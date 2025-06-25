import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.engine.model import Model as UltrlyticsModel
from ultralytics.models.yolo import YOLO
from ultralytics.models.rtdetr import RTDETR
from ultralytics.nn.modules import Detect, RTDETRDecoder, Concat
from ultralytics.nn.modules.conv import Conv, DWConv

class FPN(nn.Module):
    """
    Extracts features pyramid (FPN) (p3, p4, p5) output from given model as a backbone (ultralytics.nn.tasks.DetectionModel).
    """
    def __init__(self, model: UltrlyticsModel, device=torch.device("cpu")):
        super().__init__()
        assert isinstance(model, UltrlyticsModel), f"model should be instance of ultralytics.engine.model.Model"
        assert model.task == "detect", f"An error occurred in {model.model_name}. Only 'detect' task are supported, not {model.task} task yet."
        self.f = model.model.model[-1].f
        self.layers = model.to(device).model.model
        self.imgsz = model.overrides.get('imgsz')

    def forward(self, x):
        outputs = []
        for m in self.layers:
            if isinstance(m, (Detect, RTDETRDecoder)):
                return [outputs[f] for f in self.f]
            elif isinstance(m, Concat):
                x = m([outputs[f] for f in m.f])
            else:
                x = m(x) if m.f == -1 else m(outputs[m.f])
            outputs.append(x)

class HybridNet(nn.Module):
    """
    Extracts features pyramid (FPN) (p3, p4, p5) output and concatenates from given models as a backbone (ultralytics.nn.tasks.DetectionModel).
    """
    def __init__(self, models: list[UltrlyticsModel], device=torch.device("cpu")):
        super().__init__()
        self.models = [FPN(model, device) for model in models]
        assert all([len(model.f) == len(self.models[0].f) for model in self.models]), f"{[model.f for model in self.models]} should be same length."
        self.f_lenght = len(self.models[0].f)
        assert all([model.imgsz == self.models[0].imgsz for model in self.models]), f"{[model.imgsz for model in self.models]} should be same."
        self.imgsz = self.models[0].imgsz
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.imgsz, self.imgsz, device=device)
            out = self.forward(dummy)
            self.out_ch = [p.shape[1] for p in out]
            self.strides = [self.imgsz // f.shape[-1] for f in out]
        del dummy, out

    def forward(self, x):
        _outputs = []
        for model in self.models:
            out = model.forward(x)
            _outputs.append(out)
        
        outputs = []
        for feat_idx in range(len(_outputs[0])):
            outputs.append(
                torch.cat([out[feat_idx] for out in _outputs], dim=1)
            )
        return outputs

class Head(nn.Module):
    """
    Distribution of bounding boxes (l, t, r, b) and classes.
    """
    def __init__(self, nc, in_ch, regmax=16, device=torch.device("cpu")):
        super().__init__()
        self.nc = nc
        self.regmax = regmax
        reg_inter_ch = lambda ch: max((self.regmax, ch // 4, self.regmax * 4))
        cls_inter_ch = max((min(in_ch)//4, self.nc))
        self.reg_blocks = nn.ModuleList(
            nn.Sequential(
                Conv(ch, reg_inter_ch(ch), 3),
                Conv(reg_inter_ch(ch), reg_inter_ch(ch), 3),
                Conv(reg_inter_ch(ch), reg_inter_ch(ch), 3),
                nn.Conv2d(reg_inter_ch(ch), 4 * self.regmax, 1),
            )for ch in in_ch
        ).to(device)
        self.cls_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(ch, ch, 3), Conv(ch, cls_inter_ch, 1)),
                nn.Sequential(DWConv(cls_inter_ch, cls_inter_ch, 3), Conv(cls_inter_ch, cls_inter_ch, 1)),
                nn.Conv2d(cls_inter_ch, self.nc, 1),
            )for ch in in_ch
        ).to(device)

    def forward(self, x):
        # x = [[1,ch,80,80],[1,ch,40,40],[1,ch,20,20]]
        outputs = []
        for idx, x in enumerate(x):
            reg = self.reg_blocks[idx](x)
            cls = self.cls_blocks[idx](x)
            outputs.append(torch.cat((reg, cls), 1))
        return outputs

class DFL(nn.Module):
    def __init__(self, strides, imgsz=640, nc=80, regmax=16, device=torch.device("cpu")):
        super().__init__()
        self.regmax = regmax
        self.device = device
        self.nc = nc
        self.proj = torch.arange(0, self.regmax).view(1, -1, 1, 1, 1).float().to(device)
        self.strides = strides
        self.imgsz = imgsz
    
    def forward(self, x: list[torch.Tensor]):
        return self.decode(x)

    def decode(self, x: list[torch.Tensor]):
        # x = [[1,4*regmax+nc,80,80],[1,4*regmax+nc,40,40],[1,4*regmax+nc,20,20]]
        B, *_ = x[0].shape
        reg_preds = [p[:, :4 * self.regmax].reshape(B, self.regmax, 4, p.shape[-2], p.shape[-1]) for p in x]
        reg_preds = [torch.softmax(p, 1)*self.proj for p in reg_preds]
        reg_preds = [torch.sum(p, 1)*self.strides[idx] for idx, p in enumerate(reg_preds)]
        reg_preds = [self.ltrb2xyxy(p) for p in reg_preds]
        cls_preds = [p[:, 4 * self.regmax:].reshape(B, self.nc, p.shape[-2], p.shape[-1]) for p in x]
        cls_preds = [torch.softmax(p, 1) for p in cls_preds]
        out = [torch.cat((r, c), 1) for r, c in zip(reg_preds, cls_preds)]
        return out

    def encode(self, boxes: torch.Tensor):
        # boxes = [n,4]
        pass

    def ltrb2xyxy(self, dist_reg: torch.Tensor):
        # dist_reg = [n,4,ch,ch]
        B, _, _, W = dist_reg.shape
        stride = self.imgsz // W
        device = dist_reg.device
        cellx, celly = torch.meshgrid(torch.arange(W), torch.arange(W), indexing='xy')
        cx = (cellx + 0.5).to(device) * stride
        cy = (celly + 0.5).to(device) * stride
        l, t, r, b = dist_reg[:, 0], dist_reg[:, 1], dist_reg[:, 2], dist_reg[:, 3]
        x1 = cx - l; y1 = cy - t; x2 = cx + r; y2 = cy + b
        return torch.stack((x1, y1, x2, y2), 1)

class Model(nn.Module):
    def __init__(self, models: list[UltrlyticsModel], nc=80, regmax=16, device=torch.device("cpu")):
        super().__init__()
        self.body = HybridNet(models=models, device=device)        
        self.head = Head(nc=nc, regmax=regmax, in_ch=self.body.out_ch, device=device)
        self.dfl = DFL(strides=self.body.strides, imgsz=self.body.imgsz, nc=nc, regmax=regmax, device=device)
    
    def forward(self, x, train:bool=False):
        x = self.body(x)
        x = self.head(x)
        x = self.dfl(x) if not train else x
        return x
    
class ModelTrainer:
    def __init__(self, model:Model):
        self.model = model
        self.strides = model.body.strides
        self.imgsz = model.body.imgsz
        self.nc = model.dfl.nc
        self.regmax = model.dfl.regmax

    def create_targets(self, gt_boxes:np.ndarray, preds:list[torch.Tensor], epoch:int, k:int=13):
        # gt_boxes = [n,5], box format (cls_id,cx,cy,w,h), not normalized (0-imgsz)
        # preds = [[1,4*regmax+nc,p3,p3],[1,4*regmax+nc,p4,p4],[1,4*regmax+nc,p5,p5]]
        
        pass

    
    def create_c1_targets(self, target_imgsz:tuple, gt_boxes:torch.Tensor):
        # gt_boxes = [n,5], box format (cls_id,cx,cy,w,h), normalized (0-1)
        # preds = [[1,4*regmax+nc,p3,p3],[1,4*regmax+nc,p4,p4],[1,4*regmax+nc,p5,p5]]

        H,W = target_imgsz
        gt_boxes[:, [1, 3]]/=W
        gt_boxes[:, [2, 4]]/=H

        labels = [torch.zeros(1, self.regmax*4+self.nc, self.imgsz//st, self.imgsz//st) for st in self.strides]
        assignments = []

        r = lambda s: np.sqrt((s*(1/max(self.strides))))
        positive_areas = lambda s, gt: [gt[0]*s, gt[1]*s, gt[2]*r(s)*s, gt[3]*r(s)*s]


        for lb,st in zip(labels, self.strides):
            for box in gt_boxes:
                px, py, pw, ph = positive_areas(st, box) # GT Box positive area 
                for j in range(self.imgsz//st):
                    for i in range(self.imgsz//st):
                        if px < i < px + pw and py < j < py + ph: # Check grid cell if in positive area
                            










            

model1 = YOLO("pt_folder/yolo12l.pt")
model2 = RTDETR("pt_folder/rtdetr-l.pt")
device = torch.device("cuda")

model = Model(models=[model1, model2], nc=80, regmax=16, device=device)
dfl = DFL(strides=model.body.strides, imgsz=model.body.imgsz, nc=80, regmax=16, device=device)
trainer = ModelTrainer(model)

data = torch.randint(0, 80, (3, 5))
trainer.create_c1_targets((3,4), data.to(torch.float32))

k=20
input_data = torch.randn(1, 3, int(32*k), int(32*k), device=device)

print([p.shape for p in model.forward(input_data, train=False)])