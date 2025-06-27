import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
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

class Model(nn.Module):
    def __init__(self, models: list[UltrlyticsModel], nc=80, regmax=16, device=torch.device("cpu")):
        super().__init__()
        self.body = HybridNet(models=models, device=device)        
        self.head = Head(nc=nc, regmax=regmax, in_ch=self.body.out_ch, device=device)
        self.dfl = None
    
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
        self.nc = model.head.nc
        self.regmax = model.head.regmax

    def create_c2_targets(self, assignments: dict):
        # assingnments = {st: {grid_point: [gt, loss]}}
        # TODO: implemete c2 round of selecting anchor points
        pass

    def create_c1_targets(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]):
        # gt_boxes = [n,5], box format (cls_id,cx,cy,w,h), normalized[0-1]
        # preds = [[1,4*regmax+nc,p3,p3],[1,4*regmax+nc,p4,p4],[1,4*regmax+nc,p5,p5]]

        assignments = defaultdict(list) # [st, grid_point, normalized_point, [gt, loss]]
        r = lambda s: np.sqrt((s*(1/max(self.strides))))
        positive_areas = lambda s, gt:\
                            ModelTrainer.cxcywh2xyxy(
                                torch.tensor([gt[0], gt[1], gt[2]*r(s), gt[3]*r(s)])
                            ) # EPCP
        
        # Grid assignment
        for st in self.strides:
            for box in gt_boxes:
                pa = positive_areas(st, box) # GT Box positive area xyxy
                for j in range(self.imgsz//st):
                    for i in range(self.imgsz//st):
                        cx = (i+0.5)*st/self.imgsz
                        cy = (j+0.5)*st/self.imgsz
                        if (pa[0] < cx < pa[2]) and (pa[1] < cy < pa[3]): # Check grid cell if in positive area
                            assignments[(st, (j, i), (cx, cy))].append((box, -1)) # Assign to grid cell

            # Multiple Assignment Handling
            for (stride_and_loc, boxes) in assignments.items():
                if len(boxes) > 1: # Multiple assignment
                    cx, cy = stride_and_loc[2]
                    best_box = None
                    best_dist = float('inf')
                    for box,loss in boxes:
                        bcx = box[0]
                        bcy = box[1]
                        dist = (bcx - cx)**2 + (bcy - cy)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_box = box
                    assignments[stride_and_loc] = [(best_box, -1)]

        # Loss Calculation for preds tensor
        for pred in preds:
            _st = pred.shape[-1]
            for (st, (j, i), (cx, cy)), (box, loss) in assignments.items():
                if st == _st:
                    loss = self.CLA(pred[0, :, i, j], box, (cx,cy), st, self.imgsz, self.regmax, self.nc)
                    assignments[(st, (j, i), (cx, cy))] = [(box, loss)]

        return assignments

    @staticmethod            
    def CLA(pred, gt, anchor_point:tuple[int,int], stride:int, imgsz:int, regmax:int, nc:int):
        # Combined Loss of Anchors
        # pred = 4*regmax+nc , gt = 4+1
        # CLA = Lcls+ λ1*Lreg + λ2*Ldev | λ1:1.5, λ2:1 accorting to paper https://doi.org/10.3390/s23146306
        # Lcls = FocalLoss(pclsj , gi)
        # Lreg = CIoULoss(pregj , gi)
        # Ldev = |l − r|/(l+r) + |t − b|(t+b)
        
        def dev(anchor_point:torch.Tensor, gt:torch.Tensor, stride:int, imgsz:int):
            cx = anchor_point[0]
            cy = anchor_point[1]
            box_xyxy = torchvision.ops.box_convert(gt, 'cxcywh', 'xyxy')[0]
            l = cx - box_xyxy[0]
            t = cy - box_xyxy[1]
            r = box_xyxy[2] - cx
            b = box_xyxy[3] - cy
            hdev = torch.abs(l - r) / (l + r)
            vdev = torch.abs(t - b) / (t + b)
            dev = hdev + vdev
            if 0<=dev<=1: return 0
            elif 1<dev<=2: return 1

        pred_classes = pred[regmax*4:]
        gt_classes = torch.nn.functional.one_hot(gt[0].long(), num_classes=nc)

        pred_reg = pred[:regmax*4].view(regmax, 4).softmax(dim=0)
        proj = torch.arange(0, regmax).view(-1,1).float().to(pred.device)
        ltrb = torch.sum(pred_reg*proj, dim=0)
        xyxy_pred = torchvision.ops.box_convert(ltrb, 'ltrb', 'xyxy')*stride/imgsz
        xyxy_gt = torchvision.ops.box_convert(gt[1:], 'ltrb', 'xyxy')

        Lreg = torchvision.ops.ciou_loss.complete_box_iou_loss(xyxy_pred, xyxy_gt)
        Lcls = torchvision.ops.focal_loss.sigmoid_focal_loss(pred_classes, gt_classes, alpha=0.25, gamma=2)
        Ldev = dev(anchor_point, gt, stride, imgsz)

        return Lcls + 1.5*Lreg + Ldev
            
            



    @staticmethod
    def cxcywh2xyxy(x):
        # Convert nx4 boxes from [cx, cy, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    @staticmethod
    def cxcywh2ltrb(x):
        # Convert nx4 boxes from [cx, cy, w, h] to [left, top, right, bottom]
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # left
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # right
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom
        return y

    def decode(self, x: torch.Tensor):
        # x = 4*regmax+nc
        reg = x[:4*self.regmax]
        reg = reg.view(self.regmax, 4)
        reg = torch.softmax(reg, dim=0)
        proj = torch.arange(0, self.regmax).view(-1,1).float().to(x.device)
        ltrb = torch.sum(reg*proj, dim=0)

        cls = x[4*self.regmax:]
        cls = torch.softmax(cls, dim=0)
        return ltrb, cls
        
        







            

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