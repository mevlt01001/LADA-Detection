import torch
import torch.nn as nn
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
    def __init__(self, regmax, strides, device=torch.device("cpu")):
        super().__init__()
        self.regmax = regmax
        self.strides = strides
        self.regmax_tensor = torch.arange(self.regmax, device=device).view(1,1,self.regmax,1,1)
        

    def forward(self, x):
        # x = [[1,4*regmax+nc,80,80],[1,4*regmax+nc,40,40],[1,4*regmax+nc,20,20]]
        B, C, H, W = x[0].shape
        dist_ltrb = [ch[:, :4 * self.regmax].view(B, 4, self.regmax, ch.shape[2], ch.shape[3]) for ch in x]
        dist_ltrb = [torch.softmax(ch, dim=2) for ch in dist_ltrb]
        dist_ltrb = [ch*self.regmax_tensor for ch in dist_ltrb]
        ltrb = [torch.sum(ch, dim=2)*st for ch,st in zip(dist_ltrb, self.strides)]
        xyxy = [cxcywh2xyxy(ch) for ch in ltrb] # code ltrb to xyxy

        cls_score = [ch[:, 4 * self.regmax:] for ch in x]

        print(f"dist_ltrb: {[d.shape for d in ltrb]}")
        print(f"cls_score: {[c.shape for c in cls_score]}")

        return [ltrb.reshape(B, 4, -1) for ltrb in ltrb]

class Model(nn.Module):
    def __init__(self, models: list[UltrlyticsModel], nc=80, regmax=16, device=torch.device("cpu")):
        super().__init__()
        self.body = HybridNet(models=models, device=device)        
        self.head = Head(nc=nc, regmax=regmax, in_ch=self.body.out_ch, device=device)
        self.dfl = DFL(regmax=regmax, strides=self.body.strides, device=device)
    
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        x = self.dfl(x)
        return x

model1 = YOLO("pt_folder/yolo12l.pt")
model2 = RTDETR("pt_folder/rtdetr-l.pt")
device = torch.device("cuda")

model = Model(models=[model1, model2], nc=80, regmax=16, device=device)
input_data = torch.randn(1, 3, 640, 640, device=device)

torch.onnx.export(
    model,
    input_data,
    "onnx_folder/ensemble_feature_extracter.onnx",
    export_params=True,
    opset_version=19,
    input_names=["input"],
    output_names=["f3", "f2", "f1"],
)
import onnx, onnxsim
model = onnx.load("onnx_folder/ensemble_feature_extracter.onnx")
model, check = onnxsim.simplify(model)
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "onnx_folder/ensemble_feature_extracter.onnx")
