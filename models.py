import os, json
import cv2, onnx, onnxsim
import numpy as np
import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from ultralytics.engine.model import Model as UltrlyticsModel
from ultralytics.models.yolo import YOLO
from ultralytics.models.rtdetr import RTDETR
from ultralytics.nn.modules import Detect, RTDETRDecoder, Concat
from ultralytics.nn.modules.conv import Conv, DWConv
   
class FPN(torch.nn.Module):
    """
    This class extracts features pyramid (FPN) (p3, p4, p5) output from given model as a backbone (ultralytics.nn.tasks.DetectionModel).
    """
    def __init__(self, model: UltrlyticsModel, device=torch.device("cpu")):
        assert isinstance(model, UltrlyticsModel), f"model should be instance of ultralytics.engine.model.Model"
        assert model.task == "detect", f"An error occurred in {model.model_name}. Only 'detect' task are supported, not {model.task} task yet."
        super(FPN, self).__init__()
        self.model = model
        self.f = model.model.model[-1].f # list of feature map layer indices [..., p3, p4, p5, ...]
        self.layers = model.to(device).model.model
    
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
        raise ValueError(f"An error occurred in {self.model.model_name}. Detect/RTDETRDecoder layer not found.")

class HybridNet(torch.nn.Module):
    """
    This class concatenates features pyramid (FPN) (p3, p4, p5) output from given models as a backbone (ultralytics.nn.tasks.DetectionModel).
    
    Args:
        models (list[ultralytics.engine.model.Model]): list of ultralytics.engine.model.Model with 'detect' task
        imgsz (int): model input image size. Models accpets min(64, 32k) {k ∈ Natural numbers}-{0}
        device (torch.device): torch device
    """
    def __init__(self, models: list[UltrlyticsModel], imgsz:int, device=torch.device("cpu")):
        super(HybridNet, self).__init__()
        self.models = [FPN(model, device) for model in models]
        assert all([len(model.f) == len(self.models[0].f) for model in self.models]),\
            f"{[model.f for model in self.models]} should be same length."
        self.f_lenght = len(self.models[0].f)
        with torch.no_grad():
            dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
            out = self.forward(dummy)
            self.out_ch = [p.shape[1] for p in out]
            self.strides = [imgsz // f.shape[-1] for f in out]
        del dummy, out

    def forward(self, x):
        """
        Concatenates feature maps output from given models and returns list of feature maps

        Args:
            x (torch.Tensor): input image tensor shaped like (B, C, H, W) or (1,3,640,640)

        Returns:
            list[torch.Tensor]: list of feature maps
        """
        _outputs = []
        for model in self.models:
            out = model.forward(x) # [[1,ch1,80k,80k],[1,ch2,40k,40k],...,[1,chn,20k,20k]]
            _outputs.append(out)
        
        outputs = []
        for feat_idx in range(len(_outputs[0])):
            outputs.append(
                # [1,ch1+ch2+...+chn,80k,80k]
                torch.cat([out[feat_idx] for out in _outputs], dim=1)
            )
        return outputs # [[1,ch1+ch2+...+chn,80,80], [1,ch1+ch2+...+chn,40,40], ...]
    
class Head(nn.Module):
    """
    This class provides **DFL-Based** and **anchor-free** regression and classification layers which is like YOLOv11

    Args:
        nc (int): number of classes
        in_ch (int): number of input channels
        regmax (int): maximum number of anchors
        device (torch.device): device

    Methods:
        forward(self, x): accepts feature pyramid output (..., p3, p4, p5, ...) and returns DFL-Based regression and classification outputs for feature maps.
    """
    def __init__(self, nc, in_ch, regmax=16, device=torch.device("cpu")):
        super().__init__()
        reg_inter_ch = lambda ch: max((ch // 4, regmax * 4))
        cls_inter_ch = max((min(in_ch)//4, nc))
        self.reg_blocks = nn.ModuleList(
            nn.Sequential(
                Conv(ch, reg_inter_ch(ch), 3),
                Conv(reg_inter_ch(ch), reg_inter_ch(ch), 3),
                Conv(reg_inter_ch(ch), reg_inter_ch(ch), 3),
                nn.Conv2d(reg_inter_ch(ch), 4 * regmax, 1),
            )for ch in in_ch
        ).to(device)
        self.cls_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(ch, ch, 3), Conv(ch, cls_inter_ch, 1)),
                nn.Sequential(DWConv(cls_inter_ch, cls_inter_ch, 3), Conv(cls_inter_ch, cls_inter_ch, 1)),
                nn.Conv2d(cls_inter_ch, nc, 1),
            )for ch in in_ch
        ).to(device)

    def forward(self, x):
        """
        Args:
            x (list[torch.Tensor]): `HybridNet` feature pyramid output (..., p3, p4, p5, ...)

        Returns:
            outputs (list[torch.Tensor]):
            DFL-Based regression and classification outputs for each feature pyramid output.\\
            The shape of each output is (B, 4*regmax+nc, h, w)
        """
        # x = [[1,ch1+ch2+...+chn,80k,80k],[1,ch1+ch2+...+chn,40k,40k],[1,ch1+ch2+...+chn,20k,20k]]
        outputs = []
        for idx, x in enumerate(x):
            reg = self.reg_blocks[idx](x) #            [[1,4*regmax,H,W]
            cls = self.cls_blocks[idx](x) #            [[1,nc,H,W]
            outputs.append(torch.cat((reg, cls), 1)) # [[1,4*regmax+nc,H,W]
        return outputs # [[1,4*regmax+nc,80k,80k],[1,4*regmax+nc,40k,40k],[1,4*regmax+nc,20k,20k]]

class Preprocess(nn.Module):
    """
    This class is used to below1 input process for models.
    Args:
        imgsz (int): Model input image(H,W) size

    ### Image init parameters:
    - type: uint8
    - shape: (H, W, 3)
    - range: [0, 255]
    - format: BGR
    ### Models accepted parameters:
    - type: float32
    - shape: (B, 3, imgsz, imgsz)
    - range: [0.0, 1.0]
    - format: RGB
    
    Models accpets min(64, 32k) {k ∈ Natural numbers}

    """
    def __init__(self, imgsz=640):
        super().__init__()
        self.imgsz = imgsz
    def forward(self, x: torch.Tensor):
        # type conversion: uint8 -> float
        x = x.to(torch.float32)
        # BGR to RGB
        print(x.shape)
        x = torch.cat(
            [
                x[:, :, 2:],
                x[:, :, 1:2],
                x[:, :, 0:1],
            ],
            dim=-1
        )
        print(x.shape)
        # shape conversion: (H,W,3) -> (1,3,imgsz,imgsz)/255.0
        x = x.permute(2, 0, 1).unsqueeze(0) # [H,W,3] -> [1,3,H,W]
        x = F.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear")
        x = x*(1.0/255.0)
        return x

class Model(nn.Module):
    """
    This class is a combination of `HybridNet` and `Head` classes.\\
    It provides **DFL-Based** and **anchor-free** regression and classification layers which is like YOLOv11

    Args:
        models (list[UltrlyticsModel]): list of ultralytics models
        nc (int): number of classes
        regmax (int, None): maximum number of bbox distrubution bins. if None, set to `imgsz//int(np.median(self.body.strides))//2`
        imgsz (int): input image size
        device (torch.device): device

    Methods:
        forward(self, x): accepts input image tensor shaped like (H, W, C) and returns DFL-Based regression and classification outputs for feature maps.
    """
    def __init__(self, models: list[UltrlyticsModel], nc=80, regmax=None, imgsz=640, device=torch.device("cpu")):
        super().__init__()
        self.imgsz = max(64, 32*(imgsz//32))
        self.preprocess = Preprocess(imgsz=self.imgsz)
        self.body = HybridNet(models=models, device=device, imgsz=self.imgsz)
        self.nc = nc
        self.regmax = imgsz//int(np.median(self.body.strides))//2 if regmax is None else regmax
        self.head = Head(nc=nc, regmax=self.regmax, in_ch=self.body.out_ch, device=device)
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.body(x)
        x = self.head(x)
        return x
   
class ModelTrainer:
    """
    This class is a trainer for `Model` class.\\
    It provides to train `Model` class with Lightweight Anchor Dynamic Assignment (LADA) Algorithm https://doi.org/10.3390/s23146306\\

    Args:
        model (Model): HybridNet and Head models which are processed ultralytics models

    Methods:
        crate_targets(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]): creates targets
    """
    def __init__(self, model:Model, device=torch.device("cpu")):

        self.model = model
        self.strides = model.body.strides
        self.imgsz = model.imgsz
        self.nc = model.nc
        self.regmax = model.regmax
        self.device = device
        self.root = None
        self.train_ann_data = None
        self.val_ann_data = None
        
    def __load_json_data(self, ann_json_path:os.PathLike, trainval:bool=False):
        with open(ann_json_path, "r") as f:
            return json.load(f)

    def __prep_names(data:dict):
        data = data["images"]
        ann = data["annotations"]
        names = [[os.path.splitext(data[i]["file_name"]), data[i]["id"]] for i in range(len(data))]
        return names
    
    def __get_bbox_with_id(data:list, id:int):
        # data = json["annotations"]
        pass

    def calc_loss(preds, targets):
        # TODO: calc loss with DFL + Focal Loss + CIoULoss for each stride
        pass

    def crate_targets(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]):
        """
        Creates targets with LADA Assignment Algorithm https://doi.org/10.3390/s23146306\\
        This function is a combination of *create_c1_targets*, *create_c2_targets* and *create_c3_targets*\\
        Additionally, it applies **Dynamic Loss Threshold(DLT)** over *create_c3_targets* by using average loss\\
        3.3. Dynamic Loss Threshold

        Args:
            gt_boxes (torch.Tensor): ground truth boxes shaped like (n,5) (cls_id,cx,cy,w,h), normalized[0-1]
            preds (list[torch.Tensor]): (..,p3,p4,p5,..) predictions [1,4*regmax+nc,H,W]

        Returns:
            labels (list[torch.Tensor]): targets for each stride (...,p3,p4,p5,..)
        """
        c1_assignments = self.create_c1_targets(gt_boxes, preds) # EPCP areas anchor points assignment for each stride
        c2_assignments = self.create_c2_targets(c1_assignments) # EPCP areas best 9 anchor points assignment for each stride
        c3_assignments, avg_loss = self.create_c3_targets(c2_assignments) # Total best 20 anchor points assignment and average its losses
        final_assignments = defaultdict(list) # it will use to apply DLT
        
        for (st, (j, i), (cx, cy)), (box, loss) in c3_assignments.items():
            if loss < avg_loss: # Dynamic Loss Threshold(DLT)
                final_assignments[(st, (j, i), (cx, cy))].append((box, loss))
        
        labels = [torch.zeros((1, 4*self.regmax+self.nc, self.imgsz//st, self.imgsz//st), device=self.model.device) for st in self.strides] # labels for each stride
        
        for lb in labels:
            for (st, (j, i), (cx, cy)), (gt_box, loss) in final_assignments.items():
                if st == lb.shape[2]*self.imgsz//st: # Check stride
                    cls_label = torch.nn.functional.one_hot(gt_box[0].long(), self.nc)
                    x1, y1, x2, y2 = torchvision.ops.box_convert(gt_box[1:], 'cxcywh', 'xyxy')[0]
                    l = (cx - x1)*st
                    t = (cy - y1)*st
                    r = (x2 - cx)*st
                    b = (y2 - cy)*st
                    ltrb_dist = self.distribute(l, t, r, b)
                    label = torch.cat([ltrb_dist, cls_label], dim=0)
                    lb[0, :, j, i] = label

        return labels

    def create_c3_targets(self, assignments: dict, k: int=20):
        """
        Select best k anchor points from total c2_assignments

        Args:
            assignments (defaultdict(list)): c2_assignments

        Returns:
            c3_assignments (defaultdict(list)): Selected k anchor points from total c2_assignments
        """
        # assingnments = {st: {grid_point: [gt, loss]}}
        # calc select best k for each stride
        c3_assignments = defaultdict(list)
        # assignments.items() : ((st, (j, i), (cx, cy)), [(box, loss)]) | value listesi her zaman 1 elemanlı
        sorted_assignments = sorted(assignments.items(), key=lambda x: x[1][0][1])
        k = min(k, len(sorted_assignments))
        losses = []
        for (st, (j, i), (cx, cy)), (box, loss) in sorted_assignments[:k]:
            c3_assignments[(st, (j, i), (cx, cy))].append((box, loss))
            losses.append(loss)

        return c3_assignments, sum(losses)/len(losses)

    def create_c2_targets(self, assignments: dict, k: int=9):
        """
        Select best k anchor points for each stride from c1_assignments

        Args:
            assignments (defaultdict(list)): c1_assignments

        Returns:
            c2_assignments (defaultdict(list)): selected best k anchor points for each stride from c1_assignments
        """

        # select best k for each stride
        c2_assignments = defaultdict(list)
        for st in self.strides:
            stride_groups = []
            for (_st, (j, i), (cx, cy)), (box, loss) in assignments.items():
                if _st == st:
                    stride_groups.append(((_st, (j, i), (cx, cy)), (box, loss)))
            sorted_stride_groups = sorted(stride_groups, key=lambda x: x[1][1])
            k = min(k, len(sorted_stride_groups))
            for (st, (j, i), (cx, cy)), (box, loss) in sorted_stride_groups[:k]: # 
                c2_assignments[(st, (j, i), (cx, cy))].append((box, loss))

        return c2_assignments

    def create_c1_targets(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]):
        """
        Creates EPCP targets with LADA Assignment Algorithm https://doi.org/10.3390/s23146306 3.1. Equally Proportional Center Prior

        Args:
            gt_boxes (torch.Tensor): ground truth boxes shaped like (n,5) (cls_id,cx,cy,w,h), normalized[0-1]
            preds (list[torch.Tensor]): (..,p3,p4,p5,..) predictions [1,4*regmax+nc,H,W]

        Returns:
            assignments (defaultdict(list)): EPCP/C1 targets for each stride
        """
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
            for (st, (j, i), (cx, cy)), [(box, loss)] in assignments.items():
                if st == _st:
                    loss = self.CLA(pred[0, :, i, j], box, (cx,cy), st)
                    assignments[(st, (j, i), (cx, cy))] = [(box, loss)]

        return assignments
       
    def CLA(self, pred, gt, anchor_point:tuple[int,int], stride:int):
        """
        Calculates anchor loss to LADA Assignment Algorithm https://doi.org/10.3390/s23146306 3.2. Combined Loss of Anchor\\
        This funciton calculates combined loss of anchor\\
        CLA = Lcls+ λ1*Lreg + λ2*Ldev | λ1:1.5, λ2:1\\
        Lcls = FocalLoss(pcls , g)\\
        Lreg = CIoULoss(preg , g)\\ !**paper used GIoULoss**!\\
        Ldev = |l − r|/(l+r) + |t − b|/(t+b) is anchor point deviation from ground truth

        Args:
            pred (torch.Tensor): prediction tensor shaped like [4*regmax+nc]
            gt (torch.Tensor): ground truth tensor shaped like [1+4] (nc, cx, cy, w, h)
            anchor_point (tuple[int,int]): anchor point (cx,cy), normalized [0-1]
            stride (int): stride

        Returns:
            loss (torch.Tensor): loss"""
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
            if 0<=dev.item()<=1: return 0
            elif 1<dev.item()<=2: return dev - 1

        pred_classes = pred[self.regmax*4:]
        gt_classes = torch.nn.functional.one_hot(gt[0].long(), num_classes=self.nc)

        pred_reg = pred[:self.regmax*4].view(self.regmax, 4).softmax(dim=0)
        proj = torch.arange(0, self.regmax).view(-1,1).float().to(pred.device)
        ltrb = torch.sum(pred_reg*proj, dim=0)
        xyxy_pred = torchvision.ops.box_convert(ltrb, 'ltrb', 'xyxy')*stride/self.imgsz
        xyxy_gt = torchvision.ops.box_convert(gt[1:], 'ltrb', 'xyxy')

        Lreg = torchvision.ops.ciou_loss.complete_box_iou_loss(xyxy_pred, xyxy_gt)
        Lcls = torchvision.ops.focal_loss.sigmoid_focal_loss(pred_classes, gt_classes, alpha=0.25, gamma=2)
        Ldev = dev(anchor_point, gt, stride, self.imgsz)

        return Lcls + 1.5*Lreg + Ldev
            
    def distribute(self, l: torch.Tensor, t: torch.Tensor, r: torch.Tensor, b: torch.Tensor):
        """
        Distributes l, t, r, b to regmax bins

        Args:
            l (torch.Tensor): left
            t (torch.Tensor): top
            r (torch.Tensor): right
            b (torch.Tensor): bottom

        Returns:
            torch.Tensor: distributed l, t, r, b

        Examples:
            >>> regmax = 4
            >>> a, b, c, d = torch.tensor([0.4, 0.2, 1.3, 0.4])
            >>> dist(a)
            tensor([0.6000, 0.4000, 0.0000, 0.0000])
            >>> dist(c)
            tensor([0.0000, 0.7000, 0.3000, 0.0000])
        """
        # Distribute loss
        def dist(val:torch.Tensor):
            v = torch.clamp(val, 0, self.regmax - 1 - 10e-3)
            left = int(torch.floor(v).item())
            right = left + 1
            label = torch.zeros(self.regmax, device=val.device)
            if right >= self.regmax:
                label[left] = 1.0
            else:
                label[left]  = right - v.item()
                label[right] = v.item() - left
            return label
        
        return torch.cat([dist(l), dist(t), dist(r), dist(b)], dim=0)

model1 = YOLO("pt_folder/yolo12l.pt")
model2 = RTDETR("pt_folder/rtdetr-l.pt")

model = Model(models=[model1, model2], nc=5, regmax=None, imgsz=32*16, device=torch.device("cuda"))
dummy = torch.randint(0, 255, (334,553, 3), dtype=torch.uint8, device=torch.device("cuda"))
print(model.body.strides)
print(model.regmax)
out = model(dummy)
print([p.shape for p in out])