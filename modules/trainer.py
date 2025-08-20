"""
# LADA Assignment Algorithm https://doi.org/10.3390/s23146306
# Training loss is different from paper
## Training Loss = λ1*Focal Loss + λ2*CIoU Loss + λ3*DFL | λ1:0.5, λ2:1.5, λ3:7.5
"""
from typing import TYPE_CHECKING, List, Tuple, Dict
if TYPE_CHECKING:
    from .model import Model

import os, cv2
import sys
import torch
import torchvision
import numpy as np
from PIL import Image
from collections import defaultdict
from torchmetrics.detection import MeanAveragePrecision
import warnings
import math

warnings.filterwarnings("ignore")

def debug_visualize(image: torch.Tensor,
                    gt_boxes: torch.Tensor, # [N, 5]: cls, cx, cy, w, h
                    pred_boxes: torch.Tensor, # [N, 6]: xyxy, cls_score, cls_id
                    imgsz: int,
                    strides: list[int],
                    pt_by_st: list[list],
                    names: list[str],
                    ):
    
    num_strides = len(strides)

    if pt_by_st is None or not isinstance(pt_by_st, list) or len(pt_by_st) != 4:
        pt_by_st = [[], [], [], []]

    norm = []
    for candidates in pt_by_st:
        if not isinstance(candidates, list):
            candidates = []
        if len(candidates) < num_strides:
            candidates = candidates + [[] for _ in range(num_strides - len(candidates))]
        elif len(candidates) > num_strides:
            candidates = candidates[:num_strides]
        norm.append(candidates)
    pt_by_st = norm

    image = np.ascontiguousarray((image.permute(1,2,0)*255).to(device="cpu", dtype=torch.uint8).numpy())
    img2 = image.copy()
    if gt_boxes.shape[0] > 0:

        cls_idx, cxcywh = torch.split(gt_boxes, [1,4], dim=1)
        cx, cy, w, h = torch.unbind(cxcywh*imgsz, dim=1)
        boxes = torch.stack((cx-w/2, cy-h/2, cx+w/2, cy+h/2), dim=1)

        for box, idx in zip(boxes, cls_idx.long()):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(idx)]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 1)
            w,h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            image = cv2.rectangle(image, (x1, y1), (int(x1+w), int(y1-h-10)), (0,255,0), cv2.FILLED)
            image = cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    frames = np.zeros((imgsz*len(strides), imgsz*4, 3), dtype=np.uint8)

    for st_idx in range(len(strides)):
        frame = np.zeros((imgsz, imgsz*4, 3), dtype=np.uint8) # stride frame
        for c_idx, candidates in enumerate(pt_by_st):
            img_cp = image.copy()
            for assignment in candidates[st_idx]:
                cx = int(assignment[0]*imgsz)
                cy = int(assignment[1]*imgsz)
                color = assignment[2]
                img_cp = cv2.circle(img_cp, (cx, cy), 4, color, 4)
            img_cp = cv2.putText(img_cp, str(len(candidates[st_idx])) + " points", (10, imgsz-10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            frame[:, c_idx*imgsz:(c_idx+1)*imgsz, :] = img_cp
        frames[st_idx*imgsz:(st_idx+1)*imgsz, :, :] = frame

    ratio = int(imgsz/1)

    if (frames==0).all():
        for st_idx in range(len(strides)):
            frame = np.zeros((imgsz, imgsz*4, 3), dtype=np.uint8) # stride frame
            for c_idx in range(4):
                frame[:, c_idx*imgsz:(c_idx+1)*imgsz, :] = image
            frames[st_idx*imgsz:(st_idx+1)*imgsz, :, :] = frame

    mask = pred_boxes[:, 4] > 0.2
    pred_boxes = pred_boxes[mask]
    xyxy = pred_boxes[:, :4].int()
    cls_score = pred_boxes[:, 4]
    cls_id = pred_boxes[:, 5].int()

    for xyxy, cls_score, cls_id in zip(xyxy, cls_score, cls_id):

        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0,0,255), 2)
        label = f"{names[cls_id.item()]} {cls_score.item():.2f}"
        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        img2 = cv2.rectangle(img2, (x1, y1), (int(x1+w), int(y1-h-10)), (0,0,255), cv2.FILLED)
        img2 = cv2.putText(img2, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # frames[:imgsz, imgsz*4:imgsz, :] = img2

    frames = cv2.resize(frames, (ratio*4, ratio*len(strides)))

    cv2.imshow("debug_frame", cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))
    cv2.imshow("debug_img", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(404)

def progress_bar(total:int=5,
                 percentage:float=0.0,
                 epoch:int=None,
                 loss:float=None,
                 clsW:float=None,
                 ciouW:float=None,
                 dflW:float=None,
                 lr:float=None,
                 mAP_50:float=None,
                 mAP_50_95:float=None,
                 val:bool=False
                 ):
    progress = math.floor((percentage)*total)
    bar = f"{'█'*progress}{'░'*(total-progress)}"
    msg = (
    f"{f'Validation | ' if val else ''}"
    f"{f'Epoch={epoch:03}' if epoch is not None else ''} "
    f"{f'Loss(CLS%{int(clsW*10):02}, CIoU%{int(ciouW*10):02}, DFL%{int(dflW*10):02})={loss:<7.3f}' if loss is not None else ''} "
    f"{f'LR={lr:<7.6f}' if lr is not None else ''} "
    f"{f'mAP_50={mAP_50*100:<7.2f}' if mAP_50 is not None else ''} "
    f"{f'mAP_50_95={mAP_50_95*100:<7.2f}' if mAP_50_95 is not None else ''} "
    f"{f'{percentage*100:>7.2f}%{bar}  '}"
    )

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()

class LADATrainer:
    """
    Trainer for `Model` with LADA assignment (C1→C2→C3 + DLT). *** For more information, see https://doi.org/10.3390/s23146306

    Methods:
      - train(...)
      - create_targets(...)
      - calc_loss(...), __calc_dfl, __calc_CIoU
    """
    def __init__(self, model:"Model"):
        self.cls_names = model.cls_names
        self.model = model
        self.strides = model.backbone.strides
        self.imgsz = model.imgsz
        self.nc = model.nc
        self.regmax = model.regmax
        self.device = model.device
        self.proj = torch.arange(0, self.regmax, dtype=torch.float32, device=self.device).view(1, self.regmax, 1)
        self.last_ep = model.last_epoch
        self.last_batch = model.last_batch
        self.optim_state_dict = self.model.optim_state_dict
        self.sched_state_dict = self.model.sched_state_dict
        self.max_stride = max(self.strides)
        self.loss_mode = "softmax"
        self.loss_weights = torch.nn.Parameter(torch.zeros(3, device=self.device))  # [cls, ciou, dfl]

        # mAP
        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            backend='faster_coco_eval',
            class_metrics=False
        ).to(self.device)

        self.grid_cache: Dict[int, Dict[str, torch.Tensor|int]] = {}
        self._build_grid_cache()

    def _build_grid_cache(self):
        self.grid_cache.clear()
        for st in self.strides:
            H = W = self.imgsz // st
            jj, ii = torch.meshgrid(
                torch.arange(W, device=self.device),
                torch.arange(H, device=self.device),
                indexing='xy'
            )
            cx = (jj.float() + 0.5) * st / self.imgsz  # [H,W] normalized
            cy = (ii.float() + 0.5) * st / self.imgsz  # [H,W]
            self.grid_cache[st] = {
                "cx": cx, "cy": cy, "H": H, "W": W
            }
    
    def train(self,
              epoch:int,
              batch:int,
              train_path: str,
              valid_path: str=None,
              debug:bool=False,
              c2k:int=9,   # per-gt per-stride
              c3k:int=20,   # per-stride
              lr=0.001,
              max_lr=None
              ):

        train_names = np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(train_path,"images")))))
        valid_names = None if valid_path is None else np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(valid_path,"images")))))

        if self.last_ep/epoch > 0.99:
            self.last_ep = 0
            self.last_batch = 0

        if len(train_names) > 0 and self.last_batch/len(train_names) > 0.95:
            self.last_batch = 0
            self.last_ep += 1

        model = self.model.train(True)
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters(), "lr": lr},
                {"params": [self.loss_weights],"lr": lr/10, "weight_decay": 0.0},
            ]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[(lr if max_lr is None else max_lr), (lr/10 if max_lr is None else max_lr/10)],
            steps_per_epoch=len(train_names)//batch,
            epochs=epoch
        )

        if self.optim_state_dict is not None:
            optimizer.load_state_dict(self.optim_state_dict)

        if self.sched_state_dict is not None:
            scheduler.load_state_dict(self.sched_state_dict)

        for ep in range(self.last_ep, epoch):
            losses = []
            last_batch = self.last_batch
            for i in range(last_batch, len(train_names), batch):
                batch_size = batch if i+batch < len(train_names) else len(train_names)-i
                optimizer.zero_grad(set_to_none=True)

                images, gt_boxes = self.__load_data(train_names[i:batch_size+i], train_path)
                pred_boxes, preds = model.forward(images) # preds: [[B,C,p,p],...]
                batch_preds_for_loss = [[p[idx] for p in preds] for idx in range(len(gt_boxes))]  # [[C,p,p], ...]

                with torch.no_grad():
                    batch_preds_for_assign = [
                        [p[idx].detach() for p in preds] 
                        for idx in range(len(gt_boxes))] # [[[4*regmax+nc, H, W] for each stride] for each batch]
                    targets, pos, batch_pt_st = self.create_targets(gt_boxes, batch_preds_for_assign, debug, c2k=c2k, c3k=c3k)
                    if debug:
                        debug_visualize(
                            images[0], 
                            gt_boxes[0], 
                            pred_boxes[0], 
                            self.imgsz, 
                            self.strides, 
                            batch_pt_st[0],
                            model.cls_names)

                batch_pred_dicts, batch_target_dicts = self.batch_eval(pred_boxes, gt_boxes)
                self.map_metric.update(batch_pred_dicts, batch_target_dicts)
                batch_stats = self.map_metric.compute()
                self.map_metric.reset()
                map50 = batch_stats["map_50"].item()
                map50_95 = batch_stats['map'].item()

                loss, clsw, ciouw, dflw = self.calc_loss(batch_preds_for_loss, targets, pos)
                loss.backward()                
                optimizer.step()
                if max_lr is not None:
                    scheduler.step()
                losses.append(loss.item())

                progress_bar(total=20,
                             percentage=(i+batch_size)/max(1,len(train_names)),
                             epoch=ep+1,
                             loss=loss.item(),
                             clsW=clsw,
                             ciouW=ciouw,
                             dflW=dflw,
                             lr=optimizer.param_groups[0]['lr'],
                             mAP_50=map50,
                             mAP_50_95=map50_95,
                             val=False
                             )
                

            print(f" AvgLoss={sum(losses)/(len(losses)+1e-5):<6.4f}\n")
            self.map_metric.reset()
            torch.save({
                        "models": self.model.backbone.model_names,
                        "nc": self.nc,
                        "cls_names": self.cls_names,
                        "imgsz": self.imgsz,
                        "regmax": self.regmax,
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "sched_state_dict": scheduler.state_dict(),
                        "last_epoch": ep,
                        "last_batch": i+batch
                        }, f"LADA_{i+batch}.pt")


            # validation
            batch_eval_size = int(batch*1)
            if valid_path is not None and len(valid_names) > 0:
                with torch.inference_mode():
                    model = model.train(False)
                    for i in range(0, len(valid_names), batch_eval_size):
                        torch.cuda.empty_cache()
                        batch_size = batch_eval_size if i+batch_eval_size < len(valid_names) else len(valid_names)-i
                        images, gt_boxes = self.__load_data(valid_names[i:batch_size+i], valid_path)
                        pred_boxes = model.forward(images) # [B,4+nc,N]
                        batch_pred_dicts, batch_target_dicts = self.batch_eval(pred_boxes, gt_boxes)
                        self.map_metric.update(batch_pred_dicts, batch_target_dicts)

                        progress_bar(
                            val=True,
                            total=33,
                            percentage=(i+batch_size)/max(1,len(valid_names)),
                            mAP_50=None,
                            mAP_50_95=None,
                        )
                    batch_stats = self.map_metric.compute()
                    map50 = batch_stats["map_50"].item()
                    map50_95 = batch_stats['map'].item()
                    print(f"  mAP_50={map50*100:<6.4f}  mAP_50_95={map50_95*100:<6.4f}\n")

                    model = model.train(True)

            self.map_metric.reset()

        model = model.train(mode=False)
        self.model = model

    def __load_data(self, names: List[str], path: str):
        if path is None:
            return None, None

        images = []
        bboxes = []
        for file_name in names:
            image = self.__load_image(os.path.join(path,"images", file_name))
            bbox = self.__load_gt_boxes(os.path.join(path,"labels", file_name))
            if bbox is None:
                continue
            images.append(image)
            bboxes.append(bbox)

        return torch.stack(images, dim=0), bboxes

    def __load_image(self, path: str):
        img = np.array(Image.open(path+".jpg").convert("RGB").resize((self.imgsz, self.imgsz)))
        img = np.transpose(img, (2, 0, 1))/255.0 # HWC -> CHW        
        img = torch.from_numpy(img).to(device=self.device, dtype=torch.float32)
        return img

    def __load_gt_boxes(self, path: str):
        try:
            data = np.loadtxt(path+".txt", dtype=np.float32)
            if data.shape[-1] != 5:
                return None
        except:
            return None
        if data.size == 0:
            data = np.empty((0,5), dtype=np.float32)
        elif data.ndim == 1:
            data = np.reshape(data, (1,5))
        return torch.from_numpy(data).to(device=self.device) # [N,5]

    def calc_loss(self, preds: List[List[torch.Tensor]],
                  targets: List[List[torch.Tensor]],
                  positive_anchors: List[List[torch.Tensor]]):
        loss = 0.0
        cls_w = 0.0
        ciou_w = 0.0
        dfl_w = 0.0
        for pred, target, pos in zip(preds, targets, positive_anchors):
            Ltotal, w = self.__calc_loss(pred, target, pos)
            loss += Ltotal
            cls_w += w[0].item()
            ciou_w += w[1].item()
            dfl_w += w[2].item()
        return loss/len(preds), cls_w/len(preds), ciou_w/len(preds), dfl_w/len(preds)

    def __calc_loss(self, pred: list[torch.Tensor],
                    targets: list[torch.Tensor],
                    positive_anchors: list[torch.Tensor]):
        """
        pred:     per-stride [C,H,W]  (model output)
        targets:  per-stride [1,C,H,W] (create_targets output)
        positive_anchors: per-stride [[i,j], ...]
        """
        # Align targets and predictions by H×W size
        tgt_by_hw = { (t.shape[-2], t.shape[-1]): (t, positive_anchors[k]) for k, t in enumerate(targets) }
        ordered_tgts, ordered_pos = [], []
        for p in pred:
            key = (p.shape[-2], p.shape[-1])
            if key in tgt_by_hw:
                t, pos = tgt_by_hw[key]
            else:
                C = 4*self.regmax + self.nc
                t = torch.zeros((1, C, p.shape[-2], p.shape[-1]), device=self.device)
                pos = torch.empty((0,2), device=self.device, dtype=torch.long)
            ordered_tgts.append(t)
            ordered_pos.append(pos)

        targets = ordered_tgts # [[1,C,H,W]...]
        positive_anchors = ordered_pos # [[i,j]...]

        # 2) Reg and Cls Tensors
        pred_reg = [p[:4*self.regmax, :, :] for p in pred]                      # [4R,H,W]
        truth_reg = [t[0, :4*self.regmax, :, :] for t in targets]               # [4R,H,W]
        pred_cls  = [p[4*self.regmax:, :, :].reshape(self.nc, -1).T for p in pred]         # [N,nc]
        truth_cls = [t[0, 4*self.regmax:, :, :].reshape(self.nc, -1).T for t in targets]   # [N,nc]

        # 3) DFL
        dfl_loss = 0.0
        for p, t, pos in zip(pred_reg, truth_reg, positive_anchors):
            if pos.numel() == 0:
                continue
            dfl_loss += self.__calc_dfl(p, t, pos)

        # 4) CIoU
        ciou_loss = 0.0
        for p, t, pos in zip(pred_reg, truth_reg, positive_anchors):
            if pos.numel() == 0:
                continue
            ciou_loss += self.__calc_CIoU(p, t, pos)

        # 5) Classification focal
        cls_loss = 0.0
        for p, t in zip(pred_cls, truth_cls):
            t = t.to(dtype=p.dtype)
            cls_loss += torchvision.ops.sigmoid_focal_loss(p, t, reduction='mean')          

        # 6) Loss balance
        if self.loss_mode == "softmax":
            w = torch.softmax(self.loss_weights, dim=0) * 10
        else:
            w = torch.tensor([1.5,2.5,6.5], dtype=torch.float32, device=self.device)

        cls_loss *= w[0]
        ciou_loss *= w[1]
        dfl_loss *= w[2]
        total_loss = cls_loss + ciou_loss + dfl_loss

        return total_loss, w.detach()

    def __calc_dfl(self, pred:torch.Tensor, target:torch.Tensor, positive_anchors:torch.Tensor):
        # pred: [4*regmax,H,W]
        pred = pred[:, positive_anchors[:, 0], positive_anchors[:, 1]]   # [4*regmax,N]
        target = target[:, positive_anchors[:, 0], positive_anchors[:, 1]]# [4*regmax,N]

        target = target.view(4, self.regmax, target.shape[-1]) # [4,regmax,N]
        pred = pred.view(4, self.regmax, pred.shape[-1])       # [4,regmax,N]
        pred = torch.log_softmax(pred, dim=1)
        loss = torch.nn.functional.kl_div(pred, target, reduction='batchmean')*10
        return loss

    def __calc_CIoU(self, pred: torch.Tensor, target: torch.Tensor, positive_anchors: torch.Tensor):
        # pred.shape = [4*regmax, H, W]
        H=W=pred.shape[-1]
        st = self.imgsz//H
        gx,gy = torch.meshgrid(torch.arange(H, device=self.device),
                               torch.arange(W, device=self.device),
                               indexing='xy')
        gx = gx[positive_anchors[:, 0], positive_anchors[:, 1]].float() # [N]
        gy = gy[positive_anchors[:, 0], positive_anchors[:, 1]].float() # [N]
        gx = (gx+0.5)*st
        gy = (gy+0.5)*st

        pred = pred[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax,N]
        target = target[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax,N]

        pred = pred.reshape(4, self.regmax, -1) # [4,regmax,N]
        target = target.reshape(4, self.regmax, -1) # [4,regmax,N]

        pred = torch.softmax(pred, dim=1)        
        pred   = (pred  * self.proj).sum(1, keepdim=False) * st
        target = (target* self.proj).sum(1, keepdim=False) * st

        l,t,r,b = pred.split([1,1,1,1], dim=0) # [1,N]
        x1 = gx - l.squeeze(0)
        y1 = gy - t.squeeze(0)
        x2 = gx + r.squeeze(0)
        y2 = gy + b.squeeze(0)
        pred_xyxy = torch.stack([x1,y1,x2,y2], dim=1) # [N,4]

        l,t,r,b = target.split([1,1,1,1], dim=0) # [1,N]
        x1 = gx - l.squeeze(0)
        y1 = gy - t.squeeze(0)
        x2 = gx + r.squeeze(0)
        y2 = gy + b.squeeze(0)
        target_xyxy = torch.stack([x1,y1,x2,y2], dim=1) # [N,4]

        loss = torchvision.ops.complete_box_iou_loss(pred_xyxy, target_xyxy, reduction='mean')
        return loss

    def create_targets(self,
                       gt_boxes: List[torch.Tensor],    # [[N, 5]for each batch] 5: cls, cx, cy w, h (in 0-1)
                       preds: List[List[torch.Tensor]], # [[[4*regmax+nc, H, W] for each stride] for each batch]
                       debug: bool=False,
                       c2k: int=9,
                       c3k: int=20):
        labels: List[List[torch.Tensor]] = []
        positive_anchors: List[List[torch.Tensor]] = []
        batch_pt_st: List[List] = []

        for pred, gt_box in zip(preds, gt_boxes):  # batch
            if 0 in gt_box.shape:  # no object
                labels.append([torch.zeros((1, 4*self.regmax+self.nc, p.shape[-2], p.shape[-1]), device=self.device) for p in pred])
                positive_anchors.append([torch.empty((0,2), device=self.device, dtype=torch.long) for _ in range(len(pred))])
                batch_pt_st.append([[],[],[],[]] if debug else None)
                continue

            label, pos, pts = self._assign_lada_vectorized(gt_box, pred, debug, c2k=c2k, c3k=c3k)
            labels.append(label)
            positive_anchors.append(pos)
            batch_pt_st.append(pts if debug else None)

        return labels, positive_anchors, batch_pt_st

    @torch.no_grad()
    def _assign_lada_vectorized(self,
                                gt_boxes: torch.Tensor,       # [N,5] (cls,cx,cy,w,h) norm
                                preds: List[torch.Tensor],    # [[4*regmax+nc,H,W] for each stride]
                                debug: bool,
                                c2k: int,
                                c3k: int):
        """
        C1: EPCP area candidates
        C2: top-k candidates for each gt for each stride
        C3: top-k candidates for each gt (across all strides)
        Final: C3 ∩ (loss < avg_loss(C3))  [DLT per-GT]
        """
        device = self.device
        G = gt_boxes.shape[0]

        # stride index map
        st2idx = {st: i for i, st in enumerate(self.strides)}

        C1_points_by_st = [[] for _ in self.strides]
        C2_points_by_st = [[] for _ in self.strides]
        C3_points_by_st = [[] for _ in self.strides]
        FINAL_points_by_st = [[] for _ in self.strides]


        labels_per_stride: List[torch.Tensor] = []
        pos_per_stride: List[torch.Tensor] = []

        c2_candidates_per_st: dict[int, dict[str, torch.Tensor]] = {}

        color_map = {}
        if debug:
            for gb in gt_boxes:
                color_map[tuple(gb.tolist())] = np.random.randint(0, 255, 3).tolist()

        # C1 and C2: stride-based, C3: across all strides
        # 1) Collect C2 for each stride
        for p in preds:  # p: [4*regmax+nc, H, W]
            st = self.imgsz // p.shape[-1]
            cache = self.grid_cache[st]
            H, W = cache["H"], cache["W"]
            cx_map, cy_map = cache["cx"], cache["cy"]  # [H,W]

            # EPCP alanı (G,4) (xyxy, norm)
            r = math.sqrt(st / self.max_stride)
            pa_xyxy = torchvision.ops.box_convert(
                torch.stack([gt_boxes[:, 1], gt_boxes[:, 2],
                            gt_boxes[:, 3]*r, gt_boxes[:, 4]*r], dim=1),
                'cxcywh', 'xyxy'
            ).clamp_(0, 1)  # [G,4]

            # [1,H,W] → [G,H,W]
            CX = cx_map.unsqueeze(0).expand(G, H, W)
            CY = cy_map.unsqueeze(0).expand(G, H, W)
            x1, y1, x2, y2 = [pa_xyxy[:, k].view(G, 1, 1) for k in range(4)]
            inside = (CX > x1) & (CX < x2) & (CY > y1) & (CY < y2)  # [G,H,W]

            # stride index
            sidx = st2idx[st]

            if not inside.any():
                c2_candidates_per_st[st] = {
                    "i": torch.empty(0, dtype=torch.long, device=device),
                    "j": torch.empty(0, dtype=torch.long, device=device),
                    "cx": torch.empty(0, device=device),
                    "cy": torch.empty(0, device=device),
                    "gtid": torch.empty(0, dtype=torch.long, device=device),
                    "loss": torch.empty(0, device=device)
                }
                continue

            gt_id, ii, jj = inside.nonzero(as_tuple=True)  # [N_cand]
            lin = (ii * W + jj)                            # [N]
            anc_cx = CX[gt_id, ii, jj]
            anc_cy = CY[gt_id, ii, jj]

            C = p.shape[0]
            pred_flat = p.reshape(C, -1)                 # [4*regmax+nc, H*W]
            pred_cand = pred_flat.index_select(1, lin)   # [4*regmax+nc, N]

            # CLA: Lcls + 1.5*Lreg + deviation
            # Class Loss
            pred_cls = pred_cand[4*self.regmax:, :].T  # [N,nc]
            tgt_cls = torch.nn.functional.one_hot(gt_boxes[gt_id, 0].long(), self.nc).to(torch.float32)
            Lcls = torchvision.ops.sigmoid_focal_loss(pred_cls, tgt_cls, reduction='none').sum(1)  # [N]

            # GIoU loss
            pred_reg = pred_cand[:4*self.regmax, :].reshape(4, self.regmax, -1)
            pred_reg = torch.softmax(pred_reg, dim=1)
            ltrb = (pred_reg * self.proj).sum(1) * (st / self.imgsz)  # [4,N]
            l, t, r, b = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
            pred_xyxy = torch.stack([anc_cx - l, anc_cy - t, anc_cx + r, anc_cy + b], dim=1)  # [N,4]
            gt_xyxy = torchvision.ops.box_convert(gt_boxes[gt_id, 1:], 'cxcywh', 'xyxy')       # [N,4]
            Lreg = torchvision.ops.generalized_box_iou_loss(pred_xyxy, gt_xyxy, reduction='none')  # [N]

            # Deviation Loss
            gx1, gy1, gx2, gy2 = gt_xyxy.unbind(1)
            dev_h = (torch.abs((anc_cx - gx1) - (gx2 - anc_cx)) /
                    ((anc_cx - gx1) + (gx2 - anc_cx) + 1e-9))
            dev_v = (torch.abs((anc_cy - gy1) - (gy2 - anc_cy)) /
                    ((anc_cy - gy1) + (gy2 - anc_cy) + 1e-9))
            dev = torch.clamp(dev_h + dev_v - 1.0, min=0.0)
            loss = Lcls + 1.5*Lreg + dev  # [N]

            min_loss = torch.full((H*W,), float('inf'), device=device)
            min_loss = min_loss.scatter_reduce(0, lin, loss, reduce='amin', include_self=True)  # [H*W]
            mask = loss <= (min_loss[lin] + 1e-12)
            sel = torch.nonzero(mask, as_tuple=False).squeeze(1)
            order = torch.argsort(lin[sel])
            sel = sel[order]
            lin_sel = lin[sel]
            keep = torch.ones_like(lin_sel, dtype=torch.bool)
            keep[1:] = lin_sel[1:] != lin_sel[:-1] # remove duplicates
            sel = sel[keep]

            c1_i, c1_j = ii[sel], jj[sel]
            c1_cx, c1_cy = anc_cx[sel], anc_cy[sel]
            c1_gtid, c1_loss = gt_id[sel], loss[sel]

            if debug:
                for _cx, _cy, _gid in zip(c1_cx.tolist(), c1_cy.tolist(), c1_gtid.tolist()):
                    C1_points_by_st[sidx].append([_cx, _cy, color_map[tuple(gt_boxes[_gid].tolist())]])

            # C2: top-k for each GT (just for this stride)
            if c1_gtid.numel() == 0:
                c2_candidates_per_st[st] = {
                    "i": torch.empty(0, dtype=torch.long, device=device),
                    "j": torch.empty(0, dtype=torch.long, device=device),
                    "cx": torch.empty(0, device=device),
                    "cy": torch.empty(0, device=device),
                    "gtid": torch.empty(0, dtype=torch.long, device=device),
                    "loss": torch.empty(0, device=device)
                }
                continue

            sel_list = []
            for g in range(G):
                m = (c1_gtid == g)
                if not m.any():
                    continue
                l_ = c1_loss[m]
                kk = min(c2k, l_.numel())
                topk_idx = torch.topk(l_, k=kk, largest=False).indices
                idx_global = torch.nonzero(m, as_tuple=False).squeeze(1)[topk_idx]
                sel_list.append(idx_global)

            c2_idx = torch.cat(sel_list, 0) if len(sel_list) > 0 else torch.empty(0, dtype=torch.long, device=device)

            c2_i, c2_j = c1_i[c2_idx], c1_j[c2_idx]
            c2_cx, c2_cy = c1_cx[c2_idx], c1_cy[c2_idx]
            c2_gtid, c2_loss = c1_gtid[c2_idx], c1_loss[c2_idx]

            c2_candidates_per_st[st] = {
                "i": c2_i, "j": c2_j,
                "cx": c2_cx, "cy": c2_cy,
                "gtid": c2_gtid, "loss": c2_loss
            }

            if debug and c2_i.numel() > 0:
                for _cx, _cy, _gid in zip(c2_cx.tolist(), c2_cy.tolist(), c2_gtid.tolist()):
                    C2_points_by_st[sidx].append([_cx, _cy, color_map[tuple(gt_boxes[_gid].tolist())]])

        # 2) C3: top-k for each GT (across all strides)
        c3_selected_per_st = {st: {"i": [], "j": [], "cx": [], "cy": [], "gtid": [], "loss": []}
                            for st in self.strides}
        final_selected_per_st = {st: {"i": [], "j": [], "cx": [], "cy": [], "gtid": [], "loss": []}
                                for st in self.strides}

        for g in range(G):
            st_list, i_list, j_list, cx_list, cy_list, loss_list = [], [], [], [], [], []

            for st in self.strides:
                cand = c2_candidates_per_st.get(st, None)
                if cand is None or cand["i"].numel() == 0:
                    continue
                m = (cand["gtid"] == g)
                if not m.any():
                    continue

                cnt = int(m.sum().item())
                st_list.append(torch.full((cnt,), st, device=device, dtype=torch.long))
                i_list.append(cand["i"][m]);        j_list.append(cand["j"][m])
                cx_list.append(cand["cx"][m]);      cy_list.append(cand["cy"][m])
                loss_list.append(cand["loss"][m])

            if len(loss_list) == 0:
                continue

            st_all   = torch.cat(st_list, 0)
            i_all    = torch.cat(i_list, 0)
            j_all    = torch.cat(j_list, 0)
            cx_all   = torch.cat(cx_list, 0)
            cy_all   = torch.cat(cy_list, 0)
            loss_all = torch.cat(loss_list, 0)

            kk3 = min(c3k, loss_all.numel())
            c3_idx = torch.topk(loss_all, k=kk3, largest=False).indices

            # debug: C3 points
            if debug:
                for st in self.strides:
                    m3 = (st_all[c3_idx] == st)
                    if m3.any():
                        for _cx, _cy in zip(cx_all[c3_idx][m3].tolist(), cy_all[c3_idx][m3].tolist()):
                            C3_points_by_st[st2idx[st]].append([_cx, _cy, color_map[tuple(gt_boxes[g].tolist())]])

            avg_g = loss_all[c3_idx].mean() # DLT (per-GT)
            
            keep = loss_all[c3_idx] < avg_g
            final_idx = c3_idx[keep]

            for st in self.strides:
                m3 = (st_all[c3_idx] == st)
                if m3.any():
                    c3_selected_per_st[st]["i"].append(i_all[c3_idx][m3])
                    c3_selected_per_st[st]["j"].append(j_all[c3_idx][m3])
                    c3_selected_per_st[st]["cx"].append(cx_all[c3_idx][m3])
                    c3_selected_per_st[st]["cy"].append(cy_all[c3_idx][m3])
                    c3_selected_per_st[st]["gtid"].append(torch.full((int(m3.sum().item()),), g, device=device, dtype=torch.long))
                    c3_selected_per_st[st]["loss"].append(loss_all[c3_idx][m3])

                mf = (st_all[final_idx] == st)
                if mf.any():
                    final_selected_per_st[st]["i"].append(i_all[final_idx][mf])
                    final_selected_per_st[st]["j"].append(j_all[final_idx][mf])
                    final_selected_per_st[st]["cx"].append(cx_all[final_idx][mf])
                    final_selected_per_st[st]["cy"].append(cy_all[final_idx][mf])
                    final_selected_per_st[st]["gtid"].append(torch.full((int(mf.sum().item()),), g, device=device, dtype=torch.long))
                    final_selected_per_st[st]["loss"].append(loss_all[final_idx][mf])

                    if debug:
                        for _cx, _cy in zip(cx_all[final_idx][mf].tolist(), cy_all[final_idx][mf].tolist()):
                            FINAL_points_by_st[st2idx[st]].append([_cx, _cy, color_map[tuple(gt_boxes[g].tolist())]])

        def _cat_or_empty_long(lst):   # list[tensor] -> tensor[0] (long)
            return torch.cat(lst, 0) if len(lst) > 0 else torch.empty(0, dtype=torch.long, device=device)
        def _cat_or_empty_float(lst):  # list[tensor] -> tensor[0] (float)
            return torch.cat(lst, 0) if len(lst) > 0 else torch.empty(0, dtype=torch.float32, device=device)

        for p in preds:
            st = self.imgsz // p.shape[-1]
            H, W = self.grid_cache[st]["H"], self.grid_cache[st]["W"]

            cand = final_selected_per_st[st]
            i_fin = _cat_or_empty_long(cand["i"])
            j_fin = _cat_or_empty_long(cand["j"])
            cx_fin = _cat_or_empty_float(cand["cx"])
            cy_fin = _cat_or_empty_float(cand["cy"])
            gid_fin = _cat_or_empty_long(cand["gtid"])
            n_fin = int(i_fin.numel())

            L = torch.zeros((1, 4*self.regmax + self.nc, H, W), device=device)

            if n_fin > 0:
                gt_sel = gt_boxes[gid_fin]  # [N,5]
                xyxy = torchvision.ops.box_convert(gt_sel[:, 1:], 'cxcywh', 'xyxy')  # [N,4]
                x1, y1, x2, y2 = xyxy.unbind(1)

                scale = self.imgsz / st
                l = (cx_fin - x1) * scale
                t = (cy_fin - y1) * scale
                r = (x2 - cx_fin) * scale
                b = (y2 - cy_fin) * scale

                dist_ltrb = self._distribute_vec(torch.stack([l, t, r, b], dim=1), self.regmax)  # [N,4*regmax]
                cls_onehot = torch.nn.functional.one_hot(gt_sel[:, 0].long(), self.nc).to(torch.float32)
                label_vec = torch.cat([dist_ltrb, cls_onehot], dim=1)  # [N, 4*regmax+nc]

                lin_fin = (i_fin * W + j_fin)
                L2 = L.view(1, -1, H*W)[0]
                L2[:, lin_fin] = label_vec.T

                pos_per_stride.append(torch.stack([i_fin, j_fin], dim=1))
            else:
                pos_per_stride.append(torch.empty((0, 2), dtype=torch.long, device=device))

            labels_per_stride.append(L)

        points_and_strides = [C1_points_by_st, C2_points_by_st, C3_points_by_st, FINAL_points_by_st] if debug else None
        return labels_per_stride, pos_per_stride, points_and_strides

    @staticmethod
    def _distribute_vec(ltrb_pix: torch.Tensor, regmax: int) -> torch.Tensor:
        """
        DFL Distribution
        ltrb_pix: [N,4] (float, 0..∞), softmaxed output of regressors bins (regmax)
        """
        v = torch.clamp(ltrb_pix, 0, regmax-1-1e-4)              # [N,4]
        left = torch.floor(v).to(torch.long)                     # [N,4]
        right = torch.clamp(left + 1, max=regmax-1)              # [N,4]
        w_right = (v - left.to(v.dtype))                         # [N,4]
        w_left  = 1.0 - w_right                                  # [N,4]

        N = v.shape[0]
        out = v.new_zeros((N, 4, regmax))                        # [N,4,R]
        out.scatter_(2, left.unsqueeze(-1), w_left.unsqueeze(-1))
        out.scatter_(2, right.unsqueeze(-1), w_right.unsqueeze(-1))
        return out.view(N, 4*regmax)                             # [N,4R]

    def batch_eval(self, pred_boxes: List[torch.Tensor], gt_boxes: List[torch.Tensor]):
        # pred_boxes is [[N, 6],...] xyxy, cls_score, cls_id
        # gt_boxes is [[N, 5],...] cls_id, cxcywh (norm)
        batch_pred_dicts = []
        batch_target_dicts = []

        for pboxes in pred_boxes:
            # pboxes.shape is [N,6]
            if pboxes.numel() == 0:
                batch_pred_dicts.append({"boxes": torch.zeros((0, 4), device=self.device),
                                         "scores": torch.zeros((0,), device=self.device),
                                         "labels": torch.zeros((0,), device=self.device, dtype=torch.long)})
            else:
                xyxy = pboxes[:, :4]   # [N,4]
                scores = pboxes[:, 4]  # [N]
                cls = pboxes[:, 5]     # [N]
                batch_pred_dicts.append({"boxes": xyxy,
                                         "scores": scores,
                                         "labels": cls.long()})

        for tboxes in gt_boxes:
            if tboxes.numel() == 0:
                batch_target_dicts.append({"boxes": torch.zeros((0, 4), device=self.device),
                                           "labels": torch.zeros((0,), device=self.device, dtype=torch.long)})
            else:
                xyxy = torchvision.ops.box_convert(tboxes[:, 1:]*self.imgsz, "cxcywh", "xyxy")
                cls = tboxes[:, 0]
                batch_target_dicts.append({"boxes": xyxy,
                                           "labels": cls.long()})

        return batch_pred_dicts, batch_target_dicts
