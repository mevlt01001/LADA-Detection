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
        _, cx, cy, w, h = torch.unbind(gt_boxes*imgsz, dim=1)
        boxes = torch.stack((cx-w/2, cy-h/2, cx+w/2, cy+h/2), dim=1).tolist()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 1)

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
        label = f"cls:{names[cls_id.item()]} score:{cls_score.item():.2f}"
        img2 = cv2.putText(img2, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
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
                 mAP_50:float=None,
                 mAP_50_95:float=None,
                 val:bool=False
                 ):
    progress = math.floor((percentage)*total)
    bar = f"{'█'*progress}{'░'*(total-progress)}"
    msg = (
    f"{f'Validation | ' if val else ''}"
    f"{f'Epoch={epoch:03}' if epoch is not None else ''} "
    f"{f'Loss={loss:<7.2f}' if loss is not None else ''} "
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
              c3k:int=20   # per-stride
              ):

        train_names = np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(train_path,"images")))))
        valid_names = None if valid_path is None else np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(valid_path,"images")))))

        if self.last_ep/epoch > 0.95:
            self.last_ep = 0
        if len(train_names) > 0 and self.last_batch/len(train_names) > 0.95:
            self.last_batch = 0
            self.last_ep += 1

        model = self.model.train(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        if self.optim_state_dict is not None:
            optimizer.load_state_dict(self.optim_state_dict)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.008,
            epochs=max(1, epoch - self.last_ep),
            steps_per_epoch=math.ceil(len(train_names)/max(1,batch))
        )
        if self.sched_state_dict is not None:
            scheduler.load_state_dict(self.sched_state_dict)

        for ep in range(self.last_ep, epoch):
            losses = []
            last_batch = self.last_batch
            for i in range(last_batch, len(train_names), batch):
                # NOT calling empty_cache() here on purpose (big slowdown).
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

                loss = self.calc_loss(batch_preds_for_loss, targets, pos)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

                progress_bar(total=33,
                             percentage=(i+batch_size)/max(1,len(train_names)),
                             epoch=ep+1,
                             loss=loss.item(),
                             mAP_50=map50,
                             mAP_50_95=map50_95,
                             val=False
                             )

                # checkpoint (eval=False → BN/statics don’t move)
                model = model.train(False)
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
                    }, f"LADA_Last.pt")
                model = model.train(True)

            print(f" AvgLoss={sum(losses)/(len(losses)+1e-5):<6.4f}\n")
            self.map_metric.reset()

            # validation
            batch_eval_size = int(batch*2)
            if valid_path is not None and len(valid_names) > 0:
                with torch.inference_mode():
                    model = model.train(False)
                    for i in range(0, len(valid_names), batch_eval_size):
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

        return torch.cat(images, dim=0), bboxes

    def __load_image(self, path: str):
        img = torch.from_numpy(np.array(Image.open(path+".jpg").convert("RGB"))).to(device=self.device)
        assert img.ndim == 3, f"Image({path}.jpg) must have 3 dimensions"
        img = self.__preprocess(img)
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

    def __preprocess(self, x: torch.Tensor):
        x = x.to(torch.float32)
        x = x.permute(2, 0, 1).unsqueeze(0) # [H,W,3] -> [1,3,H,W]
        x = torch.nn.functional.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        x = x*(1.0/255.0)
        return x

    def calc_loss(self, preds: List[List[torch.Tensor]],
                  targets: List[List[torch.Tensor]],
                  positive_anchors: List[List[torch.Tensor]]):
        loss = 0.0
        for pred, target, pos in zip(preds, targets, positive_anchors):
            loss += self.__calc_loss(pred, target, pos)
        return loss/len(preds)

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
        w = [0.5, 1.5, 7.5]
        return cls_loss*w[0] + ciou_loss*w[1] + dfl_loss*w[2]

    def __calc_dfl(self, pred:torch.Tensor, target:torch.Tensor, positive_anchors:torch.Tensor):
        # pred: [4*regmax,H,W]
        pred = pred[:, positive_anchors[:, 0], positive_anchors[:, 1]]   # [4*regmax,N]
        target = target[:, positive_anchors[:, 0], positive_anchors[:, 1]]# [4*regmax,N]

        target = target.view(4, self.regmax, target.shape[-1]) # [4,regmax,N]
        pred = pred.view(4, self.regmax, pred.shape[-1])       # [4,regmax,N]
        pred = torch.log_softmax(pred, dim=1)
        loss = torch.nn.functional.kl_div(pred, target, reduction='batchmean')
        return loss

    def __calc_CIoU(self, pred: torch.Tensor, target: torch.Tensor, positive_anchors: torch.Tensor):
        # pred.shape = [4*regmax, H, W]
        st = self.imgsz//pred.shape[-1]
        gx,gy = torch.meshgrid(torch.arange(pred.shape[-1], device=self.device),
                               torch.arange(pred.shape[-1], device=self.device),
                               indexing='xy')
        gx = gx[positive_anchors[:, 0], positive_anchors[:, 1]].float() # [N]
        gy = gy[positive_anchors[:, 0], positive_anchors[:, 1]].float() # [N]
        gx = (gx+0.5)*st / self.imgsz  # normalize to 0..1
        gy = (gy+0.5)*st / self.imgsz

        pred = pred[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax,N]
        target = target[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax,N]

        pred = pred.reshape(4, self.regmax, -1) # [4,regmax,N]
        target = target.reshape(4, self.regmax, -1) # [4,regmax,N]

        pred = torch.softmax(pred, dim=1)
        pred = (pred*self.proj).sum(1, keepdim=False) / self.regmax  # [4,N] (scale ~ 0..1)
        target = (target*self.proj).sum(1, keepdim=False) / self.regmax # [4,N]

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

        loss = torchvision.ops.generalized_box_iou_loss(pred_xyxy, target_xyxy, reduction='mean')
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
        C1: EPCP area candiadates
        C2: top-k candidates for each gt for each stride
        C3: top-k candidates for each gt
        Final: cC3 candidates > avg_loss(C3)
        """
        G = gt_boxes.shape[0]
        device = self.device
        C1_points_by_st = []
        C2_points_by_st = []
        C3_points_by_st = []
        FINAL_points_by_st = []

        # Collect all labels and positive anchors
        labels_per_stride: List[torch.Tensor] = []
        pos_per_stride: List[torch.Tensor] = []
        all_losses_for_dlt = []

        c3_candidates_per_st = {}

        color_map = {}
        if debug:
            for gb in gt_boxes:
                color_map[tuple(gb.tolist())] = np.random.randint(0, 255, 3).tolist()

        for p in preds: # [[4*regmax+nc, H, W] for each stride]
            # p.shape = [4*regmax+nc, H, W]
            st = self.imgsz // p.shape[-1]
            cache = self.grid_cache[st]
            H, W = cache["H"], cache["W"]
            cx_map, cy_map = cache["cx"], cache["cy"]        # [H,W]
            # EPCP (G,4) (xyxy, norm)
            r = math.sqrt(st / self.max_stride)
            pa_xyxy = torchvision.ops.box_convert(
                torch.stack([gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]*r, gt_boxes[:,4]*r], dim=1),
                'cxcywh', 'xyxy'
            ).clamp_(0, 1)  # [G,4]

            # [1,H,W] → [G,H,W]
            CX = cx_map.unsqueeze(0).expand(G, H, W)
            CY = cy_map.unsqueeze(0).expand(G, H, W)
            x1,y1,x2,y2 = pa_xyxy[:,0].view(G,1,1), pa_xyxy[:,1].view(G,1,1), pa_xyxy[:,2].view(G,1,1), pa_xyxy[:,3].view(G,1,1)
            inside = (CX > x1) & (CX < x2) & (CY > y1) & (CY < y2)    # [G,H,W]

            if not inside.any():
                # Empty
                labels_per_stride.append(torch.zeros((1, 4*self.regmax+self.nc, H, W), device=device))
                pos_per_stride.append(torch.empty((0,2), device=device, dtype=torch.long))
                if debug:
                    C1_points_by_st.append([])
                    C2_points_by_st.append([])
                    C3_points_by_st.append([])
                    FINAL_points_by_st.append([])
                continue

            gt_id, ii, jj = inside.nonzero(as_tuple=True)   # [N_cand]
            N = gt_id.numel()
            lin = (ii * W + jj)         # [N]
            anc_cx = CX[gt_id, ii, jj]  # [N]
            anc_cy = CY[gt_id, ii, jj]  # [N]

            # Selected Grids Features [4*regmax+nc,N]
            C = p.shape[0]
            pred_flat = p.reshape(C, -1)    # [4*regmax+nc, H*W]
            pred_cand = pred_flat.index_select(1, lin)  # [4*regmax+nc, N]

            # CLA
            # cls Loss
            pred_cls = pred_cand[4*self.regmax:, :].T                 # [N,nc]
            tgt_cls = torch.nn.functional.one_hot(gt_boxes[gt_id,0].long(), self.nc).to(torch.float32)  # [N,nc]
            Lcls = torchvision.ops.sigmoid_focal_loss(pred_cls, tgt_cls, reduction='none').sum(1)       # [N]

            # reg → ltrb (0..1 norm)
            pred_reg = pred_cand[:4*self.regmax, :].reshape(4, self.regmax, N)
            pred_reg = torch.softmax(pred_reg, dim=1)
            ltrb = (pred_reg * self.proj).sum(1)*(st / self.imgsz)       # [4,N]
            l, t, r, b = ltrb[0], ltrb[1], ltrb[2], ltrb[3]            # [N]

            # pred boxes (norm)
            px1 = anc_cx - l
            py1 = anc_cy - t
            px2 = anc_cx + r
            py2 = anc_cy + b
            pred_xyxy = torch.stack([px1,py1,px2,py2], dim=1)         # [N,4]

            # gt boxes (norm)
            gt_xyxy = torchvision.ops.box_convert(gt_boxes[gt_id,1:], 'cxcywh', 'xyxy')  # [N,4]

            # Lreg = GIoU (LADA paper: GIoU)
            Lreg = torchvision.ops.generalized_box_iou_loss(pred_xyxy, gt_xyxy, reduction='none')  # [N]

            # dev
            gx1, gy1, gx2, gy2 = gt_xyxy.unbind(1)
            dev_h = (torch.abs((anc_cx - gx1) - (gx2 - anc_cx)) / ( (anc_cx - gx1) + (gx2 - anc_cx) + 1e-9 ))
            dev_v = (torch.abs((anc_cy - gy1) - (gy2 - anc_cy)) / ( (anc_cy - gy1) + (gy2 - anc_cy) + 1e-9 ))
            dev = torch.clamp(dev_h + dev_v - 1.0, min=0.0)           # [N], <=1 → 0, >1 → dev-1

            loss = Lcls + 1.5*Lreg + dev                              # [N]

            # Best N candidates
            # lin: [N], loss: [N]  → lin based min loss
            min_loss = torch.full((H*W,), float('inf'), device=device)
            min_loss = min_loss.scatter_reduce(0, lin, loss, reduce='amin', include_self=True)  # [H*W]
            mask = loss <= (min_loss[lin] + 1e-12)
            sel = torch.nonzero(mask, as_tuple=False).squeeze(1)      # [H*W] eleman
            # guarenteed for each (i,j): only one gt
            order = torch.argsort(lin[sel])
            sel = sel[order]
            lin_sel = lin[sel]
            keep = torch.ones_like(lin_sel, dtype=torch.bool)
            keep[1:] = lin_sel[1:] != lin_sel[:-1]
            sel = sel[keep]

            # C1:
            c1_i = ii[sel]; c1_j = jj[sel]; c1_lin = lin[sel]
            c1_cx = anc_cx[sel]; c1_cy = anc_cy[sel]
            c1_gtid = gt_id[sel]; c1_loss = loss[sel]                 # [M]

            if debug:
                pts = []
                for _i, _j, _cx, _cy, _gid in zip(c1_i.tolist(), c1_j.tolist(), c1_cx.tolist(), c1_cy.tolist(), c1_gtid.tolist()):
                    pts.append([_cx, _cy, color_map[tuple(gt_boxes[_gid].tolist())]])
                C1_points_by_st.append(pts)
            else:
                C1_points_by_st.append([])

            # C2: top-k for each gt
            if c1_gtid.numel() == 0:
                if debug:
                    C2_points_by_st.append([])
                else:
                    C2_points_by_st.append([])
                # Protect C3
                c3_candidates_per_st[st] = {
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
                m = (c1_gtid == g)  # Bool Tensor [M]
                if not m.any(): # İf not include True
                    continue
                l = c1_loss[m]  # 
                kk = min(c2k, l.numel())
                topk_idx = torch.topk(l, k=kk, largest=False).indices
                idx_global = torch.nonzero(m, as_tuple=False).squeeze(1)[topk_idx]
                sel_list.append(idx_global)
            if len(sel_list) > 0:
                c2_idx = torch.cat(sel_list, 0)
            else:
                c2_idx = torch.empty(0, dtype=torch.long, device=device)

            c2_i = c1_i[c2_idx]; c2_j = c1_j[c2_idx]
            c2_cx = c1_cx[c2_idx]; c2_cy = c1_cy[c2_idx]
            c2_gtid = c1_gtid[c2_idx]; c2_loss = c1_loss[c2_idx]

            if debug:
                pts = []
                for _i, _j, _cx, _cy, _gid in zip(c2_i.tolist(), c2_j.tolist(), c2_cx.tolist(), c2_cy.tolist(), c2_gtid.tolist()):
                    pts.append([_cx, _cy, color_map[tuple(gt_boxes[_gid].tolist())]])
                C2_points_by_st.append(pts)
            else:
                C2_points_by_st.append([])

            # C3 stride top-k
            if c2_loss.numel() > 0:
                kk3 = min(c3k, c2_loss.numel())
                c3_take = torch.topk(c2_loss, k=kk3, largest=False).indices
                c3_i = c2_i[c3_take]; c3_j = c2_j[c3_take]
                c3_cx = c2_cx[c3_take]; c3_cy = c2_cy[c3_take]
                c3_gtid = c2_gtid[c3_take]; c3_loss = c2_loss[c3_take]
            else:
                c3_i = torch.empty(0, dtype=torch.long, device=device)
                c3_j = torch.empty(0, dtype=torch.long, device=device)
                c3_cx = torch.empty(0, device=device)
                c3_cy = torch.empty(0, device=device)
                c3_gtid = torch.empty(0, dtype=torch.long, device=device)
                c3_loss = torch.empty(0, device=device)

            c3_candidates_per_st[st] = {
                "i": c3_i, "j": c3_j, "cx": c3_cx, "cy": c3_cy, "gtid": c3_gtid, "loss": c3_loss
            }
            all_losses_for_dlt.append(c3_loss)

            if debug:
                pts = []
                for _i, _j, _cx, _cy, _gid in zip(c3_i.tolist(), c3_j.tolist(), c3_cx.tolist(), c3_cy.tolist(), c3_gtid.tolist()):
                    pts.append([_cx, _cy, color_map[tuple(gt_boxes[_gid].tolist())]])
                C3_points_by_st.append(pts)
            else:
                C3_points_by_st.append([])

        # DLT
        if len(all_losses_for_dlt) > 0:
            all_losses = torch.cat([x for x in all_losses_for_dlt if x.numel() > 0], 0)
            avg_loss = (all_losses.mean().item() if all_losses.numel() > 0 else float('inf'))
        else:
            avg_loss = float('inf')

        # Fianl
        for p in preds:
            st = self.imgsz // p.shape[-1]
            H = self.grid_cache[st]["H"]; W = self.grid_cache[st]["W"]
            cand = c3_candidates_per_st.get(st, None)
            if cand is None or cand["i"].numel() == 0:
                labels_per_stride.append(torch.zeros((1, 4*self.regmax+self.nc, H, W), device=device))
                pos_per_stride.append(torch.empty((0,2), device=device, dtype=torch.long))
                if debug:
                    FINAL_points_by_st.append([])
                continue

            # DLT filter
            keep = cand["loss"] < avg_loss
            i_fin = cand["i"][keep]; j_fin = cand["j"][keep]
            cx_fin = cand["cx"][keep]; cy_fin = cand["cy"][keep]
            gid_fin = cand["gtid"][keep]
            n_fin = i_fin.numel()

            # label map
            L = torch.zeros((1, 4*self.regmax+self.nc, H, W), device=device)  # [1,C,H,W]

            if n_fin > 0:
                # ltrb dist
                gt_sel = gt_boxes[gid_fin] # [N,5]
                xyxy = torchvision.ops.box_convert(gt_sel[:,1:], 'cxcywh','xyxy')  # [N,4]
                x1,y1,x2,y2 = xyxy.unbind(1)
                scale = self.imgsz / st
                l = (cx_fin - x1) * scale
                t = (cy_fin - y1) * scale
                r = (x2 - cx_fin) * scale
                b = (y2 - cy_fin) * scale
                dist_ltrb = self._distribute_vec(torch.stack([l,t,r,b], dim=1), self.regmax)  # [N,4*regmax]

                cls_onehot = torch.nn.functional.one_hot(gt_sel[:,0].long(), self.nc).to(torch.float32)  # [N,nc]
                label_vec = torch.cat([dist_ltrb, cls_onehot], dim=1)  # [N, 4*regmax+nc]

                # scatter to [1,C,H*W]
                lin_fin = (i_fin * W + j_fin)  # [N]
                L2 = L.view(1, -1, H*W)[0]     # [C,H*W]
                L2[:, lin_fin] = label_vec.T

            labels_per_stride.append(L)
            pos_per_stride.append(torch.stack([i_fin, j_fin], dim=1) if n_fin>0 else torch.empty((0,2), device=device, dtype=torch.long))

            if debug:
                pts = []
                for _cx, _cy, _gid in zip(cx_fin.tolist(), cy_fin.tolist(), gid_fin.tolist()):
                    pts.append([_cx, _cy, color_map[tuple(gt_boxes[_gid].tolist())]])
                FINAL_points_by_st.append(pts)

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
