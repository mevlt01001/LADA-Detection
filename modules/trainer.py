"""
# LADA Assignment Algorithm https://doi.org/10.3390/s23146306
# Training loss is different from paper
## Training Loss = λ1*Focal Loss + λ2*CIoU Loss + λ3*DFL | λ1:0.5, λ2:1.5, λ3:7.5
"""
from typing import TYPE_CHECKING
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
                    gt_boxes: torch.Tensor,
                    imgsz: int,
                    strides: list[int],
                    pt_by_st: list[list[list[float, float, list[int, int, int]]]],
                    ):
    # image.shape = [3, H, W]
    # gt_boxes.shape = [N, 5]
    image = np.ascontiguousarray((image.permute(1,2,0)*255).to(device="cpu", dtype=torch.uint8).numpy())
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
            img_cp = cv2.putText(img_cp, str(len(candidates[st_idx])) + " points", (10, imgsz-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            frame[:, c_idx*imgsz:(c_idx+1)*imgsz, :] = img_cp
        frames[st_idx*imgsz:(st_idx+1)*imgsz, :, :] = frame
    ratio = int(imgsz/1)

    if (frames==0).all():
        for st_idx in range(len(strides)):
            frame = np.zeros((imgsz, imgsz*4, 3), dtype=np.uint8) # stride frame
            for c_idx in range(4):                
                frame[:, c_idx*imgsz:(c_idx+1)*imgsz, :] = image
            frames[st_idx*imgsz:(st_idx+1)*imgsz, :, :] = frame

    frames = cv2.resize(frames, (ratio*4, ratio*len(strides)))

    cv2.imshow("debug_frame", cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))
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
    This class is a trainer for `Model` class.\\
    It provides to train `Model` class with Lightweight Anchor Dynamic Assignment (LADA) Algorithm https://doi.org/10.3390/s23146306\\

    Args:
        model (Model): HybridNet and Head models which are processed ultralytics models
    Methods:
        crate_targets(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]): creates targets
    """
    def __init__(self, model:"Model"):

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

        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            backend='faster_coco_eval',
            class_metrics=False
            ).to(self.device)

    def train(self, 
              epoch:int, 
              batch:int, 
              train_path: str, 
              valid_path: str=None, 
              debug:bool=False,
              c2k = 9, # Best 9 anchors for each stride
              c3k = 20 # Best 20 anchors for all strides
            ):
        

        train_names = np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(train_path,"images")))))
        valid_names = np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(valid_path,"images"))))) if valid_path is not None else None
        
        if self.last_ep/epoch > 0.95:
            self.last_ep = 0
        if self.last_batch/len(train_names) > 0.95:
            self.last_batch = 0
            self.last_ep += 1

        model = self.model.train(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        if self.optim_state_dict is not None:
            optimizer.load_state_dict(self.optim_state_dict)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3,
            epochs=epoch - self.last_ep, steps_per_epoch=max(1, len(train_names)//batch)
        )
        if self.sched_state_dict is not None:
            scheduler.load_state_dict(self.sched_state_dict)

        for ep in range(self.last_ep,epoch):
            losses = []
            last_batch = self.last_batch
            for i in range(last_batch, len(train_names), batch):
                torch.cuda.empty_cache()
                batch_size = batch if i+batch < len(train_names) else len(train_names)-i
                optimizer.zero_grad()

                images, gt_boxes = self.__load_data(train_names[i:batch_size+i], train_path)
                pred_boxes, preds = model.forward(images) # [[B,4*regmax+nc,p3,p3],[B,4*regmax+nc,p4,p4],[B,4*regmax+nc,p5,p5],...]
                batch_preds_for_loss = [[p[idx] for p in preds] for idx in range(batch_size)]# [[[4*regmax+nc,p3,p3],[4*regmax+nc,p4,p4],[4*regmax+nc,p5,p5]], ...]
                
                with torch.no_grad():
                    batch_preds_for_assign = [[p[idx].detach() for p in preds] for idx in range(batch_size)]
                    targets, pos, batch_pt_st = self.create_targets(gt_boxes, batch_preds_for_assign, debug, c2k=c2k, c3k=c3k)
                    if debug:
                        debug_visualize(images[0], gt_boxes[0], self.imgsz, self.strides, batch_pt_st[0])

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
                             percentage=(i+batch_size)/len(train_names),
                             epoch=ep+1,
                             loss=loss.item(),
                             mAP_50=map50,
                             mAP_50_95=map50_95,
                             val=False
                             )

                
                model = model.train(False)
                torch.save({
                    "models": self.model.backbone.model_names,
                    "nc": self.nc,
                    "imgsz": self.imgsz,
                    "regmax": self.regmax,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "sched_state_dict": scheduler.state_dict(),
                    "last_epoch": ep,
                    "last_batch": i+batch
                    }, f"LADA_Last.pt")
                model = model.train(True)
            print(f" AvgLoss={sum(losses)/(len(losses)+10e-5):<6.4f}\n")                
            self.map_metric.reset()

            batch = int(batch*2)
            
            if valid_path is not None:
                with torch.inference_mode():
                    model = model.train(False)
                    for i in range(0, len(valid_names), batch):
                        torch.cuda.empty_cache()
                        batch_size = batch if i+batch < len(valid_names) else len(valid_names)-i
                        images, gt_boxes = self.__load_data(valid_names[i:batch_size+i], valid_path)
                        pred_boxes = model.forward(images) # [B,4+nc,N]
                        batch_pred_dicts, batch_target_dicts = self.batch_eval(pred_boxes, gt_boxes)
                        self.map_metric.update(batch_pred_dicts, batch_target_dicts)
                        
                        progress_bar(
                            val=True,
                            total=33,
                            percentage=(i+batch_size)/len(valid_names),
                            mAP_50=None,
                            mAP_50_95=None,
                        )
                    batch_stats = self.map_metric.compute()
                    map50 = batch_stats["map_50"].item()
                    map50_95 = batch_stats['map'].item()
                    print(f"  mAP_50={map50*100:<6.4f}  mAP_50_95={map50_95*100:<6.4f}\n")

                    torch.cuda.empty_cache()
                    model = model.train(True)
                
            self.map_metric.reset()

            batch = int(batch/2)

        model = model.train(mode=False)
        self.model = model
    
    def __load_data(self, names: list[str], path: str):
        if path is None:
            return None, None

        images = []
        bboxes = []
        for file_name in names:
            image = self.__load_image(os.path.join(path+"/images", file_name))
            bbox = self.__load_gt_boxes(os.path.join(path+"/labels", file_name))
            images.append(image)
            bboxes.append(bbox)

        return torch.cat(images, dim=0), bboxes

    def __load_image(self, path: str):
        # path is a path to an image
        img = torch.from_numpy(np.array(Image.open(path+".jpg").convert("RGB"))).to(device=self.device)
        assert img.ndim == 3, f"Image({path}.jpg) must have 3 dimensions"
        img = self.__preprocess(img)
        return img
    
    def __load_gt_boxes(self, path: str):
        data = np.loadtxt(path+".txt", dtype=np.float32)
        if data.size == 0:
            data = np.empty((0,5), dtype=np.float32)
        elif data.ndim == 1:
            data = np.reshape(data, (1,5))
        return torch.from_numpy(data).to(device=self.device)# [N, 5]

    def __preprocess(self, x: torch.Tensor):
        # x is an image shaped like (H, W, 3)
        # this func converts x to (1, 3, self.imgsz, self.imgsz)
        x = x.to(torch.float32)
        x = x.permute(2, 0, 1).unsqueeze(0) # [H,W,3] -> [1,3,H,W]
        x = torch.nn.functional.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        x = x*(1.0/255.0)

        return x
    
    def calc_loss(self,preds:list[list[torch.Tensor]], 
                  targets:list[list[torch.Tensor]], 
                  positive_anchors:list[list[torch.Tensor]]):
        loss = 0
        for pred, target, pos in zip(preds, targets, positive_anchors):
            loss += self.__calc_loss(pred, target, pos)
        return loss/len(preds)

    def __calc_loss(self,pred:list[torch.Tensor], 
                  targets:list[torch.Tensor], 
                  positive_anchors:list[torch.Tensor],                  
                  ):
        # Training wil generating in batch = 1
        # gt: [N, 5] (cls_id, cx, cy, w, h)
        # pred: [1, 4*regmax+nc, H, W]
        # targets: [1, 4*regmax+nc, H, W]
        # positive_anchors: [[H, W], ...]

        total_pos = [pos.shape for pos in positive_anchors]
        # print("total_pos : ", total_pos)

        pred_reg = [p[:4*self.regmax, :, :] for p in pred] # [4*regmax, H, W]
        truth_reg = [t[0, :4*self.regmax, :, :] for t in targets] # [4*regmax, H, W]
        pred_cls = [p[4*self.regmax:, :, :].reshape(self.nc, -1).T for p in pred] # [N, nc]
        truth_cls = [t[0, 4*self.regmax:, :, :].reshape(self.nc, -1).T for t in targets] # [N, nc]

        # Distributed Focal Loss
        dfl_loss = 0
        for p, t, pos in zip(pred_reg, truth_reg, positive_anchors):
            if pos.shape[0] == 0:
                continue
            dfl_loss += self.__calc_dfl(p, t, pos)

        # Calc CIoU loss
        ciou_loss = 0 # NAN HATASI BURDA GELİYOR.
        for p, t, pos in zip(pred_reg, truth_reg, positive_anchors):
            if pos.shape[0] == 0:
                continue
            ciou_loss += self.__calc_CIoU(p, t, pos)

        # Calc Clasification Focal Loss
        cls_loss = 0
        for p, t in zip(pred_cls, truth_cls):
            cls_loss += torchvision.ops.sigmoid_focal_loss(p, t, reduction='mean')

        # print(f"cls_loss: {cls_loss}, ciou_loss: {ciou_loss}, dfl_loss: {dfl_loss}")

        w = [0.5, 1.5, 7.5]

        return cls_loss*w[0] + ciou_loss*w[1] + dfl_loss*w[2]
    
    def __calc_dfl(self, pred:torch.Tensor, target:torch.Tensor, positive_anchors:torch.Tensor):
        # pred: [4*regmax,H,W]
        # taget: [4*regmax,H,w]

        pred = pred[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax, N]
        target = target[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax, N]

        target = target.view(4, self.regmax, target.shape[-1]) # [4,regmax,N]
        pred = pred.view(4, self.regmax, pred.shape[-1]) # [4,regmax,N]
        pred = torch.log_softmax(pred, dim=1) # [4,regmax,N]
        
        loss = torch.nn.functional.kl_div(pred, target, reduction='batchmean')
        return loss

    def __calc_CIoU(self, pred: torch.Tensor, target: torch.Tensor, positive_anchors: torch.Tensor):
        # pred.shape = [4*regmax, H, W]
        # target.shape = [4*regmax, H, W]
        st = self.imgsz//pred.shape[-1]
        gx,gy = torch.meshgrid(torch.arange(pred.shape[-1], device=self.device), 
                               torch.arange(pred.shape[-1], device=self.device), 
                               indexing='xy')

        gx = gx[positive_anchors[:, 0], positive_anchors[:, 1]] # [N]
        gy = gy[positive_anchors[:, 0], positive_anchors[:, 1]] # [N]
        gx = (gx.float()+0.5)*st 
        gy = (gy.float()+0.5)*st
        pred = pred[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax, N]
        target = target[:, positive_anchors[:, 0], positive_anchors[:, 1]] # [4*regmax, N]


        pred = pred.reshape(4, self.regmax, -1) # [4,regmax,N]
        target = target.reshape(4, self.regmax, -1) # [4,regmax,N]

        pred = torch.softmax(pred, dim=1)
        pred = (pred*self.proj).sum(1, keepdim=False)*st # [4,N]
        target = (target*self.proj).sum(1, keepdim=False)*st # [4,N]

        l,t,r,b = pred.split([1,1,1,1], dim=0) # [1,N]
        x1 = gx - l
        y1 = gy - t
        x2 = gx + r
        y2 = gy + b
        pred_xyxy = torch.cat([x1,y1,x2,y2], dim=0).T # [4,N] -> [N,4]


        l,t,r,b = target.split([1,1,1,1], dim=0) # [1,N]
        x1 = gx - l
        y1 = gy - t
        x2 = gx + r
        y2 = gy + b
        target_xyxy = torch.cat([x1,y1,x2,y2], dim=0).T # [4,N] -> [N,4]

        loss = torchvision.ops.generalized_box_iou_loss(pred_xyxy, target_xyxy, reduction='mean')
        return loss

    def create_targets(self, 
                       gt_boxes:list[torch.Tensor], 
                       preds:list[list[torch.Tensor]], 
                       debug=False,
                       c2k=9,
                       c3k=20):
        labels:list[list[torch.Tensor]] = []
        positive_anchors:list[list[torch.Tensor]] = []
        batch_pt_st: list[list] = []

        for pred, gt_box in zip(preds, gt_boxes): # batch
            if 0 in gt_box.shape: # no object
                labels.append([torch.zeros((1, 4*self.regmax+self.nc, p.shape[-2], p.shape[-1]), device=self.device) for p in pred])
                positive_anchors.append([torch.empty((0,0), device=self.device) for _ in range(len(pred))])
                batch_pt_st.append([])
                continue
            label, pos, pt_st = self.__create_targets(gt_box, pred, debug, c2k, c3k)
            labels.append(label)
            positive_anchors.append(pos)
            batch_pt_st.append(pt_st)

        return labels, positive_anchors, batch_pt_st

    def __create_targets(self, 
                         gt_boxes:torch.Tensor, 
                         preds:list[torch.Tensor], 
                         debug=False, 
                         c2k=9, 
                         c3k=20):
        """
        Creates targets with LADA Assignment Algorithm https://doi.org/10.3390/s23146306\\
        This function is a combination of *create_c1_targets*, *create_c2_targets* and *create_c3_targets*\\
        Additionally, it applies **Dynamic Loss Threshold(DLT)** over *create_c3_targets* by using average loss\\
        https://doi.org/10.3390/s23146306 --> 3.3. Dynamic Loss Threshold

        Args:
            gt_boxes (torch.Tensor): ground truth boxes shaped like (n,5) (cls_id,cx,cy,w,h), normalized[0-1]
            preds (list[torch.Tensor]): (..,p3,p4,p5,..) predictions [4*regmax+nc,H,W]

        Returns:
            labels (list[torch.Tensor]): targets for each stride (...,p3,p4,p5,..)
        """
        c1_assignments = self.__candidate1(gt_boxes, preds) # EPCP areas anchor points assignment for each stride        
        c2_assignments = self.__candidate2(c1_assignments, gt_boxes, k=c2k) # EPCP areas best 9 anchor points assignment for each gt_boxfor each stride
        c3_assignments, avg_loss = self.__candidate3(c2_assignments, k=c3k) # Best 20 anchor points assignment for each stride and average of its total losses
        final_assignments = defaultdict(list) # it will use to apply Dynamic Loss Threshold, DLT

        # c3[(_st, (j, i), (cx, cy))] = [box, loss]
        for (st, (j, i), (cx, cy)), [box, loss] in c3_assignments.items():
            if loss < avg_loss: # Dynamic Loss Threshold(DLT)
                final_assignments[(st, (j, i))] = [(cx, cy),(box, loss)]


        labels = [
            torch.zeros((1, 4*self.regmax+self.nc, self.imgsz//st, self.imgsz//st), device=self.device) 
            for st in self.strides
            ] # labels for each stride
        
        positive_anchors:list[torch.Tensor] = []

        for lb in labels:
            pos = []
            for (st, (j, i)), [(cx, cy), (gt_box, loss)] in final_assignments.items():
                if st == self.imgsz//lb.shape[-1]: # Check stride
                    pos.append([i, j])
                    cls_label = torch.nn.functional.one_hot(gt_box[0].long(), self.nc)
                    x1, y1, x2, y2 = torchvision.ops.box_convert(gt_box[1:], 'cxcywh', 'xyxy')
                    l = (cx - x1)*st
                    t = (cy - y1)*st
                    r = (x2 - cx)*st
                    b = (y2 - cy)*st
                    ltrb_dist = self.distribute(l, t, r, b)
                    label = torch.cat([ltrb_dist, cls_label], dim=0)
                    lb[0, :, i, j] = label

            positive_anchors.append(torch.tensor(pos, dtype=torch.long, device=self.device))

        if debug:
            box_and_color = {}
            for gt_box in gt_boxes:
                box_and_color[tuple(gt_box.tolist())] = np.random.randint(0, 255, 3).tolist()

            c1_points = []
            for _st in self.strides:
                st_points = []
                for (st, (j, i), (cx, cy)), [gt_box, loss] in c1_assignments.items():
                    if st == _st:
                        st_points.append([cx, cy, box_and_color[tuple(gt_box.tolist())]])
                c1_points.append(st_points)

            c2_points = []
            for _st in self.strides:
                st_points = []
                for (st, (j, i), (cx, cy)), [gt_box, loss] in c2_assignments.items():
                    if st == _st:
                        st_points.append([cx, cy, box_and_color[tuple(gt_box.tolist())]])
                c2_points.append(st_points)

            c3_points = []
            for _st in self.strides:
                st_points = []
                for (st, (j, i), (cx, cy)), [gt_box, loss] in c3_assignments.items():
                    if st == _st:
                        st_points.append([cx, cy, box_and_color[tuple(gt_box.tolist())]])
                c3_points.append(st_points)

            final_points = []
            for _st in self.strides:
                st_points = []
                for (st, (j, i)), [(cx, cy), (gt_box, loss)] in final_assignments.items():
                    if st == _st:
                        st_points.append([cx, cy, box_and_color[tuple(gt_box.tolist())]])
                final_points.append(st_points)

            points_and_strides = [c1_points, c2_points, c3_points, final_points]

            return labels, positive_anchors, points_and_strides

        return labels, positive_anchors, None

    def __candidate3(self, assignments: dict, k: int=20):
        """
        Select best k anchor points for each stride from c2_assignments

        Args:
            assignments (defaultdict(list)): c2_assignments

        Returns:
            c3_assignments (defaultdict(list)): Selected k anchor points for each stride from c2_assignments
            avg_loss (float): average of its losses
        """
        # c2[(st, (j, i), (cx, cy))] = [(box, loss)]

        c3 = defaultdict(list)
        losses = []
        for st in self.strides:
            stride_group = []
            for (_st, (j, i), (cx, cy)), [box, loss] in assignments.items():
                if _st == st:
                    stride_group.append(((_st, (j, i), (cx, cy)), [box, loss]))

            end = min(k, len(stride_group))
            stride_group = sorted(stride_group, key=lambda x:x[1][1])
            
            for (_st, (j, i), (cx, cy)), [box, loss] in stride_group[:end]:
                c3[(_st, (j, i), (cx, cy))] = [box, loss]
                losses.append(loss)

        avg_loss = sum(losses)/(len(losses)+1e-6)

        return c3, avg_loss

    def __candidate2(self, c1_assignments: dict, gt_boxes: torch.Tensor, k: int=9):
        """
        Select best k anchor points for each gtbox for each stride from c1_assignments

        Args:
            assignments (defaultdict(list)): c1_assignments
            gt_boxes (torch.Tensor): ground truth boxes
            k (int, optional): Number of point for each gt_box for each stride. Default is 9.

        Returns:
            c2_assignments (defaultdict(list)): selected best k anchor points for each gt_box for each stride from c1_assignments
        """
        # c1[(st, (j, i), (cx, cy))] = [best_box, best_loss]
        # select best k for each stride

        c2 = defaultdict(list)
                
        for st in self.strides:
            for box in gt_boxes:
                boxes_groups = []
                for (_st, (j, i), (cx, cy)), [_box, loss] in c1_assignments.items():
                    if _st == st and _box.tolist() == box.tolist():
                        boxes_groups.append(((_st, (j, i), (cx, cy)), (box, loss)))

                end = min(k, len(boxes_groups))
                sorted_boxes_groups = sorted(boxes_groups, key=lambda x: x[1][1])
                for (_st, (j, i), (cx, cy)), (box, loss) in sorted_boxes_groups[:end]:
                    c2[(st, (j, i), (cx, cy))] = [box, loss]
                    
        return c2

    def __candidate1(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]):
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
        def EPCP(s, gt):
            # gt: [cls,cx,cy,w,h] (norm)
            r = math.sqrt(s/max(self.strides))
            pa = torchvision.ops.box_convert(
                torch.tensor([gt[1], gt[2], gt[3]*r, gt[4]*r], dtype=torch.float32, device=self.device),
                'cxcywh', 'xyxy').clamp(0, 1)
            return pa
        
        # Grid assignment
        for st in self.strides:
            for box in gt_boxes:
                pa = EPCP(st, box) # GT Box positive area xyxy, EPCP
                i0 = int(torch.floor(pa[1]*self.imgsz/st))
                j0 = int(torch.floor(pa[0]*self.imgsz/st))
                i1 = int(torch.ceil(pa[3]*self.imgsz/st))
                j1 = int(torch.ceil(pa[2]*self.imgsz/st))
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        cx = (j+0.5)*st/self.imgsz
                        cy = (i+0.5)*st/self.imgsz
                        if (pa[0].item() < cx < pa[2].item()) and (pa[1].item() < cy < pa[3].item()): # Check grid cell if in positive area
                            assignments[(st, (j, i), (cx, cy))].append((box)) # Assign to grid cell         

        # Loss assignment
        c1 = defaultdict(list)
        for pred in preds:
            _st = self.imgsz//pred.shape[-1]
            for (st, (j, i), (cx, cy)), boxes in assignments.items():
                if st == _st:
                    best_box = None
                    best_loss = float('inf')
                    for box in boxes:
                        loss = self.CLA(pred[:, i, j], box, (cx,cy), st)
                        if loss < best_loss:
                            best_loss = loss
                            best_box = box
                    c1[(st, (j, i), (cx, cy))] = [best_box, best_loss]
        
        return c1
       
    def CLA(self, pred:torch.Tensor, gt:torch.Tensor, anchor_point:tuple[int,int], stride:int):
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
        
        def dev(anchor_point:tuple, gt:torch.Tensor, stride:int, imgsz:int):
            cx = anchor_point[0]
            cy = anchor_point[1]
            box_xyxy = torchvision.ops.box_convert(gt[1:], 'cxcywh', 'xyxy')
            l = cx - box_xyxy[0]
            t = cy - box_xyxy[1]
            r = box_xyxy[2] - cx
            b = box_xyxy[3] - cy
            hdev = abs(l - r) / (l + r)
            vdev = abs(t - b) / (t + b)
            dev = hdev + vdev
            if 0<=dev.item()<=1: return 0
            elif 1<dev.item(): return dev - 1

        pred_classes = pred[self.regmax*4:]
        gt_classes = torch.nn.functional.one_hot(gt[0].long(), num_classes=self.nc).to(torch.float32)

        pred_reg = pred[:self.regmax*4].reshape(4, self.regmax).softmax(dim=1)
        
        ltrb = torch.sum(pred_reg*self.proj.squeeze(-1), dim=1)*stride/self.imgsz
        xyxy_pred = torch.tensor([
            anchor_point[0]-ltrb[0], 
            anchor_point[1]-ltrb[1], 
            anchor_point[0]+ltrb[2], 
            anchor_point[1]+ltrb[3]], 
            device=ltrb.device)
        xyxy_gt = torchvision.ops.box_convert(gt[1:], 'cxcywh', 'xyxy')
        Lreg = torchvision.ops.ciou_loss.complete_box_iou_loss(xyxy_pred, xyxy_gt, reduction="sum")
        Lcls = torchvision.ops.focal_loss.sigmoid_focal_loss(pred_classes, gt_classes, alpha=0.25, gamma=2, reduction="sum")
        Ldev = dev(anchor_point, gt, stride, self.imgsz)
        loss = Lcls + 1.5*Lreg + Ldev
        # print(f"Lcls = {Lcls}, Lreg = {Lreg}, Ldev = {Ldev}, CLA = {loss}")
        # print("pred_reg :", pred)
        # print("ltrb :", ltrb)
        # print("xyxy_pred raw :", xyxy_pred)
        # print("xyxy_gt :", xyxy_gt)
        # w = xyxy_pred[2] - xyxy_pred[0]
        # h = xyxy_pred[3] - xyxy_pred[1]
        # print("w,h =", w.item(), h.item())
        return loss
            
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
            v = torch.clamp(val, 0, self.regmax-1-10e-4)
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
    
    def batch_eval(self, pred_boxes: list[torch.Tensor], gt_boxes:list[torch.Tensor]):
        # pred_boxes is [[N, 6],...] xyxy, cls_score, cls_id
        # gt_boxes is [[N, 5],...] cls_id, xyxy
        batch_pred_dicts = []
        batch_target_dicts = []

        for pboxes in pred_boxes:
            # pboxes.shape is [N,6]
            xyxy = pboxes[:, :4] # [N, 4]
            scores = pboxes[:, 4] # [N]
            cls = pboxes[:, 5] # [N]

            if xyxy.shape[0] == 0:
                batch_pred_dicts.append({"boxes": torch.zeros((0, 4), device=self.device),
                                    "scores": torch.zeros((0,), device=self.device),
                                    "labels": torch.zeros((0,), device=self.device, dtype=torch.long)})
            else:
                batch_pred_dicts.append({"boxes": xyxy,
                                    "scores": scores,
                                    "labels": cls.long()})
        for tboxes in gt_boxes:
            # gboxes.shape is [N,5]
            if tboxes.numel() == 0:
                batch_target_dicts.append({"boxes": torch.zeros((0, 4), device=self.device),
                                    "labels": torch.zeros((0,), device=self.device, dtype=torch.long)})
            else:
                    
                xyxy = torchvision.ops.box_convert(tboxes[:, 1:]*self.imgsz, "cxcywh", "xyxy") # [N, 4]
                cls = tboxes[:, 0] # [N]

                batch_target_dicts.append({"boxes": xyxy,
                                    "labels": cls.long()})
                
        return batch_pred_dicts, batch_target_dicts

            
        