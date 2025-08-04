"""
# LADA Assignment Algorithm https://doi.org/10.3390/s23146306
# Training loss is different from paper
## Training Loss = λ1*Focal Loss + λ2*CIoU Loss + λ3*DFL | λ1:0.5, λ2:1.5, λ3:7.5
"""

import os, cv2
import sys
import torch
import torchvision
import numpy as np
from modules import Model
from PIL import Image
from collections import defaultdict
from torchmetrics.detection import MeanAveragePrecision 
import warnings

warnings.filterwarnings("ignore")

class Trainer:
    """
    This class is a trainer for `Model` class.\\
    It provides to train `Model` class with Lightweight Anchor Dynamic Assignment (LADA) Algorithm https://doi.org/10.3390/s23146306\\

    Args:
        model (Model): HybridNet and Head models which are processed ultralytics models
    Methods:
        crate_targets(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]): creates targets
    """
    def __init__(self, model:Model):

        self.model = model
        self.strides = model.backbone.strides
        self.imgsz = model.imgsz
        self.nc = model.nc
        self.regmax = model.regmax
        self.device = model.device
        self.proj = torch.arange(0, self.regmax, dtype=torch.float32, device=self.device).view(1, self.regmax, 1)

        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            backend='faster_coco_eval',
            class_metrics=False
            ).to(self.device)

    def train(self, epoch:int, batch:int, train_path: str, valid_path: str=None):

        train_names = np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(train_path,"images")))))
        valid_names = np.array(list(set(os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(valid_path,"images"))))) if valid_path is not None else None

        model = self.model.train(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for ep in range(epoch):
            losses = []

            for i in range(0, len(train_names), batch):
                batch_size = batch if i+batch < len(train_names) else len(train_names)-i
                optimizer.zero_grad()
                images, gt_boxes = self.__load_data(train_names[i:batch_size+i], train_path)
                pred_boxes, preds = model.forward(images) # [[B,4*regmax+nc,p3,p3],[B,4*regmax+nc,p4,p4],[B,4*regmax+nc,p5,p5],...]
                preds = [[p[idx] for p in preds] for idx in range(batch_size)]# [[[4*regmax+nc,p3,p3],[4*regmax+nc,p4,p4],[4*regmax+nc,p5,p5]], ...]
                targets, pos = self.create_targets(gt_boxes, preds)

                batch_pred_dicts, batch_target_dicts = self.batch_eval(pred_boxes, gt_boxes)
                self.map_metric.update(batch_pred_dicts, batch_target_dicts)
                batch_stats = self.map_metric.compute()
                map50 = batch_stats["map_50"].item()
                map50_95 = batch_stats['map'].item()
                self.map_metric.reset()

                loss = self.calc_loss(preds, targets, pos)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                total = 33
                progress = (batch_size+i)*total//len(train_names)  # dolu hücre sayısı
                bar = f"{'█'*progress}{'▒'*(total-progress)}"

                msg = (
                    f"Epoch={ep+1:03}/{epoch:03} "
                    f"Loss={loss.item():<6.2f} "
                    f"mAP@50={map50*100:<6.3f} "
                    f"mAP@50:95={map50_95*100:<6.3f} | "
                    f"{bar} %{(batch_size+i)/len(train_names)*100:6.2f}"
                )
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()
            
            if valid_path is not None:
                torch.cuda.empty_cache()
                model = model.train(False)
                
                pred_dicts = []
                target_dicts = []

                for i in range(0, len(valid_names), batch):
                    batch_size = batch if i+batch < len(valid_names) else len(valid_names)-i
                    images, gt_boxes = self.__load_data(valid_names[i:batch_size+i], valid_path)
                    pred_boxes = model.forward(images) # [B,4+nc,N]
                    batch_pred_dicts, batch_target_dicts = self.batch_eval(pred_boxes, gt_boxes)
                    pred_dicts.extend(batch_pred_dicts)
                    target_dicts.extend(batch_target_dicts)

                self.map_metric.update(pred_dicts, target_dicts)
                batch_stats = self.map_metric.compute()
                map50 = batch_stats["map_50"].item()
                map50_95 = batch_stats['map'].item()
                print(f" AvgLoss={sum(losses)/len(losses):<6.4f}\n\t"
                      f" Valid | mAP@50={map50*100:<6.3f} mAP@50:95={map50_95*100:<6.3f}\n")
                self.map_metric.reset()
                torch.cuda.empty_cache()
                model = model.train(True)
           

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
        if data.ndim == 1:
            data = data[None, :]
        elif data.ndim == 0:
            data = data[None, None]
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

        loss = torchvision.ops.complete_box_iou_loss(pred_xyxy, target_xyxy, reduction='mean')
        return loss

    def create_targets(self, gt_boxes:list[torch.Tensor], preds:list[list[torch.Tensor]]):
        labels:list[list[torch.Tensor]] = []
        positive_anchors:list[list[torch.Tensor]] = []
        for pred, gt_box in zip(preds, gt_boxes):
            if 0 in gt_box.shape:
                labels.append([torch.zeros((1, 4*self.regmax+self.nc, p.shape[-2], p.shape[-1]), device=self.device) for p in pred])
                positive_anchors.append([torch.empty((0,0), device=self.device) for _ in range(len(pred))])
                continue
            label, pos = self.__create_targets(gt_box, pred)
            labels.append(label)
            positive_anchors.append(pos)
        return labels, positive_anchors

    def __create_targets(self, gt_boxes:torch.Tensor, preds:list[torch.Tensor]):
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
        c2_assignments = self.__candidate2(c1_assignments) # EPCP areas best 9 anchor points assignment for each stride
        c3_assignments, avg_loss = self.__candidate3(c2_assignments) # Total best 20 anchor points assignment and average of its losses
        final_assignments = defaultdict(list) # it will use to apply Dynamic Loss Threshold, DLT

        for (st, (j, i), (cx, cy)), [(box, loss)] in c3_assignments.items():
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


        return labels, positive_anchors

    def __candidate3(self, assignments: dict, k: int=20):
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
        sorted_assignments = sorted(assignments.items(), key=lambda x: x[1][0][1])
        k = min(k, len(sorted_assignments))
        losses = []
        for (st, (j, i), (cx, cy)), [(box, loss)] in sorted_assignments[:k]:
            c3_assignments[(st, (j, i), (cx, cy))].append((box, loss))
            losses.append(loss)
        avg_loss = sum(losses)/len(losses) if len(losses) > 0 else 0
        return c3_assignments, avg_loss

    def __candidate2(self, assignments: dict, k: int=9):
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
            for (_st, (j, i), (cx, cy)), [(box, loss)] in assignments.items():
                if _st == st:
                    stride_groups.append(((_st, (j, i), (cx, cy)), (box, loss)))
            sorted_stride_groups = sorted(stride_groups, key=lambda x: x[1][1])
            k = min(k, len(sorted_stride_groups))
            for (st, (j, i), (cx, cy)), (box, loss) in sorted_stride_groups[:k]: # 
                c2_assignments[(st, (j, i), (cx, cy))].append((box, loss))

        return c2_assignments

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
        r = lambda s: np.sqrt((s*(1/max(self.strides)))) # EPCP scale factor. All anchors are positive for max(self.strides), other strides include same anchor points count.
        positive_areas = lambda s, gt:\
                            torchvision.ops.box_convert(
                                torch.tensor([gt[1], gt[2], gt[3]*r(s), gt[4]*r(s)]),
                                'cxcywh', 'xyxy'
                            ) # EPCP
        
        # Grid assignment
        for st in self.strides:
            for box in gt_boxes:                          
                pa = positive_areas(st, box) # GT Box positive area xyxy, PA cxcywh                
                for i in range(self.imgsz//st):
                    for j in range(self.imgsz//st):
                        cx = (j+0.5)*st/self.imgsz
                        cy = (i+0.5)*st/self.imgsz
                        if (pa[0].item() < cx < pa[2].item()) and (pa[1].item() < cy < pa[3].item()): # Check grid cell if in positive area
                            assignments[(st, (j, i), (cx, cy))].append((box, -1)) # Assign to grid cell         

            # Multiple Assignment Handling
            for (st, (j, i), (cx, cy)) , boxes in assignments.items():
                if len(boxes) > 1: # Multiple assignment
                    best_box = None
                    best_dist = float('inf')
                    for box,loss in boxes:
                        bcx = box[1]
                        bcy = box[2]
                        dist = (bcx - cx)**2 + (bcy - cy)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_box = box
                    assignments[(st, (j, i), (cx, cy))] = [(best_box, -1)]

        # Loss Calculation for preds tensor
        for pred in preds:
            _st = self.imgsz//pred.shape[-1]
            for (st, (j, i), (cx, cy)), [(box, loss)] in assignments.items(): # Assingments'e atama yapılmıyor.
                if st == _st:
                    loss = self.CLA(pred[:, i, j], box, (cx,cy), st)
                    assignments[(st, (j, i), (cx, cy))] = [(box, loss)]

        return assignments
       
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
        
        def dev(anchor_point:torch.Tensor, gt:torch.Tensor, stride:int, imgsz:int):
            cx = anchor_point[0]
            cy = anchor_point[1]
            box_xyxy = torchvision.ops.box_convert(gt[1:], 'cxcywh', 'xyxy').squeeze(0)
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

    def _decode_pred_batch(self, preds_boxes: list[torch.Tensor]):

        batch_dicts = []
       
        for pred in preds_boxes:
            # pred.shape is [N,6] xyxy, cls_score, cls_id
            xyxy = pred[:, :4] # [N, 4]
            _scores = pred[:, 4] # [N]
            cls = pred[:, 5] # [N]

            if xyxy.shape[0] == 0:
                batch_dicts.append({"boxes": torch.zeros((0, 4), device=self.device),
                                    "scores": torch.zeros((0,), device=self.device),
                                    "labels": torch.zeros((0,), device=self.device, dtype=torch.long)})
            else:
                batch_dicts.append({"boxes": xyxy,
                                    "scores": _scores,
                                    "labels": cls.long()})
        return batch_dicts
    
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

            
        