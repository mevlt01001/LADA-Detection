# LADA Assignment Algorithm https://doi.org/10.3390/s23146306
# Training loss is different from paper
# Training Loss = λ1*Focal Loss + λ2*CIoU Loss + λ3*DFL | λ1:0.5, λ2:1.5, λ3:7.5

import torch
import torchvision
import numpy as np
from modules import Model
from collections import defaultdict

class Trainer:
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
        
    def DFL_decode(self, pred:list[torch.Tensor]):
        # pred: list of tensors shaped like [1,4*regmax+nc,H,W]
        # TODO: returns default dict, keys are (stride, (row, col)) and value is (cx, cy, w, h) normalized 0-1
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
        c3_assignments, avg_loss = self.create_c3_targets(c2_assignments) # Total best 20 anchor points assignment and average of its losses
        final_assignments = defaultdict(list) # it will use to apply Dynamic Loss Threshold, DLT
        
        for (st, (j, i), (cx, cy)), (box, loss) in c3_assignments.items():
            if loss < avg_loss: # Dynamic Loss Threshold(DLT)
                final_assignments[(st, (j, i), (cx, cy))].append((box, loss))
        
        labels = [torch.zeros((1, 4*self.regmax+self.nc, self.imgsz//st, self.imgsz//st), device=self.device) for st in self.strides] # labels for each stride
        
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
        r = lambda s: np.sqrt((s*(1/max(self.strides)))) # EPCP scale factor. All anchors are positive for max(self.strides), other strides include same anchor points count.
        positive_areas = lambda s, gt:\
                            torchvision.ops.box_convert(
                                torch.tensor([[gt[1], gt[2], gt[3]*r(s), gt[4]*r(s)]]),
                                'xyxy', 'cxcywh'
                            ) # EPCP
        
        # Grid assignment
        for st in self.strides:
            for box in gt_boxes:
                pa = positive_areas(st, box) # GT Box positive area xyxy, PA cxcywh
                for j in range(self.imgsz//st):
                    for i in range(self.imgsz//st):
                        cx = (i+0.5)*st/self.imgsz
                        cy = (j+0.5)*st/self.imgsz
                        if (pa[0] < cx < pa[2]) and (pa[1] < cy < pa[3]): # Check grid cell if in positive area
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
        ltrb = torch.sum(pred_reg*proj, dim=0)*stride/self.imgsz
        xyxy_pred = torch.tensor([
            anchor_point[0]-ltrb[0], 
            anchor_point[1]-ltrb[1], 
            anchor_point[0]+ltrb[2], 
            anchor_point[1]+ltrb[3]], 
            device=ltrb.device)
        xyxy_gt = torchvision.ops.box_convert(gt[1:], 'cxycwh', 'xyxy')[0]

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
