import torch, torchvision

class Postprocess(torch.nn.Module):
    def __init__(self, score_thres=None, iou_thres=0.5, max_det=300):
        super().__init__()
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
    
    @torch.no_grad()
    def forward(self, x:torch.Tensor):
        """
        Args:
            x (torch.Tensor): DFL-Based regression and classification outputs for each feature pyramid output.\
                The shape of each output is (B,4*regmax+nc,H,W)

        Returns:
            output (list[torch.Tensor]): List of nms outputs for each batch. [[N, 6(xyxy, score, cls)], [N, 6], [N, 6], ...]
        """
        # x.shape is (B, 4+nc, N)
        x = x.permute(0, 2, 1)
        xyxy, scores = torch.split(x, (4, x.shape[2]-4), dim=2)

        score_vals, cls_ids = scores.max(2)
        mask = score_vals > (0.005 if self.score_thres is None else self.score_thres)

        out = []
        for _xyxy, _score_vals, _cls_ids, _mask in zip(xyxy, score_vals, cls_ids, mask):
            _xyxy = _xyxy[_mask]
            _score_vals = _score_vals[_mask]
            _cls_ids = _cls_ids[_mask]
            if _xyxy.numel() == 0:
                out.append(torch.zeros((0,6), device=_xyxy.device))
                continue
            # class-aware NMS
            keep = torchvision.ops.batched_nms(_xyxy, _score_vals, _cls_ids, self.iou_thres)
            keep = keep[: self.max_det]  # per-image top-K
            _xyxy = _xyxy[keep]
            _score_vals = _score_vals[keep]
            _cls_ids = _cls_ids[keep]
            out.append(torch.cat([_xyxy, _score_vals.unsqueeze(-1), _cls_ids.unsqueeze(-1)], 1))

        return out
           