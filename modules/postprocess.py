import torch, torchvision

class Postprocess(torch.nn.Module):
    def __init__(self, score_thres=None, iou_thres=0.5):
        super().__init__()
        self.score_thres = score_thres
        self.iou_thres = iou_thres
    
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

        x = x.permute(0, 2, 1) # x.shape is (B, N, 4+nc)
        xyxy, scores = torch.split(x, (4, x.shape[2]-4), dim=2) # xyxy.shape is (B, N, 4), cls.shape is (B,N, nc)

        score_vals, cls_ids = scores.max(2) # score_vals.shape is (B,N), cls_ids.shape is (B,N)
        mask = score_vals > 0.005 if self.score_thres is None else score_vals > self.score_thres  # mask.shape is (B,N)

        out = []

        for _xyxy, _score_vals, _cls_ids, _mask in zip(xyxy, score_vals, cls_ids, mask):
            _xyxy = _xyxy[_mask]
            _score_vals = _score_vals[_mask]
            _cls_ids = _cls_ids[_mask]
            selected_idx = torchvision.ops.nms(_xyxy, _score_vals, iou_threshold=self.iou_thres)
            _xyxy = _xyxy[selected_idx]
            _score_vals = _score_vals[selected_idx]
            _cls_ids = _cls_ids[selected_idx]
            out.append(torch.cat([_xyxy, _score_vals.unsqueeze(-1), _cls_ids.unsqueeze(-1)], 1))

        return out
           