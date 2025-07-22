import torch, torchvision

class Postprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x:torch.Tensor):
        # x.shape is (B, 4+nc, N)

        x = x.permute(0, 2, 1).squeeze(0) # x.shape is (N, 4+nc)
        x = x[:, :5] # x.shape is (N, 4)
        xyxy, cls = torch.split(x, (4, 1), dim=-1) # xyxy.shape is (N, 4), cls.shape is (N, 1)
        selected_idx = torchvision.ops.nms(xyxy, cls.squeeze(-1), iou_threshold=0.5)
        xyxy = xyxy[selected_idx]
        cls = cls[selected_idx]
        out = torch.cat((xyxy, cls), dim=-1) # out.shape is (N, 5)
        score_mask = out[:, 4] > 0.5
        out = out[score_mask]
        return out