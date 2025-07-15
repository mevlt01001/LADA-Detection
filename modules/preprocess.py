import torch
import torch.nn.functional as F

class Preprocess(torch.nn.Module):
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
        x = torch.cat(
            [
                x[:, :, 2:],
                x[:, :, 1:2],
                x[:, :, 0:1],
            ],
            dim=-1
        )
        # shape conversion: (H,W,3) -> (1,3,imgsz,imgsz)/255.0
        x = x.permute(2, 0, 1).unsqueeze(0) # [H,W,3] -> [1,3,H,W]
        x = F.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear")
        x = x*(1.0/255.0)
        return x
