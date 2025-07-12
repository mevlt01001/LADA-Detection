from . import *

class DFL(torch.nn.Module):
    def __init__(self, reg_max, nc, imgsz):
        super(DFL, self).__init__()
        self.reg_max = reg_max #
        self.nc = nc
        self.imgsz = imgsz # 
        self.proj = torch.arange(1,reg_max+1, dtype=torch.float32)

    def forward(self, x:list[torch.Tensor]):
        # x = [[1,4*regmax+nc,8k,8k], [1,4*regmax+nc,4k,4k], [1,4*regmax+nc,2k,2k], ...]
        for pred in x:
            # pred.shape is [1,4*regmax+nc,H,W]
            B, C, H, W = pred.shape
            st = self.imgsz//H

            reg = pred[:, :4*self.reg_max, ...].view(B, 4, self.reg_max, H, W) # [B,4,regmax,H,W]
            cls = pred[:, self.reg_max:, ...] # [B,nc,H,W]

            cls = cls.softmax(1) # Get Class Probabilities Distribution
            
            reg = reg.softmax(2) # Get LTRB Regression Probabilities (for regmax) Distribution
            reg = (reg * self.proj.to(pred.device)).sum(2, keepdim=False)*st/self.imgsz # [B,4,H,W] Get LTRB Regression distance
            # TODO: ltrb to xywh

        return None        