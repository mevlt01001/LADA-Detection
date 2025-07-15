# in the next step, DFL will work one step. Now, this calculation spereted for each feature map
import torch

class DFL(torch.nn.Module):
    def __init__(self, regmax, nc, imgsz):
        super(DFL, self).__init__()
        self.regmax = regmax #
        self.nc = nc
        self.imgsz = imgsz # 
        self.proj = torch.arange(1,regmax+1, dtype=torch.float32).view(1,1, regmax, 1, 1)

    def forward(self, x:list[torch.Tensor]):
        # x = [[1,4*regmax+nc,8k,8k], [1,4*regmax+nc,4k,4k], [1,4*regmax+nc,2k,2k], ...]
        outs = []
        for pred in x:
            # pred.shape is [1,4*regmax+nc,H,W]
            B, C, H, W = pred.shape
            st = self.imgsz//H

            reg = pred[:, :4*self.regmax, ...].view(B, 4, self.regmax, H, W) # [B,4,regmax,H,W]
            cls = pred[:, 4*self.regmax:, ...] # [B,nc,H,W]

            cls = cls.softmax(1) # Get Class Probabilities Distribution
            
            reg = reg.softmax(2) # Get LTRB Regression Probabilities (for regmax) Distribution
            reg = (reg * self.proj.to(pred.device)).sum(2, keepdim=False)*st/self.imgsz # [B,4,H,W] Get LTRB Regression distance presented hom much stride distanca to pixel, normalized 0-1.
            
            gcx, gcy = torch.meshgrid(torch.arange(W, device=pred.device), torch.arange(H, device=pred.device), indexing='xy')
            gcx = gcx.view(1, H, W).expand(B, H, W).float() # [B,H,W]
            gcy = gcy.view(1, H, W).expand(B, H, W).float() # [B,H,W]
            gcx = (gcx+0.5)/st # grid center x points [0+(0.5/stride), 1-(0.5/stride)]
            gcy = (gcy+0.5)/st # grid center y points [0+(0.5/stride), 1-(0.5/stride)]
            
            l = reg[:, 0, ...] # [B,H,W]
            t = reg[:, 1, ...] # [B,H,W]
            r = reg[:, 2, ...] # [B,H,W]
            b = reg[:, 3, ...] # [B,H,W]

            x1 = gcx - l
            x2 = gcx + r
            y1 = gcy - t
            y2 = gcy + b

            reg = torch.stack((x1, y1, x2, y2), 1) # [B,4,H,W]
            out = torch.cat((reg, cls), 1) # [B,4+nc,H,W]
            out = torch.reshape(out, (B, 4+self.nc, -1)) # [B,4*regmax+nc,H*W]
            outs.append(out)

        out = torch.cat(outs, -1) # [B,4+nc,k*H*W]
        return out
    
