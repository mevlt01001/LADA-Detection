import torch

class DFL(torch.nn.Module):
    """
    This class process **DFL-Based** and **anchor-free** regression and classification layers outputs which is like YOLOv11

    Args:
        regmax (int): maximum number of prob bins
        nc (int): number of classes
        imgsz (int): input image size
        grid_sizes (list[int]): the field of **`HybridBody`**
        device (torch.device): device
        onnx_nms_out (bool): whether to split regression and classification outputs (xyxy, yxyx, scores)

    """
    def __init__(self, 
            regmax, 
            nc, 
            imgsz, 
            grid_sizes: list[int], 
            device=torch.device("cpu"),
            onnx_nms_out=False
            ):
        
        super(DFL, self).__init__()
        self.regmax = regmax
        self.nc = nc
        self.imgsz = imgsz
        self.grid_sizes = grid_sizes
        self.split = onnx_nms_out
        self.proj = torch.arange(0,regmax, dtype=torch.float32, device=device).view(1, 1, regmax, 1)
        strides = []
        for gs in self.grid_sizes:
            st, repeat = self.imgsz//gs, gs**2            
            strides.extend([st]*repeat)
        self.strides = torch.tensor(strides, device=device)

    def forward(self, x:list[torch.Tensor]):
        """
        Args:
            x (list[torch.Tensor]): DFL-Based regression and classification outputs for each feature pyramid output.\\
            The shape of each output is (B, 4*regmax+nc, h, w)

        Returns:
            output (torch.Tensor):
            Concantenated regression and classification outputs (B, 4*regmax+nc, N)

        # Algorithm:
            - Concatenate all outputs
            - Split regression and classification outputs
                - Regression Part:
                    - Calculate centers for each feature pyramid output
                    - Apply softmax to distrubuted regression (ltrb) outputs as regmax as far
                    - Convert ltrb to xyxy
                - Classification Part:
                    - Apply sigmoid to classification outputs
            - Concatenate regression and classification outputs
        """
        # x = [[1,4*regmax+nc,8k,8k], [1,4*regmax+nc,4k,4k], [1,4*regmax+nc,2k,2k], ...]

        centers = [
            torch.meshgrid(torch.arange(gs), torch.arange(gs), indexing='xy')
            for gs in self.grid_sizes
        ]

        gx = [(c[0].float()+0.5).reshape(-1) for c in centers]
        gy = [(c[1].float()+0.5).reshape(-1) for c in centers]

        gx = torch.cat(gx, dim=0).to(x[0].device)*self.strides
        gy = torch.cat(gy, dim=0).to(x[0].device)*self.strides

        B, *_ = x[0].shape

        # flat all outputs
        x = [torch.reshape(x, (B, -1, x.shape[-1]**2)) for x in x]
        # x = [[1,4*regmax+nc,8k*8k], [1,4*regmax+nc,4k*4k], [1,4*regmax+nc,2k*2k], ...]
        combined = torch.cat(x, dim=-1)
        reg, cls = torch.split(combined, [4*self.regmax, self.nc], dim=1)
        cls = cls.sigmoid()
        reg = reg.reshape(B, 4, self.regmax, reg.shape[-1]).softmax(2) # [B,4,regmax,N], softmaxed regmax channel
        # reg = self.conv(reg).squeeze(2) slower way
        reg = (reg*self.proj).sum(2)*self.strides
        # convert ltrb to xyxy
        l,t,r,b = torch.split(reg, [1,1,1,1], dim=1)
        x1 = gx - l
        y1 = gy - t
        x2 = gx + r
        y2 = gy + b
        scores = cls # [B,nc,N]
        xyxy = torch.cat([x1,y1,x2,y2], dim=1) # [B,4,N]
        yxyx = torch.cat([y1,x1,y2,x2], dim=1) # [B,4,N]
        out = (xyxy.permute(0,2,1), yxyx.permute(0,2,1), scores) if self.split else (torch.cat([xyxy,scores], dim=1))

        return out