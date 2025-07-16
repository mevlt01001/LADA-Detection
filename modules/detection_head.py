from . import *

class Head(torch.nn.Module):
    """
    This class provides **DFL-Based** and **anchor-free** regression and classification layers which is like YOLOv11

    Args:
        nc (int): number of classes
        in_ch (list[int]): input channels
        regmax (int): maximum number of prob bins
        device (torch.device): device

    Methods:
        forward(self, x): accepts feature pyramid output (..., p3, p4, p5, ...) and returns DFL-Based regression and classification outputs for feature maps.
    """
    def __init__(self, nc, in_ch, regmax, device:torch.device="cpu"):
        super().__init__()
        reg_inter_ch = lambda ch: max((ch // 4, regmax * 4))
        cls_inter_ch = max((min(in_ch)//4, nc))
        self.reg_blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                Conv(ch, reg_inter_ch(ch), 3),
                Conv(reg_inter_ch(ch), reg_inter_ch(ch), 3),
                Conv(reg_inter_ch(ch), reg_inter_ch(ch), 3),
                torch.nn.Conv2d(reg_inter_ch(ch), 4 * regmax, 1),
            )for ch in in_ch
        ).to(device)
        self.cls_blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Sequential(DWConv(ch, ch, 3), Conv(ch, cls_inter_ch, 1)),
                torch.nn.Sequential(DWConv(cls_inter_ch, cls_inter_ch, 3), Conv(cls_inter_ch, cls_inter_ch, 1)),
                torch.nn.Conv2d(cls_inter_ch, nc, 1),
            )for ch in in_ch
        ).to(device)

    def forward(self, x):
        """
        Args:
            x (list[torch.Tensor]): `HybridNet` feature pyramid output (..., p3, p4, p5, ...)

        Returns:
            outputs (list[torch.Tensor]):
            DFL-Based regression and classification outputs for each feature pyramid output.\\
            The shape of each output is (B, 4*regmax+nc, h, w)
        """
        # x = [[1,ch1+ch2+...+chn,80k,80k],[1,ch1+ch2+...+chn,40k,40k],[1,ch1+ch2+...+chn,20k,20k]]
        outputs = []
        for idx, x in enumerate(x):
            reg = self.reg_blocks[idx](x) #            [[1,4*regmax,H,W]
            cls = self.cls_blocks[idx](x) #            [[1,nc,H,W]
            outputs.append(torch.cat((reg, cls), 1)) # [[1,4*regmax+nc,H,W]
        return outputs # [[1,4*regmax+nc,80k,80k],[1,4*regmax+nc,40k,40k],[1,4*regmax+nc,20k,20k]]
