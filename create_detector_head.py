import torch, torch.nn as nn
from ultralytics.nn.modules.conv import Conv, DWConv

class Head(nn.Module):
    def __init__(self, nc=1, f_ch=[640, 1024, 1280], ch=[128, 128, 128]):
        super().__init__()
        self.f1_conv = nn.Conv2d(f_ch[0], ch[0], 1)
        self.f2_conv = nn.Conv2d(f_ch[1], ch[1], 1)
        self.f3_conv = nn.Conv2d(f_ch[2], ch[2], 1)
        self.nc = nc
        self.reg_max = 12
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if False
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
    

class Head_p3(Head):
    def __init__(self, nc=1, f_ch=[640, 1024, 1280], ch=[128, 128, 128]):
        super().__init__(nc, f_ch, ch)

    def forward(self, f1):
        f1 = self.f1_conv(f1)
        f1 = torch.nn.functional.interpolate(f1, scale_factor=1.5, mode="nearest")
        p3 = torch.cat((self.cv2[0](f1), self.cv3[0](f1)), 1)
        return p3
    
class Head_p4(Head):
    def __init__(self, nc=1, f_ch=[640, 1024, 1280], ch=[128, 128, 128]):
        super().__init__(nc, f_ch, ch)

    def forward(self, f2):
        f2 = self.f2_conv(f2)
        f2 = torch.nn.functional.interpolate(f2, scale_factor=1.5, mode="nearest")
        p4 = torch.cat((self.cv2[1](f2), self.cv3[1](f2)), 1)
        return p4

class Head_p5(Head):
    def __init__(self, nc=1, f_ch=[640, 1024, 1280], ch=[128, 128, 128]):
        super().__init__(nc, f_ch, ch)

    def forward(self, f3):
        f3 = self.f3_conv(f3)
        f3 = torch.nn.functional.interpolate(f3, scale_factor=1.5, mode="nearest")
        p5 = torch.cat((self.cv2[2](f3), self.cv3[2](f3)), 1)
        return p5