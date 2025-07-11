from modules import *

class Model(torch.nn.Module):
    """
    This class is a combination of `HybridBody` and `Head` classes.\\
    It provides **DFL-Based** and **anchor-free** regression and classification layers which is like YOLOv11

    Args:
        models (list[UltrlyticsModel]): list of ultralytics models
        nc (int): number of classes
        regmax (int, None): maximum number of bbox distrubution bins. if None, set to `imgsz//int(np.median(self.body.strides))//2`
        imgsz (int): input image size
        device (torch.device): device

    Methods:
        forward(self, x): accepts input image tensor shaped like (H, W, C) and returns DFL-Based regression and classification outputs for feature maps.
    """
    def __init__(self, models: list[UltrlyticsModel], nc=80, regmax=None, imgsz=640, device=torch.device("cpu")):
        super().__init__()
        self.imgsz = max(64, 32*(imgsz//32))
        self.nc = nc
        self.preprocess = Preprocess(imgsz=self.imgsz).to(device)
        self.body = HybridBody(models=models, device=device, imgsz=self.imgsz)
        self.regmax = self.imgsz//int(np.median(self.body.strides))//2 if regmax is None else regmax
        self.head = Head(nc=nc, regmax=self.regmax, in_ch=self.body.out_ch, device=device)
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.body(x)
        x = self.head(x)
        return x
   