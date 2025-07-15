from modules import *

class Model(torch.nn.Module):
    """
    This class is a combination of `HybridBody` and `Head` classes.\\
    It provides **DFL-Based** and **anchor-free** regression and classification layers which is like YOLOv11

    Args:
        models (list[UltrlyticsModel]): list of ultralytics models
        nc (int): number of classes
        regmax (int, None): maximum number of bbox distrubution bins. if None, set to\
            `imgsz//int(np.median(self.body.strides))//2`
        imgsz (int): input image size
        device (torch.device): device

    Methods:
        forward(self, x): accepts input image tensor shaped like (H, W, C) and\
            returns DFL-Based regression and classification outputs for feature maps.
    """
    def __init__(self, models: list[UltrlyticsModel], nc=80, regmax=None, imgsz=640, device=torch.device("cpu")):
        super().__init__()
        self.imgsz = max(64, 32*(imgsz//32))
        self.nc = nc
        self.device = device
        self.backbones = self._load_backbones(models)
        # TODO: Hybrid Backbones creation with self.backbones

        self.preprocess = Preprocess(imgsz=self.imgsz).to(device)
        self.body = HybridBody(models=models, device=device, imgsz=self.imgsz)
        self.regmax = self.imgsz//int(np.median(self.body.strides))//2 if regmax is None else regmax
        self.head = Head(nc=nc, regmax=self.regmax, in_ch=self.body.out_ch, device=device)
    
    def _load_backbones(self, models: list[UltrlyticsModel]):
        available_tasks = ["detect"]
        
        # Load backbones
        backbones:list[Body] = []
        for model in models:
            # Control that, given model is ultralytics.engine.model.Model and model.task is in available_tasks
            assert isinstance(model, UltrlyticsModel), f"Given model is not ultralytics.engine.model.Model object"
            assert model.task in available_tasks, f"Model {model.task} is not supported. Available tasks: {available_tasks}"
            backbone = Body(model, device=self.device), backbones.append(backbone)

        # Control that, All of Backbones must have same task.
        assert all(backbone.task for backbone in backbones) == backbones[0].task, "Models must have the same task"
        
        return backbones


        

    def forward(self, x):
        x = self.preprocess(x)
        x = self.body(x)
        x = self.head(x)
        return x
   