from modules import UltrlyticsModel, Preprocess, Postprocess, Body, HybridBody, Head, DFL, torch

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
    def __init__(self, models: list[UltrlyticsModel], nc:int, imgsz:int, regmax:int=None, device=torch.device("cpu"),
                preprocess:bool=False,
                postprocess:bool=False,
                dfl:bool=True
                ):
        super().__init__()
        self.imgsz = max(64, 32*(imgsz//32))
        self.nc = nc
        self.regmax = regmax
        self.device = device
        self.preprocess = Preprocess(imgsz=self.imgsz) if preprocess else None
        self.postprocess = Postprocess() if postprocess else None
        self.backbone = self._load_hybrid_backbone(models) # HybridBackbone, Body
        self.head = Head(nc=nc, regmax=self.regmax, in_ch=self.backbone.out_ch, device=device)
        self.dfl = DFL(regmax=self.regmax, nc=nc, imgsz=self.imgsz, device=device, grid_sizes=self.backbone.grid_sizes) if dfl else None
    
    def _load_hybrid_backbone(self, models: list[UltrlyticsModel]):
        available_tasks = ["detect"]

        # Load backbones
        backbones:list[Body] = []
        for model in models:
            # Control that, given model is ultralytics.engine.model.Model and model.task is in available_tasks
            assert isinstance(model, UltrlyticsModel), f"Given model is not ultralytics.engine.model.Model object"
            assert model.task in available_tasks, f"Model {model.task} is not supported. Available tasks: {available_tasks}"
            # Create a Body instance as 'backbone' to extract feature maps up to the detect layers of the model
            backbone = Body(model, device=self.device)
            backbones.append(backbone)

        # Control that, All of Backbones must have same task.
        assert all([backbone.task==backbones[0].task for backbone in backbones]), "Models must have the same task"
        # Control that, All of Backbones must have same size of detect_feats_from
        assert all(len(backbone.detect_feats_from)==len(backbones[0].detect_feats_from) for backbone in backbones), "All Backbones must have same size of detect_feats_from"
        # Create HybridBackbone to concatenate all backbones feature maps
        hybrid_backbone = HybridBody(models=backbones, imgsz=self.imgsz, regmax=self.regmax, device=self.device)
        self.regmax = hybrid_backbone.regmax
        return hybrid_backbone
        
    def forward(self, x):
        x = self.preprocess.forward(x) if self.preprocess is not None else x
        x = self.backbone.forward(x)
        x = self.head.forward(x)
        x = self.dfl.forward(x) if self.dfl is not None else x
        x = self.postprocess.forward(x) if self.postprocess is not None else x
        return x
    
    def train_forward(self, x, preprocess:bool=False, dfl:bool=False):
        x = self.preprocess.forward(x) if preprocess else x
        x = self.backbone.forward(x)
        x = self.head.forward(x)
        x = self.dfl.forward(x) if dfl else x
        return x

   