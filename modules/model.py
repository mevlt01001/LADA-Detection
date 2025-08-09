from modules import UltrlyticsModel, Postprocess, Body, HybridBody, Head, DFL, YOLO, RTDETR
import torch
import os

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
    def __init__(self, 
                 models: list[UltrlyticsModel], 
                 nc:int, 
                 imgsz:int, 
                 regmax:int=None, 
                 device=torch.device("cpu"),
                 last_epoch:int=0,
                 last_batch:int=0,
                 optim_state_dict:dict=None,
                 sched_state_dict:dict=None
                ):
        super().__init__()
        self.imgsz = max(64, 32*(int(imgsz)//32))
        self.nc = nc
        self.regmax = regmax
        self.device = device
        self.last_epoch = last_epoch
        self.last_batch = last_batch
        self.optim_state_dict = optim_state_dict
        self.sched_state_dict = sched_state_dict
        self.backbone = self._load_hybrid_backbone(models) # HybridBackbone, Body
        self.head = Head(nc=nc, regmax=self.regmax, in_ch=self.backbone.out_ch, device=device)
        self.dfl = DFL(regmax=self.regmax, nc=nc, imgsz=self.imgsz, device=device, grid_sizes=self.backbone.grid_sizes)        
        self.postprocess = Postprocess()

    @classmethod
    def from_ckpt(cls, checkpoint_path:os.PathLike, device=torch.device("cpu")):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

        inst = cls(
            models=[YOLO(model) if 'yolo' in model else RTDETR(model) for model in ckpt["models"]],
            nc=ckpt["nc"],
            imgsz=ckpt["imgsz"],
            regmax=ckpt["regmax"],
            last_epoch=ckpt["last_epoch"],
            last_batch=ckpt["last_batch"],
            optim_state_dict=ckpt["optim_state_dict"],
            sched_state_dict=ckpt["sched_state_dict"],
            device=device
        )

        inst.load_state_dict(ckpt["model_state_dict"])

        return inst

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
        x = self.backbone.forward(x)
        fpn = self.head.forward(x)
        x = self.dfl.forward(fpn)
        x = self.postprocess.forward(x)
        return x if not self.training else (x, fpn)
    

   