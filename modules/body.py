from . import UltrlyticsModel, Detect, RTDETRDecoder, Concat
import numpy as np
import torch

class Body(torch.nn.Module):
    """
    This class extracts features pyramid (FPN) (p3, p4, p5,...) output from given model as a backbone (ultralytics.nn.tasks.DetectionModel).
    """
    def __init__(self, model: UltrlyticsModel, device=torch.device("cpu")):
        super(Body, self).__init__()
        self.model_name = model.model_name
        self.task = model.task
        self.detect_feats_from = model.model.model[-1].f # list of feature map layer indices [..., p3, p4, p5, ...]
        self.layers = torch.nn.ModuleList(model.to(device).model.model[:-1])
        # print(self.info())
    
    def forward(self, x):
        outputs = []
        for m in self.layers:
            if isinstance(m, (Detect, RTDETRDecoder)):
                return [outputs[f] for f in self.detect_feats_from]
            elif isinstance(m, Concat):
                x = m([outputs[f] for f in m.f])
            else:
                x = m(x) if m.f == -1 else m(outputs[m.f])
            outputs.append(x)
        return [outputs[f] for f in self.detect_feats_from]
        
    
    def info(self):
        _info = f"""
        Model name: {self.model_name}
        Task: {self.task}
        Detect feats from: {self.detect_feats_from}
        Num layers: {len(self.layers)}
        """
        return _info
    
class HybridBody(torch.nn.Module):
    """
    This class concatenates features pyramid (FPN) (p3, p4, p5) output from given models as a backbone (ultralytics.nn.tasks.DetectionModel).
    
    Args:
        models (list[ultralytics.engine.model.Model]): list of ultralytics.engine.model.Model with 'detect' task
        imgsz (int): model input image size. Models accpets min(64, 32k) {k ∈ Natural numbers}-{0}
        device (torch.device): torch device
    """
    def __init__(self, models: list[Body], imgsz:int, regmax:int=None, device=torch.device("cpu")):
        super(HybridBody, self).__init__()
        self.model_names = [model.model_name for model in models]
        self.models = torch.nn.ModuleList(models)
        self.num_feats = len(self.models[0].detect_feats_from) # All models have same size of detect_feats_from
        self.imgsz = imgsz

        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
            out = self.forward(dummy)

        self.out_ch = [p.shape[1] for p in out]
        self.strides = [imgsz // f.shape[-1] for f in out]
        self.grid_sizes = [f.shape[-1] for f in out]

        self.regmax = int(min(self.grid_sizes)/2)+4 if regmax is None else regmax
        print(self)

    def __str__(self):
        return f"""
        HybridBody:
        models: {self.model_names}
        num_feats: {self.num_feats}
        imgsz: {self.imgsz}
        out_ch: {self.out_ch}
        strides: {self.strides}
        grid_sizes: {self.grid_sizes}
        regmax: {self.regmax}
        """

    def forward(self, x):
        """
        Concatenates feature maps output from given models and returns list of feature maps

        Args:
            x (torch.Tensor): input image tensor shaped like (B, C, H, W) or (1,3,640,640)

        Returns:
            list[torch.Tensor]: list of feature maps
        """
        models_outputs = [] # [[[1,ch1,80k,80k],[1,ch2,40k,40k],...,[1,chn,20k,20k]], ...]
        for model in self.models:
            out = model.forward(x) # [[1,ch1,80k,80k],[1,ch2,40k,40k],...,[1,chn,20k,20k]]
            models_outputs.append(out)
        
        outputs = []
        for feat_idx in range(self.num_feats): #
            hybrid_out = [] # [1,ch1,80k,80k], [1,ch2,80k,80k], ...
            for model_feats in models_outputs: # feature maps [p3, p4, p5, ...] from different models
                hybrid_out.append(model_feats[feat_idx]) # Every models_outputs' same detect feats
            if any(f.shape[-1] != hybrid_out[0].shape[-1] for f in hybrid_out): # if feature maps have not the same resolution
                max_res = max(f.shape[-1] for f in hybrid_out)
                for idx, f in enumerate(hybrid_out):
                    if f.shape[-1] < max_res:
                        f = torch.nn.functional.interpolate(f, size=(max_res, max_res), mode="bilinear")
                        hybrid_out[idx] = f
            # [1,ch1+ch2+...+chn,80k,80k]
            hybrid_out = torch.cat(hybrid_out, dim=1)
            outputs.append(hybrid_out)
        return outputs # [[1,ch1+ch2+...+chn,80,80], [1,ch1+ch2+...+chn,40,40], ...]
 