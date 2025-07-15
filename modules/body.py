from . import *

class Body(torch.nn.Module):
    """
    This class extracts features pyramid (FPN) (p3, p4, p5) output from given model as a backbone (ultralytics.nn.tasks.DetectionModel).
    """
    def __init__(self, model: UltrlyticsModel, device=torch.device("cpu")):
        assert isinstance(model, UltrlyticsModel), f"model should be instance of ultralytics.engine.model.Model"
        assert model.task == "detect", f"An error occurred in {model.model_name}. Only 'detect'\
            task are supported, not {model.task} task yet."
        super(Body, self).__init__()
        self.model = model
        self.f = model.model.model[-1].f # list of feature map layer indices [..., p3, p4, p5, ...]
        self.layers = model.to(device).model.model
    
    def forward(self, x):
        outputs = []
        for m in self.layers:
            if isinstance(m, (Detect, RTDETRDecoder)):
                return [outputs[f] for f in self.f]
            elif isinstance(m, Concat):
                x = m([outputs[f] for f in m.f])
            else:
                x = m(x) if m.f == -1 else m(outputs[m.f])
            outputs.append(x)
        raise ValueError(f"An error occurred in {self.model.model_name}. Detect/RTDETRDecoder layer not found.")
    
class HybridBody(torch.nn.Module):
    """
    This class concatenates features pyramid (FPN) (p3, p4, p5) output from given models as a backbone (ultralytics.nn.tasks.DetectionModel).
    
    Args:
        models (list[ultralytics.engine.model.Model]): list of ultralytics.engine.model.Model with 'detect' task
        imgsz (int): model input image size. Models accpets min(64, 32k) {k ∈ Natural numbers}-{0}
        device (torch.device): torch device
    """
    def __init__(self, models: list[UltrlyticsModel], imgsz:int, device=torch.device("cpu")):
        super(HybridBody, self).__init__()
        self.models = [Body(model, device) for model in models]
        assert all([len(model.f) == len(self.models[0].f) for model in self.models]),\
            f"Detect Layers inputs ({[model.f for model in self.models]}) should be same length."
        self.f_lenght = len(self.models[0].f)
        with torch.no_grad():
            dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
            out = self.forward(dummy)
            self.out_ch = [p.shape[1] for p in out]
            self.strides = [imgsz // f.shape[-1] for f in out]
        del dummy, out

    def forward(self, x):
        """
        Concatenates feature maps output from given models and returns list of feature maps

        Args:
            x (torch.Tensor): input image tensor shaped like (B, C, H, W) or (1,3,640,640)

        Returns:
            list[torch.Tensor]: list of feature maps
        """
        _outputs = []
        for model in self.models:
            out = model.forward(x) # [[1,ch1,80k,80k],[1,ch2,40k,40k],...,[1,chn,20k,20k]]
            _outputs.append(out)
        
        outputs = []
        for feat_idx in range(len(_outputs[0])):
            outputs.append(
                # [1,ch1+ch2+...+chn,80k,80k]
                torch.cat([out[feat_idx] for out in _outputs], dim=1)
            )
        return outputs # [[1,ch1+ch2+...+chn,80,80], [1,ch1+ch2+...+chn,40,40], ...]
 