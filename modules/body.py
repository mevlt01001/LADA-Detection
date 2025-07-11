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