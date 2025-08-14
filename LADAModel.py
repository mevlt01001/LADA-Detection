from modules.model import Model
from modules.trainer import LADATrainer
from modules import YOLO, RTDETR, UltrlyticsModel
import os
import torch
import yaml as YAML
import onnx, onnxsim

class LADAModel:
    def __init__(self,
                 models: list[UltrlyticsModel]|str,
                 nc:int=None, 
                 imgsz:int=None, 
                 regmax:int=None, 
                 device=torch.device("cpu"),
                 last_epoch:int=0,
                 last_batch:int=0,
                 optim_state_dict:dict=None,
                 sched_state_dict:dict=None):
        
        _type = type(models)
        
        if _type == str:
            if os.path.splitext(models)[-1] == ".pt":
                self.model = Model.from_ckpt(models, device)
        else:    
            self.model = Model(
                models=models,
                nc=nc,
                imgsz=imgsz,
                regmax=regmax,
                device=device,
                last_epoch=last_epoch,
                last_batch=last_batch,
                optim_state_dict=optim_state_dict,
                sched_state_dict=sched_state_dict
            )
    
    def load(self, path:os.PathLike, device=torch.device("cpu")):
        model = Model.from_ckpt(path, device)
        self.model = model

    def train(self, 
              epoch:int, 
              batch:int, 
              data_yaml: str=None,
              train_path: str=None, 
              valid_path: str=None,
              debug:bool=False,
              c2k = 9, # Best 9 anchors for each stride
              c3k = 20 # Best 20 anchors for all strides
            ):
        if data_yaml is not None:
            with open(data_yaml, "r", encoding="utf-8") as f:
                yaml = YAML.safe_load(f)

            train_path = os.path.dirname(os.path.abspath(os.path.join(data_yaml,yaml["train"])))
            valid_path = os.path.dirname(os.path.abspath(os.path.join(data_yaml,yaml["val"]))) if "val" in yaml else None
           
            self.nc = yaml["nc"] if "nc" in yaml else yaml["names"] if "names" in yaml else None
            if self.nc is None:
                raise ValueError("Specified data_yaml does not contain nc or names.")
        elif train_path is None:
            raise ValueError("You must specify either data_yaml or train_path.")
        
        trainer = LADATrainer(model=self.model)
        trainer.train(
            epoch=epoch,
            batch=batch,
            train_path=train_path,
            valid_path=valid_path,
            debug=debug,
            c2k=c2k,
            c3k=c3k
        )
        self = trainer.model

    def export(self, 
               format:str, 
               path:str, 
               optim_state_dict=None, 
               sched_state_dict=None,
               last_epoch:int=0,
               last_batch:int=0):
        formats = ["onnx", "pt"]
        assert format in formats, f"Format {format} is not supported. Available formats: {formats}"

        path = "model" if path is None else path
        path = path if path.endswith(f".{format}") else path + f".{format}" # path/dir/file.format
        par_dir = os.path.dirname(path) # path/dir/
        os.makedirs(par_dir, exist_ok=True) if par_dir != "" else None

        files = os.listdir(par_dir) if par_dir != "" else [] # files in path/dir/

        count = len([1 for file in files if path.split("/")[-1] in file]) # count files with same name
        path = f"{os.path.splitext(path)[0]}_{count}.{format}"
        
        dummy = torch.zeros(1, 3, self.model.imgsz, self.model.imgsz, device=self.model.device)
        
        if format == "onnx":
            torch.onnx.export(
                self.model, 
                dummy, 
                path,
                input_names=["images"],
                output_names=["output"]
                )
            mdl = onnx.load(path)
            mdl, check = onnxsim.simplify(mdl, check_n=3)
            if check:
                mdl = onnx.shape_inference.infer_shapes(mdl)
            else:
                Warning("ONNX model could not be simplified")
            onnx.save(mdl, path)
        elif format == "pt":
            torch.save({
                    "models": self.model.backbone.model_names,
                    "nc": self.nc,
                    "imgsz": self.model.imgsz,
                    "regmax": self.model.regmax,
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": optim_state_dict,
                    "sched_state_dict": sched_state_dict(),
                    "last_epoch": last_epoch,
                    "last_batch": last_batch
                    }, path)
