from modules.model import Model
from modules.trainer import LADATrainer
from modules import YOLO, RTDETR, UltrlyticsModel
import os, cv2
import torch
import yaml as YAML
import onnx, onnxsim

class LADAModel:
    def __init__(self,
                 models: list[UltrlyticsModel]|str,
                 imgsz:int=640, 
                 regmax:int=None, 
                 device=torch.device("cpu"),
                 last_epoch:int=0,
                 last_batch:int=0,
                 optim_state_dict:dict=None,
                 sched_state_dict:dict=None):
        
        self.models=models
        self.imgsz=imgsz
        self.regmax=regmax
        self.device=device
        self.last_epoch=last_epoch
        self.last_batch=last_batch
        self.optim_state_dict=optim_state_dict
        self.sched_state_dict=sched_state_dict

    def train(self, 
              epoch:int, 
              batch:int, 
              data_yaml: str=None,
              train_path: str=None, 
              valid_path: str=None,
              debug:bool=False,
              c2k = 9, # Best 9 anchors for each stride
              c3k = 20, # Best 20 anchors for all strides,
              lr = 0.001,
            ):
        
        if data_yaml is not None:
            with open(data_yaml, "r", encoding="utf-8") as f:
                yaml = YAML.safe_load(f)

        train_path = os.path.dirname(os.path.abspath(os.path.join(data_yaml,yaml["train"])))
        valid_path = os.path.dirname(os.path.abspath(os.path.join(data_yaml,yaml["val"]))) if "val" in yaml else None
        valid_path = valid_path if os.path.exists(valid_path) else None        
        
        self.cls_names = yaml["names"] if "names" in yaml else None
        self.nc = yaml["nc"] if "nc" in yaml else len(self.cls_names) if self.cls_names is not None else None

        if self.cls_names is None or self.nc is None:
            raise ValueError("Specified data_yaml does not contain nc or names.")
        
        elif train_path is None:
            raise ValueError("You must specify either data_yaml or train_path.")
        
        self.__load_model()

        trainer = LADATrainer(model=self.model)
        trainer.train(
            epoch=epoch,
            batch=batch,
            train_path=train_path,
            valid_path=valid_path,
            debug=debug,
            c2k=c2k,
            c3k=c3k,
            lr=lr
        )
        self = trainer.model

    def export(self, 
               format:str, 
               path:str):
        
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

        self.__load_model()
        
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
                    "cls_names": self.cls_names,
                    "imgsz": self.imgsz,
                    "regmax": self.regmax,
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": self.model.optim_state_dict(),
                    "sched_state_dict": self.model.sched_state_dict(),
                    "last_epoch": self.last_epoch,
                    "last_batch": self.last_batch,
                    }, path)

    def __load_model(self):
        if type(self.models) == str:
            self.model = Model.from_ckpt(
                checkpoint_path=self.models,
                device=self.device
            )
        else:
            self.model = Model(
                models=self.models,
                imgsz=self.imgsz,
                regmax=self.regmax,
                device=self.device,
                last_epoch=self.last_epoch,
                last_batch=self.last_batch,
                optim_state_dict=self.optim_state_dict,
                sched_state_dict=self.sched_state_dict,
                nc=self.nc,
                names=self.cls_names
            )

    def predict(self, 
                source:str|torch.Tensor, 
                save_path:str="inference"):
        self.__load_model()
        self.model = self.model.train(False)
        if type(source) == str:
            if os.path.exists(source):
                img = cv2.imread(source)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.model.imgsz, self.model.imgsz))
                data = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device=self.model.device)/255.0

        out = self.model.forward(data)[0]        
        xyxy = out[:, :4]
        cls_score = out[:, 4]
        cls_id = out[:, 5].long()
        
        save_path = "inference" if save_path is None else save_path
        os.makedirs(save_path, exist_ok=True)
        cnt = os.listdir(save_path)
        save_path = f"{save_path}/inf_{len(cnt)}"

        for box, score, idx in zip(xyxy, cls_score, cls_id):
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.cls_names[idx.item()]}: {score.item():.2f}"
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            img = cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(f"{save_path}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))