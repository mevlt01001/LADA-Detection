from math import e
import onnx
import torch
import onnxsim
from sor4onnx import rename
from snc4onnx import combine
from onnx.utils import extract_model

rtdetr_path = "onnx_folder/rtdetr-l.onnx"
rtdetr_new_path = "onnx_folder/rtdetr-cutted.onnx"
yolo_path = "onnx_folder/yolo12l.onnx"
yolo_new_path = "onnx_folder/yolo-cutted.onnx"

new_input_rtdetr = ["images"]
new_output_rtdetr = ["/model.21/Add_output_0", "/model.24/Add_output_0", "/model.27/Add_output_0"]# [1,256,80,80], [1,256,40,40], [1,256,20,20], 
new_input_yolo = ["images"]
new_output_yolo = ["/model.14/Concat_output_0", "/model.17/Concat_output_0", "/model.20/Concat_output_0"]# [1,384,80,80], [1,768,40,40], [1,1024,20,20]

#extract model
rtdetr_cutted = extract_model(rtdetr_path, rtdetr_new_path, new_input_rtdetr, new_output_rtdetr)
yolo_cutted = extract_model(yolo_path, yolo_new_path, new_input_yolo, new_output_yolo)

rtdetr_cutted = onnx.load(rtdetr_new_path)
rtdetr_cutted, check = onnxsim.simplify(rtdetr_cutted)
rtdetr_cutted = rename(old_new=['images', 'rtdetr_in'], onnx_graph=rtdetr_cutted)
onnx.save(rtdetr_cutted, rtdetr_new_path)

yolo_cutted = onnx.load(yolo_new_path)
yolo_cutted, check = onnxsim.simplify(yolo_cutted)
yolo_cutted = rename(old_new=['images', 'yolo_in'], onnx_graph=yolo_cutted)
onnx.save(yolo_cutted, yolo_new_path)

class ensemble_post(torch.nn.Module):
    def __init__(self):
        super(ensemble_post, self).__init__()
        
    def forward(self, r1, r2, r3, y1, y2, y3):
        x1 = torch.cat([r1, y1], dim=1)
        x2 = torch.cat([r2, y2], dim=1)
        x3 = torch.cat([r3, y3], dim=1)
        return x1, x2, x3

class ensemble_pre(torch.nn.Module):
    def __init__(self):
        super(ensemble_pre, self).__init__()
        
    def forward(self, x):
        return x,x

torch.onnx.export(
    model=ensemble_post(),
    args=(torch.randn([1,256,80,80], dtype=torch.float32),
          torch.randn([1,256,40,40], dtype=torch.float32),
          torch.randn([1,256,20,20], dtype=torch.float32),
          torch.randn([1,384,80,80], dtype=torch.float32),
          torch.randn([1,768,40,40], dtype=torch.float32),
          torch.randn([1,1024,20,20], dtype=torch.float32)),
    f="onnx_folder/ensemble_post.onnx",
    opset_version=19,
    input_names=["r1", "r2", "r3", "y1", "y2", "y3"],
    output_names=["f1", "f2", "f3"],
)

torch.onnx.export(
    model=ensemble_pre(),
    args=(torch.randn([1,3,640,640], dtype=torch.float32),),
    f="onnx_folder/ensemble_pre.onnx",
    opset_version=19,
    input_names=["in"],
    output_names=["in1", "in2"],
)

ensemble_post_onnx = onnx.load("onnx_folder/ensemble_post.onnx")
ensemble_post_onnx, check = onnxsim.simplify(ensemble_post_onnx)
onnx.save(ensemble_post_onnx, "onnx_folder/ensemble_post.onnx")

ensemble_pre_onnx = onnx.load("onnx_folder/ensemble_pre.onnx")
ensemble_pre_onnx, check = onnxsim.simplify(ensemble_pre_onnx)
onnx.save(ensemble_pre_onnx, "onnx_folder/ensemble_pre.onnx")

r_combined = combine(
    onnx_graphs=[
        rtdetr_cutted,
        ensemble_post_onnx,
    ],
    op_prefixes_after_merging=[
        "rt", "post"
    ],
    srcop_destop=[
        [rtdetr_cutted.graph.output[0].name, ensemble_post_onnx.graph.input[0].name,
        rtdetr_cutted.graph.output[1].name, ensemble_post_onnx.graph.input[1].name,
        rtdetr_cutted.graph.output[2].name, ensemble_post_onnx.graph.input[2].name]
    ]
)

ensemble_model = combine(
    onnx_graphs=[
        yolo_cutted,
        r_combined,
    ],
    op_prefixes_after_merging=[
        "init", "post"
    ],
    srcop_destop=[
        [yolo_cutted.graph.output[0].name, r_combined.graph.input[1].name,
        yolo_cutted.graph.output[1].name, r_combined.graph.input[2].name,
        yolo_cutted.graph.output[2].name, r_combined.graph.input[3].name]
    ]
)

ensemble_model = combine(
    onnx_graphs=[
        ensemble_pre_onnx,
        ensemble_model
    ],
    srcop_destop=[
        [ensemble_pre_onnx.graph.output[0].name, ensemble_model.graph.input[0].name,
        ensemble_pre_onnx.graph.output[1].name, ensemble_model.graph.input[1].name]
    ]
)
ensemble_model, check = onnxsim.simplify(ensemble_model)

ensemble_post_onnx = combine(
    onnx_graphs=[
        ensemble_post_onnx,
        ensemble_model
    ],
    srcop_destop=[
        [ensemble_post_onnx.graph.output[0].name, ensemble_model.graph.output[0].name,
        ensemble_post_onnx.graph.output[1].name, ensemble_model.graph.output[1].name,
        ensemble_post_onnx.graph.output[2].name, ensemble_model.graph.output[2].name]
    ],
    op_prefixes_after_merging=[
        "post", "ensemble"
    ],
    output_onnx_file_path="onnx_folder/ensemble_model.onnx"
)
ensemble_model, check = onnxsim.simplify(ensemble_model)
onnx.save(ensemble_model, "onnx_folder/ensemble_model.onnx")    