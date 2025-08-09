"""
It is a collection of modules used in Hybrid LADA detection model
"""

import torch
import numpy as np
from ultralytics.engine.model import Model as UltrlyticsModel
from ultralytics.models.yolo import YOLO
from ultralytics.models.rtdetr import RTDETR
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules import Detect, RTDETRDecoder, Concat
   
from .preprocess import Preprocess
from .postprocess import Postprocess
from .body import Body, HybridBody
from .detection_head import Head
from .dfl_layer import DFL
from .model import Model
from .trainer import Trainer