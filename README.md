# The Lightweight Anchor Dynamic Assignment (LADA) Algorithm for Object Detection
*-Just implementation for [Ultraliytics]() Object Detection Models. **Here is [LADA]() paper.**-*

This repository provides a training pipeline for [Ultralytics]() object detection models that replaces the standard label/anchor assignment with **LADA**.

## Anchor Assignment

“Anchors” (more generally: candidate boxes or points) are a way to **propose multiple potential object locations per image region.** The idea traces back to **Region Proposal Networks** (RPN) in [Faster R-CNN](), and was later adapted to one-stage detectors like [SSD](). Modern YOLO variants historically used anchor-based heads, while newer versions tend to be **anchor-free** (predicting distances from a grid point, not box).

**LADA** is a lightweight dynamic assignment strategy designed to pick which candidates (anchor boxes or anchor points) should be labeled positive for each ground-truth object during training. Instead of relying on a heavy matching routine or fixed IoU thresholds per level, LADA uses simple heuristics (CLA) to select a small, high-quality set of positives, which:
- educes noisy assignments,
- keeps computations low,
- improves training stability on crowded scenes.

In this repo, “anchors” can mean either classic anchor boxes or the grid points used by anchor-free heads. 

*LADA operates at the assignment level, not tied to a specific head design.*

## YOLO Models Architecture

Ultralytics-style YOLO models are typically structured as **Backbone → Head → Detector**. (The classical “Neck”/FPN-PAN is implemented inside the “Head” block in Ultralytics configs.)

### Backbone

**Goal**: extract multi-scale semantic features from the image by convolutional blocks.

Outputs: a feature pyramid {..,P3, P4, P5, ..} at strides {..8, 16, 32..} (exact set depends on the model and image size). The backbone takes and returns a list of tensors with shapes:

input: 
- image/s: [B, 3, H, W]

output:
- P3: [B, C3, H/8, W/8 ]
- P4: [B, C4, H/16, W/16]
- P5: [B, C5, H/32, W/32]

These multi-scale maps (output) carry both low-level localization cues (high resolution) and high-level semantics (low resolution).
























## LADA Algortihm

The Lightweight Anchor Dynamic Assignment



### Features:
- Auto Regression bins count (regmax).
- LADA c1, c2, c3, final anchors point visualization.
- Hybrid model concanitation.
    - Out channels interpolation.
- Preprocess layer
- Postprocess layer
- ONNX NMS layer integration.
- Resume Training.
### Files and what are they doing?
- **modules/body.py**:
    - Includes two classes: **Body, HybridBody**
        - **Body Class**:
            - Seperates ultralytic models feature/detection layers (Detect, RtdetrDecoder), Return it's outputs.
        - **HybridBody Class**: *Associated with: **Body***
            - If there is more than one models, This class concanate feature layers along Channel dimensions. If layer does not have equal resolution interpolate maximum resolution.
            - Calculates optimum regression distance (LTRB) probality binaries count (regmax).
- **modules/detection_head.py**
    - Includes one class: **Head**
        - **Head**:
            - Creates regression (c:4*regmax) and classification(c: number of class) blocks for each feature layers with same resolution.
- **modules/dfl_layer.py**:
    - includes one class: **DFL**:
        - **DFL**: 
            - This is non-trainable class. Every feature layers returns Tensor shaped like [B,4(LTRB)*regmax+nc,H,W], this class convert LTRB to XYXY for each feature layers.
            - Optionally use `onnx_nms_out=true` to use faster [ONNX NMS](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html).
- **modules/postprocess.py**:
    - Includes one class: **Postprocess**:
        - **Postprocess**:
            - This is one of non-trinable class. Apply confidence score threshold and Non-Maximum Suppression for anchor point regressions. Returns Bounding box coordinates, selected class and it's score.
- **modules/preprocess.py**:
    - Includes one class: **Preprocess**:
        - **Preprocess**:
            - This is non-Trainable class. Model trained with 640x640 resolutions does not work well with different resolutions. This class resize data to model resolution with GPU acceralated.
- **modules/model.py**: 
    - Includes one class: **Model**
        - **Model**: *Associated with: **Preprocess**, **HybridBody**, **Head**, **DFL**, **Postprocess***
            - Main LADA Model. Data flow fallowing class: **Preprocess**, **HybridBody**, **Head**, **DFL**, **Postprocess**
- **modules/trainer.py**:
    - Includes one class: **LADATrainer**
        - **LADATrainer**: *Associated with: **Model***
            - Trains **Model** with LADA [Lightweight Anchor Dynamic Assignmet](https://doi.org/10.3390/s23146306) Algorithm. 
    


