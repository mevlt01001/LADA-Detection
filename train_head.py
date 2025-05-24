import torch
import torch.nn.functional as F
from create_detector_head import Head

def get_cell_center(row, col, grid_size):
    """
    returns the center of the cell given by row and col
    """
    stride = 640 / grid_size
    return torch.tensor([col*stride + stride/2, row*stride + stride/2], dtype=torch.float32)
    
def find_cell(box_xyxy, grid_size, img_size=640):
    # box_xyxy: Tensor[4] (x1,y1,x2,y2) piksel cinsinden
    cx = (box_xyxy[0] + box_xyxy[2]) * 0.5
    cy = (box_xyxy[1] + box_xyxy[3]) * 0.5
    stride = img_size / grid_size
    col  = int(cx / stride)
    row  = int(cy / stride)
    return col, row

def cxcywh2xyxy(bbox: torch.tensor):
    # bbox.shape is (n, 4)
    cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    return xyxy

def encode_dfl_label(value, reg_max, eps=1e-6):
    """
    value  : skaler tensor veya float, 0 ≤ v < reg_max
    reg_max: bin sayısı
    """
    v = torch.clamp(value, 0, reg_max - 1 - eps)    # güvenli aralık
    left  = torch.floor(v).long()
    right = left + 1

    label = torch.zeros(reg_max, dtype=v.dtype, device=v.device)

    if right >= reg_max:               # v ≈ reg_max-1
        label[left] = 1.0
    elif left == right:                # tam sayı
        label[left] = 1.0
    else:                              # ara değer
        label[left]  = right - v
        label[right] = v - left
    return label

def distribution(l, t, r, b, reg_max, nc, cls=0):
    """
    l, t, r, b: float, her biri bir mesafe (label olarak, ör: 3.2)
    reg_max: kaç aralık olacak (örn: 12, 16)
    nc: class sayısı
    cls: ground truth class indexi
    """
    # DFL encoding
    l_label = encode_dfl_label(l, reg_max)
    t_label = encode_dfl_label(t, reg_max)
    r_label = encode_dfl_label(r, reg_max)
    b_label = encode_dfl_label(b, reg_max)
    # Concat: ltrb
    ltrb_dist = torch.cat([l_label, t_label, r_label, b_label], dim=0)  # shape [4*reg_max]
    # Class label: one-hot (ör: cls=0 için [1,0,0,...])
    class_vec = torch.zeros(nc)
    class_vec[cls] = 1.0
    # Tümünü birleştir: shape [4*reg_max + nc]
    full_label = torch.cat([ltrb_dist, class_vec], dim=0)
    return full_label  # shape [4*reg_max+nc]
    
def bbox2dist(grid_size, boxes_xyxy, reg_max=12, nc=1, img_size=640):
    label = torch.zeros((1, 4*reg_max + nc, grid_size, grid_size))
    stride = img_size / grid_size
    for box in boxes_xyxy:                     # tek tek kutular
        col, row = find_cell(box, grid_size, img_size)
        cx_cell, cy_cell = get_cell_center(row, col, grid_size)
        # piksel mesafeler
        l = (cx_cell - box[0]) / stride
        t = (cy_cell - box[1]) / stride
        r = (box[2] - cx_cell) / stride
        b = (box[3] - cy_cell) / stride
        label[0, :, row, col] = distribution(l, t, r, b, reg_max, nc, cls=0)
    return label

def get_all_cell_centers(grid_size, img_size=640, device='cpu'):
    stride = img_size / grid_size
    ys, xs = torch.meshgrid(torch.arange(grid_size, device=device),
                            torch.arange(grid_size, device=device),
                            indexing='ij')
    centers = torch.stack((xs, ys), dim=-1).float() * stride + stride * 0.5
    return centers  # [H,W,2] piksel

def dist2bbox(dist_preds, cell_centers, reg_max=16, img_size=640, *, from_logits=True):
    """
    dist_preds : [B, 4*reg_max + nc, H, W]
                 (logits ise from_logits=True, prob ise False)
    cell_centers: [H, W, 2]  (cx, cy) piksel
    """
    B, C, H, W = dist_preds.shape
    stride = img_size / H

    # 1) Ayrıştır
    class_preds = dist_preds[:, 4*reg_max:, ...]          # [B,nc,H,W]
    dist_preds  = dist_preds[:, :4*reg_max, ...]          # [B,4*reg_max,H,W]

    # 2) LTRB dağılımını olasılığa çevir
    preds = dist_preds.view(B, 4, reg_max, H, W)
    if from_logits:
        preds = preds.softmax(2)                          # [B,4,R,H,W]

    # 3) Beklenen değer
    project = torch.arange(reg_max, device=preds.device, dtype=preds.dtype)
    dist = (preds * project[None, None, :, None, None]).sum(2) * stride  # [B,4,H,W]

    # 4) Piksele projeksiyon
    cx = cell_centers[..., 0].to(dist.dtype).unsqueeze(0)  # [1,H,W]
    cy = cell_centers[..., 1].to(dist.dtype).unsqueeze(0)
    l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]

    x1, y1 = cx - l, cy - t
    x2, y2 = cx + r, cy + b
    boxes = torch.stack((x1, y1, x2, y2), 1)              # [B,4,H,W]

    return torch.cat((boxes, class_preds), 1)              # [B,4+nc,H,W]




reg_max, nc, gs = 12, 1, 160
cxcywh_norm = torch.tensor([[0.2, 0.92, 0.1, 0.1]]) # 320,320,64,64
cxcywh_pix  = cxcywh_norm * 640
xyxy_pix    = cxcywh2xyxy(cxcywh_pix) # 

dist = bbox2dist(gs, xyxy_pix, reg_max, nc)          # [1,49,gs,gs]
centers = get_all_cell_centers(gs)                   # [gs,gs,2]
out = dist2bbox(dist, centers, reg_max, from_logits=False)              # [1,4+nc,gs,gs]

# En yüksek class skorunun olduğu hücre
scores = out[:,4,:,:][0]                                # [1,gs,gs]
row, col = torch.nonzero(scores==scores.max(), as_tuple=True)
print('recovered xyxy :', out[0,:4,row,col])         # ≈ orijinal kutu (piksel)
xyxy = out[0,:4,row,col]
cx = (xyxy[0] + xyxy[2]) * 0.5
cy = (xyxy[1] + xyxy[3]) * 0.5
w = xyxy[2] - xyxy[0]
h = xyxy[3] - xyxy[1]
print('recovered cxcywh :', torch.tensor([cx, cy, w, h])/640) # output: recovered cxcywh : tensor([0.5170, 0.5170, 0.3971, 0.3971])



