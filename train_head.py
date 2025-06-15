import torch, torchvision
import torch.nn.functional as F
from create_detector_head import Head_p3, Head_p4, Head_p5
import cv2, os
import numpy as np
import xml.etree.ElementTree as ET
import onnxruntime

def get_cell_center(row, col, grid_size):
    """
    returns the center of the cell given by row and col
    """
    stride = 640 / grid_size
    return torch.tensor([col*stride + stride/2, row*stride + stride/2], dtype=torch.float32)
    
def find_center_cell(box_xyxy, grid_size, img_size=640):
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

def encode_dfl_label(value, reg_max, device='cpu'):
    v = torch.clamp(value, 0, reg_max - 1)
    left = int(torch.floor(v).item())
    right = left + 1
    label = torch.zeros(reg_max, device=device)
    if right >= reg_max:
        label[left] = 1.0
    else:
        label[left]  = right - v.item()
        label[right] = v.item() - left
    return label

def distribution(l, t, r, b, reg_max, nc, cls=0, device='cpu'):
    # Düzeltme 2: 'device' parametresi eklendi ve aşağıya iletildi
    l_label = encode_dfl_label(l, reg_max, device=device)
    t_label = encode_dfl_label(t, reg_max, device=device)
    r_label = encode_dfl_label(r, reg_max, device=device)
    b_label = encode_dfl_label(b, reg_max, device=device)
    
    ltrb_dist = torch.cat([l_label, t_label, r_label, b_label], dim=0)
    
    class_vec = torch.zeros(nc, device=device) # Düzeltme 2
    class_vec[cls] = 1.0
    
    full_label = torch.cat([ltrb_dist, class_vec], dim=0)
    return full_label

def is_available_for_gs(bbox, gs, regmax=12, imgsize=640):
    stride = imgsize / gs
    max_ltrb = stride * regmax
    boolean = bbox[2]-bbox[0] <= max_ltrb*2 and bbox[3]-bbox[1] <= max_ltrb*2
    return boolean

def assign_C1_anchor_points(grid_size, boxes_xyxy, reg_max=12, img_size=640):

    stride = img_size / grid_size

    grids_and_boxes = dict()

    for box_id,box in enumerate(boxes_xyxy):
        if not is_available_for_gs(box, grid_size, reg_max, img_size): continue
        start_col = int(box[0] / stride)
        start_row = int(box[1] / stride)
        end_col = int(box[2] / stride)
        end_row = int(box[3] / stride)

        for p_row in range(start_row, end_row + 1):
            for p_col in range(start_col, end_col + 1):
                if not (0 <= p_row < grid_size and 0 <= p_col < grid_size):
                    continue
                
                point = (p_row, p_col)
                if not point in grids_and_boxes: grids_and_boxes[point] = []
                grids_and_boxes[point].append(box_id)

    final_grids_and_boxes = dict()
    # çakışmaları çözme adımı
    for anchor_point, candidates in grids_and_boxes.items():
        if len(candidates) == 1: 
            final_grids_and_boxes[anchor_point] = [boxes_xyxy[candidates[0]]]
            continue
        else: # o grid için 1 den fazla aday box atanmış, çakışma var.
            # 1- hangi GT'ler için çakışma gerçekleşmiş tespti et
            # 2- Çakışma gerçekleşen point e hangi GT daha yakın bul ve o point ile o gt'yi eşleştir
            
            selected_box_id = -1
            nearest_dist = float('inf')
            p_cx, p_cy = get_cell_center(anchor_point[0], anchor_point[1], grid_size)
            for box_id in candidates:
                box = boxes_xyxy[box_id]
                box_cx, box_cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                dist = torch.sqrt((p_cx - box_cx)**2 + (p_cy - box_cy)**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    selected_box_id = box_id

            final_grids_and_boxes[anchor_point] = [boxes_xyxy[selected_box_id]]

    return final_grids_and_boxes

def dist2bbox(dist_preds, reg_max=12, img_size=640, from_logits=True):
    """
    dist_preds : [B, 4*reg_max + nc, H, W]
    cell_centers: [H, W, 2]  (cx, cy) piksel
    """

    B, C, H, W = dist_preds.shape
    stride = img_size / H

    class_preds = dist_preds[:, 4*reg_max:, ...]          # [B,nc,H,W]
    dist_preds  = dist_preds[:, :4*reg_max, ...]          # [B,4*reg_max,H,W]

    preds = dist_preds.view(B, 4, reg_max, H, W)
    if from_logits:
        preds = preds.softmax(2)                          # [B,4,R,H,W]

    project = torch.arange(reg_max, device=preds.device, dtype=preds.dtype)
    dist = (preds * project[None, None, :, None, None]).sum(2) * stride  # [B,4,H,W]

    cell_centers = get_all_cell_centers(H, img_size=img_size, device=dist.device)
    cx = cell_centers[..., 0].to(dist.dtype).unsqueeze(0)  # [1,H,W]
    cy = cell_centers[..., 1].to(dist.dtype).unsqueeze(0)
    l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]

    x1, y1 = cx - l, cy - t
    x2, y2 = cx + r, cy + b
    boxes = torch.stack((x1, y1, x2, y2), 1)              # [B,4,H,W]

    return torch.cat((boxes, class_preds), 1)              # [B,4+nc,H,W]

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1, 4) to box2(n, 4)
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # Complete IoU https://arxiv.org/abs/1911.08287v1
                v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/abs/1902.09630
    return iou  # IoU

def select_best_k_points(assignment_map, decoded_preds, k=5, epoch=0):

    box_preds = decoded_preds[0, :4, ...].permute(1, 2, 0)   # (H, W, 4)
    cls_preds = decoded_preds[0, 4:, ...].permute(1, 2, 0) # (H, W, nc)
    
    candidates_with_scores = []
    for point, gt_box_list in assignment_map.items():
        p_row, p_col = point
        
        gt_box_tensor = torch.tensor(gt_box_list[0], dtype=torch.float32, device=box_preds.device).unsqueeze(0)
        pred_box_tensor = box_preds[p_row, p_col, :].unsqueeze(0)

        iou_score = bbox_iou(gt_box_tensor, pred_box_tensor, CIoU=True, xywh=False)
        
        class_score = cls_preds[p_row, p_col, 0].sigmoid()
        
        alignment_score = iou_score * class_score
        
        candidates_with_scores.append({
            'point': point,
            'gt_box_tuple': tuple(gt_box_list[0]), 
            'score': alignment_score.item()
        })
        
    candidates_by_gt = {}
    for cand in candidates_with_scores:
        gt_key = cand['gt_box_tuple']
        if gt_key not in candidates_by_gt: candidates_by_gt[gt_key] = []
        candidates_by_gt[gt_key].append(cand)
        
    final_positives_map = {}
    for gt_key, candidate_list in candidates_by_gt.items():
        candidate_list.sort(key=lambda c: c['score'], reverse=True)

        if epoch == 0:
                top_k_candidates = candidate_list
        else:
            topk = min(k, len(candidate_list))
            top_k_candidates = candidate_list[:topk]

        for cand in top_k_candidates:
            final_positives_map[cand['point']] = [list(cand['gt_box_tuple'])]
            
    return final_positives_map

def create_targets(positive_assignments, grid_size, reg_max=12, nc=1, img_size=640, device='cpu'):

    assert img_size == 640, "img_size must be 640"
    stride = img_size / grid_size
    target = torch.zeros(1, 4 * reg_max + nc, grid_size, grid_size, dtype=torch.float32, device=device)
    
    for point, boxes in positive_assignments.items():
        box = boxes[0]
        p_cx, p_cy = get_cell_center(point[0], point[1], grid_size)
        p_cx, p_cy = p_cx.to(device), p_cy.to(device)
        l = (p_cx - box[0]) / stride
        t = (p_cy - box[1]) / stride
        r = (box[2] - p_cx) / stride
        b = (box[3] - p_cy) / stride
        
        # distribution fonksiyonuna device'ı iletiyoruz
        target[0, :, point[0], point[1]] = distribution(l, t, r, b, reg_max=reg_max, nc=nc, cls=0, device=device)
        
    return target

def decode_boxes_from_dfl(pred_dist, anchor_points_centers, stride, reg_max, from_logits=False):

    dfl_dist = pred_dist.view(-1, 4, reg_max)
    if from_logits:
        dfl_dist = F.softmax(dfl_dist, dim=2)
    project = torch.arange(reg_max, device=dfl_dist.device, dtype=torch.float32)
    distances = (dfl_dist * project).sum(2)
    l_px, t_px, r_px, b_px = (distances * stride).chunk(4, dim=1)
    cx, cy = anchor_points_centers.chunk(2, dim=1)
    return torch.cat([cx - l_px, cy - t_px, cx + r_px, cy + b_px], dim=1)

def compute_loss(preds, targets, grid_size, reg_max=12, nc=1, img_size=640):
    device = preds.device
    B, C, H, W = preds.shape
    
    # ... (pos_mask ve loss_cls hesaplamaları önceki gibi doğru) ...
    target_cls = targets[:, 4 * reg_max:, :, :]
    pos_mask = (target_cls > 0).any(dim=1)
    num_pos = pos_mask.sum()

    if num_pos == 0:
        pred_scores = preds[:, 4 * reg_max:, :, :]
        loss_cls = torchvision.ops.sigmoid_focal_loss(pred_scores, target_cls, alpha=0.75, gamma=2.0, reduction='mean')
        return loss_cls, torch.tensor([0.0, 0.0, loss_cls], device=device)

    pred_scores = preds[:, 4 * reg_max:, :, :]
    loss_cls = torchvision.ops.sigmoid_focal_loss(inputs=pred_scores, targets=target_cls, alpha=0.75, gamma=2.0, reduction='sum') / num_pos
    
    # Pozitif örnekleri seçme
    pos_mask_flat = pos_mask.view(-1)
    preds_flat = preds.permute(0, 2, 3, 1).reshape(-1, C)
    pos_preds = preds_flat[pos_mask_flat]
    targets_flat = targets.permute(0, 2, 3, 1).reshape(-1, C)
    pos_targets = targets_flat[pos_mask_flat]
    
    # --- DFL KAYBI İÇİN DOĞRU HESAPLAMA (SENİN DÜZELTMENLE) ---
    target_dist = pos_targets[:, :4 * reg_max]
    pred_dist_logits = pos_preds[:, :4 * reg_max]

    # Dağılımları 4 ayrı grup olarak yeniden şekillendir: [num_pos, 4, reg_max]
    pred_dist_logits_reshaped = pred_dist_logits.view(-1, 4, reg_max)
    target_dist_reshaped = target_dist.view(-1, 4, reg_max)
    
    # Softmax'ı her bir 4 dağılım için reg_max boyutu (dim=2) boyunca uygula
    loss_dfl = F.kl_div(
        F.log_softmax(pred_dist_logits_reshaped, dim=2), 
        target_dist_reshaped, 
        reduction='batchmean'
    )

    # --- Kutu Regresyon Kaybı (L_box) ---
    # ... (Bu kısım aynı kalır, decode_boxes_from_dfl de artık bu mantığa göre çalışır) ...
    stride = img_size / H
    all_cell_centers = get_all_cell_centers(H, img_size=img_size, device=device)
    pos_anchor_centers = all_cell_centers.view(-1, 2)[pos_mask_flat]
    
    pred_boxes_pos = decode_boxes_from_dfl(pred_dist_logits, pos_anchor_centers, stride, reg_max, from_logits=True)
    target_boxes_pos = decode_boxes_from_dfl(target_dist, pos_anchor_centers, stride, reg_max, from_logits=False)
    
    iou = bbox_iou(pred_boxes_pos, target_boxes_pos, CIoU=True, xywh=False)
    loss_box = (1.0 - iou).mean()
    
    # Toplam Kayıp
    w_cls, w_dfl, w_box = 0.5,1.5,7.5
    total_loss = (loss_cls * w_cls) + (loss_dfl * w_dfl) + (loss_box * w_box)
    
    return total_loss, torch.stack([loss_box, loss_dfl, loss_cls]).detach()

def get_all_cell_centers(grid_size, img_size=640, device='cpu'):
    stride = img_size / grid_size
    ys, xs = torch.meshgrid(torch.arange(grid_size, device=device),
                            torch.arange(grid_size, device=device),
                            indexing='ij')
    centers = torch.stack((xs, ys), dim=-1).float() * stride + stride * 0.5
    return centers  # [H,W,2] piksel

def get_names(images_path):
    files = os.listdir(images_path)
    names = []
    for file in files:
        name, ext = os.path.splitext(file)
        names.append(name)
    return np.asarray(names)

def get_images_and_xyxy(names, images_path, labels_path, img_size=640, device='cpu'):
    """
    **Description**:
        This func takes batch names (images and labels have to same names except of extentions) and returns images and xyxy bbox coordinates by create list of images and labels path.
        ***NOTE**: Preprocess step is not included for images. Because Ensemble model has its own preprocess.*
    **Parameters**:
        `names`: images and labels' names array\n
        `images_path`: path to images\n
        `labels_path`: path to labels\n
        `img_size`: 640 (must!)\n
    **Returns**:
        `images`: list of images\n
        `xyxy`: list of boxes coordinates for 640x640 image size
        
    """
    assert img_size == 640, "img_size must be 640"

    images = []
    _labels = []

    for name in names:
        img = cv2.imread(images_path + '/' + name + '.jpg').astype(np.float32)
        w, h = img.shape[1], img.shape[0]
        images.append(img)

        tree = ET.parse(labels_path + '/' + name + '.xml')
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if not (name == "person"):
                raise Exception("Not a person")
            
            bndbox = obj.find('bndbox')
            x1 = float(bndbox.find('xmin').text)/w*img_size
            y1 = float(bndbox.find('ymin').text)/h*img_size
            x2 = float(bndbox.find('xmax').text)/w*img_size
            y2 = float(bndbox.find('ymax').text)/h*img_size

            boxes.append([x1, y1, x2, y2])
        _labels.append(boxes)
    
    return images, _labels

def DFL_decode(distributed_preds, reg_max, img_size, nc=1, conf_thresh=0.25, iou_thresh=0.45):
    """
    Modelin ham çıktısını işleyerek anlamlı sınırlayıcı kutulara dönüştürür.
    Bu fonksiyon, DFL (Distribution Focal Loss) tabanlı bir nesne tespit modelinin
    ürettiği ham tensörü alır. Tensör içindeki olasılık dağılımlarını kullanarak
    sınırlayıcı kutu (bounding box) koordinatlarını (ltrb formatından xyxy formatına)
    çözer. Ardından, düşük güven skoruna sahip kutuları eler (confidence thresholding)
    ve son olarak çakışan kutuları temizlemek için Non-Maximum Suppression (NMS)
    uygular. Fonksiyon, batch boyutu 1 olan girdiler için tasarlanmıştır.

    Args:
        distributed_preds (torch.Tensor): Modelin ham çıktı tensörü.
            Beklenen şekil: `[B, 4*reg_max+nc, H, W]`.
        reg_max (int): DFL için kullanılan bin (kategori) sayısı.
        img_size (int): Modele verilen girdi görüntüsünün boyutu (örn: 640).
            Stride hesaplaması için kullanılır.
        nc (int): Veri setindeki toplam sınıf sayısı.
        conf_thresh (float): Güven skoru için alt eşik. Bu değerin
            altındaki tespitler elenir.
        iou_thresh (float): NMS için kullanılan IoU (Intersection over Union)
            eşiği. Bu eşikten daha fazla çakışan kutulardan düşük skorlu
            olanlar elenir.

    Returns:
        torch.Tensor: Tespit edilen nihai nesneleri içeren tensör. Şekli
            `[N, 6]`'dır, burada `N` tespit edilen nesne sayısıdır. Her satır
            şu formattadır: `(x1, y1, x2, y2, score, class_id)`.
                - `x1, y1, x2, y2`: Kutunun sol üst ve sağ alt koordinatları.
                - `score`: Tespitin güven skoru.
                - `class_id`: Tespit edilen sınıfın kimliği (indeksi).
    """
    preds_tensor = distributed_preds[0] # Şekil: [C, H, W] batch_size=1

    device = preds_tensor.device
    H, W = preds_tensor.shape[1:]
    stride = img_size / H
    
    pred_dist, pred_scores = torch.split(preds_tensor, [4 * reg_max, nc], dim=0)

    pred_dist = pred_dist.view(4, reg_max, H, W).softmax(1)
    project = torch.arange(reg_max, device=device, dtype=torch.float32)
    dist = (pred_dist * project.view(1, -1, 1, 1)).sum(1)

    cell_centers = get_all_cell_centers(grid_size=H, device=device)
    cx, cy = cell_centers[..., 0], cell_centers[..., 1]
    l, t, r, b = dist
    
    boxes_all = torch.stack([cx - l, cy - t, cx + r, cy + b], dim=-1)*stride

    boxes_flat = boxes_all.view(-1, 4) # Şekil: [H*W, 4]
    scores_flat = pred_scores.sigmoid().view(nc, -1).T # Şekil: [H*W, nc]

    conf_scores, class_ids = scores_flat.max(1) # En yüksek skora sahip sınıfı bul
    
    mask = conf_scores > conf_thresh
    
    boxes_filtered = boxes_flat[mask]
    scores_filtered = conf_scores[mask]
    class_ids_filtered = class_ids[mask]
    
    if boxes_filtered.shape[0] == 0:
        return torch.empty((0, 6), device=device)

    keep = torchvision.ops.nms(boxes_filtered, scores_filtered, iou_threshold=iou_thresh)
    
    final_boxes = boxes_filtered[keep]
    final_scores = scores_filtered[keep]
    final_class_ids = class_ids_filtered[keep]

    return torch.cat([
        final_boxes,
        final_scores.unsqueeze(1),
        final_class_ids.unsqueeze(1)
    ], dim=1)

def create_target_dist(gt_boxes, pred_dist, nc=1, grid_size=120, reg_max=12, img_size=640, max_k=10, epoch=0):
    with torch.no_grad():
        assignmets_map = assign_C1_anchor_points(grid_size, gt_boxes, reg_max, img_size)
        pred_boxes = dist2bbox(pred_dist, reg_max, img_size)
        positive_assignments = select_best_k_points(assignmets_map, pred_boxes, k=max_k, epoch=epoch)
        targets = create_targets(positive_assignments, grid_size=grid_size, reg_max=reg_max, nc=nc)
        return targets

os.makedirs("trained_models", exist_ok=True)
if __name__ == '__main__':
    EPOCHS = 20
    BATCH_SIZE = 4
    IMG_SIZE = 640
    LEARNING_RATE = 0.01
    GRID_SIZE = 120
    REG_MAX = 16
    NC = 1 # Sınıf sayısı
    K_BEST = 12 # Her GT için seçilecek en iyi aday sayısı
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Veri ve Modelleri Yükle
    images_path = "/home/neuron/datasets/head_detection/images"
    labels_path = "/home/neuron/datasets/head_detection/labels"
    names = get_names(images_path)

    # ONNX modelini yükle (sadece özellik çıkarımı için, eğitilmeyecek)
    ensemble_model = onnxruntime.InferenceSession("onnx_folder/ensemble_model.onnx", providers=['CUDAExecutionProvider'])

    model = Head_p3(nc=NC, f_ch=[640, 1024, 1280], ch=[128, 128, 128]).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.635)
    total_loses = []
    cls_losses = []
    dfl_losses = []
    reg_losses = []
    best_loss = float('inf')

    # 3. Eğitim

    for epoch in range(EPOCHS):
        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dfl_loss = 0.0
        epoch_reg_loss = 0.0

        for batch_start in range(0, len(names), BATCH_SIZE):
            optimizer.zero_grad()

            batch_end = min(batch_start + BATCH_SIZE, len(names))
            batch = names[batch_start:batch_end]
            images, gt = get_images_and_xyxy(batch, images_path, labels_path, img_size=IMG_SIZE) # np.ndarray, python nested list

            batch_loss = []
            batch_cls_loss = []
            batch_dfl_loss = []
            batch_box_loss = []

            for image, xyxy_gt in zip(images, gt): # her resim ve labeli için
                feats, *_ = ensemble_model.run(None, {'in': image}) 
                feats = torch.from_numpy(feats).to(device) 
                dist_pred = model.forward(feats) # 1, 4xregmax+nc, gs, gs
                dist_target = create_target_dist(xyxy_gt, dist_pred, nc=NC, grid_size=GRID_SIZE, reg_max=REG_MAX, img_size=IMG_SIZE, max_k=K_BEST, epoch=epoch).to(device)
                loss, loss_details = compute_loss(dist_pred, dist_target, grid_size=GRID_SIZE, reg_max=REG_MAX, nc=NC)

                batch_loss.append(loss)
                batch_cls_loss.append(loss_details[2].item())
                batch_dfl_loss.append(loss_details[1].item())
                batch_box_loss.append(loss_details[0].item())

            if batch_loss:

                batch_loss = torch.stack(batch_loss).mean()
                batch_cls_loss_val = np.mean(batch_cls_loss)
                batch_dfl_loss_val = np.mean(batch_dfl_loss)
                batch_box_loss_val = np.mean(batch_box_loss)

                batch_loss.backward()
                optimizer.step()
                if batch_loss.item() < best_loss:
                    best_loss = batch_loss.item()
                print(f"LR: {scheduler.get_last_lr()}, Epoch: {epoch+1}/{EPOCHS}, Batch: {batch_end}/{len(names)}, Best Loss: {best_loss:.4f}, Current Loss: {batch_loss.item():.4f}, boxL: {batch_box_loss_val:.4f}, clsL: {batch_cls_loss_val:.8f}, DFL: {batch_dfl_loss_val:.8f}")
                scheduler.step()
                total_loses.append(batch_loss.item())
                cls_losses.append(batch_cls_loss_val)
                dfl_losses.append(batch_dfl_loss_val)
                reg_losses.append(batch_box_loss_val)

                torch.save(model.state_dict(), f'trained_models/model_e{epoch+1}_b{batch_start//BATCH_SIZE}_l{batch_loss.item():4f}.pt')
                

    import matplotlib.pyplot as plt
    plt.plot(total_loses, label='Total Loss')
    plt.plot(cls_losses, label='Cls Loss', marker='.-.')
    plt.plot(dfl_losses, label='DFL Loss', marker='.-.')
    plt.plot(reg_losses, label='Reg Loss', marker='.-.')
    plt.ylabel('Loss')
    plt.xlabel('iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('losses.png')
        

                

