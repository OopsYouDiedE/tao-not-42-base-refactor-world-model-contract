import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision



# =====================================================================

def flow_to_color(flow_np, max_mag=None):
    """光流 → BGR 颜色。方向决定色相 (hue)，幅度决定明度 (value)。

    关键：不再做"每张图各自减中位数、各自除以自身 max"的处理——那会让每个面板
    用不同的零点和不同的尺度，导致 GT/Pred/各分量之间颜色不可比。这里改为接收一个
    跨所有面板共享的 max_mag，使同一条光流向量在任何面板都映射到完全相同的颜色。
    """
    flow_np = flow_np.astype(np.float32)
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    if max_mag is None:
        max_mag = float(mag.max())
    hsv = np.zeros((*flow_np.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 90.0 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (max_mag + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def mask_to_color(m_np):
    """[0,1] 概率图 → 固定 0..1 标准的 VIRIDIS 伪彩，保证 pred 与 GT 同标准。"""
    return cv2.applyColorMap((np.clip(m_np, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)




def depth_to_color(depth_map, d_min=None, d_max=None):
    d_min = d_min if d_min is not None else depth_map.min()
    d_max = d_max if d_max is not None else depth_map.max()
    d_norm = (depth_map - d_min) / \
        (d_max - d_min) if d_max > d_min else np.zeros_like(depth_map)
    return cv2.applyColorMap((np.clip(d_norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)



def extract_instances(preds, score_thresh=0.3, nms_thresh=0.5, max_det=20, with_nms=True):
    obj_list = preds.get("objectness", [])
    box_list = preds.get("boxes", [])

    if not isinstance(obj_list, list):
        obj_list, box_list = [obj_list], [box_list]
        cls_list = [preds.get("classification")
                    ] if "classification" in preds else []
        coef_list = [preds.get("mask_coefficients")
                     ] if "mask_coefficients" in preds else []
    else:
        cls_list = preds.get("classification", [])
        coef_list = preds.get("mask_coefficients", [])

    B = obj_list[0].shape[0] if obj_list else 0
    device = obj_list[0].device if obj_list else torch.device("cuda")
    H_img, W_img = (obj_list[0].shape[2] * 8,
                    obj_list[0].shape[3] * 8) if obj_list else (0, 0)
    results = []

    for b in range(B):
        all_scores, all_boxes, all_masks_info, all_classes = [], [], [], []
        for i, (obj, box) in enumerate(zip(obj_list, box_list)):
            stride = 8 * (2 ** i)
            if box is None:
                continue

            obj_score = torch.sigmoid(obj[b, 0])
            # 由于实际训练为类无关的物体发现 (class-agnostic object discovery)，分类头未训练，
            # 乘法融合会导致预测置信度被未训练的分类概率严重抑制，因此直接使用 obj_score 作为置信度。
            final_score = obj_score

            valid = final_score > score_thresh
            if not valid.any():
                continue

            sel_scores = final_score[valid]
            decoded_boxes = box[b][:, valid].T
            cy, cx = valid.nonzero(as_tuple=True)

            grid_x_norm = (cx.float() * stride + stride / 2.0) / W_img
            grid_y_norm = (cy.float() * stride + stride / 2.0) / H_img

            pl_norm = decoded_boxes[:, 0] * stride / W_img
            pt_norm = decoded_boxes[:, 1] * stride / H_img
            pr_norm = decoded_boxes[:, 2] * stride / W_img
            pb_norm = decoded_boxes[:, 3] * stride / H_img

            decoded_boxes_norm = torch.stack([
                torch.clamp(grid_x_norm - pl_norm, 0.0, 1.0),
                torch.clamp(grid_y_norm - pt_norm, 0.0, 1.0),
                torch.clamp(grid_x_norm + pr_norm, 0.0, 1.0),
                torch.clamp(grid_y_norm + pb_norm, 0.0, 1.0)
            ], dim=-1)

            all_scores.append(sel_scores)
            all_boxes.append(decoded_boxes_norm)

            if cls_list and i < len(cls_list) and cls_list[i] is not None:
                all_classes.append(torch.argmax(
                    cls_list[i][b, :, cy, cx].T, dim=-1))
            else:
                all_classes.append(torch.zeros_like(
                    sel_scores, dtype=torch.long))

            if coef_list and i < len(coef_list) and coef_list[i] is not None:
                all_masks_info.append(coef_list[i][b, :, cy, cx].T)

        if not all_scores:
            results.append(None)
            continue

        all_scores = torch.cat(all_scores, dim=0)
        all_boxes = torch.cat(all_boxes, dim=0)
        all_classes = torch.cat(all_classes, dim=0)

        if with_nms:
            boxes_scaled = all_boxes * torch.tensor([W_img, H_img, W_img, H_img], device=device)
            keep = torchvision.ops.nms(boxes_scaled, all_scores, nms_thresh)[:max_det]
        else:
            # 端到端检测，无需 NMS，直接取 Top-K
            num_keep = min(max_det, all_scores.shape[0])
            _, keep = torch.topk(all_scores, num_keep)

        protos = preds.get("mask_prototypes")
        protos = protos[0] if isinstance(protos, list) else protos
        masks_bool = None

        if protos is not None and all_masks_info:
            all_masks_info = torch.cat(all_masks_info, dim=0)
            masks = F.interpolate(
                torch.einsum(
                    "kp,phw->khw", all_masks_info[keep], protos[b]).unsqueeze(0),
                size=(H_img, W_img), mode="bilinear", align_corners=False
            )[0]

            x1, y1, x2, y2 = (
                all_boxes[keep] * torch.tensor([W_img, H_img, W_img, H_img], device=device)).unbind(-1)
            rows = torch.arange(H_img, device=device).view(1, H_img, 1)
            cols = torch.arange(W_img, device=device).view(1, 1, W_img)

            masks_bool = (masks > 0) & (cols >= x1.view(-1, 1, 1)) & (cols < x2.view(-1,
                                                                                     1, 1)) & (rows >= y1.view(-1, 1, 1)) & (rows < y2.view(-1, 1, 1))

        results.append({"scores": all_scores[keep], "boxes": all_boxes[keep],
                       "masks": masks_bool, "classes": all_classes[keep]})

    return results


