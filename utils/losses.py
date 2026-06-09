import numpy as np
import torch
import torch.nn.functional as F

from utils.geometry import *
from utils.matching import compute_sinkhorn_matching

@torch.no_grad()
def flow_epe_px(pred_flow_norm, gt_flow_norm, valid_mask=None, img_size=256):
    """
    pred_flow_norm / gt_flow_norm: [B, 2, H, W]
    当前代码约定：flow_norm = flow_px * 2 / img_size
    """
    pred_px = pred_flow_norm * (img_size / 2.0)
    gt_px = gt_flow_norm * (img_size / 2.0)
    epe = torch.linalg.vector_norm(pred_px - gt_px, dim=1)

    if valid_mask is not None:
        valid = valid_mask.float().expand_as(epe)
        valid_sum = valid.sum()
        return (epe * valid).sum() / valid_sum if valid_sum > 0 else torch.tensor(0.0, device=epe.device)

    return epe.mean()


@torch.no_grad()
def depth_metrics(pred_depth, gt_depth, valid_mask):
    pred = pred_depth.clamp(min=1e-4)
    gt = gt_depth.clamp(min=1e-4)
    valid = valid_mask.bool()

    pred = pred[valid]
    gt = gt[valid]

    if pred.numel() == 0:
        z = torch.tensor(0.0, device=pred_depth.device)
        return {"AbsRel": z, "RMSElog": z, "Delta1": z}

    abs_rel = (pred - gt).abs().div(gt).mean()
    rmse_log = torch.sqrt(((torch.log(pred) - torch.log(gt)) ** 2).mean())

    ratio = torch.maximum(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).float().mean()

    return {
        "AbsRel": abs_rel,
        "RMSElog": rmse_log,
        "Delta1": delta1,
    }

# =====================================================================

def focal_loss(preds_logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(
        preds_logits, targets, reduction="none")
    return (alpha * (1 - torch.exp(-bce)) ** gamma * bce).mean()


def dfl_loss(pred_dist, target_distances, reg_max=16):
    if pred_dist.shape[-1] == 4:
        return torch.zeros(pred_dist.shape[:-1], device=pred_dist.device, dtype=pred_dist.dtype)

    tl = torch.clamp(target_distances.long(), 0, reg_max - 1)
    tr = torch.clamp(target_distances.long() + 1, 0, reg_max - 1)
    wl = tr.float() - target_distances
    wr = 1.0 - wl

    pred_dist = pred_dist.reshape(-1, 4, reg_max)
    loss_left = F.cross_entropy(
        pred_dist.reshape(-1, reg_max), tl.reshape(-1), reduction="none").reshape(wl.shape)
    loss_right = F.cross_entropy(
        pred_dist.reshape(-1, reg_max), tr.reshape(-1), reduction="none").reshape(wr.shape)

    return (loss_left * wl + loss_right * wr).mean(dim=-1)


def giou_loss(preds, targets):
    pl, pt, pr, pb = preds[..., :4].unbind(-1)
    tl, tt, tr, tb = targets[..., :4].unbind(-1)

    inter_area = (torch.min(pl, tl) + torch.min(pr, tr)) * \
        (torch.min(pt, tt) + torch.min(pb, tb))
    union_area = (pl + pr) * (pt + pb) + (tl + tr) * \
        (tt + tb) - inter_area + 1e-6

    convex_w = torch.max(pl, tl) + torch.max(pr, tr)
    convex_h = torch.max(pt, tt) + torch.max(pb, tb)
    convex_area = convex_w * convex_h + 1e-6

    iou = inter_area / union_area
    giou = iou - (convex_area - union_area) / convex_area
    return 1.0 - giou


def ssim_loss(x, y):
    pad_x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    pad_y = F.pad(y, (1, 1, 1, 1), mode="reflect")

    mu_x = F.avg_pool2d(pad_x, 3, 1)
    mu_y = F.avg_pool2d(pad_y, 3, 1)

    sigma_x = F.avg_pool2d(pad_x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(pad_y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(pad_x * pad_y, 3, 1) - mu_x * mu_y

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
        ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def edge_aware_smoothness_loss(depth, img):
    norm_depth = (depth.float(
    ) / torch.clamp(depth.mean(dim=[2, 3], keepdim=True).float(), min=1e-4)).to(depth.dtype)

    depth_dx = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])
    img_dx = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)

    depth_dy = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])
    img_dy = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

    return (depth_dx * torch.exp(-img_dx)).mean() + (depth_dy * torch.exp(-img_dy)).mean()

# 重构阶段性课程学习限制，实现 2D/3D 物理与运动特征的解耦训练
def get_loss_weights(step=None):
    return {
        "obj":   1.0,
        "box":   1.5,
        "mask":  1.0,
        "depth": 1.5,
        "ego":   2.0,
        "flow":  1.0,
        "attr":  0.5,
        "anom":  0.1,
        "track": 1.0,
    }


def get_ema_loss(name, current_val, alpha=0.95, ema_state=None):
    if ema_state is None:
        ema_state = {}
    with torch.no_grad():
        val = current_val.detach()
        if name not in ema_state:
            ema_state[name] = val.clone() if val > 0.0 else torch.tensor(
                1.0, device=val.device)
        if val > 0.0:
            ema_state[name] = ema_state[name] * alpha + val * (1.0 - alpha)
        return torch.clamp(ema_state[name], min=1e-4) if val > 0.0 else torch.tensor(1.0, device=val.device)

# =====================================================================
# 端到端追踪损失函数
# =====================================================================


def compute_track_loss(preds, targets, step, assignments=None, diag=None):
    if assignments is None:
        assignments = {}
    if "track_boxes" not in preds:
        device = next(iter(preds.values())
                      ).device if preds else torch.device("cuda")
        return torch.tensor(0., device=device)

    track_boxes = preds["track_boxes"]
    track_alive = preds["track_alive"]
    B, T, N, _ = track_boxes.shape
    device = track_boxes.device

    track_gt_boxes = targets.get("track_gt_boxes")
    track_gt_valid = targets.get("track_gt_valid")

    if track_gt_boxes is None or track_gt_valid is None:
        return torch.tensor(0., device=device)

    # track_gt_boxes 初始形状为 [B, T, MAX_INSTANCES, 4]
    # track_gt_valid 初始形状为 [B, T, MAX_INSTANCES]
    # 但如果 targets 是通过 _extract_target_chunk 传递的，它会被展平为 [B * T, ...]
    # 让我们重建 [B, T, ...] 形状
    if track_gt_boxes.dim() == 3:
        track_gt_boxes = track_gt_boxes.view(B, T, -1, 4)
    if track_gt_valid.dim() == 2:
        track_gt_valid = track_gt_valid.view(B, T, -1)

    loss_box = torch.tensor(0., device=device)
    loss_alive = torch.tensor(0., device=device)
    n_matched_total = 0

    # 在 GPU 上批量计算所有 B 和 T 的代价矩阵
    flat_pred_boxes = track_boxes.flatten(0, 1)      # [B*T, N, 4]
    flat_gt_boxes = track_gt_boxes.flatten(0, 1)      # [B*T, M, 4]
    cost_matrix_all = torch.cdist(flat_pred_boxes.detach(), flat_gt_boxes, p=1) # [B*T, N, M]

    b_list, t_list, q_list, g_list = [], [], [], []
    # assignments 状态由外部传入，跨 Chunk 持久化

    for t in range(T):
        alive_t = track_alive[:, t, :, 0]
        alive_target = torch.zeros(B, N, device=device)

        for b in range(B):
            idx = b * T + t
            valid_mask = track_gt_valid[b, t]
            
            if not valid_mask.any():
                continue
            
            valid_ids = torch.where(valid_mask)[0].tolist()

            used_queries = {
                q for (bb, _gid), q in assignments.items()
                if bb == b
            }

            # 1. 已绑定且当前可见的 GT，继续监督同一个 query
            new_gt_ids = []
            for gt_idx in valid_ids:
                key = (b, int(gt_idx))
                if key in assignments:
                    qi = assignments[key]
                    if qi < N:
                        alive_target[b, qi] = 1.0
                        b_list.append(b)
                        t_list.append(t)
                        q_list.append(qi)
                        g_list.append(gt_idx)
                else:
                    new_gt_ids.append(gt_idx)

            # 2. 新出现 GT 才使用 GPU Greedy 分配空闲 query
            if len(new_gt_ids) > 0:
                free_queries = [q for q in range(N) if q not in used_queries]
                if len(free_queries) == 0:
                    continue

                # cost shape: [len(free_queries), len(new_gt_ids)]
                cost = cost_matrix_all[idx][free_queries][:, new_gt_ids].clone()

                # GPU 原生 Sinkhorn 算法求软分配概率矩阵 (Zero CPU-GPU Sync)
                P_opt = compute_sinkhorn_matching(cost, epsilon=0.1, iters=10)

                # 将软概率向量化硬化匹配 (0 次微观 CPU 同步阻断)
                fq_tensor = torch.tensor(free_queries, device=device, dtype=torch.long)
                gt_tensor = torch.tensor(new_gt_ids, device=device, dtype=torch.long)

                # 1. 沿列方向 (GT 维度) 寻找每个 Query 匹配的最大概率
                max_probs, matched_cols = torch.max(P_opt, dim=1)

                # 2. 设定匹配阈值筛选有效匹配 (完全在 GPU 上完成)
                valid_matches = max_probs > 0.1

                # 3. 提取匹配的 Query 和 GT 的原始索引
                matched_qs = fq_tensor[valid_matches]
                matched_gs = gt_tensor[matched_cols[valid_matches]]

                # 4. 更新 alive_target (利用 GPU 级高级索引，0 次 CPU 同步)
                alive_target[b, matched_qs] = 1.0

                # 5. 回写 assignments 字典与列表统计 (单次批量传输)
                mq_list = matched_qs.tolist()
                mg_list = matched_gs.tolist()
                seen_gts = set()
                for qi, gt_idx in zip(mq_list, mg_list):
                    gt_idx_int = int(gt_idx)
                    if gt_idx_int not in seen_gts:
                        seen_gts.add(gt_idx_int)
                        qi_int = int(qi)
                        assignments[(b, gt_idx_int)] = qi_int
                        q_list.append(qi_int)
                        g_list.append(gt_idx_int)
                        b_list.append(b)
                        t_list.append(t)

        loss_alive = loss_alive + F.binary_cross_entropy_with_logits(
            alive_t, alive_target
        )

    if len(b_list) > 0:
        b_idx = torch.tensor(b_list, dtype=torch.long, device=device)
        t_idx = torch.tensor(t_list, dtype=torch.long, device=device)
        q_idx = torch.tensor(q_list, dtype=torch.long, device=device)
        g_idx = torch.tensor(g_list, dtype=torch.long, device=device)

        pred_boxes_matched = track_boxes[b_idx, t_idx, q_idx]
        gt_boxes_matched = track_gt_boxes[b_idx, t_idx, g_idx]
        
        loss_box = F.smooth_l1_loss(pred_boxes_matched, gt_boxes_matched, beta=0.1, reduction="sum")
        n_matched_total = len(b_list)

        # [诊断] 拆分 t=0（GT 播种帧，几乎免费）与 t>0（真正的时序传播）的框误差。
        # Track loss 整体很小可能只是因为 t=0 被 GT 喂了；只有 t>0 也低、且场景有运动，
        # 才证明它真的在“追踪”而非冻住首帧框。
        if diag is not None:
            with torch.no_grad():
                per_match = F.smooth_l1_loss(
                    pred_boxes_matched, gt_boxes_matched, beta=0.1, reduction="none").mean(dim=-1)
                is_t0 = (t_idx == 0)
                diag["TrackBoxT0"] = per_match[is_t0].mean().detach() if is_t0.any() \
                    else torch.tensor(0., device=device)
                diag["TrackBoxTpos"] = per_match[~is_t0].mean().detach() if (~is_t0).any() \
                    else torch.tensor(0., device=device)
    else:
        n_matched_total = 1

    n_matched_total = max(n_matched_total, 1)
    loss_box = loss_box / n_matched_total
    loss_alive = loss_alive / T

    return 1.5 * loss_box + 0.5 * loss_alive

def compute_instance_loss(preds, targets):
    B = preds["objectness"][0].shape[0]
    device = preds["objectness"][0].device
    num_scales = len(preds["objectness"])

    loss_obj = torch.tensor(0.0, device=device)
    loss_box = torch.tensor(0.0, device=device)
    loss_mask = torch.tensor(0.0, device=device)

    for i in range(num_scales):
        p_obj, t_obj = preds["objectness"][i], targets["obj_dense"][i]

        loss_obj += focal_loss(p_obj, t_obj)
        if "dense_objectness" in preds:
            loss_obj += focal_loss(preds["dense_objectness"][i], t_obj) * 0.5

        pos_mask = t_obj[:, 0] > 0.5

        pb = preds["boxes"][i].permute(0, 2, 3, 1)
        tb = targets["bboxes_dense"][i].permute(0, 2, 3, 1)
        pdist = preds["box_dist"][i].permute(0, 2, 3, 1)
        giou = giou_loss(pb, tb)
        box_l = (giou * 1.5 + dfl_loss(pdist, tb, 32) * 0.5) * pos_mask.float()
        pos_sum = pos_mask.float().sum()
        loss_box += box_l.sum() / pos_sum if pos_sum > 0 else torch.tensor(0.0, device=device)

        pos_mask_b = pos_mask.bool()
        if pos_mask_b.any():
            b_idx, y_idx, x_idx = torch.where(pos_mask_b)
            mc = preds["mask_coefficients"][i].permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]
            protos_n = preds["mask_prototypes"][b_idx]
            pred_logits = torch.einsum("nc,nchw->nhw", mc, protos_n)

            H, W = targets["seg_raw"].shape[1], targets["seg_raw"].shape[2]
            stride = 8 * (2 ** i)
            gy = (y_idx.float() * stride + stride / 2.0).long().clamp(0, H - 1)
            gx = (x_idx.float() * stride + stride / 2.0).long().clamp(0, W - 1)

            inst_ids = targets["seg_raw"][b_idx, gy, gx]
            gt_masks_full = (targets["seg_small"][b_idx] == inst_ids.view(-1, 1, 1)).float()
            if gt_masks_full.shape[-2:] != pred_logits.shape[-2:]:
                gt_masks_full = F.interpolate(gt_masks_full.unsqueeze(1), size=pred_logits.shape[-2:], mode="nearest").squeeze(1)

            tb = targets["bboxes_dense"][i].permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]
            mask_stride = H / pred_logits.shape[-2]
            gy_m = (y_idx.float() * stride + stride / 2.0) / mask_stride
            gx_m = (x_idx.float() * stride + stride / 2.0) / mask_stride
            x1 = (gx_m - tb[:, 0] * stride / mask_stride).clamp(0, pred_logits.shape[-1] - 1)
            y1 = (gy_m - tb[:, 1] * stride / mask_stride).clamp(0, pred_logits.shape[-2] - 1)
            x2 = (gx_m + tb[:, 2] * stride / mask_stride).clamp(0, pred_logits.shape[-1] - 1)
            y2 = (gy_m + tb[:, 3] * stride / mask_stride).clamp(0, pred_logits.shape[-2] - 1)

            rows = torch.arange(pred_logits.shape[-2], device=device).view(1, -1, 1)
            cols = torch.arange(pred_logits.shape[-1], device=device).view(1, 1, -1)
            box_mask = (cols >= x1.view(-1, 1, 1)) & (cols <= x2.view(-1, 1, 1)) & \
                       (rows >= y1.view(-1, 1, 1)) & (rows <= y2.view(-1, 1, 1))

            pred_logits_crop = pred_logits.masked_fill(~box_mask, -10.0)
            gt_masks_crop = gt_masks_full * box_mask.float()

            intersection = (torch.sigmoid(pred_logits_crop) * gt_masks_crop).sum(dim=(1, 2))
            union = torch.sigmoid(pred_logits_crop).sum(dim=(1, 2)) + gt_masks_crop.sum(dim=(1, 2))
            bce = F.binary_cross_entropy_with_logits(pred_logits_crop, gt_masks_crop, reduction="none")
            focal_bce = (0.25 * (1 - torch.exp(-bce)) ** 2 * bce).mean(dim=(1, 2))

            gt_area = gt_masks_crop.sum(dim=(1, 2))
            dice_loss = 1.0 - (2.0 * intersection + gt_area * 0.01 + 1e-4) / \
                              (union + gt_area * 0.01 + 1e-4)
            valid_mask_inst = (inst_ids > 0).float()
            inst_sum = valid_mask_inst.sum()
            loss_mask += ((dice_loss * 2.0 + focal_bce) * valid_mask_inst).sum() / inst_sum if inst_sum > 0 else torch.tensor(0.0, device=device)

    return loss_obj, loss_box, loss_mask

def compute_attribute_loss(preds, targets):
    if "attributes" not in preds:
        return torch.tensor(0.0, device=next(iter(preds.values())).device)

    loss = torch.tensor(0.0, device=preds["attributes"][0].device)
    n_terms = 0

    for i, pred_attr in enumerate(preds["attributes"]):
        obj = targets["obj_dense"][i]
        pos = obj[:, 0] > 0.5

        if pos.sum() == 0:
            continue

        init_dyn = targets["initial_dynamic_dense"][i][:, 0]
        cur_mov = targets["current_moving_dense"][i][:, 0]
        init_valid = targets.get("initial_dynamic_valid_dense", None)
        cur_valid = targets.get("current_moving_valid_dense", None)

        if init_valid is not None:
            init_pos = pos & (init_valid[i][:, 0] > 0.5)
        else:
            init_pos = pos

        if init_pos.any():
            loss_init = F.binary_cross_entropy_with_logits(
                pred_attr[:, 0], init_dyn.float(), reduction="none"
            )
            init_sum = init_pos.float().sum()
            loss = loss + (loss_init * init_pos.float()).sum() / init_sum if init_sum > 0 else loss
            n_terms += 1

        if cur_valid is not None:
            cur_pos = pos & (cur_valid[i][:, 0] > 0.5)
        else:
            cur_pos = pos

        if cur_pos.any():
            loss_cur = F.binary_cross_entropy_with_logits(
                pred_attr[:, 1], cur_mov.float(), reduction="none"
            )
            cur_sum = cur_pos.float().sum()
            loss = loss + (loss_cur * cur_pos.float()).sum() / cur_sum if cur_sum > 0 else loss
            n_terms += 1

    return loss / max(n_terms, 1)


def compute_physics_loss(preds, targets, img_t=None, img_next=None, mode="supervised", step=0, ema_state=None, assignments=None):
    device = preds["depth"].device
    H, W = preds["depth"].shape[-2:]
    w = get_loss_weights(step)

    loss_obj, loss_box, loss_mask = compute_instance_loss(preds, targets)

    # [诊断] Obj 召回率 / 正样本平均置信度：focal loss 的数值会被“背景坍缩”骗到极小，
    # 这两个指标不可作弊——坍缩时 ObjRecall≈0，真学到时 ObjRecall→1。
    obj_recall = torch.tensor(0.0, device=device)
    obj_pos_conf = torch.tensor(0.0, device=device)
    if "objectness" in preds:
        with torch.no_grad():
            n_pos_total = torch.tensor(0.0, device=device)
            for i in range(len(preds["objectness"])):
                t_pos = targets["obj_dense"][i][:, 0] > 0.5
                if t_pos.any():
                    p = torch.sigmoid(preds["objectness"][i][:, 0])[t_pos]
                    obj_recall = obj_recall + (p > 0.5).float().sum()
                    obj_pos_conf = obj_pos_conf + p.sum()
                    n_pos_total = n_pos_total + t_pos.float().sum()
            n_pos_total = n_pos_total.clamp(min=1.0)
            obj_recall = obj_recall / n_pos_total
            obj_pos_conf = obj_pos_conf / n_pos_total
    loss_ego, loss_depth, loss_flow = [torch.tensor(0.0, device=device) for _ in range(3)]
    loss_track = torch.tensor(0.0, device=device)
    loss_attr = torch.tensor(0.0, device=device)

    B, T = preds["track_boxes"].shape[:2]
    # 强制屏蔽所有 t=0 的时序预测（由于没有历史信息，第一帧预测纯属盲猜，屏蔽以保护前期特征）
    if "has_next" in targets:
        has_next_mod = targets["has_next"].view(B, T).clone()
        has_next_mod[:, 0] = False
        targets["has_next"] = has_next_mod.flatten(0, 1)

    if mode == "supervised" and "cam_pos_t" in targets and "cam_pos_next" in targets:
        R_n_inv = quaternion_to_matrix(
            targets["cam_quat_next"]).transpose(1, 2)
        trans_diff = torch.bmm(
            R_n_inv, (targets["cam_pos_t"] - targets["cam_pos_next"]).unsqueeze(-1)).squeeze(-1)
        rot_diff = matrix_to_6d(
            torch.bmm(R_n_inv, quaternion_to_matrix(targets["cam_quat_t"])))
        gt_pose = torch.cat([trans_diff, rot_diff], dim=1)

        pred_pose_vec = torch.cat([preds["ego_pose"]["t"], preds["ego_pose"]["rot6d"]], dim=-1)
        l_ego_raw = F.smooth_l1_loss(pred_pose_vec, gt_pose, reduction="none")
        ego_mask = targets["has_next"].view(B * T, 1).float() if "has_next" in targets else torch.ones_like(l_ego_raw)
        loss_ego = (l_ego_raw * ego_mask).sum() / (ego_mask.sum() * 9 + 1e-6)

        v_d_mask = (~targets["sky_mask"]).float()
        v_d_mask_sum = v_d_mask.sum()
        l_depth_base = (F.smooth_l1_loss(
            preds["log_depth"], targets["log_depth"], reduction="none") * v_d_mask).sum() / v_d_mask_sum if v_d_mask_sum > 0 else torch.tensor(0.0, device=device)

        # log 空间梯度正则：∂log_depth/∂logit = -1（常数），消除远景梯度爆炸
        log_d_pred = preds["log_depth"]
        log_d_gt   = targets["log_depth"]

        pd_dx = log_d_pred[:, :, 1:] - log_d_pred[:, :, :-1]
        td_dx = log_d_gt[:, :, 1:]   - log_d_gt[:, :, :-1]
        mask_dx = v_d_mask[:, :, 1:] * v_d_mask[:, :, :-1]
        mask_dx_sum = mask_dx.sum().clamp(min=1)
        l_depth_dx = F.smooth_l1_loss(
            pd_dx * mask_dx, td_dx * mask_dx, reduction="sum") / mask_dx_sum

        pd_dy = log_d_pred[:, 1:, :] - log_d_pred[:, :-1, :]
        td_dy = log_d_gt[:, 1:, :]   - log_d_gt[:, :-1, :]
        mask_dy = v_d_mask[:, 1:, :] * v_d_mask[:, :-1, :]
        mask_dy_sum = mask_dy.sum().clamp(min=1)
        l_depth_dy = F.smooth_l1_loss(
            pd_dy * mask_dy, td_dy * mask_dy, reduction="sum") / mask_dy_sum

        loss_depth = l_depth_base + 0.5 * (l_depth_dx + l_depth_dy)

        if img_t is not None:
            l_edge_smooth = edge_aware_smoothness_loss(
                preds["depth"].unsqueeze(1), img_t
            )
            loss_depth = loss_depth + 0.05 * l_edge_smooth

    ret_flow_epe = torch.tensor(0.0, device=device)
    if w["flow"] > 0 and preds.get("flow") is not None and "flow_target" in targets:
        if "has_next" in targets:
            has_n = targets["has_next"].view(-1, 1, 1, 1).float()
            has_n_sum = has_n.sum()
            l_flow_raw = F.smooth_l1_loss(
                preds["flow"], targets["flow_target"], reduction="none", beta=0.05) * has_n
            loss_flow = l_flow_raw.sum() / (has_n_sum *
                                             preds["flow"].shape[1] * H * W) if has_n_sum > 0 else torch.tensor(0.0, device=device)

            if has_n_sum > 0:
                # 光流场梯度匹配：∇flow_pred ≈ ∇flow_GT（与深度梯度正则同构）
                # 强迫光流场具有正确的空间方向结构，而不仅仅是像素均值
                fp, fg = preds["flow"], targets["flow_target"]
                fp_dx = fp[:, :, :, 1:] - fp[:, :, :, :-1]
                fg_dx = fg[:, :, :, 1:] - fg[:, :, :, :-1]
                fp_dy = fp[:, :, 1:, :] - fp[:, :, :-1, :]
                fg_dy = fg[:, :, 1:, :] - fg[:, :, :-1, :]
                has_dx = has_n.expand_as(fp_dx)
                has_dy = has_n.expand_as(fp_dy)
                l_flow_dx = (F.smooth_l1_loss(fp_dx, fg_dx, reduction="none", beta=0.05) * has_dx).sum() / (has_n_sum * 2 * H * (W - 1) + 1e-6)
                l_flow_dy = (F.smooth_l1_loss(fp_dy, fg_dy, reduction="none", beta=0.05) * has_dy).sum() / (has_n_sum * 2 * (H - 1) * W + 1e-6)
                loss_flow = loss_flow + 0.5 * (l_flow_dx + l_flow_dy)

                # 边缘感知光流平滑：图像边缘处允许光流突变，平坦区域强制平滑
                if img_t is not None:
                    flow_x = preds["flow"][:, :1] * has_n  # dx 分量
                    flow_y = preds["flow"][:, 1:] * has_n  # dy 分量
                    l_fs_x = edge_aware_smoothness_loss(flow_x, img_t)
                    l_fs_y = edge_aware_smoothness_loss(flow_y, img_t)
                    loss_flow = loss_flow + 0.02 * (l_fs_x + l_fs_y) * 0.5
        else:
            loss_flow = F.smooth_l1_loss(preds["flow"], targets["flow_target"], beta=0.05)

    if preds.get("flow") is not None and "flow_target" in targets:
        ret_flow_epe = flow_epe_px(
            preds["flow"].detach(),
            targets["flow_target"].detach(),
            valid_mask=targets.get("has_next", None).view(-1, 1, 1)
            if "has_next" in targets else None,
            img_size=W,
        )

    depth_abs_rel = torch.tensor(0.0, device=device)
    depth_rmse_log = torch.tensor(0.0, device=device)
    depth_delta1 = torch.tensor(0.0, device=device)
    if mode == "supervised" and "depth" in targets and "sky_mask" in targets:
        with torch.no_grad():
            d_metrics = depth_metrics(
                preds["depth"].detach(),
                targets["depth"].detach(),
                ~targets["sky_mask"].detach(),
            )
            depth_abs_rel = d_metrics["AbsRel"]
            depth_rmse_log = d_metrics["RMSElog"]
            depth_delta1 = d_metrics["Delta1"]

    warped_img = None
    if img_t is not None:
        if img_next is not None:
            K, K_inv = generate_intrinsics(
                H,
                W,
                device,
                focal_length=targets.get("camera_focal_length", None),
                sensor_width=targets.get("camera_sensor_width", None),
                dtype=preds["depth"].dtype,
            )
            pred_pose_vec = torch.cat([preds["ego_pose"]["t"], preds["ego_pose"]["rot6d"]], dim=-1)
            warped_img, _ = inverse_warp(
                img_next,
                preds["depth"].unsqueeze(1),
                pred_pose_vec,
                K,
                K_inv,
                depth_is_distance=True,
            )

    photo_err = torch.tensor(0.0, device=device)
    if warped_img is not None:
        with torch.no_grad():
            valid = (warped_img.abs().sum(dim=1, keepdim=True) > 0.01).float()
            photo_err = ((warped_img - img_t).abs() * valid).sum() / (valid.sum() * 3 + 1e-6)

    loss_anom = preds["feature_error"].mean().clamp(max=10.0)

    track_diag = {}
    if w.get("track", 0) > 0 and "track_boxes" in preds:
        loss_track = compute_track_loss(preds, targets, step, assignments, diag=track_diag)

    if w.get("attr", 0) > 0 and "attributes" in preds:
        loss_attr = compute_attribute_loss(preds, targets)

    loss_components = {
        "Obj": loss_obj, "Box": loss_box, "Mask": loss_mask,
        "Depth": loss_depth, "Ego": loss_ego,
        "Flow": loss_flow, "Anom": loss_anom, "Attr": loss_attr
    }

    if ema_state is not None:
        tot = torch.tensor(0.0, device=device)
        for k, l in loss_components.items():
            wk = w.get(k.lower(), 0)
            if wk > 0:
                tot = tot + wk * (l / get_ema_loss(k.lower(), l, ema_state=ema_state))
        if w.get("track", 0) > 0:
            tot = tot + w["track"] * (loss_track / get_ema_loss("track", loss_track, ema_state=ema_state))
    else:
        tot = sum(w.get(k.lower(), 0) * l for k, l in loss_components.items())
        tot += w["track"] * loss_track

    ret_dict = {k: v.detach() for k, v in loss_components.items() if w.get(k.lower(), 0) > 0}
    if w["track"] > 0:
        ret_dict["Track"] = loss_track.detach()
        ret_dict["TrackBoxT0"] = track_diag.get("TrackBoxT0", torch.tensor(0.0, device=device))
        ret_dict["TrackBoxTpos"] = track_diag.get("TrackBoxTpos", torch.tensor(0.0, device=device))

    ret_dict["ObjRecall"] = obj_recall.detach()
    ret_dict["ObjPosConf"] = obj_pos_conf.detach()
    ret_dict["FlowEPEpx"] = ret_flow_epe.detach()
    ret_dict["DepthAbsRel"] = depth_abs_rel.detach()
    ret_dict["DepthRMSElog"] = depth_rmse_log.detach()
    ret_dict["DepthDelta1"] = depth_delta1.detach()
    ret_dict["PhotoErr"] = photo_err.detach()
    ret_dict["Tot"] = tot.detach()

    return tot, ret_dict, warped_img

# =====================================================================

@torch.no_grad()
def update_ema_teacher(student_model, teacher_model, momentum=0.996):
    """更新 EMA 教师网络 (I8: 停止梯度)。"""
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data = param_t.data * momentum + param_s.data * (1.0 - momentum)


def info_nce_loss(z_pred, z_target, temperature=0.07):
    """控制变量反事实对比损失。fp32(I4), 分母安全(I1)。
    z_pred: [B, N, d] 预测特征
    z_target: [B, N, d] 教师网络输出目标特征
    """
    B, N, d = z_pred.shape
    # Flatten to [B*N, d]
    z_pred = z_pred.reshape(B * N, d)
    z_target = z_target.reshape(B * N, d)
    
    # normalize (I1: eps=1e-4)
    z_pred = F.normalize(z_pred, dim=-1, eps=1e-4)
    z_target = F.normalize(z_target, dim=-1, eps=1e-4)
    
    # logits (fp32 I4)
    logits = torch.matmul(z_pred.float(), z_target.float().T) / temperature
    labels = torch.arange(B * N, device=z_pred.device)
    
    return F.cross_entropy(logits, labels)


def gaussian_nll_loss(mu, sigma, target, active_mask):
    """高斯负对数似然损失 (JEPA 概率云对齐)。
    包含了两项的博弈:
    1. ||mu - target||^2 / sigma^2 : 迫使 mu 逼近 target。如果不准，模型会放大 sigma 逃避惩罚。
    2. log(sigma^2) : 惩罚过大的方差，迫使模型在可能的情况下尽可能确信 (反偷懒)。
    """
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=mu.device)
    
    # I1: 限制 sigma 的下界，防止除 0 和 log(0)
    sigma2 = (sigma ** 2).clamp(min=1e-4) 
    
    # 均方误差项
    mse = F.mse_loss(mu, target, reduction='none')
    
    # NLL = (mse / sigma^2) + log(sigma^2)
    nll = (mse / sigma2) + torch.log(sigma2)
    
    # 仅对存在概率 > 0.5 的活跃 Slot 计算
    valid_nll = (nll * active_mask.unsqueeze(-1)).sum(-1)
    
    return valid_nll.sum() / active_mask.sum()


def _sinkhorn_batched(cost, epsilon=0.1, iters=20):
    """批量 Sinkhorn。cost:[B,K,M] → 软分配 P:[B,K,M]。纯 tensor,GPU 零同步。"""
    P = torch.exp(-cost / epsilon)                       # [B,K,M]
    u = torch.ones_like(P[:, :, 0])                      # [B,K]
    v = torch.ones_like(P[:, 0, :])                      # [B,M]
    for _ in range(iters):
        u = 1.0 / ((P * v.unsqueeze(1)).sum(-1) + 1e-8)  # [B,K]
        v = 1.0 / ((P * u.unsqueeze(-1)).sum(1) + 1e-8)  # [B,M]
    return u.unsqueeze(-1) * P * v.unsqueeze(1)


def action_plan_loss(pred, gt, w_onset=2.0, w_dur=1.0, w_key=0.5, w_exist=1.0):
    """DETR 式集合匹配:K 个预测动作 ↔ M 个 GT 待击打动作。全批量,GPU 零同步。

    pred: {key_logits[B,K,n_keys], onset[B,K], duration[B,K], exist[B,K]}
    gt  : {onset[B,M], duration[B,M], track[B,M], valid[B,M]}  (来自 env.get_upcoming_actions)

    匹配代价 = w_onset·|Δonset| + w_dur·|Δdur| + w_key·(1-p_key)。匹配用 Sinkhorn(detach)。
    匹配上的:回归 onset/时长 + 键分类 + exist→1;未匹配的:exist→0。
    指标返回 0 维 tensor(调用方按需 .item(),热路径不同步)。
    """
    on_p, dur_p, key_p, ex_p = pred["onset"], pred["duration"], pred["key_logits"], pred["exist"]
    gon, gdur, gtrk, gval = gt["onset"], gt["duration"], gt["track"], gt["valid"]
    B, K = on_p.shape
    M = gon.shape[1]
    n_keys = key_p.shape[-1]
    eps = 1e-6

    with torch.no_grad():
        cp = key_p.softmax(-1)                                       # [B,K,n_keys]
        key_cost = 1.0 - cp.gather(2, gtrk.clamp(0, n_keys - 1)      # [B,K,M]
                                   .unsqueeze(1).expand(B, K, M))
        cost = (w_onset * (on_p.unsqueeze(2) - gon.unsqueeze(1)).abs()
                + w_dur * (dur_p.unsqueeze(2) - gdur.unsqueeze(1)).abs()
                + w_key * key_cost)
        cost = cost + (1.0 - gval).unsqueeze(1) * 1e4               # 屏蔽无效 GT 列
        P = _sinkhorn_batched(cost)                                 # [B,K,M]
        sel = P.argmax(dim=1)                                       # [B,M] 每个 GT 选一个槽

    on_sel = on_p.gather(1, sel)                                    # [B,M]
    dur_sel = dur_p.gather(1, sel)
    key_sel = key_p.gather(1, sel.unsqueeze(-1).expand(B, M, n_keys))  # [B,M,n_keys]

    gsum = gval.sum().clamp(min=1.0)
    L_on = (F.smooth_l1_loss(on_sel, gon, reduction="none", beta=0.2) * gval).sum() / gsum
    L_dur = (F.smooth_l1_loss(dur_sel, gdur, reduction="none", beta=0.2) * gval).sum() / gsum
    ce = F.cross_entropy(key_sel.reshape(B * M, n_keys),
                         gtrk.reshape(B * M).clamp(0, n_keys - 1), reduction="none").reshape(B, M)
    L_key = (ce * gval).sum() / gsum

    # exist 目标:匹配到的槽→1(OR 语义,防同槽冲突),其余→0
    ex_t = torch.zeros_like(ex_p).scatter_reduce_(1, sel, gval, reduce="amax", include_self=True)
    L_ex = F.binary_cross_entropy(ex_p, ex_t)

    total = w_onset * L_on + w_dur * L_dur + w_key * L_key + w_exist * L_ex

    with torch.no_grad():
        off = 1.0 - ex_t
        metrics = {
            "OnsetMAEms": 1000.0 * (on_sel - gon).abs().mul(gval).sum() / gsum,
            "DurMAEms": 1000.0 * (dur_sel - gdur).abs().mul(gval).sum() / gsum,
            "KeyAcc": ((key_sel.argmax(-1) == gtrk).float() * gval).sum() / gsum,
            "ExistOn": (ex_p * ex_t).sum() / ex_t.sum().clamp(min=1.0),
            "ExistOff": (ex_p * off).sum() / off.sum().clamp(min=1.0),
        }
    return total, metrics


def gaussian_action_loss(pred_action, target_times, current_time, sigma_gauss=0.03):
    """高斯涂抹动作监督损失。"""
    # y = exp(-(Δt)^2 / (2 * σ^2))
    dt = current_time.unsqueeze(1) - target_times # [B, num_targets]
    y_target = torch.exp(-(dt**2) / (2 * sigma_gauss**2))
    # 取该时刻最接近的按键事件
    y_target, _ = y_target.max(dim=1) 
    
    return F.binary_cross_entropy(pred_action, y_target.unsqueeze(-1).expand_as(pred_action))
