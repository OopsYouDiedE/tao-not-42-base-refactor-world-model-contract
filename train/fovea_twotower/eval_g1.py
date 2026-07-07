# -*- coding: utf-8 -*-
"""G1 闸门评测:YOLOE 域内向量校准 vs 零样本文本基线(mIoU,预登记判据)。

判据(knowledge/design_fovea_yolo_fasttower.md §2,先于结果登记):
    铁矿 mask mIoU ≥ 0.5 且比零样本文本 prompt 基线高 ≥ +0.3,640×360 原始帧。

数据:runs/data/calib640/*.npz(collect_calib640 产出:原始帧+位姿+raycast+GT方块坐标)。
GT:已知方块世界坐标 + 逐帧位姿 → 针孔投影(MC 竖直 FOV 70°,眼高 1.62)→ 前脸四角掩膜
    (投影件在 net/fovea_twotower/seg_head.py)。
    投影正确性用 --mode gtvis 人工目检(叠加图),并用 raycast 命中帧做定量核对:
    准星(图心)应落在所指方块的投影掩膜内。

校准(="调向量"):训练集帧上,GT 掩膜内的特征格采 post-BN 单位嵌入 → 每类均值再归一
    = 类原型;背景原型取 GT 外随机格。测试集(留出局)上:pf 提案 → 中心嵌入 → 与
    原型库 argmax 命名 → iron_ore 提案掩膜并集 vs GT 掩膜算 IoU。
基线:另起 native YOLOE 实例 set_classes(文本 PE) predict,iron ore 类掩膜并集同口径。

用法:
    PYTHONPATH=. .venv/bin/python train/fovea_twotower/eval_g1.py --mode gtvis
    PYTHONPATH=. .venv/bin/python train/fovea_twotower/eval_g1.py --mode run
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from net.fovea_twotower.seg_head import (FOV_V, ConvSegHead, cam_basis,  # noqa: F401
                                         gt_label_img, gt_masks, project_block)
from net.fovea_twotower.token_stream import CLASSES
from net.fovea_twotower.yolo_unified import (PAD_TOP, STRIDES, UnifiedYoloe26,
                                             pad384)

BG = "background"


def load_eps(data_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        eps.append(dict(name=os.path.basename(f)[:-4],
                        frames=z["frames"], pose=z["pose"],
                        ray_xyz=z["ray_xyz"], ray_key=z["ray_key"],
                        gt=json.loads(str(z["gt_blocks"]))))
    assert eps, f"{data_dir} 无数据"
    return eps


# ── gtvis:投影目检 + raycast 定量核对 ────────────────────────────────
def mode_gtvis(eps, out_dir, n_vis=8):
    os.makedirs(out_dir, exist_ok=True)
    colors = {"iron_ore": (0, 255, 255), "coal_ore": (255, 0, 255),
              "dirt": (0, 165, 255)}
    hit = tot = 0
    for ep in eps:
        for t in range(len(ep["frames"])):
            key = str(ep["ray_key"][t])
            if not any(c in key for c in ("iron_ore", "coal_ore", "dirt")):
                continue
            cls = next(c for c in CLASSES if c in key)
            if list(ep["ray_xyz"][t]) not in [list(b) for b in ep["gt"].get(cls, [])]:
                continue                     # raycast 命中的须是 GT 方块本体
            pts = project_block(*ep["ray_xyz"][t], ep["pose"][t])
            tot += 1
            if pts is not None:
                m = np.zeros((384, 640), np.uint8)
                cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1)
                hit += int(m[180 + PAD_TOP, 320] > 0)   # 准星=原图心(pad 后 y+12)
    print(f"[gtvis] raycast 核对:准星落在所指方块投影内 {hit}/{tot} "
          f"({hit / max(tot, 1):.2f})")
    k = 0
    for ep in eps[::4]:
        for t in range(0, len(ep["frames"]), 17):
            if k >= n_vis:
                break
            img = pad384(ep["frames"][t].transpose(1, 2, 0)).copy()
            ms = gt_masks(ep["gt"], ep["pose"][t])
            for cls, m in ms.items():
                img[m] = (0.5 * img[m] + 0.5 * np.array(colors[cls])).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"gt_{ep['name']}_t{t}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            k += 1
    print(f"[gtvis] {k} 张叠加图 → {out_dir}")


# ── run:校准 + G1 评测 ───────────────────────────────────────────────
def harvest_cells(u, eps_list, erode=5, stride_t=2, rng=None):
    """帧 → P3 格级 (单位嵌入 X [N,512], 标签 Y [N])。

    只用 P3(跨尺度 BN 统计不同,混采污染空间);训练侧 erode>0 腐蚀 GT 掩膜,
    剔除边界混合格(格级诊断:腐蚀采样下线性可分 acc 0.845/铁 recall 0.92)。"""
    rng = rng or np.random.default_rng(0)
    X, Y = [], []
    for ep in eps_list:
        for t in range(0, len(ep["frames"]), stride_t):
            img = pad384(ep["frames"][t].transpose(1, 2, 0))
            ms = gt_masks(ep["gt"], ep["pose"][t])
            if not any(m.any() for m in ms.values()):
                continue
            m = u.embed(img)[0]                       # [1,512,h,w] post-BN P3
            s = STRIDES[0]
            hh, ww = m.shape[-2:]
            gy, gx = np.mgrid[0:hh, 0:ww]
            py = np.clip((gy * s + s // 2), 0, 383)
            px = np.clip((gx * s + s // 2), 0, 639)
            e = torch.nn.functional.normalize(m[0], dim=0).permute(1, 2, 0)
            taken = np.zeros((hh, ww), bool)
            for ci, cls in enumerate(CLASSES):
                mm = ms[cls].astype(np.uint8)
                if erode:
                    mm = cv2.erode(mm, np.ones((erode, erode), np.uint8))
                sel = mm.astype(bool)[py, px]
                taken |= ms[cls][py, px]              # 排负样本用未腐蚀掩膜
                if sel.any():
                    X.append(e[torch.from_numpy(sel)].cpu())
                    Y.append(np.full(int(sel.sum()), ci))
            idx = np.argwhere(~taken)
            if len(idx):
                pick = idx[rng.choice(len(idx), min(len(idx), 40), replace=False)]
                X.append(e[pick[:, 0], pick[:, 1]].cpu())
                Y.append(np.full(len(pick), len(CLASSES)))
    return torch.cat(X).numpy(), np.concatenate(Y)


def fit_vectors(Xtr, Ytr):
    """"调向量"第 1.5 级:每类一个学习向量+偏置(多分类逻辑回归,冻结其余一切)。

    均值原型(第 1 级)格级 acc 仅 0.65 → 不弃用作对照;线性头 acc 0.845。
    返回 (W [C+1,512], b [C+1], protos [C+1,512])。"""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, Ytr)
    protos = np.stack([Xtr[Ytr == k].mean(0) / np.linalg.norm(Xtr[Ytr == k].mean(0))
                       for k in range(len(CLASSES) + 1)])
    print(f"[calib] 格样本 {len(Ytr)} | 线性头训练集 acc={lr.score(Xtr, Ytr):.3f}")
    return (torch.tensor(lr.coef_, dtype=torch.float32),
            torch.tensor(lr.intercept_, dtype=torch.float32),
            torch.tensor(protos, dtype=torch.float32))


def iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union else None


def pred_mask_dense(u, img, W, b, delta_iron=0.0, tta=False):
    """稠密口径(G1 分割主判):P3 逐格单位嵌入 → 学习向量打分 → 概率图双线性上采样
    → argmax 掩膜。可选翻转 TTA 与铁类 logit 偏置(在训练分割上标定,修类不平衡过预测)。

    动机(首轮诊断):pf 提案对十字形 GT 的召回上限仅 ~0.49,提案并集口径够不到 0.5;
    逐格分类 = YOLOE cls 分支的逐锚语义本义,仍是"只调向量"(每类一向量+偏置)。
    已证死路:2× 分辨率推理 0.248(纹理尺度偏离骨干训练分布);提案边界融合 0.36。"""
    def _prob(im):
        emb = u.embed(im)[0]                          # [1,512,h,w]
        e = torch.nn.functional.normalize(emb[0], dim=0).permute(1, 2, 0)
        lg = e @ W.to(e.device).T + b.to(e.device)    # [h,w,C+1]
        pr = lg.softmax(-1).permute(2, 0, 1)[None]
        return torch.nn.functional.interpolate(
            pr, size=(384, 640), mode="bilinear", align_corners=False)[0]
    p = _prob(img)
    if tta:
        p2 = _prob(np.ascontiguousarray(img[:, ::-1]))
        p = (p + torch.flip(p2, [-1])) / 2
    lg = p.clamp_min(1e-8).log()
    lg[0] += delta_iron
    lab = lg.argmax(0).cpu().numpy()                  # [384,640]
    return {c: lab == k for k, c in enumerate(CLASSES)}


def pred_mask_proposal(u, img, W, b, conf=0.05):
    """提案口径(token 流工况):pf 提案 → 掩膜池化嵌入 → 学习向量命名 → 掩膜并集。"""
    boxes, confs, masks = u.propose(img, conf)
    out = {c: np.zeros((384, 640), bool) for c in CLASSES}
    if masks is None or not len(boxes):
        return out
    emb = u.embed(img)
    e = u.proposal_embed(emb, boxes, masks=masks)     # [N,512] 掩膜池化
    logit = (e @ W.to(e.device).T + b.to(e.device)).cpu().numpy()
    for j in range(len(boxes)):
        k = int(np.argmax(logit[j]))
        if k < len(CLASSES):
            out[CLASSES[k]] |= masks[j]
    return out


# ── conv 头训练(阶梯第 2 级;结构在 net/fovea_twotower/seg_head.py) ─────
def train_conv_head(u, tr, epochs=6, lr=3e-4, dev="cuda", neg_frac=0.35):
    """缓存冻结 P3 嵌入 → 像素 CE(类逆频权重)训 ConvSegHead。

    neg_frac:纯负帧(无任何 GT 类可见,含 --hard_neg 纯石墙房)按比例掺入——
    v6 教训:铁类假阳性(远距斜角灰墙冒充铁)是教师锁定率 0.38 的唯一残余瓶颈。"""
    feats, labs = [], []
    n_pos = n_neg = 0
    rng = np.random.default_rng(0)
    for ep in tr:
        for t in range(0, len(ep["frames"]), 2):
            img = pad384(ep["frames"][t].transpose(1, 2, 0))
            lab = gt_label_img(ep["gt"], ep["pose"][t])
            if (lab != len(CLASSES)).sum() < 100:
                if rng.random() > neg_frac or n_neg >= max(n_pos, 40):
                    continue
                n_neg += 1
            else:
                n_pos += 1
            feats.append(u.embed(img)[0][0].half().cpu())
            labs.append(torch.from_numpy(lab))
    print(f"[conv] 缓存 {len(feats)} 帧冻结嵌入(正 {n_pos}/纯负 {n_neg})")
    cnt = torch.stack([(torch.stack(labs) == k).sum() for k in range(len(CLASSES) + 1)])
    wgt = (cnt.sum() / (cnt.float() + 1)).sqrt()
    wgt = (wgt / wgt.mean()).to(dev)
    head = ConvSegHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for e in range(epochs):
        idx = rng.permutation(len(feats))
        tot = 0.0
        for i0 in range(0, len(idx), 8):
            bi = idx[i0:i0 + 8]
            x = torch.stack([feats[i] for i in bi]).float().to(dev)
            y = torch.stack([labs[i] for i in bi]).to(dev)
            loss = torch.nn.functional.cross_entropy(head(x), y, weight=wgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
        print(f"[conv] epoch {e} loss={tot / len(idx):.4f}")
    return head.eval()


def pred_mask_conv(u, img, head, tta=False):
    def _lg(im):
        return head(u.embed(im)[0].float())[0]         # [C+1,384,640]
    with torch.no_grad():
        lg = _lg(img)
        if tta:
            lg = (lg + torch.flip(_lg(np.ascontiguousarray(img[:, ::-1])), [-1])) / 2
        lab = lg.argmax(0).cpu().numpy()
    return {c: lab == k for k, c in enumerate(CLASSES)}


def mode_run(args):
    eps = load_eps(args.data)
    if args.test_dir:                                 # 终审口径:独立测试目录(未被变体选择污染)
        tr, te = eps, load_eps(args.test_dir)
    else:
        test_names = {ep["name"] for ep in eps[::4]}  # 1/4 局留出(按局切,不跨局)
        tr = [e for e in eps if e["name"] not in test_names]
        te = [e for e in eps if e["name"] in test_names]
    print(f"[g1] train {len(tr)} eps / test {len(te)} eps"
          + (f" (test_dir={args.test_dir})" if args.test_dir else ""))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = UnifiedYoloe26(device=dev)
    if args.head == "conv":
        conv_head = train_conv_head(u, tr, dev=dev)
    Xtr, Ytr = harvest_cells(u, tr, erode=5)
    W, b, protos = fit_vectors(Xtr, Ytr)

    # 基线:native 文本 prompt(独立实例,允许其 predict 融合)
    from ultralytics import YOLOE
    base = YOLOE("runs/checkpoints/yoloe-26l-seg.pt")
    tnames = ["iron ore", "coal ore", "dirt", "stone"]
    base.set_classes(tnames, base.get_text_pe(tnames))

    ious_cal, ious_prop, ious_txt = [], [], []
    per_cls = {c: [] for c in CLASSES}
    for ep in te:
        for t in range(0, len(ep["frames"]), 2):
            img = pad384(ep["frames"][t].transpose(1, 2, 0))
            gt = gt_masks(ep["gt"], ep["pose"][t])
            if gt["iron_ore"].sum() < args.min_gt_px:
                continue
            pred = (pred_mask_conv(u, img, conv_head, tta=args.tta)
                    if args.head == "conv" else
                    pred_mask_dense(u, img, W, b, delta_iron=args.delta_iron,
                                    tta=args.tta))
            v = iou(pred["iron_ore"], gt["iron_ore"])
            if v is not None:
                ious_cal.append(v)
            for c in CLASSES:
                if gt[c].sum() >= args.min_gt_px:
                    vv = iou(pred[c], gt[c])
                    if vv is not None:
                        per_cls[c].append(vv)
            pp = pred_mask_proposal(u, img, W, b, conf=args.conf)
            ious_prop.append(iou(pp["iron_ore"], gt["iron_ore"]) or 0.0)
            r = base.predict(img, imgsz=(384, 640), conf=0.03, verbose=False,
                             device=dev)[0]
            mtxt = np.zeros((384, 640), bool)
            if r.masks is not None and r.boxes is not None:
                md = r.masks.data.cpu().numpy() > 0.5
                if md.shape[-2:] != (384, 640):
                    md = np.stack([cv2.resize(x.astype(np.uint8), (640, 384),
                                              interpolation=cv2.INTER_NEAREST) > 0
                                   for x in md])
                for j, c in enumerate(r.boxes.cls.cpu().numpy()):
                    if tnames[int(c)] == "iron ore":
                        mtxt |= md[j]
            ious_txt.append(iou(mtxt, gt["iron_ore"]) or 0.0)

    miou_cal = float(np.mean(ious_cal)) if ious_cal else 0.0
    miou_txt = float(np.mean(ious_txt)) if ious_txt else 0.0
    delta = miou_cal - miou_txt
    verdict = "PASS" if (miou_cal >= 0.5 and delta >= 0.3) else "FAIL"
    res = dict(n_frames=len(ious_cal), miou_calibrated_dense=miou_cal,
               miou_proposal_union=float(np.mean(ious_prop)) if ious_prop else 0.0,
               miou_text_baseline=miou_txt, delta=delta,
               per_class_dense={c: float(np.mean(v)) if v else None
                                for c, v in per_cls.items()},
               gate="dense mIoU>=0.5 且 delta>=+0.3", verdict=verdict,
               conf=args.conf, fov=FOV_V, delta_iron=args.delta_iron,
               tta=args.tta, head=args.head,
               test_dir=args.test_dir or "(内部1/4留出)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[g1] {json.dumps(res, indent=2, ensure_ascii=False)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gtvis", "run"], default="run")
    p.add_argument("--data", default="runs/data/calib640")
    p.add_argument("--out", default="runs/g1_yoloe_calib.json")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--test_dir", default="",
                   help="独立测试目录(终审口径;空=内部 1/4 留出)")
    p.add_argument("--delta_iron", type=float, default=0.0,
                   help="铁类 logit 偏置(训练分割标定值 -1.5)")
    p.add_argument("--tta", action="store_true", help="水平翻转 TTA")
    p.add_argument("--head", choices=["vector", "conv"], default="vector",
                   help="校准头级别:vector=每类一向量(阶梯1);conv=轻量分割头(阶梯2)")
    p.add_argument("--min_gt_px", type=int, default=250)
    p.add_argument("--vis_out", default="runs/probe_yoloe/gtvis")
    args = p.parse_args()
    if args.mode == "gtvis":
        mode_gtvis(load_eps(args.data), args.vis_out)
    else:
        mode_run(args)


if __name__ == "__main__":
    main()
