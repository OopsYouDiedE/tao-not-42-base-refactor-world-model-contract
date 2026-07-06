# -*- coding: utf-8 -*-
"""YOLOE 对 Minecraft 分割的"仅调向量"可行性探针 + 快塔解析头原型(离线诊断)。

回答两问(见 knowledge/design_fovea_yolo_fasttower.md):
  Q2 仅调向量能否适配 Minecraft 分割:对同一批 Minecraft 帧比较三种"向量"口径——
     text 文本 prompt(get_text_pe)、visual 视觉 prompt(SAVPE 原型)、prompt-free 内置词表。
     报告每帧检出数/置信/掩膜覆盖。判据登记:见设计文档 §验证。
  Q3 YOLO 作快塔解析头:parse_head 把每帧检出转成固定 [K,D] 目标 token 流(类比 DINO
     patch token 喂慢塔),验证解析管线端到端跑通并给出 [T,K,D]。

数据源:runs/data/s8_full/*.npz 的 frames(C2 采矿帧,setblock 铁矿=准星中心)。
注:s8 帧为 126×126 凹区裁剪,偏暗且 OOD;干净结论需换 640×360 原始渲染(见文档 §混淆)。

用法(CPU 即可,慢;GPU 更快):
    PYTHONPATH=. python tests/probe_yoloe_minecraft.py \
        --weights runs/checkpoints/yoloe-11l-seg.pt --n 4 --out runs/probe_yoloe
"""
import argparse
import glob
import os

import cv2
import numpy as np


def load_mc_frames(data_dir, n, out_dir, only_score_pos=True):
    """从 s8_full npz 取 n 条(score>0)轨迹各 2 帧,存 640×640 png,返回路径列表。"""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        if len(paths) >= n * 2:
            break
        z = np.load(f)
        if only_score_pos and float(z["score"]) <= 0:
            continue
        fr = z["frames"]                                   # [T,3,126,126] u8
        for ti in (len(fr) // 2, int(len(fr) * 0.8)):
            img = fr[ti].transpose(1, 2, 0)                # HWC RGB
            big = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
            p = os.path.join(out_dir, f"{os.path.basename(f)[:-4]}_t{ti}.png")
            cv2.imwrite(p, cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
            paths.append(p)
    return paths


def parse_head(res, K=8, n_cls=5):
    """YOLO 检出 → 固定 [K,7] 目标 token:[cls/n_cls, cx, cy, w, h, conf, area](归一化)。

    按 conf 取 top-K,不足补零。空间归一到 [0,1],类别归一,置信/面积原样。
    这就是"快塔解析头"的输出契约:每帧一组 object slot,时序拼成 [T,K,7]。
    """
    b = res.boxes
    H, W = res.orig_shape
    toks = np.zeros((K, 7), np.float32)
    if b is not None and len(b) > 0:
        idx = np.argsort(-b.conf.cpu().numpy())[:K]
        for j, i in enumerate(idx):
            x1, y1, x2, y2 = b.xyxy[i].cpu().numpy()
            toks[j] = [float(b.cls[i]) / n_cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                       (x2 - x1) / W, (y2 - y1) / H, float(b.conf[i]),
                       (x2 - x1) * (y2 - y1) / (W * H)]
    return toks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/checkpoints/yoloe-11l-seg.pt")
    p.add_argument("--weights_pf", default="runs/checkpoints/yoloe-26l-seg-pf.pt")
    p.add_argument("--data", default="runs/data/s8_full")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.03)
    p.add_argument("--device", default="cpu")
    p.add_argument("--classes", nargs="+",
                   default=["iron ore", "stone", "wall", "block", "ground"])
    p.add_argument("--out", default="runs/probe_yoloe")
    args = p.parse_args()

    from ultralytics import YOLOE
    frames = load_mc_frames(args.data, args.n, os.path.join(args.out, "frames"))
    assert frames, f"{args.data} 无 score>0 帧"
    print(f"[probe] {len(frames)} 帧 | 类={args.classes}", flush=True)

    m = YOLOE(args.weights)
    m.set_classes(args.classes, m.get_text_pe(args.classes))

    # Q2: 文本 prompt
    print("== Q2 文本 prompt 检出 ==")
    seq = []
    for fp in frames:
        r = m.predict(fp, conf=args.conf, verbose=False, device=args.device)[0]
        b = r.boxes
        confs = [] if b is None else [round(float(s), 3) for s in b.conf]
        print(f"  {os.path.basename(fp)}: {len(confs)} det conf={confs[:6]}")
        seq.append(parse_head(r))                          # Q3: 顺带出解析 token

    # Q3: 解析 token 流
    seq = np.stack(seq)                                    # [T,K,7]
    print(f"== Q3 解析头 token 流 shape={seq.shape}(每帧 {seq.shape[1]} 目标槽×{seq.shape[2]} 维)")
    np.save(os.path.join(args.out, "parse_tokens.npy"), seq)

    # 对照: prompt-free 内置词表(校准该 backbone 是否"看得见"结构)
    if args.weights_pf and os.path.exists(args.weights_pf):
        print("== 对照 prompt-free 内置词表(标签会是真实世界概念=域外证据)==")
        pf = YOLOE(args.weights_pf)
        r = pf.predict(frames[0], conf=0.2, verbose=False, device=args.device)[0]
        b = r.boxes
        labs = [] if b is None else [(r.names[int(c)], round(float(s), 2))
                                     for c, s in zip(b.cls, b.conf)][:6]
        print(f"  {os.path.basename(frames[0])}: {0 if b is None else len(b)} det {labs}")
        r.save(filename=os.path.join(args.out, "pf_overlay.png"))
    print(f"[probe] 产出 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
