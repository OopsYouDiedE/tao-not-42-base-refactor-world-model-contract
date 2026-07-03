"""gaming500 闸门测试 Stage A:冻结 DINOv3 特征 + 30Hz 子帧动作抽取(小规模)。

设计出处:knowledge/design_gaming500_consume.md(变率对齐)、2026-07-03 三轮辩论收敛结论
(前置探针:Δz→dx 线性可解码性;子帧有序动作条件)。本脚本是**一次性探针工具**,不入 net/train。

对每个选中段:
  · JPEG → 176px resize → 冻结 DINOv3 ViT-S/16 → patch 特征 [N,384,11,11] fp16
  · 相邻图像帧间 30Hz 子帧动作:有序 (dx,dy,keys)×≤3 + 子帧 mask + 聚合量(和/OR)+ gui/dt
输出 runs/g500_gates/feats/<game>__<seg>.npz + meta.json。

用法:
    python tests/probe_g500_extract.py --data-dir runs/data/g500_h5 \
        --out runs/g500_gates/feats --frames-per-seg 2200 --batch 48
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.dino_tokenizer import DinoTokenizer            # noqa: E402
from train.gaming500.dataset import unpack_keys, N_KEYS  # noqa: E402

# 游戏 → 该游戏最多取几段(鼠标视角类为主;terraria 为 2D 光标对照组)
GAME_QUOTA = {
    "aim-lab": 2, "fortnite": 1, "ghost-of-tsushima": 1,
    "palworld": 1, "alan-wake": 1, "terraria": 1,
}
MAX_SUB = 3          # 子帧槽位上限(15Hz 名义 2 子帧,实际 1~3,padding+mask)


def pick_segments(data_dir):
    """扫描分片,按 GAME_QUOTA 选段(段内帧数优先)。返回 [(path, seg, n_imgs)]。"""
    cand = {}
    for p in sorted(os.listdir(data_dir)):
        if not p.endswith(".h5"):
            continue
        fp = os.path.join(data_dir, p)
        with h5py.File(fp, "r") as f:
            for game in f:
                if game not in GAME_QUOTA:
                    continue
                for name in f[game]:
                    g = f[game][name]
                    if "jpeg" in g:
                        cand.setdefault(game, []).append(
                            (fp, f"{game}/{name}", int(g["jpeg"].shape[0])))
    out = []
    for game, quota in GAME_QUOTA.items():
        segs = sorted(cand.get(game, []), key=lambda s: -s[2])[:quota]
        out.extend(segs)
        if not segs:
            print(f"⚠️  {game}: 本地分片无段,跳过", flush=True)
    return out


def subframe_actions(g, n_img):
    """段内相邻图像帧的 30Hz 子帧有序动作。返回 dict(各 [n_img-1, ...])。"""
    fidx = g["frame_idx"][:n_img].astype(np.int64)
    dx_f, dy_f = g["dx"][:], g["dy"][:]
    keys_f, gui_f = g["keys"][:], g["gui"][:]
    M = dx_f.shape[0]
    T = n_img - 1
    sub_dx = np.zeros((T, MAX_SUB), np.float32)
    sub_dy = np.zeros((T, MAX_SUB), np.float32)
    sub_keys = np.zeros((T, MAX_SUB, N_KEYS), np.uint8)
    sub_mask = np.zeros((T, MAX_SUB), np.uint8)
    agg = {k: np.zeros(T, np.float32) for k in ("dx", "dy")}
    keys_or = np.zeros((T, N_KEYS), np.uint8)
    gui = np.zeros(T, np.uint8)
    dt = np.zeros(T, np.int32)
    for j in range(T):
        a, b = fidx[j], fidx[j + 1]
        lo, hi = min(a + 1, M), min(b + 1, M)
        dt[j] = b - a
        if hi <= lo:
            continue
        seg_dx, seg_dy = dx_f[lo:hi], dy_f[lo:hi]
        seg_keys = unpack_keys(keys_f[lo:hi])
        agg["dx"][j], agg["dy"][j] = seg_dx.sum(), seg_dy.sum()
        keys_or[j] = np.bitwise_or.reduce(seg_keys, axis=0)
        gui[j] = np.uint8(gui_f[lo:hi].any())
        k = min(hi - lo, MAX_SUB)                       # 超过槽位时尾并入最后一槽
        for s in range(k):
            s_lo = lo + s
            s_hi = lo + s + 1 if s < k - 1 else hi
            sub_dx[j, s] = seg_dx[s_lo - lo:s_hi - lo].sum()
            sub_dy[j, s] = seg_dy[s_lo - lo:s_hi - lo].sum()
            sub_keys[j, s] = np.bitwise_or.reduce(seg_keys[s_lo - lo:s_hi - lo], axis=0)
            sub_mask[j, s] = 1
    return dict(sub_dx=sub_dx, sub_dy=sub_dy, sub_keys=sub_keys, sub_mask=sub_mask,
                dx=agg["dx"], dy=agg["dy"], keys=keys_or, gui=gui, dt=dt)


@torch.no_grad()
def encode_segment(tok, g, n_img, img_size, batch, threads, device):
    """JPEG[n_img] → DINO patch 特征 [n_img,384,11,11] fp16(分批,与在跑任务共存)。"""
    blobs = [g["jpeg"][i] for i in range(n_img)]

    def dec(blob):
        bgr = cv2.imdecode(blob, cv2.IMREAD_COLOR)
        bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    feats = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for i0 in range(0, n_img, batch):
            imgs = list(ex.map(dec, blobs[i0:i0 + batch]))
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
            with torch.autocast(device, dtype=torch.bfloat16):
                f = tok.encode(x.to(device))            # [B,384,11,11]
            feats.append(f.to(torch.float16).cpu())
    return torch.cat(feats).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="runs/data/g500_h5")
    ap.add_argument("--out", default="runs/g500_gates/feats")
    ap.add_argument("--frames-per-seg", type=int, default=2200)
    ap.add_argument("--img-size", type=int, default=176)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    tok = DinoTokenizer(kind="dinov3").to(device).eval()
    segs = pick_segments(args.data_dir)
    meta = []
    for fp, seg, n_total in segs:
        n = min(n_total, args.frames_per_seg)
        name = seg.replace("/", "__")
        out_p = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(out_p):
            print(f"✔ 已存在,跳过 {name}", flush=True)
            meta.append(dict(seg=seg, n=n, file=f"{name}.npz"))
            continue
        t0 = time.time()
        with h5py.File(fp, "r") as f:
            g = f[seg]
            acts = subframe_actions(g, n)
            feats = encode_segment(tok, g, n, args.img_size, args.batch,
                                   args.threads, device)
        np.savez_compressed(out_p, feats=feats, **acts)
        meta.append(dict(seg=seg, n=n, file=f"{name}.npz"))
        print(f"✔ {seg}: {n} 帧, {time.time()-t0:.0f}s, "
              f"|dx|p50={np.percentile(np.abs(acts['dx']), 50):.1f} "
              f"p99={np.percentile(np.abs(acts['dx']), 99):.1f}", flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as fo:
        json.dump(meta, fo, ensure_ascii=False, indent=1)
    print(f"完成:{len(meta)} 段 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
