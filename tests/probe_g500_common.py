"""gaming500 闸门测试公共件:特征缓存加载、时间切分、描述子、动作编码。

被 probe_g500_gate0.py / probe_g500_gate12.py 共用。一次性探针工具,不入 net/train。
特征缓存由 tests/probe_g500_extract.py 生成(feats [N,384,11,11] fp16 + 子帧动作)。
"""
import json
import os

import numpy as np
import torch

HOLDOUT_FRAC = 0.15          # 每段时间尾部 15% 为 holdout(防时间漏洩)
N_KEYS = 20


def load_segments(feat_dir):
    """读全部段缓存。返回 list[dict],每段含 feats/动作数组/game/split 边界。"""
    with open(os.path.join(feat_dir, "meta.json")) as f:
        meta = json.load(f)
    segs = []
    for m in meta:
        d = np.load(os.path.join(feat_dir, m["file"]))
        n = d["feats"].shape[0]
        t_cut = int(n * (1 - HOLDOUT_FRAC))            # 转移 j 属 train ⇔ j+1 帧 < t_cut
        segs.append(dict(
            game=m["seg"].split("/")[0], seg=m["seg"], n=n, t_cut=t_cut,
            feats=d["feats"],                          # [N,384,11,11] fp16
            sub_dx=d["sub_dx"], sub_dy=d["sub_dy"],    # [T,3]
            sub_keys=d["sub_keys"], sub_mask=d["sub_mask"],
            dx=d["dx"], dy=d["dy"], keys=d["keys"], gui=d["gui"], dt=d["dt"]))
    return segs


def game_dx_std(segs):
    """逐游戏 dx/dy 归一化尺度(train 区非零位移 std,clamp 下限防除零)。"""
    acc = {}
    for s in segs:
        v = np.concatenate([s["dx"][:s["t_cut"] - 1], s["dy"][:s["t_cut"] - 1]])
        acc.setdefault(s["game"], []).append(v)
    return {g: max(float(np.std(np.concatenate(vs))), 1.0) for g, vs in acc.items()}


def descriptor(feats_t, feats_t1):
    """Δz 描述子:全局均值 + 行均值 + 列均值(保留平移方向的空间梯度信息)。

    Parameters
    ----------
    feats_t, feats_t1 : ndarray fp16, [B,384,11,11]

    Returns
    -------
    ndarray fp32, [B, 384*(1+11+11)] = [B, 8832]
    """
    dz = feats_t1.astype(np.float32) - feats_t.astype(np.float32)
    g = dz.mean(axis=(2, 3))                           # [B,384]
    rows = dz.mean(axis=3).reshape(dz.shape[0], -1)    # [B,384*11]
    cols = dz.mean(axis=2).reshape(dz.shape[0], -1)    # [B,384*11]
    return np.concatenate([g, rows, cols], axis=1)


def frame_descriptor(feats_t):
    """单帧 z_t 描述子(对照组),布局同 descriptor。"""
    z = feats_t.astype(np.float32)
    g = z.mean(axis=(2, 3))
    rows = z.mean(axis=3).reshape(z.shape[0], -1)
    cols = z.mean(axis=2).reshape(z.shape[0], -1)
    return np.concatenate([g, rows, cols], axis=1)


def pooled_frames(feats):
    """[N,384,11,11] → 4×4 均值池化展平 [N,6144] fp32(动力学的观测降维,PCA 前置)。"""
    x = torch.from_numpy(feats.astype(np.float32))
    p = torch.nn.functional.adaptive_avg_pool2d(x, 4)  # [N,384,4,4]
    return p.reshape(p.shape[0], -1).numpy()


def fit_pca(x_train, k=256, seed=0):
    """PCA(torch.pca_lowrank)。返回 (mean [D], comps [D,k]),投影 y=(x-mean)@comps。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x_train.shape[0], generator=g)[:8000]
    xs = torch.from_numpy(x_train[idx.numpy()])
    mean = xs.mean(0)
    _, _, v = torch.pca_lowrank(xs - mean, q=k, niter=4)
    return mean.numpy(), v.numpy()


def mulaw(x, mu=255.0, scale=4.0):
    """mu-law 压缩(归一化位移的非线性压缩;scale 控制饱和点)。"""
    y = np.clip(x / scale, -1, 1)
    return np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)


def encode_action(seg, j, std, mode):
    """转移 j 的动作向量。mode: none(0 维占位)/agg(24 维)/sub(3×24+3=75 维)。

    agg 每子槽布局:[dx_n, dy_n, mulaw(dx_n), mulaw(dy_n), keys20] = 24 维。
    sub = 3 个有序子槽(不足 padding 0)+ sub_mask 3 维;dt 并入两种模式末尾(1 维)。
    """
    dtv = np.array([seg["dt"][j] / 2.0], np.float32)
    if mode == "none":
        return dtv * 0.0                               # 1 维零占位(保持接口齐整)
    if mode == "agg":
        dxn, dyn = seg["dx"][j] / std, seg["dy"][j] / std
        v = np.concatenate([[dxn, dyn, mulaw(dxn), mulaw(dyn)],
                            seg["keys"][j].astype(np.float32)])
        return np.concatenate([v, dtv]).astype(np.float32)
    slots = []
    for s in range(3):
        dxn = seg["sub_dx"][j, s] / std
        dyn = seg["sub_dy"][j, s] / std
        slots.append(np.concatenate([[dxn, dyn, mulaw(dxn), mulaw(dyn)],
                                     seg["sub_keys"][j, s].astype(np.float32)]))
    v = np.concatenate(slots + [seg["sub_mask"][j].astype(np.float32), dtv])
    return v.astype(np.float32)


ACTION_DIM = {"none": 1, "agg": 25, "sub": 76}
