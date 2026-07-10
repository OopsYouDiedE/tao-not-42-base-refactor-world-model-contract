# -*- coding: utf-8 -*-
"""DINO 瞄准可学性探针:冻结 patch 网格 + ridge,回归准星→最近目标角偏移。

沿革:原为 §8 "DINO vs YOLOE" 裁决探针;2026-07-10 用户直接拍板 DINO、YOLOE 整线
废弃,双臂对比作废,本探针降级为**单臂验证**:确认 DINO patch 特征对瞄准任务可学
(预期 R² 显著 >0,地形分层 hole/slope 不塌)。若 FAIL 则触发 fovea 双尺度臂(dino_fovea)。

数据清单(采集需活环境,本探针只管评测):runs/probe_aim/manifest.jsonl,每行
  {"img": "<path>", "dx_deg": <float>, "dy_deg": <float>, "terrain": "flat|hole|slope"}
dx/dy = 准星到最近可交互目标的角偏移(度);标签由采集侧 raycast/GT 给(特权信息只进训练侧)。

用法:python tests/probe_dino_aim.py --manifest runs/probe_aim/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ridge_r2(x: np.ndarray, y: np.ndarray, lam: float = 1e-3,
             folds: int = 5, seed: int = 0) -> float:
    """K 折交叉验证的 ridge R²。D>N 时走对偶式(Gram N×N),D 十万级也可解。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    r2s = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        xm = x[tr].mean(0)
        xt, xv = x[tr] - xm, x[te] - xm
        yc = y[tr] - y[tr].mean(0)
        if xt.shape[1] > len(tr):                    # 对偶:w = Xᵀ(XXᵀ+λnI)⁻¹y
            a = np.linalg.solve(xt @ xt.T + lam * len(tr) * np.eye(len(tr)), yc)
            pred = xv @ (xt.T @ a) + y[tr].mean(0)
        else:
            w = np.linalg.solve(xt.T @ xt + lam * len(tr) * np.eye(xt.shape[1]),
                                xt.T @ yc)
            pred = xv @ w + y[tr].mean(0)
        ss_res = ((y[te] - pred) ** 2).sum(0)
        ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum(0) + 1e-9
        r2s.append(float((1 - ss_res / ss_tot).mean()))
    return float(np.mean(r2s))


_PX_MEAN = (0.485, 0.456, 0.406)
_PX_STD = (0.229, 0.224, 0.225)


def feats_dino(imgs: list, fovea: bool, device: str, kind: str = "dinov3") -> np.ndarray:
    """臂1/1b:冻结 DINO patch 网格,**展平保空间结构**(池化毁方向信息的教训)。

    全图 resize 到 224×384(16 整除);fovea 臂加准星 25% 裁剪(裁剪不缩放,
    规避 2× 放大的纹理尺度漂移教训)。
    """
    from net.backbone import load_backbone  # 懒加载:重依赖只在被选臂时引入
    bb, patch, _dim, n_reg = load_backbone(kind)
    bb = bb.eval().to(device)
    mean = torch.tensor(_PX_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_PX_STD, device=device).view(1, 3, 1, 1)

    def grid(x: torch.Tensor) -> torch.Tensor:      # 入参尺寸须被 patch 整除(调用侧保证)
        assert x.shape[-2] % patch == 0 and x.shape[-1] % patch == 0
        return bb(pixel_values=(x - mean) / std).last_hidden_state[:, 1 + n_reg:]

    out = []
    with torch.no_grad():
        for im in imgs:
            x = torch.as_tensor(np.ascontiguousarray(im), dtype=torch.float32,
                                device=device).permute(2, 0, 1)[None] / 255.0
            fs = [grid(torch.nn.functional.interpolate(x, (224, 384), mode="bilinear"))]
            if fovea:                                # 准星裁剪原分辨率 25% 区域
                h, w = x.shape[-2:]
                crop = x[..., int(h * .375):int(h * .625), int(w * .375):int(w * .625)]
                fs.append(grid(torch.nn.functional.interpolate(
                    crop, (96, 160), mode="nearest")))  # 最近邻:不引入新纹理尺度
            out.append(torch.cat([f.flatten() for f in fs]).float().cpu().numpy())
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/probe_aim/manifest.jsonl")
    ap.add_argument("--arms", default="dino", help="dino,dino_fovea")
    args = ap.parse_args()
    mf = Path(args.manifest)
    if not mf.exists():
        sys.exit(f"缺数据清单 {mf}(采集需活环境,见模块 docstring)")
    rows = [json.loads(ln) for ln in mf.read_text().splitlines() if ln.strip()]
    from PIL import Image
    imgs = [np.asarray(Image.open(r["img"]).convert("RGB")) for r in rows]
    y = np.array([[r["dx_deg"], r["dy_deg"]] for r in rows], np.float32)
    terr = np.array([r.get("terrain", "flat") for r in rows])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {}
    for arm in args.arms.split(","):
        x = {"dino": lambda: feats_dino(imgs, False, device),
             "dino_fovea": lambda: feats_dino(imgs, True, device)}[arm]()
        rep = {"all": ridge_r2(x, y)}
        for t in np.unique(terr):
            m = terr == t
            if m.sum() >= 20:
                rep[str(t)] = ridge_r2(x[m], y[m])
        report[arm] = rep
        print(arm, json.dumps(rep, indent=None), flush=True)
    out = Path("runs/probe_aim/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"写入 {out};判据:R² 显著>0 且地形分层不塌,FAIL 则跑 dino_fovea 臂。")


if __name__ == "__main__":
    main()
