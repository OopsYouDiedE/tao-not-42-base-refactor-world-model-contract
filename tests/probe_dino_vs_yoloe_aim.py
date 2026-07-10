# -*- coding: utf-8 -*-
"""§8 预登记裁决探针:DINO patch 网格 vs YOLOE 提案 token,回归准星→最近目标角偏移。

预登记(knowledge/design_bitter_lesson_map_integration.md §8,2026-07-10):
  预测:DINO 臂 ≥ YOLOE 臂,且地形样本(hole/slope)上显著更强。
  判据:臂1(DINO) ≥ 臂2(YOLOE) ⇒ YOLOE 退役获得与词表退役同强度实测依据;
       臂1 输在瞄准精度 ⇒ 跑臂1b(fovea 双尺度:全图粗 + 准星裁剪细,裁剪不缩放);
       臂1b 仍输 ⇒ 维持路线 2,§8 作废。
  纪律:探针出结果前路线 2(YOLOE)仍是现行裁决,不得先删 YOLOE 代码。

数据清单(采集需活环境,本探针只管评测):runs/probe_aim/manifest.jsonl,每行
  {"img": "<path>", "dx_deg": <float>, "dy_deg": <float>, "terrain": "flat|hole|slope"}
dx/dy = 准星到最近可交互目标的角偏移(度);标签由采集侧 raycast/GT 给(特权信息只进训练侧)。

用法:python tests/probe_dino_vs_yoloe_aim.py --manifest runs/probe_aim/manifest.jsonl
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


def feats_yoloe(imgs: list, device: str, max_det: int = 64) -> np.ndarray:
    """臂2:YOLOE 类别无关提案 token [geo6 ⊕ e_j(512)],按 conf 降序,pad 0 后展平。"""
    from net.fovea_twotower.yolo_unified import UnifiedYoloe26  # 懒加载
    yu = UnifiedYoloe26(max_det=max_det)
    out = []
    with torch.no_grad():
        for im in imgs:
            boxes, confs, _m = yu.propose(im)
            e = yu.proposal_embed(yu.embed(im), boxes).cpu().numpy() \
                if len(boxes) else np.zeros((0, 512), np.float32)
            buf = np.zeros((max_det, 518), np.float32)
            for j in np.argsort(-confs)[:max_det]:
                x1, y1, x2, y2 = boxes[j]
                geo = [(x1 + x2) / 2 / 640, (y1 + y2) / 2 / 384, (x2 - x1) / 640,
                       (y2 - y1) / 384, confs[j], (x2 - x1) * (y2 - y1) / (640 * 384)]
                buf[j] = np.concatenate([np.asarray(geo, np.float32), e[j]])
            out.append(buf.reshape(-1))
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/probe_aim/manifest.jsonl")
    ap.add_argument("--arms", default="dino,yoloe", help="dino,dino_fovea,yoloe")
    args = ap.parse_args()
    mf = Path(args.manifest)
    if not mf.exists():
        sys.exit(f"缺数据清单 {mf}(采集需活环境,见模块 docstring;探针出结果前"
                 f"路线 2 仍现行)")
    rows = [json.loads(ln) for ln in mf.read_text().splitlines() if ln.strip()]
    from PIL import Image
    imgs = [np.asarray(Image.open(r["img"]).convert("RGB")) for r in rows]
    y = np.array([[r["dx_deg"], r["dy_deg"]] for r in rows], np.float32)
    terr = np.array([r.get("terrain", "flat") for r in rows])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {}
    for arm in args.arms.split(","):
        x = {"dino": lambda: feats_dino(imgs, False, device),
             "dino_fovea": lambda: feats_dino(imgs, True, device),
             "yoloe": lambda: feats_yoloe(imgs, device)}[arm]()
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
    print(f"写入 {out};对照 §8 预登记判据裁决。")


if __name__ == "__main__":
    main()
