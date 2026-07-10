# -*- coding: utf-8 -*-
"""DINO 瞄准可学性探针:采集(活 CraftGround)+ 评测(冻结 patch 网格 + ridge)。

沿革:原为 §8 "DINO vs YOLOE" 裁决探针;2026-07-10 用户直接拍板 DINO、YOLOE 整线
废弃,双臂对比作废,本探针降级为**单臂验证**:确认 DINO patch 特征对瞄准任务可学
(预期 R² 显著 >0,地形分层 hole/slope 不塌)。若 FAIL 则触发 fovea 双尺度臂(dino_fovea)。

数据清单:runs/probe_aim/manifest.jsonl,每行
  {"img": "<path>", "dx_deg": <float>, "dy_deg": <float>, "terrain": "flat|hole|slope"}
dx/dy = 准星到最近目标(raycast 扫描到的树干)的角偏移,单位 = env 位姿度;
标签由采集侧 raycast + env pose 给(特权信息只进训练侧,不进部署回路)。

采集端顺带做 yaw/pitch 与 CraftGround 的位姿符号标定(fit_angle_map 实测,不写死):
cmd→env 增益、env yaw/pitch 对几何参考角(北=-z,东=+x,down 正)的 (sign, offset),
逐 episode 写入 runs/probe_aim/pose_calib.json。

用法:
  DISPLAY=:1 python tests/probe_dino_aim.py --collect 240      # 采集(Xorg+RAW,与训练同卡)
  python tests/probe_dino_aim.py --arms dino --kind dinov3     # 评测
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ridge_r2(x: np.ndarray, y: np.ndarray, lam: float = 1e-3,
             folds: int = 5, seed: int = 0, groups: np.ndarray | None = None) -> float:
    """K 折交叉验证的 ridge R²。D>N 时走对偶式(Gram N×N),D 十万级也可解。

    groups 非空时按组留出(如按 world seed):同场景样本相关,随机折会泄漏、R² 偏乐观。
    """
    rng = np.random.default_rng(seed)
    if groups is not None:
        uniq = list(dict.fromkeys(groups))
        split = [np.nonzero(groups == g)[0] for g in uniq]
    else:
        idx = rng.permutation(len(x))
        split = [idx[f::folds] for f in range(folds)]
    r2s = []
    for te in split:
        tr = np.setdiff1d(np.arange(len(x)), te)
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


# ────────────────────────────────────────────────────── 采集(活环境,训练侧)

def _geom_angles(eye: np.ndarray, block_xyz) -> tuple[float, float]:
    """眼→方块中心的几何参考角(度)。参考系:北=-z,东=+x,pitch 向下为正。
    env 角与参考角的 (sign, offset) 由 fit_angle_map 实测,不预设 MC 口径。"""
    v = np.array([block_xyz[0] + 0.5, block_xyz[1] + 0.5, block_xyz[2] + 0.5]) - eye
    gy = float(np.degrees(np.arctan2(v[0], -v[2])))
    gp = float(np.degrees(np.arctan2(-v[1], float(np.hypot(v[0], v[2])))))
    return gy, gp


def _terrain(full) -> str:
    """地形分层(heightmap,采集侧标签):hole=近旁有 ≥3 格深落差;slope=起伏 ≥3 格。"""
    hi = list(full.height_info)
    if not hi:
        return "flat"
    near = [h.height for h in hi
            if abs(h.x + 0.5 - full.x) <= 6 and abs(h.z + 0.5 - full.z) <= 6]
    if not near:
        return "flat"
    rel = np.asarray(near, np.float64) - full.y
    if float(rel.min()) <= -3:
        return "hole"
    if float(rel.max() - rel.min()) >= 3:
        return "slope"
    return "flat"


def collect(n_target: int, out_dir: Path, port: int, max_ep: int = 24) -> None:
    """采 (帧, 准星→树干角偏移, 地形) 清单。渲染选型:与训练同卡并行 ⇒ Xorg+RAW。"""
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.initial_environment_config import Difficulty, GameMode, WorldType
    from craftground.proto import observation_space_pb2 as pb
    from craftground.screen_encoding_modes import ScreenEncodingMode

    from net.calibration import fit_angle_map, wrap_deg

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows, calib_log = [], []

    for ep in range(max_ep):
        if len(rows) >= n_target:
            break
        seed = str(int(rng.integers(0, 1 << 30)))
        env_cfg = InitialEnvironmentConfig(
            image_width=640, image_height=360, gamemode=GameMode.SURVIVAL,
            difficulty=Difficulty.PEACEFUL, world_type=WorldType.DEFAULT, seed=seed,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            request_raycast=True, requires_heightmap=True)
        env_cfg.set_allow_mob_spawn(False)
        env_cfg.freeze_time(True)
        env_cfg.freeze_weather(True)
        env = CraftGroundEnvironment(env_cfg,
                                     action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                                     port=port + ep, find_free_port=True, verbose=False)
        try:
            obs, _ = env.reset()
            for _ in range(60):
                obs = env.step(no_op_v2())[0]

            def cam(yaw=0.0, pitch=0.0, wait=1):
                nonlocal obs
                a = no_op_v2()
                a["camera_yaw"], a["camera_pitch"] = float(yaw), float(pitch)
                obs = env.step(a)[0]
                for _ in range(wait):                     # 渲染延迟 1 tick,落定后再读
                    obs = env.step(no_op_v2())[0]
                return obs["full"]

            # ① cmd→env 位姿增益(特权 pose,训练侧):±10° 对称对,净漂移 0
            f0 = obs["full"]
            y0, p0 = float(f0.yaw), float(f0.pitch)
            f1 = cam(yaw=10.0)
            g_yaw = float(wrap_deg(float(f1.yaw) - y0)) / 10.0
            cam(yaw=-10.0)
            f2 = cam(pitch=10.0)
            g_pitch = (float(f2.pitch) - p0) / 10.0
            cam(pitch=-10.0)
            if abs(g_yaw) < 0.5 or abs(g_pitch) < 0.5:    # 测不出,如实登记后跳过
                calib_log.append(dict(ep=ep, seed=seed, fail="pose gain unmeasured",
                                      g_yaw=g_yaw, g_pitch=g_pitch))
                continue

            # ② 树候选:heightmap(48×48 列含顶层 block_name)先验筛掉无树 seed
            full = obs["full"]
            cand = sorted(
                ((h.x, h.z) for h in full.height_info
                 if "leaves" in h.block_name or "log" in h.block_name),
                key=lambda c: (c[0] + 0.5 - full.x) ** 2 + (c[1] + 0.5 - full.z) ** 2)
            if not cand:
                print(f"[ep{ep}] seed={seed} 无树(heightmap),跳过", flush=True)
                calib_log.append(dict(ep=ep, seed=seed, fail="no trees in heightmap"))
                continue

            # ③ 位姿符号标定扫描:双俯仰行 × 各 12 步 yaw,raycast 命中 → 圆周拟合。
            # 单行扫描 env pitch 无变差,平坦地形上拟合病态(符号可翻),故必须两行。
            hits = []
            for dpc in (15.0, 20.0):                      # 累计 +15°、+35° 两行
                cam(pitch=dpc)
                for _ in range(12):
                    full = cam(yaw=30.0)
                    rc = full.raycast_result
                    if rc.type != pb.HitResult.BLOCK:
                        continue
                    eye = np.array([full.x, full.y + 1.62, full.z])
                    b = (rc.target_block.x, rc.target_block.y, rc.target_block.z)
                    if float(np.linalg.norm(np.array(b) + 0.5 - eye)) > 40:
                        continue
                    gy, gp = _geom_angles(eye, b)
                    hits.append((float(full.yaw), float(full.pitch), gy, gp))
            cam(pitch=-35.0)                              # 回平视
            h = np.asarray(hits, np.float64) if hits else np.zeros((0, 4))
            if len(hits) < 12 or float(h[:, 1].std()) < 5.0:   # 命中量/俯仰变差双门
                calib_log.append(dict(ep=ep, seed=seed, fail=f"hits={len(hits)}"))
                continue
            s_yaw, b_yaw, r_yaw = fit_angle_map(h[:, 0], h[:, 2])
            s_pit, b_pit, r_pit = fit_angle_map(h[:, 1], h[:, 3])
            calib_log.append(dict(ep=ep, seed=seed, g_yaw=round(g_yaw, 3),
                                  g_pitch=round(g_pitch, 3),
                                  yaw=dict(sign=s_yaw, offset=round(b_yaw, 2),
                                           resid=round(r_yaw, 2)),
                                  pitch=dict(sign=s_pit, offset=round(b_pit, 2),
                                             resid=round(r_pit, 2)),
                                  n_hits=len(hits), n_trees=len(cand)))
            print(f"[ep{ep}] seed={seed} 标定 yaw(s={s_yaw},b={b_yaw:.1f},r={r_yaw:.1f}) "
                  f"pitch(s={s_pit},b={b_pit:.1f},r={r_pit:.1f}) 树候选={len(cand)}",
                  flush=True)
            if r_yaw > 5.0 or r_pit > 5.0:
                continue                                  # 标定不可信:不产标签

            def env_angles(full, xyz):                    # 几何角 → env 角(实测映射)
                eye = np.array([full.x, full.y + 1.62, full.z])
                gy, gp = _geom_angles(eye, xyz)
                return s_yaw * gy + b_yaw, s_pit * gp + b_pit

            def turn_to(ty, tp=None, iters=3):            # 绝对角控制(cmd 用实测增益)
                nonlocal obs
                full = obs["full"]
                for _ in range(iters):
                    dy = float(wrap_deg(ty - float(full.yaw)))
                    dp = 0.0 if tp is None else float(np.clip(tp - float(full.pitch),
                                                              -60, 60))
                    if abs(dy) < 1.5 and abs(dp) < 1.5:
                        break
                    full = cam(yaw=np.clip(dy, -60, 60) / g_yaw, pitch=dp / g_pitch)
                return full

            # ④ 逐候选:走近 → 细扫锁树干 → 扰动采样
            done_ep = 0
            for cx, cz in cand[:4]:
                full = obs["full"]
                for _ in range(30):                       # 走近(≤30×8 tick)
                    d = float(np.hypot(cx + 0.5 - full.x, cz + 0.5 - full.z))
                    if d < 13:
                        break
                    ty, _tp = env_angles(full, (cx, full.y + 1.62, cz))
                    turn_to(ty, tp=0.0, iters=1)
                    x0, z0 = full.x, full.z
                    a = no_op_v2()
                    a["forward"], a["sprint"] = True, True
                    for _ in range(8):
                        obs = env.step(a)[0]
                    full = obs["full"]
                    if float(np.hypot(full.x - x0, full.z - z0)) < 1.0:
                        a["jump"] = True                  # 卡住:带跳重试
                        for _ in range(8):
                            obs = env.step(a)[0]
                        full = obs["full"]
                else:
                    continue                              # 走不到,换候选
                logs = {}                                 # 细扫:3 档俯仰 × ±50° yaw
                ty0, _ = env_angles(full, (cx, full.y + 1.62, cz))
                for p_abs in (5.0, 12.0, 20.0):           # 几何 down 角 → env 角(实测映射)
                    turn_to(ty0 - 50.0, tp=s_pit * p_abs + b_pit)
                    for _ in range(20):
                        full = cam(yaw=5.0 / g_yaw)
                        rc = full.raycast_result
                        if (rc.type == pb.HitResult.BLOCK
                                and "log" in rc.target_block.translation_key):
                            logs[(rc.target_block.x, rc.target_block.y,
                                  rc.target_block.z)] = True
                if not logs:
                    print(f"[ep{ep}] 候选({cx},{cz}) 细扫无 log 命中", flush=True)
                    continue
                full = obs["full"]
                eye = np.array([full.x, full.y + 1.62, full.z])
                tgt = min(logs, key=lambda b: float(
                    np.linalg.norm(np.array(b) + 0.5 - eye)))
                n_ok = 0
                for _k in range(8):                       # 每树 8 个扰动样本
                    ty, tp = env_angles(obs["full"], tgt)
                    turn_to(ty, tp)                       # 先对准
                    py = rng.uniform(-18, 18)             # 已知扰动(env 度)
                    pp = rng.uniform(-10, 10)
                    full = cam(yaw=py / g_yaw, pitch=pp / g_pitch)
                    ty, tp = env_angles(full, tgt)
                    dx = float(wrap_deg(ty - float(full.yaw)))
                    dy_ = float(tp - float(full.pitch))
                    if abs(dx) > 30 or abs(dy_) > 25:     # 目标太偏(图外),弃样本
                        continue
                    img_p = out_dir / "frames" / f"ep{ep}_{len(rows)}.png"
                    Image.fromarray(np.asarray(obs["rgb"], np.uint8)).save(img_p)
                    rows.append(dict(img=str(img_p), dx_deg=round(dx, 3),
                                     dy_deg=round(dy_, 3), terrain=_terrain(full),
                                     seed=seed, block=str(tgt)))
                    n_ok += 1
                done_ep += n_ok
                print(f"[ep{ep}] 树 {tgt} 采 {n_ok},rows={len(rows)}", flush=True)
                if done_ep >= 24 or len(rows) >= n_target:
                    break
        finally:
            env.close()

    with (out_dir / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "pose_calib.json").write_text(json.dumps(calib_log, indent=2))
    print(f"采集完成:{len(rows)} 样本,{len(calib_log)} 条标定记录 → {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/probe_aim/manifest.jsonl")
    ap.add_argument("--arms", default="dino", help="dino,dino_fovea")
    ap.add_argument("--kind", default="dinov3", help="dinov3(gated)|dinov2(开放)")
    ap.add_argument("--collect", type=int, default=0,
                    help=">0:活环境采集 N 个样本后退出(需 DISPLAY)")
    ap.add_argument("--port", type=int, default=8850)
    args = ap.parse_args()
    if args.collect:
        collect(args.collect, Path(args.manifest).parent, args.port)
        return
    mf = Path(args.manifest)
    if not mf.exists():
        sys.exit(f"缺数据清单 {mf}(采集需活环境,见模块 docstring)")
    rows = [json.loads(ln) for ln in mf.read_text().splitlines() if ln.strip()]
    imgs = [np.asarray(Image.open(r["img"]).convert("RGB")) for r in rows]
    y = np.array([[r["dx_deg"], r["dy_deg"]] for r in rows], np.float32)
    terr = np.array([r.get("terrain", "flat") for r in rows])
    seeds = np.array([str(r.get("seed", i)) for i, r in enumerate(rows)])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {"n": len(rows), "kind": args.kind}
    for arm in args.arms.split(","):
        x = {"dino": lambda: feats_dino(imgs, False, device, args.kind),
             "dino_fovea": lambda: feats_dino(imgs, True, device, args.kind)}[arm]()
        rep = {"all": ridge_r2(x, y)}
        if len(set(seeds)) >= 3:                      # 留 seed:同场景相关样本不跨折泄漏
            rep["all_by_seed"] = ridge_r2(x, y, groups=seeds)
        for t in np.unique(terr):
            m = terr == t
            if m.sum() >= 20:
                rep[str(t)] = ridge_r2(x[m], y[m])
                if len(set(seeds[m])) >= 3:
                    rep[f"{t}_by_seed"] = ridge_r2(x[m], y[m], groups=seeds[m])
        report[arm] = rep
        print(arm, json.dumps(rep, indent=None), flush=True)
    out = Path("runs/probe_aim/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"写入 {out};判据:R² 显著>0 且地形分层不塌,FAIL 则跑 dino_fovea 臂。")


if __name__ == "__main__":
    main()
