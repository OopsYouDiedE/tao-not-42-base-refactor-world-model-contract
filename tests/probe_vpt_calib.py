#!/usr/bin/env python3
"""VPT 真实数据上的键位无关离线标定探针(net/calibration.fit_action_flow_map 的实测验收)。

⚠ 实测结论(2026-07-10,负结果,已录 knowledge/README.md §4):
    人类录像上单帧**全局**相位相关不可靠——静态覆盖层(HUD/F3 调试屏/手持地图)把
    全局峰锁死在零位移;快转头帧运动模糊使世界纹理弱于覆盖层。实测增益衰减 ~7×
    (deg/mouse_px 0.021 vs 官方约定 0.15),相机主导帧(|dx|>30)的流比值中位数≈0.0028;
    裁掉 HUD 后仍无改善(Player374 全程 F3 覆盖,比值≈0.0003)。
    若日后重启此路线,需分块中位数流(网格分块各自相位相关取中位数)级别的鲁棒化;
    BC 数据侧的相机单位改按上游数据集格式常量 0.15 deg/px(见 bc_vpt_warmstart.py)。

背景（见 knowledge/README.md §4）：fit_action_flow_map 只在合成数据上
验证过;本探针在真实 BASALT/VPT 承包商录像上跑通,回答三个问题:
  1. 真实录像(运动模糊/行走径向流/动态实体)下 flow 证据的可用率(conf 门控通过率);
  2. 相机通道增益 M[0,0]/M[1,1] 是否跨 clip 一致(中位数±IQR)——一致才可当数据集常量;
  3. 派生 deg/mouse_px:M 给出 screen_px/mouse_px,除以 px_per_deg(=H/FOV_v,FOV 取
     MC 默认 70° 假设,报告中显式标注该假设)→ 与 VPT 官方 run 代码的
     CAMERA_SCALER=360/2400=0.15 deg/px 对照(该常量不在 vendored net/vpt_lib 里,
     出自上游 Video-Pre-Training 仓的 run_inverse_dynamics_model/agent 侧)。
     这是**数据集格式常量的实测核对**,不是往代码里绑物理参数。

产出:runs/probe_vpt_calib/report.json + 终端摘要。BC 暖启动的 camera_scale
(mouse_px ↔ CAM_MAX_DEG 归一)以本探针实测值为准。

用法:
    python -m tests.probe_vpt_calib --data runs/data/vpt_early --clips 6 --frames 1500
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net.calibration import fit_action_flow_map, flow_shift  # noqa: E402
from train.minecraft.vpt_dataset import VPT_KEYS, _pair_list  # noqa: E402

FOV_V_DEG = 70.0          # 报告假设:MC 默认竖直 FOV(承包商默认设置);仅用于换算展示
VPT_CAMERA_SCALER = 0.15  # 官方 VPT 动作空间常量(deg/mouse_px),对照锚


def collect_pairs(mp4, jsonl, n_frames, hw=(90, 160), conf_min=1.2):
    """一段 clip 的 (flow[N,2], action[N,22], conf通过率)。GUI 帧与低置信 flow 剔除。"""
    acts, guis = [], []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            kb = d.get("keyboard", {})
            acts.append([d["mouse"]["dx"], d["mouse"]["dy"]]
                        + [float(kb.get(k, 0)) for k in VPT_KEYS])
            guis.append(bool(d.get("gui", False)))
    cap = cv2.VideoCapture(mp4)
    flows, kept_a, total, passed = [], [], 0, 0
    prev = None
    for t in range(min(n_frames, len(acts) - 1)):
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            # 动作 t-1 → 帧间 (t-1, t) 的流;GUI 态无相机响应,跳过
            if not guis[t - 1] and not guis[t]:
                total += 1
                dx, dy, conf = flow_shift(prev, gray)
                if conf >= conf_min:
                    passed += 1
                    flows.append([dx, dy])
                    kept_a.append(acts[t - 1])
        prev = gray
    cap.release()
    if not flows:
        return None
    return (np.asarray(flows, np.float32), np.asarray(kept_a, np.float32),
            passed / max(total, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/data/vpt_early")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--frames", type=int, default=1500, help="每 clip 取前 N 帧")
    ap.add_argument("--out", default="runs/probe_vpt_calib")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pairs = _pair_list(args.data)[:args.clips]
    if not pairs:
        raise SystemExit(f"{args.data} 无成对 mp4/jsonl")

    px_per_deg_v = 90.0 / FOV_V_DEG
    per_clip = []
    for mp4, jsonl in pairs:
        got = collect_pairs(mp4, jsonl, args.frames)
        if got is None:
            print(f"⤫ {os.path.basename(mp4)}: 无可用 flow 证据")
            continue
        flows, acts, pass_rate = got
        m, r2 = fit_action_flow_map(flows, acts)
        # 约定核对:flow_shift 返回"场景右移为 dx>0";右转(dx_mouse>0)⇒ 场景左移 ⇒ M[0,0]<0
        gain_x = float(m[0, 0])          # screen_px / mouse_px(带符号)
        gain_y = float(m[1, 1])
        deg_per_px = abs(gain_x) / px_per_deg_v
        rec = dict(clip=os.path.basename(mp4), n=len(flows),
                   conf_pass=round(pass_rate, 3), r2=round(r2, 3),
                   gain_x=round(gain_x, 4), gain_y=round(gain_y, 4),
                   deg_per_mouse_px=round(deg_per_px, 4),
                   key_col_max=round(float(np.abs(m[:, 2:]).max()), 4))
        per_clip.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    if not per_clip:
        raise SystemExit("全部 clip 无证据,标定失败")
    dpp = np.array([c["deg_per_mouse_px"] for c in per_clip])
    summary = dict(
        n_clips=len(per_clip),
        deg_per_mouse_px_median=round(float(np.median(dpp)), 4),
        deg_per_mouse_px_iqr=[round(float(np.percentile(dpp, 25)), 4),
                              round(float(np.percentile(dpp, 75)), 4)],
        official_anchor=VPT_CAMERA_SCALER,
        fov_assumption_deg=FOV_V_DEG,
        conf_pass_median=round(float(np.median([c["conf_pass"] for c in per_clip])), 3),
        r2_median=round(float(np.median([c["r2"] for c in per_clip])), 3),
        per_clip=per_clip)
    Path(args.out, "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n== 汇总 ==")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
