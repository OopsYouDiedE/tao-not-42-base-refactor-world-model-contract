"""环境查看器:把 ProceduralRhythmEnv 渲染成视频 + 帧网格 PNG,直观看环境长啥样。

用法:
    python view_env.py --steps 80 --device cuda            # tracer(单音符)
    python view_env.py --steps 80 --multi --device cuda    # 多音符
产出(默认 env_view/):
    env.mp4           env[0] 随时间的完整回放(可拖动)
    timeline.png      env[0] 沿时间采样的帧网格(看单音符一生:出生→下落→穿盲区→判定线→重生)
    variety.png       同一时刻多个并行环境(看随机:轨道/颜色/速度各不同)
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.rhythm.rhythm_env import ProceduralRhythmEnv


def to_bgr(f):
    """[3,H,W] in [0,1] RGB tensor -> HxWx3 uint8 BGR。"""
    img = (f.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def annotate(bgr, env, step, state=None):
    H, W = bgr.shape[:2]
    cv2.putText(bgr, f"t={step}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    # 盲区边界(env 已把该带涂黑,这里标出来)
    y0, y1 = env.blind_y0, env.blind_y1
    cv2.line(bgr, (0, y0), (W, y0), (0, 140, 255), 1)
    cv2.line(bgr, (0, y1), (W, y1), (0, 140, 255), 1)
    cv2.putText(bgr, "BLIND", (W - 62, (y0 + y1) // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
    cv2.putText(bgr, "HIT", (W - 40, env.hit_y_px - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    if state is not None and state.get("active", 0) > 0.5:
        occ = "OCC" if state.get("occluded", False) else "vis"
        cv2.putText(bgr, f"y={state['y']:.0f} v={state['speed']:.0f} {occ}",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    return bgr


def grid(cells, cols):
    H, W = cells[0].shape[:2]
    blank = np.zeros((H, W, 3), np.uint8)
    rows = (len(cells) + cols - 1) // cols
    cells = cells + [blank] * (rows * cols - len(cells))
    return np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--multi", action="store_true", help="多音符模式(默认 tracer 单音符)")
    ap.add_argument("--out", default="env_view")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = ProceduralRhythmEnv(batch_size=args.batch, device=args.device,
                              tracer_mode=not args.multi)
    env.reset()

    frames, states, variety = [], [], None
    for s in range(args.steps):
        env.step(args.dt)
        img = env.render()                       # [B,3,256,256]
        frames.append(img[0].clone())
        if env.tracer_mode:
            st = env.get_tracer_state()
            states.append({k: (v[0].item() if torch.is_tensor(v) else v) for k, v in st.items()})
        else:
            states.append(None)
        if s == args.steps // 2:
            variety = [to_bgr(img[b]) for b in range(min(args.batch, 6))]

    # --- mp4 回放(env[0]) ---
    H, W = 256, 256
    try:
        vw = cv2.VideoWriter(os.path.join(args.out, "env.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
        for s, f in enumerate(frames):
            vw.write(annotate(to_bgr(f), env, s, states[s]))
        vw.release()
        mp4_ok = True
    except Exception as e:
        mp4_ok = False
        print(f"[mp4 skipped: {e}]")

    # --- timeline.png: 沿时间采样 15 帧 ---
    n = min(15, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)
    cells = [annotate(to_bgr(frames[i]), env, int(i), states[i]) for i in idxs]
    cv2.imwrite(os.path.join(args.out, "timeline.png"), grid(cells, cols=5))

    # --- variety.png: 中间时刻多个并行环境 ---
    if variety is not None:
        for b, c in enumerate(variety):
            cv2.putText(c, f"env#{b}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.out, "variety.png"), grid(variety, cols=3))

    print(f"saved -> {args.out}/timeline.png, variety.png" + (", env.mp4" if mp4_ok else ""))


if __name__ == "__main__":
    main()
