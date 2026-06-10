"""为 Minecraft 世界模型准备样本数据（VPTDataset 兼容格式）。

VPTDataset（utils/vpt_dataset.py）读取成对的 `.mp4` + `.jsonl`:
  - <name>.mp4   : [T,H,W,3] 视频
  - <name>.jsonl : 每帧一行 {"mouse":{"dx","dy"}, "keyboard":{"key_w":0/1,...}}
                   第一行可带 "task": "<自然语言任务>"

真实的 OpenAI VPT/BASALT contractor 数据动作 schema 与本仓库 VPTDataset 的精简
parser 并不一致（需额外转换器），所以默认生成**离线合成样本**:动作与画面强相关
(鼠标→全局平移、key_w→前进放大、key_attack→闪红),保证 train_minecraft.py 端到端
可跑且逆动力学有可学信号。`--source vpt` 仅打印真实数据的获取指引,不直接下载。

用法:
    python utils/download_sample_data.py                    # 默认: 8 段合成样本 -> runs/vpt_sample/
    python utils/download_sample_data.py --clips 16 --frames 120 --out runs/vpt_sample
    python utils/download_sample_data.py --source vpt        # 打印真实 VPT 数据获取指引
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 必须与 utils/vpt_dataset.py 的 KEYS 完全一致(顺序无所谓,名字要对得上)
KEYS = ["key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak",
        "key_sprint", "key_attack", "key_use", "key_drop", "key_inventory"]
KEYS += [f"key_hotbar.{i}" for i in range(1, 10)]

TASKS = ["chop a tree", "dig straight down", "build a dirt tower",
         "find a cave", "collect flowers", "explore the plains"]


def sample_action(rng):
    """采样一帧动作:鼠标(连续) + 键盘(稀疏 0/1)。返回 (dx, dy, kb_dict)。"""
    dx = float(rng.normal(0, 6.0))            # 视角水平转动(像素/帧)
    dy = float(rng.normal(0, 4.0))            # 视角垂直转动
    kb = {k: 0 for k in KEYS}
    # 移动键:有惯性地随机按下
    if rng.random() < 0.6:
        kb["key_w"] = 1                        # 多数时间前进
    if rng.random() < 0.15:
        kb["key_a"] = 1
    if rng.random() < 0.15:
        kb["key_d"] = 1
    if rng.random() < 0.25:
        kb["key_attack"] = 1                   # 挖掘/攻击 -> 画面闪红
    if rng.random() < 0.08:
        kb["key_space"] = 1                    # 跳跃 -> 画面上抬
    if rng.random() < 0.05:
        hot = rng.integers(1, 10)
        kb[f"key_hotbar.{hot}"] = 1
    return dx, dy, kb


def render_frame(state, kb, size):
    """根据世界状态渲染一帧 BGR 图。动作通过 state 影响画面,逆动力学才有信号。"""
    H = W = size
    img = np.full((H, W, 3), 40, np.uint8)               # 深灰背景(天空/地面)
    # 地平线随俯仰(cam_y)上下移动
    horizon = int(np.clip(H * 0.5 + state["cam_y"] * 0.3, 20, H - 20))
    img[horizon:, :] = (60, 90, 60)                       # 草地(BGR)
    # 一个"目标方块",x 位置随视角水平偏移(cam_x)滚动,大小随前进(depth)变化
    bx = int((state["cam_x"] % W))
    sz = int(np.clip(state["depth"], 8, size // 2))
    color = (40, 40, 200) if kb.get("key_attack", 0) else (150, 110, 70)  # 攻击->红
    y0 = horizon - sz
    cv2.rectangle(img, (bx - sz // 2, y0), (bx + sz // 2, horizon), color, -1)
    # 准星
    c = size // 2
    cv2.line(img, (c - 6, c), (c + 6, c), (230, 230, 230), 1)
    cv2.line(img, (c, c - 6), (c, c + 6), (230, 230, 230), 1)
    return img


def make_clip(path_mp4, path_jsonl, frames, size, fps, task, seed):
    rng = np.random.default_rng(seed)
    state = {"cam_x": float(rng.integers(0, size)), "cam_y": 0.0, "depth": size * 0.25}
    vw = cv2.VideoWriter(path_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter 打不开 {path_mp4}(缺 mp4v 编码器?)")
    with open(path_jsonl, "w", encoding="utf-8") as jf:
        for t in range(frames):
            dx, dy, kb = sample_action(rng)
            # 动作更新世界状态(确定性物理 -> 画面变化与动作严格相关)
            state["cam_x"] += dx + (3.0 if kb["key_d"] else 0.0) - (3.0 if kb["key_a"] else 0.0)
            state["cam_y"] = np.clip(state["cam_y"] + dy - (12.0 if kb["key_space"] else 0.0), -40, 40)
            state["depth"] += (2.5 if kb["key_w"] else -1.0)
            state["depth"] = float(np.clip(state["depth"], 8, size * 0.45))
            vw.write(render_frame(state, kb, size))
            rec = {"mouse": {"dx": dx, "dy": dy}, "keyboard": kb}
            if t == 0:
                rec["task"] = task
            jf.write(json.dumps(rec) + "\n")
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/vpt_sample", help="输出目录(runs/ 默认 gitignore)")
    ap.add_argument("--clips", type=int, default=8, help="生成多少段视频")
    ap.add_argument("--frames", type=int, default=120, help="每段帧数(需 >= train 的 seq_len)")
    ap.add_argument("--size", type=int, default=128, help="帧边长(正方形)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["synthetic", "vpt"], default="synthetic",
                    help="synthetic=离线合成(默认); vpt=打印真实数据获取指引")
    args = ap.parse_args()

    if args.source == "vpt":
        print("真实 OpenAI VPT/BASALT contractor 数据:")
        print("  索引: https://openaipublic.blob.core.windows.net/minecraft-rl/index/all_6xx_Jun_29.json")
        print("  每条含成对的 .mp4 与 .jsonl(动作),公开可下。")
        print("  注意:其动作 schema 与本仓库 VPTDataset 的精简 parser 不同,需写一个")
        print("        转换器映射到 {mouse:{dx,dy}, keyboard:{key_*}} 格式后才能直接喂。")
        print("  调试管线请先用默认的 --source synthetic。")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"=== 生成 {args.clips} 段合成 VPT 样本 -> {args.out}/ "
          f"({args.frames} 帧 @ {args.size}px {args.fps}fps) ===")
    for i in range(args.clips):
        name = f"clip_{i:03d}"
        task = TASKS[i % len(TASKS)]
        make_clip(os.path.join(args.out, f"{name}.mp4"),
                  os.path.join(args.out, f"{name}.jsonl"),
                  args.frames, args.size, args.fps, task, seed=args.seed + i)
        print(f"  [{i + 1}/{args.clips}] {name}.mp4 + .jsonl  task='{task}'")
    print(f"完成。下一步:python train/train_minecraft.py --data_dir {args.out}")


if __name__ == "__main__":
    main()
