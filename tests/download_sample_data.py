"""为 Minecraft 世界模型准备样本数据(VPTDataset 兼容格式)。

VPTDataset 读取成对 `.mp4` + `.jsonl`(每帧一行 {"mouse":{dx,dy}, "keyboard":{...}})。
本脚本生成**离线合成样本**:动作与画面强相关(鼠标→全局平移、key_w→前进放大、
key_attack→闪红),并带**不可逆状态**(has_item/airborne):持物为画面角落一小块叠层
(像素能量小、却永久改变后续帧),跳跃区分地面跳(有效)与空中跳(无效)。

`--counterfactual` 额外产出"低像素能量 × 高后果"参数化族:同一初始状态分叉出
{可逆: 前进+后退 / 不可逆: 合成 / null: 原地 / 开关背包 / 地面跳 / 空中跳} 多支,
每帧 jsonl 标注 has_item/airborne/branch/reach_state_id(供路径无关与反捷径探针监督)。
`--pixel_energy` 把"持物叠层的像素幅度"与"是否不可逆"解耦,用于检验模型按后果而非像素
分配重要性。`--source vpt` 仅打印真实数据获取指引。
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

KEYS = ["key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak",
        "key_sprint", "key_attack", "key_use", "key_drop", "key_inventory"]
KEYS += [f"key_hotbar.{i}" for i in range(1, 10)]

TASKS = ["chop a tree", "dig straight down", "build a dirt tower",
         "find a cave", "collect flowers", "explore the plains"]

# 反事实分支:名 → (branch_id, 动作脚本生成器名)。分支集合是数据参数,不是写死的用例。
CF_BRANCHES = ["reversible", "craft", "null", "toggle_inv", "ground_jump", "air_jump"]


def _empty_kb():
    return {k: 0 for k in KEYS}


def sample_action(rng):
    """采样一帧动作(默认合成模式)。返回 (dx, dy, kb)。"""
    dx = float(rng.normal(0, 6.0))
    dy = float(rng.normal(0, 4.0))
    kb = _empty_kb()
    if rng.random() < 0.6:
        kb["key_w"] = 1
    if rng.random() < 0.15:
        kb["key_a"] = 1
    if rng.random() < 0.15:
        kb["key_d"] = 1
    if rng.random() < 0.25:
        kb["key_attack"] = 1
    if rng.random() < 0.08:
        kb["key_space"] = 1
    if rng.random() < 0.05:
        kb["key_use"] = 1
        kb["key_inventory"] = 1               # 开背包 + use → 置位 has_item
    return dx, dy, kb


def init_state(rng, size):
    return {"cam_x": float(rng.integers(0, size)), "cam_y": 0.0, "depth": size * 0.25,
            "has_item": 0, "airborne": 0}


def step_state(state, dx, dy, kb, size):
    """确定性物理:可逆相机/depth + 不可逆 has_item;airborne 区分地面跳/空中跳。"""
    state["cam_x"] += dx + (3.0 if kb["key_d"] else 0.0) - (3.0 if kb["key_a"] else 0.0)
    state["cam_y"] = float(np.clip(state["cam_y"] + dy - (12.0 if kb["key_space"] else 0.0), -40, 40))
    state["depth"] = float(np.clip(state["depth"] + (2.5 if kb["key_w"] else -1.0), 8, size * 0.45))
    # 不可逆:开背包 + use 永久置位 has_item(此后走动/相机都改变不了)
    if kb.get("key_use") and kb.get("key_inventory"):
        state["has_item"] = 1
    # airborne:地面按 space 才起跳(有效);已在空中再按 space 无效(空中跳=no-op)
    if kb.get("key_space") and state["airborne"] == 0:
        state["airborne"] = 3
    elif state["airborne"] > 0:
        state["airborne"] -= 1
    return state


def render_frame(state, kb, size, pixel_energy=1.0):
    """据世界状态渲染一帧 BGR 图。has_item 叠角落小块(像素幅度 ∝ pixel_energy,后果与之解耦)。"""
    H = W = size
    img = np.full((H, W, 3), 40, np.uint8)
    horizon = int(np.clip(H * 0.5 + state["cam_y"] * 0.3 - (6.0 if state["airborne"] else 0.0), 20, H - 20))
    img[horizon:, :] = (60, 90, 60)
    bx = int((state["cam_x"] % W))
    sz = int(np.clip(state["depth"], 8, size // 2))
    color = (40, 40, 200) if kb.get("key_attack", 0) else (150, 110, 70)
    cv2.rectangle(img, (bx - sz // 2, horizon - sz), (bx + sz // 2, horizon), color, -1)
    # 持物叠层:右下角小色块。边长随 pixel_energy 缩放(默认很小 ⇒ 低像素能量、高后果)
    if state["has_item"]:
        s = max(2, int(round(6 * pixel_energy)))
        cv2.rectangle(img, (W - s - 2, H - s - 2), (W - 2, H - 2), (60, 200, 230), -1)
    c = size // 2
    cv2.line(img, (c - 6, c), (c + 6, c), (230, 230, 230), 1)
    cv2.line(img, (c, c - 6), (c, c + 6), (230, 230, 230), 1)
    return img


def _write_clip(path_mp4, path_jsonl, frames, records, size, fps):
    vw = cv2.VideoWriter(path_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter 打不开 {path_mp4}(缺 mp4v 编码器?)")
    for f in frames:
        vw.write(f)
    vw.release()
    with open(path_jsonl, "w", encoding="utf-8") as jf:
        for rec in records:
            jf.write(json.dumps(rec) + "\n")


def make_clip(path_mp4, path_jsonl, frames_n, size, fps, task, seed, pixel_energy=1.0):
    """默认合成 clip(动作随机 + 不可逆状态)。"""
    rng = np.random.default_rng(seed)
    state = init_state(rng, size)
    frames, records = [], []
    for t in range(frames_n):
        dx, dy, kb = sample_action(rng)
        state = step_state(state, dx, dy, kb, size)
        frames.append(render_frame(state, kb, size, pixel_energy))
        rec = {"mouse": {"dx": dx, "dy": dy}, "keyboard": kb,
               "has_item": int(state["has_item"]), "airborne": int(bool(state["airborne"]))}
        if t == 0:
            rec["task"] = task
        records.append(rec)
    _write_clip(path_mp4, path_jsonl, frames, records, size, fps)


def _branch_action(branch, t, frames_n, rng):
    """反事实分支的逐帧动作脚本。返回 (dx, dy, kb)。"""
    kb = _empty_kb()
    half = frames_n // 2
    if branch == "reversible":            # 前进一半再后退一半(净位移≈0,可逆)
        kb["key_w" if t < half else "key_s"] = 1
        dx = 5.0 if t < half else -5.0
        return dx, 0.0, kb
    if branch == "craft":                 # 中途开背包 + use → 永久 has_item
        if t == half:
            kb["key_use"] = 1
            kb["key_inventory"] = 1
        return 0.0, 0.0, kb
    if branch == "null":                  # 原地不动
        return 0.0, 0.0, kb
    if branch == "toggle_inv":            # 开/关背包但不 use(无不可逆后果)
        if t in (half, half + 1):
            kb["key_inventory"] = 1
        return 0.0, 0.0, kb
    if branch == "ground_jump":           # 地面起跳(有效)
        if t == half:
            kb["key_space"] = 1
        return 0.0, 0.0, kb
    if branch == "air_jump":              # 先跳起,空中再连按 space(后续无效)
        if t in (half, half + 2, half + 3):
            kb["key_space"] = 1
        return 0.0, 0.0, kb
    return 0.0, 0.0, kb


def make_counterfactual_set(out, scenario, frames_n, size, fps, seed, pixel_energy):
    """同一初始状态分叉出 CF_BRANCHES 多支;reach_state_id = 末态 has_item(0/1)。"""
    for branch in CF_BRANCHES:
        rng = np.random.default_rng(seed)
        state = init_state(rng, size)
        frames, records = [], []
        bid = CF_BRANCHES.index(branch)
        for t in range(frames_n):
            dx, dy, kb = _branch_action(branch, t, frames_n, rng)
            state = step_state(state, dx, dy, kb, size)
            frames.append(render_frame(state, kb, size, pixel_energy))
            rec = {"mouse": {"dx": dx, "dy": dy}, "keyboard": kb,
                   "has_item": int(state["has_item"]), "airborne": int(bool(state["airborne"])),
                   "branch": bid}
            if t == 0:
                rec["task"] = f"counterfactual:{branch}"
            records.append(rec)
        reach = int(state["has_item"])         # 同末态(同 reach_state_id)的支构成路径无关对
        for rec in records:
            rec["reach_state_id"] = reach
        name = f"cf_{scenario:03d}_{branch}"
        _write_clip(os.path.join(out, f"{name}.mp4"), os.path.join(out, f"{name}.jsonl"),
                    frames, records, size, fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/vpt_sample")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--counterfactual", action="store_true",
                    help="生成低像素×高后果反事实参数族(分支组 + 不可逆 GT + 双路径对)")
    ap.add_argument("--pixel_energy", type=float, default=1.0,
                    help="持物叠层像素幅度(与是否不可逆解耦;调小 → 小像素高后果)")
    ap.add_argument("--source", choices=["synthetic", "vpt"], default="synthetic")
    args = ap.parse_args()

    if args.source == "vpt":
        print("真实 OpenAI VPT/BASALT contractor 数据:")
        print("  索引: https://openaipublic.blob.core.windows.net/minecraft-rl/index/all_6xx_Jun_29.json")
        print("  每条含成对 .mp4 与 .jsonl;schema 与本仓库精简 parser 不同,需写转换器。")
        print("  调试管线请先用默认 --source synthetic。")
        return

    os.makedirs(args.out, exist_ok=True)
    if args.counterfactual:
        print(f"=== 生成 {args.clips} 组反事实族(每组 {len(CF_BRANCHES)} 支)-> {args.out}/ "
              f"({args.frames} 帧 @ {args.size}px, pixel_energy={args.pixel_energy}) ===")
        for i in range(args.clips):
            make_counterfactual_set(args.out, i, args.frames, args.size, args.fps,
                                    seed=args.seed + i, pixel_energy=args.pixel_energy)
            print(f"  [{i + 1}/{args.clips}] cf_{i:03d}_* ({len(CF_BRANCHES)} 支)")
        print(f"完成。{args.out}/ 可由 train/minecraft/vpt_dataset.VPTStreamDataset 读取"
          f"(训练入口待新基座落地后补)")
        return

    print(f"=== 生成 {args.clips} 段合成 VPT 样本 -> {args.out}/ "
          f"({args.frames} 帧 @ {args.size}px {args.fps}fps) ===")
    for i in range(args.clips):
        name = f"clip_{i:03d}"
        task = TASKS[i % len(TASKS)]
        make_clip(os.path.join(args.out, f"{name}.mp4"), os.path.join(args.out, f"{name}.jsonl"),
                  args.frames, args.size, args.fps, task, seed=args.seed + i,
                  pixel_energy=args.pixel_energy)
        print(f"  [{i + 1}/{args.clips}] {name}.mp4 + .jsonl  task='{task}'")
    print(f"完成。{args.out}/ 可由 train/minecraft/vpt_dataset.VPTStreamDataset 读取"
          f"(训练入口待新基座落地后补)")


if __name__ == "__main__":
    main()
