# -*- coding: utf-8 -*-
"""交互式课程编辑服务(纯 CPU 渲染,常驻 env,文件协议驱动)。

回答用户要求:写一句命令→看执行效果(截图)→再写→直到课程调好→存课程。
渲染走 Xvfb(:99)+ 软件 GL(llvmpipe),0 显存,可与 GPU 训练并行。

协议(文件驱动,跨进程):
  控制:向 <dir>/inbox.jsonl 追加一行 JSON,支持:
    {"cmd": ["tp @p ~ ~ ~", "setblock ..."], "settle": 5}  中途注入命令(add_commands)+沉降+快照
    {"noop": N}                                            走 N 帧 no-op(推进/沉降)
    {"reset": ["gamemode survival @p", ...]}               fast-reset 并附带构造命令(重置课程)
    {"snapshot": true}                                     仅截图(不发命令)
    {"save_course": "name"}                                把累计已注入命令存成课程 courses/name.json
    {"quit": true}
  回执:每处理一条,写 <dir>/latest.png(obs["rgb"]) + <dir>/latest.json(seq/库存/坐标/last)
        并向 <dir>/outbox.jsonl 追加一行(含 seq,供轮询确认已处理)。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. \
    python tests/integration/curriculum_repl.py --dir runs/curriculum_repl --port 8500
"""
import argparse
import json
import os
import time
import shutil

import numpy as np
from PIL import Image


def log(dir_, msg):
    with open(os.path.join(dir_, "server.log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def dump_inventory(full_obs):
    inv = []
    try:
        for it in full_obs.inventory:
            if getattr(it, "count", 0) > 0:
                inv.append({"key": it.translation_key, "count": it.count})
    except Exception as e:  # noqa
        inv = [{"error": repr(e)}]
    return inv


def player_xyz(full_obs):
    try:
        return [round(full_obs.x, 2), round(full_obs.y, 2), round(full_obs.z, 2)]
    except Exception:  # noqa
        return None


def save_png(rgb, path):
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    Image.fromarray(arr).save(path)
    return list(arr.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="runs/curriculum_repl")
    p.add_argument("--port", type=int, default=8500)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    os.makedirs(os.path.join(args.dir, "courses"), exist_ok=True)
    inbox = os.path.join(args.dir, "inbox.jsonl")
    outbox = os.path.join(args.dir, "outbox.jsonl")
    open(inbox, "a").close()
    open(outbox, "a").close()

    log(args.dir, f"boot: DISPLAY={os.environ.get('DISPLAY')} "
                  f"LIBGL_ALWAYS_SOFTWARE={os.environ.get('LIBGL_ALWAYS_SOFTWARE')}")

    # 延迟导入,确保 env 变量在 java 子进程启动前已就位
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    cfg = InitialEnvironmentConfig(
        image_width=args.width, image_height=args.height,
        screen_encoding_mode=ScreenEncodingMode.RAW,  # 软件 GL 无 CUDA,不能 ZEROCOPY
        initial_extra_commands=["gamemode survival @p"],
    )
    log(args.dir, "launching craftground (llvmpipe, RAW)... 首启较慢")
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    log(args.dir, "env.reset() ...")
    obs, _ = env.reset()
    applied = []  # 累计已注入命令(用于存课程)

    def snapshot(last_desc):
        png_shape = save_png(obs["rgb"], os.path.join(args.dir, "latest.png"))
        rec = {"seq": seq, "last": last_desc, "img_shape": png_shape,
               "xyz": player_xyz(obs["full"]), "inventory": dump_inventory(obs["full"])}
        json.dump(rec, open(os.path.join(args.dir, "latest.json"), "w"),
                  ensure_ascii=False, indent=1)
        with open(outbox, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(args.dir, f"seq={seq} done: {last_desc} inv={rec['inventory']}")

    seq = 0
    snapshot("boot")  # seq0 = 初始画面
    cursor = 0
    log(args.dir, "READY (poll inbox)")
    while True:
        try:
            lines = open(inbox).read().splitlines()
        except FileNotFoundError:
            lines = []
        if cursor >= len(lines):
            time.sleep(0.4)
            continue
        line = lines[cursor].strip()
        cursor += 1
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:  # noqa
            log(args.dir, f"bad json: {line!r} {e}")
            continue
        seq += 1
        try:
            if msg.get("quit"):
                snapshot("quit")
                env.close()
                log(args.dir, "quit.")
                return
            if "reset" in msg:
                cmds = msg["reset"] or ["gamemode survival @p"]
                obs, _ = env.reset(options={"fast_reset": True, "extra_commands": cmds})
                applied = list(cmds)
                snapshot(f"reset({len(cmds)} cmds)")
                continue
            if "cmd" in msg:
                cmds = msg["cmd"]
                env.add_commands(list(cmds))
                applied.extend(cmds)
                settle = int(msg.get("settle", 5))
                for _ in range(max(1, settle)):
                    obs, *_ = env.step(noop)
                snapshot(f"cmd(+{len(cmds)}, settle={settle})")
                continue
            if "act" in msg:
                # 发自定义 V2 动作(在 no_op 基础上覆盖给定键),repeat 帧后快照
                a = dict(noop)
                a.update(msg["act"] or {})
                for _ in range(max(1, int(msg.get("repeat", 1)))):
                    obs, *_ = env.step(a)
                snapshot(f"act({msg['act']} x{msg.get('repeat',1)})")
                continue
            if "noop" in msg:
                n = int(msg["noop"])
                for _ in range(max(1, n)):
                    obs, *_ = env.step(noop)
                snapshot(f"noop({n})")
                continue
            if msg.get("snapshot"):
                obs, *_ = env.step(noop)
                snapshot("snapshot")
                continue
            if "save_course" in msg:
                name = str(msg["save_course"])
                path = os.path.join(args.dir, "courses", f"{name}.json")
                json.dump({"name": name, "commands": applied,
                           "note": "fast_reset extra_commands 可复现此课程初始态"},
                          open(path, "w"), ensure_ascii=False, indent=1)
                snapshot(f"save_course({name}: {len(applied)} cmds)")
                continue
            log(args.dir, f"unknown msg: {msg}")
        except Exception as e:  # noqa
            import traceback
            log(args.dir, f"ERROR on seq={seq}: {e}\n{traceback.format_exc()}")
            with open(outbox, "a") as f:
                f.write(json.dumps({"seq": seq, "error": repr(e)}) + "\n")


if __name__ == "__main__":
    main()
