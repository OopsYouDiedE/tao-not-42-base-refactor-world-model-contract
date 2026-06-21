"""诊断：跑第一轮，到 10 仿真秒时把 40 个环境的相机画面拼成 5×8 一张图保存。

聚光灯亮起后(0~2s 触发)，理论上 40 个环境里总有一些相机恰好朝着被照亮的物体，画面里应能看到明亮物体。
若一个都看不到 → 渲染/聚光灯/可见性代码有误。本诊断用【零动作】(相机不动)，可见与否纯由"物体是否恰好落在
初始视锥内"决定。输出: montage_10s.png（不会自动删除，留着看）。
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from utils.godot_rl import shared_mem_env as E
from utils.godot_rl.launch import launch_godot, kill_godot

CAPTURE_SIM_T = 10.0       # 累计到 10 仿真秒时截图
GRID_ROWS, GRID_COLS = 5, 8
BRIGHT_THRESH = 200        # 像素(任一通道)亮度超过此值视为"被照亮"
MIN_BRIGHT_PIXELS = 12     # 亮像素超过这么多，判定该环境"看得到物体"
OUT_PNG = os.path.join(E.PROJECT_DIR, "montage_10s.png")


def save_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr, "RGB").save(path)
        return "PIL"
    except ImportError:
        pass
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)
        return "imageio"
    except ImportError:
        return None


def main():
    log_path = os.path.join(E.PROJECT_DIR, "_diag_godot.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = launch_godot(log=log)
    ok = False
    try:
        env = E.GodotTrainEnv(connect_timeout_s=40)
        print(f"已连接 {E.NUM_ENVS} 环境。零动作推进到 {CAPTURE_SIM_T:.0f} 仿真秒后截图 ...")

        zeros_c = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
        zeros_d = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)

        # 软件渲染/Linux 首帧含一次性着色器编译；预热吃掉它。
        assert env.warmup(timeout_ms=120000, frames=2), "预热未收到渲染帧"
        env.read_meta()
        env.send_action(zeros_c, zeros_d)

        sim_t = 0.0
        imgs = None
        while sim_t < CAPTURE_SIM_T:
            if not env.wait_obs(3000):
                break
            imgs = env.read_images()
            meta = env.read_meta()
            sim_t += float(meta[0, E.M_SIM_DT])
            env.send_action(zeros_c, zeros_d)

        env.close()
        if imgs is None:
            print("[FAIL] 没拿到图像。")
            return 1

        # 拼 5×8
        montage = np.zeros((GRID_ROWS * E.IMAGE_HEIGHT, GRID_COLS * E.IMAGE_WIDTH, 3), np.uint8)
        bright_counts = np.zeros(E.NUM_ENVS, np.int64)
        for i in range(E.NUM_ENVS):
            r, c = divmod(i, GRID_COLS)
            montage[r * E.IMAGE_HEIGHT:(r + 1) * E.IMAGE_HEIGHT,
                    c * E.IMAGE_WIDTH:(c + 1) * E.IMAGE_WIDTH] = imgs[i]
            bright_counts[i] = int((imgs[i].max(axis=2) > BRIGHT_THRESH).sum())

        visible = np.where(bright_counts > MIN_BRIGHT_PIXELS)[0]

        # A 验证：各环境画面必须互不相同（隔离没坏），且方向正确（地面灰应在下半部更亮）。
        n_unique = len({imgs[i].tobytes() for i in range(E.NUM_ENVS)})
        top_half = imgs[:, :E.IMAGE_HEIGHT // 2].mean()
        bot_half = imgs[:, E.IMAGE_HEIGHT // 2:].mean()
        print(f"不重复画面数        : {n_unique}/{E.NUM_ENVS}  (应=40，全同则=1 表示隔离坏了)")
        print(f"上/下半亮度         : 上={top_half:.1f} 下={bot_half:.1f}  (地面灰在下→下应更亮；若反则上下翻转)")

        backend = save_png(montage, OUT_PNG)
        print("-" * 56)
        print(f"截图时累计仿真时间 : {sim_t:.2f}s")
        if backend:
            print(f"已保存拼图(5×8)    : {OUT_PNG}  (via {backend})")
        else:
            np.save(OUT_PNG.replace(".png", ".npy"), montage)
            print(f"无 PIL/imageio，改存 : {OUT_PNG.replace('.png', '.npy')}")
        print(f"看得到亮物体的环境 : {len(visible)}/{E.NUM_ENVS}  -> {visible.tolist()}")
        print(f"各环境亮像素数(top) : {sorted(bright_counts.tolist(), reverse=True)[:10]}")
        if len(visible) > 0:
            print("=> [ PASS ] 有环境能看到被照亮的物体，渲染/聚光灯正常。")
            ok = True
            return 0
        else:
            print("=> [ FAIL ] 没有任何环境看到亮物体 —— 渲染/聚光灯/可见性代码可能有误。")
            return 1
    finally:
        kill_godot(proc)
        log.close()
        if ok:
            try:
                os.remove(log_path)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
