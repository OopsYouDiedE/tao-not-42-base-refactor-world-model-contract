"""端到端自动化测试（同步握手版）：验证 Godot(生产者) -> Python(消费者) 的数据通道。

校验三件事：
  (A) Python 是否真的收到图像（字节数正确 + 像素非全零 + 指纹随时间变化为活数据 + 帧号在涨）；
  (B) 40 个环境的"帧数/步数"方差是否为 0（所有环境步调一致）；
  (C) Python 侧接收帧率 —— 握手下每回合恰一帧新观测，无重复、无丢帧（帧号严格 +1）。
退出码：0 = 全部通过，1 = 有失败。
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

GODOT_LOG = os.path.join(E.PROJECT_DIR, "_godot_test_run.log")

CONNECT_TIMEOUT_S = 40.0
WARMUP_CYCLES = 20
MEASURE_SECONDS = 6.0

_SAMPLE_ENV = 20
_SAMPLE_ROW = E.IMAGE_HEIGHT // 2


def main():
    print("=" * 64)
    print(" Godot <-> Python 同步握手 端到端自动化测试")
    print("=" * 64)

    print(f"[*] 启动 Godot: {E.TRAIN_SCENE}  (日志 -> {GODOT_LOG})")
    log = open(GODOT_LOG, "w", encoding="utf-8", errors="replace")
    proc = launch_godot(log=log)

    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    try:
        try:
            envh = E.GodotTrainEnv(connect_timeout_s=CONNECT_TIMEOUT_S)
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            return 1
        print("[*] 已连接共享内存与握手。预热中（吃掉首帧着色器编译）...")
        # 软件渲染/Linux 首帧含一次性着色器编译；预热吃掉它。
        if not envh.warmup(timeout_ms=120000, frames=2):
            print("[FAIL] 预热未收到渲染帧。")
            return 1

        # 预热：再跑若干回合丢弃（物理未稳定）。
        for _ in range(WARMUP_CYCLES):
            if proc.poll() is not None:
                print("[FAIL] Godot 进程在预热阶段已退出，详见日志。")
                return 1
            if not envh.wait_obs(2000):
                continue
            envh.send_action(zeros_cont, zeros_disc)

        # ---------- 正式测量 ----------
        print(f"[*] 测量 {MEASURE_SECONDS:.0f} 秒（全速同步回合，每回合读整图+元数据）...")
        cycles = 0
        timeouts = 0
        frame_ids = []
        max_frame_var = 0.0
        max_steps_var = 0.0
        first_frame_mean = None
        last_frame_mean = None
        last_steps_mean = None
        image_seen_nonzero = False
        last_imgs = None
        img_fingerprints = set()

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < MEASURE_SECONDS:
            if proc.poll() is not None:
                print("[FAIL] Godot 进程在测量中途退出，详见日志。")
                return 1
            if not envh.wait_obs(2000):
                timeouts += 1
                continue
            meta = envh.read_meta()
            imgs = envh.read_images()
            envh.send_action(zeros_cont, zeros_disc)
            cycles += 1

            frames = meta[:, E.M_FRAME]
            steps = meta[:, E.M_STEPS]
            max_frame_var = max(max_frame_var, float(np.var(frames)))
            max_steps_var = max(max_steps_var, float(np.var(steps)))

            fmean = float(np.mean(frames))
            if first_frame_mean is None:
                first_frame_mean = fmean
            last_frame_mean = fmean
            last_steps_mean = float(np.mean(steps))
            frame_ids.append(int(round(fmean)))

            if not image_seen_nonzero and imgs.any():
                image_seen_nonzero = True
            img_fingerprints.add(hash(imgs[_SAMPLE_ENV, _SAMPLE_ROW].tobytes()))
            last_imgs = imgs

        elapsed = time.perf_counter() - t0
        envh.close()

        recv_fps = cycles / elapsed if elapsed > 0 else 0.0

        ids = np.array(frame_ids)
        gaps = np.diff(ids) if len(ids) >= 2 else np.array([1])
        lossless = bool(np.all(gaps == 1))

        bytes_per_read = last_imgs.size if last_imgs is not None else 0
        img_mean = float(last_imgs.mean())
        img_min = int(last_imgs.min())
        img_max = int(last_imgs.max())
        per_env_mean = last_imgs.reshape(E.NUM_ENVS, -1).mean(axis=1)
        frames_increased = (last_frame_mean is not None and first_frame_mean is not None
                            and last_frame_mean > first_frame_mean)

        test_a = (bytes_per_read == E.TOTAL_IMAGES_BYTES and image_seen_nonzero
                  and frames_increased and len(img_fingerprints) > 1)
        test_b = (max_frame_var == 0.0 and max_steps_var == 0.0)
        test_c = (recv_fps > 0.0 and lossless)

        print()
        print("-" * 64)
        print("结果")
        print("-" * 64)
        print(f"测量时长            : {elapsed:.2f} s")
        print(f"完成同步回合        : {cycles}   (等待超时 {timeouts} 次)")
        print()
        print("[A] Python 是否真的收到图像?")
        print(f"    每回合图像字节     : {bytes_per_read}  (期望 {E.TOTAL_IMAGES_BYTES})")
        print(f"    图像非全零         : {image_seen_nonzero}")
        print(f"    像素 mean/min/max  : {img_mean:.1f} / {img_min} / {img_max}")
        print(f"    各环境像素均值范围 : [{per_env_mean.min():.1f}, {per_env_mean.max():.1f}]")
        print(f"    数据为活(帧号在涨) : {frames_increased}  "
              f"(mean帧号 {first_frame_mean:.0f} -> {last_frame_mean:.0f}, "
              f"末帧步数 {last_steps_mean:.0f})")
        print(f"    图像指纹种类(env20一行): {len(img_fingerprints)}  (>1 表示画面随时间变化)")
        print(f"    => {'PASS' if test_a else 'FAIL'}")
        print()
        print("[B] 40 环境的帧数/步数方差是否为 0?")
        print(f"    期间最大帧号方差    : {max_frame_var}")
        print(f"    期间最大步数方差    : {max_steps_var}")
        print(f"    => {'PASS' if test_b else 'FAIL'}")
        print()
        print("[C] Python 侧接收帧率?")
        print(f"    接收速率           : {recv_fps:.1f} FPS  (= 唯一帧速率，握手下每回合恰一帧)")
        print(f"    帧号连续(+1)无丢帧  : {lossless}  (最大跳 {int(gaps.max())})")
        print(f"    => {'PASS' if test_c else 'FAIL'}")
        print("-" * 64)

        all_pass = test_a and test_b and test_c
        print(f"总判定: {'[ ALL PASS ]' if all_pass else '[ FAIL ]'}")
        return 0 if all_pass else 1

    finally:
        kill_godot(proc)
        log.close()
        print("[*] 已关闭 Godot。")


if __name__ == "__main__":
    sys.exit(main())
