"""
40 环境训练管线的 Python 端封装（与 Main.cs 对应）。

通用接口（所有模型/环境固定）：动作 = 10 连续 + 30 离散；观测 = 128×128×3 图像 + 元数据。
元数据每环境 5 个 float32：[frameCount, steps, sim_dt(=steps*dt), reward, done]。
  —— sim_dt 即"过了几个物理帧 × 步长"，务必喂给模型。

握手（事件，自动重置）：ObsReady(Godot→Python) / ActReady(Python→Godot)。
  循环：wait_obs → read_images/read_meta → 策略算动作 → send_action。
  Godot 收到 action 前绝不步进，故 Python 必然读到每一帧（帧号严格 +1，无丢帧）。

计时器精度辅助（仅 Windows）：set_timer_resolution / reset_timer_resolution。
  在基准测试中使用故意放慢消费者时调用，让 time.sleep 精度达到毫秒级。
"""

import ctypes
import mmap
import sys
import time

import numpy as np

try:
    import win32event
except ImportError:
    print("请先安装 pywin32: pip install pywin32")
    sys.exit(2)

# Godot 运行环境与项目路径全局配置
GODOT_EXE = r"C:\Users\iii\Desktop\Godot_v4.6.1-stable_mono_win64\Godot_v4.6.1-stable_mono_win64.exe"
PROJECT_DIR = r"C:\Users\iii\Documents\godot_meta_rl"
TRAIN_SCENE = "res://train_main.tscn"

NUM_ENVS = 40
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 3
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS
TOTAL_IMAGES_BYTES = NUM_ENVS * IMAGE_BYTES          # 1,966,080

META_PER_ENV = 5                                      # frameCount, steps, sim_dt, reward, done
TOTAL_META_BYTES = NUM_ENVS * META_PER_ENV * 4        # 800

CONT_DIM = 10
DISC_DIM = 30
CONT_BYTES = NUM_ENVS * CONT_DIM * 4                  # 1600
DISC_BYTES = NUM_ENVS * DISC_DIM * 4                  # 4800

META_OFFSET = TOTAL_IMAGES_BYTES
CONT_OFFSET = TOTAL_IMAGES_BYTES + TOTAL_META_BYTES
DISC_OFFSET = CONT_OFFSET + CONT_BYTES
SEQ_OFFSET = DISC_OFFSET + DISC_BYTES          # 异步模式 seqlock 序号(int32)
TOTAL_SHM_SIZE = SEQ_OFFSET + 4

MAP_NAME = "GodotRL_SharedMem"
OBS_READY_NAME = "GodotRL_ObsReady"
ACT_READY_NAME = "GodotRL_ActReady"

# 元数据列索引
M_FRAME = 0
M_STEPS = 1
M_SIM_DT = 2
M_REWARD = 3
M_DONE = 4

_SYNCHRONIZE = 0x00100000
_EVENT_MODIFY_STATE = 0x0002
_WAIT_OBJECT_0 = 0x0


def set_timer_resolution(ms=1):
    """提高 Windows 计时器精度，让 time.sleep(亚毫秒~毫秒) 可靠。返回是否成功。

    进程级设置一次即可；主要给基准测试故意放慢消费者时用（让 sleep 精确）；握手本身不依赖它。
    """
    try:
        return ctypes.windll.winmm.timeBeginPeriod(int(ms)) == 0
    except Exception:
        return False


def reset_timer_resolution(ms=1):
    """还原 Windows 计时器精度。"""
    try:
        ctypes.windll.winmm.timeEndPeriod(int(ms))
    except Exception:
        pass


class GodotTrainEnv:
    """连接 Main.cs 编排的 40 环境，握手收发。"""

    def __init__(self, connect_timeout_s=40.0):
        deadline = time.time() + connect_timeout_s
        self.obs_ready = self._open(
            lambda: win32event.OpenEvent(_SYNCHRONIZE, False, OBS_READY_NAME), deadline)
        self.act_ready = self._open(
            lambda: win32event.OpenEvent(_EVENT_MODIFY_STATE, False, ACT_READY_NAME), deadline)
        self.shm = self._open(
            lambda: mmap.mmap(-1, TOTAL_SHM_SIZE, tagname=MAP_NAME, access=mmap.ACCESS_WRITE), deadline)
        if not (self.obs_ready and self.act_ready and self.shm):
            raise RuntimeError("连接 Godot 失败（事件或共享内存未就绪）。")

    @staticmethod
    def _open(factory, deadline):
        while time.time() < deadline:
            try:
                return factory()
            except Exception:
                time.sleep(0.2)
        return None

    def wait_obs(self, timeout_ms=2000):
        """等一帧新观测。True=拿到；False=超时。"""
        return win32event.WaitForSingleObject(self.obs_ready, timeout_ms) == _WAIT_OBJECT_0

    def read_meta(self):
        """返回 (NUM_ENVS, 5) float32：[frameCount, steps, sim_dt, reward, done]。"""
        self.shm.seek(META_OFFSET)
        raw = self.shm.read(TOTAL_META_BYTES)
        return np.frombuffer(raw, dtype=np.float32).reshape(NUM_ENVS, META_PER_ENV)

    def read_images(self):
        """返回 (NUM_ENVS, H, W, 3) uint8。"""
        self.shm.seek(0)
        raw = self.shm.read(TOTAL_IMAGES_BYTES)
        return np.frombuffer(raw, dtype=np.uint8).reshape(NUM_ENVS, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

    def send_action(self, cont, disc):
        """写动作并通知 Godot。cont:(NUM_ENVS,CONT_DIM) float32, disc:(NUM_ENVS,DISC_DIM) int32。"""
        c = np.ascontiguousarray(cont, dtype=np.float32).reshape(NUM_ENVS, CONT_DIM)
        d = np.ascontiguousarray(disc, dtype=np.int32).reshape(NUM_ENVS, DISC_DIM)
        self.shm.seek(CONT_OFFSET)
        self.shm.write(c.tobytes())
        self.shm.seek(DISC_OFFSET)
        self.shm.write(d.tobytes())
        win32event.SetEvent(self.act_ready)

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass
