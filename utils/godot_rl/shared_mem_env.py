"""Godot 40 环境训练管线的跨平台共享内存驱动（与 assets/godot_meta_rl/Main.cs 对应）。

对外接口：GodotTrainEnv（连接/握手/收发）、shm_path（后备文件路径工厂）、
set_timer_resolution / reset_timer_resolution（Windows 计时器精度，跨平台空操作）、
以及共享内存布局常量（NUM_ENVS / *_OFFSET / *_DIM / M_* 列索引）。

通用接口（所有模型/环境固定）：动作 = 10 连续 + 30 离散；观测 = 128×128×3 图像 + 元数据。
元数据每环境 5 个 float32：[frameCount, steps, sim_dt(=steps*dt), reward, done]。sim_dt 即
"过了几个物理帧 × 步长"，务必喂给模型。

跨平台共享内存（Win/Linux 自动识别）：
  - 共享内存用【文件后端 mmap】：双方打开同一个文件（shm_path()）；Windows 与 Linux 的
    mmap/.NET MemoryMappedFile 都支持文件后端，无需命名内核对象（Linux 上 .NET 不支持命名 MMF/事件）。
  - 握手用【共享内存内的轮询计数器】(seqlock)，不依赖 Windows 命名事件：
      ObsSeq(Godot→Python)：Godot 发布一帧观测后 +1。
      ActSeq(Python→Godot)：Python 写完动作后置为它消费的 ObsSeq（应答）。
    循环：wait_obs(轮询 ObsSeq≠ActSeq) → read_images/read_meta → 策略算动作 → send_action(写 ActSeq 应答)。
    Godot 收到应答前绝不步进，故 Python 必然读到每一帧（帧号严格 +1，无丢帧）。
"""

import ctypes
import mmap
import os
import struct
import sys
import tempfile
import time

import numpy as np

IS_WINDOWS = sys.platform == "win32"

# 仓库根：utils/godot_rl/ 向上两级。Godot 工程默认在 assets/godot_meta_rl，可用 RL_PROJECT_DIR 覆盖。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GODOT_EXE = os.environ.get("GODOT_EXE", "godot")
PROJECT_DIR = os.environ.get("RL_PROJECT_DIR", os.path.join(_REPO_ROOT, "assets", "godot_meta_rl"))
TRAIN_SCENE = "res://train_main.tscn"

NUM_ENVS = 40
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 3
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS   # 49152
TOTAL_IMAGES_BYTES = NUM_ENVS * IMAGE_BYTES           # 1,966,080

META_PER_ENV = 5                                      # frameCount, steps, sim_dt, reward, done
TOTAL_META_BYTES = NUM_ENVS * META_PER_ENV * 4        # 800

CONT_DIM = 10
DISC_DIM = 30
CONT_BYTES = NUM_ENVS * CONT_DIM * 4                  # 1600
DISC_BYTES = NUM_ENVS * DISC_DIM * 4                  # 4800

META_OFFSET = TOTAL_IMAGES_BYTES
CONT_OFFSET = TOTAL_IMAGES_BYTES + TOTAL_META_BYTES
DISC_OFFSET = CONT_OFFSET + CONT_BYTES
SEQ_OFFSET = DISC_OFFSET + DISC_BYTES                 # 异步模式 seqlock 序号(int32)
OBS_SEQ_OFFSET = SEQ_OFFSET + 4                       # 锁步握手：Godot 发布观测后 +1
ACT_SEQ_OFFSET = OBS_SEQ_OFFSET + 4                   # 锁步握手：Python 写完动作后置为已消费的 ObsSeq
TOTAL_SHM_SIZE = ACT_SEQ_OFFSET + 4

MAP_NAME = "GodotRL_SharedMem"                        # 后备文件名（不再作为命名内核对象）

# 元数据列索引
M_FRAME = 0
M_STEPS = 1
M_SIM_DT = 2
M_REWARD = 3
M_DONE = 4


def shm_path():
    """共享内存后备文件的跨平台路径（Win/Linux 自动识别）。

    优先 RL_SHM_PATH 环境变量；否则放系统临时目录下的固定文件名，保证 Godot 与 Python 指向同一文件。

    Returns
    -------
    str
        后备文件绝对路径。
    """
    p = os.environ.get("RL_SHM_PATH")
    if p:
        return p
    return os.path.join(tempfile.gettempdir(), MAP_NAME + ".bin")


def set_timer_resolution(ms=1):
    """提高 Windows 计时器精度，让 time.sleep(亚毫秒~毫秒) 可靠。返回是否成功；非 Windows 直接 False。

    进程级设置一次即可；主要给基准测试故意放慢消费者时用（让 sleep 精确）；握手本身不依赖它。
    """
    if not IS_WINDOWS:
        return False
    try:
        return ctypes.windll.winmm.timeBeginPeriod(int(ms)) == 0
    except Exception:
        return False


def reset_timer_resolution(ms=1):
    """还原 Windows 计时器精度。非 Windows 空操作。"""
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.winmm.timeEndPeriod(int(ms))
    except Exception:
        pass


class GodotTrainEnv:
    """连接 Main.cs 编排的 40 环境，文件后端共享内存 + 轮询计数器握手收发。"""

    def __init__(self, connect_timeout_s=40.0, poll_sleep_s=0.0002):
        self._poll_sleep = poll_sleep_s
        deadline = time.time() + connect_timeout_s
        self._file, self.shm = self._open_shm(deadline)
        if self.shm is None:
            raise RuntimeError("连接 Godot 失败（共享内存文件未就绪）。")
        # _consumed：Python 侧"已消费到第几帧 ObsSeq"，由 wait_obs 推进（消费一次性，门控测试依赖此语义：
        # 不发动作就不应再返回新帧）。send_action 把它写进 ActSeq 向 Godot 应答。
        # 初值取 ActSeq（已应答进度），保证连接后第一帧待应答观测会被消费，且不重复/不漏。
        self._consumed = self._read_i32(ACT_SEQ_OFFSET)

    def _open_shm(self, deadline):
        """轮询等待 Godot 创建并撑满共享内存文件，然后映射。返回 (file, mmap)。"""
        path = shm_path()
        while time.time() < deadline:
            try:
                if os.path.exists(path) and os.path.getsize(path) >= TOTAL_SHM_SIZE:
                    f = open(path, "r+b")
                    mm = mmap.mmap(f.fileno(), TOTAL_SHM_SIZE, access=mmap.ACCESS_WRITE)
                    return f, mm
            except Exception:
                pass
            time.sleep(0.2)
        return None, None

    def _read_i32(self, offset):
        return struct.unpack_from("<i", self.shm, offset)[0]

    def _write_i32(self, offset, value):
        struct.pack_into("<i", self.shm, offset, value)

    def wait_obs(self, timeout_ms=2000):
        """等一帧【尚未消费】的新观测（消费一次性）。True=拿到；False=超时。

        判据：ObsSeq != _consumed 表示出现了一帧 Python 还没消费的观测。锁步下 Godot 收到应答前不步进，
        ObsSeq 每应答一次才 +1，故不漏帧；不发 send_action 时 ObsSeq 不再增长，wait_obs 必超时（门控语义）。
        """
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            obs = self._read_i32(OBS_SEQ_OFFSET)
            if obs != self._consumed:
                self._consumed = obs
                return True
            if time.perf_counter() >= deadline:
                return False
            if self._poll_sleep:
                time.sleep(self._poll_sleep)

    def warmup(self, timeout_ms=120000, frames=2):
        """吃掉首批渲染帧（软件渲染/Linux 上首帧着色器编译可能数秒~十几秒），避免计入正式测量/训练吞吐。

        逐帧收发零动作；用长超时容忍首帧的一次性着色器编译。返回是否成功收到 frames 帧。
        """
        for _ in range(frames):
            if not self.wait_obs(timeout_ms):
                return False
            self.send_action(np.zeros((NUM_ENVS, CONT_DIM), np.float32),
                             np.zeros((NUM_ENVS, DISC_DIM), np.int32))
        return True

    def read_meta(self):
        """返回 (NUM_ENVS, 5) float32：[frameCount, steps, sim_dt, reward, done]。"""
        raw = self.shm[META_OFFSET:META_OFFSET + TOTAL_META_BYTES]
        return np.frombuffer(raw, dtype=np.float32).reshape(NUM_ENVS, META_PER_ENV)

    def read_images(self):
        """返回 (NUM_ENVS, H, W, 3) uint8。"""
        raw = self.shm[0:TOTAL_IMAGES_BYTES]
        return np.frombuffer(raw, dtype=np.uint8).reshape(NUM_ENVS, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

    def send_action(self, cont, disc):
        """写动作并向 Godot 应答（置 ActSeq=已消费的 ObsSeq）。

        cont:(NUM_ENVS,CONT_DIM) float32, disc:(NUM_ENVS,DISC_DIM) int32。
        """
        c = np.ascontiguousarray(cont, dtype=np.float32).reshape(NUM_ENVS, CONT_DIM)
        d = np.ascontiguousarray(disc, dtype=np.int32).reshape(NUM_ENVS, DISC_DIM)
        self.shm[CONT_OFFSET:CONT_OFFSET + CONT_BYTES] = c.tobytes()
        self.shm[DISC_OFFSET:DISC_OFFSET + DISC_BYTES] = d.tobytes()
        # 应答：把 ActSeq 置为已消费帧的 ObsSeq；Godot 轮询 ActSeq==ObsSeq 才推进，
        # 保证它步进所用动作正是对本帧观测算出的。
        self._write_i32(ACT_SEQ_OFFSET, self._consumed)

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass
