#!/usr/bin/env python3
"""CraftGround 渲染路径基准:Xvfb-RAW / Xorg-RAW / Xorg-ZEROCOPY / EGL 无 X 四臂(一次性诊断)。

对外接口:patch_craftground_native()、probe_egl_headless() -> dict、run_bench(args) -> dict、main()。

用法(a/b/c 臂同一动作序列,前置 X server 需已起好;d 臂不需要 X):
    python tests/bench_render_craftground.py --arm xvfb-raw      --display :99
    python tests/bench_render_craftground.py --arm xorg-raw      --display :1
    python tests/bench_render_craftground.py --arm xorg-zerocopy --display :1
    python tests/bench_render_craftground.py --arm egl-probe
加 --e2e 后口径改为"端到端到策略可用张量":每步统一测到 [3,90,160] float32 已在
cuda:0(RAW=env.step+CPU 下采样+H2D;ZEROCOPY=env.step+GPU 侧变换,不落 CPU),
并输出分桶 p50/p95 ms、进程 CPU%、GPU util%、显存增量。
结果追加写 runs/zerocopy_bench/results.jsonl。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch  # craftground 原生库要求先 import torch 防段错误
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("runs/zerocopy_bench")


def patch_craftground_native() -> None:
    """运行时修补 craftground 2.6.15 的两处上游 bug(tests 层 shim,不改 site-packages 文件)。

    1. 打包 bug:`environment/observation_converter.py` 用相对导入
       `from .craftground_native import ...`,但原生 .so 只存在于包根
       `craftground/craftground_native*.so` → 预注册 sys.modules 别名。
    2. 属性名 bug:`initialize_zerocopy` 写 `self.observation_tensor_type`,
       而 `convert` 分派读 `self.internal_type` → 包一层,初始化后回填。
    """
    import craftground.craftground_native as native

    sys.modules["craftground.environment.craftground_native"] = native

    from craftground.environment import observation_converter as oc

    _orig = oc.ObservationConverter.initialize_zerocopy

    def _fixed(self, ipc_handle: bytes):
        _orig(self, ipc_handle)
        self.internal_type = self.observation_tensor_type

    oc.ObservationConverter.initialize_zerocopy = _fixed


def probe_egl_headless() -> dict:
    """EGL device-platform 无 X 探测:枚举 GPU → surfaceless context → FBO 清屏回读。

    只验证驱动层可用性(与 CraftGround 无关);全程 unset DISPLAY。

    Returns
    -------
    dict
        {"arm": "egl-probe", "ok": bool, "renderer": str, "gl_version": str,
         "readback": list[int](期望 [0,255,0,255]), "n_devices": int}
    """
    import ctypes

    os.environ.pop("DISPLAY", None)
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglQueryString.restype = ctypes.c_char_p

    def proc(name, restype, *argtypes):
        p = egl.eglGetProcAddress(name.encode())
        assert p, f"eglGetProcAddress({name}) 为空:libEGL 无 device 扩展"
        return ctypes.CFUNCTYPE(restype, *argtypes)(p)

    dev_t = ctypes.c_void_p
    query_devices = proc("eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int,
                         ctypes.POINTER(dev_t), ctypes.POINTER(ctypes.c_int))
    get_platform_display = proc("eglGetPlatformDisplayEXT", ctypes.c_void_p,
                                ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)
    devs, n = (dev_t * 16)(), ctypes.c_int()
    assert query_devices(16, devs, ctypes.byref(n))
    egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    egl.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
    egl.eglCreateContext.restype = ctypes.c_void_p
    egl.eglCreateContext.argtypes = [ctypes.c_void_p] * 3 + [ctypes.POINTER(ctypes.c_int)]
    egl.eglMakeCurrent.argtypes = [ctypes.c_void_p] * 4
    res = dict(arm="egl-probe", ok=False, renderer="", gl_version="",
               readback=[], n_devices=n.value)
    for i in range(n.value):
        dpy = get_platform_display(0x313F, devs[i], None)  # EGL_PLATFORM_DEVICE_EXT
        ma, mi = ctypes.c_int(), ctypes.c_int()
        if not dpy or not egl.eglInitialize(ctypes.c_void_p(dpy), ma, mi):
            continue
        egl.eglBindAPI(0x30A2)                             # EGL_OPENGL_API
        att = (ctypes.c_int * 11)(0x3033, 1, 0x3040, 8,    # PBUFFER, OPENGL_BIT
                                  0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3038)
        cfgs, m = (ctypes.c_void_p * 4)(), ctypes.c_int()
        egl.eglChooseConfig(ctypes.c_void_p(dpy), att, cfgs, 4, ctypes.byref(m))
        if m.value == 0:
            continue
        ctx = egl.eglCreateContext(ctypes.c_void_p(dpy), cfgs[0], None, None)
        if not ctx or not egl.eglMakeCurrent(ctypes.c_void_p(dpy), None, None,
                                             ctypes.c_void_p(ctx)):
            continue                                       # 需 EGL_KHR_surfaceless_context
        from OpenGL import GL                              # context 已 current
        res["renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        res["gl_version"] = GL.glGetString(GL.GL_VERSION).decode()
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, 64, 64, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, tex, 0)
        GL.glViewport(0, 0, 64, 64)
        GL.glClearColor(0, 1, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        px = GL.glReadPixels(0, 0, 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        res["readback"] = [int(b) for b in px[:4]]
        res["ok"] = ("NVIDIA" in res["renderer"]
                     and res["readback"] == [0, 255, 0, 255])
        break
    res["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return res


def _action(no_op, rng: np.random.Generator, i: int) -> dict:
    """确定性动作序列:前 20 步 no-op,之后随机键鼠(各臂共用同一 seed=0 序列)。

    Parameters
    ----------
    no_op : callable
        craftground no_op_v2 工厂。
    rng : np.random.Generator
        动作随机源(调用方以 seed=0 初始化)。
    i : int
        步序号(帧)。

    Returns
    -------
    dict
        V2 动作字典(bool 键 + camera_pitch/camera_yaw, float, 度)。
    """
    a = no_op()
    if i < 20:
        return a
    a["forward"] = bool(rng.random() < 0.5)
    a["jump"] = bool(rng.random() < 0.1)
    a["attack"] = bool(rng.random() < 0.3)
    a["camera_yaw"] = float(rng.normal(0, 5))
    a["camera_pitch"] = float(rng.normal(0, 2))
    return a


def _proc_tree_rss_mb(*pids: int) -> float:
    """返回若干进程及其全部子孙的 RSS 合计(MiB);进程已退出的忽略。"""
    total, seen = 0, set()
    for pid in pids:
        try:
            procs = [psutil.Process(pid)]
            procs += procs[0].children(recursive=True)
        except psutil.NoSuchProcess:
            continue
        for p in procs:
            if p.pid in seen:
                continue
            seen.add(p.pid)
            try:
                total += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
    return total / 2**20


def _gpu_mem_mb() -> dict:
    """nvidia-smi 逐进程显存(MiB),{pid: MiB};无 GPU 进程时为空 dict。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        pid, mem = line.split(",")
        res[int(pid)] = int(mem)
    return res


class _GpuSampler:
    """后台线程按 interval 秒采样 nvidia-smi,收集 GPU util% 与全卡显存 MiB。"""

    def __init__(self, interval: float = 1.0):
        import threading
        self.interval, self.utils, self.mems = interval, [], []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip()
            if out:
                u, m = out.split(",")
                self.utils.append(int(u))
                self.mems.append(int(m))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def _pctile(xs: list, q: float) -> float:
    """列表分位数(ms 口径,保留 2 位)。"""
    return round(float(np.percentile(np.array(xs) * 1e3, q)), 2)


def run_bench(args) -> dict:
    """跑一臂基准并返回指标 dict(steps/s、reset 秒、RSS/显存 MiB)。

    obs["rgb"]:RAW 为 np.uint8 [H,W,3];ZEROCOPY 为 torch.uint8 cuda [H,W,3]。

    --e2e 口径(端到端到策略可用张量,终点统一为 [3,90,160] float32 @cuda:0):
      RAW      step 桶 = env.step;conv 桶 = PIL 下采样 + float32(CPU,同 grpo_pixel
               `small()`);upload 桶 = torch.from_numpy → H2D → permute + synchronize。
      ZEROCOPY step 桶 = env.step(含 converter 在 GPU 上 clone+flip,系该模式固有);
               conv 桶 = permute+float+F.interpolate 到 90×160(全程 GPU)+ synchronize;
               upload 桶 = 0(帧不落 CPU)。
    分桶各报 p50/p95 ms;另报进程 CPU%(python 与 java 树,区间均值)、
    GPU util%/显存(后台 1s 采样均值)与相对基线的显存增量。
    """
    gpu_mem_base = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip())
    os.environ["DISPLAY"] = args.display
    patch_craftground_native()

    from craftground import CraftGroundEnvironment
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.initial_environment_config import (Difficulty, GameMode,
                                                        InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.screen_encoding_modes import ScreenEncodingMode

    mode = (ScreenEncodingMode.ZEROCOPY if args.arm.endswith("zerocopy")
            else ScreenEncodingMode.RAW)
    cfg = InitialEnvironmentConfig(
        image_width=args.width, image_height=args.height,
        gamemode=GameMode.SURVIVAL, difficulty=Difficulty.PEACEFUL,
        world_type=WorldType.DEFAULT, seed="12345",
        screen_encoding_mode=mode)
    cfg.set_allow_mob_spawn(False)
    cfg.freeze_time(True)
    cfg.freeze_weather(True)
    env = CraftGroundEnvironment(
        cfg, action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        port=args.port, find_free_port=True, verbose=False,
        verbose_gradle=args.verbose_gradle)

    t0 = time.time()
    obs, _ = env.reset()
    t_reset = time.time() - t0
    for _ in range(60):                       # 等 "Loading terrain..."(不计入 sps 窗口)
        obs = env.step(no_op_v2())[0]

    rng = np.random.default_rng(0)
    dev = str(getattr(obs["rgb"], "device", "cpu"))
    java_pid = env.process.pid if env.process else -1
    zerocopy = mode == ScreenEncodingMode.ZEROCOPY

    py_proc = psutil.Process(os.getpid())
    java_procs = []
    try:
        jp = psutil.Process(java_pid)
        java_procs = [jp] + jp.children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    for p in [py_proc] + java_procs:
        p.cpu_percent(None)                    # prime:之后读到的是区间均值

    t_step, t_conv, t_up = [], [], []
    with _GpuSampler() as smp:
        t1 = time.time()
        for i in range(args.steps):
            s0 = time.perf_counter()
            obs = env.step(_action(no_op_v2, rng, i))[0]
            rgb = obs["rgb"]
            s1 = time.perf_counter()
            if not args.e2e:                   # 旧口径:只验 obs 落点
                if zerocopy:
                    assert rgb.is_cuda, f"ZEROCOPY 帧不在 GPU: {rgb.device}"
                else:
                    assert rgb.shape == (args.height, args.width, 3), rgb.shape
                t_step.append(s1 - s0)
                continue
            if zerocopy:                       # GPU 侧变换,帧不落 CPU
                assert rgb.is_cuda
                x = rgb.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(90, 160), mode="bilinear",
                                  align_corners=False)[0]
                torch.cuda.synchronize()
                s2 = time.perf_counter()
                s3 = s2                        # upload 桶 = 0
            else:                              # CPU 下采样(同 grpo_pixel small())+ H2D
                small = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8))
                                   .resize((160, 90)), np.float32)
                s2 = time.perf_counter()
                x = torch.from_numpy(small).to("cuda:0").permute(2, 0, 1)
                torch.cuda.synchronize()
                s3 = time.perf_counter()
            assert x.shape == (3, 90, 160) and x.is_cuda and x.dtype == torch.float32
            t_step.append(s1 - s0)
            t_conv.append(s2 - s1)
            t_up.append(s3 - s2)
        dt = time.time() - t1
    cpu_py = py_proc.cpu_percent(None)
    cpu_java = sum(p.cpu_percent(None) for p in java_procs
                   if p.is_running())

    res = dict(
        arm=args.arm, e2e=bool(args.e2e), display=args.display, steps=args.steps,
        wh=[args.width, args.height],
        sps=round(args.steps / dt, 1), reset_s=round(t_reset, 1),
        rgb_device=dev, rgb_dtype=str(obs["rgb"].dtype),
        rss_mb=round(_proc_tree_rss_mb(os.getpid(), java_pid), 0),
        gpu_mem_mb_by_pid=_gpu_mem_mb(),
        step_ms_p50=_pctile(t_step, 50), step_ms_p95=_pctile(t_step, 95),
        cpu_pct_python=round(cpu_py, 1), cpu_pct_java=round(cpu_java, 1),
        gpu_util_pct=round(float(np.mean(smp.utils)), 1) if smp.utils else None,
        gpu_mem_mb_mean=round(float(np.mean(smp.mems)), 0) if smp.mems else None,
        gpu_mem_mb_delta=round(float(np.mean(smp.mems)) - gpu_mem_base, 0)
        if smp.mems else None,
        ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.e2e:
        res.update(conv_ms_p50=_pctile(t_conv, 50), conv_ms_p95=_pctile(t_conv, 95),
                   upload_ms_p50=_pctile(t_up, 50), upload_ms_p95=_pctile(t_up, 95))
    env.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True,
                    choices=["xvfb-raw", "xorg-raw", "xorg-zerocopy", "egl-probe"])
    ap.add_argument("--display", default=None, help="X display,如 :99 / :1(egl-probe 不需要)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--port", type=int, default=8023)
    ap.add_argument("--e2e", action="store_true",
                    help="端到端口径:测到 [3,90,160] float32 @cuda:0 为止,含分桶延时")
    ap.add_argument("--verbose-gradle", action="store_true")
    args = ap.parse_args()

    if args.arm == "egl-probe":
        res = probe_egl_headless()
    else:
        assert args.display, "a/b/c 臂必须给 --display"
        res = run_bench(args)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.jsonl").open("a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
