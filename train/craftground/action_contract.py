# -*- coding: utf-8 -*-
"""CraftGround V2 动作契约的单一定义:键序 / 相机 mu-law 分箱 / 帧堆叠。

对外接口:
    V2_KEYS, CAM_BINS, CAM_MAX_DEG — 契约常量（与 PixelTowerConfiguration 对齐）
        由训练端断言一致,AGENTS §8:领域常量归 train/)
    bins_to_deg(b)   — mu-law 分箱 → 度(采样端解码)
    deg_to_bins(deg) — 度 → mu-law 分箱(BC 数据端编码;与 bins_to_deg 互逆,单测锚定)
    stack_frames(imgs, s) — 帧堆叠(旧→新,开局首帧填充;采样/更新/BC 三侧逐字节同序)

采样端(grpo_pixel)与 BC 暖启动端(bc_vpt_warmstart)共同消费本模块——
"采样 π = 更新 π = BC 目标编码"的口径一致性以单一定义保证(AGENTS §10 合并触发)。
"""
import numpy as np

# CraftGround V2 的 20 个二值键（与 PixelTowerConfiguration.n_keys=20 一致）
V2_KEYS = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use",
           "drop", "inventory", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4",
           "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9"]
CAM_BINS = 11
CAM_MAX_DEG = 18.0                     # 每 tick 相机增量上限(与 StudentPolicy 同口径)
CAM_MU = 8.0                           # mu-law 压缩系数(与 net/vpt_lib 口径同源)


def bins_to_deg(b: np.ndarray) -> np.ndarray:
    """mu-law 分箱 → 度。bin 中心 [-1,1] 经 mu-law 解压后乘 CAM_MAX_DEG。

    Parameters
    ----------
    b : np.ndarray, int, 任意形状,取值 [0, CAM_BINS-1]

    Returns
    -------
    np.ndarray, float32 同形状,单位:度/tick,范围 [-CAM_MAX_DEG, CAM_MAX_DEG]
    """
    x = (b.astype(np.float32) / (CAM_BINS - 1)) * 2 - 1          # [-1,1]
    v = np.sign(x) * (np.power(1 + CAM_MU, np.abs(x)) - 1) / CAM_MU
    return v * CAM_MAX_DEG


def deg_to_bins(deg: np.ndarray) -> np.ndarray:
    """度 → mu-law 分箱(bins_to_deg 的逆;bin 中心处 encode∘decode 恒等)。

    Parameters
    ----------
    deg : np.ndarray, float, 任意形状,单位:度/tick(超界截到 ±CAM_MAX_DEG)

    Returns
    -------
    np.ndarray, int64 同形状,取值 [0, CAM_BINS-1]
    """
    v = np.clip(deg.astype(np.float32) / CAM_MAX_DEG, -1.0, 1.0)
    x = np.sign(v) * np.log1p(CAM_MU * np.abs(v)) / np.log1p(CAM_MU)   # mu-law 压缩,[-1,1]
    return np.rint((x + 1) / 2 * (CAM_BINS - 1)).astype(np.int64)


def stack_frames(imgs: np.ndarray, s: int) -> np.ndarray:
    """[T,H,W,3] → [T,3s,H,W]:每 tick 取最近 s 帧沿通道拼接(旧→新),开局用首帧填充。

    与采样端 rollout 的 deque 堆叠**逐字节同序**——这是"采样 π = 更新 π"的一部分。
    """
    t_n = len(imgs)
    idx = np.clip(np.arange(t_n)[:, None] + np.arange(-(s - 1), 1)[None, :], 0, None)
    return imgs[idx].transpose(0, 1, 4, 2, 3).reshape(t_n, s * 3, *imgs.shape[1:3])
