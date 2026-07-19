"""VPT 动作与张量之间的统一动作契约。

契约（全仓库唯一布局，与 vpt_video_dataset 的 N_MOUSE=2 切片严格一致）：
    动作向量 = [相机 dx, 相机 dy(归一化)] ⊕ multi-hot 二值键(VPT_KEYS) = ACTION_DIM 维。
    ⚠️ 鼠标在前(索引 0,1)、键在后(索引 2..21)。历史版本曾是键在前/相机在后,
    与 vpt_video_dataset._action_vec 互相矛盾会把鼠标和前两个键静默对调。
  - Colab 端:`encode_vpt_jsonl` 从 VPT `.jsonl` 单帧解析(schema 假设见注释,需按真文件校准)。
模型只认 ACTION_DIM 维向量,两端共用 → 本机验证过的模型/编码器在 Colab 原样可用。

相机离散分箱(camera_to_bin/bin_to_camera):**只用于逆动力学的监督目标**,模型的
动作输入仍是连续 [dx,dy]。MSE 回归下"恒预测 0"是近似最优的平凡解(dx/dy 分布 =
尖峰在 0 + 重尾大转身);改成 mu-law 分箱分类后,0 只是 CAMERA_BINS 个类之一,
基率解的 CE 等于边缘熵,任何真实信号都能压过它。VPT 原版同样用 mu-law 离散相机。
"""
import math

import torch

VPT_KEYS = [
    "key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak", "key_sprint",
    "key_attack", "key_use", "key_drop", "key_inventory",
    "key_hotbar.1", "key_hotbar.2", "key_hotbar.3", "key_hotbar.4", "key_hotbar.5",
    "key_hotbar.6", "key_hotbar.7", "key_hotbar.8", "key_hotbar.9",
]                                       # 20 个二值键
N_KEYS = len(VPT_KEYS)
N_MOUSE = 2                              # 连续相机 (dx, dy),位于向量头部
ACTION_DIM = N_MOUSE + N_KEYS
CAMERA_SCALE = 10.0                      # 相机归一化尺度(px/帧;按真数据分布校准)
CAMERA_BINS = 11                         # mu-law 分箱数(奇数 ⇒ 中心 bin 恰为 0)
CAMERA_MU = 10.0                         # mu-law 压缩系数:小位移区分辨率高,重尾被压进边缘 bin


def camera_to_bin(x):
    """归一化相机值 [-1,1] → mu-law bin 索引 ∈ [0, CAMERA_BINS-1](long)。

    y = sign(x)·ln(1+μ|x|)/ln(1+μ) ∈ [-1,1],均匀切 CAMERA_BINS 段。
    被 camera_scale 截断在 ±1 的大转身落入边缘 bin——分类目标下截断只是
    "≥最大档"一档,不再像 MSE 那样污染整个回归目标。
    """
    x = torch.as_tensor(x, dtype=torch.float32).clamp(-1.0, 1.0)
    y = torch.sign(x) * torch.log1p(CAMERA_MU * x.abs()) / math.log1p(CAMERA_MU)
    return ((y + 1.0) / 2.0 * (CAMERA_BINS - 1)).round().long()


def bin_to_camera(idx):
    """bin 索引 → 归一化相机值(bin 中心;camera_to_bin 的逆)。可视化/推理解码用。"""
    y = idx.float() / (CAMERA_BINS - 1) * 2.0 - 1.0
    return torch.sign(y) * (torch.expm1(y.abs() * math.log1p(CAMERA_MU))) / CAMERA_MU

_KEY_IDX = {k: i for i, k in enumerate(VPT_KEYS)}
# VPT jsonl 的键名 → 我们的键名(部分;按真文件补全)
_KEYMAP = {
    "key.keyboard.w": "key_w", "key.keyboard.s": "key_s",
    "key.keyboard.a": "key_a", "key.keyboard.d": "key_d",
    "key.keyboard.space": "key_space", "key.keyboard.left.shift": "key_sneak",
    "key.keyboard.left.control": "key_sprint", "key.keyboard.q": "key_drop",
    "key.keyboard.e": "key_inventory",
}
_KEYMAP.update({f"key.keyboard.{i}": f"key_hotbar.{i}" for i in range(1, 10)})


def encode_vpt_jsonl(d):
    """单帧 VPT jsonl dict → [ACTION_DIM] 张量(fp32)。布局:[dx, dy, 20 键]。

    假设 schema:d["keys"]=按下的键名列表;d["mouse"]={"buttons":[..],"dx":..,"dy":..}。
    左键(0)→attack,右键(1)→use。相机 dx/dy 归一化并 clamp 到 [-1,1]。真文件若字段名不同,改这里。
    """
    v = torch.zeros(ACTION_DIM, dtype=torch.float32)
    for k in d.get("keys", []):
        name = _KEYMAP.get(k)
        if name is not None:
            v[N_MOUSE + _KEY_IDX[name]] = 1.0
    mouse = d.get("mouse", {}) or {}
    for b in mouse.get("buttons", []) or []:
        if b == 0:
            v[N_MOUSE + _KEY_IDX["key_attack"]] = 1.0
        elif b == 1:
            v[N_MOUSE + _KEY_IDX["key_use"]] = 1.0
    v[0] = float(max(-1.0, min(1.0, (mouse.get("dx", 0.0) or 0.0) / CAMERA_SCALE)))
    v[1] = float(max(-1.0, min(1.0, (mouse.get("dy", 0.0) or 0.0) / CAMERA_SCALE)))
    return v
