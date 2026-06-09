"""VPT 动作 ↔ 张量 的统一动作契约(动作条件世界模型用)。

契约:每帧动作 = multi-hot 二值键(VPT_KEYS) ⊕ 连续相机 (dx, dy 归一化) = ACTION_DIM 维。
  - Colab 端:`encode_vpt_jsonl` 从 VPT `.jsonl` 单帧解析(schema 假设见注释,需按真文件校准)。
  - 本机端:`rhythm_to_vpt` 把 rhythm 4 键代理动作映射成同形状,用于离线验证训练有效性。
模型只认 ACTION_DIM 维向量,两端共用 → 本机验证过的模型/编码器在 Colab 原样可用。
"""
import torch

VPT_KEYS = [
    "forward", "back", "left", "right", "jump", "sneak", "sprint",
    "attack", "use", "drop", "inventory",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
]                                       # 20 个二值键
N_KEYS = len(VPT_KEYS)
ACTION_DIM = N_KEYS + 2                  # + 连续相机 (dx, dy)
CAMERA_SCALE = 10.0                      # 相机归一化尺度(px/帧;按真数据分布校准)

_KEY_IDX = {k: i for i, k in enumerate(VPT_KEYS)}
# VPT jsonl 的键名 → 我们的键名(部分;按真文件补全)
_KEYMAP = {
    "key.keyboard.w": "forward", "key.keyboard.s": "back",
    "key.keyboard.a": "left", "key.keyboard.d": "right",
    "key.keyboard.space": "jump", "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint", "key.keyboard.q": "drop",
    "key.keyboard.e": "inventory",
}
_KEYMAP.update({f"key.keyboard.{i}": f"hotbar.{i}" for i in range(1, 10)})


def encode_vpt_jsonl(d):
    """单帧 VPT jsonl dict → [ACTION_DIM] 张量(fp32)。

    假设 schema:d["keys"]=按下的键名列表;d["mouse"]={"buttons":[..],"dx":..,"dy":..}。
    左键(0)→attack,右键(1)→use。相机 dx/dy 归一化并 clamp 到 [-1,1]。真文件若字段名不同,改这里。
    """
    v = torch.zeros(ACTION_DIM, dtype=torch.float32)
    for k in d.get("keys", []):
        name = _KEYMAP.get(k)
        if name is not None:
            v[_KEY_IDX[name]] = 1.0
    mouse = d.get("mouse", {}) or {}
    for b in mouse.get("buttons", []) or []:
        if b == 0:
            v[_KEY_IDX["attack"]] = 1.0
        elif b == 1:
            v[_KEY_IDX["use"]] = 1.0
    v[N_KEYS] = float(max(-1.0, min(1.0, (mouse.get("dx", 0.0) or 0.0) / CAMERA_SCALE)))
    v[N_KEYS + 1] = float(max(-1.0, min(1.0, (mouse.get("dy", 0.0) or 0.0) / CAMERA_SCALE)))
    return v


def rhythm_to_vpt(rhythm_action):
    """rhythm 4-lane 按键 [B, n] → VPT 形状动作 [B, ACTION_DIM]。

    把 n 个 lane 键映射到前 n 个二值键(hotbar 风格),相机维=0。
    仅用于本机代理验证:让代理数据与 VPT 同契约,模型/编码器无需改即可两端通用。
    """
    B, n = rhythm_action.shape
    v = torch.zeros(B, ACTION_DIM, device=rhythm_action.device, dtype=torch.float32)
    v[:, :n] = rhythm_action
    return v
