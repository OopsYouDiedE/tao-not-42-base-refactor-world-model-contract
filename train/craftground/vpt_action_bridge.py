"""CraftGround 离散动作 → VPT 连续动作契约的桥接(离线→在线知识直通)。

动机:离线 Dreamer4 在 VPT 22 维动作契约(train/minecraft/vpt_action,鼠标 2 + 二值键 20,
末位追加 Δt/DT_NORM 条件)上学到"动作→潜变化"映射。在线若用 27 维 one-hot,
action_proj 只能重新学,动作语义知识全部丢弃。本模块把 27 个离散动作翻译成同一契约,
使离线 checkpoint **整体**(含 action_proj)迁移到在线。

跨域 import 说明:vpt_action 是全仓唯一动作契约(其 docstring 自述),桥接必须消费它;
这是契约依赖而非实现依赖。

映射约定:
  - 移动/交互键 → 对应二值键(sprint 语义含 forward)。
  - 相机:V2 动作的 pitch/yaw 以度计(±15 小幅/±30 大幅,见 train/craftground/env.py),
    归一化 dx=yaw/CAM_DEG_SCALE, dy=pitch/CAM_DEG_SCALE(正 yaw=向右,正 pitch=向下,
    与 VPT 鼠标 dx/dy 符号一致);CAM_DEG_SCALE=30 使大幅转头≈±1(与离线 camera_scale
    取 p95 的标定原则一致:典型幅度落 [-1,1],重尾截断)。
  - Δt 条件:在线 20Hz 单步 Δt=1 帧 ⇒ 末位恒 1/DT_NORM。
"""
import torch

from train.craftground.env import DISCRETE_TO_V2
from train.minecraft.vpt_action import ACTION_DIM, N_MOUSE, VPT_KEYS
from train.minecraft.train_dreamer4 import DT_NORM

CAM_DEG_SCALE = 30.0

# V2 dict 布尔键 → VPT 契约键名(hotbar/drop/inventory 在 27 动作表中不出现)
_KEY_OF = {"forward": "key_w", "back": "key_s", "left": "key_a", "right": "key_d",
           "jump": "key_space", "sneak": "key_sneak", "sprint": "key_sprint",
           "attack": "key_attack", "use": "key_use"}
_IDX = {k: i for i, k in enumerate(VPT_KEYS)}


def build_action_table(device="cpu"):
    """[27, ACTION_DIM+1] 查表:离散动作 id → VPT 契约向量 ⊕ Δt/DT_NORM。"""
    table = torch.zeros(len(DISCRETE_TO_V2), ACTION_DIM + 1)
    for i, v2 in enumerate(DISCRETE_TO_V2):
        for v2_key, vpt_key in _KEY_OF.items():
            if v2.get(v2_key):
                table[i, N_MOUSE + _IDX[vpt_key]] = 1.0
        table[i, 0] = max(-1.0, min(1.0, v2["camera_yaw"] / CAM_DEG_SCALE))    # dx
        table[i, 1] = max(-1.0, min(1.0, v2["camera_pitch"] / CAM_DEG_SCALE))  # dy
        table[i, -1] = 1.0 / DT_NORM
    return table.to(device)
