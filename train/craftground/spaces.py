"""Minecraft RL 环境的观测、动作、成就空间定义。

定义了：
  - ObservationSpace: (C=3, H=360, W=640) RGB 图像（Craftground 原生分辨率）
  - ActionSpace: 27 维离散动作（Minecraft 标准）
  - AchievementSpace: 成就向量（Binary）
  - Encoder: YOLO26s Backbone + 多尺度融合

对外接口：
    OBS_SHAPE, NUM_ACTIONS, NUM_ACHIEVEMENTS, ENCODER_CONFIG
"""

# 观测空间（Craftground 原生分辨率：640x360 → 填充到 384x640 以兼容 YOLO）
OBS_SHAPE_NATIVE = (3, 360, 640)  # 原生 Craftground 输出
OBS_SHAPE = (3, 384, 640)  # 填充后的形状（384 = 最小的 ≥360 的 32 倍数）
OBS_DTYPE = "float32"  # [0, 1] 归一化

# 图像编码器配置
ENCODER_CONFIG = {
    "type": "YOLO26s",
    "backbone": "YOLO26s",  # YOLOv8-s（目标检测预训练）
    "fusion": "multi_scale",  # P3, P4, P5 多尺度融合
    "fusion_method": "weighted",  # 学习权重的融合
    "output_dim": 512,  # 编码后的特征维度
    "total_params": "~12M",  # YOLO26s backbone (~11.2M) + fusion head
    "input_resolution": (384, 640),  # (H, W)，兼容 YOLO 的 32 倍数要求
}

# 动作空间（Minecraft Java 版的离散动作）
# 基础动作：前后左右、跳跃、看上看下、潜行、冲刺等
# 工具动作：破坏、放置、交互、合成等
NUM_ACTIONS = 27  # Craftground 标准离散空间维度

ACTION_NAMES = {
    0: "noop",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
    5: "jump",
    6: "sneak",
    7: "sprint",
    8: "look_up",
    9: "look_down",
    10: "attack",  # 破坏方块/攻击实体
    11: "use",     # 放置方块/交互
    12: "hotbar_1",  # 快捷栏选择
    13: "hotbar_2",
    14: "hotbar_3",
    15: "hotbar_4",
    16: "hotbar_5",
    17: "hotbar_6",
    18: "hotbar_7",
    19: "hotbar_8",
    20: "hotbar_9",
    21: "inventory",  # 打开背包
    22: "drop",      # 丢弃物品
    23: "craft",     # 合成
    24: "place_craft",  # 放置工作台
    25: "furnace",   # 炉子
    26: "wait",      # 等待（无操作）
}

# 成就空间
from train.craftground.achievements import ALL_ACHIEVEMENTS

NUM_ACHIEVEMENTS = len(ALL_ACHIEVEMENTS)

# 常见的门控成就组（用于课程学习）
CHECKPOINT_ACHIEVEMENTS = {
    "初期": ["minecraft.story.mine_wood"],
    "石器": ["minecraft.story.mine_stone"],
    "铁器": ["minecraft.story.smelt_iron"],
    "钻石": ["minecraft.story.obtain_diamond"],
    "下界": ["minecraft.story.enter_the_nether"],
    "末影": ["minecraft.story.enter_the_end"],
}
