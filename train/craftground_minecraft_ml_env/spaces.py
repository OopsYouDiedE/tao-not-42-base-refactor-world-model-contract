"""Minecraft RL 环境的观测、动作、成就空间定义。

定义了：
  - ObservationSpace: (C=3, H=64, W=64) RGB 图像
  - ActionSpace: 27 维离散动作（Minecraft 标准）
  - AchievementSpace: 成就向量（Binary）

对外接口：
    OBS_SHAPE, NUM_ACTIONS, NUM_ACHIEVEMENTS
"""

# 观测空间
OBS_SHAPE = (3, 64, 64)  # RGB 图像，64x64
OBS_DTYPE = "float32"  # [0, 1] 归一化

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
from train.craftground_minecraft_ml_env.achievements import ALL_ACHIEVEMENTS

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

print(f"🎮 Minecraft 空间定义：")
print(f"   - 观测: {OBS_SHAPE}")
print(f"   - 动作: {NUM_ACTIONS} 个离散动作")
print(f"   - 成就: {NUM_ACHIEVEMENTS} 个成就")
