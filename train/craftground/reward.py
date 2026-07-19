"""从 Craftground 观测构造奖励与成就信号。

Craftground 基类 `step` 的 reward 恒为 0，info 不含成就。本模块从
`obs["full"]`（protobuf ObservationSpaceMessage）的库存/统计量自行构造：

  1. 成就检测：基于库存 translation_key 的子串规则，映射到 ALL_ACHIEVEMENTS。
  2. 稠密内在奖励：每获得一种**新物品类型** +小额奖励（鼓励资源/工具进阶，
     天然指向成就），保证策略梯度始终有非零信号。

成就奖励（每个新成就 +1）由上层 env_interface 统一发放，本模块只负责
"当前满足哪些成就" 的检测与稠密内在奖励，避免重复计分。
"""

from collections import deque
from typing import Dict, List, Set, Tuple

from train.craftground.achievements import ALL_ACHIEVEMENTS

# 成就名 → 索引（与 env_interface 的成就向量列对齐）
ACHIEVEMENT_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(ALL_ACHIEVEMENTS)}

# 基于库存 translation_key 的成就检测规则。
# translation_key 形如 "block.minecraft.oak_log" / "item.minecraft.iron_ingot"。
# 每条规则：成就名 → 判定函数(库存中所有 translation_key 的集合) -> bool。
# 说明：
#   - 部分成就（下界/末地/繁殖/种植）无法仅从库存可靠检测，未列入 → 永不误报，
#     管线正确且可扩展，常见早期成就能正常触发。
#   - story.root 是 Minecraft 根进度（游戏开始即授予），由 step 计数单独处理。
def _has(keys: Set[str], *subs: str) -> bool:
    return any(any(s in k for s in subs) for k in keys)


ACHIEVEMENT_ITEM_RULES = {
    # 主线科技树
    "minecraft.story.mine_wood": lambda keys: _has(keys, "_log"),
    "minecraft.story.punch_tree": lambda keys: _has(keys, "_log"),
    "minecraft.story.mine_stone": lambda keys: _has(keys, "cobblestone"),
    "minecraft.story.obtain_armor": lambda keys: _has(
        keys, "_helmet", "_chestplate", "_leggings", "_boots"
    ),
    "minecraft.story.smelt_iron": lambda keys: _has(keys, "iron_ingot"),
    "minecraft.story.obtain_diamond": lambda keys: _has(keys, "diamond"),
    # 基础采集/合成（课程塑形，难度低、出现频繁）
    "craftground.gather.wool": lambda keys: _has(keys, "_wool"),
    "craftground.gather.flower": lambda keys: _has(
        keys, "dandelion", "poppy", "_tulip", "allium", "cornflower",
        "oxeye_daisy", "lily_of_the_valley", "blue_orchid", "azure_bluet"
    ),
    "craftground.gather.dirt": lambda keys: _has(keys, "block.minecraft.dirt"),
    "craftground.gather.sand": lambda keys: _has(keys, "block.minecraft.sand"),
    "craftground.gather.seeds": lambda keys: _has(keys, "_seeds"),
    "craftground.gather.sapling": lambda keys: _has(keys, "_sapling"),
    "craftground.gather.coal": lambda keys: _has(keys, "item.minecraft.coal", "charcoal"),
    "craftground.gather.dye": lambda keys: _has(keys, "_dye"),
    "craftground.gather.food": lambda keys: _has(
        keys, "apple", "bread", "_beef", "_porkchop", "_chicken", "_mutton",
        "carrot", "potato", "melon_slice", "sweet_berries"
    ),
    "craftground.craft.planks": lambda keys: _has(keys, "_planks"),
    "craftground.craft.stick": lambda keys: _has(keys, "item.minecraft.stick"),
    "craftground.craft.crafting_table": lambda keys: _has(keys, "crafting_table"),
    # 水桶（"item.minecraft.bucket" 只匹配空桶；水/岩浆桶 key 不含该子串）
    "craftground.craft.bucket": lambda keys: _has(keys, "item.minecraft.bucket"),
    "craftground.use.water_bucket": lambda keys: _has(keys, "water_bucket"),
    "craftground.use.lava_bucket": lambda keys: _has(keys, "lava_bucket"),
}

# 深度成就：玩家 y < 阈值时触发（Minecraft 1.21 世界底 y=-64，钻石最密 y≈-59）。
DEPTH_THRESHOLDS = [
    ("craftground.depth.y_below_40", 40),
    ("craftground.depth.y_below_0", 0),
    ("craftground.depth.y_below_neg50", -50),
]

# 距离成就：离出生点水平位移 >= 阈值（格）时触发。
DISTANCE_THRESHOLDS = [
    ("craftground.explore.dist_200", 200.0),
    ("craftground.explore.dist_1000", 1000.0),
]

# ── 死档(idle/退化策略)检测 ──────────────────────────────────────────
# 防止策略塌缩成"原地不动"或"无意义乱钻"。触发 = 小额惩罚 + 强制重开(force_done)。
# 惩罚刻意放轻(默认 0.5)：主要靠"结束烂局"，避免惩罚太狠让角色摆烂/自杀刷重置。
ANTI_IDLE_ENABLED = False
SEA_LEVEL = 45              # Minecraft 1.18+/1.21 海平面 y (调低为 45 避免河道、沙滩等常规地表误判为深渊死档)
IDLE_WINDOW = 1200          # 地表 idle 滑窗步数(20Hz=60秒)
IDLE_MIN_DISP = 10.0        # 窗口内离窗口起点净位移 < 此值(格) → 地表卡死
BELOW_SEA_GRACE = 100       # 无镐处于海平面下连续步数 > 此值(5秒) 才触发(防地形瞬时误判)
DEAD_POLICY_PENALTY = 0.0   # 我们主动重置(idle/无镐深入)的小额惩罚(可设 0 完全靠重置)
DEATH_PENALTY = 0.0         # 游戏内死亡(被怪/摔/岩浆/溺水/饿死)重罚——比主动重置狠
# 地表 idle 重置(规则②)前期关闭:早期随机策略位移本就小,60秒/<10格的 idle 判定
# 会把"在树边站着砍木"这种正确行为也当成卡死、每分钟强制重开,扼杀探索自举。
# 对比实验证实:老 run(无此机制)16.5万步解锁3个成就,开此机制后61.6万步零非root成就。
# 等成就跑起来(如解锁 mine_stone)再考虑重开。死亡-2 与规则①(无镐下海平面)保留。
IDLE_RESET_ENABLED = False

# 每获得一种新物品类型的稠密内在奖励
NEW_ITEM_BONUS = 0.1


def extract_inventory_keys(full_obs) -> Set[str]:
    """从 ObservationSpaceMessage 提取库存中 count>0 的物品 translation_key 集合。"""
    keys: Set[str] = set()
    inventory = getattr(full_obs, "inventory", None)
    if inventory is None:
        return keys
    for item in inventory:
        if getattr(item, "count", 0) > 0:
            tk = getattr(item, "translation_key", "") or ""
            if tk:
                keys.add(tk)
    return keys


def detect_achievements(item_keys: Set[str], episode_step: int) -> Set[str]:
    """根据当前库存与步数，返回**当前满足**的成就名集合。

    Args:
        item_keys: 当前库存所有物品的 translation_key 集合
        episode_step: 当前 episode 步数（用于 root 进度）

    Returns:
        满足条件的成就名集合（含 root）
    """
    satisfied: Set[str] = set()

    # root：游戏开始即授予（步数 >= 1 视为已进入游戏）
    if episode_step >= 1 and "minecraft.story.root" in ACHIEVEMENT_TO_IDX:
        satisfied.add("minecraft.story.root")

    for name, rule in ACHIEVEMENT_ITEM_RULES.items():
        if name in ACHIEVEMENT_TO_IDX and rule(item_keys):
            satisfied.add(name)

    return satisfied


class RewardShaper:
    """单环境的奖励/成就状态机（按 episode 维护）。"""

    def __init__(self, anti_idle: bool = ANTI_IDLE_ENABLED):
        self.anti_idle = anti_idle
        self.seen_item_keys: Set[str] = set()
        self.unlocked: Set[str] = set()  # 已解锁成就名
        self.spawn_xz = None             # 出生点水平坐标 (x, z)，首步记录
        self.pos_window = deque(maxlen=IDLE_WINDOW)  # 近 IDLE_WINDOW 步的 (x,z)
        self.below_sea_no_pick_steps = 0             # 无镐处于海平面下的连续步数

    def reset(self):
        self.seen_item_keys.clear()
        self.unlocked.clear()
        self.spawn_xz = None
        self.pos_window.clear()
        self.below_sea_no_pick_steps = 0

    def _check_termination(self, full_obs) -> float:
        """终止检测：返回应施加的终止惩罚（>0 即强制重开本 episode，0=不重开）。

        - 游戏内死亡（被怪/摔/岩浆/溺水/饿死）：DEATH_PENALTY（重罚，不受 anti_idle 开关影响）
        - 主动重置(死档)，DEAD_POLICY_PENALTY：
            规则①：无镐 + 海平面以下(y<63)持续 BELOW_SEA_GRACE 步
            规则②：地表(y>=63) + 滑窗内净位移 < IDLE_MIN_DISP（卡死）
        """
        # 游戏死亡优先：被 kill 而非我们主动杀 → 重罚
        if getattr(full_obs, "is_dead", False):
            return DEATH_PENALTY

        if not self.anti_idle:
            return 0.0
        x = getattr(full_obs, "x", None)
        y = getattr(full_obs, "y", None)
        z = getattr(full_obs, "z", None)
        if x is None or y is None or z is None:
            return 0.0

        below_sea = y < SEA_LEVEL
        has_pickaxe = _has(self.seen_item_keys, "_pickaxe")

        # 规则①：无镐下到海平面以下（给 grace 步，避免地形瞬时下陷误判）
        if below_sea and not has_pickaxe:
            self.below_sea_no_pick_steps += 1
        else:
            self.below_sea_no_pick_steps = 0
        if self.below_sea_no_pick_steps > BELOW_SEA_GRACE:
            return DEAD_POLICY_PENALTY

        # 规则②：地表卡死（仅在地表才查；地下挖矿不算 idle）——前期关闭，见 IDLE_RESET_ENABLED
        if IDLE_RESET_ENABLED:
            self.pos_window.append((x, z))
            if not below_sea and len(self.pos_window) >= IDLE_WINDOW:
                ox, oz = self.pos_window[0]
                disp = ((x - ox) ** 2 + (z - oz) ** 2) ** 0.5
                if disp < IDLE_MIN_DISP:
                    return DEAD_POLICY_PENALTY

        return 0.0

    def _detect_position_achievements(self, full_obs) -> Set[str]:
        """从玩家坐标检测深度/距离里程碑成就。"""
        out: Set[str] = set()
        x = getattr(full_obs, "x", None)
        y = getattr(full_obs, "y", None)
        z = getattr(full_obs, "z", None)
        if x is None or y is None or z is None:
            return out
        # 深度：y 越低越深
        for name, y_thresh in DEPTH_THRESHOLDS:
            if name in ACHIEVEMENT_TO_IDX and y < y_thresh:
                out.add(name)
        # 距离：离出生点水平距离
        if self.spawn_xz is None:
            self.spawn_xz = (x, z)
        dx = x - self.spawn_xz[0]
        dz = z - self.spawn_xz[1]
        dist = (dx * dx + dz * dz) ** 0.5
        for name, d_thresh in DISTANCE_THRESHOLDS:
            if name in ACHIEVEMENT_TO_IDX and dist >= d_thresh:
                out.add(name)
        return out

    def compute(self, full_obs, episode_step: int) -> Tuple[float, List[int], bool]:
        """计算本步内在奖励、本步新解锁成就索引、是否强制重开(死档)。

        Returns:
            intrinsic_reward: float，稠密内在奖励(含死档小额惩罚)
            new_achievement_indices: list[int]，本步新解锁成就的索引（喂给 successes）
            force_done: bool，True 表示触发死档应重开本 episode
        """
        item_keys = extract_inventory_keys(full_obs)

        # 稠密内在奖励：新出现的物品类型
        new_items = item_keys - self.seen_item_keys
        intrinsic = NEW_ITEM_BONUS * len(new_items)
        self.seen_item_keys |= new_items

        # 成就检测用**累积出现过**的物品(seen_item_keys)而非当前快照：
        # 否则物品被消耗(如原木合成木板)后会漏报对应成就。
        satisfied = detect_achievements(self.seen_item_keys, episode_step)
        # 位置型成就（深度/距离）
        satisfied |= self._detect_position_achievements(full_obs)
        new_unlocked = satisfied - self.unlocked
        self.unlocked |= new_unlocked
        new_indices = [ACHIEVEMENT_TO_IDX[name] for name in new_unlocked]

        # 终止检测：游戏死亡(-2) 或 死档(idle/无镐深入, -0.5) → 惩罚 + 强制重开
        term_penalty = self._check_termination(full_obs)
        force_done = getattr(full_obs, "is_dead", False) and (episode_step > 10)
        intrinsic -= term_penalty

        return intrinsic, new_indices, force_done
