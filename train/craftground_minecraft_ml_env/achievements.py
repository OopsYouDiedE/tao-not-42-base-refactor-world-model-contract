"""Minecraft 1.21 成就定义与分类。

Minecraft 包含数百个成就，分为几大类：
  - Story（故事/进度）：主线任务
  - Nether（下界）：下界相关
  - Husbandry（农业）：动物和农业
  - Adventure（冒险）：探索和战斗
  - Inherited（其他）：杂项

本模块提供分类和依赖关系。
"""

# Minecraft 1.21 故事线成就（主线进度）
STORY_ACHIEVEMENTS = {
    "minecraft.story.root": "Minecraft",
    "minecraft.story.mine_wood": "Getting Wood",
    "minecraft.story.punch_tree": "Punch Tree",
    "minecraft.story.mine_stone": "Stone Age",
    "minecraft.story.obtain_armor": "Getting an Upgrade",
    "minecraft.story.smelt_iron": "Acquire Hardware",
    "minecraft.story.obtain_diamond": "Diamonds!",
    "minecraft.story.enter_the_nether": "We Need to Go Deeper",
    "minecraft.story.follow_ender_eye": "The End?",
    "minecraft.story.enter_the_end": "The End",
}

# Nether 下界成就
NETHER_ACHIEVEMENTS = {
    "minecraft.nether.return_from_end": "Return to Sender",
    "minecraft.nether.find_fortress": "Spooky Scary Skeleton",
    "minecraft.nether.obtain_ancient_debris": "Hidden in the Depths",
    # ... 更多下界成就
}

# Husbandry 农业成就
HUSBANDRY_ACHIEVEMENTS = {
    "minecraft.husbandry.plant_seed": "Planting Time",
    "minecraft.husbandry.breed_an_animal": "The Parrots and the Bats",
    "minecraft.husbandry.breed_all_animals": "Two by Two",
    # ... 更多农业成就
}

# CraftGround 自定义"基础采集/合成"成就（课程塑形用）。
# 这些都能从库存 translation_key 可靠检测，难度低、出现频繁，
# 给早期探索一个稠密信号（缓解主线科技树过于稀疏导致的"撞不开"）。
# 用 craftground.* 前缀，明示是我们加的塑形目标、非官方 advancement。
GATHERING_ACHIEVEMENTS = {
    "craftground.gather.wool": "Collect Wool",
    "craftground.gather.flower": "Collect Flower",
    "craftground.gather.dirt": "Collect Dirt",
    "craftground.gather.sand": "Collect Sand",
    "craftground.gather.seeds": "Collect Seeds",
    "craftground.gather.sapling": "Collect Sapling",
    "craftground.gather.coal": "Collect Coal",
    "craftground.gather.dye": "Obtain Dye",
    "craftground.gather.food": "Obtain Food",
    "craftground.craft.planks": "Make Planks",
    "craftground.craft.stick": "Make Sticks",
    "craftground.craft.crafting_table": "Make Crafting Table",
    # 水桶相关（库存可检测）
    "craftground.craft.bucket": "Make Bucket",
    "craftground.use.water_bucket": "Fill Water",
    "craftground.use.lava_bucket": "Fill Lava",
}

# 探索成就：按深度(绝对 y 越低越深)与离出生点水平距离，分级里程碑。
# 数据来自 obs["full"] 的 x/y/z（已确认字段存在）。连续进度型，信号稠密平滑，
# 专治主线过稀疏导致的探索撞不开。里程碑式(越过阈值一次)，避免无界刷分。
EXPLORATION_ACHIEVEMENTS = {
    # 深度（Minecraft 1.21：世界底 y=-64，钻石最密 y≈-59）
    "craftground.depth.y_below_40": "Underground (y<40)",
    "craftground.depth.y_below_0": "Deepslate (y<0)",
    "craftground.depth.y_below_neg50": "Diamond Zone (y<-50)",
    # 距离：离出生点水平位移里程碑（200 当前长度可达；1000 需拉长 episode）
    "craftground.explore.dist_200": "Journey 200m",
    "craftground.explore.dist_1000": "Expedition 1000m",
}

# 成就依赖关系（表示解锁顺序约束）
# 例如：要获得 "mine_stone"，需要先有 "mine_wood"
ACHIEVEMENT_DEPENDENCIES = {
    "minecraft.story.mine_stone": ["minecraft.story.mine_wood"],
    "minecraft.story.obtain_armor": ["minecraft.story.mine_stone"],
    "minecraft.story.smelt_iron": ["minecraft.story.obtain_armor"],
    "minecraft.story.obtain_diamond": ["minecraft.story.smelt_iron"],
    "minecraft.story.enter_the_nether": ["minecraft.story.obtain_diamond"],
    "minecraft.story.follow_ender_eye": ["minecraft.story.enter_the_nether"],
    "minecraft.story.enter_the_end": ["minecraft.story.follow_ender_eye"],
}

# 所有成就的统一列表（用于 AD 成就向量）
ALL_ACHIEVEMENTS = list(
    STORY_ACHIEVEMENTS.keys()
    | NETHER_ACHIEVEMENTS.keys()
    | HUSBANDRY_ACHIEVEMENTS.keys()
    | GATHERING_ACHIEVEMENTS.keys()
    | EXPLORATION_ACHIEVEMENTS.keys()
)
ALL_ACHIEVEMENTS.sort()

print(f"📊 Minecraft 成就统计：")
print(f"   - 故事线: {len(STORY_ACHIEVEMENTS)}")
print(f"   - 下界: {len(NETHER_ACHIEVEMENTS)}")
print(f"   - 农业: {len(HUSBANDRY_ACHIEVEMENTS)}")
print(f"   - 基础采集/合成: {len(GATHERING_ACHIEVEMENTS)}")
print(f"   - 探索(深度/距离): {len(EXPLORATION_ACHIEVEMENTS)}")
print(f"   - 总数: {len(ALL_ACHIEVEMENTS)}")
