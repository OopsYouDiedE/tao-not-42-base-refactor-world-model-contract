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
)
ALL_ACHIEVEMENTS.sort()

print(f"📊 Minecraft 成就统计：")
print(f"   - 故事线: {len(STORY_ACHIEVEMENTS)}")
print(f"   - 下界: {len(NETHER_ACHIEVEMENTS)}")
print(f"   - 农业: {len(HUSBANDRY_ACHIEVEMENTS)}")
print(f"   - 总数: {len(ALL_ACHIEVEMENTS)}")
