"""观察驱动的多起点课程银行。

本模块只描述可验证的课程抽样合同，不负责伪造世界状态或替代真实执行。
每一个可进入执行队列的快照都必须来自一次真实观测或一个成功轨迹的稳定末态。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable


DIMENSIONS = (
    "stage",
    "inventory_loadout",
    "biome_terrain",
    "hazard",
    "health_hunger",
    "route",
    "strategy",
    "version_mechanic",
)


@dataclass(frozen=True)
class Capability:
    capability_id: str
    title: str
    stage: str
    prerequisites: tuple[str, ...]
    dimensions: dict[str, str]
    version_mechanic: str
    task: str


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    provenance: str
    feasible: bool
    feasibility_evidence: tuple[str, ...]
    dimensions: dict[str, str]
    source_run: str | None = None
    parent_snapshot_id: str | None = None
    failure_count: int = 0


CAPABILITIES: tuple[Capability, ...] = (
    Capability("approach-observed-log", "接近已观测树干", "early", ("可见目标",),
               {"inventory_loadout": "empty", "biome_terrain": "tree_crown", "hazard": "fall", "health_hunger": "healthy", "route": "surface", "strategy": "safe_descent"},
               "Minecraft Java 1.21.x 基础移动", "从树冠安全接近已观测树干。"),
    Capability("iron-pickaxe-diamond", "铁镐采钻石", "iron", ("iron_pickaxe", "visible_diamond_ore", "safe_mining_route"),
               {"inventory_loadout": "iron_pickaxe", "biome_terrain": "underground", "hazard": "cave", "health_hunger": "supplied", "route": "mine", "strategy": "target_mining"},
               "Minecraft Java 1.21.x 钻石矿石挖掘", "以铁镐采掘真实可见的钻石矿石。"),
    Capability("diamond-pickaxe-obsidian", "钻石镐采黑曜石", "diamond", ("diamond_pickaxe", "visible_obsidian", "safe_lava_control"),
               {"inventory_loadout": "diamond_pickaxe", "biome_terrain": "underground", "hazard": "lava", "health_hunger": "supplied", "route": "mine", "strategy": "controlled_mining"},
               "Minecraft Java 1.21.x 黑曜石挖掘", "以钻石镐采掘可验证黑曜石并处理邻近岩浆。"),
    Capability("supply-and-pillaring", "垫脚方块与食物补给", "early", ("blocks", "food"),
               {"inventory_loadout": "blocks_food", "biome_terrain": "surface", "hazard": "fall", "health_hunger": "recovering", "route": "surface", "strategy": "resource_management"},
               "Minecraft Java 1.21.x 饥饿与放置机制", "在保留返程余量的前提下补给和搭建。"),
    Capability("village-use", "村庄利用", "early", ("visible_village", "safe_route"),
               {"inventory_loadout": "basic", "biome_terrain": "village", "hazard": "low", "health_hunger": "healthy", "route": "surface", "strategy": "trading_looting"},
               "Minecraft Java 1.21.x 村庄结构与交易", "利用已观测村庄的床、工作站和可行交易。"),
    Capability("iron-golem-combat", "铁傀儡战斗", "iron", ("visible_iron_golem", "weapon", "escape_route"),
               {"inventory_loadout": "weapon_armor", "biome_terrain": "village", "hazard": "combat", "health_hunger": "supplied", "route": "surface", "strategy": "hit_and_retreat"},
               "Minecraft Java 1.21.x 铁傀儡 AI", "在明确撤离路线下完成铁傀儡战斗。"),
    Capability("general-combat", "一般战斗", "early", ("visible_hostile", "weapon_or_safe_fallback"),
               {"inventory_loadout": "basic_weapon", "biome_terrain": "surface_or_cave", "hazard": "combat", "health_hunger": "healthy", "route": "surface_or_cave", "strategy": "spacing"},
               "Minecraft Java 1.21.x 敌对生物 AI", "针对实际观测敌对生物选择距离、掩体和撤离。"),
    Capability("nether-portal-build", "搭建下界门", "diamond", ("obsidian", "ignition", "portal_site"),
               {"inventory_loadout": "obsidian_ignition", "biome_terrain": "surface", "hazard": "dimension_transition", "health_hunger": "supplied", "route": "portal", "strategy": "portal_build"},
               "Minecraft Java 1.21.x 下界门", "用黑曜石搭建并点燃下界门；末影之眼不作为下界路线。"),
    Capability("nether-fortress-survival", "堡垒与烈焰人生存", "nether", ("active_portal", "fire_resistance_plan", "visible_fortress_or_blaze"),
               {"inventory_loadout": "nether_ready", "biome_terrain": "nether_fortress", "hazard": "fire_combat", "health_hunger": "supplied", "route": "nether", "strategy": "cover_and_retreat"},
               "Minecraft Java 1.21.x 下界堡垒与烈焰人", "在可回退快照下处理烈焰人火球和火焰伤害。"),
    Capability("iron-farm", "刷铁机", "iron", ("village_resources", "villagers", "zombie_or_equivalent"),
               {"inventory_loadout": "construction", "biome_terrain": "village", "hazard": "construction", "health_hunger": "supplied", "route": "base", "strategy": "farm_build"},
               "以目标版本实测的刷铁机机制", "仅对经版本验证的结构进行搭建和产量检查。"),
    Capability("boat-enderman", "船机制处理末影人", "end", ("boat", "visible_enderman", "safe_area"),
               {"inventory_loadout": "boat_weapon", "biome_terrain": "end_or_nether_warped", "hazard": "enderman_combat", "health_hunger": "supplied", "route": "combat_arena", "strategy": "boat_control"},
               "Minecraft Java 1.21.x 船与末影人碰撞机制", "在已验证船机制和安全区域内处理末影人。"),
    Capability("eye-find-stronghold", "末影之眼寻找要塞", "end_ready", ("ender_eye", "overworld_route"),
               {"inventory_loadout": "ender_eye", "biome_terrain": "overworld", "hazard": "expedition", "health_hunger": "supplied", "route": "stronghold_search", "strategy": "triangulation"},
               "Minecraft Java 1.21.x 末影之眼定位要塞", "仅在主世界用末影之眼寻找要塞和末地门。"),
    Capability("dragon-loadouts", "不同装备组合击杀末影龙", "end", ("active_end_portal", "combat_loadout", "recovery_plan"),
               {"inventory_loadout": "end_combat_variants", "biome_terrain": "end", "hazard": "boss_combat", "health_hunger": "supplied", "route": "end_portal", "strategy": "crystal_then_dragon"},
               "Minecraft Java 1.21.x 末影龙战斗", "覆盖远程、近战、床爆等经版本验证的装备组合。"),
)


def validate_snapshot(record: SnapshotRecord) -> list[str]:
    """返回不可进入训练队列的原因；空列表表示记录格式和证据都充分。"""
    problems: list[str] = []
    if not record.snapshot_id:
        problems.append("缺少快照 ID")
    if not record.provenance:
        problems.append("缺少来源链")
    if not record.feasible:
        problems.append("快照尚未通过可行性验证")
    if not record.feasibility_evidence:
        problems.append("缺少真实观察或成功末态证据")
    missing = [dimension for dimension in DIMENSIONS if dimension not in record.dimensions]
    if missing:
        problems.append("缺少维度标记：" + "、".join(missing))
    return problems


def capability_eligible(snapshot: SnapshotRecord, capability: Capability) -> bool:
    """只根据已登记快照的真实标签筛掉显然不可能的能力课。"""
    if validate_snapshot(snapshot):
        return False
    for dimension, expected in capability.dimensions.items():
        actual = snapshot.dimensions.get(dimension, "")
        if actual != expected and expected not in {"surface_or_cave", "end_combat_variants"}:
            return False
    return True


def stratified_sample(records: Iterable[SnapshotRecord], *, limit: int) -> list[SnapshotRecord]:
    """按能力维度覆盖和失败重采样选择快照，避免始终沿出生点线性推进。"""
    eligible = [record for record in records if not validate_snapshot(record)]
    selected: list[SnapshotRecord] = []
    seen: dict[str, Counter[str]] = defaultdict(Counter)
    remaining = eligible[:]
    while remaining and len(selected) < limit:
        def priority(record: SnapshotRecord) -> tuple[int, int, str]:
            novel = sum(seen[dimension][value] == 0 for dimension, value in record.dimensions.items())
            return (novel, record.failure_count, record.snapshot_id)
        chosen = max(remaining, key=priority)
        selected.append(chosen)
        remaining.remove(chosen)
        for dimension, value in chosen.dimensions.items():
            seen[dimension][value] += 1
    return selected


def coverage_matrix(records: Iterable[SnapshotRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if not validate_snapshot(record):
            for dimension, value in record.dimensions.items():
                result[dimension][value] += 1
    return {dimension: dict(sorted(values.items())) for dimension, values in sorted(result.items())}


def design_document(actual_snapshot: SnapshotRecord) -> dict[str, Any]:
    """生成可审计的课程银行设计；仅 actual_snapshot 属于本轮已观测数据。"""
    return {
        "contract": {
            "sampling": "按快照维度分层，先满足能力覆盖下限，再以失败次数提高重采样权重。",
            "validity": "每个快照必须记录来源、真实观察或成功末态的可行性证据和全部八个维度；禁止构造不可能装备或资源状态。",
            "curriculum": "课程不是出生点线性序列；从各科技阶段、装备、地形、风险和策略的合格快照中混合抽样。",
            "route_correction": "末影之眼仅用于主世界寻找要塞/末地门；进入下界依赖黑曜石下界门。",
            "failure_resampling": "失败轨迹保留其快照标签和失败原因，在满足安全与物理前置后提高其抽样权重。",
        },
        "dimensions": list(DIMENSIONS),
        "capabilities": [asdict(capability) for capability in CAPABILITIES],
        "observed_snapshot_only": asdict(actual_snapshot),
        "observed_snapshot_validation": validate_snapshot(actual_snapshot),
        "observed_coverage": coverage_matrix([actual_snapshot]),
        "eligible_capabilities_for_observed_snapshot": [
            capability.capability_id for capability in CAPABILITIES if capability_eligible(actual_snapshot, capability)
        ],
        "execution_boundary": "本文件的其余能力是待真实观察与验证后才可入队的课程定义，不代表本轮已执行或已拥有对应快照。",
    }
