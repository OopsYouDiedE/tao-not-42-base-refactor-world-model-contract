"""Crafter 成就目标空间 (train/crafter/goal.py)。

把 22 个 Crafter 成就当作**文本目标**:用冻结 MiniLM(train/minecraft/task_text.py 的
TaskTextEncoder)把成就的自然语英文句一次性编码成 [22, 384] 冻结向量,供 DreamerV3
的 goal 条件化 actor 做"文本点乘判定动作概率"(YOLOE 文本-动作对齐搬运)。

对外接口:
    GoalSpace — 持有 22 个目标的文本嵌入 + 科技树前置表 + "下一前沿目标"采样器。

设计见 knowledge plan / [[crafter-1m-run-config]]。目标只条件化行为,不改 vanilla 奖励。
"""
import random

import torch

from train.crafter.ad_buffer import ACHIEVEMENTS

# 成就 → 自然语英文句(给 MiniLM 编码;语义相近的指令向量相近)。
GOAL_TEXTS = {
    "collect_wood": "collect wood from a tree",
    "place_table": "place a crafting table",
    "make_wood_pickaxe": "make a wood pickaxe",
    "make_wood_sword": "make a wood sword",
    "collect_stone": "mine stone with a pickaxe",
    "place_stone": "place a stone block",
    "make_stone_pickaxe": "make a stone pickaxe",
    "make_stone_sword": "make a stone sword",
    "collect_coal": "mine coal",
    "collect_iron": "mine iron ore",
    "place_furnace": "place a furnace",
    "make_iron_pickaxe": "make an iron pickaxe",
    "make_iron_sword": "make an iron sword",
    "collect_diamond": "mine a diamond",
    "collect_sapling": "collect a sapling",
    "place_plant": "plant a sapling",
    "eat_plant": "eat a grown plant",
    "collect_drink": "drink water",
    "eat_cow": "eat a cow",
    "defeat_zombie": "defeat a zombie",
    "defeat_skeleton": "defeat a skeleton",
    "wake_up": "sleep and wake up",
}

# 科技树前置(近似,够用即可):某成就的"前置成就全解锁"才算可达前沿。
# 不求精确,只用于把目标采样偏向"够得着的下一步"。
PREREQS = {
    "collect_wood": [],
    "place_table": ["collect_wood"],
    "make_wood_pickaxe": ["place_table"],
    "make_wood_sword": ["place_table"],
    "collect_stone": ["make_wood_pickaxe"],
    "place_stone": ["collect_stone"],
    "make_stone_pickaxe": ["collect_stone", "place_table"],
    "make_stone_sword": ["collect_stone", "place_table"],
    "collect_coal": ["make_wood_pickaxe"],
    "collect_iron": ["make_stone_pickaxe"],
    "place_furnace": ["collect_stone"],
    "make_iron_pickaxe": ["collect_iron", "place_furnace", "collect_coal"],
    "make_iron_sword": ["collect_iron", "place_furnace", "collect_coal"],
    "collect_diamond": ["make_iron_pickaxe"],
    "collect_sapling": [],
    "place_plant": ["collect_sapling"],
    "eat_plant": ["place_plant"],
    "collect_drink": [],
    "eat_cow": [],
    "defeat_zombie": [],
    "defeat_skeleton": [],
    "wake_up": [],
}


class GoalSpace:
    """成就目标空间:文本嵌入查表 + 下一前沿目标采样。

    Args:
        encoder_kind: "minilm"(冻结句向量)或 "mock"(哈希伪嵌入,离线冒烟)。
        device:       目标嵌入张量所在设备。

    属性:
        embeddings: [N, text_dim] 冻结目标嵌入(N=22)。
        text_dim:   嵌入维(MiniLM = 384)。
    """

    def __init__(self, encoder_kind: str = "minilm", device: str = "cpu"):
        from train.minecraft.task_text import TaskTextEncoder
        self.names = list(ACHIEVEMENTS)
        self.n = len(self.names)
        self._idx = {a: i for i, a in enumerate(self.names)}
        enc = TaskTextEncoder(kind=encoder_kind, device="cpu")
        texts = [GOAL_TEXTS[a] for a in self.names]
        self.embeddings = enc.encode(texts).to(device).float()   # [N, text_dim]
        self.text_dim = self.embeddings.shape[-1]
        self.device = device

    def embedding(self, goal_ids: torch.Tensor) -> torch.Tensor:
        """goal_ids[...] long → [..., text_dim] 嵌入(在 self.device)。"""
        return self.embeddings[goal_ids.to(self.embeddings.device)]

    def next_frontier(self, unlocked) -> int:
        """给定已解锁成就名集合,返回一个"下一前沿"目标 id。

        优先:前置全满足且未解锁的成就;退化:任意未解锁;再退化:任意成就。
        """
        unlocked = set(unlocked)
        cands = [i for i, a in enumerate(self.names)
                 if a not in unlocked and set(PREREQS[a]).issubset(unlocked)]
        if not cands:
            cands = [i for i, a in enumerate(self.names) if a not in unlocked]
        if not cands:
            cands = list(range(self.n))
        return random.choice(cands)
