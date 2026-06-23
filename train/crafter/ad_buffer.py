"""Achievement Distillation 示范缓冲区 (train/crafter/ad_buffer.py)。

对外接口:
    ACHIEVEMENTS    — Crafter 22 个成就名称列表(固定,与 crafter.Env 一致)。
    AchievementBuffer — 按成就分组存储(obs, action)示范,支持均衡随机采样。
"""
import numpy as np
import torch

ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]
N_ACHIEVEMENTS = len(ACHIEVEMENTS)


class AchievementBuffer:
    """按成就分组存储(obs, action)示范对的循环缓冲区。

    为避免高频成就(如 collect_wood)淹没低频成就,采样时对有数据的成就
    等量采样后合并,而非从全局扁平列表随机取。

    Args:
        cap_per_achievement: 每个成就最多保留的(obs, action)步数。
        device:              tensor 存储设备。

    Obs 存储为 float32 CPU tensor,采样时转移到 device。
    """

    def __init__(self, cap_per_achievement: int = 100, device: str = "cuda"):
        self.cap = cap_per_achievement
        self.device = device
        # 每个成就独立的 obs/action 列表(循环队列语义,用 list + 手动截断)
        self._obs: dict[str, list] = {a: [] for a in ACHIEVEMENTS}
        self._act: dict[str, list] = {a: [] for a in ACHIEVEMENTS}

    def add_demo(self, achievement: str, obs_seq: list, action_seq: list) -> None:
        """追加一段示范序列到对应成就槽。

        Args:
            achievement: 成就名称,须在 ACHIEVEMENTS 中。
            obs_seq:     list of (C, H, W) float32 CPU tensor。
            action_seq:  list of int。
        """
        buf_obs = self._obs[achievement]
        buf_act = self._act[achievement]
        for o, a in zip(obs_seq, action_seq):
            buf_obs.append(o.cpu())
            buf_act.append(int(a))
            if len(buf_obs) > self.cap:
                buf_obs.pop(0)
                buf_act.pop(0)

    def sample(self, batch_size: int):
        """从有数据的成就中均衡采样。

        Returns:
            obs:     (B, C, H, W) float32 tensor on self.device,或 None。
            actions: (B,) long tensor on self.device,或 None。
        """
        active = [a for a in ACHIEVEMENTS if self._obs[a]]
        if not active:
            return None, None

        per_ach = max(1, batch_size // len(active))
        obs_list, act_list = [], []
        for ach in active:
            n = len(self._obs[ach])
            k = min(per_ach, n)
            idx = np.random.choice(n, size=k, replace=False)
            obs_list.extend(self._obs[ach][i] for i in idx)
            act_list.extend(self._act[ach][i] for i in idx)

        obs_batch = torch.stack(obs_list).to(self.device)
        act_batch = torch.tensor(act_list, dtype=torch.long, device=self.device)
        return obs_batch, act_batch

    def total_steps(self) -> int:
        """所有成就槽中已存储的(obs, action)总步数。"""
        return sum(len(v) for v in self._obs.values())

    def coverage(self) -> int:
        """有至少一个示范的成就数。"""
        return sum(1 for v in self._obs.values() if v)

    def covered_names(self) -> list:
        """已解锁(有示范)的成就名,按 ACHIEVEMENTS 固定顺序。"""
        return [a for a in ACHIEVEMENTS if self._obs[a]]


# 科技树"硬墙"成就:都有前置依赖,解锁它们才算越过 vanilla DreamerV3 卡死的易成就层。
# make_wood_pickaxe 是第一道闸(需木头+工作台),其后逐级解锁石/煤/铁/钻石链。
HARD_ACHIEVEMENTS = [
    "make_wood_pickaxe", "collect_stone", "place_stone", "make_wood_sword",
    "make_stone_pickaxe", "make_stone_sword", "place_furnace", "collect_coal",
    "collect_iron", "make_iron_pickaxe", "make_iron_sword", "collect_diamond",
    "eat_plant",
]
