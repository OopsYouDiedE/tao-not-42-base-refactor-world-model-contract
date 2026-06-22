"""Crafter 成就 → 语言任务条件 (train/crafter/crafter_tasks.py)。

对外接口:
    ACHIEVEMENT_SENTENCES — 22 个成就的自然语言指令(与 ad_buffer.ACHIEVEMENTS 同序)。
    build_ach_embed       — 用冻结句编码器把 22 句指令编成嵌入矩阵 E [U, d_g](单位球行)。
    GoalSampler           — 按 env 维护"当前目标成就",reset/done 时重采样,给出 task_emb。

这是 YOLOE 文本侧的域落地:任务用语言描述、编码进与计划嵌入同一空间(见 knowledge/yoloworld.md §3)。
net/ 不读文件/不 import 域,故 E 与 task_emb 在本层算好后以张量注入智能体。
"""
import torch

from train.crafter.ad_buffer import ACHIEVEMENTS
from train.minecraft.task_text import TaskTextEncoder

# 成就 → 指令句(与 ACHIEVEMENTS 一一对应)。语义相近的成就句向量相近,
# 使点乘选择对新指令有外推可能(冻结句空间的先验)。
ACHIEVEMENT_SENTENCES = {
    "collect_coal": "mine coal from rocks",
    "collect_diamond": "mine a diamond",
    "collect_drink": "drink water to quench thirst",
    "collect_iron": "mine iron ore",
    "collect_sapling": "collect a sapling",
    "collect_stone": "mine stone",
    "collect_wood": "collect wood from trees",
    "defeat_skeleton": "defeat a skeleton",
    "defeat_zombie": "defeat a zombie",
    "eat_cow": "hunt and eat a cow",
    "eat_plant": "eat a ripe plant",
    "make_iron_pickaxe": "craft an iron pickaxe",
    "make_iron_sword": "craft an iron sword",
    "make_stone_pickaxe": "craft a stone pickaxe",
    "make_stone_sword": "craft a stone sword",
    "make_wood_pickaxe": "craft a wooden pickaxe",
    "make_wood_sword": "craft a wooden sword",
    "place_furnace": "place a furnace",
    "place_plant": "plant a sapling in the ground",
    "place_stone": "place a stone block",
    "place_table": "place a crafting table",
    "wake_up": "wake up after sleeping",
}


def build_ach_embed(encoder: TaskTextEncoder, device="cpu") -> torch.Tensor:
    """22 句成就指令 → 嵌入矩阵 E [U, d_g](行 = 单位向量)。

    Args:
        encoder: 冻结句编码器(TaskTextEncoder)。
        device:  返回张量设备。

    Returns:
        E: [U, d_g] float32,行序与 ACHIEVEMENTS 一致。
    """
    sents = [ACHIEVEMENT_SENTENCES[a] for a in ACHIEVEMENTS]
    E = encoder.encode(sents)                       # [U, d_g],encode 已 L2 归一
    return E.to(device)


class GoalSampler:
    """按 env 维护当前目标成就索引,给出对应任务句向量。

    Args:
        ach_embed: E [U, d_g](单位球)。
        n_envs:    并行环境数。
        device:    task_emb 设备。
        seed:      采样种子。

    每 env 一个目标索引 ∈ [0, U)。done/reset 时调用 resample(mask) 重采样。
    """

    def __init__(self, ach_embed: torch.Tensor, n_envs: int, device="cpu", seed=0):
        self.E = ach_embed.to(device)
        self.U = ach_embed.shape[0]
        self.n_envs = n_envs
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.goal = torch.randint(0, self.U, (n_envs,), generator=self.gen)

    def resample(self, mask):
        """对 mask 为 1 的 env 重采样目标成就。

        Args:
            mask: [n_envs] float/bool(1 = 该 env 需重采样,通常 = done)。
        """
        m = mask.bool().cpu()
        if m.any():
            new = torch.randint(0, self.U, (int(m.sum()),), generator=self.gen)
            self.goal[m] = new

    def task_emb(self):
        """当前各 env 目标的任务句向量 [n_envs, d_g](on device)。"""
        return self.E[self.goal.to(self.E.device)]

    def goal_idx(self):
        """当前各 env 目标索引 [n_envs] long(on device)。"""
        return self.goal.to(self.device)
