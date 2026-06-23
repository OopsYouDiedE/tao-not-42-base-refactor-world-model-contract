"""Go-Explore 状态缓存 (train/crafter/state_cache.py)。

对外接口:
    StateCache — 按 stage(已解锁成就数)分桶的 crafter.Env 快照池;push 存档、sample 取档。

破解"塌缩到易成就":把"刚踏入各阶段那一刻"的世界状态深拷贝存档,训练时让 env 以一定概率
**空降**到任意深度的存档点续探(并赋该处的下一前沿目标),省去每局从零重爬科技树。
快照可行性见 deepcopy(crafter.Env) 实测(独立、可确定性复现,~97KB/状态)。

stage 用"已解锁成就数"分桶(0–22);采样时**按桶均匀**(而非按数量),使稀有的深层桶
获得不成比例的曝光 ⇒ 偏向探索深层阶段。设计见 plan。
"""
import numpy as np


class StateCache:
    """按 stage 分桶的状态快照池(每桶 reservoir 截断)。

    Args:
        cap_per_stage: 每个 stage 桶最多保留的快照数。
        seed:          采样/淘汰 RNG 种子。

    快照(bundle)是 dict:{"env": deepcopy(crafter.Env), "obs": CPU tensor,
    "prev_ach": dict, "unlocked": set[str]}(由 VecCrafterEnv.snapshot 产出)。
    """

    def __init__(self, cap_per_stage: int = 50, seed: int = 0):
        self.cap = cap_per_stage
        self.buckets: dict[int, list] = {}
        self.rng = np.random.RandomState(seed)
        self._n = 0

    def __len__(self):
        return self._n

    def push(self, bundle, stage: int) -> None:
        """存档:stage 桶未满则追加,满则随机替换(reservoir 式,保多样)。"""
        b = self.buckets.setdefault(stage, [])
        if len(b) < self.cap:
            b.append(bundle)
            self._n += 1
        else:
            b[self.rng.randint(self.cap)] = bundle

    def sample(self):
        """取档:先按非空桶均匀选 stage(偏向稀有深层),再桶内随机。空池返回 None。"""
        stages = [s for s, b in self.buckets.items() if b]
        if not stages:
            return None
        s = stages[self.rng.randint(len(stages))]
        b = self.buckets[s]
        return b[self.rng.randint(len(b))]

    def coverage(self) -> int:
        """已有快照的 stage 数(诊断:课程覆盖到多深)。"""
        return sum(1 for b in self.buckets.values() if b)
