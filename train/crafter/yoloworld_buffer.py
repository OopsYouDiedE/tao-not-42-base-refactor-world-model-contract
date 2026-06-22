"""YOLO-World-Dreamer 序列回放缓冲区 (train/crafter/yoloworld_buffer.py)。

对外接口:
    GoalSequenceReplay — 在 DreamerV3 序列回放基础上,额外存逐步**成就 multi-hot** 与
                         **目标索引**;采样定长 (B, L) 窗口,支持 HER 事后重标目标。

存储落 CPU(obs uint8 压缩),采样转训练设备。HER:对一部分窗口,把目标重标为该窗口内
真实解锁过的某个成就 → 稀疏目标条件信号变稠密(见 knowledge/yoloworld.md §8 / GoalSampler)。
non-wrapping 容量(cap 用尽即停写),与 SequenceReplay 同约定。
"""
import numpy as np
import torch


class GoalSequenceReplay:
    """含成就/目标的定长序列回放缓冲。

    Args:
        capacity:        每 env 最大步数(总容量 = capacity × n_envs)。
        n_envs:          并行环境数。
        obs_shape:       单帧 (C, H, W)。
        num_actions:     离散动作数(one-hot 存)。
        n_achievements:  成就数 U。

    存储张量(CPU):
        obs:      [cap, n_envs, C, H, W] uint8。
        action:   [cap, n_envs, A] float32 one-hot。
        reward:   [cap, n_envs] float32(环境原始奖励,记录用)。
        cont:     [cap, n_envs] float32(1 = 延续)。
        ach:      [cap, n_envs, U] uint8(该步累计已解锁成就 multi-hot)。
        goal:     [cap, n_envs] int16(采集时的目标成就索引)。
        is_first: [cap, n_envs] float32(1 = 轨迹起点)。
    """

    def __init__(self, capacity, n_envs, obs_shape, num_actions, n_achievements):
        c, h, w = obs_shape
        self.capacity = capacity
        self.n_envs = n_envs
        self.num_actions = num_actions
        self.U = n_achievements
        self.obs = torch.zeros(capacity, n_envs, c, h, w, dtype=torch.uint8)
        self.action = torch.zeros(capacity, n_envs, num_actions, dtype=torch.float32)
        self.reward = torch.zeros(capacity, n_envs, dtype=torch.float32)
        self.cont = torch.zeros(capacity, n_envs, dtype=torch.float32)
        self.ach = torch.zeros(capacity, n_envs, n_achievements, dtype=torch.uint8)
        self.goal = torch.zeros(capacity, n_envs, dtype=torch.int16)
        self.is_first = torch.zeros(capacity, n_envs, dtype=torch.float32)
        self._ptr = 0
        self._full = False

    def __len__(self):
        return self.capacity if self._full else self._ptr

    def add(self, obs, action_onehot, reward, cont, ach, goal, is_first):
        """追加一步(全 env)。形状见类 docstring 对应字段(去掉 cap 维)。"""
        if self._ptr >= self.capacity:
            self._full = True
            return
        t = self._ptr
        self.obs[t] = (obs.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
        self.action[t] = action_onehot.cpu()
        self.reward[t] = reward.cpu()
        self.cont[t] = cont.cpu()
        self.ach[t] = ach.to(torch.uint8).cpu()
        self.goal[t] = goal.to(torch.int16).cpu()
        self.is_first[t] = is_first.cpu()
        self._ptr += 1

    def can_sample(self, seq_len):
        return len(self) >= seq_len + 1

    def sample(self, batch_size, seq_len, device, her_ratio=0.0):
        """采样 batch_size 条长 seq_len 的窗口。

        Args:
            batch_size: 窗口数 B。
            seq_len:    窗口长 L。
            device:     返回设备。
            her_ratio:  ∈[0,1],被 HER 重标的窗口比例;重标目标 = 该窗口内
                        真实解锁过的某个成就(无解锁则保留原目标)。

        Returns(均在 device):
            obs [B,L,C,H,W] float∈[0,1] / action [B,L,A] / reward [B,L] /
            cont [B,L] / ach [B,L,U] float / goal [B,L] long / is_first [B,L]。
        """
        n = len(self)
        max_start = n - seq_len
        starts = np.random.randint(0, max_start + 1, size=batch_size)
        envs = np.random.randint(0, self.n_envs, size=batch_size)

        obs = torch.empty(batch_size, seq_len, *self.obs.shape[2:], dtype=torch.uint8)
        action = torch.empty(batch_size, seq_len, self.num_actions)
        reward = torch.empty(batch_size, seq_len)
        cont = torch.empty(batch_size, seq_len)
        ach = torch.empty(batch_size, seq_len, self.U, dtype=torch.uint8)
        goal = torch.empty(batch_size, seq_len, dtype=torch.long)
        is_first = torch.empty(batch_size, seq_len)
        for i, (s, e) in enumerate(zip(starts, envs)):
            sl = slice(s, s + seq_len)
            obs[i] = self.obs[sl, e]
            action[i] = self.action[sl, e]
            reward[i] = self.reward[sl, e]
            cont[i] = self.cont[sl, e]
            ach[i] = self.ach[sl, e]
            goal[i] = self.goal[sl, e].long()
            is_first[i] = self.is_first[sl, e]

        if her_ratio > 0.0:
            self._relabel(goal, ach, her_ratio)

        return (
            obs.to(device).float() / 255.0,
            action.to(device),
            reward.to(device),
            cont.to(device),
            ach.to(device).float(),
            goal.to(device),
            is_first.to(device),
        )

    @staticmethod
    def _relabel(goal, ach, her_ratio):
        """HER 事后重标(就地改 goal)。

        对随机 her_ratio 比例的窗口,若窗口内有解锁过的成就(ach 在 L 维 any),
        从中随机挑一个作整窗新目标;否则不改。
        """
        B, L = goal.shape
        unlocked = ach.any(dim=1)                       # [B, U] 窗口内是否解锁过
        pick = torch.rand(B) < her_ratio
        for i in range(B):
            if not pick[i]:
                continue
            cand = unlocked[i].nonzero(as_tuple=False).flatten()
            if cand.numel() == 0:
                continue
            new_g = int(cand[np.random.randint(0, cand.numel())])
            goal[i] = new_g
