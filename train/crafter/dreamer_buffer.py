"""DreamerV3 序列回放缓冲区 (train/crafter/dreamer_buffer.py)。

对外接口:
    SequenceReplay — 按 (时间, env) 存全部交互流,采样定长 (B, L) 序列窗口供世界模型训练。

存储落在 CPU(obs 以 uint8 压缩),采样的小批量再转到训练设备。窗口可跨 episode 边界,
边界由 is_first 标记(RSSM 在窗口内据此重置初始状态)。non-wrapping 容量:cap 用尽即停止写入
(训练侧据 total_steps 控制 cap),避免环形窗口跨写指针的复杂度。
"""
import numpy as np
import torch


class SequenceReplay:
    """定长序列回放缓冲区。

    Args:
        capacity:    每个 env 的最大存储步数(总容量 = capacity × n_envs)。
        n_envs:      并行环境数(每步同时写入 n_envs 列)。
        obs_shape:   单帧观测 (C, H, W)。
        num_actions: 离散动作数(动作以 one-hot 存储)。

    存储张量(均在 CPU):
        obs:      [cap, n_envs, C, H, W] uint8(∈ [0, 255])。
        action:   [cap, n_envs, A] float32 one-hot。
        reward:   [cap, n_envs] float32。
        cont:     [cap, n_envs] float32(1 = 延续)。
        is_first: [cap, n_envs] float32(1 = 该步为轨迹起点)。
    """

    def __init__(self, capacity, n_envs, obs_shape, num_actions):
        c, h, w = obs_shape
        self.capacity = capacity
        self.n_envs = n_envs
        self.num_actions = num_actions
        self.obs = torch.zeros(capacity, n_envs, c, h, w, dtype=torch.uint8)
        self.action = torch.zeros(capacity, n_envs, num_actions, dtype=torch.float32)
        self.reward = torch.zeros(capacity, n_envs, dtype=torch.float32)
        self.cont = torch.zeros(capacity, n_envs, dtype=torch.float32)
        self.is_first = torch.zeros(capacity, n_envs, dtype=torch.float32)
        # 每步活动的目标 id(goal 条件化用;vanilla 路径恒为 0,内存可忽略)。
        self.goal = torch.zeros(capacity, n_envs, dtype=torch.int16)
        self._ptr = 0
        self._full = False

    def __len__(self):
        return self.capacity if self._full else self._ptr

    def add(self, obs, action_onehot, reward, cont, is_first, goal_ids=None):
        """追加一步(全 env)。

        Args:
            obs:           [n_envs, C, H, W] float ∈ [0, 1]。
            action_onehot: [n_envs, A] float。
            reward:        [n_envs] float。
            cont:          [n_envs] float(1 = 延续,0 = 终止)。
            is_first:      [n_envs] float。
            goal_ids:      [n_envs] long/int 该步目标 id;None = 0(vanilla)。
        """
        if self._ptr >= self.capacity:
            self._full = True
            return  # 容量用尽:停止写入(non-wrapping)
        t = self._ptr
        self.obs[t] = (obs.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
        self.action[t] = action_onehot.cpu()
        self.reward[t] = reward.cpu()
        self.cont[t] = cont.cpu()
        self.is_first[t] = is_first.cpu()
        if goal_ids is not None:
            self.goal[t] = goal_ids.to(torch.int16).cpu()
        self._ptr += 1

    def can_sample(self, seq_len):
        return len(self) >= seq_len + 1

    def sample(self, batch_size, seq_len, device):
        """采样 batch_size 条长度 seq_len 的序列窗口。

        Returns(均在 device):
            obs:      [B, L, C, H, W] float ∈ [0, 1]。
            action:   [B, L, A] float。
            reward:   [B, L] float。
            cont:     [B, L] float。
            is_first: [B, L] float。
        """
        n = len(self)
        max_start = n - seq_len
        starts = np.random.randint(0, max_start + 1, size=batch_size)
        envs = np.random.randint(0, self.n_envs, size=batch_size)

        obs = torch.empty(batch_size, seq_len, *self.obs.shape[2:], dtype=torch.uint8)
        action = torch.empty(batch_size, seq_len, self.num_actions)
        reward = torch.empty(batch_size, seq_len)
        cont = torch.empty(batch_size, seq_len)
        is_first = torch.empty(batch_size, seq_len)
        goal = torch.empty(batch_size, seq_len, dtype=torch.int16)
        for i, (s, e) in enumerate(zip(starts, envs)):
            sl = slice(s, s + seq_len)
            obs[i] = self.obs[sl, e]
            action[i] = self.action[sl, e]
            reward[i] = self.reward[sl, e]
            cont[i] = self.cont[sl, e]
            is_first[i] = self.is_first[sl, e]
            goal[i] = self.goal[sl, e]

        return (
            obs.to(device).float() / 255.0,
            action.to(device),
            reward.to(device),
            cont.to(device),
            is_first.to(device),
            goal.to(device).long(),
        )
