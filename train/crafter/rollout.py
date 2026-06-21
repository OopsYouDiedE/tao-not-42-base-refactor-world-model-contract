"""PPO 轨迹缓冲区 (train/crafter/rollout.py)。

对外接口:
    RolloutBuffer — 收集 n_envs × n_steps 步,计算 GAE 优势,迭代 minibatch。
"""
import torch


class RolloutBuffer:
    """PPO on-policy 轨迹缓冲区。

    按 (time, env) 存储数据;compute_gae 后通过 get_minibatches 展平并乱序
    切 minibatch。

    Args:
        n_envs:      并行环境数。
        n_steps:     每次 rollout 每 env 收集的步数。
        obs_shape:   单步观测形状 (C, H, W)。
        gamma:       折扣因子。
        gae_lambda:  GAE λ。
        device:      存储设备。
    """

    def __init__(self, n_envs: int, n_steps: int, obs_shape: tuple,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: str = "cuda"):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        N, E = n_steps, n_envs
        self.obs = torch.zeros(N, E, *obs_shape, device=device)
        self.actions = torch.zeros(N, E, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(N, E, device=device)
        self.rewards = torch.zeros(N, E, device=device)
        self.dones = torch.zeros(N, E, device=device)
        self.values = torch.zeros(N, E, device=device)

        self._ptr = 0
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        """追加一步数据。

        Args:
            obs:      (E, C, H, W) float32。
            action:   (E,) long。
            log_prob: (E,) float32。
            reward:   (E,) float32。
            done:     (E,) float32 (0/1)。
            value:    (E,) float32。
        """
        t = self._ptr
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self._ptr += 1

    def compute_gae(self, last_value: torch.Tensor, last_done: torch.Tensor) -> None:
        """计算 GAE 优势与 λ-return。

        Args:
            last_value: (E,) float32 — rollout 末尾下一步的 V(s')。
            last_done:  (E,) float32 — rollout 末尾是否 terminal。
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nonterminal = 1.0 - last_done.float()
                next_val = last_value
            else:
                nonterminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_val * nonterminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values
        self._ptr = 0

    def get_minibatches(self, minibatch_size: int):
        """展平并乱序切分 minibatch 的迭代器。

        优势标准化(mean=0,std=1)在此处进行,见 I1:ε=1e-4。

        Yields:
            (obs, actions, old_log_probs, advantages, returns, values)
            每项均为 (B, ...) tensor,B = minibatch_size(最后一批可能较小)。
        """
        total = self.n_envs * self.n_steps
        obs = self.obs.view(total, *self.obs.shape[2:])
        actions = self.actions.view(total)
        log_probs = self.log_probs.view(total)
        adv = self.advantages.view(total)
        ret = self.returns.view(total)
        val = self.values.view(total)

        # I1: ε ≥ 1e-4
        adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        idx = torch.randperm(total, device=self.device)
        for start in range(0, total, minibatch_size):
            mb = idx[start: start + minibatch_size]
            yield obs[mb], actions[mb], log_probs[mb], adv[mb], ret[mb], val[mb]
