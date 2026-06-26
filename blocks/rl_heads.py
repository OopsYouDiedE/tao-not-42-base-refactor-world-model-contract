"""L1 primitive 积木库 - RL 策略/价值头与 fan-in MLP (blocks/rl_heads.py)。

对外接口:
    FanInMLP             — FanInInitReLULayer 堆叠的 MLP(VPT/IMPALA 风格初始化)。
    CategoricalActionHead— 离散动作头:logits=log_softmax,含 sample/log_prob/entropy/kl。
    NormalizeEwma        — 指数滑动均值/方差归一化器(价值目标白化,不参与梯度)。
    ScaledMSEHead        — 带 EWMA 归一化的标量价值头(回归归一化后的目标)。

来源:结构照搬 OpenAI VPT / snu-mllab Achievement-Distillation。task-agnostic RL 组件,故落 blocks/。
数值约定:NormalizeEwma 的除法分母 clamp(min≥1e-2),方差 clamp(min=1e-2),承袭 I1/I4。
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from blocks.impala import FanInInitReLULayer


class FanInMLP(nn.Module):
    """fan-in 初始化的多层感知机(隐层 ReLU+归一化,输出层线性)。

    Args:
        insize/outsize: 输入/输出维度。
        nhidlayer:      隐藏层数。
        hidsize:        隐藏层宽度。
        dense_init_norm_kwargs: 隐藏层归一化配置(本仓 layer_norm=True)。

    Forward:
        x: (B, insize) → (B, outsize), 同 dtype。
    """

    def __init__(
        self,
        insize: int,
        nhidlayer: int,
        outsize: int,
        hidsize: int,
        dense_init_norm_kwargs: Dict = {},
    ):
        super().__init__()
        insizes = [insize] + nhidlayer * [hidsize]
        outsizes = nhidlayer * [hidsize] + [outsize]
        self.layers = nn.ModuleList()
        for i, (isz, osz) in enumerate(zip(insizes, outsizes)):
            use_activation = i < nhidlayer
            init_scale = 1.4 if use_activation else 1.0
            norm_kwargs = dense_init_norm_kwargs if use_activation else {}
            self.layers.append(
                FanInInitReLULayer(
                    isz,
                    osz,
                    layer_type="linear",
                    use_activation=use_activation,
                    init_scale=init_scale,
                    **norm_kwargs,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CategoricalActionHead(nn.Module):
    """离散动作头。线性层后取 log_softmax 作为 logits(已是 log 概率)。

    Args:
        insize:      输入特征维度。
        num_actions: 动作数。
        init_scale:  正交初始化增益(默认 0.01,小方差输出)。
    """

    def __init__(self, insize: int, num_actions: int, init_scale: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(insize, num_actions)
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, insize) → logits (B, num_actions) = log_softmax。"""
        return F.log_softmax(self.linear(x), dim=-1)

    def log_prob(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """logits (B, A), actions (B, 1) long → log_prob (B, 1)。"""
        return torch.gather(logits, dim=-1, index=actions)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """logits (B, A) → entropy (B, 1)。"""
        probs = torch.exp(logits)
        return -torch.sum(probs * logits, dim=-1, keepdim=True)

    def sample(self, logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Gumbel-max 采样(或 argmax)。logits (B, A) → actions (B, 1) long。"""
        if deterministic:
            return torch.argmax(logits, dim=-1, keepdim=True)
        u = torch.rand_like(logits)
        u[u == 1.0] = 0.999
        gumbels = logits - torch.log(-torch.log(u))
        return torch.argmax(gumbels, dim=-1, keepdim=True)

    def kl_divergence(self, logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
        """KL(q‖p),q/p 均为 log_softmax 后的 logits。→ (B, 1)。"""
        return torch.sum(torch.exp(logits_q) * (logits_q - logits_p), dim=-1, keepdim=True)


class NormalizeEwma(nn.Module):
    """指数滑动(EWMA)均值/方差归一化器(buffer 形式,不参与梯度)。

    训练态下用批统计量以 beta 滑动更新 running mean / mean_sq;denormalize 逆变换。

    Args:
        insize:    被归一化的标量维度。
        beta:      EWMA 衰减系数。
        epsilon:   去偏项分母下界(I1:≥1e-2)。
    """

    def __init__(self, insize: int, norm_axes: int = 1, beta: float = 0.99, epsilon: float = 1e-2):
        super().__init__()
        self.norm_axes = norm_axes
        self.beta = beta
        self.epsilon = epsilon
        self.running_mean = nn.Parameter(torch.zeros(insize), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(insize), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def running_mean_var(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean**2).clamp(min=1e-2)
        return mean, var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_detach = x.detach()
            batch_mean = x_detach.mean(dim=tuple(range(self.norm_axes)))
            batch_mean_sq = (x_detach**2).mean(dim=tuple(range(self.norm_axes)))
            w = self.beta
            self.running_mean.mul_(w).add_(batch_mean * (1.0 - w))
            self.running_mean_sq.mul_(w).add_(batch_mean_sq * (1.0 - w))
            self.debiasing_term.mul_(w).add_(1.0 * (1.0 - w))
        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        return x * torch.sqrt(var) + mean


class ScaledMSEHead(nn.Module):
    """标量价值头:线性输出 + EWMA 归一化(在归一化空间回归)。

    Args:
        insize/outsize: 输入维度 / 输出维度(价值为 1)。
        init_scale:     正交初始化增益(默认 0.1)。
        norm_kwargs:    传给 NormalizeEwma。
    """

    def __init__(self, insize: int, outsize: int, init_scale: float = 0.1, norm_kwargs: Dict = {}):
        super().__init__()
        self.linear = nn.Linear(insize, outsize)
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)
        self.normalizer = NormalizeEwma(outsize, **norm_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer(x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize(x)

    def mse_loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """pred 与 归一化后的 targ 的逐元素 MSE(不 reduce)。"""
        return F.mse_loss(pred, self.normalizer(targ), reduction="none")
