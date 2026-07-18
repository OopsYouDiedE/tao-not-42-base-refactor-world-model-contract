"""L1 primitive 积木库 - 正则与随机潜空间 (blocks/regularization.py)"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_FP16 = 1e-4  # I1: fp16 安全 epsilon(绝不用 1e-12)


class StochLatent(nn.Module):
    """随机潜在。gaussian: reparam + tanh 有界;categorical: 直通估计。采样 fp32(I4),返回 KL≥0。"""

    def __init__(self, in_dim, z_dim, kind="gaussian"):
        super().__init__()
        assert kind in ("gaussian", "categorical")
        self.kind, self.z_dim = kind, z_dim
        self.proj = nn.Linear(in_dim, 2 * z_dim if kind == "gaussian" else z_dim)

    def forward(self, h):
        if self.kind == "gaussian":
            mu, logsig = self.proj(h).float().chunk(2, dim=-1)       # fp32, I4
            logsig = logsig.clamp(-8.0, 8.0)
            sigma = torch.exp(logsig)
            z = mu + sigma * torch.randn_like(mu)                    # reparam
            kl = 0.5 * (mu ** 2 + sigma ** 2 - 2 * logsig - 1).sum(-1)
            return torch.tanh(z).to(h.dtype), kl                     # I3 有界
        logits = self.proj(h).float()
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs.reshape(-1, self.z_dim), 1).reshape(probs.shape[:-1])
        onehot = F.one_hot(idx, self.z_dim).float()
        z = onehot + probs - probs.detach()                         # straight-through
        kl = (probs * (F.log_softmax(logits, -1) + math.log(self.z_dim))).sum(-1)
        return z.to(h.dtype), kl


class SIGReg(nn.Module):
    """Sliced 各向同性高斯正则(防表征坍缩)。

    来源:LeWM(github.com/lucas-maes/le-wm,见其 LICENSE)。
    思路:用随机单位方向把高维 embedding 投影成 1D,再用 Epps-Pulley 经验特征函数检验,
    把每个投影分布钉到标准正态 N(0,1)——目标实部 `E[cos(t·s)]=exp(-t²/2)`、虚部 `E[sin(t·s)]=0`。
    单 GPU、无需 EMA target、无需负样本即可防坍缩。统计项只进入 loss，不进入前向（I6）。

    Args:
        knots: 梯形积分节点数(t∈[0,3])。
        num_proj: 每次前向重采样的随机投影方向数(Monte-Carlo slicing)。
        eps: 投影方向归一化分母下界(I1,fp16 安全)。

    Shape:
        proj: [G, B, D] —— G 独立分组(各自一份检验,如时间步;无分组传 1), B 样本维(分布在此维上估计), D 特征维。
        返回: 0 维标量 ∈ [0, ∞)。完美高斯 → 小常量(O(1),不随 B 趋 0);坍缩(常量/低秩)→ O(B) 大(判别力在比值,非绝对值)。

    Dtype: 投影与 cos/sin/mean 全程 fp32(I4);窗 `exp(-t²/2)` 有界(I2);归一化分母 clamp(I1)。
    """

    def __init__(self, knots=17, num_proj=1024, eps=EPS_FP16):
        super().__init__()
        self.num_proj = num_proj
        self.eps = eps
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)  # 梯形权重(偶被积函数 ⇒ 内点 ×2)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)                          # N(0,1) 特征函数实部 = 高斯窗(I2 有界)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)              # 积分权重 ⊙ 窗:抑制大 t 噪声

    def forward(self, proj):
        x = proj.float()                                               # I4
        D = x.size(-1)
        A = torch.randn(D, self.num_proj, device=x.device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp(min=self.eps)   # I1:单位方向,分母 clamp
        s_t = (x @ A).unsqueeze(-1) * self.t                           # [G,B,num_proj,knots] fp32
        ecf_re = s_t.cos().mean(-3)                                    # 经验特征函数实部 E_B[cos(t·s)]
        ecf_im = s_t.sin().mean(-3)                                    # 虚部 E_B[sin(t·s)]
        err = (ecf_re - self.phi).square() + ecf_im.square()          # 与 N(0,1) 特征函数之差²
        statistic = (err @ self.weights) * x.size(-2)                 # 梯形积分,×B 为 Epps-Pulley 标度
        return statistic.mean()                                        # 各分组/投影上平均


class BoundedActivation(nn.Module):
    """I3 执行者。depth: exp-clamp; flow: s·tanh; pos: softplus+ε; prob: sigmoid。"""

    def __init__(self, kind, scale=1.5, eps=EPS_FP16, clamp=4.6):
        super().__init__()
        assert kind in ("depth", "flow", "pos", "prob")
        self.kind, self.scale, self.eps, self.clamp = kind, scale, eps, clamp

    def forward(self, x):
        if self.kind == "depth":
            return torch.exp(x.clamp(-self.clamp, self.clamp))
        if self.kind == "flow":
            return self.scale * torch.tanh(x)
        if self.kind == "pos":
            return F.softplus(x) + self.eps
        return torch.sigmoid(x.clamp(-15.0, 15.0))
