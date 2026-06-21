"""L1 primitive 积木库 - 动力学与门控更新 (blocks/dynamics.py)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_FP16 = 1e-4


class ConvGRUCell(nn.Module):
    """ConvGRU 单元。凸更新 (1-z)h+zn ⇒ 非扩张。"""

    def __init__(self, c):
        super().__init__()
        self.conv_z = nn.Conv2d(2 * c, c, 3, padding=1)
        self.conv_r = nn.Conv2d(2 * c, c, 3, padding=1)
        self.conv_n = nn.Conv2d(2 * c, c, 3, padding=1)

    def forward(self, x, h):
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * n


class GatedResidual(nn.Module):
    """唯一允许的残差形式: x + γ.clamp(±gmax)·f(x)。

    ⚠️ γ 受限是**经验性稳定化**,不是非扩张定理:严格 I5 需 |γ|·Lip(f) ≤ 1,
    而 f(无谱归一化的 MLP/注意力)的 Lipschitz 常数无界。另注意:严格非扩张的
    递归映射是收缩映射 ⇒ 指数遗忘,会杀长期记忆——"稳定但不严格收缩"才是
    工程上正确的点位,勿为追求 I5 字面成立而加谱归一化。
    """

    def __init__(self, submodule, gmax=0.5, init=0.1):
        super().__init__()
        self.f = submodule
        self.gmax = gmax
        self.gamma = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x, *args, **kwargs):
        g = self.gamma.clamp(min=-self.gmax, max=self.gmax)
        return x + g * self.f(x, *args, **kwargs)


class FiLM(nn.Module):
    """条件仿射调制 x*(1+γ)+β。末层零初始 ⇒ 冷启动恒等。"""

    def __init__(self, cond_dim, c, hidden=None):
        super().__init__()
        hidden = hidden or max(cond_dim * 4, c * 2)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * c))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        params = self.mlp(cond)
        while params.dim() < x.dim():
            params = params.unsqueeze(-1)
        gamma, beta = params.chunk(2, dim=1)
        return x * (1 + gamma) + beta


class Accumulator(nn.Module):
    """Accumulator, 精确累加。reg = reg + x @ W"""
    def __init__(self, in_dim, d):
        super().__init__()
        self.W_hat = nn.Parameter(torch.randn(in_dim, d) * 0.1)
        self.M_hat = nn.Parameter(torch.randn(in_dim, d) * 0.1)

    def weight(self):
        return torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)   # ∈(-1,1)

    def forward(self, reg, x):
        return reg + x @ self.weight()


class DiscreteRouter(nn.Module):
    """对 K 分支硬选择,Gumbel-softmax 直通可微。"""
    def __init__(self, in_dim, K, tau=1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, K)
        self.tau = tau

    def forward(self, h, hard=True):
        logits = self.proj(h).float()                           # fp32 I4
        if self.training:
            y = F.gumbel_softmax(logits, tau=self.tau, hard=hard)
        else:
            idx = logits.argmax(-1)
            y = F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
        return y, logits.argmax(-1)


# DreamerV3 的向量 GRU 单元(LayerNorm + 凸更新):与 ConvGRUCell 同属 I5/I7 安全递归算子
# ——LayerNorm(I7)+ 凸组合 update*cand+(1-update)*state(I5 非扩张),RSSM 的确定性状态 deter
# 由它递推;update_bias=-1 让初始更新门偏向"保持旧状态",state 以单元素列表传入(承袭 Keras 接口)。
# 原样照抄 NM512/dreamerv3-torch 的 networks.py(类体逐字 1:1,见 blocks/NOTICE.dreamerv3)。
class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
