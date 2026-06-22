"""L1 primitive 积木库 - 多层感知机 (blocks/mlp.py)。

对外接口:
    MLP — Linear→[LayerNorm]→激活 的堆叠,可选 symlog 输入预处理。

与任务无关的可复用前馈算子:DreamerV3 的 RSSM 内部投影、reward/cont/value/actor 头,
以及 Dreamer4 的各预测头都由它组装。输出**原始 logits/特征**(无分布封装),
概率/分布参数化(DiscDist/Bernoulli/OneHotDist)由 net/ 在外部组合(职责分离,见 blocks/distributions.py)。
所有宽度/层数/激活/归一化经构造参数注入,**不写死**。
"""
import torch.nn as nn

from blocks.distributions import symlog


class MLP(nn.Module):
    """多层感知机:[Linear → LayerNorm → act] × layers → Linear(out)。

    隐藏层用 LayerNorm(I7:递归/rollout 路径不用 BatchNorm);输出层为裸 Linear
    (无归一化/激活,值域由下游分布封装约束)。layers=0 时退化为单个 Linear(in→out)。

    Args:
        in_dim: 输入特征维。
        out_dim: 输出维(分布参数个数,如 reward 头的 255、actor 头的动作数)。
        hidden: 隐藏层宽度。
        layers: 隐藏层数(每层 Linear+Norm+act);0 = 纯线性映射。
        act: 激活层**类**(可调用,无参构造),默认 nn.SiLU。
        norm: 隐藏层是否插入 LayerNorm(I7)。
        symlog_input: True 则前向先对输入做 symlog 压缩(重尾标量输入用)。

    Forward:
        x: [..., in_dim] → [..., out_dim],dtype 随输入。
    """

    def __init__(self, in_dim, out_dim, hidden=512, layers=2, act=nn.SiLU,
                 norm=True, symlog_input=False):
        super().__init__()
        self.symlog_input = symlog_input
        mods = []
        c = in_dim
        for _ in range(layers):
            mods.append(nn.Linear(c, hidden, bias=not norm))
            if norm:
                mods.append(nn.LayerNorm(hidden, eps=1e-3))
            mods.append(act())
            c = hidden
        mods.append(nn.Linear(c, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        if self.symlog_input:
            x = symlog(x)
        return self.net(x)
