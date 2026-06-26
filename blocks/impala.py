"""L1 primitive 积木库 - IMPALA 残差卷积编码器与 fan-in 初始化层 (blocks/impala.py)。

对外接口:
    FanInInitReLULayer — conv/linear + 可选归一化(GroupNorm/LayerNorm)+ ReLU,权重按
                         fan-in 范数缩放初始化(VPT/IMPALA 风格)。
    ImpalaCNN          — 多级 CnnDownStack 残差卷积塔 + dense 投影,task-agnostic 图像编码器。

来源:结构照搬 OpenAI VPT / snu-mllab Achievement-Distillation 的 IMPALA-CNN 实现。
归一化默认走 GroupNorm/LayerNorm(不用 BatchNorm),承袭仓内 I7 约定。
"""
import math
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FanInInitReLULayer(nn.Module):
    """归一化 + (conv/linear) + ReLU,权重按 fan-in L2 范数缩放初始化。

    归一化优先级:batch_norm > group_norm_groups > layer_norm(均默认关闭)。
    本仓约定不开 batch_norm(I7)。无归一化时该层带 bias,否则无 bias。

    Args:
        inchan/outchan: 输入/输出通道(或 linear 的 in/out features)。
        layer_type:     "conv" | "linear" | "conv3d"。
        init_scale:     权重缩放系数(按 fan-in 范数归一后乘此值)。
        group_norm_groups: GroupNorm 分组数;None 关闭。
        layer_norm:     是否用 LayerNorm(对 inchan)。
        use_activation: 是否在末尾施加 ReLU。

    Forward:
        x: (B, inchan, ...) 或 (B, inchan) → (B, outchan, ...) / (B, outchan), 同 dtype。
    """

    def __init__(
        self,
        inchan: int,
        outchan: int,
        layer_type: str = "conv",
        init_scale: float = 1.0,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs,
    ):
        super().__init__()

        # 归一化(I7: rollout 路径不用 BatchNorm;此处默认 GroupNorm/LayerNorm)
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, **layer_kwargs)
        self.use_activation = use_activation

        # fan-in 范数缩放初始化
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x


class CnnBasicBlock(nn.Module):
    """两层 3×3 conv 的残差块(通道不变)。"""

    def __init__(self, inchan: int, init_scale: float = 1.0, init_norm_kwargs: Dict = {}):
        super().__init__()
        s = math.sqrt(init_scale)
        self.conv0 = FanInInitReLULayer(
            inchan, inchan, kernel_size=3, padding=1, init_scale=s, **init_norm_kwargs
        )
        self.conv1 = FanInInitReLULayer(
            inchan, inchan, kernel_size=3, padding=1, init_scale=s, **init_norm_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv1(self.conv0(x))


class CnnDownStack(nn.Module):
    """一级下采样塔:firstconv → maxpool(/2) → 若干 CnnBasicBlock。"""

    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1.0,
        pool: bool = True,
        post_pool_groups: Optional[int] = None,
        init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool

        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan, outchan, kernel_size=3, padding=1, **first_conv_init_kwargs
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(
                    outchan,
                    init_scale=init_scale / math.sqrt(nblock),
                    init_norm_kwargs=init_norm_kwargs,
                    **kwargs,
                )
                for _ in range(nblock)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape: Sequence[int]) -> Tuple[int, int, int]:
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        return (self.outchan, h, w)


class ImpalaCNN(nn.Module):
    """IMPALA 残差卷积编码器:多级 CnnDownStack + dense 投影。

    Args:
        inshape:  输入图像形状 (C, H, W)。
        chans:    各级输出通道,长度 = 下采样级数(每级空间 /2)。
        outsize:  dense 投影输出维度。
        nblock:   每级残差块数。
        init_norm_kwargs:       卷积层归一化配置(本仓 batch_norm=False, group_norm_groups=1)。
        dense_init_norm_kwargs: dense 层归一化配置(本仓 layer_norm=True)。

    Forward:
        x: (B, C, H, W) float32 [0,1] → (B, outsize) float32。
    """

    def __init__(
        self,
        inshape: Sequence[int],
        chans: Sequence[int],
        outsize: int,
        nblock: int,
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        curshape = inshape
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=1.0 / math.sqrt(len(chans)),
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = FanInInitReLULayer(
            math.prod(curshape),
            outsize,
            layer_type="linear",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stack in self.stacks:
            x = stack(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense(x)
        return x
