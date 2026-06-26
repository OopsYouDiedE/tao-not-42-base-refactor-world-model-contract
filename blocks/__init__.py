"""基础 block 层。

`dreamer 系` — 原样 vendored 自 NM512/dreamerv3-torch 的可复用算子(MIT,见 blocks/NOTICE.dreamerv3):
    distributions(symlog/two-hot/one-hot 分布)、sequence(static_scan/λ-return)、
    conv(same-pad / 通道 LayerNorm)、dynamics.GRUCell(向量 GRU)。
"""
from blocks.encodings import PositionalEmbed, ContinuousTimeEncoding, SpatialPosEmbed, sinusoidal_time_encoding
from blocks.dynamics import (ConvGRUCell, GatedResidual, FiLM, Accumulator, DiscreteRouter,
                             GRUCell)
from blocks.regularization import StochLatent, SIGReg, BoundedActivation
from blocks.attention import (PreLNAttn, ProtoDecode, SlotCompetitiveAttn,
                              Mamba2Block, MHABlock)
from blocks.conv import Conv2dSamePad, ImgChLayerNorm
from blocks.encoder import ConvEncoder
from blocks.decoder import ConvDecoder
from blocks.mlp import MLP
from blocks.impala import FanInInitReLULayer, ImpalaCNN
from blocks.rl_heads import (FanInMLP, CategoricalActionHead, NormalizeEwma,
                             ScaledMSEHead)
from blocks.sequence import static_scan, static_scan_for_lambda_return, lambda_return
from blocks.quantization import VectorQuantizer
from blocks.distributions import (
    symlog, symexp, SampleDist, OneHotDist, DiscDist, MSEDist, SymlogDist,
    ContDist, Bernoulli, UnnormalizedHuber, SafeTruncatedNormal, TanhBijector,
)

__all__ = [
    "ConvGRUCell",
    "GatedResidual", "FiLM", "PreLNAttn", "PositionalEmbed", "ProtoDecode",
    "SlotCompetitiveAttn", "Mamba2Block", "MHABlock", "ConvEncoder", "ConvDecoder", "MLP",
    "FanInInitReLULayer", "ImpalaCNN", "FanInMLP", "CategoricalActionHead",
    "NormalizeEwma", "ScaledMSEHead",
    "StochLatent", "SIGReg", "BoundedActivation", "Accumulator", "DiscreteRouter",
    "ContinuousTimeEncoding", "SpatialPosEmbed", "sinusoidal_time_encoding",
    "VectorQuantizer",
    # dreamer 系 vendored 算子
    "GRUCell", "Conv2dSamePad", "ImgChLayerNorm",
    "static_scan", "static_scan_for_lambda_return", "lambda_return",
    "symlog", "symexp", "SampleDist", "OneHotDist", "DiscDist", "MSEDist",
    "SymlogDist", "ContDist", "Bernoulli", "UnnormalizedHuber",
    "SafeTruncatedNormal", "TanhBijector",
]


