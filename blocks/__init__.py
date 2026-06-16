"""基础 block 层。

`primitives` — 17 个 L1 primitive(我们的,数值不变量 I1–I8 焊进实现)。
`yolo` — 1:1 复刻官方 YOLO 的底层通用算子。
`dreamer 系` — 原样 vendored 自 NM512/dreamerv3-torch 的可复用算子(MIT,见 net/dreamer/NOTICE):
    distributions(symlog/two-hot/one-hot 分布)、sequence(static_scan/λ-return)、
    conv(same-pad / 通道 LayerNorm)、dynamics.GRUCell(向量 GRU)。
"""
from blocks.spatial import Warp, GlobalTransformApply, BEVSplat, rot6d_to_matrix, make_4x4
from blocks.similarity import LocalCorr, SoftArgmaxFlow, box_iou
from blocks.encodings import PositionalEmbed, ContinuousTimeEncoding, SpatialPosEmbed, sinusoidal_time_encoding
from blocks.dynamics import (ConvGRUCell, GatedResidual, FiLM, Accumulator, DiscreteRouter,
                             GRUCell)
from blocks.regularization import StochLatent, SIGReg, BoundedActivation
from blocks.attention import PreLNAttn, ProtoDecode, SlotCompetitiveAttn
from blocks.conv import Conv2dSamePad, ImgChLayerNorm
from blocks.sequence import static_scan, static_scan_for_lambda_return, lambda_return
from blocks.distributions import (
    symlog, symexp, SampleDist, OneHotDist, DiscDist, MSEDist, SymlogDist,
    ContDist, Bernoulli, UnnormalizedHuber, SafeTruncatedNormal, TanhBijector,
)

__all__ = [
    "Warp", "GlobalTransformApply", "LocalCorr", "SoftArgmaxFlow", "ConvGRUCell",
    "GatedResidual", "FiLM", "PreLNAttn", "PositionalEmbed", "ProtoDecode",
    "SlotCompetitiveAttn",
    "StochLatent", "SIGReg", "BoundedActivation", "Accumulator", "DiscreteRouter",
    "BEVSplat", "ContinuousTimeEncoding", "SpatialPosEmbed", "sinusoidal_time_encoding",
    "rot6d_to_matrix", "make_4x4", "box_iou",
    # dreamer 系 vendored 算子
    "GRUCell", "Conv2dSamePad", "ImgChLayerNorm",
    "static_scan", "static_scan_for_lambda_return", "lambda_return",
    "symlog", "symexp", "SampleDist", "OneHotDist", "DiscDist", "MSEDist",
    "SymlogDist", "ContDist", "Bernoulli", "UnnormalizedHuber",
    "SafeTruncatedNormal", "TanhBijector",
]

