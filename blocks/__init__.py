"""基础 block 层。

`primitives` — 17 个 L1 primitive(我们的,数值不变量 I1–I8 焊进实现)。
`yolo` — 1:1 复刻官方 YOLO 的底层通用算子。
"""
from blocks.primitives import (
    Warp, GlobalTransformApply, LocalCorr, SoftArgmaxFlow, ConvGRUCell,
    GatedResidual, FiLM, PreLNAttn, PositionalEmbed, ProtoDecode, StochLatent,
    SIGReg, BoundedActivation, Accumulator, DiscreteRouter, BEVSplat,
    ContinuousTimeEncoding, SpatialPosEmbed, rot6d_to_matrix, make_4x4, box_iou,
)

__all__ = [
    "Warp", "GlobalTransformApply", "LocalCorr", "SoftArgmaxFlow", "ConvGRUCell",
    "GatedResidual", "FiLM", "PreLNAttn", "PositionalEmbed", "ProtoDecode",
    "StochLatent", "SIGReg", "BoundedActivation", "Accumulator", "DiscreteRouter",
    "BEVSplat", "ContinuousTimeEncoding", "SpatialPosEmbed",
    "rot6d_to_matrix", "make_4x4", "box_iou",
]
