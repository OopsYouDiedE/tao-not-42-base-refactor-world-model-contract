"""LLM 指导层结构超参 schema(纯 dataclass,无 IO)(net/guidance/config.py)。

对外接口:
    GuidanceConfig — 语义奖励头 / 目标条件化 / 指导总线节拍的全量结构超参。

设计见 [knowledge/design_llm_deep_integration.md]:LLM/VLM 以异步双系统形式嵌合,
下行只条件化行为层(actor/critic/语义奖励头),不条件化世界模型动力学。
"""
from dataclasses import dataclass


@dataclass
class GuidanceConfig:
    """LLM 指导层结构超参。

    Attributes:
        goal_text_dim:   子目标文本嵌入维(冻结 MiniLM = 384)。
        units:           语义奖励头 MLP 宽度。
        mlp_layers:      语义奖励头 MLP 隐藏层数。
        reward_bins:     语义奖励 two-hot symexp 离散分布桶数(对齐 DreamerV3 reward 头)。
        reward_coef:     语义 shaping 奖励混入想象回报的系数(0 = 通道置零,北极星防火墙档)。
        replan_period_s: VLM 异步重规划节拍(秒);仅供运行时参考,不进前向。
        stale_after_s:   LLM 计划陈旧阈值(秒),超时降级到静态计划。
    """
    goal_text_dim: int = 384
    units: int = 512
    mlp_layers: int = 2
    reward_bins: int = 255
    reward_coef: float = 1.0
    replan_period_s: float = 1.5
    stale_after_s: float = 8.0
