"""YOLO-World-Dreamer:DreamerV3 世界模型基座 + YOLO 双头 + YOLOE 文本点乘的目标条件规划器。

对外接口:
    YoloWorldConfig   — 结构超参 schema(纯 dataclass)。
    YoloWorld         — 智能体(世界模型 + 256 候选小头 + 双头行为线)。
    build_yoloworld   — 工厂构造。
    WorldModel / ProposalHead / DualHeadBehavior — 部件(测试/装配用)。

设计见 knowledge/yoloworld.md。
"""
from net.yoloworld.config import YoloWorldConfig
from net.yoloworld.world_model import WorldModel
from net.yoloworld.heads import ProposalHead, select_score
from net.yoloworld.behavior import DualHeadBehavior, Critic
from net.yoloworld.agent import YoloWorld, build_yoloworld

__all__ = [
    "YoloWorldConfig", "WorldModel", "ProposalHead", "select_score",
    "DualHeadBehavior", "Critic", "YoloWorld", "build_yoloworld",
]
