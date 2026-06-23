"""DreamerV3 世界模型智能体 (net/dreamerv3/)。

从 blocks 算子库清白重建的 DreamerV3(Hafner 等 arXiv:2301.04104):
离散 32×32 随机隐变量 RSSM + 卷积编/解码 + two-hot symexp 奖励/价值 + 想象 actor-critic。
非 vendored,**全部部件由 blocks 组装**(GRUCell/ConvEncoder/ConvDecoder/OneHotDist/
DiscDist/Bernoulli/MSEDist/lambda_return/static_scan/MLP)。设计见 knowledge/dreamer.md。

对外接口:
    DreamerV3Config — 结构超参 dataclass。
    RSSM / WorldModel / ImagBehavior — 部件。
    DreamerV3 / build_dreamerv3 — 智能体与工厂。
"""
from net.dreamerv3.config import DreamerV3Config
from net.dreamerv3.rssm import RSSM
from net.dreamerv3.world_model import WorldModel
from net.dreamerv3.behavior import ImagBehavior
from net.dreamerv3.agent import DreamerV3, build_dreamerv3
from net.dreamerv3.planner import Planner

__all__ = [
    "DreamerV3Config", "RSSM", "WorldModel", "ImagBehavior",
    "DreamerV3", "build_dreamerv3", "Planner",
]
