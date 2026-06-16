"""DreamerV3(PyTorch)— 原样 vendored 自 NM512/dreamerv3-torch(MIT,见 NOTICE)。

本子包是与 MinecraftWorldModel 并列的第二个世界模型,按本仓 net/blocks 分层物理拆分:
    底层算子(分布/扫描/GRU/conv)→ blocks/,初始化与训练胶水 → utils/nn,
    本包只留模型与装配:networks(RSSM/编解码/MLP)、models(WorldModel/ImagBehavior)、
    config(可加载配置 + build_dreamer)、_compat(原 tools 名字空间垫片)。

对外接口:
    build_dreamer / make_config / DREAMER_DEFAULTS — 加载与配置。
    WorldModel / ImagBehavior / RSSM — 模型部件。
"""
from net.dreamer.config import build_dreamer, make_config, DREAMER_DEFAULTS
from net.dreamer.models import WorldModel, ImagBehavior, RewardEMA
from net.dreamer.networks import RSSM, MultiEncoder, MultiDecoder, ConvEncoder, ConvDecoder, MLP

__all__ = [
    "build_dreamer", "make_config", "DREAMER_DEFAULTS",
    "WorldModel", "ImagBehavior", "RewardEMA",
    "RSSM", "MultiEncoder", "MultiDecoder", "ConvEncoder", "ConvDecoder", "MLP",
]
