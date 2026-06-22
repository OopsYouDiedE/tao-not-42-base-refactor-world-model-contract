"""Dreamer4 可扩展 Transformer 世界模型智能体 (net/dreamer4/)。

从 blocks 算子库组装的 Dreamer 4(2025,《Training Agents Inside of Scalable World Models》)
结构:连续潜 token tokenizer(ConvEncoder/Decoder + 可选 VQ)+ 因果时空 Transformer 动力学
(MHABlock 空间/时间注意 + 动作 AdaLN 调制)+ shortcut-forcing 流匹配速度头 + reward/cont/
actor/critic 头。**仅构建,不提供训练循环**(流匹配 + 想象 actor-critic 训练待补)。

对外接口:
    Dreamer4Config — 结构超参 dataclass。
    Tokenizer / SpaceTimeTransformer / ShortcutHead / WorldModel — 部件。
    Dreamer4 / build_dreamer4 — 智能体与工厂。
"""
from net.dreamer4.config import Dreamer4Config
from net.dreamer4.tokenizer import Tokenizer
from net.dreamer4.dynamics import SpaceTimeTransformer, ShortcutHead
from net.dreamer4.world_model import WorldModel
from net.dreamer4.agent import Dreamer4, build_dreamer4

__all__ = [
    "Dreamer4Config", "Tokenizer", "SpaceTimeTransformer", "ShortcutHead",
    "WorldModel", "Dreamer4", "build_dreamer4",
]
