"""LLM 指导层(net/guidance/):语义奖励头 + 结构 schema。

对外接口(显式 re-export):
    GuidanceConfig, SemanticRewardHead, build_semantic_reward

运行时异步总线在 utils/guidance_bus.py(横向层);子目标文本编码用冻结句向量编码器
(train/minecraft/task_text.py,依赖注入进总线)。设计见 [knowledge/design_llm_deep_integration.md]。
"""
from net.guidance.config import GuidanceConfig
from net.guidance.heads import SemanticRewardHead, build_semantic_reward

__all__ = ["GuidanceConfig", "SemanticRewardHead", "build_semantic_reward"]
