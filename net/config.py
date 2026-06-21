"""世界模型基座的类型化结构配置(纯 dataclass,无 IO)。

net 只持有 schema,不读 yaml、不做文件 IO(yaml 读取在 utils,装配/领域校验在 train)。

注:统一世界基座清白重设计期间,旧 Δz-JEPA 的 ModelConfig 与 RSSM 切片的 RSSMConfig
及其全部子配置(Dynamics/Adapter/Effect/Predictor/Heads)已删除(见 git 历史)。本文件
当前仅保留与冻结视觉骨干相关的 BackboneConfig;新基座的 FoundationConfig schema 待构建期补入。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackboneConfig:
    """冻结视觉骨干选择。测试 mock 骨干走依赖注入(AGENTS §2),不在此处。"""
    kind: str = "dinov3"            # dinov3=ViT-S/16(默认,gated)| dinov2=ViT-S/14(开放,备选)
    weights: Optional[str] = None  # 覆盖 HF repo id(换更大变体);None=用 kind 默认 repo
