"""VPT 行为克隆网络模块 (net/bc/)。

对外接口:
    BCPolicy / build_bc_policy — 冻结骨干 + 因果时序 Transformer + 相机/按键头。
    BCConfig — 结构配置 dataclass。
"""
from net.bc.config import BCConfig
from net.bc.policy import BCPolicy, CondPolicy, TextCondPolicy, build_bc_policy

__all__ = ["BCConfig", "BCPolicy", "CondPolicy", "TextCondPolicy", "build_bc_policy"]
