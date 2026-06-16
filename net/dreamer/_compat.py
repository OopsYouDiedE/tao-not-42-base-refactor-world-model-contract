"""vendored Dreamer 代码的兼容垫片(原 tools.py 名字空间的重聚合)。

DreamerV3 的可复用算子已按本仓分层物理拆出:概率分布/符号对数→`blocks.distributions`、
序列扫描/λ-return→`blocks.sequence`、初始化与训练胶水→`utils.nn`。本模块把这些名字
重新聚合到一个命名空间,使 `networks.py` / `models.py` 仍以 `tools.X` 调用、函数体保持
1:1 原样(`from net.dreamer import _compat as tools`)。本文件只做 re-export,无逻辑。

对外接口(= 原 tools.py 中被模型代码引用的子集):
    symlog / symexp、static_scan / lambda_return、weight_init / uniform_weight_init、
    to_np / tensorstats、RequiresGrad / Optimizer、以及各 Dist 包装类。
"""
from blocks.distributions import (
    symlog,
    symexp,
    SampleDist,
    OneHotDist,
    DiscDist,
    MSEDist,
    SymlogDist,
    ContDist,
    Bernoulli,
    UnnormalizedHuber,
    SafeTruncatedNormal,
    TanhBijector,
)
from blocks.sequence import static_scan, static_scan_for_lambda_return, lambda_return
from utils.nn import (
    to_np,
    tensorstats,
    weight_init,
    uniform_weight_init,
    RequiresGrad,
    Optimizer,
)
