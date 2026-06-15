"""动力学核:对拼好的 token 序列做一次序列内推演。

对外接口:
    build_dynamics(cfg, d) — 按 DynamicsConfig 造动力学核(nn.Module);
        契约 forward(X:[B,L,d] float) -> [B,L,d] float。换 attention 变体在此扩 kind。
"""
import torch.nn as nn

from net.config import DynamicsConfig


def build_dynamics(cfg: DynamicsConfig, d: int) -> nn.Module:
    """按配置构造动力学核。

    kind="transformer": nn.TransformerEncoder(num_layers 层,nhead 头,
        dim_feedforward=d*ffn_mult,gelu,dropout);batch_first ⇒ 输入 [B,L,d]。
    dropout 默认 0:mu 直喂回归损失,train/eval 前向须一致(正则交 SIGReg,见 knowledge)。
    """
    if cfg.kind == "transformer":
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=d * cfg.ffn_mult,
            batch_first=True, activation="gelu", dropout=cfg.dropout)
        return nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
    raise ValueError(f"未知 dynamics kind: {cfg.kind}")
