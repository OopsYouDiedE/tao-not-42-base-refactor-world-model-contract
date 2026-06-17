"""世界模型的类型化结构配置(纯 dataclass,无 IO)。

对外接口:
    BackboneConfig / EncoderConfig / DynamicsConfig / HeadsConfig / XiConfig — 各部件超参。
    ModelConfig — 顶层装配配置;ModelConfig.from_dict(plain_dict) 从 yaml 解析出的 dict 构造
                  (缺键取默认 = 今日写死值,未知键报错)。

net 只持有 schema,不读 yaml、不做文件 IO(yaml 读取在 utils.config_io,装配/领域校验在 train)。
"""
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class BackboneConfig:
    """冻结视觉骨干选择。测试 mock 骨干走依赖注入(AGENTS §2),不在此处。"""
    kind: str = "dinov3"            # dinov3=ViT-S/16(默认,gated)| dinov2=ViT-S/14(开放,备选)
    weights: Optional[str] = None  # 覆盖 HF repo id(换更大变体);None=用 kind 默认 repo


@dataclass
class DynamicsConfig:
    """动力学核(token 序列内推演)。"""
    kind: str = "transformer"
    num_layers: int = 4
    nhead: int = 8                 # 须整除 d
    ffn_mult: int = 4              # dim_feedforward = d * ffn_mult
    dropout: float = 0.0           # 0:mu 直喂回归损失,train/eval 前向须一致;正则交 SIGReg


@dataclass
class HeadsConfig:
    """解码头选项。"""
    n_cam_bins: int = 11           # 必须 == domains.minecraft.vpt_action.CAMERA_BINS(训练端断言)


@dataclass
class ModelConfig:
    """MinecraftWorldModel 的顶层结构配置。"""
    d: int = 384
    K: int = 5                     # 动作查询数
    J: int = 8                     # 历史动作长度
    act_dim: int = 22              # 必须 == domains.minecraft.vpt_action.ACTION_DIM(训练端断言)
    max_skip: int = 8              # 区间动作序列上限(= 数据集 frame_skip 上限,装配时由 train 注入)
    state_dec_mult: int = 2        # state_dec 隐藏维 = d * state_dec_mult
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)

    _SUB = {"backbone": BackboneConfig,
            "dynamics": DynamicsConfig, "heads": HeadsConfig}

    @classmethod
    def from_dict(cls, d):
        """从 plain dict(yaml 解析结果)构造,缺键取默认,未知键报错(防 yaml 拼写漏配)。"""
        d = dict(d or {})
        top = {f.name for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k in cls._SUB:
                kw[k] = _sub_from_dict(cls._SUB[k], v)
            elif k in top:
                kw[k] = v
            else:
                raise ValueError(f"ModelConfig 未知字段: {k}")
        return cls(**kw)


def _sub_from_dict(sub_cls, d):
    d = dict(d or {})
    valid = {f.name for f in fields(sub_cls)}
    bad = set(d) - valid
    if bad:
        raise ValueError(f"{sub_cls.__name__} 未知字段: {sorted(bad)}")
    return sub_cls(**d)
