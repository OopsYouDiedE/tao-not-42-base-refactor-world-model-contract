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
    """动力学核 = 序列↔序列预测器 P_ψ 的 Transformer 主干(对拼好的时空 token 集合做一次推演)。"""
    kind: str = "transformer"
    num_layers: int = 4
    nhead: int = 8                 # 须整除 d
    ffn_mult: int = 4              # dim_feedforward = d * ffn_mult
    dropout: float = 0.0           # 0:mu 直喂回归损失,train/eval 前向须一致;正则交 SIGReg


@dataclass
class AdapterConfig:
    """冻结骨干之上的可训练编码 adapter(patch 自注意 + 因子化潜)。"""
    num_layers: int = 2            # PreLNAttn + MLP 残差块层数
    nhead: int = 8                 # 须整除 d
    ffn_mult: int = 4              # adapter 内 MLP 隐藏维 = d * ffn_mult
    z_inv_kind: str = "gaussian"   # z_inv 随机潜种类:gaussian|categorical(StochLatent)
    beta_kl: float = 1e-3          # z_inv 信息瓶颈 KL 权重(太大会丢掉小而关键的位)
    ema_decay: float = 0.996       # EMA 教师跟随 online adapter 的衰减(JEPA 目标,I8)


@dataclass
class EffectConfig:
    """效应词表 𝔤⊕𝒟:𝒟=对潜变化 Δz_inv 量化的事件码本;𝔤=可逆生成元算子组。"""
    event_vocab_size: int = 64     # 𝒟 事件码本大小(替换原写死的动作 VQ-512)
    n_generators: int = 8          # 𝔤 可逆生成元算子数


@dataclass
class PredictorConfig:
    """序列↔序列对齐采样与目标超参(数学 (2)/(4))。"""
    horizon: int = 6               # 未来 query 相对最大上下文截止的最大跨度(帧)
    n_context_cutoffs: int = 2     # 每个 target 采样的上下文截止个数 |𝒦|(≥2 才有一致性约束)
    lambda_agree: float = 1.0      # 多上下文一致性损失权重(数学 (4) 第二项)


@dataclass
class HeadsConfig:
    """解码头选项。"""
    n_cam_bins: int = 11           # 必须 == domains.minecraft.vpt_action.CAMERA_BINS(训练端断言)


@dataclass
class RSSMConfig:
    """帧级 RSSM + 后继特征世界模型的结构配置(net/rssm.py;独立实验路径,不入 ModelConfig)。

    设计与数学见 knowledge/rssm_sf_design.md。stoch = d_rev(高斯) + inv_groups*inv_classes(离散);
    feat = deter + stoch。free_nats/dyn_scale/rep_scale 为 free-bits balanced KL 三参。
    """
    embed_dim: int = 384           # 冻结骨干池化嵌入维(grounding 目标维;DINO ViT-S=384)
    act_dim: int = 22              # 动作维(== vpt_action.ACTION_DIM)
    deter: int = 256               # 确定递归态 h 维(GRU 隐藏)
    hidden: int = 256              # 先验/后验/各头 MLP 隐藏维
    d_rev: int = 32                # 可逆连续随机态维(高斯,可逆相机/平移)
    inv_groups: int = 8            # 不可逆离散随机态组数(事件)
    inv_classes: int = 8           # 每组类别数
    min_std: float = 0.1           # 高斯 z_rev 标准差下界(I1/I3)
    unimix: float = 0.01           # categorical 均匀混合比(防死类)
    free_nats: float = 1.0         # free bits:每序列步 KL 免费额度(防 posterior collapse)
    dyn_scale: float = 0.5         # KL 先验侧(dyn)权重(> rep_scale ⇒ 先验向后验靠)
    rep_scale: float = 0.1         # KL 后验侧(rep)权重
    sf_dim: int = 1                # 后继特征维 = φ 个数(切片=1:has_item)
    sf_hidden: int = 256           # 后继特征头 MLP 隐藏维

    @property
    def stoch_dim(self) -> int:
        return self.d_rev + self.inv_groups * self.inv_classes

    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch_dim


@dataclass
class ModelConfig:
    """MinecraftWorldModel 的顶层结构配置。"""
    d: int = 384
    d_rev: int = 256               # 可逆连续潜维(相机/平移);d_rev + d_inv == d
    d_inv: int = 128               # 不可逆离散潜维(事件/持物);后果与路径无关性落在此
    K: int = 5                     # 动作查询数
    J: int = 8                     # 历史动作长度
    act_dim: int = 22              # 必须 == domains.minecraft.vpt_action.ACTION_DIM(训练端断言)
    max_skip: int = 8              # 区间动作序列上限(= 数据集 frame_skip 上限,装配时由 train 注入)
    state_dec_mult: int = 2        # state_dec 隐藏维 = d * state_dec_mult
    unfreeze_backbone_layers: int = 0  # >0 时解冻 backbone 顶部 N 层(探针失败时的升级逃生口,默认关)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    effect: EffectConfig = field(default_factory=EffectConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)

    _SUB = {"backbone": BackboneConfig, "adapter": AdapterConfig,
            "dynamics": DynamicsConfig, "effect": EffectConfig,
            "predictor": PredictorConfig, "heads": HeadsConfig}

    def __post_init__(self):
        if self.d_rev + self.d_inv != self.d:
            raise ValueError(
                f"d_rev({self.d_rev}) + d_inv({self.d_inv}) 必须 == d({self.d})")

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
