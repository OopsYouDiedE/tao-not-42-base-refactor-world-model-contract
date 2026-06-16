"""DreamerV3 的可加载配置与装配入口(摆脱对原仓 argparse/yaml/gym 的依赖)。

对外接口:
    DREAMER_DEFAULTS    — 原仓 configs.yaml `defaults` 段的结构超参(逐字段照搬,纯 Python dict)。
    make_config(...)     — 由默认值 + 覆盖项构造一个 SimpleNamespace,字段与原仓 config 对象一致。
    build_dreamer(...)   — "完整加载":构造 WorldModel + ImagBehavior 并返回容器(= 完整 DreamerV3 模型)。

原仓用 argparse 解析 configs.yaml 成 Namespace、用 gym spaces 描述观测空间。本仓 net/ 不读文件、
不引 gym;故此处把 `defaults` 段的数值原样落成 Python dict(模型加载后逐位等价于原仓默认配置),
并提供最小 obs-space 垫片。模型逻辑(networks/models)保持 1:1 原样,只是这里替掉外围装配胶水。
"""
import copy
from types import SimpleNamespace

from net.dreamer.models import WorldModel, ImagBehavior


# 原仓 configs.yaml `defaults` 段中模型构造所需的字段(数值逐字段照搬)。
DREAMER_DEFAULTS = dict(
    precision=32,
    # World Model — RSSM
    dyn_hidden=512,
    dyn_deter=512,
    dyn_stoch=32,
    dyn_discrete=32,
    dyn_rec_depth=1,
    dyn_mean_act="none",
    dyn_std_act="sigmoid2",
    dyn_min_std=0.1,
    grad_heads=["decoder", "reward", "cont"],
    units=512,
    act="SiLU",
    norm=True,
    encoder=dict(
        mlp_keys="$^", cnn_keys="image", act="SiLU", norm=True, cnn_depth=32,
        kernel_size=4, minres=4, mlp_layers=5, mlp_units=1024, symlog_inputs=True,
    ),
    decoder=dict(
        mlp_keys="$^", cnn_keys="image", act="SiLU", norm=True, cnn_depth=32,
        kernel_size=4, minres=4, mlp_layers=5, mlp_units=1024, cnn_sigmoid=False,
        image_dist="mse", vector_dist="symlog_mse", outscale=1.0,
    ),
    actor=dict(
        layers=2, dist="normal", entropy=3e-4, unimix_ratio=0.01, std="learned",
        min_std=0.1, max_std=1.0, temp=0.1, lr=3e-5, eps=1e-5, grad_clip=100.0,
        outscale=1.0,
    ),
    critic=dict(
        layers=2, dist="symlog_disc", slow_target=True, slow_target_update=1,
        slow_target_fraction=0.02, lr=3e-5, eps=1e-5, grad_clip=100.0, outscale=0.0,
    ),
    reward_head=dict(layers=2, dist="symlog_disc", loss_scale=1.0, outscale=0.0),
    cont_head=dict(layers=2, loss_scale=1.0, outscale=1.0),
    dyn_scale=0.5,
    rep_scale=0.1,
    kl_free=1.0,
    weight_decay=0.0,
    unimix_ratio=0.01,
    initial="learned",
    # Training (优化器构造需要)
    model_lr=1e-4,
    opt_eps=1e-8,
    grad_clip=1000,
    opt="adam",
    # Behavior
    discount=0.997,
    discount_lambda=0.95,
    imag_horizon=15,
    imag_gradient="dynamics",
    imag_gradient_mix=0.0,
    reward_EMA=True,
)

# 字典型字段(覆盖时按 key 深合并,而非整体替换)。
_DICT_FIELDS = ("encoder", "decoder", "actor", "critic", "reward_head", "cont_head")


class _Box:
    """gym.spaces.Box 的最小垫片:只暴露 `.shape`(MultiEncoder/MultiDecoder 仅读 shape)。"""

    def __init__(self, shape):
        self.shape = tuple(shape)


class _ObsSpace:
    """gym.spaces.Dict 的最小垫片:`.spaces` 为 {名称: _Box}。

    shapes: {名称: 形状元组}。图像键约定形如 (H, W, C)(与原仓 cnn_keys/HWC 约定一致)。
    """

    def __init__(self, shapes):
        self.spaces = {k: _Box(v) for k, v in shapes.items()}


def make_config(num_actions, device="cpu", **overrides):
    """构造与原仓 config 对象同构的 SimpleNamespace。

    num_actions: 动作维度(env 相关,无默认,必传)。
    device: 'cpu' / 'cuda:0' 等;原仓默认 'cuda:0',本仓默认 'cpu' 以便离线加载/测试。
    overrides: 覆盖任意默认字段;字典型字段(encoder/decoder/actor/critic/reward_head/cont_head)
               按 key 深合并(只改给定子键,其余保持默认)。
    """
    cfg = copy.deepcopy(DREAMER_DEFAULTS)
    for k, v in overrides.items():
        if k in _DICT_FIELDS and isinstance(v, dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    cfg["num_actions"] = num_actions
    cfg["device"] = device
    return SimpleNamespace(**cfg)


def build_dreamer(num_actions, obs_shapes=None, device="cpu", **overrides):
    """完整加载 DreamerV3:WorldModel(编码+RSSM+解码+奖励/继续头)+ ImagBehavior(actor-critic)。

    num_actions: 动作维度。
    obs_shapes:  观测空间 {名称: 形状};默认单图像观测 {"image": (64, 64, 3)}(DreamerV3 视觉任务标准 64×64)。
    device:      构造设备(默认 'cpu')。
    overrides:   透传 make_config 的结构超参覆盖。
    返回:SimpleNamespace(config, obs_space, world_model, behavior)。
         world_model 与 behavior 即完整 DreamerV3 模型,可直接前向/训练(见 tests/unit/test_dreamer.py)。
    """
    obs_shapes = obs_shapes or {"image": (64, 64, 3)}
    config = make_config(num_actions, device=device, **overrides)
    obs_space = _ObsSpace(obs_shapes)
    world_model = WorldModel(obs_space, None, 0, config).to(device)
    behavior = ImagBehavior(config, world_model).to(device)
    return SimpleNamespace(
        config=config,
        obs_space=obs_space,
        world_model=world_model,
        behavior=behavior,
    )
