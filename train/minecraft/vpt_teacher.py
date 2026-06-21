"""minerl-free VPT teacher:加载 OpenAI VPT(.model+.weights),前向出动作分布,
**边缘化到我们的 22 维契约**(相机 dx/dy 各 11-bin 分布 + 20 键 Bernoulli 概率)。

只依赖 vendored [net/vpt_lib](net/vpt_lib)(已去 minerl/gym3/attr);用于把 rl-from-foundation-2x
当 teacher 蒸馏进世界模型(soft 分布 KL)。

动作空间对齐(见 net/vpt_lib/action_mapping.py 实测):
  - VPT `buttons` = 8641 类联合分类(8 个互斥组 × camera-meta 的笛卡尔积 + inventory);
    每个组合激活哪些 minerl 按钮由 `BUTTON_IDX_TO_FACTORED` [8641,20] 预计算 ⇒
    逐键边缘 P(key)= softmax(buttons) @ BUTTON_IDX_TO_FACTORED,再按我们键序重排。
    attack/use 是其中两个按钮 ⇒ 蒸 VPT 自动修复 F1(模型终于看得到"挖/放")。
  - VPT `camera` = 121 类(11×11),mu-law(maxval=10, mu=10),与我们 camera_to_bin 同分箱 ⇒
    bin 对 bin 直接对齐。按 camera-meta on/off 门控折算等效分布,再边缘化成 dx[11]/dy[11]。

⚠ 单帧 recurrent:VPT 策略带 transformer 记忆,必须**按帧顺序**调用 step();新 episode 前 reset()。
⚠ 相机轴序(yaw/pitch ↔ 我们的 dx/dy)以 VPT 内部约定为准,蒸进新头时自洽;若日后与
  数据集监督头混用,需核对是否需要 x/y 对调(见 colab-camera-units 备忘)。

用法:
  # 真权重(用户环境):
  python train/minecraft/vpt_teacher.py --model /path/rl-from-foundation-2x.model --weights /path/....weights
  # 无权重冒烟(验证 minerl-free 装配 + 边缘化适配器,用小随机网络):
  python train/minecraft/vpt_teacher.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402

from net.vpt_lib.policy import MinecraftAgentPolicy           # noqa: E402
from net.vpt_lib.action_mapping import CameraHierarchicalMapping  # noqa: E402
from net.vpt_lib.actions import Buttons                       # noqa: E402
from net.vpt_lib.gym3_types import DictType                   # noqa: E402

AGENT_RESOLUTION = (128, 128)

# 我们的 22 维契约键序(与 train/minecraft/vpt_action.VPT_KEYS 严格一致)
OUR_KEYS = (["key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak", "key_sprint",
             "key_attack", "key_use", "key_drop", "key_inventory"]
            + [f"key_hotbar.{i}" for i in range(1, 10)])
# 我们的键名 → VPT Buttons.ALL 里的 minerl 按钮名
_OUR_TO_MINERL = {
    "key_w": "forward", "key_a": "left", "key_s": "back", "key_d": "right", "key_space": "jump",
    "key_sneak": "sneak", "key_sprint": "sprint", "key_attack": "attack", "key_use": "use",
    "key_drop": "drop", "key_inventory": "inventory",
    **{f"key_hotbar.{i}": f"hotbar.{i}" for i in range(1, 10)},
}

# 无权重冒烟用的小配置(只验证装配/适配器,随机权重 ⇒ 形状无需与真模型一致,故缩小以提速)
SMOKE_POLICY_KWARGS = dict(
    # maxlen = memory_size - timesteps,须 >0(见 masked_attention.MaskedAttention)
    attention_heads=4, attention_mask_style="clipped_causal", attention_memory_size=32,
    diff_mlp_embedding=False, hidsize=128, img_shape=[128, 128, 3], impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1}, impala_width=1,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=2, only_img_input=True, pointwise_ratio=4, pointwise_use_activation=False,
    recurrence_is_residual=True, recurrence_type="transformer", timesteps=16,
    use_pointwise_layer=True, use_pre_lstm_ln=False,
)
SMOKE_PI_HEAD_KWARGS = dict(temperature=2.0)


def _load_model_pickle(model_path):
    """从 .model pickle 取 policy_kwargs / pi_head_kwargs(同 run_agent.py 的解包路径)。"""
    import pickle
    p = pickle.load(open(model_path, "rb"))
    policy_kwargs = p["model"]["args"]["net"]["args"]
    pi_head_kwargs = p["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def build_policy(policy_kwargs, pi_head_kwargs, device="cpu"):
    """造 MinecraftAgentPolicy(动作空间固定为 CameraHierarchicalMapping(11))。"""
    # VPT 的 recurrent initial_state 用 torch_util.DEFAULT_DEVICE(默认探测到 cuda)建零张量;
    # 与我们传入的 device 同步,否则 state(cuda)与输入(cpu)设备不一致。
    from net.vpt_lib.torch_util import set_default_torch_device
    set_default_torch_device(torch.device(device))
    mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_space = DictType(**mapper.get_action_space_update())
    policy = MinecraftAgentPolicy(action_space=action_space, policy_kwargs=policy_kwargs,
                                  pi_head_kwargs=pi_head_kwargs)
    return policy.to(device), mapper


class VPTTeacher:
    """加载好的 VPT 策略 + 到我们契约的边缘化适配器。"""

    def __init__(self, model_path=None, weights_path=None, device="cpu"):
        self.device = device
        if model_path is not None:
            policy_kwargs, pi_head_kwargs = _load_model_pickle(model_path)
        else:
            policy_kwargs, pi_head_kwargs = SMOKE_POLICY_KWARGS, SMOKE_PI_HEAD_KWARGS
        self.policy, self.mapper = build_policy(policy_kwargs, pi_head_kwargs, device)
        if weights_path is not None:
            sd = torch.load(weights_path, map_location=device)
            miss, unexp = self.policy.load_state_dict(sd, strict=False)
            print(f"[vpt] load_state_dict: missing={len(miss)} unexpected={len(unexp)}"
                  + ("  ⚠ 缺失非空,检查 .model 是否与 .weights 同尺寸" if miss else "  ✓ 全部按名命中"))
        self.policy.eval()

        # 边缘化所需的预计算矩阵(numpy → torch,常驻)
        self.button_to_keys = torch.from_numpy(
            self.mapper.BUTTON_IDX_TO_FACTORED.astype(np.float32)).to(device)      # [8641,20] minerl 序
        self.meta_off = torch.from_numpy(
            self.mapper.BUTTON_IDX_TO_CAMERA_META_OFF.astype(np.float32)).to(device)  # [8641] camera-meta off
        self.key_perm = torch.tensor(
            [Buttons.ALL.index(_OUR_TO_MINERL[k]) for k in OUR_KEYS], device=device)  # minerl→我们键序
        self.n_cam = self.mapper.n_camera_bins        # 11
        self.center = self.n_cam // 2                 # 5
        self.reset()

    def reset(self):
        """重置 recurrent 记忆(每个新 episode 之前调用)。"""
        self.state = self.policy.initial_state(1)
        self._first = torch.tensor([False], device=self.device)

    def _preprocess(self, img_rgb):
        """[H,W,3] uint8 RGB → [1,128,128,3] uint8 tensor(VPT 用 INTER_LINEAR resize)。"""
        if img_rgb.shape[:2] != AGENT_RESOLUTION:
            img_rgb = cv2.resize(img_rgb, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(img_rgb)[None]).to(self.device)

    @torch.no_grad()
    def step(self, img_rgb):
        """前向一帧,返回 VPT 动作分布 pd(dict:'camera' log-probs[…,121],'buttons' log-probs[…,8641]);
        内部维护并推进 recurrent state。"""
        obs = {"img": self._preprocess(img_rgb)}
        pd, _vpred, self.state = self.policy.get_output_for_observation(
            obs, self.state, self._first)
        return pd

    @torch.no_grad()
    def to_contract(self, pd):
        """VPT 分布 → 我们 22 维契约的 **soft 目标**:
            dx[11], dy[11](相机分箱概率,已折 camera-meta 门控)、keys[20](逐键概率,我们键序)。
        """
        btn = pd["buttons"].reshape(-1).exp()                    # [8641] 组合概率
        cam = pd["camera"].reshape(self.n_cam, self.n_cam).exp() # [11,11] = [x(yaw), y(pitch)]
        # 逐键边缘
        p_minerl = btn @ self.button_to_keys                     # [20] minerl 按钮序
        p_keys = p_minerl[self.key_perm].clamp(0.0, 1.0)         # [20] 我们键序
        # 相机:meta off ⇒ 等效相机=中心 bin;meta on ⇒ 用 camera 头分布
        p_meta_off = (btn * self.meta_off).sum()
        eff = (1.0 - p_meta_off) * cam
        eff[self.center, self.center] = eff[self.center, self.center] + p_meta_off
        p_dx = eff.sum(dim=1)                                    # over y → dx 分布[11]
        p_dy = eff.sum(dim=0)                                    # over x → dy 分布[11]
        return {"dx": p_dx, "dy": p_dy, "keys": p_keys}


def _smoke():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help=".model 文件(不给则用小随机网络验证装配)")
    ap.add_argument("--weights", default=None, help=".weights 文件")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    mode = "真模型" if args.model else "小随机网络(仅验证 minerl-free 装配 + 适配器)"
    print(f"=== VPT teacher 冒烟 | {mode} | device={args.device} ===")
    teacher = VPTTeacher(args.model, args.weights, args.device)
    n_params = sum(p.numel() for p in teacher.policy.parameters())
    print(f"  策略参数: {n_params / 1e6:.1f}M | buttons 类数={teacher.button_to_keys.shape[0]} "
          f"| camera bins={teacher.n_cam}")

    # 任意分辨率假帧,内部 resize 到 128;连跑 3 帧验证 recurrent state 推进
    teacher.reset()
    for f in range(3):
        dummy = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        pd = teacher.step(dummy)
        tgt = teacher.to_contract(pd)
    print(f"  pd 形状: camera={tuple(pd['camera'].shape)} buttons={tuple(pd['buttons'].shape)}")
    print(f"  契约目标: dx[{tgt['dx'].numel()}] sum={tgt['dx'].sum():.3f} | "
          f"dy[{tgt['dy'].numel()}] sum={tgt['dy'].sum():.3f} | keys[{tgt['keys'].numel()}] "
          f"范围[{tgt['keys'].min():.3f},{tgt['keys'].max():.3f}]")
    ka = tgt["keys"][OUR_KEYS.index("key_attack")].item()
    print(f"  key_attack 概率={ka:.4f}(真模型上应 >0 ⇒ F1 修复;随机网络此值无意义)")
    # 断言:分布归一、概率合法
    assert abs(tgt["dx"].sum().item() - 1.0) < 1e-3 and abs(tgt["dy"].sum().item() - 1.0) < 1e-3
    assert tgt["keys"].min() >= 0.0 and tgt["keys"].max() <= 1.0
    print("  ✓ 装配 + 边缘化适配器通过(minerl-free)")


if __name__ == "__main__":
    _smoke()
