"""VPT teacher 真实权重加载 (net/vpt/adapter.py)。

对外接口:
    VPTTeacher — 加载真实 VPT foundation-model 权重。
"""
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.vpt_lib.policy import MinecraftPolicy


class VPTTeacher(nn.Module):
    """VPT teacher，加载真实权重。

    Args:
        model_path:     VPT .model 文件路径
        weights_path:   VPT .weights 文件路径
        target_hidsize: 学生模型隐藏维度（用于对齐）
        target_actions: 学生动作数（27）

    Forward:
        obs: (B, 3, H, W) float32 [0,1]
        → logits: (B, 27), hidden: (B, target_hidsize)
    """

    def __init__(
        self,
        model_path: str,
        weights_path: str,
        target_hidsize: int = 256,
        target_actions: int = 27,
    ):
        super().__init__()

        # 加载 VPT 配置
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

        policy_kwargs = params['model']['args']['net']['args']
        self.vpt_hidsize = policy_kwargs['hidsize']

        # 初始化 VPT policy net
        self.net = MinecraftPolicy(**policy_kwargs)

        # 加载权重
        weights = torch.load(weights_path, map_location='cpu')
        # 移除 "net." 前缀（如果存在）
        if any(k.startswith('net.') for k in weights.keys()):
            weights = {k.replace('net.', '', 1): v for k, v in weights.items() if k.startswith('net.')}
        self.net.load_state_dict(weights)
        self.net.eval()

        # 投影层
        self.hidden_proj = nn.Linear(self.vpt_hidsize, target_hidsize)
        self.action_proj = nn.Linear(self.vpt_hidsize, target_actions)

        # Recurrent state
        self.state = None

        print(f"✓ VPT teacher: {self.vpt_hidsize}D → {target_hidsize}D, {target_actions} actions")

    def forward(self, obs: torch.Tensor, first: bool = True):
        """
        Args:
            obs: (B, 3, H, W) [0,1] RGB
            first: 是否为episode开始

        Returns:
            logits: (B, 27)
            hidden: (B, target_hidsize)
        """
        B = obs.shape[0]

        # 128×128 resize
        if obs.shape[-2:] != (128, 128):
            obs = F.interpolate(obs, size=(128, 128), mode='bilinear', align_corners=False)

        # 转换为 VPT 格式: (B,C,H,W) → (B,T=1,H,W,C)
        obs_vpt = obs.permute(0, 2, 3, 1).unsqueeze(1)  # (B,H,W,C) + T dim

        # 初始化state
        if self.state is None or first:
            self.state = self.net.initial_state(B)
            if self.state is not None:
                def to_device(x):
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        return x.to(obs.device)
                    if isinstance(x, (list, tuple)):
                        return type(x)(to_device(t) for t in x)
                    return x
                self.state = [to_device(s) for s in self.state]

        # VPT forward
        with torch.no_grad():
            ob_dict = {"img": obs_vpt}  # (B,T=1,H,W,C)
            context = {"first": torch.tensor([[first]] * B, device=obs.device)}  # (B,T=1)
            (pi_latent, _), self.state = self.net(ob_dict, self.state, context)
            pi_latent = pi_latent.squeeze(1)  # (B,T=1,D) → (B,D)

        # 投影
        hidden = self.hidden_proj(pi_latent)
        logits = self.action_proj(pi_latent)

        return logits, hidden
