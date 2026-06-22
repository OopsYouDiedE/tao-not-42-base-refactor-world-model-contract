"""YOLO-World-Dreamer 智能体装配 (net/yoloworld/agent.py)。

对外接口:
    YoloWorld       — 持有 WorldModel + ProposalHead + DualHeadBehavior;
                      policy() 用小头点乘选序列做环境交互(环内无 rollout)。
    build_yoloworld — 由 YoloWorldConfig(或字段覆盖)一行构造智能体。

net/ 只装配结构、不含训练循环/优化器/数据加载(那些在 train/crafter/)。设计见 knowledge/yoloworld.md。
"""
import torch
import torch.nn as nn

from net.yoloworld.config import YoloWorldConfig
from net.yoloworld.world_model import WorldModel
from net.yoloworld.heads import ProposalHead, select_score
from net.yoloworld.behavior import DualHeadBehavior


class YoloWorld(nn.Module):
    """YOLO-World-Dreamer 智能体(世界模型 + 256 候选小头 + 双头行为线)。

    Args:
        cfg: YoloWorldConfig。
    """

    def __init__(self, cfg: YoloWorldConfig):
        super().__init__()
        self.cfg = cfg
        self.world_model = WorldModel(cfg)
        self.proposal = ProposalHead(cfg)
        self.behavior = DualHeadBehavior(cfg)

    def set_ach_embed(self, E):
        """注入成就描述嵌入矩阵 E [U, d_g](域常量,见 behavior.set_ach_embed)。"""
        self.behavior.set_ach_embed(E)

    @torch.no_grad()
    def policy(self, obs, state, is_first, task_emb, training=True):
        """单步递归策略(YOLOE 点乘选序列,执行首动作,receding-horizon)。

        Args:
            obs:      [B, C, H, W] float ∈ [0, 1]。
            state:    (latent dict, prev_action [B, A]) 或 None(轨迹起点)。
            is_first: [B] float(1 = 该 env 刚 reset)。
            task_emb: [B, d_g] 当前任务句向量。
            training: True 按 α 采样候选 + 采样首动作探索;False 取 argmax 贪心。

        Returns:
            action_idx:    [B] long。
            action_onehot: [B, A] float。
            state:         更新后的 (latent, action_onehot)。
        """
        wm = self.world_model
        B = obs.shape[0]
        device = obs.device
        if state is None:
            latent = wm.dynamics.initial(B, device)
            prev_action = torch.zeros(B, self.cfg.num_actions, device=device)
        else:
            latent, prev_action = state

        embed = wm.encoder(wm.preprocess_image(obs))
        latent, _ = wm.dynamics.obs_step(latent, prev_action, embed, is_first)
        feat = wm.dynamics.get_feat(latent)

        plan_logits, p, e = self.proposal(feat, task_emb)            # [B,K,H,A]/[B,K]/[B,K,d_g]
        score = select_score(p, e, task_emb, self.cfg.select_beta)   # [B, K]
        if training:
            k = torch.distributions.Categorical(logits=score).sample()  # [B]
        else:
            k = score.argmax(dim=-1)
        # 取所选候选的首步动作
        first = plan_logits[torch.arange(B, device=device), k, 0]    # [B, A]
        if training:
            action_idx = torch.distributions.Categorical(logits=first).sample()
        else:
            action_idx = first.argmax(dim=-1)
        action_onehot = torch.nn.functional.one_hot(
            action_idx, self.cfg.num_actions).float()
        return action_idx, action_onehot, (latent, action_onehot)


def build_yoloworld(device="cuda", **overrides) -> YoloWorld:
    """构造 YoloWorld 智能体。

    Args:
        device: 目标设备。
        **overrides: 覆盖 YoloWorldConfig 任意字段。

    Returns:
        已移到 device 的 YoloWorld。
    """
    cfg = YoloWorldConfig(**overrides)
    return YoloWorld(cfg).to(device)
