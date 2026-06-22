"""YOLO-World-Dreamer 世界模型 (net/yoloworld/world_model.py)。

对外接口:
    WorldModel — ConvEncoder + RSSM + ConvDecoder + reward/cont/**成就** 头;
                 observe 出后验状态、loss() 给重建+奖励+终止+成就+KL 的世界模型总损失。

与 net/dreamerv3 的世界模型同构,额外加一个**成就预测头** ψ(s)∈[0,1]^U(多标签):
它把"当前状态完成了哪些成就"压进隐状态,使行为线能在想象内由 ρ^g=w(g)·ψ 自算
任务条件势函数奖励(YOLOE 在状态层的点乘,见 knowledge/yoloworld.md §3)。
从 blocks 组装,图像在 [-0.5, 0.5] 域重建。
"""
import torch
import torch.nn as nn

from blocks import ConvEncoder, ConvDecoder, MLP, DiscDist, Bernoulli, MSEDist
from net.yoloworld.config import YoloWorldConfig
from net.dreamerv3.rssm import RSSM


class WorldModel(nn.Module):
    """目标条件世界模型(表征 + 动力学 + 重建/奖励/终止/成就预测)。

    Args:
        cfg: YoloWorldConfig 结构超参。

    观测约定:image ∈ [0, 1] 的 [B, T, C, H, W];内部 preprocess 移到 [-0.5, 0.5]。
    """

    def __init__(self, cfg: YoloWorldConfig):
        super().__init__()
        self.cfg = cfg
        c, h, w = cfg.obs_shape

        self.encoder = ConvEncoder(
            in_channels=c, depths=cfg.enc_depths,
            kernel=cfg.conv_kernel, stride=cfg.conv_stride, flatten=True)
        embed_dim = self.encoder.feature_dim((h, w))

        self.dynamics = RSSM(
            stoch=cfg.dyn_stoch, deter=cfg.dyn_deter, hidden=cfg.dyn_hidden,
            discrete=cfg.dyn_discrete, num_actions=cfg.num_actions,
            embed_dim=embed_dim, rec_depth=cfg.dyn_rec_depth,
            unimix_ratio=cfg.unimix_ratio)

        feat_dim = cfg.feat_dim
        self.feat_dim = feat_dim

        self.decoder = ConvDecoder(
            feat_dim=feat_dim, out_channels=c, depths=cfg.dec_depths,
            kernel=cfg.conv_kernel, stride=cfg.conv_stride, min_res=cfg.dec_min_res)
        self.reward = MLP(feat_dim, cfg.reward_bins, hidden=cfg.units,
                          layers=cfg.mlp_layers)
        self.cont = MLP(feat_dim, 1, hidden=cfg.units, layers=cfg.mlp_layers)
        # 成就头 ψ(s):多标签 logits → BCE;前向取 sigmoid 得 [0,1]^U
        self.ach = MLP(feat_dim, cfg.n_achievements, hidden=cfg.units,
                       layers=cfg.mlp_layers)

    # ── 头分布封装 ──────────────────────────────────────────────────────────
    def reward_dist(self, feat):
        """feat → reward 的 two-hot symexp 离散分布(DiscDist)。"""
        logits = self.reward(feat)
        return DiscDist(logits, device=logits.device)

    def cont_dist(self, feat):
        """feat → 终止延续概率的伯努利分布。"""
        logits = self.cont(feat)
        return Bernoulli(torch.distributions.independent.Independent(
            torch.distributions.bernoulli.Bernoulli(logits=logits), 1))

    def image_dist(self, feat):
        """feat → 图像重建分布(MSEDist,像素和聚合)。"""
        return MSEDist(self.decoder(feat), agg="sum")

    def ach_prob(self, feat):
        """feat → 成就完成度 ψ(s) ∈ [0,1]^U(各成就独立 sigmoid)。"""
        return torch.sigmoid(self.ach(feat))

    @staticmethod
    def preprocess_image(image):
        """[0, 1] → [-0.5, 0.5](重建域居中)。"""
        return image - 0.5

    # ── 损失 ──────────────────────────────────────────────────────────────────
    def loss(self, obs, action, reward, cont, ach, is_first):
        """世界模型总损失与后验状态。

        Args:
            obs:      [B, T, C, H, W] float ∈ [0, 1]。
            action:   [B, T, A] float one-hot;action[t] = 在 obs[t] 处执行的动作。
            reward:   [B, T] float。
            cont:     [B, T] float(1 = 延续)。
            ach:      [B, T, U] float(0/1 multi-hot,该步累计已解锁成就)。
            is_first: [B, T] float(1 = 轨迹起点)。

        Returns:
            total: 标量损失(重建 + 奖励 + 终止 + 成就 + KL)。
            post:  后验状态 dict(字段首维 B、次维 T),供想象起点。
            metrics: dict[str, float]。
        """
        embed = self.encoder(self.preprocess_image(obs))
        # RSSM 因果对齐:obs[t] 的先验由 (state[t-1], 进入 obs[t] 的动作 action[t-1]) 推出。
        prev_action = torch.cat(
            [torch.zeros_like(action[:, :1]), action[:, :-1]], dim=1)
        post, prior = self.dynamics.observe(embed, prev_action, is_first)
        feat = self.dynamics.get_feat(post)

        image_loss = -self.image_dist(feat).log_prob(
            self.preprocess_image(obs)).mean()
        reward_loss = -self.reward_dist(feat).log_prob(reward.unsqueeze(-1)).mean()
        cont_loss = -self.cont_dist(feat).log_prob(cont.unsqueeze(-1)).mean()
        # 成就头:多标签 BCE(I3:走 logits + BCEWithLogits,数值稳定)。
        # Crafter 成就稀疏,无权重 BCE 会让 ψ 退化为全 0 → ρ^g=w·ψ≈0 → 目标条件奖励消失。
        # 按 batch 逐成就的负正比设 pos_weight(钳 [1,50]),把正样本梯度抬回来。
        pos = ach.sum(dim=(0, 1))                                # [U]
        neg = ach[..., 0].numel() - pos
        pos_weight = (neg / (pos + 1.0)).clamp(1.0, 50.0)
        ach_loss = nn.functional.binary_cross_entropy_with_logits(
            self.ach(feat), ach, pos_weight=pos_weight)
        kl_loss, kl_dyn, kl_rep = self.dynamics.kl_loss(
            post, prior, self.cfg.kl_free,
            self.cfg.kl_dyn_scale, self.cfg.kl_rep_scale)

        total = (image_loss + reward_loss + cont_loss
                 + self.cfg.ach_scale * ach_loss + kl_loss)
        metrics = {
            "wm_total": total.item(),
            "image": image_loss.item(),
            "reward": reward_loss.item(),
            "cont": cont_loss.item(),
            "ach": ach_loss.item(),
            "kl_dyn": kl_dyn.item(),
            "kl_rep": kl_rep.item(),
        }
        return total, post, metrics
