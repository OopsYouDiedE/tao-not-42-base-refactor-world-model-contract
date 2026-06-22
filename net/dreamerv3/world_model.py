"""DreamerV3 世界模型 (net/dreamerv3/world_model.py)。

对外接口:
    WorldModel — ConvEncoder + RSSM + ConvDecoder + reward/cont 头;
                 observe 出后验状态、loss() 给重建+奖励+终止+KL 的世界模型总损失。

从 blocks 组装:图像编/解码用 blocks.ConvEncoder/ConvDecoder,递归世界状态用本包 RSSM,
标量头分布用 blocks.DiscDist(reward,two-hot symexp)/ blocks.Bernoulli(cont)/ blocks.MSEDist(图像)。
图像在 [-0.5, 0.5] 域上重建。设计见 [knowledge/dreamer.md]。
"""
import torch
import torch.nn as nn

from blocks import ConvEncoder, ConvDecoder, MLP, DiscDist, Bernoulli, MSEDist
from net.dreamerv3.config import DreamerV3Config
from net.dreamerv3.rssm import RSSM


class WorldModel(nn.Module):
    """DreamerV3 世界模型(表征 + 动力学 + 重建/奖励/终止预测)。

    Args:
        cfg: DreamerV3Config 结构超参。

    观测约定:image ∈ [0, 1] 的 [B, T, C, H, W];内部 preprocess 移到 [-0.5, 0.5]。
    """

    def __init__(self, cfg: DreamerV3Config):
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

        feat_dim = (cfg.dyn_stoch * cfg.dyn_discrete if cfg.dyn_discrete
                    else cfg.dyn_stoch) + cfg.dyn_deter
        self.feat_dim = feat_dim

        self.decoder = ConvDecoder(
            feat_dim=feat_dim, out_channels=c, depths=cfg.dec_depths,
            kernel=cfg.conv_kernel, stride=cfg.conv_stride, min_res=cfg.dec_min_res)
        self.reward = MLP(feat_dim, cfg.reward_bins, hidden=cfg.units,
                          layers=cfg.mlp_layers)
        self.cont = MLP(feat_dim, 1, hidden=cfg.units, layers=cfg.mlp_layers)

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

    @staticmethod
    def preprocess_image(image):
        """[0, 1] → [-0.5, 0.5](重建域居中)。"""
        return image - 0.5

    # ── 损失 ──────────────────────────────────────────────────────────────────
    def loss(self, obs, action, reward, cont, is_first):
        """世界模型总损失与后验状态。

        Args:
            obs:      [B, T, C, H, W] float ∈ [0, 1]。
            action:   [B, T, num_actions] float(one-hot);action[t] = 在 obs[t] 处执行的动作。
            reward:   [B, T] float;reward[t] = 在 obs[t] 执行 action[t] 所得奖励。
            cont:     [B, T] float(1 = 延续,0 = 终止)。
            is_first: [B, T] float(1 = 轨迹起点)。

        Returns:
            total: 标量损失(重建 + 奖励 + 终止 + KL)。
            post:  后验状态 dict(字段首维 B、次维 T),供想象起点。
            metrics: dict[str, float] 各分项标量。
        """
        embed = self.encoder(self.preprocess_image(obs))
        # RSSM 因果对齐:obs[t] 的先验由 (state[t-1], 进入 obs[t] 的动作) 推出,即 action[t-1]。
        # 故喂入右移一位的 prev_action(序列首步补零);序列中段的 episode 重置由 obs_step
        # 据 is_first 把 prev_action 清零处理。这与想象 img_step(state, departing_action) 的因果方向一致。
        prev_action = torch.cat(
            [torch.zeros_like(action[:, :1]), action[:, :-1]], dim=1)
        post, prior = self.dynamics.observe(embed, prev_action, is_first)
        feat = self.dynamics.get_feat(post)

        image_loss = -self.image_dist(feat).log_prob(
            self.preprocess_image(obs)).mean()
        reward_loss = -self.reward_dist(feat).log_prob(reward.unsqueeze(-1)).mean()
        cont_loss = -self.cont_dist(feat).log_prob(cont.unsqueeze(-1)).mean()
        kl_loss, kl_dyn, kl_rep = self.dynamics.kl_loss(
            post, prior, self.cfg.kl_free,
            self.cfg.kl_dyn_scale, self.cfg.kl_rep_scale)

        total = image_loss + reward_loss + cont_loss + kl_loss
        metrics = {
            "wm_total": total.item(),
            "image": image_loss.item(),
            "reward": reward_loss.item(),
            "cont": cont_loss.item(),
            "kl_dyn": kl_dyn.item(),
            "kl_rep": kl_rep.item(),
        }
        return total, post, metrics
