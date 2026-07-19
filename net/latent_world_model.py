"""动作条件化的离散潜状态世界模型及其训练契约。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.residual import ResidualFeedForwardBlock


@dataclass(frozen=True)
class LatentWorldModelConfiguration:
    """潜状态世界模型配置。"""

    observation_dim: int = 384
    action_dim: int = 22
    d: int = 1024
    stochastic_variables: int = 32
    stochastic_classes: int = 16
    dynamics_layers: int = 12
    event_dim: int = 0
    inventory_dim: int = 0


@dataclass
class LatentWorldModelState:
    """RSSM 状态。

    ``deterministic`` Shape 为 ``[B,d]``，``stochastic`` Shape 为
    ``[B,V,C]``，二者 Dtype 与模型计算 Dtype 一致。
    """

    deterministic: torch.Tensor
    stochastic: torch.Tensor


@dataclass
class LatentWorldModelPrediction:
    """一次想象转移的预测。"""

    next_state: LatentWorldModelState
    prior_logits: torch.Tensor
    observation: torch.Tensor
    reward: torch.Tensor
    continuation_logits: torch.Tensor
    event_logits: torch.Tensor | None
    inventory_delta: torch.Tensor | None


class LatentWorldModel(nn.Module):
    """以冻结视觉特征为观测、以结构化动作为条件的离散 RSSM。"""

    def __init__(self, configuration: LatentWorldModelConfiguration):
        super().__init__()
        if configuration.dynamics_layers < 1:
            raise ValueError("dynamics_layers 必须大于零")
        if configuration.stochastic_variables < 1 or configuration.stochastic_classes < 2:
            raise ValueError("离散潜状态至少需要一个变量和两个类别")
        self.configuration = configuration
        stochastic_dim = (
            configuration.stochastic_variables * configuration.stochastic_classes
        )
        self.observation_in = nn.Linear(configuration.observation_dim, configuration.d)
        self.action_in = nn.Linear(configuration.action_dim + 1, configuration.d)
        self.recurrent = nn.GRUCell(configuration.d + stochastic_dim, configuration.d)
        self.dynamics = nn.Sequential(*[
            ResidualFeedForwardBlock(configuration.d)
            for _ in range(configuration.dynamics_layers)
        ])
        self.prior = nn.Linear(configuration.d, stochastic_dim)
        self.posterior = nn.Sequential(
            nn.Linear(2 * configuration.d, configuration.d),
            nn.GELU(),
            nn.Linear(configuration.d, stochastic_dim),
        )
        self.feature_projection = nn.Linear(configuration.d + stochastic_dim, configuration.d)
        self.observation_head = nn.Linear(configuration.d, configuration.observation_dim)
        self.reward_head = nn.Linear(configuration.d, 1)
        self.continuation_head = nn.Linear(configuration.d, 1)
        self.event_head = (
            nn.Linear(configuration.d, configuration.event_dim)
            if configuration.event_dim else None
        )
        self.inventory_head = (
            nn.Linear(configuration.d, configuration.inventory_dim)
            if configuration.inventory_dim else None
        )

    def _reshape_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """将 ``[B,V*C]`` 变为 ``[B,V,C]``。"""
        return logits.reshape(
            logits.shape[0],
            self.configuration.stochastic_variables,
            self.configuration.stochastic_classes,
        )

    @staticmethod
    def _sample(logits: torch.Tensor, training: bool) -> torch.Tensor:
        """以 fp32 采样直通 one-hot，评估时采用 argmax。"""
        logits_fp32 = logits.float()
        if training:
            sample = F.gumbel_softmax(logits_fp32, tau=1.0, hard=True, dim=-1)
        else:
            index = logits_fp32.argmax(dim=-1)
            sample = F.one_hot(index, logits.shape[-1]).float()
        return sample.to(dtype=logits.dtype)

    def _feature(self, state: LatentWorldModelState) -> torch.Tensor:
        """将确定性和离散状态拼接为 ``[B,d]`` 特征。"""
        stochastic = state.stochastic.flatten(1)
        return self.feature_projection(torch.cat([state.deterministic, stochastic], dim=-1))

    def initialize(self, observation: torch.Tensor) -> tuple[LatentWorldModelState, torch.Tensor]:
        """由首个冻结视觉观测 ``[B,observation_dim]`` 初始化后验状态。"""
        deterministic = self.observation_in(observation)
        posterior_logits = self._reshape_logits(
            self.posterior(torch.cat([deterministic, deterministic], dim=-1)),
        )
        stochastic = self._sample(posterior_logits, self.training)
        return LatentWorldModelState(deterministic, stochastic), posterior_logits

    def imagine(
        self,
        state: LatentWorldModelState,
        action: torch.Tensor,
        dt: torch.Tensor,
    ) -> LatentWorldModelPrediction:
        """执行一次仅依赖旧状态和动作的潜空间想象。

        Parameters
        ----------
        state : LatentWorldModelState
            当前潜状态。
        action : torch.Tensor
            Shape ``[B,action_dim]``，连续相机加结构化键位。
        dt : torch.Tensor
            Shape ``[B,1]``，单位秒，分母或时基不得隐含在模型外。
        """
        if dt.ndim != 2 or dt.shape[-1] != 1:
            raise ValueError("dt 必须为 [B,1]")
        action_token = self.action_in(torch.cat([action.float(), dt.float()], dim=-1))
        recurrent_input = torch.cat([action_token, state.stochastic.flatten(1)], dim=-1)
        deterministic = self.recurrent(
            recurrent_input.to(dtype=state.deterministic.dtype), state.deterministic,
        )
        deterministic = self.dynamics(deterministic)
        prior_logits = self._reshape_logits(self.prior(deterministic))
        stochastic = self._sample(prior_logits, self.training)
        next_state = LatentWorldModelState(deterministic, stochastic)
        feature = self._feature(next_state)
        return LatentWorldModelPrediction(
            next_state=next_state,
            prior_logits=prior_logits,
            observation=self.observation_head(feature),
            reward=self.reward_head(feature).squeeze(-1),
            continuation_logits=self.continuation_head(feature).squeeze(-1),
            event_logits=self.event_head(feature) if self.event_head is not None else None,
            inventory_delta=(
                self.inventory_head(feature) if self.inventory_head is not None else None
            ),
        )

    def observe(
        self,
        predicted_state: LatentWorldModelState,
        observation: torch.Tensor,
    ) -> tuple[LatentWorldModelState, torch.Tensor]:
        """以真实下一观测修正想象先验并返回后验 logits。"""
        observation_token = self.observation_in(observation)
        posterior_logits = self._reshape_logits(self.posterior(torch.cat(
            [predicted_state.deterministic, observation_token], dim=-1,
        )))
        stochastic = self._sample(posterior_logits, self.training)
        return (
            LatentWorldModelState(predicted_state.deterministic, stochastic),
            posterior_logits,
        )


def balanced_categorical_kl_loss(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    dynamic_weight: float = 0.8,
    free_nats: float = 1.0,
) -> torch.Tensor:
    """计算 Dreamer 风格的平衡离散 KL，所有概率运算使用 fp32。"""
    if not 0.0 <= dynamic_weight <= 1.0:
        raise ValueError("dynamic_weight 必须位于 [0,1]")

    def categorical_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        p_log = F.log_softmax(p_logits.float(), dim=-1)
        q_log = F.log_softmax(q_logits.float(), dim=-1)
        probability = p_log.exp().clamp(min=1e-4)
        return (probability * (p_log - q_log)).sum(dim=-1).sum(dim=-1)

    dynamic = categorical_kl(posterior_logits.detach(), prior_logits)
    representation = categorical_kl(posterior_logits, prior_logits.detach())
    dynamic = dynamic.clamp(min=free_nats).mean()
    representation = representation.clamp(min=free_nats).mean()
    return dynamic_weight * dynamic + (1.0 - dynamic_weight) * representation


def build_latent_world_model(
    configuration: LatentWorldModelConfiguration,
) -> LatentWorldModel:
    """构造不包含视觉骨干、数据读取或环境进程的世界模型核心。"""
    return LatentWorldModel(configuration)
