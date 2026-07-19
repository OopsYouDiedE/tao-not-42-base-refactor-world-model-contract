"""语言条件化的时空视觉快塔。

对外接口：SpatiotemporalFastTowerConfiguration、MemoryProvider、NullMemory、
StructuredActionOutput、SpatiotemporalFastTower、build_spatiotemporal_fast_tower。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatiotemporalFastTowerConfiguration:
    """快塔结构配置。

    Attributes
    ----------
    visual_dim : int
        冻结视觉编码器 patch token 的通道数。
    text_dim : int
        冻结文本编码器 token 的通道数。
    action_dim : int
        历史结构化动作向量维数。
    d : int
        快塔内部通道数。
    grid_hw : tuple[int, int]
        当前帧 patch 网格高宽，默认 18×32。
    action_horizon : int
        一次预测的未来动作步数，部署只执行前 1–2 步后重规划。
    """

    visual_dim: int = 384
    text_dim: int = 384
    action_dim: int = 22
    d: int = 768
    heads: int = 12
    spatial_layers: int = 4
    temporal_layers: int = 8
    grid_hw: tuple[int, int] = (18, 32)
    camera_bins: int = 11
    action_horizon: int = 4
    max_history: int = 16
    max_text_tokens: int = 64
    dropout: float = 0.0


class MemoryProvider(nn.Module):
    """可选长期记忆 token 的接口。"""

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """返回记忆 token。

        Parameters
        ----------
        batch_size : int
            batch 数量。
        device : torch.device
            输出设备。

        Returns
        -------
        torch.Tensor
            Shape ``[B, M, d]``，Dtype float32/模型 dtype。
        """
        raise NotImplementedError


class NullMemory(MemoryProvider):
    """默认无记忆实现，不引入地图或递归状态。"""

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """返回空 token 序列。

        Returns
        -------
        torch.Tensor
            Shape ``[B, 0, d]``，Dtype float32。
        """
        return torch.empty(batch_size, 0, self.d, device=device)


@dataclass
class StructuredActionOutput:
    """互斥结构化动作分布的 logits，前导维为 ``[B,K]``。"""

    camera_logits: torch.Tensor
    move_fb_logits: torch.Tensor
    move_lr_logits: torch.Tensor
    stance_logits: torch.Tensor
    hotbar_logits: torch.Tensor
    button_logits: torch.Tensor

    def legacy_key_probabilities(self) -> torch.Tensor:
        """展开为旧接口的 20 键边缘概率。

        Returns
        -------
        torch.Tensor
            Shape ``[B,K,20]``，Dtype float32。键序为 CraftGround V2_KEYS。
        """
        fb = F.softmax(self.move_fb_logits.float(), dim=-1)
        lr = F.softmax(self.move_lr_logits.float(), dim=-1)
        stance = F.softmax(self.stance_logits.float(), dim=-1)
        hotbar = F.softmax(self.hotbar_logits.float(), dim=-1)
        buttons = torch.sigmoid(self.button_logits.float())
        return torch.cat(
            [fb[..., 2:3], fb[..., 0:1], lr[..., 0:1], lr[..., 2:3],
             buttons[..., 0:1], stance[..., 1:2], stance[..., 2:3],
             buttons[..., 1:5], hotbar[..., 1:]],
            dim=-1,
        )

    def sample_legacy(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """从结构化分布采样并展开旧相机/20 键接口。

        Parameters
        ----------
        deterministic : bool
            True 使用 argmax/阈值，False 按分布采样。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            相机 Shape ``[B,K,2]``、Dtype int64；按键 Shape ``[B,K,20]``、Dtype int32。
            前后、左右、姿态和 hotbar 各组构造上互斥。
        """
        def categorical(logits: torch.Tensor) -> torch.Tensor:
            """按最后一维的类别分布采样，输入 ``[B,C]`` logits。"""
            if deterministic:
                return logits.argmax(dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            sampled = torch.multinomial(p.reshape(-1, p.shape[-1]), 1)
            return sampled.reshape(p.shape[:-1])

        camera = self.camera_logits.argmax(dim=-1) if deterministic else torch.stack(
            [categorical(self.camera_logits[..., axis, :]) for axis in range(2)], dim=-1,
        )
        fb = categorical(self.move_fb_logits)
        lr = categorical(self.move_lr_logits)
        stance = categorical(self.stance_logits)
        hotbar = categorical(self.hotbar_logits)
        if deterministic:
            events = self.button_logits >= 0
        else:
            events = torch.bernoulli(torch.sigmoid(self.button_logits.float())).bool()
        hotbar_keys = F.one_hot((hotbar - 1).clamp(min=0), num_classes=9)
        hotbar_keys = hotbar_keys * (hotbar > 0).unsqueeze(-1)
        keys = torch.cat(
            [(fb == 2).unsqueeze(-1), (fb == 0).unsqueeze(-1),
             (lr == 0).unsqueeze(-1), (lr == 2).unsqueeze(-1),
             events[..., 0:1], (stance == 1).unsqueeze(-1),
             (stance == 2).unsqueeze(-1), events[..., 1:5], hotbar_keys.bool()],
            dim=-1,
        ).to(dtype=torch.int32)
        return camera, keys


class _Encoder(nn.Module):
    """Pre-LN Transformer 编码器。"""

    def __init__(self, d: int, heads: int, layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码 token 序列 ``[B,L,d]``，保持 Shape 与 Dtype。"""
        return self.norm(self.body(x))


class SpatiotemporalFastTower(nn.Module):
    """时空视觉、文本、动作历史到结构化动作分布的快塔。"""

    def __init__(
        self,
        configuration: SpatiotemporalFastTowerConfiguration,
        memory: MemoryProvider | None = None,
    ):
        super().__init__()
        if configuration.d % configuration.heads:
            raise ValueError("d 必须能被 heads 整除")
        if configuration.action_horizon < 1:
            raise ValueError("action_horizon 必须大于零")
        gh, gw = configuration.grid_hw
        if gh % 2 or gw % 2:
            raise ValueError("grid_hw 必须能被 2×2 历史池化整除")
        self.configuration = configuration
        self.memory = memory if memory is not None else NullMemory(configuration.d)
        self.visual_in = nn.Linear(configuration.visual_dim, configuration.d)
        self.text_in = nn.Linear(configuration.text_dim, configuration.d)
        self.action_in = nn.Linear(configuration.action_dim + 1, configuration.d)
        self.goal_scale = nn.Linear(configuration.d, configuration.d)
        self.goal_bias = nn.Linear(configuration.d, configuration.d)
        self.spatial_pos = nn.Parameter(torch.zeros(1, gh * gw, configuration.d))
        self.history_pos = nn.Parameter(torch.zeros(1, configuration.max_history, configuration.d))
        self.text_pos = nn.Parameter(torch.zeros(1, configuration.max_text_tokens, configuration.d))
        self.spatial = _Encoder(configuration.d, configuration.heads, configuration.spatial_layers, configuration.dropout)
        self.temporal = _Encoder(configuration.d, configuration.heads, configuration.temporal_layers, configuration.dropout)
        self.action_history = _Encoder(configuration.d, configuration.heads, 1, configuration.dropout)
        self.action_queries = nn.Parameter(
            torch.empty(1, configuration.action_horizon, 6, configuration.d),
        )
        self.state_query = nn.Parameter(torch.empty(1, 1, configuration.d))
        self.cross_norm_q = nn.LayerNorm(configuration.d)
        self.cross_norm_kv = nn.LayerNorm(configuration.d)
        self.cross = nn.MultiheadAttention(configuration.d, configuration.heads, batch_first=True, dropout=configuration.dropout)
        self.camera_head = nn.Linear(configuration.d, 2 * configuration.camera_bins)
        self.move_fb_head = nn.Linear(configuration.d, 3)
        self.move_lr_head = nn.Linear(configuration.d, 3)
        self.stance_head = nn.Linear(configuration.d, 3)
        self.hotbar_head = nn.Linear(configuration.d, 10)
        self.button_head = nn.Linear(configuration.d, 5)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.history_pos, std=0.02)
        nn.init.trunc_normal_(self.text_pos, std=0.02)
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        nn.init.trunc_normal_(self.state_query, std=0.02)

    def _text(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """投影文本 token 并计算有掩码汇总。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            token Shape ``[B,L,d]`` 与汇总 Shape ``[B,d]``，Dtype float32/模型 dtype。
        """
        if tokens.shape[1] > self.configuration.max_text_tokens:
            raise ValueError("文本 token 数超过 max_text_tokens")
        x = self.text_in(tokens) + self.text_pos[:, :tokens.shape[1]]
        weight = mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x * weight
        denom = weight.sum(dim=1).float().clamp(min=1e-4).to(dtype=x.dtype)
        summary = (x * weight).sum(dim=1) / denom
        return x, summary

    def _pool_history(self, history: torch.Tensor) -> torch.Tensor:
        """历史 patch 做 2×2 空间池化。

        Returns
        -------
        torch.Tensor
            Shape ``[B,H,gh/2*gw/2,Dv]``，保持输入 Dtype。
        """
        b, h, _, d = history.shape
        gh, gw = self.configuration.grid_hw
        x = history.reshape(b, h, gh // 2, 2, gw // 2, 2, d)
        return x.mean(dim=(3, 5)).reshape(b, h, (gh // 2) * (gw // 2), d)

    def _encode_context(
        self,
        current_patches: torch.Tensor,
        history_patches: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        past_actions: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将完整空间网格、连续历史、文本和已执行动作编码为注意力上下文。

        Parameters
        ----------
        current_patches : torch.Tensor
            Shape ``[B,576,Dv]``，Dtype float32/bfloat16，当前帧 DINO patch。
        history_patches : torch.Tensor
            Shape ``[B,H,576,Dv]``，Dtype float32/bfloat16，历史 DINO patch。
        text_tokens : torch.Tensor
            Shape ``[B,L,Dt]``，Dtype float32/bfloat16，冻结文本编码器输出。
        text_mask : torch.Tensor
            Shape ``[B,L]``，Dtype bool，文本有效位。
        past_actions : torch.Tensor
            Shape ``[B,H+1,A]``，Dtype float32，已请求/执行动作历史。
        dt : torch.Tensor
            Shape ``[B,H+1,1]``，Dtype float32，单位秒。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            上下文 token ``[B,N,d]`` 与任务汇总 ``[B,d]``。
        """
        b, n, _ = current_patches.shape
        gh, gw = self.configuration.grid_hw
        if n != gh * gw:
            raise ValueError(f"当前 patch 数必须为 {gh * gw}")
        h = history_patches.shape[1]
        if h > self.configuration.max_history - 1:
            raise ValueError("历史帧数超过 max_history-1")
        if history_patches.shape[2] != gh * gw:
            raise ValueError(f"历史 patch 数必须为 {gh * gw}")
        if past_actions.shape[:2] != (b, h + 1):
            raise ValueError("past_actions 必须与历史帧对齐且不包含当前待预测动作")
        if dt.shape != (b, h + 1, 1):
            raise ValueError("dt 必须为 [B,H+1,1]")
        text, goal = self._text(text_tokens, text_mask)
        current = self.visual_in(current_patches) + self.spatial_pos
        current = current * (1.0 + torch.tanh(self.goal_scale(goal))[:, None])
        current = current + self.goal_bias(goal)[:, None]
        current = self.spatial(current)

        pooled_current = current.reshape(b, gh, gw, self.configuration.d)
        pooled_current = pooled_current.reshape(b, gh // 2, 2, gw // 2, 2, self.configuration.d)
        pooled_current = pooled_current.mean(dim=(2, 4)).flatten(1, 2)
        history = self.visual_in(self._pool_history(history_patches))
        history = history * (1.0 + torch.tanh(self.goal_scale(goal))[:, None, None])
        history = history + self.goal_bias(goal)[:, None, None]
        frames = torch.cat([history, pooled_current[:, None]], dim=1)
        frames = frames + self.history_pos[:, :h + 1, None]
        _, t, p, d = frames.shape
        temporal = frames.permute(0, 2, 1, 3).reshape(b * p, t, d)
        temporal = self.temporal(temporal)[:, -1].reshape(b, p, d)

        action = self.action_in(torch.cat([past_actions.float(), dt.float()], dim=-1))
        action = self.action_history(action.to(current.dtype))
        memory = self.memory(b, current.device).to(dtype=current.dtype)
        kv = torch.cat([current, temporal, text, action, memory], dim=1)
        return kv, goal

    def forward_with_state(
        self,
        current_patches: torch.Tensor,
        history_patches: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        past_actions: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[StructuredActionOutput, torch.Tensor]:
        """计算动作块和供价值学习使用的策略状态。"""
        kv, goal = self._encode_context(
            current_patches, history_patches, text_tokens, text_mask,
            past_actions, dt,
        )
        b = current_patches.shape[0]
        query = self.action_queries.expand(b, -1, -1, -1) + goal[:, None, None]
        query = query.flatten(1, 2)
        attended, _ = self.cross(
            self.cross_norm_q(query), self.cross_norm_kv(kv), self.cross_norm_kv(kv),
            need_weights=False,
        )
        query = (query + attended).reshape(b, self.configuration.action_horizon, 6, -1)
        state_query = self.state_query.expand(b, -1, -1) + goal[:, None]
        state_attended, _ = self.cross(
            self.cross_norm_q(state_query), self.cross_norm_kv(kv), self.cross_norm_kv(kv),
            need_weights=False,
        )
        state = (state_query + state_attended)[:, 0]
        output = StructuredActionOutput(
            camera_logits=self.camera_head(query[:, :, 0]).reshape(
                b, self.configuration.action_horizon, 2, self.configuration.camera_bins,
            ),
            move_fb_logits=self.move_fb_head(query[:, :, 1]),
            move_lr_logits=self.move_lr_head(query[:, :, 2]),
            stance_logits=self.stance_head(query[:, :, 3]),
            hotbar_logits=self.hotbar_head(query[:, :, 4]),
            button_logits=self.button_head(query[:, :, 5]),
        )
        return output, state

    def forward(
        self,
        current_patches: torch.Tensor,
        history_patches: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        past_actions: torch.Tensor,
        dt: torch.Tensor,
    ) -> StructuredActionOutput:
        """计算未来 ``action_horizon`` 步的结构化动作 logits。"""
        output, _ = self.forward_with_state(
            current_patches, history_patches, text_tokens, text_mask,
            past_actions, dt,
        )
        return output


def build_spatiotemporal_fast_tower(
    configuration: SpatiotemporalFastTowerConfiguration,
    memory: MemoryProvider | None = None,
) -> SpatiotemporalFastTower:
    """构造快塔 v2。

    Returns
    -------
    SpatiotemporalFastTower
        未加载视觉或文本骨干权重的快塔核心。
    """
    return SpatiotemporalFastTower(configuration, memory)
