"""PixelTower:从零训练的像素快塔(GRPO 规范下的唯一快塔结构)。

设计立场(苦涩的教训 / Sutton 2019):
  **不为单个游戏打感知补丁。** 输入是原始像素,不是 YOLOE 解析槽位、不是手标的
  log/iron/coal/dirt 分割头、不是 8 角凸包造的树干 GT。那条路线(net/fovea_twotower/
  token_stream.py + g1_conv_head + g1_vectors)已按用户裁决退役——它把大量人力压进
  游戏特定先验,最终仍卡在执行层(R-A:mIoU 0.322 而 wood_rate=0)。
  同理退役:net/bc/policy.py 的冻结 DINOv3 骨干路线。

结构(自回归策略,预测 a_t | o_{≤t}, a_{<t}, goal):
    帧像素 [B,T,3,H,W] ──conv stem(可训)──┐
    goal_vec [B,goal_dim](慢塔文本指示) ──┤ goal_q / goal_bias(FiLM 式条件)
    上一步动作 a_{t-1} [B,T,n_mouse+n_keys] ──act_embed──┘
                     ⊕ + 位置编码 → [MHABlock(causal) + FFN] × layers
                                  → cam_head [B,T,k,n_mouse,camera_bins]  mu-law 分箱
                                  → key_head [B,T,k,n_keys]               独立二值

相机头用分箱 + CE 而非回归:MSE 下"恒预测 0"是平凡解（见 vpt_action_contract.py）。
与 TrackNavTower 的动作头口径逐字一致(camera_bins=11 / n_keys=20 / chunk_k),
故 StudentPolicy._decode 可直接复用;差别只在**输入端换成像素**。

本模块只有结构,不含损失与领域常量(那些在 train/ 侧)。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from blocks.attention import MHABlock


@dataclass
class PixelTowerConfiguration:
    """PixelTower 结构超参(纯 dataclass)。

    Attributes:
        img_hw:      输入分辨率(H, W)。BC 默认使用 90x160。
        d:           时序骨干宽度。
        heads:       自注意力头数(须整除 d)。
        layers:      因果 Transformer 块数。
        goal_dim:    慢塔指示向量维(冻结句向量;MiniLM=384)。
        n_mouse:     连续相机维数(yaw, pitch)。
        camera_bins: 相机 mu-law 分箱数。
        n_keys:      二值按键数。
        chunk_k:     动作分块;k=1(R-B 已裁决:分块对反应式快塔有害)。
    """

    img_hw: tuple[int, int] = (90, 160)
    d: int = 256
    heads: int = 4
    layers: int = 3
    dropout: float = 0.0    # 策略梯度正确性:采样 π 与更新 π 必须同网络;dropout 会造成两次
                            # 独立随机 mask ⇒ log π(a) 打在另一个网络实例上(2026-07-10 修复)
    goal_dim: int = 384
    n_mouse: int = 2
    camera_bins: int = 11
    n_keys: int = 20
    max_len: int = 256
    chunk_k: int = 1
    frame_stack: int = 4    # 帧堆叠 S:stem 吃 [3S,H,W]。单帧测不出任何速度量(自身移动/
                            # 目标接近/相机惯性),而相机控制是伺服问题——误差导数是基本控制量。
                            # DQN(2013) 起所有像素策略都堆帧或带递归;S=4 同时把长程记忆职责
                            # 明确划给地图(空间)与慢塔(任务)。(2026-07-10,设计文档 §7 D1)
    key_prior: float = 0.05  # 按键先验:bias←logit(p)。零 bias 下 sigmoid(0)=0.5 ⇒ 随机初始化
                             # 每 tick 期望按下 n_keys/2=10 个键(背包反复开合、hotbar 全按),
                             # "抽搐乱动"是结构必然。p=0.05 ⇒ 起点期望 1 键/tick,把"人类多数
                             # tick 只按少数键"写进初始分布——先验注入,非采样后手工屏蔽。


class _FFN(nn.Module):
    """Pre-LN 前馈残差块: x + W2(GELU(W1(LN(x))))。[B,L,d]→[B,L,d]。"""

    def __init__(self, d: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, mult * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mult * d, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class _ConvStem(nn.Module):
    """IMPALA 风格的小卷积干:[B,C,H,W] → [B,d]。从零训练,无预训练先验。

    C = 3 × frame_stack(堆叠帧沿通道拼接,旧→新);帧差即速度,一阶导数可观测。
    """

    def __init__(self, d: int, img_hw: tuple[int, int], in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, in_ch, *img_hw)).flatten(1).shape[1]
        self.proj = nn.Linear(n, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class PixelTower(nn.Module):
    """像素输入、goal 条件的因果策略塔。

    Shapes:
        img   [B,T,3·frame_stack,H,W] float32 in [0,1](堆叠帧沿通道拼接,旧→新)
        goal  [B,goal_dim] float32(L2 归一的冻结句向量;无指导时传零向量)
        prev  [B,T,n_mouse+n_keys] float32(上一步动作,cam 已归一到 [-1,1])
        →  cam_logits [B,T,k,n_mouse,camera_bins],  key_logits [B,T,k,n_keys]

    注意:goal 与 prev **必须真的接进来**。旧 grpo_r1.update 把两者都喂零
    (`g = torch.zeros(1,1)`、`prev[1:,0]=0`),导致慢塔指示根本不进梯度。
    """

    def __init__(self, configuration: PixelTowerConfiguration):
        super().__init__()
        self.configuration = configuration
        d = configuration.d
        self.stem = _ConvStem(d, configuration.img_hw, in_ch=3 * configuration.frame_stack)
        self.goal_q = nn.Linear(configuration.goal_dim, d)
        self.goal_bias = nn.Linear(configuration.goal_dim, d)
        self.act_embed = nn.Linear(configuration.n_mouse + configuration.n_keys, d)
        self.pos = nn.Parameter(torch.zeros(1, configuration.max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(configuration.layers):
            self.blocks.append(MHABlock(d, configuration.heads, dropout=configuration.dropout, causal=True))
            self.blocks.append(_FFN(d, dropout=configuration.dropout))
        self.norm = nn.LayerNorm(d)
        self.cam_head = nn.Linear(d, configuration.chunk_k * configuration.n_mouse * configuration.camera_bins)
        self.key_head = nn.Linear(d, configuration.chunk_k * configuration.n_keys)
        with torch.no_grad():  # 按键先验注入（见 PixelTowerConfiguration.key_prior）
            p = configuration.key_prior
            self.key_head.bias.fill_(float(torch.log(torch.tensor(p / (1 - p)))))

    def forward(
        self,
        images: torch.Tensor,
        goals: torch.Tensor,
        previous_actions: torch.Tensor,
    ):
        batch_size, sequence_length = images.shape[:2]
        configuration = self.configuration
        hidden_states = self.stem(
            images.reshape(batch_size * sequence_length, *images.shape[2:]),
        ).reshape(batch_size, sequence_length, configuration.d)
        hidden_states = (
            hidden_states * (1.0 + self.goal_q(goals)[:, None])
            + self.goal_bias(goals)[:, None]
        )
        hidden_states = (
            hidden_states + self.act_embed(previous_actions)
            + self.pos[:, :sequence_length]
        )
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)
        camera_logits = self.cam_head(hidden_states).view(
            batch_size,
            sequence_length,
            configuration.chunk_k,
            configuration.n_mouse,
            configuration.camera_bins,
        )
        key_logits = self.key_head(hidden_states).view(
            batch_size, sequence_length, configuration.chunk_k, configuration.n_keys,
        )
        return camera_logits, key_logits


def build_pixel_tower(configuration: PixelTowerConfiguration) -> PixelTower:
    return PixelTower(configuration)
