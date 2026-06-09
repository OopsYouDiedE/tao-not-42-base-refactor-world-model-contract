import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.primitives import (
    GlobalTransformApply, PreLNAttn, GatedResidual, BoundedActivation,
    ContinuousTimeEncoding, SpatialPosEmbed, PositionalEmbed
)

# =====================================================================
# 1. Peripheral Vision
# =====================================================================

class PeripheralVision(nn.Module):
    """余光感知系统。输入 64x64，输出 M 个全局 Token 和显著性热力图。"""
    
    def __init__(self, d, M=2):
        super().__init__()
        self.d = d
        self.M = M
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32), # I7: no BatchNorm
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU()
        ) # output: 8x8x128
        
        self.saliency_head = nn.Conv2d(128, 1, kernel_size=1)
        self.saliency_act = BoundedActivation("prob") # Sigmoid
        
        self.pos = PositionalEmbed(128)
        self.feat_proj = nn.Linear(128, d)
        self.queries = nn.Parameter(torch.randn(1, M, d) * 0.02)
        self.attn = PreLNAttn(d, heads=4, mode="cross")
        
    def forward(self, x):
        # x: [B, 3, 64, 64]
        feat = self.net(x) # [B, 128, 8, 8]
        
        saliency = self.saliency_act(self.saliency_head(feat)).squeeze(1) # [B, 8, 8]
        
        pos = self.pos(8, 8, device=feat.device) # [1, 128, 8, 8]
        feat = feat + pos
        feat_flat = feat.flatten(2).transpose(1, 2) # [B, 64, 128]
        feat_flat = self.feat_proj(feat_flat) # [B, 64, d]
        
        q = self.queries.expand(feat.shape[0], -1, -1) # [B, M, d]
        tokens = self.attn(q, feat_flat) # [B, M, d]
        
        return tokens, saliency

# =====================================================================
# 2. Foveated Vision (Minimal Custom ViT-Tiny)
# =====================================================================

class MLP(nn.Module):
    """标准 MLP with LayerNorm."""
    def __init__(self, d, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d)
        )
    def forward(self, x):
        return self.net(x)

class FoveatedVision(nn.Module):
    """主眼中央凹 ViT。输入 64x64 局部裁剪，输出 1 个高清 Token。"""
    
    def __init__(self, d, patch_size=8, img_size=64, layers=4):
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d) * 0.02)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': PreLNAttn(d, heads=4, mode="self"),
                'ffn': GatedResidual(MLP(d, d * 4), gmax=1.0)
            }) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d)
        
    def forward(self, x):
        # x: [B, 3, 64, 64]
        B = x.shape[0]
        x = self.patch_embed(x) # [B, d, 8, 8]
        x = x.flatten(2).transpose(1, 2) # [B, 64, d]
        
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, d]
        x = torch.cat((cls_tokens, x), dim=1) # [B, 65, d]
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block['attn'](x)
            x = block['ffn'](x)
            
        x = self.norm(x)
        return x[:, 0, :].unsqueeze(1) # [B, 1, d]

# =====================================================================
# 3. Slot Binder (Vision-to-Slot)
# =====================================================================

class SlotBinder(nn.Module):
    """将全局与局部感知 Token 绑定到实体 Slot 上。"""
    
    def __init__(self, d):
        super().__init__()
        self.attn = PreLNAttn(d, heads=4, mode="cross")
        # I5: 增益受限。使用可学习的 scalar
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.ln = nn.LayerNorm(d)
        
    def forward(self, Z, P):
        # Z: [B, N, d] (Slots)
        # P: [B, M+1, d] (Perception tokens)
        
        # attn 内部有残差 Z + CrossAttn(Z, P)，但我们想用门控残差，
        # 为了复用 PreLNAttn 的层归一化和注意力，我们可以手动提取注意力输出
        # 因为 PreLNAttn 返回 q + attn_out，所以我们减去 q 得到 attn_out
        z_out = self.attn(Z, P) 
        delta_Z = z_out - Z
        
        gate = torch.sigmoid(self.gate) # clamp to (0, 1)
        Z_new = Z + gate * self.ln(delta_Z) # I5
        
        return Z_new

# =====================================================================
# 4. Action Encoder
# =====================================================================

class ActionEncoder(nn.Module):
    """编码历史动作流。"""
    def __init__(self, t_hist, n_keys, J, d):
        super().__init__()
        self.J = J
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(t_hist * n_keys, 128),
            nn.SiLU(),
            nn.Linear(128, J * d)
        )
        
    def forward(self, a_raw):
        # a_raw: [B, t_hist, n_keys]
        B = a_raw.shape[0]
        x = a_raw.reshape(B, -1)
        tokens = self.net(x).view(B, self.J, self.d)
        return tokens

# =====================================================================
# 5. Core Transformer Brain
# =====================================================================

class WorldTransformerBlock(nn.Module):
    """单层脑内世界 Transformer (无掩码，双向)。"""
    def __init__(self, d, heads=8):
        super().__init__()
        self.attn = PreLNAttn(d, heads=heads, mode="self")
        self.ffn = GatedResidual(MLP(d, d * 4), gmax=1.0)
        
    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

# =====================================================================
# 6. Decoder Heads
# =====================================================================

class StateDecoder(nn.Module):
    """输出未来预测的概率云、存在概率和可控性闸门 c_i。

    c_i ∈ [0,1]^N 表示每个 Slot 受动作控制的程度。
    c_i → 1: 该 Slot 的变化主要由动作引起（可控前景）。
    c_i → 0: 该 Slot 的变化来自环境随机性（不可控背景）。
    c 由逆动力学头接地：InvDyn((Z_next - Z_t) ⊙ c) → â_t。

    Shape:
        z: [B, N, d]
        return: mu [B,N,d], sigma [B,N,d], exist_p [B,N], c [B,N,1]
    """
    def __init__(self, d, N):
        super().__init__()
        self.mu_head = nn.Linear(d, d)
        self.sigma_head = nn.Linear(d, d)
        self.exist_head = nn.Linear(d, 1)
        self.sigma_act = BoundedActivation("pos") # softplus + eps (I2, I3)
        self.exist_act = BoundedActivation("prob") # sigmoid (I3)
        # 可控性闸门：必须是动态预测的！因为 Slot 绑定的物体是动态变化的。
        self.c_head = nn.Linear(d, 1)
        
    def forward(self, z):
        mu = self.mu_head(z)
        sigma = self.sigma_act(self.sigma_head(z))
        exist_p = self.exist_act(self.exist_head(z)).squeeze(-1)
        c = torch.sigmoid(self.c_head(z))  # [B, N, 1]
        return mu, sigma, exist_p, c

class InverseDynamicsHead(nn.Module):
    """轻量级逆动力学读出头。

    接收两帧隐变量的加权残差 (Z_{t+1} - Z_t) ⊙ c，预测导致该转移的动作。
    刻意保持极小（单隐层 MLP），只是挂在潜向量后面的一个读出探针。
    其梯度回流到 c_logit，驱动 c_i 自适应极化（动作可解释 → c→1，否则 → 0）。

    Shape:
        delta_z_weighted: [B, N, d]  即 (Z_next - Z_t) * c
        return: [B, n_keys] 动作 logits
    """
    def __init__(self, d, N, n_keys):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 128),
            nn.SiLU(),
            nn.Linear(128, n_keys),
        )

    def forward(self, delta_z_weighted):
        # 对 N 个 slot 做均值池化再预测动作
        # 注意：这里的池化是在逆动力学的"已经被 c 加权后"的残差上做的，
        # 低 c 的 slot 已经被压到近零，不会稀释信号。
        pooled = delta_z_weighted.mean(dim=1)  # [B, d]
        return self.net(pooled)  # [B, n_keys]




class DecoderHeads(nn.Module):
    """包含动作、注视、唤醒等解码器。"""
    def __init__(self, d, n_keys):
        super().__init__()
        
        self.action_head = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, n_keys)
        )
        self.action_act = BoundedActivation("prob") # Sigmoid
        
        self.gaze_head = nn.Linear(d, 3)
        self.gaze_xy_act = BoundedActivation("flow", scale=1.0) # Tanh (-1 to 1)
        self.gaze_s_act = BoundedActivation("prob") # Sigmoid (0 to 1)
        
        self.wake_head = nn.Linear(d, 1)
        self.wake_act = BoundedActivation("pos") # Softplus + eps

        # 动作计划头(DETR 式):每个动作查询 → (按哪个键, onset Δt, 时长, 是否真有)
        self.plan_key = nn.Linear(d, n_keys)          # 击打哪条轨道/键
        self.plan_onset = nn.Linear(d, 1)             # 多久后按下(秒)
        self.plan_dur = nn.Linear(d, 1)               # 按住多久(秒);0≈tap
        self.plan_exist = nn.Linear(d, 1)             # 该计划槽是否对应真动作
        self.plan_pos_act = BoundedActivation("pos")  # softplus+eps ⇒ onset/时长 ≥0
        self.plan_prob_act = BoundedActivation("prob")  # sigmoid

    def decode_action(self, u_token):
        return self.action_act(self.action_head(u_token))

    def decode_action_plan(self, u_tokens):
        """u_tokens: [B, K, d] → 一次性 K 个带时长的定时动作。"""
        return {
            "key_logits": self.plan_key(u_tokens),                         # [B,K,n_keys]
            "onset": self.plan_pos_act(self.plan_onset(u_tokens)).squeeze(-1),  # [B,K]
            "duration": self.plan_pos_act(self.plan_dur(u_tokens)).squeeze(-1), # [B,K]
            "exist": self.plan_prob_act(self.plan_exist(u_tokens)).squeeze(-1), # [B,K]
        }
        
    def decode_gaze(self, g_token):
        out = self.gaze_head(g_token)
        g_x, g_y = self.gaze_xy_act(out[:, 0:2]).unbind(-1)
        g_s = self.gaze_s_act(out[:, 2])
        return g_x, g_y, g_s
        
    def decode_wake(self, w_token):
        return self.wake_act(self.wake_head(w_token)).squeeze(-1)

# =====================================================================
# 7. Main Model: Tao-Not-42
# =====================================================================

class TaoNot42Model(nn.Module):
    """完整具身世界模型顶层架构。"""
    
    def __init__(self, d=256, N=32, M=2, J=2, n_keys=10, t_hist=10, layers=8, K=6):
        super().__init__()
        self.d = d
        self.N = N
        self.M = M
        self.J = J
        self.K = K   # 一步规划的动作查询数
        self.n_keys = n_keys
        
        self.peripheral = PeripheralVision(d, M)
        self.foveated = FoveatedVision(d)
        self.fovea_pos = SpatialPosEmbed(d)   # 给中央凹 token 贴全局注视位置 (g_x,g_y,g_s)
        self.binder = SlotBinder(d)

        self.time_enc = ContinuousTimeEncoding(d)
        self.action_enc = ActionEncoder(t_hist, n_keys, J, d)
        
        self.blocks = nn.ModuleList([WorldTransformerBlock(d) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d)
        
        self.state_dec = StateDecoder(d, N)
        self.heads = DecoderHeads(d, n_keys)
        self.inv_dyn = InverseDynamicsHead(d, N, n_keys)
        
        # Learnable Placeholders (Must break symmetry!)
        # u 现在是 K 个互不相同的动作查询(randn 破对称,各查询天生不同)
        self.u_placeholder = nn.Parameter(torch.randn(1, K, d) * 0.02)
        self.g_placeholder = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.w_placeholder = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.err_placeholder = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        
    def crop_fovea(self, img, g_x, g_y, g_s, target_size=64):
        """可导裁剪 STN (使用 F.affine_grid 和 grid_sample)"""
        B = img.shape[0]
        device, dtype = img.device, img.dtype
        
        # theta shape: [B, 2, 3]
        # | g_s  0   g_x |
        # |  0  g_s  g_y |
        theta = torch.zeros(B, 2, 3, device=device, dtype=torch.float32) # I4: fp32 grid
        theta[:, 0, 0] = g_s
        theta[:, 1, 1] = g_s
        theta[:, 0, 2] = g_x
        theta[:, 1, 2] = g_y
        
        grid = F.affine_grid(theta, (B, img.shape[1], target_size, target_size), align_corners=True)
        crop = F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
        return crop.to(dtype)
        
    def encode(self, img, Z, g_prev):
        """纯感知编码：提取 Z_target（stop-grad 目标）。

        只执行视觉扫描 + SlotBinder 绑定，不经过 Transformer 推演。
        用于 JEPA 训练中生成 stop-grad 的目标隐变量。

        Args:
            img: [B, 3, H, W] 原始全景图
            Z: [B, N, d] 上一时刻脑内实体
            g_prev: tuple(g_x, g_y, g_s) 注视指令 [B]

        Returns:
            Z_encoded: [B, N, d] 经感知修正后的隐变量
            saliency: [B, 8, 8] 显著性热力图
        """
        img_low = F.interpolate(img, size=(64, 64), mode="bilinear", align_corners=False)
        G_tokens, saliency = self.peripheral(img_low)

        g_x, g_y, g_s = g_prev
        crop = self.crop_fovea(img, g_x, g_y, g_s)
        e_fov = self.foveated(crop)
        e_fov = e_fov + self.fovea_pos(g_x, g_y, g_s).unsqueeze(1)

        P = torch.cat([G_tokens, e_fov], dim=1)  # [B, M+1, d]
        Z_encoded = self.binder(Z, P)  # [B, N, d]
        return Z_encoded, saliency

    def forward(self, img, Z, h, a_raw, dt, g_prev, has_error=False):
        """
        前向闭环推理。
        img: [B, 3, H, W] 原始全景图
        Z: [B, N, d] 上一时刻脑内实体
        h: [B, 1, d] 假说 Token
        a_raw: [B, t_hist, n_keys] 动作历史
        dt: [B, 1] 预测跨度 (秒)
        g_prev: tuple(g_x, g_y, g_s) 上次注视指令 [B]
        has_error: bool 是否注入惊奇误差
        """
        B = img.shape[0]
        
        # 1. 视觉扫描
        img_low = F.interpolate(img, size=(64, 64), mode="bilinear", align_corners=False)
        G_tokens, saliency = self.peripheral(img_low)
        
        g_x, g_y, g_s = g_prev
        crop = self.crop_fovea(img, g_x, g_y, g_s)
        e_fov = self.foveated(crop)                                  # [B, 1, d] 仅内容
        e_fov = e_fov + self.fovea_pos(g_x, g_y, g_s).unsqueeze(1)   # 注入全局注视位置:内容→内容@位置

        P = torch.cat([G_tokens, e_fov], dim=1) # [B, M+1, d]
        
        # 2. 绑定到实体
        Z_new = self.binder(Z, P) # [B, N, d]
        
        # 3. 构造 Transformer 输入
        a_tokens = self.action_enc(a_raw) # [B, J, d]
        tau_token = self.time_enc(dt).unsqueeze(1) # [B, 1, d]
        
        # 扩展 placeholders
        u_p = self.u_placeholder.expand(B, -1, -1)
        g_p = self.g_placeholder.expand(B, -1, -1)
        w_p = self.w_placeholder.expand(B, -1, -1)
        
        seq = [Z_new, h, a_tokens, tau_token, u_p, g_p, w_p]
        if has_error:
            seq.append(self.err_placeholder.expand(B, -1, -1))
            
        X = torch.cat(seq, dim=1) # [B, L, d]
        
        # 4. Transformer 推演
        for block in self.blocks:
            X = block(X)
            
        X = self.final_norm(X)
            
        # 5. 提取特征并解码
        # X order: [N(slots), 1(h), J(a), 1(tau), K(u), 1(g), 1(w), ...]

        out_Z = X[:, 0:self.N, :]
        out_h = X[:, self.N:self.N+1, :]

        base = self.N + 1 + self.J + 1   # 第一个动作查询的位置
        out_u = X[:, base:base+self.K, :]   # [B, K, d] K 个动作查询
        out_g = X[:, base+self.K, :]
        out_w = X[:, base+self.K+1, :]

        mu, sigma, exist_p, c = self.state_dec(out_Z)

        action_plan = self.heads.decode_action_plan(out_u)        # K 个定时带时长动作
        action = self.heads.decode_action(out_u[:, 0, :])         # 兼容旧单动作(query0)
        g_x_new, g_y_new, g_s_new = self.heads.decode_gaze(out_g)
        T_wake = self.heads.decode_wake(out_w)

        return {
            "mu": mu, "sigma": sigma, "exist_p": exist_p,
            "c": c,              # [1, N, 1] 可控性闸门（广播到 [B, N, 1]）
            "Z_out": out_Z,      # 经 transformer 处理后的"当前"实体信念(读出/递归用)
            "h_next": out_h,
            "action": action,
            "action_plan": action_plan,
            "gaze": (g_x_new, g_y_new, g_s_new),
            "T_wake": T_wake,
            "saliency": saliency
        }
