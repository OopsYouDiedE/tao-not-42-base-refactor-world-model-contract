import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinecraftDecoderHeads(nn.Module):
    """Minecraft 复合动作解码器：同时预测鼠标移动和大量键盘按键。"""
    def __init__(self, d, num_keyboard_keys=20):
        super().__init__()
        # 鼠标预测 (dx, dy 回归)
        self.mouse_head = nn.Sequential(
            nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 2)
        )
        # 键盘预测 (20 个独立二分类)
        self.keyboard_head = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, num_keyboard_keys)
        )
        
        # 动作规划头 (DETR 式)
        self.plan_mouse = nn.Linear(d, 2)
        self.plan_keyboard = nn.Linear(d, num_keyboard_keys)
        self.plan_onset = nn.Linear(d, 1)             # 多久后按下(秒)
        self.plan_dur = nn.Linear(d, 1)               # 按住多久(秒)
        self.plan_exist = nn.Linear(d, 1)             # 该计划槽是否有效

    def decode_action(self, u_token):
        mouse = self.mouse_head(u_token)
        kb_logits = self.keyboard_head(u_token)
        kb_prob = torch.sigmoid(kb_logits)
        return torch.cat([mouse, kb_prob], dim=-1)

    def decode_action_plan(self, u_tokens):
        """预测未来长程动作计划"""
        return {
            "mouse": self.plan_mouse(u_tokens), 
            "keyboard": torch.sigmoid(self.plan_keyboard(u_tokens)),
            "onset": F.softplus(self.plan_onset(u_tokens)).squeeze(-1),
            "duration": F.softplus(self.plan_dur(u_tokens)).squeeze(-1),
            "exist": torch.sigmoid(self.plan_exist(u_tokens)).squeeze(-1)
        }

class MinecraftInverseDynamicsHead(nn.Module):
    def __init__(self, d, num_keyboard_keys=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU()
        )
        self.mouse_out = nn.Linear(64, 2)
        self.kb_out = nn.Linear(64, num_keyboard_keys)
        
    def forward(self, residual_z):
        # residual_z: [B, N, d]
        pooled = residual_z.mean(dim=1)
        feat = self.net(pooled)
        mouse = self.mouse_out(feat)
        kb_prob = torch.sigmoid(self.kb_out(feat))
        return torch.cat([mouse, kb_prob], dim=-1)

def sinusoidal_time_encoding(t_vec, d):
    """
    连续绝对时间戳编码。
    t_vec: [B] 时间戳(秒)
    返回: [B, 1, d]
    """
    B = t_vec.shape[0]
    pe = torch.zeros(B, d, device=t_vec.device)
    position = t_vec.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=t_vec.device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d % 2 != 0:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(1) # [B, 1, d]

class MockDINOv2(nn.Module):
    """如果本地没有 transformers，用一个卷积网络模拟 DINOv2 输出 Patch Tokens"""
    def __init__(self, d=384):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8), nn.ReLU(),
            nn.Conv2d(64, d, kernel_size=4, stride=4), nn.ReLU()
        )
    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.net(x) # [B, d, h, w]
        B, d, h, w = feat.shape
        return feat.view(B, d, h*w).transpose(1, 2) # [B, M, d]

class MinecraftWorldModel(nn.Module):
    """整合了时空掩码、DINOv2视觉、文本编码器的终极世界模型"""
    def __init__(self, d=384, N=16, K=5, J=8, act_dim=22):
        super().__init__()
        self.d = d
        self.N = N # 实体槽数量
        self.K = K # 动作查询数量
        
        # 多模态骨干
        self.vision_encoder = MockDINOv2(d)
        self.text_proj = nn.Linear(768, d) # 假设从 DistilBERT 的 768 映射过来
        
        # 槽位记忆
        self.slots = nn.Parameter(torch.randn(1, N, d))
        from net.tao_not_42 import SlotBinder
        self.binder = SlotBinder(d)
        
        # 内部递归状态 (mu, sigma, c, exist)
        out_dim = d * 3 + 1
        self.state_dec = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d*2),
            nn.SiLU(),
            nn.Linear(d*2, out_dim)
        )
        
        # Transformer
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=d*4, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=4)
        
        # 复合动作
        self.action_enc = nn.Linear(act_dim, d)
        self.heads = MinecraftDecoderHeads(d, num_keyboard_keys=act_dim-2)
        self.inv_dyn = MinecraftInverseDynamicsHead(d, num_keyboard_keys=act_dim-2)
        
        # Placeholders
        self.u_placeholder = nn.Parameter(torch.randn(1, K, d))
        self.text_placeholder = nn.Parameter(torch.randn(1, 1, d)) # 如果任务没有文本时的占位
        self.h_placeholder = nn.Parameter(torch.randn(1, 1, d))

    def encode_vision(self, patch_tokens, t_vec, Z_prev):
        """带时间戳注入的视觉编码"""
        # 1. 空间掩码 (Spatial Masking) - 随机丢弃 30% Patches
        if self.training:
            B, M, d = patch_tokens.shape
            mask = torch.rand(B, M, device=patch_tokens.device) > 0.3
            # 这里简单处理：被遮挡的赋予零向量
            patch_tokens = patch_tokens * mask.unsqueeze(-1)
            
        # 3. 注入绝对时间戳 (Time Anchoring)
        time_pe = sinusoidal_time_encoding(t_vec, self.d) # [B, 1, d]
        patch_tokens = patch_tokens + time_pe
        
        # 4. 绑定到实体槽
        Z_new = self.binder(Z_prev, patch_tokens)
        return Z_new

    def forward(self, patch_tokens, Z, h, a_raw, t_vec, task_feat=None, drop_frame=False):
        B = patch_tokens.shape[0]
        
        # 视觉感知与时空掩码
        if drop_frame and self.training:
            # 时间丢帧 (Blind Rollout): 直接沿用 Z，不看画面
            Z_new = Z
        else:
            Z_new = self.encode_vision(patch_tokens, t_vec, Z)
            
        # 任务文本编码
        if task_feat is not None:
            text_token = self.text_proj(task_feat).unsqueeze(1) # [B, 1, d]
        else:
            text_token = self.text_placeholder.expand(B, -1, -1)
            
        # 序列拼接
        a_tokens = self.action_enc(a_raw) # [B, J, d] (历史动作)
        u_p = self.u_placeholder.expand(B, -1, -1)
        h_token = h if h is not None else self.h_placeholder.expand(B, -1, -1)
        
        # [N slots, 1 text, 1 h, J actions, K action_queries]
        X = torch.cat([Z_new, text_token, h_token, a_tokens, u_p], dim=1)
        
        # Transformer 推演
        X = self.blocks(X)
        
        # 解码
        out_Z = X[:, 0:self.N, :]
        out_h = X[:, self.N+1:self.N+2, :]
        base = self.N + 1 + 1 + a_tokens.shape[1]
        out_u = X[:, base:base+self.K, :]
        
        # 状态预测
        out_state = self.state_dec(out_Z)
        mu = out_state[:, :, :self.d]
        sigma = F.softplus(out_state[:, :, self.d:2*self.d]) + 1e-4
        c = torch.sigmoid(out_state[:, :, 2*self.d:3*self.d]) # [B, N, 1]
        exist_p = torch.sigmoid(out_state[:, :, 3*self.d:])
        
        action_plan = self.heads.decode_action_plan(out_u)
        
        return {
            "mu": mu, "sigma": sigma, "c": c, "exist_p": exist_p,
            "Z_out": out_Z, "h_next": out_h,
            "action_plan": action_plan
        }
