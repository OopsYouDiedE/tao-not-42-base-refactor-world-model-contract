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

    def decode_action_plan(self, u_tokens):
        """预测未来长程动作计划。onset 累积 softplus 参数化 ⇒ 沿查询维单调,
        消除查询置换对称(与 tao 版 DecoderHeads.decode_action_plan 同理)。"""
        onset_inc = F.softplus(self.plan_onset(u_tokens)).squeeze(-1)   # [B,K] 正增量
        return {
            "mouse": self.plan_mouse(u_tokens),
            "keyboard": torch.sigmoid(self.plan_keyboard(u_tokens)),
            "onset": torch.cumsum(onset_inc, dim=1),
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
        self.J = J # 历史动作长度(训练/可视化按此构造 a_raw)
        
        # 多模态骨干
        self.vision_encoder = MockDINOv2(d)
        
        # JEPA 目标编码的固定查询锚(buffer 不训练):目标 = binder(slots, patch_next),
        # 只是下一帧观察的函数,与递归状态解耦(消除 binder 增益→0 的平凡不动点)。
        self.register_buffer("slots", torch.randn(1, N, d))
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
        
        # Transformer。dropout=0(PyTorch 默认 0.1):本模型的 mu 直接喂 JEPA 回归损失,
        # train 模式下 dropout 的随机置零+1/(1-p) 重缩放会让 train/eval 前向输出系统性
        # 不一致——已观测到 eval 期 |mu-z_tg| 比训练期大数倍,可视化/评估全部失真。
        # 正则交给 SIGReg 与 patch 掩码,不用 dropout。
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=d*4,
                                           batch_first=True, activation="gelu", dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=4)
        
        # 复合动作
        self.action_enc = nn.Linear(act_dim, d)
        self.heads = MinecraftDecoderHeads(d, num_keyboard_keys=act_dim-2)
        self.inv_dyn = MinecraftInverseDynamicsHead(d, num_keyboard_keys=act_dim-2)
        
        # Placeholders
        self.u_placeholder = nn.Parameter(torch.randn(1, K, d))
        self.text_placeholder = nn.Parameter(torch.randn(1, 1, d))

    @torch.no_grad()
    def encode_target(self, img_next):
        """JEPA 内容目标:vision → binder(固定锚 slots) − 锚,stop-grad。

        不注入绝对时间 PE(数学原因):目标若含 PE(t_next),mu 被迫精确预测纯时钟
        相位——与任务零互信息的 nuisance;且 t_vec 带 [0,1e4] 随机偏移,该分量在
        样本间剧烈变化,抬高早期 loss 地板、占用 slot 容量。时间条件保留在在线侧
        (encode_vision)即可。固定锚与递归状态解耦的原因见 slots 定义处注释。

        减锚(同一 nuisance 论证):binder 输出 = anchor + gate·LN(δ),其中 anchor
        是与帧内容零互信息的常数(逐元素 RMS≈1),而内容项 gate·LN(δ) 的跨帧变化
        远小于它——保留锚则 |mu−z_tg| 几乎全在度量"复现一个静态随机向量",
        persistence 基线(锚自动抵消)≈0 而模型误差被锚主导,预测质量完全不可读;
        inv-dyn 的 (z_tg−Z_enc) 也被同一常数偏置淹没。减锚后目标=纯内容增量,
        pred 损失、persistence 基线、inv-dyn 残差三者在同一坐标系下可比。
        """
        patch = self.vision_encoder(img_next)
        anchor = self.slots.expand(img_next.shape[0], -1, -1)
        return self.binder(anchor, patch) - anchor

    def encode_vision(self, patch_tokens, t_vec, Z_prev):
        """带时间戳注入的视觉编码"""
        # 1. 空间掩码:掩码率逐样本 ~U(0, 0.3),而非恒定 30%。
        #    恒定掩码率下模型从未见过"全 patch"输入,eval(无掩码)成为分布外前向,
        #    binder/Transformer 的输出统计随之偏移;把 0 纳入训练分布后,
        #    eval 行为=掩码率取下确界,train/eval 一致。
        if self.training:
            B, M, d = patch_tokens.shape
            ratio = torch.rand(B, 1, device=patch_tokens.device) * 0.3
            mask = torch.rand(B, M, device=patch_tokens.device) >= ratio
            patch_tokens = patch_tokens * mask.unsqueeze(-1)
            
        # 3. 注入绝对时间戳 (Time Anchoring)
        time_pe = sinusoidal_time_encoding(t_vec, self.d) # [B, 1, d]
        patch_tokens = patch_tokens + time_pe
        
        # 4. 绑定到实体槽
        Z_new = self.binder(Z_prev, patch_tokens)
        return Z_new

    def forward(self, patch_tokens, Z, h, a_raw, t_vec):
        B = patch_tokens.shape[0]
        Z_new = self.encode_vision(patch_tokens, t_vec, Z)
        text_token = self.text_placeholder.expand(B, -1, -1)
        a_tokens = self.action_enc(a_raw)
        u_p = self.u_placeholder.expand(B, -1, -1)
        h_token = h
        
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
        c = torch.sigmoid(out_state[:, :, 2*self.d:3*self.d]) # [B, N, d] 逐维可控闸(注意:tao 版为逐 slot 标量)
        exist_p = torch.sigmoid(out_state[:, :, 3*self.d:])

        action_plan = self.heads.decode_action_plan(out_u)

        return {
            "mu": mu, "sigma": sigma, "c": c, "exist_p": exist_p,
            "Z_out": out_Z,
            "Z_enc": Z_new,   # 纯感知编码(未经 Transformer、未见动作):逆动力学专用,防动作直通作弊
            "h_next": out_h,
            "action_plan": action_plan
        }
