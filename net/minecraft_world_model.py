"""Minecraft 世界模型(Δz-JEPA 版)。

本版修复了上一版"训练几乎无效"的四个结构性缺陷(诊断见 2026-06-11 复盘):

1. **预测目标改为 Δz**:旧版预测绝对潜表征 z_{t+1},其中静态场景内容占能量 ~99.8%、
   动力学只占 ~0.2%——模型容量全部花在"重编码当前帧",persistence 基线不可战胜。
   新版预测 Δz = sg[enc(img_{t+1}) − enc(img_t)],persistence 退化为"预测 0",
   动力学占目标能量 100%,动作信息成为唯一可用的预测来源。

2. **在线/目标统一坐标系**:旧版在线 binder 用递归状态 Z_prev 做查询、patch 注入
   时间 PE、目标减锚而在线不减——同一帧两条路径编码结果不同,模型被迫先学一个
   状态相关的坐标变换;inv-dyn 的差信号被失配噪声淹没(信噪比 ~1/20)。新版在线/
   目标共用 encode 形式:binder(固定锚, patch) − 锚,时间 PE 移到 Transformer 的
   h token 上注入,感知输入不被污染。

3. **EMA 目标编码器**:旧版目标与在线共享同一份正在训练的权重(仅 no_grad),
   目标非平稳且唯一梯度压力指向"让目标好预测"(= 抹平视觉差异);SIGReg 施加在
   Z_out 上防不到这条路。新版 encode_target 用 EMA 副本——目标慢速跟踪在线权重,
   BYOL/JEPA 式的稳定靶;SIGReg 由训练侧施加到在线 z_obs(坍缩发生的位置)。

4. **撤掉异方差 σ 支路 + c 改回逐 slot 标量**:σ 自由时 NLL 靠标定残差下降
   (学会"把梯度静音"而非"把误差降低"),pred 曲线失真;先用纯 MSE,μ 学会前
   不给泄压阀。c 旧版逐维([B,N,d],6144 个闸门)靠一个池化标量损失塑形,梯度
   过度弥散;改回 tao 版逐 slot 标量。

训练时序(teacher forcing):每步感知输入 = 在线编码 z_obs(t),μ 预测 Δz(t→t+dt);
跨步记忆只走 h token。开环推演(可视化/推理)用 ẑ(t+dt) = ẑ(t) + μ(t) 累积。

可变 Δt(jumpy prediction):每个转移的跨度 dt ~ U{1..max_skip}(帧)由数据集采样,
模型同时接收 (a) 区间内**完整的原始动作序列** a_cur(信息无损,带有效位区分零填充
与真·无操作)、(b) 聚合动作历史 a_hist、(c) ContinuousTimeEncoding(dt) 条件 token。
数学动机:固定步长允许模型学"默认漂移先验";Δt 可变后,唯一能解释 Δz 的就是把
区间内动作逐个积分——这正是开环推演所需的能力,且动作效应随 Δt 累积而编码噪声
地板不变,大 Δt 样本信噪比更高。
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.primitives import ContinuousTimeEncoding

N_CAMERA_BINS = 11   # 与 utils.vpt_action.CAMERA_BINS 一致(net 层不 import utils,训练端校验)


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
    """从潜变化 ΔZ 反推动作。鼠标 = mu-law 分箱**分类**(logits),键盘 = 20 独立二分类。

    鼠标弃回归改分类的原因:dx/dy 分布是"尖峰在 0 + 重尾大转身",MSE 下恒预测 0
    即近似最优(上一版实测 mouse loss 钉死在边缘方差处一动不动);分类目标下
    基率解的 CE = 边缘熵,任何真实信号都能压过它。分箱定义在 utils.vpt_action
    (camera_to_bin/bin_to_camera),与 VPT 原版的 mu-law 离散相机一致。
    """
    def __init__(self, d, num_keyboard_keys=20, n_cam_bins=N_CAMERA_BINS):
        super().__init__()
        self.n_cam_bins = n_cam_bins
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU()
        )
        self.mouse_out = nn.Linear(64, 2 * n_cam_bins)
        self.kb_out = nn.Linear(64, num_keyboard_keys)

    def forward(self, residual_z):
        # residual_z: [B, N, d] → mouse_logits [B, 2, n_bins], kb_prob [B, n_keys]
        pooled = residual_z.mean(dim=1)
        feat = self.net(pooled)
        mouse_logits = self.mouse_out(feat).view(-1, 2, self.n_cam_bins)
        kb_prob = torch.sigmoid(self.kb_out(feat))
        return mouse_logits, kb_prob


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
    """随机冻结卷积,模拟 DINOv2 输出 Patch Tokens——**仅供无网络的管线冒烟测试**。

    注意:本版中视觉骨干统一冻结(见 extract_feats),mock 即随机特征投影;
    真实训练请用 --encoder dinov2。
    """
    def __init__(self, d=384):
        super().__init__()
        self.embed_dim = d
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8), nn.ReLU(),
            nn.Conv2d(64, d, kernel_size=4, stride=4), nn.ReLU()
        )
    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.net(x) # [B, d, h, w]
        B, d, h, w = feat.shape
        return feat.view(B, d, h*w).transpose(1, 2) # [B, M, d]


def _load_backbone(kind, weights=None):
    """torch.hub 加载冻结视觉骨干。返回 (module, patch_size)。

    dinov2: ViT-S/14,embed_dim=384,权重完全开放,直接下载。
    dinov3: ViT-S/16,embed_dim=384,稠密特征质量显著优于 v2(gram anchoring),
            且 patch=16 整除 128;但权重 **gated**——需在 Meta/HuggingFace 接受
            许可证后获得下载 URL/本地路径,经 weights 参数传入。
    """
    try:
        if kind == "dinov2":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"), 14
        if kind == "dinov3":
            kwargs = {"weights": weights} if weights else {}
            return torch.hub.load("facebookresearch/dinov3", "dinov3_vits16", **kwargs), 16
    except Exception as ex:
        hint = ("DINOv3 权重是 gated 的:在 https://github.com/facebookresearch/dinov3 "
                "接受许可证后,把获得的下载 URL 或本地 .pth 路径经 --encoder_weights 传入。"
                if kind == "dinov3" else "需要网络访问 torch.hub;离线冒烟测试请改用 --encoder mock。")
        raise RuntimeError(f"{kind} 加载失败({ex})。{hint}") from ex
    raise ValueError(f"未知 encoder: {kind}")


class MinecraftWorldModel(nn.Module):
    """Δz-JEPA 世界模型:统一锚坐标系感知 + EMA 目标 + Transformer 动力学推演。"""

    def __init__(self, d=384, N=16, K=5, J=8, act_dim=22, n_cam_bins=N_CAMERA_BINS,
                 ema_decay=0.99, max_skip=8, encoder="dinov2", encoder_weights=None):
        super().__init__()
        self.d = d
        self.N = N # 实体槽数量
        self.K = K # 动作查询数量
        self.J = J # 历史动作长度(训练/可视化按此构造 a_hist)
        self.S = max_skip # 区间动作序列最大长度(= 数据集 frame_skip 上限)
        self.ema_decay = ema_decay
        self.encoder_kind = encoder

        # 视觉骨干:**冻结**的预训练 DINOv2(ViT-S/14)。prime-leaf-7 实测:从零训练
        # 的 mock 卷积在 2 个 clip 上把纹理背熟(train pred 0.19 vs holdout ~1.7×基线,
        # 鼠标读出 holdout ≈ 随机),泛化失败的主因之一是表征没有先验。冻结预训练
        # 骨干给目标编码一个独立于本任务数据的意义来源——这正是 JEPA 的前提。
        # 骨干冻结且在线/目标共享 ⇒ 特征每帧只需提取一次(extract_feats),
        # EMA 只需覆盖可训练部分(proj + binder),省一半视觉计算与 21M 参数副本。
        if encoder in ("dinov2", "dinov3"):
            self.backbone, self._patch = _load_backbone(encoder, encoder_weights)
            enc_dim = self.backbone.embed_dim          # ViT-S = 384
        else:
            self.backbone = MockDINOv2(d)
            enc_dim = d
            self._patch = None
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        # DINOv2 期望 ImageNet 归一化输入(我们的帧是 [0,1])
        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 在线感知:冻结特征 → 可训练投影 → binder(固定锚) − 锚。锚是 buffer 不训练,
        # 在线/目标共用 ⇒ 两条路径编码同一帧得到同一坐标系下的同一向量(至 EMA 滞后)。
        # compete=True:slot 维竞争注意力(防多个 slot 冗余绑定同一区域,见 SlotBinder)。
        self.proj = nn.Linear(enc_dim, d)
        self.register_buffer("slots", torch.randn(1, N, d))
        from net.tao_not_42 import SlotBinder
        self.binder = SlotBinder(d, compete=True)

        # EMA 目标编码器(JEPA 靶):深拷贝可训练部分、不收梯度、不进 optimizer
        # (requires_grad=False ⇒ 优化器过滤,仍随 state_dict 保存/加载)。
        self.proj_ema = copy.deepcopy(self.proj)
        self.binder_ema = copy.deepcopy(self.binder)
        for p in self._ema_params():
            p.requires_grad_(False)

        # 状态解码:μ(Δz 预测, d) + c(逐 slot 标量可控闸, 1) + exist(1)。
        # σ 已撤(异方差 NLL 的"σ 标定残差"泄压阀让 loss 下降与误差脱钩);
        # c 回到逐 slot 标量(逐维 6144 闸门 × 池化标量损失 = 梯度过度弥散)。
        out_dim = d + 2
        self.state_dec = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d*2),
            nn.SiLU(),
            nn.Linear(d*2, out_dim)
        )
        # 末层零初始化:冷启动 μ=0(恰为 persistence 基线 ⇒ 归一化 pred 损失从 1.0
        # 起步,而非 |随机μ|²/|Δz|² ~ 1e4)、c=σ(0)=0.5(闸门居中,等待 inv-dyn 极化)。
        nn.init.zeros_(self.state_dec[-1].weight)
        nn.init.zeros_(self.state_dec[-1].bias)

        # Transformer。dropout=0(PyTorch 默认 0.1):本模型的 mu 直接喂回归损失,
        # train 模式下 dropout 的随机置零+1/(1-p) 重缩放会让 train/eval 前向输出
        # 系统性不一致。正则交给 SIGReg(训练端施加在 z_obs 上),不用 dropout。
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=d*4,
                                           batch_first=True, activation="gelu", dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=4)

        # 复合动作。act_dim+1:末位是**有效位**——区分零填充与真·无操作(全零动作
        # 是合法输入"什么都没按",不能用全零判别 padding)。历史 token 有效位恒 1。
        self.action_enc = nn.Linear(act_dim + 1, d)
        # 区间内动作的序内位置嵌入(次序携带信息:先转头后前进 ≠ 先前进后转头)
        self.act_pos = nn.Parameter(torch.randn(1, max_skip, d) * 0.02)
        # Δt 条件编码(契约:以帧为单位,见 primitives.ContinuousTimeEncoding)
        self.dt_enc = ContinuousTimeEncoding(d)
        self.heads = MinecraftDecoderHeads(d, num_keyboard_keys=act_dim-2)
        self.inv_dyn = MinecraftInverseDynamicsHead(d, num_keyboard_keys=act_dim-2,
                                                    n_cam_bins=n_cam_bins)

        # Placeholders
        self.u_placeholder = nn.Parameter(torch.randn(1, K, d))
        self.text_placeholder = nn.Parameter(torch.randn(1, 1, d))

    def train(self, mode=True):
        """冻结骨干永远保持 eval(drop_path/随机性关闭),其余模块正常切换。"""
        super().train(mode)
        self.backbone.eval()
        return self

    # ---------------- 感知编码(在线 / EMA 目标,同一坐标系) ----------------

    def _ema_params(self):
        for m in (self.proj_ema, self.binder_ema):
            yield from m.parameters()

    @torch.no_grad()
    def extract_feats(self, img):
        """冻结骨干提取 patch 特征(无梯度,在线/目标共用,每帧只算一次)。

        DINOv2:ImageNet 归一化 + 分辨率自动对齐到 patch=14 的倍数(128→126),
        输出 x_norm_patchtokens [B, M, 384]。mock:随机冻结卷积(冒烟用)。
        """
        if self.encoder_kind in ("dinov2", "dinov3"):
            H, W = img.shape[-2:]
            ps = self._patch
            H2, W2 = max(ps, (H // ps) * ps), max(ps, (W // ps) * ps)
            if (H2, W2) != (H, W):
                img = F.interpolate(img, size=(H2, W2), mode="bilinear", align_corners=False)
            img = (img - self._in_mean) / self._in_std
            out = self.backbone.forward_features(img)
            return out["x_norm_patchtokens"] if isinstance(out, dict) else out
        return self.backbone(img)

    def encode_obs(self, img=None, feats=None):
        """在线感知编码(梯度只进 proj+binder):feats → proj → binder(固定锚) − 锚。

        不加时间 PE、不做 patch 掩码:时间条件移到 forward 的 h token 上注入,
        感知输入保持纯内容——这是 inv-dyn 差信号 (z_tg − z_obs) 干净的前提。
        减锚:binder 输出 = 锚 + gate·LN(δ),锚是与内容零互信息的常数,减掉后
        编码 = 纯内容增量,与 encode_target 同坐标系。
        传 feats(extract_feats 的输出)可跳过骨干重复计算。
        """
        if feats is None:
            feats = self.extract_feats(img)
        patch = self.proj(feats)
        anchor = self.slots.expand(feats.shape[0], -1, -1)
        return self.binder(anchor, patch) - anchor

    @torch.no_grad()
    def encode_target(self, img=None, feats=None):
        """JEPA 目标编码:与 encode_obs 同构,但走 EMA 投影/binder + no_grad。

        骨干本身冻结 ⇒ 非平稳性只可能来自 proj/binder,EMA 恰好罩住这两处。
        在线权重的任何坍缩动作要经过 ema_decay 的低通才会出现在目标里,期间
        SIGReg(施加在在线 z_obs 上)与 inv-dyn(要求 Δz 可读出动作)有时间纠偏。
        """
        if feats is None:
            feats = self.extract_feats(img)
        patch = self.proj_ema(feats)
        anchor = self.slots.expand(feats.shape[0], -1, -1)
        return self.binder_ema(anchor, patch) - anchor

    @torch.no_grad()
    def ema_update(self, decay=None):
        """目标编码器 EMA 跟踪在线权重:θ_tg ← τ·θ_tg + (1−τ)·θ。每个优化步后调用。"""
        tau = self.ema_decay if decay is None else decay
        pairs = [(self.proj, self.proj_ema),
                 (self.binder, self.binder_ema)]
        for online, target in pairs:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.lerp_(po.detach(), 1.0 - tau)
            for bo, bt in zip(online.buffers(), target.buffers()):
                bt.copy_(bo)

    # ---------------- 动力学推演 ----------------

    def forward(self, z_ref, h, a_hist, a_cur, dt, t_vec):
        """一段可变跨度的动力学推演:预测 z 从 t 到 t+dt 的增量。

        z_ref:  [B,N,d] 当前帧潜表征(闭环 = encode_obs(img_t);开环 = 上一步 ẑ+μ)。
        a_hist: [B,J,A] 过去 J 个转移的聚合动作(严格过去,当前区间不在内)。
        a_cur:  [B,S,A] 当前区间内完整的原始动作序列(右侧零填充)。
        dt:     [B]     当前转移的帧跨度(决定 a_cur 的有效长度与 Δt 条件)。
        返回 mu = **Δz 预测**(z(t+dt) 的估计 = z_ref + mu),c = 逐 slot 可控闸。
        时间 PE 加在 h token 上(不污染感知);Δt 编码为独立条件 token。
        """
        B = z_ref.shape[0]
        text_token = self.text_placeholder.expand(B, -1, -1)
        u_p = self.u_placeholder.expand(B, -1, -1)
        h_token = h + sinusoidal_time_encoding(t_vec, self.d).to(h.dtype)
        dt_token = self.dt_enc(dt).to(z_ref.dtype).unsqueeze(1)        # [B,1,d]

        ones_h = torch.ones(B, a_hist.shape[1], 1, device=z_ref.device, dtype=a_hist.dtype)
        ah = self.action_enc(torch.cat([a_hist, ones_h], dim=-1))
        S = a_cur.shape[1]
        valid = (torch.arange(S, device=z_ref.device).unsqueeze(0)
                 < dt.unsqueeze(1)).to(a_cur.dtype).unsqueeze(-1)      # [B,S,1]
        ac = self.action_enc(torch.cat([a_cur, valid], dim=-1)) + self.act_pos[:, :S]

        # [N slots, 1 text, 1 h, 1 dt, J hist, S cur, K action_queries]
        X = torch.cat([z_ref, text_token, h_token, dt_token, ah, ac, u_p], dim=1)

        # Transformer 推演
        X = self.blocks(X)

        # 解码
        out_Z = X[:, 0:self.N, :]
        out_h = X[:, self.N+1:self.N+2, :]
        base = self.N + 3 + ah.shape[1] + S
        out_u = X[:, base:base+self.K, :]

        out_state = self.state_dec(out_Z)
        mu = out_state[:, :, :self.d]                                  # Δz 预测
        c = torch.sigmoid(out_state[:, :, self.d:self.d+1])           # [B,N,1] 逐 slot 标量
        exist_p = torch.sigmoid(out_state[:, :, self.d+1:])

        action_plan = self.heads.decode_action_plan(out_u)

        return {
            "mu": mu, "c": c, "exist_p": exist_p,
            "Z_out": out_Z,
            "h_next": out_h,
            "action_plan": action_plan
        }
