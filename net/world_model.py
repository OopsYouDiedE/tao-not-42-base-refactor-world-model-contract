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
from net.config import ModelConfig
from net.slots import build_binder
from net.backbone import build_backbone
from net.dynamics import build_dynamics
from net.heads import DecoderHeads, InverseDynamicsHead


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


class MinecraftWorldModel(nn.Module):
    """Δz-JEPA 世界模型:统一锚坐标系感知 + EMA 目标 + Transformer 动力学推演。"""

    def __init__(self, config, backbone=None):
        """config: ModelConfig(结构超参,默认值 = 今日写死值,见 net.config)。
        backbone: 非空 = 依赖注入的冻结骨干(仅测试 mock,AGENTS §2 只许在 tests/;须自带
                  .embed_dim);为空则按 config.backbone 经 build_backbone 加载。
        """
        super().__init__()
        self.config = config
        self.d = config.d
        self.N = config.N          # 实体槽数量
        self.K = config.K          # 动作查询数量
        self.J = config.J          # 历史动作长度(训练/可视化按此构造 a_hist)
        self.S = config.max_skip   # 区间动作序列最大长度(= 数据集 frame_skip 上限)
        self.ema_decay = config.ema_decay
        act_dim = config.act_dim
        # 局部别名:下文沿用裸名,装配逻辑一字不改;各部件超参来自 config。
        d, N, K, max_skip = self.d, self.N, self.K, self.S
        n_cam_bins = config.heads.n_cam_bins
        inv_dyn_ctx = config.heads.inv_dyn_ctx
        d_xi = config.xi.d_xi

        # 视觉骨干:**冻结**的预训练 ViT(默认 DINOv3 ViT-S/16,见 net.backbone)。冻结预训练
        # 骨干给目标编码一个独立于本任务数据的意义来源(JEPA 前提);在线/目标共享 ⇒ 特征每帧
        # 只提取一次(extract_feats),EMA 只覆盖可训练部分(proj + binder)。
        # backbone 非空 = 依赖注入(测试 mock,kind="injected" → extract_feats 走 mock 分支)。
        self.backbone, self._patch, enc_dim, self._n_reg, self.encoder_kind = \
            build_backbone(config.backbone, injected=backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        # DINOv2/v3 期望 ImageNet 归一化输入(我们的帧是 [0,1];与 HF processor 同款常数)
        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 在线感知:冻结特征 → 可训练投影 → binder(固定锚) − 锚。锚是 buffer 不训练,
        # 在线/目标共用 ⇒ 两条路径编码同一帧得到同一坐标系下的同一向量(至 EMA 滞后)。
        # binder 见 config.encoder(默认 competitive:slot 维竞争注意力,防多 slot 冗余绑定同区)。
        self.proj = nn.Linear(enc_dim, d)
        self.register_buffer("slots", torch.randn(1, N, d))
        self.binder = build_binder(config.encoder, d)

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
        hidden = d * config.state_dec_mult
        self.state_dec = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
        # 末层零初始化:冷启动 μ=0(恰为 persistence 基线 ⇒ 归一化 pred 损失从 1.0
        # 起步,而非 |随机μ|²/|Δz|² ~ 1e4)、c=σ(0)=0.5(闸门居中,等待 inv-dyn 极化)。
        nn.init.zeros_(self.state_dec[-1].weight)
        nn.init.zeros_(self.state_dec[-1].bias)

        # 动力学核(默认 Transformer,层数/头数/ffn/dropout 见 config.dynamics)。
        # dropout=0:mu 直喂回归损失,train/eval 前向须一致;正则交 SIGReg,见 net.dynamics。
        self.blocks = build_dynamics(config.dynamics, d)

        # 复合动作。act_dim+1:末位是**有效位**——区分零填充/空槽与真·无操作
        # (全零动作是合法输入"什么都没按",不能用全零判别 padding)。
        # a_cur 的有效位由 dt 推出;a_hist 的由训练侧滚动维护(开头空槽 = 0)。
        # 同帧并发动作(w+attack+鼠标)合在一个向量的正交维度上——multi-hot 无
        # 叠加损失,不拆多 token(拆开徒增同帧置换对称与 token 数)。
        self.action_enc = nn.Linear(act_dim + 1, d)
        # 区间内动作的序内位置嵌入(次序携带信息:先转头后前进 ≠ 先前进后转头;
        # 区间内逐帧等距,序数 = 相对当前时刻的帧偏移,即已是时间编码)
        self.act_pos = nn.Parameter(torch.randn(1, max_skip, d) * 0.02)
        # 历史动作不用序数位置嵌入而用**时间编码**(dt_enc(t_hist),单位帧):
        # 可变 Δt 下序数定位不了"多久之前"——第 j 个历史槽可能是 3 帧前也可能是
        # 24 帧前;历史条目的位置必须由"距当前推算时刻的帧数"给出。
        # Δt 条件编码(契约:以帧为单位,见 primitives.ContinuousTimeEncoding)
        self.dt_enc = ContinuousTimeEncoding(d)
        self.heads = DecoderHeads(d, num_keyboard_keys=act_dim-2,
                                  n_cam_bins=n_cam_bins)
        self.inv_dyn = InverseDynamicsHead(d, num_keyboard_keys=act_dim-2,
                                           n_cam_bins=n_cam_bins, enc_dim=enc_dim,
                                           use_ctx=inv_dyn_ctx)

        # Placeholders
        self.u_placeholder = nn.Parameter(torch.randn(1, K, d))
        self.text_placeholder = nn.Parameter(torch.randn(1, 1, d))
        # 任务文本条件:冻结句向量(utils.task_text,384 维)→ 线性投影 → text token。
        # 不传 task_emb 时回退 placeholder(可学常数,等价"无条件")。
        self.task_proj = nn.Linear(384, d)

        # 随机隐变量 ξ(Dreamer 式,确定性开环修补的结构性接班人):Δz 中"转身
        # 揭示的新内容"本质不可预测,确定性 μ 只能输出模式平均/摆烂(实证:α 课程
        # 到纯 ẑ 后 pred_open 仍回 0.99)。ξ 给不可预测部分一个**有价格的去处**:
        #   后验 q(ξ|ctx, Δz):训练时看真实 Δz;先验 p(ξ|ctx):只看当下;
        #   KL(q‖p) 把两者拉近——β_kl 是通道的价格;
        #   μ = f(z, a, ξ):闭环训练用后验采样(主干不再为不知道的事赔钱),
        #     开环/eval 用先验均值(诚实口径),想象/推演从先验采样。
        # xi_proj 零初始化:通道从静默开始,有利可图才被打开(KL 曲线 = 用量计)。
        # ⚠ 后验视图(deep-yogurt-28 复盘修):旧版用 Δz.mean(slots) ——"新内容"
        # 恰是**槽级**局部细节,跨 16 槽一平均互相抵消 ⇒ 通道无信息可装(kl<free,
        # 闲置)。改为逐槽 φ 投影(d→phi)后 mean+max 双池化:保留槽级新奇,
        # 作弊带宽仍由 phi/d_xi/KL 把守(max 池化专门保住"某个槽突现的意外")。
        self.d_xi = d_xi
        phi = config.xi.phi if config.xi.phi is not None else max(8, d // 8)
        self.xi_dz_phi = nn.Linear(d, phi)            # 逐槽 Δz → phi(降维限带宽)
        self.xi_prior_net = nn.Sequential(
            nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, 2 * d_xi))
        self.xi_post_net = nn.Sequential(
            nn.Linear(3 * d + 2 * phi, d), nn.SiLU(), nn.Linear(d, 2 * d_xi))
        self.xi_proj = nn.Linear(d_xi, d)
        nn.init.zeros_(self.xi_proj.weight)
        nn.init.zeros_(self.xi_proj.bias)

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

        DINOv2/v3(HF):ImageNet 归一化 + 分辨率对齐到 patch 的倍数(v3 patch16:128→128;
        v2 patch14:128→126),取 last_hidden_state 切掉 CLS+register → [B, M, enc_dim]。
        mock:随机冻结卷积(冒烟用)。
        """
        if self.encoder_kind in ("dinov2", "dinov3"):
            H, W = img.shape[-2:]
            ps = self._patch
            H2, W2 = max(ps, (H // ps) * ps), max(ps, (W // ps) * ps)
            if (H2, W2) != (H, W):
                img = F.interpolate(img, size=(H2, W2), mode="bilinear", align_corners=False)
            img = (img - self._in_mean) / self._in_std
            lhs = self.backbone(pixel_values=img).last_hidden_state  # [B, 1+n_reg+M, enc_dim]
            return lhs[:, 1 + self._n_reg:, :]                       # 切 CLS + register → 纯 patch
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

    # ---------------- 随机隐变量 ξ ----------------

    def _xi_ctx(self, z_ref, h, dt):
        """ξ 先验/后验共用的上下文特征:[B, 3d](槽池化 + 记忆 + Δt 编码)。"""
        return torch.cat([z_ref.mean(dim=1), h.squeeze(1),
                          self.dt_enc(dt).to(z_ref.dtype)], dim=-1)

    def xi_prior(self, z_ref, h, dt):
        """p(ξ|当下) → (mu, logvar)。开环/eval 用均值,想象推演用采样。"""
        o = self.xi_prior_net(self._xi_ctx(z_ref, h, dt))
        return o[:, :self.d_xi], o[:, self.d_xi:].clamp(-6.0, 3.0)

    def xi_posterior(self, z_ref, h, dt, dz_tg):
        """q(ξ|当下, Δz) → (mu, logvar)。仅训练用。

        Δz 视图:逐槽 φ 投影后 mean+max 双池化——保留**槽级**新奇(新内容是
        局部细节,跨槽平均会抵消;max 专门保住某个槽突现的意外),带宽由 phi 把守。
        """
        phi = self.xi_dz_phi(dz_tg.to(z_ref.dtype))      # [B,N,phi]
        dz_view = torch.cat([phi.mean(dim=1), phi.amax(dim=1)], dim=-1)   # [B,2phi]
        ctx = torch.cat([self._xi_ctx(z_ref, h, dt), dz_view], dim=-1)
        o = self.xi_post_net(ctx)
        return o[:, :self.d_xi], o[:, self.d_xi:].clamp(-6.0, 3.0)

    @staticmethod
    def xi_sample(mu, logvar):
        """重参数化采样(梯度可穿)。"""
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    # ---------------- 动力学推演 ----------------

    def forward(self, z_ref, h, a_hist, a_cur, dt, t_vec, t_hist=None, hist_valid=None,
                task_emb=None, xi=None):
        """一段可变跨度的动力学推演:预测 z 从 t 到 t+dt 的增量。

        **时间锚契约**:t_vec 是本次前向的"现在"——脑内世界以它(正弦 PE,注入
        h token)自定位;它同时是一切前向输出的 0 时刻(动作规划头的 onset 从
        这一刻起算,单位 = 帧,与 dt 同单位)。历史/当前区间的定位都相对此刻:
        t_hist 给出各历史条目距"现在"的帧数,a_cur 的序数位置 = 未来帧偏移。

        z_ref:  [B,N,d] 当前帧潜表征(闭环 = encode_obs(img_t);开环 = 上一步 ẑ+μ)。
        a_hist: [B,J,A] 过去 J 个转移的聚合动作(严格过去,当前区间不在内)。
        t_hist: [B,J]   各历史转移**结束时刻**距"现在"的帧数(0 = 刚刚结束;
                None = 全 0,仅兼容旧调用)。可变 Δt 下序数定位不了时间,
                历史 token 的位置编码必须由它给出(dt_enc,与 dt 同单位)。
        hist_valid: [B,J] 历史有效位(0 = 序列开头尚未填充的空槽;None = 全 1)。
                全零动作是合法的"什么都没按",不能用全零判别空槽。
        a_cur:  [B,S,A] 当前区间内完整的原始动作序列(右侧零填充)。
        dt:     [B]     当前转移的帧跨度(决定 a_cur 的有效长度与 Δt 条件)。
        task_emb: [B,384] 冻结任务文本句向量(None = 无条件 placeholder)。
                条件是否被利用取决于数据:单任务数据下文本是常数(零互信息),
                四任务混采时它解释任务间行为方差,并给 plan 头坍缩意图多峰。
        xi:     [B,d_xi] 随机隐变量(None = 先验均值,即 eval/开环的诚实默认;
                训练闭环传后验采样,想象推演传先验采样)。
        返回 mu = **Δz 预测**(z(t+dt) 的估计 = z_ref + mu),c = 逐 slot 可控闸。
        时间 PE 加在 h token 上(不污染感知);Δt 编码为独立条件 token。
        """
        B = z_ref.shape[0]
        J = a_hist.shape[1]
        text_token = (self.text_placeholder.expand(B, -1, -1) if task_emb is None
                      else self.task_proj(task_emb.to(z_ref.dtype)).unsqueeze(1))
        u_p = self.u_placeholder.expand(B, -1, -1)
        h_token = h + sinusoidal_time_encoding(t_vec, self.d).to(h.dtype)
        dt_token = self.dt_enc(dt).to(z_ref.dtype).unsqueeze(1)        # [B,1,d]

        if t_hist is None:
            t_hist = torch.zeros(B, J, device=z_ref.device)
        if hist_valid is None:
            hist_valid = torch.ones(B, J, device=z_ref.device, dtype=a_hist.dtype)
        ah = self.action_enc(torch.cat([a_hist, hist_valid.unsqueeze(-1)], dim=-1)) \
            + self.dt_enc(t_hist).to(z_ref.dtype).view(B, J, self.d)
        S = a_cur.shape[1]
        valid = (torch.arange(S, device=z_ref.device).unsqueeze(0)
                 < dt.unsqueeze(1)).to(a_cur.dtype).unsqueeze(-1)      # [B,S,1]
        ac = self.action_enc(torch.cat([a_cur, valid], dim=-1)) + self.act_pos[:, :S]

        if xi is None:
            xi = self.xi_prior(z_ref, h, dt)[0]          # 诚实默认:先验均值
        xi_token = self.xi_proj(xi.to(z_ref.dtype)).unsqueeze(1)

        # [N slots, 1 text, 1 h, 1 dt, 1 xi, J hist, S cur, K action_queries]
        X = torch.cat([z_ref, text_token, h_token, dt_token, xi_token, ah, ac, u_p],
                      dim=1)

        # Transformer 推演
        X = self.blocks(X)

        # 解码
        out_Z = X[:, 0:self.N, :]
        out_h = X[:, self.N+1:self.N+2, :]
        base = self.N + 4 + ah.shape[1] + S
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
