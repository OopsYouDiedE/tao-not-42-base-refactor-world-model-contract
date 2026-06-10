"""L1 primitive 积木库。

规格见 knowledge/net_blocks.md。每个 primitive 不可再拆、可独立单测,
数值不变量 I1-I8 焊进实现(用对积木即满足)。命名遵循职责,无 custom_/my_ 前缀。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_FP16 = 1e-4  # I1: fp16 安全 epsilon(绝不用 1e-12)


def _base_grid(h, w, device, dtype=torch.float32):
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0)  # [2,H,W] 顺序 (x,y)


class Warp(nn.Module):
    """局部光流重采样。flow=(dx,dy) 像素位移。坐标 fp32(I4);双线性凸插值 ⇒ 非扩张(I5)。"""

    def forward(self, feat, flow):
        B, C, H, W = feat.shape
        base = _base_grid(H, W, feat.device).unsqueeze(0)            # [1,2,H,W] fp32
        coords = base + flow.float()                                  # fp32, I4
        gx = 2.0 * coords[:, 0] / max(W - 1, 1) - 1.0
        gy = 2.0 * coords[:, 1] / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)                         # [B,H,W,2]
        out = F.grid_sample(feat.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.to(feat.dtype)


class GlobalTransformApply(nn.Module):
    """全局仿射屏幕空间变换。theta:[B,2,3]。fp32 网格(I4),非扩张(I5)。"""

    def forward(self, feat, theta):
        B, C, H, W = feat.shape
        grid = F.affine_grid(theta.float(), (B, C, H, W), align_corners=True)
        out = F.grid_sample(feat.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.to(feat.dtype)


class LocalCorr(nn.Module):
    """有界半径余弦相关。输入先 ℓ2norm(eps=1e-4, I1)。输出 ∈[-1,1]。禁止全局注意力。"""

    def __init__(self, radius=4):
        super().__init__()
        self.r = radius

    def forward(self, a, b):
        r = self.r
        a = F.normalize(a, dim=1, eps=EPS_FP16)
        b = F.normalize(b, dim=1, eps=EPS_FP16)
        B, C, H, W = a.shape
        b_pad = F.pad(b, (r, r, r, r))
        outs = []
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                bs = b_pad[:, :, dy:dy + H, dx:dx + W]
                outs.append((a * bs).sum(dim=1, keepdim=True))
        return torch.cat(outs, dim=1)                                # [B,(2r+1)^2,H,W]


class SoftArgmaxFlow(nn.Module):
    """corr → 期望位移 ∈[-r,r]。fp32 softmax(I4),输出有界(I3)。"""

    def __init__(self, radius=4, tau=1.0):
        super().__init__()
        self.r, self.tau = radius, tau
        offs = [[dx, dy] for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)]
        self.register_buffer("offsets", torch.tensor(offs, dtype=torch.float32))

    def forward(self, corr):
        w = F.softmax(corr.float() / self.tau, dim=1)                # fp32, I4
        off = self.offsets.to(corr.device)                          # [K,2]
        fx = (w * off[None, :, 0, None, None]).sum(1, keepdim=True)
        fy = (w * off[None, :, 1, None, None]).sum(1, keepdim=True)
        return torch.cat([fx, fy], dim=1).to(corr.dtype)            # [B,2,H,W]


class ConvGRUCell(nn.Module):
    """ConvGRU 单元。凸更新 (1-z)h+zn ⇒ 非扩张。"""

    def __init__(self, c):
        super().__init__()
        self.conv_z = nn.Conv2d(2 * c, c, 3, padding=1)
        self.conv_r = nn.Conv2d(2 * c, c, 3, padding=1)
        self.conv_n = nn.Conv2d(2 * c, c, 3, padding=1)

    def forward(self, x, h):
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * n


class GatedResidual(nn.Module):
    """唯一允许的残差形式: x + γ.clamp(±gmax)·f(x)。

    ⚠️ γ 受限是**经验性稳定化**,不是非扩张定理:严格 I5 需 |γ|·Lip(f) ≤ 1,
    而 f(无谱归一化的 MLP/注意力)的 Lipschitz 常数无界。另注意:严格非扩张的
    递归映射是收缩映射 ⇒ 指数遗忘,会杀长期记忆——"稳定但不严格收缩"才是
    工程上正确的点位,勿为追求 I5 字面成立而加谱归一化。
    """

    def __init__(self, submodule, gmax=0.5, init=0.1):
        super().__init__()
        self.f = submodule
        self.gmax = gmax
        self.gamma = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x, *args, **kwargs):
        g = self.gamma.clamp(min=-self.gmax, max=self.gmax)
        return x + g * self.f(x, *args, **kwargs)


class FiLM(nn.Module):
    """条件仿射调制 x*(1+γ)+β。末层零初始 ⇒ 冷启动恒等。"""

    def __init__(self, cond_dim, c, hidden=None):
        super().__init__()
        hidden = hidden or max(cond_dim * 4, c * 2)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * c))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        params = self.mlp(cond)
        while params.dim() < x.dim():
            params = params.unsqueeze(-1)
        gamma, beta = params.chunk(2, dim=1)
        return x * (1 + gamma) + beta


class PreLNAttn(nn.Module):
    """Pre-LN 多头注意 + 残差。mode∈{self,cross}。

    默认 need_weights=False ⇒ 走 PyTorch 融合 SDPA 快路径(need_weights=True 会强制
    物化注意力矩阵、禁用 flash/mem-efficient kernel,全模型显著拖慢)。
    可视化需要注意力图时把 store_attn 置 True:该次前向走慢路径,
    头平均注意力权重存入 last_attn(detach,[B, L_q, L_kv])。
    """

    def __init__(self, d, heads=4, mode="self"):
        super().__init__()
        assert mode in ("self", "cross")
        self.mode = mode
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.store_attn = False
        self.last_attn = None

    def forward(self, q, kv=None):
        if self.mode == "self" or kv is None:
            kv = q
        out, w = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
                           need_weights=self.store_attn)
        if self.store_attn and w is not None:
            self.last_attn = w.detach()
        return q + out


class PositionalEmbed(nn.Module):
    """2D 正弦位置嵌入(sine2d)。返回 [1,d,H,W]。"""

    def __init__(self, d, kind="sine2d"):
        super().__init__()
        assert kind == "sine2d", "ray 模式在 P2 几何头实现"
        assert d % 4 == 0, "d 必须能被 4 整除"
        self.d = d

    def forward(self, h, w, device="cpu"):
        d = self.d
        dq = d // 4
        div = torch.exp(torch.arange(dq, device=device).float()
                        * (-math.log(10000.0) / max(dq, 1)))
        y = torch.arange(h, device=device).float()[:, None] * div[None, :]   # [h,dq]
        x = torch.arange(w, device=device).float()[:, None] * div[None, :]   # [w,dq]
        ey = torch.cat([y.sin(), y.cos()], dim=1)                            # [h,2dq]
        ex = torch.cat([x.sin(), x.cos()], dim=1)                            # [w,2dq]
        emb = torch.zeros(d, h, w, device=device)
        emb[:2 * dq] = ey.t()[:, :, None].expand(2 * dq, h, w)
        emb[2 * dq:] = ex.t()[:, None, :].expand(2 * dq, h, w)
        return emb.unsqueeze(0)


class ProtoDecode(nn.Module):
    """σ(clamp(einsum(coeff,proto),±15))。无参 (I3)。"""

    def forward(self, coeff, proto):
        logit = torch.einsum("bnk,bkhw->bnhw", coeff, proto)
        return torch.sigmoid(logit.clamp(-15.0, 15.0))


class StochLatent(nn.Module):
    """随机潜在。gaussian: reparam + tanh 有界;categorical: 直通估计。采样 fp32(I4),返回 KL≥0。"""

    def __init__(self, in_dim, z_dim, kind="gaussian"):
        super().__init__()
        assert kind in ("gaussian", "categorical")
        self.kind, self.z_dim = kind, z_dim
        self.proj = nn.Linear(in_dim, 2 * z_dim if kind == "gaussian" else z_dim)

    def forward(self, h):
        if self.kind == "gaussian":
            mu, logsig = self.proj(h).float().chunk(2, dim=-1)       # fp32, I4
            logsig = logsig.clamp(-8.0, 8.0)
            sigma = torch.exp(logsig)
            z = mu + sigma * torch.randn_like(mu)                    # reparam
            kl = 0.5 * (mu ** 2 + sigma ** 2 - 2 * logsig - 1).sum(-1)
            return torch.tanh(z).to(h.dtype), kl                     # I3 有界
        logits = self.proj(h).float()
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs.reshape(-1, self.z_dim), 1).reshape(probs.shape[:-1])
        onehot = F.one_hot(idx, self.z_dim).float()
        z = onehot + probs - probs.detach()                         # straight-through
        kl = (probs * (F.log_softmax(logits, -1) + math.log(self.z_dim))).sum(-1)
        return z.to(h.dtype), kl


class SIGReg(nn.Module):
    """Sliced 各向同性高斯正则(防表征坍缩)。

    来源:LeWM(github.com/lucas-maes/le-wm,见其 LICENSE)。
    思路:用随机单位方向把高维 embedding 投影成 1D,再用 Epps-Pulley 经验特征函数检验,
    把每个投影分布钉到标准正态 N(0,1)——目标实部 `E[cos(t·s)]=exp(-t²/2)`、虚部 `E[sin(t·s)]=0`。
    单 GPU、无需 EMA target、无需负样本即可防坍缩。弃 Mamba 递归凸更新后,递归潜序列失去
    结构性防坍缩保险,本正则近乎必需(见 knowledge/mental_world.md §5.2)。统计项,只进 loss、不进前向(I6 精神)。

    Args:
        knots: 梯形积分节点数(t∈[0,3])。
        num_proj: 每次前向重采样的随机投影方向数(Monte-Carlo slicing)。
        eps: 投影方向归一化分母下界(I1,fp16 安全)。

    Shape:
        proj: [G, B, D] —— G 独立分组(各自一份检验,如时间步;无分组传 1), B 样本维(分布在此维上估计), D 特征维。
        返回: 0 维标量 ∈ [0, ∞)。完美高斯 → 小常量(O(1),不随 B 趋 0);坍缩(常量/低秩)→ O(B) 大(判别力在比值,非绝对值)。

    Dtype: 投影与 cos/sin/mean 全程 fp32(I4);窗 `exp(-t²/2)` 有界(I2);归一化分母 clamp(I1)。
    """

    def __init__(self, knots=17, num_proj=1024, eps=EPS_FP16):
        super().__init__()
        self.num_proj = num_proj
        self.eps = eps
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)  # 梯形权重(偶被积函数 ⇒ 内点 ×2)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)                          # N(0,1) 特征函数实部 = 高斯窗(I2 有界)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)              # 积分权重 ⊙ 窗:抑制大 t 噪声

    def forward(self, proj):
        x = proj.float()                                               # I4
        D = x.size(-1)
        A = torch.randn(D, self.num_proj, device=x.device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp(min=self.eps)   # I1:单位方向,分母 clamp
        s_t = (x @ A).unsqueeze(-1) * self.t                           # [G,B,num_proj,knots] fp32
        ecf_re = s_t.cos().mean(-3)                                    # 经验特征函数实部 E_B[cos(t·s)]
        ecf_im = s_t.sin().mean(-3)                                    # 虚部 E_B[sin(t·s)]
        err = (ecf_re - self.phi).square() + ecf_im.square()          # 与 N(0,1) 特征函数之差²
        statistic = (err @ self.weights) * x.size(-2)                 # 梯形积分,×B 为 Epps-Pulley 标度
        return statistic.mean()                                        # 各分组/投影上平均


def rot6d_to_matrix(x, eps=EPS_FP16):
    """6D → SO(3) Gram-Schmidt(eps fp16 安全, I1)。"""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def make_4x4(R, t):
    B = R.shape[0]
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t.reshape(B, 3)
    return T


def box_iou(a, b, kind="iou", eps=1e-7):
    """xyxy 框 IoU/GIoU,fp32(I4)。a,b: [...,4]。"""
    a, b = a.float(), b.float()
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter + eps
    iou = inter / union
    if kind == "iou":
        return iou
    lt_c = torch.min(a[..., :2], b[..., :2])
    rb_c = torch.max(a[..., 2:], b[..., 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1] + eps
    return iou - (area_c - union) / area_c


class BoundedActivation(nn.Module):
    """I3 执行者。depth: exp-clamp; flow: s·tanh; pos: softplus+ε; prob: sigmoid。"""

    def __init__(self, kind, scale=1.5, eps=EPS_FP16, clamp=4.6):
        super().__init__()
        assert kind in ("depth", "flow", "pos", "prob")
        self.kind, self.scale, self.eps, self.clamp = kind, scale, eps, clamp

    def forward(self, x):
        if self.kind == "depth":
            return torch.exp(x.clamp(-self.clamp, self.clamp))
        if self.kind == "flow":
            return self.scale * torch.tanh(x)
        if self.kind == "pos":
            return F.softplus(x) + self.eps
        return torch.sigmoid(x.clamp(-15.0, 15.0))


# =====================================================================
# 2.15-2.17  数值-符号-关系 子底 + 地图投影
# =====================================================================

class Accumulator(nn.Module):
    """NAC/NALU 式精确累加。`reg' = reg + x @ W`,`W = tanh(Ŵ)⊙σ(M̂) ∈(-1,1)` 偏向 {-1,0,1}。

    线性、无饱和 ⇒ 计数/资源**可外推**(对比 MLP 换数值范围即崩)。输出**有意不有界**(计数器需增长)。
    """

    def __init__(self, in_dim, d):
        super().__init__()
        self.W_hat = nn.Parameter(torch.randn(in_dim, d) * 0.1)
        self.M_hat = nn.Parameter(torch.randn(in_dim, d) * 0.1)

    def weight(self):
        return torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)   # ∈(-1,1)

    def forward(self, reg, x):
        return reg + x @ self.weight()


class DiscreteRouter(nn.Module):
    """对 K 分支硬选择,Gumbel-softmax 直通可微。`FiLM` 是软调制,做不了离散切换。"""

    def __init__(self, in_dim, K, tau=1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, K)
        self.tau = tau

    def forward(self, h, hard=True):
        logits = self.proj(h).float()                           # fp32 I4
        if self.training:
            y = F.gumbel_softmax(logits, tau=self.tau, hard=hard)
        else:
            idx = logits.argmax(-1)
            y = F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
        return y, logits.argmax(-1)


class BEVSplat(nn.Module):
    """图像特征经深度抬升 + 位姿 scatter 到俯视 BEV 格(3D→BEV)。坐标 fp32(I4)。

    像素 →(K_inv)射线 →(×depth)相机系点 →(pose)世界点 →(x,z)量化到 BEV 格 → scatter_add。
    scatter_add 守恒:在范围内像素的特征质量被保留。2D 地图退化为 `GlobalTransformApply`,不需它。
    """

    def __init__(self, bev_hw=(64, 64), x_range=(-10.0, 10.0), z_range=(0.0, 20.0)):
        super().__init__()
        self.Hb, self.Wb = bev_hw
        self.x_range, self.z_range = x_range, z_range

    def forward(self, feat, depth, K_inv, pose):
        # feat[B,C,H,W] depth[B,1,H,W] K_inv[B,3,3] pose[B,4,4](cam→world)
        B, C, H, W = feat.shape
        v, u = torch.meshgrid(torch.arange(H, device=feat.device, dtype=torch.float32),
                              torch.arange(W, device=feat.device, dtype=torch.float32),
                              indexing="ij")
        pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)         # [H,W,3]
        rays = torch.einsum("bij,hwj->bhwi", K_inv.float(), pix)      # [B,H,W,3] fp32
        pts_cam = rays * depth.permute(0, 2, 3, 1).float()           # [B,H,W,3]
        pts_world = (torch.einsum("bij,bhwj->bhwi", pose[:, :3, :3].float(), pts_cam)
                     + pose[:, :3, 3].float()[:, None, None, :])      # [B,H,W,3]
        x, z = pts_world[..., 0], pts_world[..., 2]
        gx = ((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.Wb).long()
        gz = ((z - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * self.Hb).long()
        valid = (gx >= 0) & (gx < self.Wb) & (gz >= 0) & (gz < self.Hb)
        idx = (gz.clamp(0, self.Hb - 1) * self.Wb + gx.clamp(0, self.Wb - 1)).view(B, -1)
        feat_flat = (feat * valid.unsqueeze(1)).flatten(2)           # [B,C,HW],无效像素清零
        bev = feat.new_zeros(B, C, self.Hb * self.Wb)
        bev.scatter_add_(2, idx.unsqueeze(1).expand(B, C, H * W), feat_flat)
        return bev.view(B, C, self.Hb, self.Wb)


class ContinuousTimeEncoding(nn.Module):
    """连续时间编码 τ(Δt)。对 Δt 连续可导, fp32(I4)。

    ⚠️ 单位契约:Δt 必须以**帧**为单位喂入(预测跨度 / 距上次观测的帧数),不能传秒。
    频率组 div∈[1, 1e-4] 按整数帧量程标定(同原版 Transformer 正弦 PE);传秒级小量(如 0.05)
    会让所有通道角度趋近 0 ⇒ sin≈0、cos≈1,编码退化成常量、低频通道失效。
    见 knowledge/mental_world.md §3。
    """

    def __init__(self, d):
        super().__init__()
        assert d % 2 == 0
        self.d = d
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        self.register_buffer("div", div)

    def forward(self, dt):
        dt = dt.float().view(-1, 1)
        angles = dt * self.div.unsqueeze(0)
        emb = torch.zeros(dt.shape[0], self.d, device=dt.device, dtype=torch.float32)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


class SpatialPosEmbed(nn.Module):
    """连续 (x, y, scale) 点坐标的 Fourier 位置编码。

    给注视裁剪(fovea)token 贴**全局位置**:多频段 Fourier 特征 + 线性投影 → [B,d],
    加到观测 token 上,使脑内世界知道"这片高清内容来自世界的哪儿、多近"。
    缺它则脑子只拿到内容、不知方位,无法把观测摆回世界。尺度取对数(乘性 ⇒ 加性)。

    Args:
        d: 输出维度(与 token 维一致)。
        num_bands: 几何频段数 2^0·π … 2^(num_bands-1)·π。

    Shape:
        x, y, s: [B](或可广播到 [B])。x,y∈ 归一化坐标(约 [-1,1]),s∈(0,1] 缩放。
        return: [B, d]。

    Dtype: 坐标 fp32(I4);log(s) 前 clamp(I1)。
    """

    def __init__(self, d, num_bands=8, eps=EPS_FP16):
        super().__init__()
        self.eps = eps
        freqs = (2.0 ** torch.arange(num_bands, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(3 * num_bands * 2, d)

    def forward(self, x, y, s):
        coords = torch.stack([
            x.float(), y.float(),
            torch.log(s.float().clamp(min=self.eps)),    # I1:log 前 clamp;尺度乘性 ⇒ 取对数
        ], dim=-1)                                        # [B,3] fp32 (I4)
        ang = coords.unsqueeze(-1) * self.freqs           # [B,3,num_bands]
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)  # [B, 3·2·num_bands]
        return self.proj(feat).to(x.dtype if torch.is_floating_point(x) else torch.float32)

