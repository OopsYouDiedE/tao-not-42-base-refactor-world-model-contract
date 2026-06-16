"""L1 primitive 积木库 - 时空与位置编码 (blocks/encodings.py)"""
import math

import torch
import torch.nn as nn

EPS_FP16 = 1e-4  # I1: fp16 安全 epsilon(绝不用 1e-4)


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
