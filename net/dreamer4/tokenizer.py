"""Dreamer4 帧 tokenizer (net/dreamer4/tokenizer.py)。

对外接口:
    Tokenizer — 把每帧图像编码成一组连续潜 token、并能从 token 还原图像。

从 blocks 组装:ConvEncoder(flatten=False 取空间特征图)→ 线性投影成每空间位置一个 token;
可选 blocks.VectorQuantizer 做离散码本瓶颈。解码经 ConvDecoder 还原图像。
潜 token 网格 = 编码器下采样后的空间分辨率(H/stride^L × W/stride^L)。
"""
import math

import torch
import torch.nn as nn

from blocks import ConvEncoder, ConvDecoder, VectorQuantizer
from net.dreamer4.config import Dreamer4Config


class Tokenizer(nn.Module):
    """帧 ↔ 潜 token 编解码器。

    Args:
        cfg: Dreamer4Config。

    Shape 约定:
        图像 image:  [B, T, C, H, W] float ∈ [0, 1](内部 preprocess 到 [-0.5, 0.5])。
        token tokens: [B, T, S, D],S = 空间 token 数,D = token_dim。
    """

    def __init__(self, cfg: Dreamer4Config):
        super().__init__()
        self.cfg = cfg
        c, h, w = cfg.obs_shape
        self.encoder = ConvEncoder(
            in_channels=c, depths=cfg.enc_depths,
            kernel=cfg.conv_kernel, stride=cfg.conv_stride, flatten=False)
        # 编码后空间网格
        hh, ww = h, w
        for _ in cfg.enc_depths:
            hh = math.ceil(hh / cfg.conv_stride)
            ww = math.ceil(ww / cfg.conv_stride)
        self.grid = (hh, ww)
        self.num_tokens = hh * ww
        enc_ch = cfg.enc_depths[-1]

        self.to_token = nn.Linear(enc_ch, cfg.token_dim)
        self.use_vq = cfg.use_vq
        if cfg.use_vq:
            self.vq = VectorQuantizer(dim=cfg.token_dim, n_embed=cfg.vq_codes)

        self.decoder = ConvDecoder(
            feat_dim=self.num_tokens * cfg.token_dim, out_channels=c,
            depths=cfg.dec_depths, kernel=cfg.conv_kernel,
            stride=cfg.conv_stride, min_res=cfg.dec_min_res)

    @staticmethod
    def preprocess_image(image):
        """[0, 1] → [-0.5, 0.5]。"""
        return image - 0.5

    def encode(self, image):
        """图像 → 潜 token。

        Returns:
            tokens: [B, T, S, D]。
            vq_loss: 标量(use_vq 时为承诺损失,否则 0)。
        """
        feat = self.encoder(self.preprocess_image(image))     # [B, T, C', h', w']
        b, t, ch, hh, ww = feat.shape
        feat = feat.permute(0, 1, 3, 4, 2).reshape(b, t, hh * ww, ch)
        tokens = self.to_token(feat)                          # [B, T, S, D]
        vq_loss = torch.zeros((), device=tokens.device)
        if self.use_vq:
            tokens, _, vq_loss = self.vq(tokens)
        return tokens, vq_loss

    def decode(self, tokens):
        """潜 token → 重建图像 [B, T, C, H, W](值域 [-0.5, 0.5])。"""
        b, t, s, d = tokens.shape
        return self.decoder(tokens.reshape(b, t, s * d))
