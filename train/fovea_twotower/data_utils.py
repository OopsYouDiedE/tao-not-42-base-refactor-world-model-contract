# -*- coding: utf-8 -*-
"""fovea-twotower 训练域的在线视觉编码与流拼接。

对外接口:
    dino_encode(dino, img_u8, bs) — 冻结 DINOv2(HF)分块前向,取 81 patch token。
    batch_to_stream(batch, dino, dev) — 视觉 + 动作流。
    batch_to_stream_msg(batch, dino, dev) — 额外并入周边消息通道。
"""
import torch

from net.fovea_twotower import act_featurize

# DINO 系骨干的 ImageNet 归一化常数(骨干预处理约定,非领域常量)
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dino_encode(dino, img_u8, bs=256):
    """[N,3,126,126] uint8 → [N,81,384] bf16 patch token(冻结,分块防峰值)。

    HF ``Dinov2Model`` 前向:``last_hidden_state`` 去 CLS(index 0)取 81 patch;
    ``interpolate_pos_encoding=True`` 使 126×126→9×9 的位置编码按插值适配
    (dinov2-small 无 register token,故 ``[:, 1:]`` 恰为 patch 序列)。
    torch.hub 私有加载已废弃,骨干统一由 ``net.backbone.build_backbone`` 提供。
    """
    outs = []
    for i in range(0, img_u8.shape[0], bs):
        x = img_u8[i:i + bs].float().div_(255)
        x = (x - IMN_MEAN.to(x.device)) / IMN_STD.to(x.device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = dino(pixel_values=x, interpolate_pos_encoding=True)
            outs.append(out.last_hidden_state[:, 1:])
    return torch.cat(outs).bfloat16()


def batch_to_stream(batch, dino, dev):
    """batch → (lat [B,L,81,384] bf16, act [B,L,·] bf16)。"""
    img = batch["img"].to(dev, non_blocking=True)      # [B,L,3,S,S] u8
    B, L = img.shape[:2]
    lat = dino_encode(dino, img.flatten(0, 1)).view(B, L, 81, 384)
    act = act_featurize(*(batch[k].to(dev) for k in
                          ("dx", "dy", "keys", "gui", "dt"))).bfloat16()
    return lat, act


def batch_to_stream_msg(batch, dino, dev):
    """batch → (lat, act, msg [B,L,N_MSG] bf16)。"""
    lat, act = batch_to_stream(batch, dino, dev)
    msg = batch["msg"].to(dev, non_blocking=True).bfloat16()
    return lat, act, msg
