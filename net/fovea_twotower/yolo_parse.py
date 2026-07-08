# -*- coding: utf-8 -*-
"""快塔(边缘视觉)YOLO 解析头 + 目标追踪/导航塔(net/fovea_twotower/yolo_parse.py)。

设计(见 knowledge/design_fovea_yolo_fasttower.md §3/§4):
  YoloParseHead — 冻结 YOLO(E) 分割/检出 → 每帧固定 K 个"目标 token"[B,K,PARSE_DIM]
      (类别 + 归一化框中心/大小 + 置信 + 面积≈深度代理)。YOLO 不可导,作冻结感知前端
      (类比冻结 DINO 出 patch token),下游塔在 token 上学习。
  TrackNavTower — goal 向量(慢塔/文本指导)作 query,从 K 个目标槽里**交叉注意选出要
      追踪/前往的目标**,再经因果时序头出动作(相机分箱 + 按键)。这把"慢塔指导选目标、
      快塔执行追踪/导航"落到结构上。

对外接口:
    YoloParseHead(weights, ...).forward(imgs) -> tokens [B,K,PARSE_DIM]
    TrackNavTower(...).forward(tokens, goal, prev_action) -> (cam_logits, key_logits)
    build_tracknav(cfg) 工厂。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from blocks.attention import MHABlock

PARSE_DIM = 7          # [cls/n_cls, cx, cy, w, h, conf, area](空间归一化)


class YoloParseHead(nn.Module):
    """冻结 YOLO(E) → 每帧 top-K 目标 token [B,K,PARSE_DIM]。

    YOLO 前向不可导(NMS/argmax),故 no_grad;下游塔把 token 当固定观测学习。
    text_classes 非空则设开放词表文本 prompt(Minecraft 类,如 ["iron ore","tree"])。
    """

    def __init__(self, weights: str = "runs/checkpoints/yoloe-11l-seg.pt",
                 K: int = 8, conf: float = 0.02, imgsz: int = 640,
                 text_classes=None, device: str = "cpu"):
        super().__init__()
        from ultralytics import YOLOE
        self.model = YOLOE(weights)
        self.n_cls = 80
        if text_classes:
            self.model.set_classes(text_classes, self.model.get_text_pe(text_classes))
            self.n_cls = len(text_classes)
        self.K, self.conf, self.imgsz, self.device_name = K, conf, imgsz, device

    @torch.no_grad()
    def forward(self, imgs) -> torch.Tensor:
        """imgs: [B,3,H,W] float[0,1]/uint8 或 list[HWC uint8] → tokens [B,K,PARSE_DIM]。

        token 按 conf 降序取 top-K,不足补零;空间坐标归一到 [0,1]。
        """
        if torch.is_tensor(imgs):
            if imgs.dtype != torch.uint8:
                imgs = (imgs.clamp(0, 1) * 255).to(torch.uint8)
            imgs = [im.permute(1, 2, 0).cpu().numpy() for im in imgs]   # CHW→HWC list
        res = self.model.predict(imgs, conf=self.conf, imgsz=self.imgsz,
                                 verbose=False, device=self.device_name)
        toks = torch.zeros(len(res), self.K, PARSE_DIM)
        for bi, r in enumerate(res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            H, W = r.orig_shape
            order = torch.argsort(b.conf.cpu(), descending=True)[:self.K]
            for j, i in enumerate(order):
                x1, y1, x2, y2 = b.xyxy[i].cpu()
                toks[bi, j] = torch.tensor([
                    float(b.cls[i]) / self.n_cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                    (x2 - x1) / W, (y2 - y1) / H, float(b.conf[i]),
                    (x2 - x1) * (y2 - y1) / (W * H)])
        return toks


class _FFN(nn.Module):
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(mult * d, d))

    def forward(self, x):
        return x + self.net(self.norm(x))


@dataclass
class TrackNavConfig:
    """TrackNavTower 结构超参(纯 dataclass)。"""
    parse_dim: int = PARSE_DIM
    d: int = 256
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1
    goal_dim: int = 384          # 慢塔/文本指导向量维(MiniLM=384)
    n_mouse: int = 2
    camera_bins: int = 11
    n_keys: int = 20
    max_len: int = 256


class TrackNavTower(nn.Module):
    """goal 选目标 + 因果时序头 → 动作(相机分箱 + 按键)。

    Shapes:
        tokens [B,T,K,parse_dim], goal [B,goal_dim], prev_action [B,T,n_mouse+n_keys]
        → cam_logits [B,T,n_mouse,camera_bins], key_logits [B,T,n_keys]
    机制:goal 投影成 query,对 K 个目标槽做单头交叉注意 → 每帧"被指导选中的目标"表征;
    加上一步动作 + goal 全局偏置 + 位置编码 → 因果 Transformer → 动作头。
    """

    def __init__(self, cfg: TrackNavConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.slot_proj = nn.Linear(cfg.parse_dim, d)
        self.goal_q = nn.Linear(cfg.goal_dim, d)
        self.goal_bias = nn.Linear(cfg.goal_dim, d)
        self.act_embed = nn.Linear(cfg.n_mouse + cfg.n_keys, d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.scale = d ** -0.5
        trunk = []
        for _ in range(cfg.layers):
            trunk.append(MHABlock(d, heads=cfg.heads, causal=True, dropout=cfg.dropout))
            trunk.append(_FFN(d, dropout=cfg.dropout))
        self.trunk = nn.ModuleList(trunk)
        self.out_norm = nn.LayerNorm(d)
        self.cam_head = nn.Linear(d, cfg.n_mouse * cfg.camera_bins)
        self.key_head = nn.Linear(d, cfg.n_keys)

    def forward(self, tokens, goal, prev_action):
        B, T, K, _ = tokens.shape
        d = self.cfg.d
        slots = self.slot_proj(tokens)                     # [B,T,K,d]
        q = self.goal_q(goal)[:, None, None, :].expand(B, T, 1, d)   # 指导 query
        attn = (q @ slots.transpose(-1, -2)) * self.scale  # [B,T,1,K]
        sel = (attn.softmax(-1) @ slots).squeeze(2)        # [B,T,d] 被选目标表征
        x = (sel + self.act_embed(prev_action)
             + self.goal_bias(goal)[:, None, :] + self.pos[:, :T])
        for blk in self.trunk:
            x = blk(x)
        x = self.out_norm(x)
        cam = self.cam_head(x).view(B, T, self.cfg.n_mouse, self.cfg.camera_bins)
        return cam, self.key_head(x)


def build_tracknav(cfg: TrackNavConfig = None) -> TrackNavTower:
    """工厂:按 TrackNavConfig 构建 TrackNavTower(缺省=默认结构)。"""
    return TrackNavTower(cfg or TrackNavConfig())
