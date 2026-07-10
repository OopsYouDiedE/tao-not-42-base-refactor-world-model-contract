# -*- coding: utf-8 -*-
"""TokenPolicyTower:v2 目标结构的骨架(2026-07-10 设计文档 §3.1/§7)。

替代 FiLM+flatten 的条件机制:
    KV = [ 视觉 token(DINO patch 或 YOLOE 提案,探针门控,§8) ]
       ⊕ [ 地图 token(net/map_io.MapReader) ]
       ⊕ [ 语言 token(subgoal 原文 UTF-8 字节嵌入,从零学——A1:开放码本,
            无 MiniLM 反义词坍缩;grounding 靠 hindsight relabel 的 BC 数据) ]
       ⊕ [ prev 动作 token ]
    Q  = n_q 个可学策略 query(+ 可选 goal 数值向量投影,如 aim/钉点几何)
    cross-attention(Q,KV) → 汇聚 → 动作头(mu-law 11 bin CE + 20 Bernoulli,口径不动,
    key_prior 先验注入与 PixelTower 一致)。

单 tick 反应式(T=1;时序记忆归地图与慢塔,承 D1 裁决);帧间速度由视觉前端的
帧堆叠/多帧 token 承担。**建成未接线**:接线以 §8 探针裁决视觉前端后进行
(登记于 knowledge/status_built_not_wired.md)。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def encode_utf8(texts: list[str], max_len: int = 48) -> torch.Tensor:
    """字节级 tokenizer:UTF-8 bytes + 1(0 作 pad)。[B, max_len] long。开放码本,零外部依赖。"""
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, s in enumerate(texts):
        b = s.encode("utf-8")[:max_len]
        out[i, :len(b)] = torch.tensor(list(b), dtype=torch.long) + 1
    return out


@dataclass
class TokenTowerConfig:
    d: int = 256
    heads: int = 4
    n_q: int = 4              # 策略 query 数
    vis_dim: int = 384        # 视觉 token 维(DINOv3 ViT-S patch=384;YOLOE 提案=518)
    map_dim: int = 64         # 地图 token 维(MapReader.d_out)
    geo_dim: int = 16         # 数值 goal:aim_uv(2) ⊕ 钉点 xy/half+age(3) ⊕
                              # SelfCalib.physics_vector(10:增益/FOV/步速/延迟/模式,
                              # 各带有效位) ⊕ 备用(1)。物理参数是自标定测出的环境状态,
                              # 喂 query 而非写死进权重(多游戏迁移的载体)。
    lang_vocab: int = 257     # UTF-8 字节 + pad
    lang_len: int = 48
    n_mouse: int = 2
    camera_bins: int = 11
    n_keys: int = 20
    key_prior: float = 0.05


class TokenPolicyTower(nn.Module):
    """forward(vis[B,Nv,vis_dim], map_t[B,Nm,map_dim], lang[B,L]long, geo[B,geo_dim],
    prev[B,n_mouse+n_keys]) → cam[B,n_mouse,bins], key[B,n_keys]。

    各 token 组各自线性投影到 d 并加组嵌入;缺哪组传空张量([B,0,·])即可。
    """

    def __init__(self, cfg: TokenTowerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.vis_in = nn.Linear(cfg.vis_dim, d)
        self.map_in = nn.Linear(cfg.map_dim, d)
        self.lang_emb = nn.Embedding(cfg.lang_vocab, d, padding_idx=0)
        self.prev_in = nn.Linear(cfg.n_mouse + cfg.n_keys, d)
        self.group_emb = nn.Parameter(torch.zeros(4, d))     # vis/map/lang/prev
        nn.init.trunc_normal_(self.group_emb, std=0.02)
        self.queries = nn.Parameter(torch.zeros(cfg.n_q, d))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.geo_in = nn.Linear(cfg.geo_dim, d)
        self.xattn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.mix = nn.Sequential(nn.Linear(cfg.n_q * d, d), nn.GELU(), nn.Linear(d, d))
        self.cam_head = nn.Linear(d, cfg.n_mouse * cfg.camera_bins)
        self.key_head = nn.Linear(d, cfg.n_keys)
        with torch.no_grad():                                # 与 PixelTower 同款先验注入
            p = cfg.key_prior
            self.key_head.bias.fill_(float(torch.log(torch.tensor(p / (1 - p)))))

    def forward(self, vis: torch.Tensor, map_t: torch.Tensor, lang: torch.Tensor,
                geo: torch.Tensor, prev: torch.Tensor):
        b = prev.shape[0]
        c = self.cfg
        kv = torch.cat([
            self.vis_in(vis) + self.group_emb[0],
            self.map_in(map_t) + self.group_emb[1],
            self.lang_emb(lang) + self.group_emb[2],
            (self.prev_in(prev) + self.group_emb[3])[:, None],
        ], dim=1)
        pad = torch.zeros(b, kv.shape[1], dtype=torch.bool, device=kv.device)
        if lang.numel():                                     # 语言 pad 不参与注意力
            off = vis.shape[1] + map_t.shape[1]
            pad[:, off:off + lang.shape[1]] = lang == 0
        q = self.queries[None].expand(b, -1, -1) + self.geo_in(geo)[:, None]
        out, _ = self.xattn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                            key_padding_mask=pad)
        h = self.mix(out.reshape(b, -1))
        return (self.cam_head(h).view(b, c.n_mouse, c.camera_bins),
                self.key_head(h))


def build_token_tower(cfg: TokenTowerConfig) -> TokenPolicyTower:
    return TokenPolicyTower(cfg)
