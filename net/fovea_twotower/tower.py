# -*- coding: utf-8 -*-
"""凹视双塔 Step1:Context 塔 + Action 塔(net/fovea_twotower/tower.py)。

设计依据 docs/architectures/fovea-twotower-step1.md:
  流 = 每帧 [81 DINO patch token(126×126/14=9×9)] + [1 动作 token],因果交错;
  12 层 = 9×GatedDeltaNet + 3×GQA 因果自注意(每 4 层第 4 个);
  Context 塔损失 = 移一位下 token 潜变量 MSE(只计视觉位;目标过无仿射 LN,MAE 习语);
  Action 塔 = 同骨架,输入 [当前帧 81 视觉 + H 个加噪动作 token],流匹配速度场;
    历史仅经 GDN 状态播种递交(recurrent_state + conv_state),注意力只见本地窗
    ——保持因果依据 nemotron 笔记 Table 2"双向化 Mamba 无收益"。

状态接口:ContextTower.encode(..., want_states=True) → (hidden, [每 GDN 层状态]);
ActionTower.forward(..., seed=状态列表 | None) —— None 即消融组 B1(零状态)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers import GatedDeltaNet
from fla.models.utils import Cache

N_ACT = 24            # dx,dy(对数尺度),20 键,gui,dt/30 —— train/gaming500 聚合语义
N_PATCH = 81
D_LAT = 384           # DINOv2-S 潜变量维


def act_featurize(dx, dy, keys, gui, dt):
    """[...]系列 → [..., 24];位移对数尺度保符号,粗归一到 ~[-1,1]。"""
    f = lambda v: torch.sign(v) * torch.log1p(v.abs()) / 5.0
    return torch.cat([f(dx)[..., None], f(dy)[..., None], keys.float(),
                      gui[..., None].float(), dt[..., None].float() / 30.0], -1)


class _RoPE(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv", inv, persistent=False)

    def forward(self, q, k):                           # [B,H,T,D]
        t = torch.arange(q.shape[2], device=q.device).float()
        fr = torch.outer(t, self.inv)                  # [T,D/2]
        cos = fr.cos().repeat_interleave(2, -1)[None, None]
        sin = fr.sin().repeat_interleave(2, -1)[None, None]
        rot = lambda x: torch.stack([-x[..., 1::2], x[..., ::2]], -1).flatten(-2)
        return (q * cos + rot(q) * sin).to(q.dtype), (k * cos + rot(k) * sin).to(k.dtype)


class CausalAttn(nn.Module):
    """GQA 因果自注意(kv 头 = 头数/2),RoPE。"""

    def __init__(self, d, heads=6):
        super().__init__()
        self.h, self.hk = heads, max(1, heads // 2)
        self.dh = d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * self.hk * self.dh, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = _RoPE(self.dh)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.hk, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, k)
        r = self.h // self.hk
        o = F.scaled_dot_product_attention(
            q, k.repeat_interleave(r, 1), v.repeat_interleave(r, 1), is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    """PreLN 残差块;kind∈{gdn, attn}。GDN 层持有稠密 gdn_idx 供 Cache 寻址。"""

    def __init__(self, d, kind, heads=6, gdn_idx=None):
        super().__init__()
        self.kind, self.gdn_idx = kind, gdn_idx
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mix = (GatedDeltaNet(hidden_size=d, num_heads=heads, mode="chunk",
                                  layer_idx=gdn_idx)
                    if kind == "gdn" else CausalAttn(d, heads))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None):
        if self.kind == "gdn":
            y, _, cache = self.mix(self.n1(x), past_key_values=cache,
                                   use_cache=cache is not None)
        else:
            y = self.mix(self.n1(x))
        x = x + y
        return x + self.mlp(self.n2(x)), cache


def _hybrid_blocks(d, layers, heads):
    blocks, gi = [], 0
    for i in range(layers):
        if (i + 1) % 4 == 0:
            blocks.append(Block(d, "attn", heads))
        else:
            blocks.append(Block(d, "gdn", heads, gdn_idx=gi))
            gi += 1
    return nn.ModuleList(blocks)


class ContextTower(nn.Module):
    """世界塔:交错流下帧潜变量预测;encode() 供探针/播种取状态。

    n_msg>0(step2)时每帧插入 1 个消息 token(周边事件/慢通道注入):
    帧块 = [81 视觉 | 1 消息 | 1 动作],末帧无动作。n_msg=0 与 step1 完全同构。"""

    def __init__(self, d=384, layers=12, heads=6, n_msg=0, aux_msg=0.0):
        super().__init__()
        self.d, self.n_msg, self.aux_msg = d, n_msg, aux_msg
        self.blocks = _hybrid_blocks(d, layers, heads)
        self.vis_in = nn.Sequential(nn.LayerNorm(D_LAT), nn.Linear(D_LAT, d))
        self.act_in = nn.Linear(N_ACT, d)
        self.type_emb = nn.Embedding(2, d)             # 0=视觉 1=动作
        if n_msg:
            self.msg_in = nn.Linear(n_msg, d)
            self.msg_type = nn.Parameter(torch.zeros(d))
            if aux_msg:                                # step2 §4 S4a 二档:显式消息目标
                self.msg_head = nn.Linear(d, n_msg)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, D_LAT)
        self.tgt_norm = nn.LayerNorm(D_LAT, elementwise_affine=False)

    def interleave(self, lat, act, msg=None):
        """lat [B,L,81,384], act [B,L-1,24], msg [B,L,n_msg]|None → [B,T,d]。
        T = L*82-1(无消息)或 L*83-1(带消息);末帧无动作。"""
        B, L = lat.shape[:2]
        v = self.vis_in(lat) + self.type_emb.weight[0]           # [B,L,81,d]
        if msg is not None:
            m = (self.msg_in(msg) + self.msg_type)[:, :, None]   # [B,L,1,d]
            v = torch.cat([v, m], 2)                             # [B,L,82,d]
        if L == 1:
            return v[:, 0]
        a = (self.act_in(act) + self.type_emb.weight[1])[:, :, None]  # [B,L-1,1,d]
        body = torch.cat([v[:, :-1], a], 2).flatten(1, 2)
        return torch.cat([body, v[:, -1]], 1)

    def backbone(self, x, cache=None):
        for blk in self.blocks:
            x, cache = blk(x, cache)
        return x, cache

    def forward(self, lat, act, msg=None):
        """训练:下 token 潜变量 MSE(仅视觉位计损,消息/动作位不作目标)。"""
        B, L = lat.shape[:2]
        x, _ = self.backbone(self.interleave(lat, act, msg))
        h = self.norm(x)
        pred = self.head(h)                            # [B,T,384]
        T = x.shape[1]
        P = N_PATCH + 1 + (1 if msg is not None else 0)  # 帧块周期 82/83
        is_vis = torch.ones(T, dtype=torch.bool, device=x.device)
        is_vis[N_PATCH::P] = False                     # 消息位(有)或动作位
        if msg is not None:
            is_vis[N_PATCH + 1::P] = False             # 动作位
        tgt_flat = torch.zeros(B, T, D_LAT, device=x.device, dtype=pred.dtype)
        tgt_flat[:, is_vis] = self.tgt_norm(lat).flatten(1, 2).to(pred.dtype)
        m = is_vis[1:]                                 # 位置 p 预测 p+1(仅视觉位)
        loss = F.mse_loss(pred[:, :-1][:, m], tgt_flat[:, 1:][:, m])
        if msg is not None and self.aux_msg:           # 消息位预测下一帧消息
            mp = torch.arange(L - 1, device=x.device) * P + N_PATCH
            mpred = self.msg_head(h[:, mp])
            loss = loss + self.aux_msg * F.mse_loss(mpred, msg[:, 1:].to(mpred.dtype))
        return loss

    @torch.no_grad()
    def encode(self, lat, act, msg=None, want_states=False):
        """返回 (hidden [B,T,d], states|None);states=[{recurrent_state,conv_state}]×9。"""
        cache = Cache() if want_states else None
        x, cache = self.backbone(self.interleave(lat, act, msg), cache)
        states = None
        if want_states:
            states = [{"recurrent_state": cache[i]["recurrent_state"],
                       "conv_state": cache[i]["conv_state"]}
                      for i in range(len(cache))]
        return self.norm(x), states


class ActionTower(nn.Module):
    """策略塔:流匹配去噪 H 个未来动作;骨架与 Context 塔同构(可同源初始化)。

    输入序列 = [当前帧 81 视觉 token] + [H 加噪动作 token];
    历史唯一入口 = seed(冻结塔 GDN 状态);seed=None 即消融 B1。
    """

    def __init__(self, d=384, layers=12, heads=6, horizon=8, n_cmd=0):
        super().__init__()
        self.d, self.H = d, horizon
        self.blocks = _hybrid_blocks(d, layers, heads)
        self.vis_in = nn.Sequential(nn.LayerNorm(D_LAT), nn.Linear(D_LAT, d))
        self.act_in = nn.Linear(N_ACT, d)
        self.type_emb = nn.Embedding(2, d)
        self.tau_mlp = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.v_head = nn.Linear(d, N_ACT)              # 速度场
        if n_cmd:                                      # step3:钉住的指令 token
            self.cmd_in = nn.Linear(n_cmd, d)
            self.cmd_type = nn.Parameter(torch.zeros(d))

    def init_from(self, ctx: ContextTower):
        """同源初始化(TwoTower 习语):骨架+嵌入整体拷贝。"""
        self.blocks.load_state_dict(ctx.blocks.state_dict())
        self.vis_in.load_state_dict(ctx.vis_in.state_dict())
        self.act_in.load_state_dict(ctx.act_in.state_dict())
        self.type_emb.load_state_dict(ctx.type_emb.state_dict())

    def _seed_cache(self, seed, detach=True):
        if seed is None:
            return Cache()                             # 空 Cache 仍走 use_cache 路径,
        c = Cache()                                    # 保证两组前向代码路径一致
        for i, st in enumerate(seed):
            r, cv = st["recurrent_state"], st["conv_state"]
            if detach:
                r = r.detach()
                cv = tuple(t.detach() for t in cv)
            c.update(recurrent_state=r, conv_state=cv, layer_idx=i, offset=0)
        return c

    def forward(self, lat_now, x_tau, tau, seed=None, cmd=None):
        """lat_now [B,81,384], x_tau [B,H,24](加噪动作), tau [B] → 速度 [B,H,24]。
        cmd [B,n_cmd]|None:指令嵌入,作为首 token 钉在序列头(注意力可随机访问)。"""
        v = self.vis_in(lat_now) + self.type_emb.weight[0]
        a = (self.act_in(x_tau) + self.type_emb.weight[1]
             + self.tau_mlp(tau[:, None, None]))
        x = torch.cat([v, a], 1)
        if cmd is not None:
            x = torch.cat([(self.cmd_in(cmd) + self.cmd_type)[:, None], x], 1)
        cache = self._seed_cache(seed)
        for blk in self.blocks:
            x, cache = blk(x, cache)
        return self.v_head(self.norm(x[:, -self.H:]))

    @torch.no_grad()
    def sample(self, lat_now, seed=None, steps=4, generator=None, cmd=None):
        """少步 Euler 从噪声积分出动作 chunk [B,H,24]。"""
        B = lat_now.shape[0]
        x = torch.randn(B, self.H, N_ACT, device=lat_now.device,
                        dtype=lat_now.dtype, generator=generator)
        for i in range(steps):
            tau = torch.full((B,), i / steps, device=lat_now.device,
                             dtype=lat_now.dtype)
            x = x + self.forward(lat_now, x, tau, seed, cmd) / steps
        return x
