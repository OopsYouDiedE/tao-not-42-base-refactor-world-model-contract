# -*- coding: utf-8 -*-
"""W4:预训练慢塔(NVIDIA Nemotron-3-Nano-4B)接管世界模型 —— fovea-twotower-step4 §1。

与 W1 唯一的科学差异 = 主干初始化:从零 58M → 20T-token 预训练的 4B(21×Mamba2 +
4×Attn + 17×MLP, d=3136)。其余全同 W1:同数据、同交错流 [81 vis|1 msg|1 act]/帧、
同下帧潜变量 MSE(tgt_norm 域)、同 aux_msg 可选。可训参数 = 输入投影(384→3136)+
回归头(3136→384)+ LoRA(Mamba2 in/out_proj 与 Attn q/k/v/o_proj, r=16);4B 主干冻结。

S7a 的判据(step4 §2)靠此塔冻结后跑 eval_s4 同款探针;本脚本只负责训练与存 ckpt。

用法(队列腾出 GPU 后):
    PYTHONPATH=. python train/fovea_twotower/train_w4.py \
        --out runs/ftt_w4 --steps 6000 --seq 64 --accum 4
冒烟(小步、短序列,验证前向/反向通路):
    PYTHONPATH=. python train/fovea_twotower/train_w4.py --smoke
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.fovea_twotower.train_r1 import batch_to_stream
from train.gaming500.dataset import Gaming500Dataset, N_MSG

N_ACT = 24
N_PATCH = 81
D_LAT = 384
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
# LoRA 目标(已核对 modeling_nemotron_h.py:Mamba2Mixer=in/out_proj, Attention=q/k/v/o_proj)
LORA_TARGETS = ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


def batch_to_stream_msg(batch, dino, dev):
    lat, act = batch_to_stream(batch, dino, dev)
    msg = batch["msg"].to(dev, non_blocking=True).bfloat16()
    return lat, act, msg


class W4Adapter(nn.Module):
    """冻结 Nemotron 主干 + 可训投影/回归头/LoRA。交错与掩码逐行复刻 ContextTower,
    仅把主干换成经 inputs_embeds 调用的 4B。backbone 只取 last_hidden_state。"""

    def __init__(self, backbone, d, n_msg=N_MSG, aux_msg=0.0):
        super().__init__()
        self.backbone = backbone            # NemotronHModel(已 LoRA 包裹, 主干冻结)
        self.d, self.n_msg, self.aux_msg = d, n_msg, aux_msg
        self.vis_in = nn.Sequential(nn.LayerNorm(D_LAT), nn.Linear(D_LAT, d))
        self.act_in = nn.Linear(N_ACT, d)
        self.type_emb = nn.Embedding(2, d)              # 0=视觉 1=动作
        self.msg_in = nn.Linear(n_msg, d)
        self.msg_type = nn.Parameter(torch.zeros(d))
        self.head = nn.Linear(d, D_LAT)
        self.tgt_norm = nn.LayerNorm(D_LAT, elementwise_affine=False)
        if aux_msg:
            self.msg_head = nn.Linear(d, n_msg)

    def interleave(self, lat, act, msg):
        """lat [B,L,81,384], act [B,L-1,24], msg [B,L,n_msg] → [B,T,d](逐行同 tower)。"""
        B, L = lat.shape[:2]
        v = self.vis_in(lat) + self.type_emb.weight[0]
        m = (self.msg_in(msg) + self.msg_type)[:, :, None]
        v = torch.cat([v, m], 2)                        # [B,L,82,d]
        a = (self.act_in(act) + self.type_emb.weight[1])[:, :, None]  # [B,L-1,1,d]
        body = torch.cat([v[:, :-1], a], 2).flatten(1, 2)
        return torch.cat([body, v[:, -1]], 1)

    def _run_backbone(self, emb):
        """emb [B,T,d] → last_hidden_state [B,T,d];关 cache 走全序列前向。"""
        out = self.backbone(inputs_embeds=emb, use_cache=False,
                            return_dict=True)
        return out.last_hidden_state

    @torch.no_grad()
    def encode(self, lat, act, msg, want_states=False):
        """返回 (hidden [B,T,d], states|None)。states = 21 层 Mamba2 的最终 ssm_state
        列表(供 S7a STATE 探针);走带 cache 的 prefill 路径以令内核写回 ssm_state。
        act 可为空([B,0,24])→ 单帧编码(FRAME 臂,无历史)。"""
        emb = self.interleave(lat, act, msg).to(next(self.head.parameters()).dtype)
        if not want_states:
            return self._run_backbone(emb), None
        base = self.backbone.base_model.model      # peft 包裹下的 NemotronHModel
        B, T = emb.shape[0], emb.shape[1]
        # 用模型自带的混合缓存类,保证 prefill 分支写回 ssm_state
        from importlib import import_module
        mod = import_module(type(base).__module__)
        cache = mod.HybridMambaAttentionDynamicCache(
            base.config, B, dtype=emb.dtype, device=emb.device)
        cpos = torch.arange(T, device=emb.device)
        out = self.backbone(inputs_embeds=emb, use_cache=True,
                            past_key_values=cache, cache_position=cpos,
                            return_dict=True)
        states = [s for s in cache.ssm_states
                  if s is not None and s.numel() > 0 and s.dim() >= 2]
        return out.last_hidden_state, states

    def forward(self, lat, act, msg):
        B, L = lat.shape[:2]
        emb = self.interleave(lat, act, msg).to(next(self.head.parameters()).dtype)
        h = self._run_backbone(emb)
        pred = self.head(h)                             # [B,T,384]
        T = h.shape[1]
        P = N_PATCH + 2                                 # 帧块周期 83
        is_vis = torch.ones(T, dtype=torch.bool, device=h.device)
        is_vis[N_PATCH::P] = False                      # 消息位
        is_vis[N_PATCH + 1::P] = False                  # 动作位
        tgt_flat = torch.zeros(B, T, D_LAT, device=h.device, dtype=pred.dtype)
        tgt_flat[:, is_vis] = self.tgt_norm(lat).flatten(1, 2).to(pred.dtype)
        m = is_vis[1:]                                  # 位置 p 预测 p+1(仅视觉位)
        loss = F.mse_loss(pred[:, :-1][:, m], tgt_flat[:, 1:][:, m])
        if self.aux_msg:
            mp = torch.arange(L - 1, device=h.device) * P + N_PATCH
            mpred = self.msg_head(h[:, mp])
            loss = loss + self.aux_msg * F.mse_loss(mpred, msg[:, 1:].to(mpred.dtype))
        return loss


def build_model(dev, aux_msg, grad_ckpt=True, lora_r=16):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    # 模型只注册了 AutoModelForCausalLM;取其 .backbone(NemotronHModel)并丢弃 lm_head
    full = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    backbone = full.backbone
    del full.lm_head                                    # 省 ~0.8GB 词表投影
    d = backbone.config.hidden_size
    for p in backbone.parameters():
        p.requires_grad_(False)
    if grad_ckpt:
        # 非重入:冻结主干 + LoRA 时,重入式检查点会因基座输入不需梯度而丢失 LoRA 梯度;
        # 本处 inputs_embeds 来自可训投影(需梯度),非重入可靠地回传 LoRA 梯度。
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    lcfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                      target_modules=LORA_TARGETS, bias="none")
    backbone = get_peft_model(backbone, lcfg)
    model = W4Adapter(backbone, d, n_msg=N_MSG, aux_msg=aux_msg)
    # 投影/头以 bf16 参与前向,但保留 fp32 主拷贝由 AdamW 维护更稳:此处统一 bf16 简化。
    for mod in (model.vis_in, model.act_in, model.type_emb, model.head):
        mod.to(dev, torch.bfloat16)
    model.msg_in.to(dev, torch.bfloat16)
    model.msg_type.data = model.msg_type.data.to(dev, torch.bfloat16)
    model.tgt_norm.to(dev)
    if aux_msg:
        model.msg_head.to(dev, torch.bfloat16)
    model.backbone.to(dev)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_360p")
    p.add_argument("--out", default="runs/ftt_w4")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--accum", type=int, default=4, help="梯度累积(等效 bs=bs×accum)")
    p.add_argument("--bs", type=int, default=1, help="真 batch(batch_to_stream 支持 [B,L];VRAM 允许时优先加 bs 而非 accum)")
    p.add_argument("--resume", default="", help="trainer_state.pt 路径;载回模型可训键+优化器+调度器+step 续训")
    p.add_argument("--crop", default="center")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--aux-msg", type=float, default=0.0)
    p.add_argument("--unfreeze-ssm", action="store_true",
                   help="解冻 Mamba2 递归动力学参数(A_log/dt_bias/D):W4c 假说检验——"
                        "冻结递归是否为 age 编码失败的病根(W1b 从零可编码,W4b 冻结不能)")
    p.add_argument("--smoke", action="store_true", help="2 步短序列冒烟")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.seq, args.accum, args.eval_every = 2, 16, 1, 999
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    mk = lambda split: Gaming500Dataset(
        args.data, seq_len=args.seq, img_size=126, stride=args.seq,
        crop_mode=args.crop, split=split, holdout_frac=0.1, periph=True)
    dl = DataLoader(mk("train"), batch_size=args.bs, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=True)
    dl_ev = DataLoader(mk("holdout"), batch_size=1, num_workers=2)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    model = build_model(dev, args.aux_msg, grad_ckpt=True, lora_r=args.lora_r)
    ssm_params = []
    if args.unfreeze_ssm:
        # 解冻决定"状态随时间怎么写入/衰减"的递归旋钮:A_log(衰减率)、dt_bias(时间步)、
        # D(跳连)。每层仅 num_heads 大小,极廉价;放 no-WD 组(A_log 带 _no_weight_decay,
        # 对衰减动力学做 WD 会污染)。in/out_proj 仍由 LoRA 承担,conv1d 保持冻结(短程,不管长视界)。
        for nm, pp in model.backbone.named_parameters():
            if nm.endswith("A_log") or nm.endswith("dt_bias") or nm.endswith("mixer.D"):
                pp.requires_grad_(True)
                ssm_params.append(pp)
        print(f"[W4c] unfroze {len(ssm_params)} SSM 动力学张量 "
              f"({sum(p.numel() for p in ssm_params)} params)", flush=True)
    ssm_ids = {id(pp) for pp in ssm_params}
    other_params = [pp for pp in model.parameters()
                    if pp.requires_grad and id(pp) not in ssm_ids]
    train_params = other_params + ssm_params
    n_tr = sum(pp.numel() for pp in train_params) / 1e6
    opt = torch.optim.AdamW(
        [{"params": other_params, "weight_decay": 0.05},
         {"params": ssm_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / args.warmup, 0.5 * (1 + torch.cos(torch.tensor(
            min(s / args.steps, 1.0) * 3.14159)).item())))
    # ── 断点续训:长训(>数小时)必需;载回可训键(strict=False,冻结主干本就不在 ckpt)
    #    + 优化器/调度器/step。trainer_state.pt 由 eval 块随 ckpt.pt 一并落盘。
    step0 = 0
    if args.resume:
        st = torch.load(args.resume, map_location="cpu")
        _, unexpected = model.load_state_dict(st["model"], strict=False)
        assert not unexpected, f"resume 含未知键(架构不匹配): {unexpected[:3]}"
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step0 = st["step"]
        print(f"[W4] resume ← {args.resume} @step {step0}", flush=True)
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"[W4] trainable {n_tr:.2f}M | {len(dl.dataset)}/{len(dl_ev.dataset)} windows "
          f"| bs {args.bs} accum {args.accum} seq {args.seq}", flush=True)

    step, t0, it = step0, time.time(), iter(dl)
    opt.zero_grad(set_to_none=True)
    while step < args.steps:
        acc_loss = 0.0
        for _ in range(args.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            lat, act, msg = batch_to_stream_msg(batch, dino, dev)
            loss = model(lat, act, msg) / args.accum
            loss.backward()
            acc_loss += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0 or args.smoke:
            rec = {"step": step, "loss": round(acc_loss, 5),
                   "gnorm": round(float(gn), 3),
                   "sps": round((step - step0) / (time.time() - t0), 4)}
            print(f"[W4] {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ev, n = 0.0, 0
            with torch.no_grad():
                for b in dl_ev:
                    lat, act, msg = batch_to_stream_msg(b, dino, dev)
                    ev += model(lat, act, msg).item()
                    n += 1
                    if n >= 20:
                        break
            model.train()
            rec = {"step": step, "eval_loss": round(ev / max(n, 1), 5)}
            print(f"[W4] EVAL {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            # 只存可训权重(投影/头/LoRA + 若解冻则 SSM 动力学),冻结主干余部可从 HF 复原
            # 主干挂在 self.backbone.* 下;适配器(vis_in/head/…)不在其下;LoRA 含 "lora_";
            # --unfreeze-ssm 时 A_log/dt_bias/mixer.D 也已训练,必须一并存否则 eval 载回原值→实验失效
            def _keep(k):
                if not k.startswith("backbone.") or "lora_" in k:
                    return True
                return args.unfreeze_ssm and (
                    k.endswith("A_log") or k.endswith("dt_bias") or k.endswith("mixer.D"))
            sd = {k: v for k, v in model.state_dict().items() if _keep(k)}
            torch.save({"model": sd, "step": step, "args": vars(args),
                        "model_id": MODEL_ID}, os.path.join(args.out, "ckpt.pt"))
            # 续训档:同一份可训 sd + 优化器/调度器全态(AdamW m/v 约为可训参数 2×,仅落快盘)
            torch.save({"model": sd, "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step},
                       os.path.join(args.out, "trainer_state.pt"))
    print(f"[W4] done {step} steps → {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
