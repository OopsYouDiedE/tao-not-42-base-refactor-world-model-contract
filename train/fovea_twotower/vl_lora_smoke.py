#!/usr/bin/env python3
"""R-D Qwen2-VL-2B LoRA 冒烟:A2/集群 VL SFT 配置模板(工具链风险清零)。

定的设定:Qwen2-VL LoRA targets/r/lr、grad-ckpt、batch、图像分辨率(448×256)。
玩具集 200 样本 = 示范帧(448×256 重采样)+ R-C 同款状态行 → 决策文本;
peft LoRA r16 q/k/v/o(视觉塔冻结),bf16,grad-checkpointing on,batch1×累积8,
lr1e-4,200 步。闸门:loss 降≥50%、显存峰值≤20GB、无 NaN。数字入配置模板表。

对外接口:main(CLI)。用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/vl_lora_smoke.py
"""
import argparse
import json
import random

import numpy as np
import torch

from train.fovea_twotower.eval_g1 import load_eps
from train.fovea_twotower.heartbeat_sft import MS, state_line


def toy_samples(n, data_dir, rng):
    """示范帧 448×256 + 状态行 prompt → 决策文本(确定性,有可学信号)。"""
    import cv2
    eps = load_eps(data_dir)
    frames = [ep["frames"][t].transpose(1, 2, 0)
              for ep in eps for t in range(0, len(ep["frames"]), 5)]
    out = []
    for i in range(n):
        f = frames[i % len(frames)]
        img = cv2.resize(np.ascontiguousarray(f), (448, 256))
        gi = rng.randint(0, len(MS) - 2)
        stuck = rng.random() < 0.35
        inv = {MS[gi]: 1} if rng.random() < 0.5 else {}
        disp = rng.uniform(0, 1) if stuck else rng.uniform(2, 6)
        sl = state_line(i * 15, inv, None, disp, MS[gi])
        dec = "重规划" if stuck else (f"换目标:{MS[gi + 1]}" if inv else "继续")
        out.append((img, sl, dec))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data_dir", default="runs/data/calib640")
    p.add_argument("--out", default="runs/vl_lora_smoke")
    p.add_argument("--out_json", default="runs/vl_lora_smoke.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda"
    rng = random.Random(args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(          # 视觉塔冻结:只挂 LLM 注意力
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    samples = toy_samples(args.n, args.data_dir, rng)

    def encode(img, sl, dec):
        from PIL import Image
        pil = Image.fromarray(img)
        user = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "content": None,
             "text": f"观察当前帧与状态行,给出微决策(继续/重规划/换目标:<物品>)。\n{sl}"}]}]
        full = user + [{"role": "assistant", "content": dec}]
        t_full = proc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        t_pre = proc.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[t_full], images=[pil], return_tensors="pt", padding=True)
        pre = proc(text=[t_pre], images=[pil], return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[:, :pre["input_ids"].shape[1]] = -100
        enc["labels"] = labels
        return {k: v.to(dev) for k, v in enc.items()}

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses, nan = [], False
    opt.zero_grad()
    for st in range(args.steps):
        img, sl, dec = samples[rng.randrange(len(samples))]
        out = model(**encode(img, sl, dec))
        loss = out.loss / args.accum
        if not torch.isfinite(loss):
            nan = True
            break
        loss.backward()
        if (st + 1) % args.accum == 0:
            opt.step()
            opt.zero_grad()
        losses.append(float(out.loss))
        if st % 25 == 0:
            print(f"[vl] step {st} loss {float(out.loss):.4f}", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    l0 = float(np.mean(losses[:10])) if len(losses) >= 10 else (losses[0] if losses else 0)
    l1 = float(np.mean(losses[-10:])) if len(losses) >= 10 else (losses[-1] if losses else 0)
    drop = 1 - l1 / max(l0, 1e-9)
    gate = dict(loss_start=round(l0, 4), loss_end=round(l1, 4),
                loss_drop_frac=round(drop, 3), peak_vram_gb=round(peak_gb, 2),
                nan=nan, steps=len(losses),
                cfg=dict(r=16, targets="q,k,v,o(视觉冻结)", grad_ckpt=True,
                         bf16=True, batch=1, accum=args.accum, lr=args.lr,
                         img="448x256"),
                gate_loss_down_50=bool(drop >= 0.5),
                gate_vram_le_20=bool(peak_gb <= 20.0),
                gate_no_nan=bool(not nan),
                verdict="PASS" if (drop >= 0.5 and peak_gb <= 20 and not nan) else "FAIL")
    if not nan:
        model.save_pretrained(args.out)
    json.dump(gate, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
