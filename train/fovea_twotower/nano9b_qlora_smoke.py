#!/usr/bin/env python3
"""R-F Nemotron-Nano-9B-v2 QLoRA 冒烟:混合架构(Mamba2+注意力)本地工具链风险清零。

集群租卡前置条件:验 NF4 载入(~5GB)+ LoRA 反向在 3090/cu124/mamba-ssm 2.2.4
上跑通。LoRA 挂 Mamba 投影(in/out_proj)+ 注意力(qkvo);50 步 dummy 文本反向。
闸门:backward 通过、显存<24G、混合内核无报错、无 NaN。

对外接口:main(CLI)。用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/nano9b_qlora_smoke.py
"""
import argparse
import json

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_json", default="runs/nano9b_qlora_smoke.json")
    args = p.parse_args()
    dev = "cuda"

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    err = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=nf4, dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=True)
        load_gb = torch.cuda.memory_allocated() / 1e9
        model = prepare_model_for_kbit_training(model)
        # 混合架构:Mamba 投影(in/out_proj)+ 注意力(qkvo)都挂 LoRA
        names = {n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)}
        targets = [t for t in ("in_proj", "out_proj", "q_proj", "k_proj",
                               "v_proj", "o_proj", "qkv_proj") if t in names]
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=targets, task_type="CAUSAL_LM"))
        model.train()
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                lr=args.lr)
        torch.cuda.reset_peak_memory_stats()
        text = ("Minecraft 生存:检查背包,规划下一步收集。"
                "库存空,目标铁镐,需先拿木头做木镐再挖石头。")
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        losses, nan = [], False
        for st in range(args.steps):
            loss = model(input_ids=ids, labels=ids).loss
            if not torch.isfinite(loss):
                nan = True
                break
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(float(loss))
            if st % 10 == 0:
                print(f"[9b] step {st} loss {float(loss):.4f}", flush=True)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception as e:  # noqa
        import traceback
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        load_gb = peak_gb = -1.0
        losses, nan = [], True
        targets = []

    ok = bool(err is None and not nan and len(losses) == args.steps and peak_gb < 24.0)
    gate = dict(
        model=args.model, lora_targets=targets,
        load_vram_gb=round(load_gb, 2) if load_gb > 0 else None,
        peak_vram_gb=round(peak_gb, 2) if peak_gb > 0 else None,
        loss_start=round(losses[0], 4) if losses else None,
        loss_end=round(losses[-1], 4) if losses else None,
        nan=nan, error=err, steps_done=len(losses),
        gate_backward=bool(err is None and not nan and len(losses) == args.steps),
        gate_vram_lt_24=bool(peak_gb > 0 and peak_gb < 24.0),
        verdict="PASS" if ok else "FAIL")
    json.dump(gate, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
