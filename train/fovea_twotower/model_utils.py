# -*- coding: utf-8 -*-
"""fovea-twotower 评估域的 W4 慢塔重建与 SSM 隐藏状态池化。

对外接口:
    build_eval_model(ckpt_path, dev, lora_r) — 重建 W4 慢塔并载入投影/头/LoRA 权重。
    pool_ssm(states) — 21 层 Mamba2 ssm_state 逐层均池后沿层拼接。
"""
import torch

from train.fovea_twotower.train_w4 import W4Adapter, MODEL_ID, LORA_TARGETS
from train.gaming500.dataset import N_MSG


def build_eval_model(ckpt_path, dev, lora_r=16):
    """重建 W4 塔并载入 ckpt(投影/头/LoRA);eval 且不启梯度检查点(留 use_cache 通路)。"""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    full = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    backbone = full.backbone
    del full.lm_head
    for p in backbone.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                      target_modules=LORA_TARGETS, bias="none")
    backbone = get_peft_model(backbone, lcfg)
    d = backbone.base_model.model.config.hidden_size
    ck = torch.load(ckpt_path, map_location="cpu")
    aux = ck.get("args", {}).get("aux_msg", 0.0)
    model = W4Adapter(backbone, d, n_msg=N_MSG, aux_msg=aux)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    # 只允许"冻结主干的非 LoRA 权重"缺失(它们从 HF 复原);其余缺失/多余都要报警
    bad_missing = [k for k in missing
                   if k.startswith("backbone.") and "lora_" not in k]
    assert not unexpected, f"unexpected keys in ckpt: {unexpected[:5]}"
    assert len(bad_missing) == len(missing), \
        f"非主干权重缺失(适配器/LoRA 没载上):{set(missing) - set(bad_missing)}"
    return model.to(dev).bfloat16().eval(), ck


def pool_ssm(states):
    """states = [ssm_state]×21,各 (B, ...) → 每层对非状态轴均值 → (B, state)。拼接。
    对 3D (B,inter,state) 与 4D (B,head,hdim,state) 皆稳健(flatten 中间维再均值)。"""
    feats = [s.float().flatten(1, -2).mean(1) for s in states]  # 每层 (B, state_size)
    return torch.cat(feats, 1)
