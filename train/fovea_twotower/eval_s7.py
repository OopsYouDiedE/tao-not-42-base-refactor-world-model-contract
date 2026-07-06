# -*- coding: utf-8 -*-
"""S7a(Step4 核心):预训练慢塔(W4/Nemotron-4B)是否解锁**带时**消息记忆。

与 probe_b.py(S4a)逐行同协议——同样本限制、同 bearing/age 标签、同拟合与判据——
唯一差异 = 被探塔:从零 58M ContextTower → 预训练 4B W4Adapter。
STATE = 21 层 Mamba2 最终 ssm_state,每层对非状态轴取均值 → (B, state_size),
        沿层拼接(池化配方在看结果前固定,见下 pool_ssm);
FRAME = 单帧(81 视觉 + 1 消息,无历史)last_hidden_state 均池;
MSG-HIST = 裸消息缓冲(信息上界,不经塔,同 probe_b)。
判据(step4 §2,先于结果登记):acc_STATE ≥ acc_FRAME+5pt 且 R²_age(STATE) CI>0。

用法(W4 训练完、GPU 空出后):
    PYTHONPATH=. python train/fovea_twotower/eval_s7.py \
        --ckpt runs/ftt_w4/ckpt.pt --out runs/ftt_s7_360.json --n 3000
"""
import argparse
import json
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.fovea_twotower.train_w1 import batch_to_stream_msg
from train.fovea_twotower.train_w4 import W4Adapter, MODEL_ID, LORA_TARGETS
from train.fovea_twotower.probe_b import fit_acc, fit_r2, K_BACK, K_FRESH
from train.gaming500.dataset import Gaming500Dataset, N_MSG


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


@torch.no_grad()
def collect(model, dino, dl, n_max, dev):
    """逐行复刻 probe_b.collect,STATE 换成 21 层 ssm_state 池化。"""
    F_state, F_frame, F_hist, yb, ya = [], [], [], [], []
    for batch in dl:
        lat, act, msg = batch_to_stream_msg(batch, dino, dev)
        B, L = lat.shape[:2]
        t = L - 1
        evt = (msg[..., :8].sum(-1) > 0).cpu().numpy()
        bear = msg[..., :8].argmax(-1).cpu().numpy()
        keep, lbl_b, lbl_a = [], [], []
        for i in range(B):
            if evt[i, t - K_FRESH + 1:t + 1].any():
                continue
            past = np.nonzero(evt[i, t - K_BACK:t - K_FRESH + 1])[0]
            if len(past) == 0:
                continue
            j = int(past[-1]) + (t - K_BACK)
            keep.append(i)
            lbl_b.append(int(bear[i, j]))
            lbl_a.append(t - j)
        if not keep:
            continue
        k = torch.tensor(keep, device=dev)
        _, states = model.encode(lat[k], act[k], msg[k], want_states=True)
        F_state.append(pool_ssm(states).cpu())
        h_fr, _ = model.encode(lat[k, t:t + 1], act[k, :0], msg[k, t:t + 1])
        F_frame.append(h_fr.float().mean(1).cpu())
        F_hist.append(msg[k, t - K_BACK:t + 1].flatten(1).float().cpu())
        yb += lbl_b
        ya += lbl_a
        if len(yb) >= n_max:
            break
    cat = lambda xs: torch.cat(xs).numpy()
    return (cat(F_state), cat(F_frame), cat(F_hist),
            np.array(yb), np.array(ya, np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_360p")
    p.add_argument("--ckpt", default="runs/ftt_w4/ckpt.pt")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--out", default="runs/ftt_s7_360.json")
    args = p.parse_args()
    dev = "cuda"

    model, ck = build_eval_model(args.ckpt, dev)
    crop = ck.get("args", {}).get("crop", "center")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    mk = lambda split: DataLoader(
        Gaming500Dataset(args.data, seq_len=args.seq, img_size=126,
                         stride=args.seq // 2, crop_mode=crop, periph=True,
                         split=split, holdout_frac=0.1),
        batch_size=args.bs, shuffle=(split == "train"), num_workers=8)
    Xs_tr, Xf_tr, Xh_tr, yb_tr, ya_tr = collect(model, dino, mk("train"), args.n, dev)
    Xs_te, Xf_te, Xh_te, yb_te, ya_te = collect(model, dino, mk("holdout"), args.n, dev)

    res = {"tower": "W4", "model_id": MODEL_ID, "ckpt_step": ck.get("step"),
           "train_args": ck.get("args", {}),
           "n_train": len(yb_tr), "n_test": len(yb_te),
           "chance_bearing": round(1.0 / 8, 4),
           "dim_STATE": int(Xs_tr.shape[1]), "dim_FRAME": int(Xf_tr.shape[1]),
           "acc_bearing_STATE": round(fit_acc(Xs_tr, yb_tr, Xs_te, yb_te), 4),
           "acc_bearing_FRAME": round(fit_acc(Xf_tr, yb_tr, Xf_te, yb_te), 4),
           "acc_bearing_MSGHIST": round(fit_acc(Xh_tr, yb_tr, Xh_te, yb_te), 4)}
    r2s, lo_s, hi_s = fit_r2(Xs_tr, ya_tr, Xs_te, ya_te)
    r2f, lo_f, hi_f = fit_r2(Xf_tr, ya_tr, Xf_te, ya_te)
    r2h, lo_h, hi_h = fit_r2(Xh_tr, ya_tr, Xh_te, ya_te)
    res.update(r2_age_STATE=round(r2s, 4), r2_age_STATE_ci=[round(lo_s, 4), round(hi_s, 4)],
               r2_age_FRAME=round(r2f, 4), r2_age_FRAME_ci=[round(lo_f, 4), round(hi_f, 4)],
               r2_age_MSGHIST=round(r2h, 4), r2_age_MSGHIST_ci=[round(lo_h, 4), round(hi_h, 4)])
    d = res["acc_bearing_STATE"] - res["acc_bearing_FRAME"]
    ok = d >= 0.05 and lo_s > 0
    res["verdict_s7a"] = ("PASS" if ok else "FAIL") + \
        f" (Δbearing={d:+.4f}, r2_age_ci_lo={lo_s:+.4f})"
    print("[S7a]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
