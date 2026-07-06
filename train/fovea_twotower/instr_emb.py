# -*- coding: utf-8 -*-
"""Minecraft 指令小词表 → MiniLM 冻结句向量(text-conditioned 快头的条件 token)。

用途双份:①作 384 维条件喂进 TextCondPolicy(= 模型 d,免投影维度失配);
②同一句自然语指令原样交给判优 SubAgent 当"是否在执行该指令"的判据。

分阶段(用户定调"第一步简单指令,后面稍复杂"):
  SIMPLE — 单动作、基座策略在温度采样下就有行为方差的轴(挖/静止/看动),先证信号;
  DETAIL — 稍复杂的组合指令,待 SIMPLE 通了、组内多样性够了再启用。

冻结句向量而非可学 embedding:词表极小,可学 emb 退化成查表、对新指令零外推;
MiniLM 语义空间免费保留"近义指令向量相近"(见 train/minecraft/task_text.py)。

用法:
    PYTHONPATH=. python train/fovea_twotower/instr_emb.py --out runs/ftt_instr/instr_emb.pt
"""
import argparse
import os

import torch

from train.minecraft.task_text import TaskTextEncoder

# id → (阶段, 英文指令[喂模型+判据], 中文对照[仅注释/日志])
VOCAB = {
    "mine":    ("SIMPLE", "mine the iron ore straight ahead: swing and break blocks", "挖正前方的铁矿:挥击破坏方块"),
    "still":   ("SIMPLE", "stand still and do nothing, do not swing or move",         "站着别动:不挥击不移动"),
    "left":    ("SIMPLE", "turn your view to the left",                                "视角向左转"),
    "right":   ("SIMPLE", "turn your view to the right",                               "视角向右转"),
    "down":    ("SIMPLE", "look downward at the ground",                               "视角看向下方地面"),
    "forward": ("SIMPLE", "walk straight forward",                                     "笔直向前走"),
    "free":    ("SIMPLE", "act freely, do whatever seems natural",                     "自由行动"),
    # DETAIL(占位,后续启用)
    "approach_mine": ("DETAIL", "walk up to the ore wall, then mine the iron in front of you",
                      "走近矿墙,再挖正前方的铁"),
    "scan_mine":     ("DETAIL", "look down and dig, then look up to find more ore and mine it",
                      "低头挖,再抬头找更多矿继续挖"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/ftt_instr/instr_emb.pt")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    enc = TaskTextEncoder(kind="minilm", device=args.device)
    ids = list(VOCAB.keys())
    texts = [VOCAB[i][1] for i in ids]
    emb = enc.encode(texts)                     # [K,384] fp32, L2-normalized
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"ids": ids, "texts": texts,
                "phase": [VOCAB[i][0] for i in ids],
                "zh": [VOCAB[i][2] for i in ids],
                "emb": emb}, args.out)
    print(f"💾 {args.out} | {len(ids)} 指令 emb{tuple(emb.shape)}")
    for i, t, ph in zip(ids, texts, [VOCAB[k][0] for k in ids]):
        print(f"  [{ph:6s}] {i:14s} — {t}")


if __name__ == "__main__":
    main()
