#!/usr/bin/env python3
"""GRPO 长程训练 harness(预建;木材链过启动门当天点火)。

三层过程优势(experiments-index M-IRON 训练信号设计,预登记):
  ①里程碑深度分:机器可查不可逆事件链(见 MILESTONES);
  ②意图-行为一致性:声明子目标+探索覆盖增量(地图 survey 扇区递减);
  ③全败组条款:组内零成功时过程分独立撑梯度;
  封顶铁律:过程总分 < 单次成功分(精彩失败<平庸成功,防表演)。
rollout 记录 schema(采集侧填):dict(seed, inv_events=set[str], iron_lock_steps=int,
  declared_goal=str, goal_consistent_steps=int, explored_delta=int, success=bool)。
干跑冒烟:--smoke 用 Step0 episode 摘要合成假 rollout 验打分器/优势归一。

用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/grpo_harness.py --smoke
"""
import argparse
import json

import numpy as np

SUCCESS_SCORE = 10.0
PROCESS_CAP = 0.9 * SUCCESS_SCORE          # 封顶:过程总分<单次成功分
MILESTONES = [                              # (名, 事件谓词) 不可逆状态变化
    ("log", lambda r: "log" in r["inv_events"]),
    ("planks", lambda r: "planks" in r["inv_events"]),
    ("wooden_pickaxe", lambda r: "wooden_pickaxe" in r["inv_events"]),
    ("cobblestone", lambda r: "cobblestone" in r["inv_events"]),
    ("stone_pickaxe", lambda r: "stone_pickaxe" in r["inv_events"]),
    ("iron_lock", lambda r: r.get("iron_lock_steps", 0) >= 30),
    ("iron", lambda r: r.get("success", False)),
]
PER_MS = PROCESS_CAP / len(MILESTONES)     # 单里程碑分帽
CONSIST_W = 0.5 * PER_MS                   # 意图一致性权重(不越单里程碑帽)


def score_rollout(rec):
    """三层过程分 + 结果分(二值另计,叠加)。"""
    depth = 0.0
    for _name, pred in MILESTONES:
        if pred(rec):
            depth += PER_MS
    consist = 0.0
    if rec.get("declared_goal"):
        c = rec.get("goal_consistent_steps", 0) / max(rec.get("steps", 1), 1)
        consist += CONSIST_W * min(c, 1.0)
        if rec.get("explored_delta", 0) > 0:
            consist += 0.5 * CONSIST_W
    proc = min(depth + consist, PROCESS_CAP)
    return proc + (SUCCESS_SCORE if rec.get("success") else 0.0)


def group_advantage(scores, std_floor=1e-3):
    """组内相对优势(全败组条款天然成立:过程分方差>0 即有梯度)。"""
    s = np.asarray(scores, np.float64)
    return (s - s.mean()) / max(float(s.std()), std_floor)


def launch_gate(scores, wood_rate=None):
    """预注册启动门:组内过程分方差>0(操作口径=采木闭环≥0.25 或里程碑分层)。"""
    var_ok = float(np.asarray(scores).std()) > 0
    strat = len(set(np.round(scores, 3))) >= 2
    wood_ok = wood_rate is not None and wood_rate >= 0.25
    return dict(variance_gt0=var_ok, stratified=strat,
                wood_rate_ok=bool(wood_ok),
                GATE=bool(var_ok and (wood_ok or strat)))


# ── 策略更新骨架(点火时接 m_iron 式 episode 采样器) ────────────────────
def grpo_update(student, groups, opt, clip=0.2):
    """组=同 seed n 局 rollout(含逐步 token/动作序列);优势加权 BC
    (REINFORCE-with-baseline 形态,22M v17 架构复用;early-stop 按闭环)。
    groups: [(records, token_seqs, action_seqs)];此处仅定义接口,
    点火前由 rollout 采集器填充序列字段。"""
    import torch
    total = 0.0
    for recs, toks, acts in groups:
        adv = group_advantage([score_rollout(r) for r in recs])
        for a_w, tk, ac in zip(adv, toks, acts):
            if abs(float(a_w)) < 1e-6:
                continue
            cam, key = student.tower(tk[None], torch.zeros(1, 1, device=tk.device),
                                     ac[None])
            loss = float(a_w) * torch.nn.functional.cross_entropy(
                cam[0, -1], ac[-1, :2].long().clamp(0, cam.shape[-1] - 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
    return total


def smoke():
    """假 rollout=Step0 episode 摘要(inv 全空/saw_tok 分化)→验打分与优势。"""
    recs = []
    for f in ("runs/m_iron_step0.json", "runs/m_iron_step0_v5b.json"):
        try:
            d = json.load(open(f))
        except FileNotFoundError:
            continue
        for e in d["episodes"]:
            recs.append(dict(seed=e["world_seed"], inv_events=set(e["inv_end"]),
                             iron_lock_steps=e.get("saw_iron_tok", 0) and
                             (30 if e.get("saw_iron_tok", 0) >= 1000 else 0),
                             declared_goal=e.get("plan_first", ""),
                             goal_consistent_steps=0, explored_delta=0,
                             steps=e["steps"], success=e["ok"]))
    recs = recs[:16]
    scores = [score_rollout(r) for r in recs]
    adv = group_advantage(scores)
    gate = launch_gate(scores)
    out = dict(n=len(recs), scores=[round(s, 3) for s in scores],
               adv=[round(float(a), 3) for a in adv],
               all_fail=not any(r["success"] for r in recs),
               gate=gate,
               note="假rollout=Step0摘要;iron_lock(幻觉期)分层证全败组条款出梯度")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("runs/grpo_smoke.json", "w") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    assert out["all_fail"] and gate["variance_gt0"], "全败组必须仍有梯度"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        smoke()
