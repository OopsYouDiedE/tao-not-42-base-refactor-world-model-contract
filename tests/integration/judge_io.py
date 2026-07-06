#!/usr/bin/env python3
"""判优编排 I/O:组采样 groups.json ⇄ 判优 SubAgent。

判优在环外由主 agent spawn SubAgent 完成;本模块只做两件确定性的事:
  prep  —— 把每 group 组装成**匿名、乱序**的判优包(轨迹标 A/B/C..,不暴露 traj_id/
           指令归属 → 防判据泄漏;顺序随机记录 → 支持顺序偏差审计)。可为指定 group 额
           外产一份**不同乱序**的审计包,用来验证"输入序列顺序会不会影响判断"。
  apply —— 收 SubAgent 排序(每包 best→worst 的标签序列)→ 组内相对优势 advantages.json
           {traj_id: adv∈[-1,1]}(线性:最佳+1 最差-1)。审计包与主包比对排序一致性。

用法:
  PYTHONPATH=. python tests/integration/judge_io.py prep  --round_dir runs/rest_look/r0 --audit_k 1
  PYTHONPATH=. python tests/integration/judge_io.py apply --round_dir runs/rest_look/r0 \
      --rankings runs/rest_look/r0/rankings.json
"""
import argparse
import json
import os

import numpy as np

LABELS = "ABCDEFGHIJ"


def judge_frames(kf_paths, kf_steps):
    """判优给 SubAgent 的帧子集:首/中/尾(看向指令的证据主要在末帧+轨迹跨度)。"""
    if len(kf_paths) <= 3:
        return kf_paths
    mid = len(kf_paths) // 2
    idx = sorted({0, mid, len(kf_paths) - 1})
    return [kf_paths[i] for i in idx]


def build_packet(group, order, pid):
    # 判优只给帧(纯感知),**不给 net_yaw/net_pitch** —— 否则判据退化成"读数排序",
    # 顺序偏差审计也失去意义。方向答案必须从画面看出来。
    disp = []
    for lab, ti in zip(LABELS, order):
        tr = group["trajectories"][ti]
        disp.append({"label": lab, "traj_id": tr["traj_id"],
                     "frames": judge_frames(tr["keyframes"], tr["keyframe_steps"])})
    return {"packet_id": pid, "group_id": group["group_id"],
            "instruction": group["instr_text"], "n": len(order), "items": disp}


def prep(args):
    gj = json.load(open(os.path.join(args.round_dir, "groups.json")))
    rng = np.random.default_rng(args.seed)
    packets = []
    for gi, g in enumerate(gj["groups"]):
        n = len(g["trajectories"])
        order = list(rng.permutation(n))
        packets.append(build_packet(g, order, g["group_id"]))
        if gi < args.audit_k:                       # 审计包:同 group 换一个乱序
            order2 = list(rng.permutation(n))
            while order2 == order and n > 1:
                order2 = list(rng.permutation(n))
            packets.append(build_packet(g, order2, g["group_id"] + "__audit"))
    out = os.path.join(args.round_dir, "judge_packets.json")
    json.dump({"round_dir": args.round_dir, "packets": packets}, open(out, "w"),
              indent=1, ensure_ascii=False)
    print(f"💾 {out} | {len(packets)} 包(含 {args.audit_k} 审计)")
    for p in packets:
        print(f"  [{p['packet_id']}] 指令='{p['instruction']}' n={p['n']} "
              f"标签={[it['label'] for it in p['items']]}")


def apply(args):
    pj = json.load(open(os.path.join(args.round_dir, "judge_packets.json")))
    rk = json.load(open(args.rankings))            # {packet_id: [labels best→worst]}
    lab2tid = {p["packet_id"]: {it["label"]: it["traj_id"] for it in p["items"]} for p in pj["packets"]}
    adv, audit = {}, []
    for pid, order in rk.items():
        if pid.endswith("__audit"):
            continue                                # 审计包不进优势,只做一致性比对
        n = len(order)
        m = lab2tid[pid]
        for rank, lab in enumerate(order):
            a = 1.0 - 2.0 * rank / max(n - 1, 1)    # best→+1  worst→-1
            adv[m[lab]] = round(float(a), 4)
    # 顺序偏差审计:主包 vs 审计包 的 traj_id 排序一致性(Spearman + top-1 一致)
    for pid in rk:
        if not pid.endswith("__audit"):
            continue
        base = pid[:-len("__audit")]
        if base not in rk:
            continue
        m0, m1 = lab2tid[base], lab2tid[pid]
        r0 = {m0[l]: i for i, l in enumerate(rk[base])}
        r1 = {m1[l]: i for i, l in enumerate(rk[pid])}
        common = sorted(set(r0) & set(r1))
        a = np.array([r0[t] for t in common]); b = np.array([r1[t] for t in common])
        rho = float(np.corrcoef(a, b)[0, 1]) if len(common) > 2 else float("nan")
        top1 = (rk[base][0] and m0[rk[base][0]]) == m1[rk[pid][0]]
        audit.append({"group": base, "spearman_order_agreement": round(rho, 3),
                      "top1_stable": bool(top1)})
    out = os.path.join(args.round_dir, "advantages.json")
    json.dump({"advantages": adv, "order_bias_audit": audit}, open(out, "w"),
              indent=1, ensure_ascii=False)
    print(f"💾 {out} | {len(adv)} 轨优势")
    if audit:
        print("🔎 顺序偏差审计:", json.dumps(audit, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prep"); pp.add_argument("--round_dir", required=True)
    pp.add_argument("--seed", type=int, default=0); pp.add_argument("--audit_k", type=int, default=1)
    pa = sub.add_parser("apply"); pa.add_argument("--round_dir", required=True)
    pa.add_argument("--rankings", required=True)
    args = ap.parse_args()
    (prep if args.cmd == "prep" else apply)(args)


if __name__ == "__main__":
    main()
