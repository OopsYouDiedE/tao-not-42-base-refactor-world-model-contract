#!/usr/bin/env python3
"""慢塔规划审计(用户 07-08 定案:教师除了检查走得好不好,还要检查慢塔分析对不对)。

双通道且互相监督:
1) 机器精判:慢塔计划 vs 配方树差额规划 missing_plan(SFT 的 ground truth)逐项对比;
2) Haiku 审推理(--judge):读库存+慢塔原文,判分析是否正确;
两者分歧率是防 Goodhart 哨兵(配方树若有错/慢塔口径漂移都会在这暴露)。
审计不进 GRPO 优势(优势只更新快塔,慢塔的错单独记账攒修复训练数据)。
"""
import argparse
import glob
import json
import re
import subprocess

import numpy as np

from train.fovea_twotower.reason_delta_sft import missing_plan, zh


def audit_events(files):
    evs = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        for rec in json.loads(str(z["recs"])):
            for t, inv, sp, text in rec.get("plan_log", []):
                expect = [zh(s) for s in missing_plan("raw_iron", set(inv))]
                evs.append(dict(src=f, t=t, inv=inv, plan=sp, expect=expect,
                                exact=sp == expect,
                                next_ok=bool(sp) and bool(expect) and sp[0] == expect[0],
                                text=text))
    return evs


def judge_events(evs, limit=20):
    lines = []
    for i, e in enumerate(evs[:limit]):
        lines.append(f"### 第{i}条\n库存:{'、'.join(zh(x) for x in e['inv']) or '(空)'}\n"
                     f"慢塔分析原文:\n{e['text']}\n")
    prompt = ("Minecraft 生存模式,总目标=获得生铁。下面每条给出智能体当时的库存和慢塔"
              "模型的规划分析原文。逐条判断该分析是否正确:核查库存是否被正确利用、"
              "补齐计划是否符合合成依赖(木头→木板→木棍/工作台→木镐→圆石→石镐→铁矿)、"
              "依赖顺序是否可执行。输出格式严格为每行一条『第N条: 正确』或"
              "『第N条: 错误,原因不超过20字』,不输出其他内容。\n\n" + "\n".join(lines))
    out = subprocess.run(["claude", "-p", "--model", "haiku"], input=prompt,
                         capture_output=True, text=True, timeout=600).stdout
    got = {}
    for m in re.finditer(r"第\s*(\d+)\s*条\s*[:：]\s*(正确|错误[^\n]*)", out):
        got[int(m.group(1))] = m.group(2)
    return got, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", nargs="+", default=sorted(glob.glob("runs/grpo_r2/g*_w*.npz")))
    p.add_argument("--judge", action="store_true")
    p.add_argument("--out", default="runs/grpo_r2/plan_audit.json")
    args = p.parse_args()
    evs = audit_events(args.npz)
    if not evs:
        print("无 plan_log 事件(g1 之前的 npz 不含)")
        return
    res = dict(n=len(evs),
               exact_acc=round(np.mean([e["exact"] for e in evs]), 3),
               next_step_acc=round(np.mean([e["next_ok"] for e in evs]), 3))
    if args.judge:
        got, raw = judge_events(evs)
        for i, v in got.items():
            evs[i]["judge"] = v
        judged = [e for e in evs if "judge" in e]
        res["judge_n"] = len(judged)
        res["judge_ok_rate"] = round(np.mean(
            [e["judge"] == "正确" for e in judged]), 3) if judged else None
        res["divergence"] = round(np.mean(
            [(e["judge"] == "正确") != e["exact"] for e in judged]), 3) if judged else None
        open(args.out.replace(".json", "_reply.txt"), "w").write(raw)
    res["events"] = evs
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "events"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
