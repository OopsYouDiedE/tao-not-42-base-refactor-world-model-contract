#!/usr/bin/env python3
"""给 J0-v2 考卷判分:读判官逐行『第N对: A/B』答案,按类别对 ground truth 出准确率。"""
import argparse
import json
import re
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("answers", help="判官答案文本文件")
    p.add_argument("--pairs", default="runs/judge_exam_v2_pairs.json")
    p.add_argument("--out", default="runs/judge_exam_v2_result.json")
    args = p.parse_args()

    truth = {x["idx"]: x for x in json.load(open(args.pairs))}
    got = {}
    for m in re.finditer(r"第\s*(\d+)\s*对\s*[:：]\s*([AB])", open(args.answers).read()):
        got[int(m.group(1))] = m.group(2)

    per_cat = defaultdict(lambda: [0, 0])
    wrong = []
    for idx, t in truth.items():
        c = per_cat[t["category"]]
        c[1] += 1
        if got.get(idx) == t["answer"]:
            c[0] += 1
        else:
            wrong.append(dict(idx=idx, category=t["category"],
                              expect=t["answer"], got=got.get(idx)))
    res = dict(parsed=len(got), total=len(truth),
               acc_by_cat={k: dict(acc=round(v[0] / v[1], 3), n=v[1])
                           for k, v in sorted(per_cat.items())},
               acc_overall=round(sum(v[0] for v in per_cat.values()) / len(truth), 3),
               wrong=wrong)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
