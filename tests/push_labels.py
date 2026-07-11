# -*- coding: utf-8 -*-
"""教师标签/词表/池 manifest → HF 公开数据仓(幂等增量)。

许可边界(2026-07-11 核查 openai/Video-Pre-Training):承包商录像 mp4/jsonl **无明示
许可**且含 Minecraft 画面(Microsoft IP 声明),**不公开再分发**;本脚本只传自产派生物:
  - runs/data/vpt_labels/*.npz + manifest.json   教师(MIT 权重)输出的蒸馏标签
  - runs/data/vpt_early_goal_vocab.json          hindsight 词表(机械生成)
  - pool_manifest.json                            段 ID→openaipublic 源 URL(纯元数据)
恢复:hf download unjustify/vpt-craftground-labels-v1 --repo-type dataset;
原始段按 pool_manifest 从 openaipublic blob 重拉。

用法:python tests/push_labels.py [--repo vpt-craftground-labels-v1] [--loop 600]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import get_hf_token  # noqa: E402

OWNER = "unjustify"
BLOB = "https://openaipublic.blob.core.windows.net/minecraft-rl/data/6.13"


def pool_manifest() -> dict:
    """池/holdout 的段 ID 与源 URL(不含任何录像内容)。"""
    out = {}
    for name, d in [("pool", Path("runs/data/vpt_early")),
                    ("holdout", Path("runs/data/vpt_holdout"))]:
        segs = sorted(p.stem for p in d.glob("*.mp4")) if d.exists() else []
        out[name] = [dict(id=s, mp4=f"{BLOB}/{s}.mp4", jsonl=f"{BLOB}/{s}.jsonl")
                     for s in segs]
    seen = Path("runs/data/vpt_early_seen.txt")
    if seen.exists():
        out["seen_evicted"] = seen.read_text().split()
    return out


def sync(api, repo: str, uploaded: set) -> int:
    from huggingface_hub import CommitOperationAdd
    ops, names = [], []
    for p in sorted(Path("runs/data/vpt_labels").glob("*.npz")):
        key = f"labels/{p.name}:{p.stat().st_size}"
        if key not in uploaded:
            ops.append(CommitOperationAdd(f"labels/{p.name}", str(p)))
            names.append(key)
    for extra, dest in [("runs/data/vpt_labels/manifest.jsonl", "labels/manifest.jsonl"),
                        ("runs/data/vpt_early_goal_vocab.json", "goal_vocab.json")]:
        if Path(extra).exists():
            ops.append(CommitOperationAdd(dest, extra))
    mpath = Path("runs/pool_manifest.json")
    mpath.write_text(json.dumps(pool_manifest(), ensure_ascii=False, indent=1))
    ops.append(CommitOperationAdd("pool_manifest.json", str(mpath)))
    api.create_commit(repo_id=f"{OWNER}/{repo}", repo_type="dataset", operations=ops,
                      commit_message=f"sync {len(names)} new label npz")
    uploaded.update(names)
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="vpt-craftground-labels-v1")
    ap.add_argument("--loop", type=int, default=0, help=">0 则每 N 秒增量同步")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=get_hf_token())
    api.create_repo(f"{OWNER}/{args.repo}", repo_type="dataset",
                    private=False, exist_ok=True)
    uploaded: set = set()
    while True:
        n = sync(api, args.repo, uploaded)
        print(f"[push_labels] +{n} npz (累计 {len(uploaded)})", flush=True)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
