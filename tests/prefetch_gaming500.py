#!/usr/bin/env python3
"""预取 gaming500-720p-hdf5 的一部分分片(快头/Action 塔下一步训练用)。

500h 数据集 = unjustify/gaming500-720p-hdf5(公开,24 片共 245GB,167 游戏轮转交错,
故少数几片即多游戏覆盖)。逐片下载、断点续传,存到 runs/data/g500_720p_shards/。
经 SOCKS 代理(本机 127.0.0.1:2080)+ HF token。

用法:
  export HF_TOKEN=... ALL_PROXY=socks5://127.0.0.1:2080 HTTPS_PROXY=socks5://127.0.0.1:2080
  python -m tests.prefetch_gaming500 --shards 0-5 --out runs/data/g500_720p_shards
"""
import argparse
import os
import time

from huggingface_hub import hf_hub_download

REPO = "unjustify/gaming500-720p-hdf5"


def parse_shards(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="0-5", help="分片索引,如 0-5 或 0,3,7")
    p.add_argument("--out", default="runs/data/g500_720p_shards")
    p.add_argument("--repo", default=REPO)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    idxs = parse_shards(args.shards)
    print(f"[prefetch] {args.repo} shards={idxs} -> {args.out}", flush=True)
    done = []
    for i in idxs:
        fn = f"shard_{i:04d}.h5"
        t0 = time.time()
        print(f"[prefetch] ↓ {fn} ...", flush=True)
        path = hf_hub_download(repo_id=args.repo, filename=fn, repo_type="dataset",
                               local_dir=args.out, resume_download=True)
        sz = os.path.getsize(path) / 1e9
        dt = time.time() - t0
        print(f"[prefetch] ✓ {fn} {sz:.2f}GB in {dt/60:.1f}min "
              f"({sz*1000/max(dt,1):.1f} MB/s) -> {path}", flush=True)
        done.append(fn)
    print(f"[prefetch] DONE {len(done)}/{len(idxs)} shards: {done}", flush=True)


if __name__ == "__main__":
    main()
