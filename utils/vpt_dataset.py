import os
import json
import random
import cv2
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np

# 动作向量布局(与 download_sample_data / colab 转换脚本严格一致):2 鼠标 + 20 键盘
VPT_KEYS = ["key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak",
            "key_sprint", "key_attack", "key_use", "key_drop", "key_inventory"] \
           + [f"key_hotbar.{i}" for i in range(1, 10)]


def _action_vec(act_dict):
    """单帧 jsonl dict -> [2+20] float tensor。"""
    mouse = act_dict.get("mouse", {"dx": 0.0, "dy": 0.0})
    kb = act_dict.get("keyboard", {})
    return torch.tensor([mouse["dx"], mouse["dy"]] + [float(kb.get(k, 0)) for k in VPT_KEYS],
                        dtype=torch.float32)


def _pair_list(data_dir):
    """目录里所有成对的 (mp4, jsonl)。"""
    pairs = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".mp4"):
            jp = os.path.join(data_dir, f[:-4] + ".jsonl")
            if os.path.exists(jp):
                pairs.append((os.path.join(data_dir, f), jp))
    return pairs


def _decode_clip(mp4_path, jsonl_path, seq_len):
    """解码一段 mp4+jsonl -> {"img":[T,3,H,W], "action":[T,22], "task":str};太短返回 None。"""
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    if len(frames) < seq_len:
        return None
    imgs = torch.stack(frames, dim=0)
    actions, task = [], ""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for t, line in enumerate(f):
            a = json.loads(line)
            if t == 0 and "task" in a:
                task = a["task"]
            actions.append(_action_vec(a))
    n = min(imgs.shape[0], len(actions))
    if n < seq_len:
        return None
    return {"img": imgs[:n], "action": torch.stack(actions[:n], dim=0), "task": task}


class VPTDataset(Dataset):
    """
    轻量化的纯离线 VPT/BASALT 数据集加载器。
    无需依赖 minerl 环境，直接读取 .mp4 和 .jsonl 裸数据。
    """
    def __init__(self, data_dir, seq_len=60, fps=20):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.fps = fps
        self.videos = []
        
        # 全量预载到内存(小数据集/局部过拟合测试用;大数据集请用 VPTStreamDataset)
        vid_paths = _pair_list(data_dir)
        print(f"[VPTDataset] Found {len(vid_paths)} video clips. Pre-loading into RAM...")
        for vid_path, json_path in vid_paths:
            clip = _decode_clip(vid_path, json_path, self.seq_len)
            if clip is not None:
                self.videos.append(clip)

    def __len__(self):
        # 局部过拟合测试，虚拟扩大 epoch 的长度
        return len(self.videos) * 20

    def __getitem__(self, idx):
        data = self.videos[idx % len(self.videos)]
        imgs_full = data["img"]
        actions_full = data["action"]
        
        # 1. 随机找一个起始点
        total_frames = imgs_full.shape[0]
        start_frame = random.randint(0, total_frames - self.seq_len)
        
        imgs = imgs_full[start_frame:start_frame+self.seq_len]
        actions = actions_full[start_frame:start_frame+self.seq_len]
        
        # 4. 随机时间偏移（破解时序过拟合）
        time_offset = random.uniform(0.0, 10000.0)
        time_steps = torch.arange(self.seq_len, dtype=torch.float32) / self.fps
        t_vec = time_steps + time_offset # [T]
        
        return {
            "img": imgs,          # [T, 3, H, W]
            "action": actions,    # [T, A_dim]
            "task_text": data["task"],
            "t_vec": t_vec        # [T] 绝对时间戳
        }


class VPTStreamDataset(IterableDataset):
    """流式 VPT 加载器(不全量预载,适合大量/长视频)。

    每个 DataLoader worker 维护一个 <=cache_size 个已解码片段的滚动缓存:
      - 缓存不足时随机载入新片段补满;
      - 每 yield refresh_every 个样本,随机换出一个旧片段(下一轮补新的)=> 流式覆盖全量;
      - 每次从缓存里随机选一个片段、随机截 seq_len 窗口 yield。
    配合 DataLoader(num_workers=K, batch_size=B) 即多进程并行喂数据,每步随机取 B 个序列。
    本身无限迭代,训练侧用固定 steps_per_epoch 截断。
    """

    def __init__(self, data_dir, seq_len=60, fps=20, cache_size=32, refresh_every=64, seed=0):
        super().__init__()
        self.seq_len, self.fps = seq_len, fps
        self.cache_size = max(1, cache_size)
        self.refresh_every = max(1, refresh_every)
        self.seed = seed
        self.pairs = _pair_list(data_dir)
        if not self.pairs:
            raise RuntimeError(f"[VPTStreamDataset] {data_dir} 里没有成对的 .mp4/.jsonl")
        print(f"[VPTStreamDataset] {len(self.pairs)} clips on disk | cache<={self.cache_size}/worker "
              f"| refresh every {self.refresh_every} samples")

    def _refill(self, cache, rng):
        attempts, cap = 0, 2 * self.cache_size + len(self.pairs) + 4
        while len(cache) < self.cache_size and attempts < cap:
            attempts += 1
            clip = _decode_clip(*rng.choice(self.pairs), self.seq_len)
            if clip is not None:
                cache.append(clip)
        if not cache:
            raise RuntimeError("[VPTStreamDataset] 没有 >= seq_len 的片段,调小 --seq_len 或下载更长数据")

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi is not None else 0))
        cache, served = [], 0
        while True:
            self._refill(cache, rng)
            clip = rng.choice(cache)
            T = clip["img"].shape[0]
            s = rng.randint(0, T - self.seq_len)
            tv = torch.arange(self.seq_len, dtype=torch.float32) / self.fps + rng.uniform(0.0, 1e4)
            served += 1
            yield {
                "img": clip["img"][s:s + self.seq_len],
                "action": clip["action"][s:s + self.seq_len],
                "task_text": clip["task"],
                "t_vec": tv,
            }
            if served % self.refresh_every == 0 and len(cache) > 0:
                cache.pop(rng.randrange(len(cache)))   # 换出一个旧片段
