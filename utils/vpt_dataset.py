import os
import json
import random
import cv2
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np

from utils.vpt_action import CAMERA_SCALE

# 动作向量布局(与 download_sample_data / colab 转换脚本严格一致):2 鼠标 + 20 键盘
# 与 utils/vpt_action.py 的契约一致:鼠标在前(索引 0,1),且按 CAMERA_SCALE 归一化。
VPT_KEYS = ["key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak",
            "key_sprint", "key_attack", "key_use", "key_drop", "key_inventory"] \
           + [f"key_hotbar.{i}" for i in range(1, 10)]


def _action_vec(act_dict, camera_scale=CAMERA_SCALE):
    """单帧 jsonl dict -> [2+20] float tensor。鼠标按 camera_scale 归一化到约 [-1,1],
    与键盘 0/1 同尺度(否则像素级 dx 主导 action_enc 线性层与逆动力学 MSE)。

    camera_scale 是**固定超参**(标定一次写死,不每次重估):合成数据(download_sample_data)
    σ≈6,默认 CAMERA_SCALE=10 即可;真 BASALT 相机以**度**计、转身可达 ±190,用 ~20
    (见 colab_demo 注释),经 train_minecraft --camera_scale 传入。
    """
    mouse = act_dict.get("mouse", {"dx": 0.0, "dy": 0.0})
    kb = act_dict.get("keyboard", {})
    s = max(float(camera_scale), 1e-6)
    dx = max(-1.0, min(1.0, mouse["dx"] / s))
    dy = max(-1.0, min(1.0, mouse["dy"] / s))
    return torch.tensor([dx, dy] + [float(kb.get(k, 0)) for k in VPT_KEYS],
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
    """流式 VPT 加载器(随机窗口按需解码;真 BASALT 长视频可用)。

    旧实现把**整段视频**解码成 float32 驻留内存:BASALT contractor 一段 5 分钟
    360×640@20fps 视频 ≈ 16.6 GB,cache_size=32/worker 的设计需要 TB 级内存——
    只在合成小样本上跑得动,真数据必然 OOM 且首样本延迟为整段解码时间。现实现:
      - 每个样本只 seek + 解码一个 seq_len 窗口(解码量降 ~T_total/seq_len 倍);
      - 帧以 uint8 返回(归一化推迟到 GPU 上做,内存与 PCIe 流量都降 4×);
      - img_size 可选下采样(360×640 → 128 训练分辨率,卷积与带宽都省);
      - 动作 jsonl 逐文件解析一次,worker 本地缓存 <=cache_size 个文件的动作表
        (动作表很小,帧不缓存)。
    refresh_every 已废弃(没有整段缓存可换出),保留参数仅为 CLI 兼容。
    本身无限迭代,训练侧用固定 steps_per_epoch 截断。
    """

    def __init__(self, data_dir, seq_len=60, fps=20, cache_size=32, refresh_every=64,
                 seed=0, img_size=None, camera_scale=CAMERA_SCALE):
        super().__init__()
        self.seq_len, self.fps = seq_len, fps
        self.cache_size = max(1, cache_size)
        self.seed = seed
        self.img_size = img_size
        self.camera_scale = camera_scale
        self.pairs = _pair_list(data_dir)
        if not self.pairs:
            raise RuntimeError(f"[VPTStreamDataset] {data_dir} 里没有成对的 .mp4/.jsonl")
        print(f"[VPTStreamDataset] {len(self.pairs)} clips on disk | 窗口按需解码(uint8) "
              f"| img_size={img_size or 'native'} | camera_scale={camera_scale:.1f} "
              f"| 动作表缓存<={self.cache_size} 文件/worker")

    def _load_meta(self, mp4, jsonl):
        """解析一个文件对的动作表与可用帧数(不解码帧)。"""
        actions, task = [], ""
        with open(jsonl, "r", encoding="utf-8") as f:
            for t, line in enumerate(f):
                a = json.loads(line)
                if t == 0 and "task" in a:
                    task = a["task"]
                actions.append(_action_vec(a, self.camera_scale))
        cap = cv2.VideoCapture(mp4)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n = min(n_frames, len(actions))
        return {"action": torch.stack(actions[:n]) if n > 0 else None,
                "task": task, "n": n}

    def _decode_window(self, mp4, start, T):
        """seek 到 start 解码 T 帧 → uint8 [T,3,H,W];不足返回 None。"""
        cap = cv2.VideoCapture(mp4)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(T):
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if self.img_size:
                f = cv2.resize(f, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(f))
        cap.release()
        if len(frames) < T:
            return None
        return torch.stack(frames).permute(0, 3, 1, 2).contiguous()

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi is not None else 0))
        meta_cache, fails = {}, 0
        while True:
            mp4, jsonl = self.pairs[rng.randrange(len(self.pairs))]
            m = meta_cache.get(mp4)
            if m is None:
                m = self._load_meta(mp4, jsonl)
                if len(meta_cache) >= self.cache_size:
                    meta_cache.pop(rng.choice(list(meta_cache.keys())))
                meta_cache[mp4] = m
            if m["n"] < self.seq_len:
                fails += 1
                if fails > 4 * len(self.pairs) + 8:
                    raise RuntimeError(
                        "[VPTStreamDataset] 没有 >= seq_len 的片段,调小 --seq_len 或下载更长数据")
                continue
            start = rng.randint(0, m["n"] - self.seq_len)
            img = self._decode_window(mp4, start, self.seq_len)
            if img is None:
                fails += 1
                continue
            fails = 0
            tv = ((start + torch.arange(self.seq_len, dtype=torch.float32)) / self.fps
                  + rng.uniform(0.0, 1e4))
            yield {
                "img": img,                                          # uint8 [T,3,H,W]
                "action": m["action"][start:start + self.seq_len].clone(),
                "task_text": m["task"],
                "t_vec": tv,
            }
