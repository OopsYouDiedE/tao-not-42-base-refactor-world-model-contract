import os
import json
import random
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

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
        
        # 寻找所有成对的 mp4 和 jsonl
        vid_paths = []
        for f in os.listdir(data_dir):
            if f.endswith(".mp4"):
                base = f[:-4]
                jsonl_path = os.path.join(data_dir, f"{base}.jsonl")
                if os.path.exists(jsonl_path):
                    vid_paths.append((os.path.join(data_dir, f), jsonl_path))
                    
        print(f"[VPTDataset] Found {len(vid_paths)} video clips. Pre-loading into RAM for maximum GPU speed...")
        
        for vid_path, json_path in vid_paths:
            # 1. 一次性读取整个视频到内存
            cap = cv2.VideoCapture(vid_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
            cap.release()
            imgs = torch.stack(frames, dim=0)
            
            # 2. 一次性读取对应的 JSONL
            actions_list = []
            task_text = ""
            with open(json_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for t, line in enumerate(lines):
                    act_dict = json.loads(line)
                    if t == 0 and "task" in act_dict:
                        task_text = act_dict["task"]
                    mouse = act_dict.get("mouse", {"dx": 0.0, "dy": 0.0})
                    kb = act_dict.get("keyboard", {})
                    KEYS = ["key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak", 
                            "key_sprint", "key_attack", "key_use", "key_drop", "key_inventory"]
                    for i in range(1, 10):
                        KEYS.append(f"key_hotbar.{i}")
                    kb_vec = [float(kb.get(k, 0)) for k in KEYS]
                    action_vec = [mouse["dx"], mouse["dy"]] + kb_vec
                    actions_list.append(torch.tensor(action_vec, dtype=torch.float32))
            
            # 对齐长度
            min_len = min(imgs.shape[0], len(actions_list))
            imgs = imgs[:min_len]
            actions = torch.stack(actions_list[:min_len], dim=0)
            
            if min_len >= self.seq_len:
                self.videos.append({
                    "img": imgs,
                    "action": actions,
                    "task": task_text
                })

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
