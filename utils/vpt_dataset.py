import os
import json
import random
import cv2
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np

from utils.vpt_action import CAMERA_SCALE, N_MOUSE

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

    核心机制:**clip 级内存缓存 + 滚动刷新**(吞吐演化史,三版教训):
      - v1 整段 float32 预载:5 分钟 360×640 一段 ≈ 16.6GB,真数据必然 OOM;
      - v2 每窗口随机 seek 解码:内存安全,但随机 seek 要从最近关键帧白解上百帧,
        实测 ~1 窗/s/worker,GPU 一步吃 batch 个窗口 ⇒ 大 batch 下 GPU 长期 0%
        利用率饿死在等数据(供需差 >3×,buffer 复用只能缓解不能根治);
      - v3(现实现)整段顺序解码到 **img_size 分辨率 uint8** 驻留内存:
        128px 下一段 5 分钟 clip 仅 ≈ 0.29GB,clip_cache=4 段/worker ≈ 1.2GB;
        顺序读零 seek 浪费(~20s/段,一次性),之后切窗口是**纯内存索引**(免费);
        每产出 clip_refresh 个窗口滚动换入一段新 clip(FIFO 逐出最老)。
        数据多样性由刷新供给,吞吐与解码彻底解耦 ⇒ GPU-bound。
    帧以 uint8 缓存/返回(归一化推迟到 GPU 上做,内存与 PCIe 流量都降 4×)。
    ⚠ img_size=None(原生分辨率)时整段缓存很大(360×640 ≈ 4GB/段),训练请务必
    设 img_size。cache_size/refresh_every 已废弃(并入 clip 缓存),仅保留 CLI 兼容。
    本身无限迭代,训练侧用固定 steps_per_epoch 截断。

    frame_skip:**可变时间跨度的上限**。每个转移独立采样间隔 Δt ~ U{1..frame_skip}
    (帧),图像只在采样点解码(跳过帧 grab() 不解码)。数学动机:
      - 20fps 下相邻帧潜表征几乎相同,固定一步预测的动力学信号被 persistence 淹没;
      - **可变** Δt 消除"固定步长默认漂移先验"——唯一能解释 Δz 的就是把区间内动作
        逐个积分,这正是开环推演所需的能力(jumpy / temporally-abstract prediction);
      - 动作效应随 Δt 近似线性累积而编码噪声地板不变 ⇒ 大 Δt 样本信噪比更高,
        混合采样自带课程。
    每个转移同时给出:**区间内完整的原始动作序列** act_seq(信息无损,右侧零填充
    到 frame_skip,有效长度=dt)与聚合动作 act_agg(鼠标区间平均/键盘 max,
    供历史 token 与逆动力学目标用——从单个 Δz 反推逐帧序列是欠定问题,反推净
    效应才良定义)。

    split:"train"/"holdout"/None。按文件名排序后扣末 holdout_n 个为 holdout
    (确定性切分,与 seed 无关)——可视化与最终评估必须用 holdout,否则展示的是
    记忆而非泛化。clip 总数不足时退化为全量并告警。

    clip_cache:每个 worker 常驻内存的整段 clip 数(滚动 FIFO)。窗口从缓存内
    随机一段随机位置切出 ⇒ batch 内样本多样性随缓存段数增长(冷启动从第 1 段
    解码完就开始供数,之后每次刷新多缓存一段直到打满)。
    clip_refresh:每产出多少个窗口换入一段新 clip(多样性/解码停顿的折中:
    一段解码 ~20s,refresh=256 时摊到每窗口 <0.1s)。

    buffer_size/buffer_reuse:窗口级滚动缓存(0/1=关闭,v2 时代的吞吐杠杆,
    clip 缓存落地后窗口切片已免费,保留仅为兼容,正常不需要开)。
    """

    def __init__(self, data_dir, seq_len=60, fps=20, cache_size=32, refresh_every=64,
                 seed=0, img_size=None, camera_scale=CAMERA_SCALE, frame_skip=1,
                 split=None, holdout_n=1, buffer_size=0, buffer_reuse=1,
                 clip_cache=4, clip_refresh=256):
        super().__init__()
        self.seq_len, self.fps = seq_len, fps
        self.cache_size = max(1, cache_size)
        self.seed = seed
        self.img_size = img_size
        self.camera_scale = camera_scale
        self.frame_skip = max(1, int(frame_skip))
        self.buffer_size = max(0, int(buffer_size))
        self.buffer_reuse = max(1, int(buffer_reuse))
        self.clip_cache = max(1, int(clip_cache))
        self.clip_refresh = max(1, int(clip_refresh))
        pairs = _pair_list(data_dir)
        if not pairs:
            raise RuntimeError(f"[VPTStreamDataset] {data_dir} 里没有成对的 .mp4/.jsonl")
        if split in ("train", "holdout"):
            if len(pairs) > holdout_n:
                pairs = pairs[:-holdout_n] if split == "train" else pairs[-holdout_n:]
            else:
                print(f"[VPTStreamDataset] ⚠ 只有 {len(pairs)} 个 clip,无法切分 "
                      f"split={split}——退化为全量(评估/可视化将与训练同源,数字偏乐观)")
        self.pairs = pairs
        print(f"[VPTStreamDataset] {len(self.pairs)} clips ({split or 'all'}) | "
              f"clip 级内存缓存(uint8) | img_size={img_size or 'native'} "
              f"| camera_scale={camera_scale:.1f} | Δt~U{{1..{self.frame_skip}}} "
              f"| 缓存={self.clip_cache} 段/worker,每 {self.clip_refresh} 窗口滚动换入")

    def _load_clip(self, mp4, jsonl):
        """顺序解码整段视频(img_size 分辨率,uint8)+ 动作表 → 常驻内存的 clip 条目。

        顺序读零 seek 浪费(随机 seek 要从最近关键帧白解上百帧,是 v2 实现 GPU
        饿死的根因);解一次之后,从该 clip 切任意窗口都是纯内存索引。
        先 resize 再 cvtColor:色彩转换在 128×128 上做比 360×640 省 ~14× 像素量,
        INTER_AREA 对 BGR/RGB 通道顺序不敏感,两步可交换。
        """
        actions, task = [], ""
        with open(jsonl, "r", encoding="utf-8") as f:
            for t, line in enumerate(f):
                a = json.loads(line)
                if t == 0 and "task" in a:
                    task = a["task"]
                actions.append(_action_vec(a, self.camera_scale))
        cap = cv2.VideoCapture(mp4)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            if self.img_size:
                f = cv2.resize(f, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        n = min(len(frames), len(actions))
        if n == 0:
            return None
        img = torch.from_numpy(np.stack(frames[:n])).permute(0, 3, 1, 2).contiguous()
        return {"img": img, "action": torch.stack(actions[:n]), "task": task, "n": n}

    def _split_actions(self, act, start, skips):
        """把 [start, start+Σskips) 的逐帧动作按转移切开。

        返回:
          act_seq [T-1, frame_skip, A] —— 每个转移区间内的**原始逐帧动作序列**,
            右侧零填充(有效长度 = dt[t],模型端据 dt 构造有效位,勿用全零判别:
            "什么都没按"本身就是全零动作)。
          act_agg [T-1, A] —— 区间净效应:鼠标取**区间平均**(平均角速度),键盘按过即 1。

        鼠标为什么是平均而不是求和:逐帧 dx 已按 camera_scale 截 ±1,求和再截 ±1
        在 frame_skip=8 下持续转身 4~5 帧就饱和——目标退化成近似三值 {-1,0,+1},
        mu-law 分箱全挤在边缘 bin,逆动力学 CE 卡在边缘分布熵的平凡解上
        (实测 mouse_acc 0.66 平台 + 可视化 dx true 贴 ±1 轨道)。区间平均与逐帧
        动作同尺度(历史 token 与当前动作可比)、对 dt 不变、几乎不饱和;模型
        另有 dt 输入,需要总转角时可自行乘回。
          dt [T-1] —— 各转移的帧跨度(float)。
        """
        A = act.shape[1]
        T1 = len(skips)
        seq = torch.zeros(T1, self.frame_skip, A)
        agg = torch.zeros(T1, A)
        pos = start
        for t, s in enumerate(skips):
            w = act[pos: pos + s]
            seq[t, :w.shape[0]] = w
            agg[t, :N_MOUSE] = (w[:, :N_MOUSE].sum(dim=0) / s).clamp(-1.0, 1.0)
            agg[t, N_MOUSE:] = w[:, N_MOUSE:].max(dim=0).values
            pos += s
        return seq, agg, torch.tensor(skips, dtype=torch.float32)

    def __iter__(self):
        wi = get_worker_info()
        if wi is not None:
            # 多 worker 时禁用 cv2 内部线程池:N 个 worker × cv2 默认全核线程
            # 会把 vCPU 超订成上下文切换,窗口解码吞吐反而下降。
            cv2.setNumThreads(0)
        rng = random.Random(self.seed + (wi.id if wi is not None else 0))
        # 窗口级滚动缓存(兼容遗留,正常关闭;quota=0 表示关闭)
        n_w = wi.num_workers if wi is not None else 1
        quota = max(1, self.buffer_size // n_w) if self.buffer_size > 0 else 0
        buf, ptr = [], 0
        clips, served, fails = {}, 0, 0     # clip 级内存缓存:mp4 -> _load_clip 条目
        while True:
            # 换入 clip:缓存空(冷启动,解完第 1 段立即开始供数)或已产出
            # clip_refresh 个窗口(滚动刷新;未满时只增,满了 FIFO 逐出最老)
            if not clips or served >= self.clip_refresh:
                mp4, jsonl = self.pairs[rng.randrange(len(self.pairs))]
                if mp4 not in clips:
                    c = self._load_clip(mp4, jsonl)
                    if c is not None:
                        if len(clips) >= self.clip_cache:
                            clips.pop(next(iter(clips)))
                        clips[mp4] = c
                served = 0
                if not clips:
                    fails += 1
                    if fails > 4 * len(self.pairs) + 8:
                        raise RuntimeError("[VPTStreamDataset] 没有可解码的 clip")
                    continue
            # 每个转移独立采样跨度 Δt ~ U{1..frame_skip}(可变间隔,见类 docstring)
            skips = [rng.randint(1, self.frame_skip) for _ in range(self.seq_len - 1)]
            span = sum(skips) + 1                             # 窗口占用的原始帧数
            cand = [c for c in clips.values() if c["n"] >= span]
            if not cand:
                fails += 1
                served = self.clip_refresh                    # 强制下轮换入新 clip
                if fails > 4 * len(self.pairs) + 8:
                    raise RuntimeError(
                        "[VPTStreamDataset] 没有足够长的片段,调小 --seq_len/--frame_skip 或下载更长数据")
                continue
            fails = 0
            m = cand[rng.randrange(len(cand))]
            start = rng.randint(0, m["n"] - span)
            # 采样帧索引(可变间隔 ⇒ 非等差);窗口 = 缓存 clip 的纯内存切片
            fidx = torch.tensor([0] + skips, dtype=torch.long).cumsum(0) + start
            img = m["img"][fidx]
            act_seq, act_agg, dt = self._split_actions(m["action"], start, skips)
            tv = fidx.float() / self.fps + rng.uniform(0.0, 1e4)
            served += 1
            sample = {
                "img": img,                # uint8 [T,3,H,W]
                "act_seq": act_seq,        # [T-1, frame_skip, A] 区间内原始动作(零填充)
                "act_agg": act_agg,        # [T-1, A] 区间净效应(历史 token/inv-dyn 目标)
                "dt": dt,                  # [T-1] 帧跨度
                "task_text": m["task"],
                "t_vec": tv,               # [T]
            }
            if quota == 0:
                yield sample
                continue
            # 窗口级滚动缓存(兼容遗留:clip 缓存落地后窗口切片已免费,正常不开)
            if len(buf) < quota:
                buf.append(sample)
            else:
                buf[ptr] = sample
                ptr = (ptr + 1) % quota
            for _ in range(self.buffer_reuse):
                yield buf[rng.randrange(len(buf))]
