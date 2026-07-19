"""读取成对的 Minecraft VPT 视频和动作标注。"""

import os
import json
import random
import threading
import time

import cv2
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np

from train.minecraft.vpt_action_contract import CAMERA_SCALE, N_MOUSE

# 动作向量布局(与 download_sample_data / colab 转换脚本严格一致):2 鼠标 + 20 键盘
# 与 vpt_action_contract.py 一致：鼠标在前（索引 0,1），并按 CAMERA_SCALE 归一化。
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


def assemble_goal(ids, aim1000, goal_mat):
    """逐帧 (词表 id, aim) → goal 向量批(hindsight relabel 契约,单测锚定)。

    ids [T] long(-1=无标签)、aim1000 [T,2] float(0..1000)、goal_mat [V,384]
    (MiniLM L2 归一句向量,词表文件由 hindsight_relabel CLI 落盘)。
    返回 goal [T,386] = 文本向量 ⊕ aim/1000；
    无标签帧全零(与既有零 goal BC 行为兼容)。
    """
    g = torch.zeros(ids.shape[0], goal_mat.shape[1] + 2)
    m = ids >= 0
    if m.any():
        g[m, :goal_mat.shape[1]] = goal_mat[ids[m]]
        g[m, goal_mat.shape[1]:] = aim1000[m] / 1000.0
    return g


def load_goal_vocab(path):
    """goal 词表 json {text: [384]} → (text→行号 dict, [V,384] 矩阵)。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    texts = sorted(d)
    mat = torch.tensor([d[t] for t in texts], dtype=torch.float32)
    return {t: i for i, t in enumerate(texts)}, mat


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

    **滚动目录模式(split=None)**:目录内容可以随时增删(配合后台滚动下载器:
    全量索引循环流式下载、磁盘只保留滑动窗口)。此模式下每次换 clip 都重扫目录、
    随机抽"当前已有"的段——下载快慢只影响数据轮换速度,训练采样永不阻塞;
    文件被删/半写导致的读取失败静默跳过重试,目录暂空时等待下载器而不报错。
    注意:滚动目录下按文件名切 train/holdout 不稳定(文件来来去去,"排序末 N 个"
    会漂移),所以滚动模式要求 holdout 放在**独立的固定目录**(train_minecraft
    --holdout_dir),本目录整体作为训练池。

    clip_cache:每个 worker 常驻内存的整段 clip 数(滚动 FIFO)。窗口从缓存内
    随机一段随机位置切出 ⇒ batch 内样本多样性随缓存段数增长(冷启动从第 1 段
    解码完就开始供数,之后随刷新逐步打满)。
    clip_refresh:每产出多少个窗口**触发一次换段**。换段解码在**后台线程**进行:
    采样循环只从"已在内存里的段"随机抽、永不等待解码——解码期间照常产出,
    解完随到随换(FIFO 逐出最老)。因此实际轮换周期 = max(产出 clip_refresh 个
    窗口的时间, 一段解码时长 ~20s);下载/解码慢只影响数据新鲜度,不影响吞吐。
    唯一允许阻塞的时刻是冷启动(缓存全空,必须同步解第一段)。

    buffer_size/buffer_reuse:窗口级滚动缓存(0/1=关闭,v2 时代的吞吐杠杆,
    clip 缓存落地后窗口切片已免费,保留仅为兼容,正常不需要开)。
    """

    def __init__(self, data_dir, seq_len=60, fps=20, cache_size=32, refresh_every=64,
                 seed=0, img_size=None, camera_scale=CAMERA_SCALE, frame_skip=1,
                 split=None, holdout_n=1, buffer_size=0, buffer_reuse=1,
                 clip_cache=4, clip_refresh=256, motion_sample=1, clip_max_frames=0,
                 goal_vocab=None):
        super().__init__()
        # goal_vocab:hindsight relabel 词表 json 路径(hindsight_relabel CLI 落盘)。
        # 提供则样本多出 "goal" [T,386] 与 "goal_on" [T] bool;词表外文本(下载器
        # 新段引入的新 item)按零 goal 处理并计数——重跑 relabel --vocab-only 可补齐。
        self.goal_idx, self.goal_mat, self.goal_unknown = None, None, 0
        if goal_vocab:
            self.goal_idx, self.goal_mat = load_goal_vocab(goal_vocab)
        # motion_sample:运动量锦标赛采样——每窗口抽 k 个候选,取采样帧差能量最大者
        # (k=1 关闭,保持均匀采样)。动机:均匀采样下静止转移占多数,persistence 基线
        # 恰赢在这些样本上;向高运动转移倾斜可把梯度集中到"动作产生效果"的地方。
        # 有偏是有意的课程,评估侧(holdout)不开启即可保持口径公正。
        self.motion_sample = max(1, int(motion_sample))
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
        # 超长段(如 gaming500 的 30 分钟录屏)整段缓存会爆内存:>0 时每次换段只解码
        # 随机起点的连续 clip_max_frames 帧(一次 keyframe seek,分布仍覆盖全段)。
        self.clip_max_frames = max(0, int(clip_max_frames))
        self.data_dir = data_dir
        self.rescan = split is None          # 滚动目录模式:换 clip 时重扫目录
        pairs = _pair_list(data_dir)
        if not pairs:
            if self.rescan:
                print(f"[VPTStreamDataset] {data_dir} 暂时为空——滚动目录模式,等待下载器填充")
            else:
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

    def _load_clip(self, mp4, jsonl, rng=None):
        """顺序解码整段视频(img_size 分辨率,uint8)+ 动作表 → 常驻内存的 clip 条目。

        顺序读零 seek 浪费(随机 seek 要从最近关键帧白解上百帧,是 v2 实现 GPU
        饿死的根因);解一次之后,从该 clip 切任意窗口都是纯内存索引。
        先 resize 再 cvtColor:色彩转换在 128×128 上做比 360×640 省 ~14× 像素量,
        INTER_AREA 对 BGR/RGB 通道顺序不敏感,两步可交换。
        文件不存在/半写坏(滚动下载器随时增删)→ 返回 None,调用方换一段重试。
        """
        actions, gflags, task = [], [], ""
        sg_ids, aims = [], []                # hindsight relabel:词表 id(-1 无标签)+ aim
        # 可选反事实 GT(download_sample_data --counterfactual 写入;真 VPT 缺这些字段)
        gt = {"has_item": [], "airborne": [], "branch": [], "reach_state_id": []}
        have_gt = False
        try:
            with open(jsonl, "r", encoding="utf-8") as f:
                for t, line in enumerate(f):
                    a = json.loads(line)
                    if t == 0 and "task" in a:
                        task = a["task"]
                    actions.append(_action_vec(a, self.camera_scale))
                    gflags.append(1.0 if a.get("gui") else 0.0)
                    if self.goal_idx is not None:
                        sg = a.get("subgoal")
                        i = self.goal_idx.get(sg, -1) if sg else -1
                        if sg and i < 0:
                            self.goal_unknown += 1
                        sg_ids.append(i)
                        aims.append(a.get("aim") or [0, 0])
                    for key in gt:
                        if key in a:
                            have_gt = True
                        gt[key].append(float(a.get(key, -1)))
        except (OSError, ValueError):
            return None
        cap = cv2.VideoCapture(mp4)
        start, limit = 0, self.clip_max_frames
        if limit and len(actions) > limit and rng is not None:
            start = rng.randrange(len(actions) - limit + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)     # 每段仅一次 seek,代价可摊
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            if self.img_size:
                # int → 方形(遗留口径);(H,W) 元组 → 保宽高比预设(cv2.resize 收 (W,H))
                hw = ((self.img_size, self.img_size) if isinstance(self.img_size, int)
                      else tuple(self.img_size))
                f = cv2.resize(f, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if limit and len(frames) >= limit:
                break
        cap.release()
        if start:
            actions, gflags = actions[start:], gflags[start:]
            sg_ids, aims = sg_ids[start:], aims[start:]
            gt = {k: v[start:] for k, v in gt.items()}
        n = min(len(frames), len(actions))
        if n == 0:
            return None
        img = torch.from_numpy(np.stack(frames[:n])).permute(0, 3, 1, 2).contiguous()
        # gui 标记(jsonl "gui" 字段,colab §2 写入):GUI 打开时画面变化(光标/物品)
        # 无法被记录的动作解释——纯标签噪声,采样时拒采。全零则存 None 免查。
        gui = torch.tensor(gflags[:n]) if any(gflags[:n]) else None
        out = {"img": img, "action": torch.stack(actions[:n]), "task": task,
               "gui": gui, "n": n}
        if self.goal_idx is not None:
            out["sg"] = torch.tensor(sg_ids[:n], dtype=torch.long)
            out["aim1000"] = torch.tensor(aims[:n], dtype=torch.float32)
        if have_gt:
            out["gt"] = {k: torch.tensor(v[:n], dtype=torch.float32) for k, v in gt.items()}
        return out

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
        pend, ldr = [], [None]              # 后台解码:结果槽 + 在途线程

        def _spawn_loader():
            """挑一段未缓存的 clip,起后台线程解码(不阻塞采样循环)。"""
            pairs = _pair_list(self.data_dir) if self.rescan else self.pairs
            if not pairs:
                return
            mp4, jsonl = pairs[rng.randrange(len(pairs))]
            if mp4 in clips:
                return                       # 抽中已缓存的段:本轮放弃,下次再抽

            def _bg():
                c = self._load_clip(mp4, jsonl, rng)
                if c is not None:
                    pend.append((mp4, c))

            ldr[0] = threading.Thread(target=_bg, daemon=True)
            ldr[0].start()

        while True:
            # 收割后台解码完成的段:随到随换(FIFO 逐出最老),采样从不等它
            if ldr[0] is not None and not ldr[0].is_alive():
                ldr[0] = None
                while pend:
                    mp4, c = pend.pop()
                    if len(clips) >= self.clip_cache:
                        clips.pop(next(iter(clips)))
                    clips[mp4] = c
                    served = 0
            if not clips:
                # 冷启动/池空:唯一允许同步阻塞的时刻(没有任何段可采)
                pairs = _pair_list(self.data_dir) if self.rescan else self.pairs
                if pairs:
                    mp4, jsonl = pairs[rng.randrange(len(pairs))]
                    c = self._load_clip(mp4, jsonl, rng)
                    if c is not None:
                        clips[mp4] = c
                        served = 0
                if not clips:
                    fails += 1
                    if self.rescan:           # 滚动目录:等下载器,不报错
                        if fails % 15 == 1:
                            print(f"[VPTStreamDataset] 等待 {self.data_dir} 出现可用 clip...")
                        time.sleep(2.0)
                        continue
                    if fails > 4 * len(self.pairs) + 8:
                        raise RuntimeError("[VPTStreamDataset] 没有可解码的 clip")
                    continue
            elif served >= self.clip_refresh and ldr[0] is None:
                _spawn_loader()              # 额度用完 → 异步换段,继续从现有缓存采样
            # 每个转移独立采样跨度 Δt ~ U{1..frame_skip}(可变间隔,见类 docstring)
            skips = [rng.randint(1, self.frame_skip) for _ in range(self.seq_len - 1)]
            span = sum(skips) + 1                             # 窗口占用的原始帧数
            cand = [c for c in clips.values() if c["n"] >= span]
            if not cand:
                fails += 1
                served = self.clip_refresh                    # 触发换段(异步)
                if not self.rescan and fails > 4 * len(self.pairs) + 8:
                    raise RuntimeError(
                        "[VPTStreamDataset] 没有足够长的片段,调小 --seq_len/--frame_skip 或下载更长数据")
                time.sleep(0.05)              # 缓存里全是短段:避免等后台解码时热旋
                continue
            # 候选抽取(motion_sample>1 时锦标赛:通过 GUI 检查的候选里取运动能量最大)
            picks = []
            for _ in range(self.motion_sample):
                mm = cand[rng.randrange(len(cand))]
                st = rng.randint(0, mm["n"] - span)
                # GUI 窗口拒采:GUI 内画面变化与记录动作零相关(标签噪声),>10% 即弃
                if mm["gui"] is not None and float(mm["gui"][st:st + span].mean()) > 0.1:
                    continue
                picks.append((mm, st))
            if not picks:
                fails += 1
                if fails % 8 == 0:
                    served = self.clip_refresh        # 反复命中 GUI 段:触发换 clip
                if not self.rescan and fails > 16 * len(self.pairs) + 64:
                    raise RuntimeError("[VPTStreamDataset] 可用 clip 几乎全是 GUI 段")
                continue
            if len(picks) == 1:
                m, start = picks[0]
            else:
                def _motion(mm, st):
                    f = mm["img"][st:st + span:max(1, span // 8)].to(torch.int16)
                    return float((f[1:] - f[:-1]).abs().float().mean())
                m, start = max(picks, key=lambda p: _motion(*p))
            fails = 0
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
            if self.goal_idx is not None:  # hindsight goal(无标签帧全零,契约见 assemble_goal)
                ids = m.get("sg")
                if ids is None:            # 未标注段(下载器新段):全零 goal
                    sample["goal"] = torch.zeros(len(fidx), self.goal_mat.shape[1] + 2)
                    sample["goal_on"] = torch.zeros(len(fidx), dtype=torch.bool)
                else:
                    sample["goal"] = assemble_goal(ids[fidx], m["aim1000"][fidx],
                                                   self.goal_mat)
                    sample["goal_on"] = ids[fidx] >= 0
            # 反事实 GT 透传(存在才加):按采样帧索引 fidx 切片 → [T];真 VPT 无此键
            if "gt" in m:
                for key, vec in m["gt"].items():
                    sample[key] = vec[fidx]
                sample["reach_id"] = sample["reach_state_id"]
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
