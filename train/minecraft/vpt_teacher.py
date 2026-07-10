# -*- coding: utf-8 -*-
"""VPT 教师(rl-from-foundation-2x)→ CraftGround V2 契约的翻译层 + 离线打标 CLI。

对外接口:
    VPTTeacher                — 教师策略加载与逐段前向(带 transformer 记忆状态)
    teacher_to_v2(pi_logits)  — 教师层级分布 → (p_keys [.,20], cam_t [.,2,11], p_cam_on)
    teacher_bin_pushforward() — 教师相机 bin(mu-law μ=10, ±10°) → 我们 bin(μ=8, ±18°)
                                的下推矩阵 P [11,11](行=教师 bin,列=我们 bin,0/1)
    remap_cam(cam_t)          — 教师 bin 边缘分布 → 我们 11-bin 分布(概率质量守恒)
    label_segment(...)        — 单段 mp4 → 教师逐帧分布(npz 载荷 dict)
    main()                    — 打标 CLI:python -m train.minecraft.vpt_teacher --pool ...

翻译原则(苦涩教训两判据,手写规则最小化):
  - 键位:教师 Buttons.ALL 与 V2_KEYS 是**同名双射**(20 键无一丢弃),置换按名字
    机械生成,零手写映射。教师联合 8641 类 → 20 独立 Bernoulli 用精确边缘化
    (`BUTTON_IDX_TO_FACTORED` 矩阵乘),丢的是键间相关结构,不是键本身。
  - 相机:教师 121 联合 → 两轴独立 11-way 用精确边缘化;camera 元动作关(meta off)
    的概率质量并入中心 bin(与上游 `to_factored` 的确定性语义一致)。教师轴序
    (pitch,yaw)(MineRL camera 约定,axis0=dy) → 我们 (dx,dy),显式换轴。
    教师 bin→度 用上游 `CameraQuantizer.undiscretize`,度→我们 bin 用
    `action_contract.deg_to_bins`,两端都是既有契约函数,无手写数值。
  - 如实丢弃并统计(归慢塔职责,不编翻译):GUI 帧(教师在 GUI 里把相机当光标、
    attack 当点击,快塔契约剔除 GUI tick)——打标仍全帧落盘,`gui` 掩码留给训练侧;
    打标统计 gui 帧占比与 GUI 帧上的教师 attack/use 质量。教师无 ESC 头,无额外丢弃。
  - 教师温度:上游 .model 的 pi_head temperature(=2.0)烧在 head 前向里,输出即
    部署口径的 log-prob("开箱能拿木头"的那套分布);打标 manifest 记录该值。

帧口径(与上游 agent.py/data_loader.py 逐条对齐):BGR→RGB、cv2.INTER_LINEAR
resize 到 128×128、uint8;20fps 与 CraftGround tick 同频,教师"度/帧"=我们"度/tick"。
GUI 光标合成(上游 data_loader 仅 GUI 帧做)不做——GUI 帧本就被快塔契约掩掉。

Shape/Dtype 契约:教师 pi_logits {"buttons":[B,T,8641] fp32 log-prob,
"camera":[B,T,121] fp32 log-prob};翻译输出 p_keys [B,T,20] fp32 概率(V2 键序)、
cam_t [B,T,2,11] fp32 概率(轴序 dx,dy;教师 bin)、p_cam_on [B,T] fp32。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.vpt_lib.action_mapping import CameraHierarchicalMapping   # noqa: E402
from net.vpt_lib.actions import Buttons, CameraQuantizer, QuantizationScheme  # noqa: E402
from net.vpt_lib.policy import MinecraftAgentPolicy               # noqa: E402
from net.vpt_lib.tree_util import tree_map                        # noqa: E402
from train.craftground.action_contract import CAM_BINS, deg_to_bins  # noqa: E402
from train.craftground.action_contract import V2_KEYS             # noqa: E402

TEACHER_RESOLUTION = (128, 128)          # 上游 agent.AGENT_RESOLUTION
# 上游 agent.ACTION_TRANSFORMER_KWARGS(VPT 仓库格式常量,非注入先验)
TEACHER_CAM_KWARGS = dict(camera_maxval=10, camera_binsize=2, mu=10,
                          quantization_scheme=QuantizationScheme.MU_LAW)

_MAPPING = CameraHierarchicalMapping(n_camera_bins=11)
N_BUTTON_COMBOS = len(_MAPPING.BUTTONS_COMBINATIONS)              # 8641
N_TEACHER_CAM = len(_MAPPING.camera_combinations)                 # 121 = 11×11
TEACHER_CAM_NULL = _MAPPING.camera_null_bin                       # 5

# 键位置换:同名双射,机械生成(V2_KEYS 名字与 Buttons.ALL 完全一致)
assert sorted(V2_KEYS) == sorted(Buttons.ALL), "教师/学生键名集合必须一致"
TEACHER_KEY_TO_V2 = [Buttons.ALL.index(k) for k in V2_KEYS]

# 联合 buttons → 20 键因子化的精确边缘化矩阵(上游预计算,[8641,20] 0/1)
_M_FACTORED = torch.from_numpy(_MAPPING.BUTTON_IDX_TO_FACTORED.astype(np.float32))
# camera 元动作开(combo 含 "camera")指示向量 [8641]
_CAM_ON = torch.from_numpy((~_MAPPING.BUTTON_IDX_TO_CAMERA_META_OFF).astype(np.float32))


def teacher_bin_pushforward() -> torch.Tensor:
    """教师相机 bin → 我们相机 bin 的下推矩阵。

    Returns
    -------
    P : torch.Tensor [11, CAM_BINS] fp32,P[t,o]=1 当教师 bin t 的中心角度落入我们
        bin o。教师 ±10° 全程被我们 ±18° 覆盖 ⇒ 每行恰一个 1,概率质量守恒。
    """
    q = CameraQuantizer(**TEACHER_CAM_KWARGS)
    deg = q.undiscretize(np.arange(11))                 # 教师 bin 中心角度
    ours = deg_to_bins(deg)                             # 我们的契约编码
    p = torch.zeros(11, CAM_BINS)
    p[torch.arange(11), torch.from_numpy(ours)] = 1.0
    return p


_PUSH = teacher_bin_pushforward()


def teacher_to_v2(pi_logits: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """教师层级动作分布 → V2 契约分布(精确边缘化,无手写规则)。

    Parameters
    ----------
    pi_logits : {"buttons": [..., 8641], "camera": [..., 121]} fp32 log-prob
        (CategoricalActionHead 输出,温度已内含)

    Returns
    -------
    p_keys : [..., 20] fp32 — V2 键序的按键边缘概率
    cam_t : [..., 2, 11] fp32 — 教师 bin 上的相机边缘分布,轴序 (dx=yaw, dy=pitch);
        camera 元动作关的质量并入中心 bin。用 remap_cam 转到我们的 bin。
    p_cam_on : [...] fp32 — camera 元动作开的概率(如实记录用)
    """
    pb = pi_logits["buttons"].float().exp()                       # [...,8641]
    p_keys = (pb @ _M_FACTORED.to(pb.device))[..., TEACHER_KEY_TO_V2]
    p_cam_on = pb @ _CAM_ON.to(pb.device)                         # [...]
    pc = pi_logits["camera"].float().exp()
    pc = pc.reshape(*pc.shape[:-1], 11, 11)                       # [pitch bin, yaw bin]
    pitch, yaw = pc.sum(-1), pc.sum(-2)                           # 精确边缘化
    on = p_cam_on.unsqueeze(-1)
    pitch, yaw = on * pitch, on * yaw                             # meta off → 中心 bin
    pitch[..., TEACHER_CAM_NULL] += 1.0 - p_cam_on
    yaw[..., TEACHER_CAM_NULL] += 1.0 - p_cam_on
    cam_t = torch.stack([yaw, pitch], dim=-2)                     # 我们的轴序 (dx, dy)
    return p_keys, cam_t, p_cam_on


def remap_cam(cam_t: torch.Tensor) -> torch.Tensor:
    """教师 bin 相机分布 → 我们 bin 分布(下推,质量守恒)。[..., 2, 11] → [..., 2, CAM_BINS]。"""
    return cam_t @ _PUSH.to(cam_t.device)


class VPTTeacher:
    """VPT 教师策略(确定性:eval 模式,前向输出分布,不采样)。

    教师是学生观测的函数:输入只有像素帧(与承包商录像同源),无特权信息。
    """

    def __init__(self, model_path: str, weights_path: str, device: str = "cpu"):
        pkl = pickle.load(open(model_path, "rb"))
        policy_kwargs = pkl["model"]["args"]["net"]["args"]
        pi_head_kwargs = pkl["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        self.temperature = pi_head_kwargs["temperature"]
        from gym3.types import DictType
        action_space = DictType(**_MAPPING.get_action_space_update())
        self.policy = MinecraftAgentPolicy(policy_kwargs=policy_kwargs,
                                           pi_head_kwargs=pi_head_kwargs,
                                           action_space=action_space).to(device)
        sd = torch.load(weights_path, map_location=device)
        self.policy.load_state_dict(sd, strict=False)             # 上游同口径(多余 aux 头容忍)
        self.policy.eval()
        self.device = device

    def initial_state(self, batch: int = 1):
        s = self.policy.initial_state(batch)
        return None if s is None else tree_map(lambda x: x.to(self.device), s)

    @torch.no_grad()
    def forward_frames(self, frames_u8: torch.Tensor, state, first0: bool,
                       autocast: bool = False):
        """一段时间块前向(状态延续)。

        Parameters
        ----------
        frames_u8 : [T, 128, 128, 3] uint8 RGB(教师口径,HWC)
        state : transformer 记忆状态(None ⇒ initial_state(1))
        first0 : 本块第 0 帧是否 episode 首帧(重置记忆)

        Returns
        -------
        pi_logits : {"buttons": [T,8641], "camera": [T,121]} fp32 log-prob(CPU)
        state : 更新后的状态
        """
        t_n = frames_u8.shape[0]
        if state is None:
            state = self.initial_state(1)
        ob = {"img": frames_u8.unsqueeze(0).to(self.device)}      # [1,T,H,W,C]
        first = torch.zeros(1, t_n, dtype=torch.bool, device=self.device)
        first[0, 0] = bool(first0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            (pd, _v, _), state = self.policy(ob, first, state)
        # 头输出 [B=1, T, 1, n](TensorType shape=(1,) 的冗余维)→ [T, n]
        return {k: v[0, :, 0].float().cpu() for k, v in pd.items()}, state


def read_teacher_frames(mp4: str) -> np.ndarray:
    """整段 mp4 → 教师输入帧 [T,128,128,3] uint8 RGB(上游预处理口径)。"""
    cap = cv2.VideoCapture(mp4)
    out = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, TEACHER_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else np.zeros((0, 128, 128, 3), np.uint8)


def load_inputs(mp4: str, jsonl: str) -> tuple[np.ndarray, np.ndarray]:
    """单段教师输入:(frames [T,128,128,3] u8, gui [T] bool),T=帧/动作行数取小。"""
    frames = read_teacher_frames(mp4)
    gui = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            gui.append(bool(json.loads(line).get("gui")))
    t_n = min(len(frames), len(gui))
    if t_n == 0:
        raise RuntimeError(f"{mp4}: 空段")
    return frames[:t_n], np.array(gui[:t_n], dtype=bool)


def label_segment(teacher: VPTTeacher, frames: np.ndarray, gui: np.ndarray,
                  chunk: int = 128, autocast: bool = False) -> dict:
    """单段打标:教师顺序前向全段(记忆跨块延续),返回 npz 载荷。

    Returns
    -------
    dict:keys [T,20] f16 概率(V2 键序)、cam [T,2,11] f16 教师 bin 边缘分布
    (轴序 dx,dy;训练侧用 remap_cam 转我们 bin)、cam_on [T] f16、
    top_combo [T] u16(教师最可能 buttons 联合类,审计用)、gui [T] bool。
    """
    t_n = len(frames)
    frames = torch.from_numpy(frames)
    keys, cams, ons, tops = [], [], [], []
    state = None
    for i0 in range(0, t_n, chunk):
        pd, state = teacher.forward_frames(frames[i0:i0 + chunk], state,
                                           first0=(i0 == 0), autocast=autocast)
        pk, ct, on = teacher_to_v2(pd)
        keys.append(pk.half()); cams.append(ct.half()); ons.append(on.half())
        tops.append(pd["buttons"].argmax(-1).to(torch.int32))
    return dict(keys=torch.cat(keys).numpy(),
                cam=torch.cat(cams).numpy(),
                cam_on=torch.cat(ons).numpy(),
                top_combo=torch.cat(tops).numpy().astype(np.uint16),
                gui=gui)


def main() -> None:
    ap = argparse.ArgumentParser(description="VPT 教师离线打标(逐段 npz + manifest)")
    ap.add_argument("--pool", nargs="+", default=["runs/data/vpt_early", "runs/data/vpt_holdout"])
    ap.add_argument("--out", default="runs/data/vpt_labels")
    ap.add_argument("--model", default="runs/data/models/vpt_teacher/2x.model")
    ap.add_argument("--weights", default="runs/data/models/vpt_teacher/rl-from-foundation-2x.weights")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--decoders", type=int, default=2,
                    help="cv2 解码预取线程数(GPU 前向若快于解码,加大到不再是瓶颈)")
    ap.add_argument("--limit", type=int, default=0, help=">0 只标前 N 段(冒烟)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        with manifest.open() as f:
            done = {json.loads(l)["seg"] for l in f if l.strip()}
    wsha = hashlib.sha256(open(args.weights, "rb").read(1 << 24)).hexdigest()[:12]

    teacher = VPTTeacher(args.model, args.weights, device=args.device)
    n_par = sum(p.numel() for p in teacher.policy.parameters())
    print(f"教师 {Path(args.weights).name} ({n_par/1e6:.0f}M, T={teacher.temperature}) "
          f"@ {args.device}", flush=True)

    pairs = []
    for pool in args.pool:
        for mp4 in sorted(Path(pool).glob("*.mp4")):
            jl = mp4.with_suffix(".jsonl")
            if jl.exists() and mp4.stem not in done:
                pairs.append((mp4, jl))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"待打标 {len(pairs)} 段(已完成 {len(done)})", flush=True)

    # 解码预取:下一段在后台线程解码,GPU 前向与 cv2 解码流水线化(墙钟≈max 而非和)
    import queue as _q
    import threading as _th
    n_dec = max(1, args.decoders)
    pre: _q.Queue = _q.Queue(maxsize=n_dec + 1)

    def _producer(shard: int):
        for mp4, jl in pairs[shard::n_dec]:
            try:
                pre.put((mp4, load_inputs(str(mp4), str(jl))))
            except (RuntimeError, OSError, ValueError) as e:   # 滚动池半写段:跳过
                pre.put((mp4, e))
        pre.put(None)

    for k in range(n_dec):
        _th.Thread(target=_producer, args=(k,), daemon=True).start()
    # bf16 autocast 与上游 xf.py 的 dtype 断言冲突(transformer 记忆状态 fp32,
    # Q/K dtype 不齐);教师前向保持 fp32(上游部署口径,不动 vendored 代码)
    use_ac = False
    i, ended = -1, 0
    while ended < n_dec:
        item = pre.get()
        if item is None:
            ended += 1
            continue
        i += 1
        mp4, loaded = item
        t0 = time.time()
        if isinstance(loaded, Exception):
            print(f"[{i}] {mp4.stem} 失败:{loaded}", flush=True)
            continue
        try:
            payload = label_segment(teacher, *loaded, chunk=args.chunk, autocast=use_ac)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"[{i}] {mp4.stem} 失败:{e}", flush=True)
            continue
        np.savez_compressed(out / f"{mp4.stem}.npz", **payload)
        t_n = len(payload["gui"])
        rec = dict(seg=mp4.stem, n=t_n, gui_frac=round(float(payload["gui"].mean()), 4),
                   cam_on_mean=round(float(payload["cam_on"].astype(np.float32).mean()), 4),
                   attack_p_mean=round(float(payload["keys"][:, V2_KEYS.index("attack")]
                                             .astype(np.float32).mean()), 4),
                   teacher=Path(args.weights).name, weights_sha12=wsha,
                   temperature=teacher.temperature, translate_v=1,
                   fps_label=round(t_n / max(time.time() - t0, 1e-3), 1))
        with manifest.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 10 == 0:
            print(f"[{i}/{len(pairs)}] {mp4.stem} n={t_n} "
                  f"{rec['fps_label']} fps", flush=True)
    print("打标完成", flush=True)


if __name__ == "__main__":
    main()
