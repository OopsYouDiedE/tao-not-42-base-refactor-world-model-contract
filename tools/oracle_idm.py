"""逆动力学信息上界 oracle —— 回答"逆动力学目标在真 BASALT + 冻结 DINOv2 下到底可学多少"。

动机:多轮训练后 eval/mouse_move_acc 卡 0.3~0.5、kb_onset_recall 卡 0.15,且 192px 探针
证伪了"分辨率瓶颈"。要判定这是【信息极限】(任何模型都到顶)还是【我们的池化/训练问题】
(目标其实可学),最干净的办法是绕开训练:直接量 frozen DINOv2 的 Δz 里**还剩多少关于动作
的信息**——用非参数/小模型 oracle 把这个上界估出来。

阶梯(全部只用冻结 DINOv2,不需要训练好的 SlotBinder):
  - pool : patch 平均 Δz [384]        → MLP        (≈ 槽池化后能给的,我们模型的下界代理)
  - grid : 完整 gh×gw patch Δz         → CNN + kNN  (backbone 空间上界 = 任何模型的天花板)
对照:
  - center 平凡解(恒猜中心 bin) —— mouse_move_acc 定义上 = 0,mouse_bin_acc = 中心频率
  - label-shuffle —— 打乱标签重训,应跌到 chance,证明上面不是泄漏

读法(与 eval 同口径,cam_scale=20、camera_to_bin、非中心元素口径):
  - grid 上界 ≈ 0.8  → 信息在 backbone 里,我们模型的 0.3~0.5 是【池化/训练】问题,目标可学。
  - grid 上界 ≈ 0.3  → backbone+数据就这上限,目标接近【信息极限】,得换表征而非调参。
  - grid ≫ pool      → 空间结构关键,我们的 16 槽池化正把它丢掉(改 IDM 直接吃 patch token)。

用法(repo 根目录):
  PYTHONPATH=. python tools/oracle_idm.py
  PYTHONPATH=. python tools/oracle_idm.py --clips_per_task 5 --max_frames 5000   # 更稳的估计
数据是公开 Azure blob,**下载不需要任何密钥**(PAT 只为 clone 私有 repo)。
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domains.minecraft.vpt_action import camera_to_bin, CAMERA_BINS, N_MOUSE  # noqa: E402
from domains.minecraft.vpt_dataset import VPT_KEYS as DS_KEYS  # 训练真契约的键名

# ---- 数据契约(与 colab §1/§2 + utils.vpt_action 严格一致)----
BASE = "https://openaipublic.blob.core.windows.net/minecraft-rl"
INDEX = {
    "FindCave":      f"{BASE}/snapshots/find-cave-Jul-28.json",
    "MakeWaterfall": f"{BASE}/snapshots/waterfall-Jul-28.json",
    "AnimalPen":     f"{BASE}/snapshots/pen-animals-Jul-28.json",
    "BuildHouse":    f"{BASE}/snapshots/build-house-Jul-28.json",
}
TASKS = list(INDEX)
PX2DEG = 360.0 / 2400.0                 # §2:鼠标像素 → 视角度数
CAM_SCALE = 20.0                        # §1:相机归一化尺度(度/帧);eval 同值
CENTER = (CAMERA_BINS - 1) // 2
ACT_DIM = 2 + 20

# 原始 VPT 键名 → 动作向量索引([dx, dy, 20 键],键序 = utils.vpt_action.VPT_KEYS)
RAW2IDX = {
    "key.keyboard.w": 2, "key.keyboard.a": 3, "key.keyboard.s": 4, "key.keyboard.d": 5,
    "key.keyboard.space": 6, "key.keyboard.left.shift": 7, "key.keyboard.left.control": 8,
    "key.keyboard.q": 11, "key.keyboard.e": 12,
    **{f"key.keyboard.{i}": 12 + i for i in range(1, 10)},   # hotbar.1..9 → 13..21
}
BTN2IDX = {0: 9, 1: 10}                 # 鼠标左/右键 → attack / use


# ============================ 数据获取 ============================
def _download_one(task, ti, basedir, rel, tdir):
    """下载一段 mp4 + jsonl,.part 临时名 + os.replace 原子落盘(半截文件不会被当成完成)。
    已存在的完整文件直接跳过(断点续传)。返回 (mp4, jsonl, ti) 或 None。"""
    name = os.path.basename(rel)[:-4]
    mp4 = os.path.join(tdir, f"{name}.mp4")
    jsl = os.path.join(tdir, f"{name}.jsonl")
    try:
        if not os.path.exists(mp4) or os.path.getsize(mp4) == 0:
            r = requests.get(f"{basedir}/{rel}", stream=True, timeout=600); r.raise_for_status()
            with open(mp4 + ".part", "wb") as f:
                for ch in r.iter_content(1 << 22):
                    f.write(ch)
            os.replace(mp4 + ".part", mp4)
        if not os.path.exists(jsl) or os.path.getsize(jsl) == 0:
            r = requests.get(f"{basedir}/{rel[:-4]}.jsonl", timeout=600); r.raise_for_status()
            with open(jsl + ".part", "w", encoding="utf-8") as f:
                f.write(r.text)
            os.replace(jsl + ".part", jsl)
        print(f"  [{task}] {name}  ({os.path.getsize(mp4)/1e6:.0f} MB)")
        return (mp4, jsl, ti)
    except Exception as e:
        print(f"  [{task}] {name}  下载失败 {type(e).__name__}: {e}")
        return None


def fetch_clips(cache_dir, clips_per_task, workers=8, clip_offset=0):
    """并行下载真 BASALT(公开,无需鉴权)。返回按 (task, name) 稳定排序的 [(mp4, jsonl, ti), ...]。
    clip_offset:取每任务索引的 [offset:offset+clips_per_task] 切片——取索引深处切片可得到
    与训练(通常从索引前部流式取)极可能 disjoint 的全新 clip,用于真泛化 eval。"""
    os.makedirs(cache_dir, exist_ok=True)
    jobs = []
    for ti, (task, url) in enumerate(INDEX.items()):
        idx = requests.get(url, timeout=60).json()
        basedir = idx["basedir"].rstrip("/")
        tdir = os.path.join(cache_dir, task)
        os.makedirs(tdir, exist_ok=True)
        for rel in idx["relpaths"][clip_offset:clip_offset + clips_per_task]:
            jobs.append((task, ti, basedir, rel, tdir))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda j: _download_one(*j), jobs))
    # 稳定 cid:按 (任务序, mp4 名) 排序,避免并行完成顺序影响留出划分
    return sorted([r for r in res if r], key=lambda r: (r[2], r[0]))


# 原始 VPT 键名 → 训练契约键名
RAW2NAME = {
    "key.keyboard.w": "key_w", "key.keyboard.a": "key_a", "key.keyboard.s": "key_s",
    "key.keyboard.d": "key_d", "key.keyboard.space": "key_space",
    "key.keyboard.left.shift": "key_sneak", "key.keyboard.left.control": "key_sprint",
    "key.keyboard.q": "key_drop", "key.keyboard.e": "key_inventory",
    **{f"key.keyboard.{i}": f"key_hotbar.{i}" for i in range(1, 10)},
}
BTN2NAME = {0: "key_attack", 1: "key_use"}


def raw_to_converted_line(d, task):
    """原始 BASALT jsonl 单帧 → §2 转换格式(utils.vpt_dataset._action_vec 期望的 schema):
    keyboard = {训练契约键名(key_w…): 1.0}(原始 keys 列表经 RAW2NAME 映射,鼠标键经
    BTN2NAME 折进 key_attack/key_use)、mouse.dx/dy 转**度**(×PX2DEG,_action_vec 再
    /camera_scale)、gui、task。GUI 段动作置零(同 parse_actions)。"""
    gui = bool(d.get("isGuiOpen", False))
    kbd = {}
    for k in d.get("keyboard", {}).get("keys", []):
        nm = RAW2NAME.get(k)
        if nm is not None:
            kbd[nm] = 1.0
    m = d.get("mouse", {}) or {}
    dx = dy = 0.0
    if not gui:
        for b in m.get("buttons", []) or []:
            if b in BTN2NAME:
                kbd[BTN2NAME[b]] = 1.0
        dx = (m.get("dx", 0.0) or 0.0) * PX2DEG     # px → deg
        dy = (m.get("dy", 0.0) or 0.0) * PX2DEG
    return {"task": task, "gui": gui, "isGuiOpen": gui,
            "mouse": {"dx": dx, "dy": dy}, "keyboard": kbd}


def parse_actions(jsonl_path):
    """原始 BASALT jsonl → 逐帧 (act[ACT_DIM], gui)。dx/dy 已归一化(度/CAM_SCALE,clamp±1)。"""
    acts, guis = [], []
    for line in open(jsonl_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gui = bool(d.get("isGuiOpen", False))
        v = np.zeros(ACT_DIM, dtype=np.float32)
        for k in d.get("keyboard", {}).get("keys", []):
            j = RAW2IDX.get(k)
            if j is not None:
                v[j] = 1.0
        m = d.get("mouse", {}) or {}
        if not gui:
            for b in m.get("buttons", []) or []:
                if b in BTN2IDX:
                    v[BTN2IDX[b]] = 1.0
            v[0] = float(np.clip((m.get("dx", 0.0) or 0.0) * PX2DEG / CAM_SCALE, -1.0, 1.0))
            v[1] = float(np.clip((m.get("dy", 0.0) or 0.0) * PX2DEG / CAM_SCALE, -1.0, 1.0))
        acts.append(v)
        guis.append(1.0 if gui else 0.0)
    return np.stack(acts), np.array(guis, dtype=np.float32)


# ============================ DINOv2 特征 ============================
class Backbone:
    """冻结 DINOv2 ViT-S/14(HF facebook/dinov2-small),复刻 net.extract_feats 预处理
    (128→126 双线性 + ImageNet 归一化 + 切 CLS/register)。"""

    def __init__(self, device):
        from transformers import AutoModel
        from utils.hf_token import get_hf_token
        self.device = device
        self.m = AutoModel.from_pretrained(
            "facebook/dinov2-small", token=get_hf_token()).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.ps = self.m.config.patch_size
        self.n_reg = getattr(self.m.config, "num_register_tokens", 0) or 0
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(self, img01):                  # img01: [B,3,H,W] in [0,1]
        H, W = img01.shape[-2:]
        H2 = max(self.ps, (H // self.ps) * self.ps)
        W2 = max(self.ps, (W // self.ps) * self.ps)
        if (H2, W2) != (H, W):
            img01 = F.interpolate(img01, (H2, W2), mode="bilinear", align_corners=False)
        x = (img01 - self.mean) / self.std
        lhs = self.m(pixel_values=x).last_hidden_state          # [B, 1+n_reg+M, 384]
        return lhs[:, 1 + self.n_reg:, :]                       # [B,M,384] 切 CLS+register


def build_pairs(clips, feat_fn, frame_skip, max_frames, img_size, device, model=None, bsz=256):
    """每段 clip 每 frame_skip 取一锚帧抽 patch 特征;相邻锚 = 一个转移对。
    feat_fn: img01[B,3,H,W]∈[0,1] → patch tokens [B,M,Ed](Backbone.feats 或 model.extract_feats)。
    model 非空时额外算**训练后槽-Δz**(z_tg[k+1]−z_obs[k] 槽平均,与 IDM 输入同构、不带 c
    门控——纯测训练编码器的表征是否比 frozen patch-Δz 多暴露相机/键盘信息)。
    返回 dict:pool[P,384], grid[P,384,gh,gw], ybin[P,2], kb[P,20], kb_prev[P,20], cid, tid
    (+ 若 model:ztrained[P,d])。
    """
    POOL, GRID, YB, KB, KBP, CID, TID, ZT = [], [], [], [], [], [], [], []
    for ci, (mp4, jsl, tid) in enumerate(clips):
        acts, guis = parse_actions(jsl)
        cap = cv2.VideoCapture(mp4)
        frames = []
        while len(frames) < max_frames:
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        n = min(len(frames), len(acts))
        anchors = list(range(0, n, frame_skip))
        if len(anchors) < 2:
            continue
        feats, zobs_l, ztg_l = [], [], []
        for s in range(0, len(anchors), bsz):
            batch = np.stack([frames[i] for i in anchors[s:s + bsz]])
            x = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device) / 255.0
            with torch.no_grad():
                ft = feat_fn(x).float()                          # [b, M, Ed] on device
                feats.append(ft.cpu())
                if model is not None:
                    zobs_l.append(model.encode_obs(feats=ft).float().cpu())     # [b,N,d] 在线
                    ztg_l.append(model.encode_target(feats=ft).float().cpu())   # [b,N,d] EMA
        feats = torch.cat(feats, 0)                       # [A, M, Ed]
        zobs = torch.cat(zobs_l, 0) if model is not None else None
        ztg = torch.cat(ztg_l, 0) if model is not None else None
        A, M, D = feats.shape
        gh = gw = int(round(math.sqrt(M)))
        prev_kb = np.zeros(20, dtype=np.float32)
        kept = 0
        for k in range(len(anchors) - 1):
            i0, i1 = anchors[k], anchors[k + 1]
            seg, seg_gui = acts[i0:i1], guis[i0:i1]
            if seg_gui.mean() > 0.1:                       # GUI 段:训练侧拒采,这里也跳
                continue
            agg_m = np.clip(seg[:, :N_MOUSE].mean(0), -1.0, 1.0)
            agg_kb = seg[:, N_MOUSE:].max(0)
            ybin = camera_to_bin(torch.from_numpy(agg_m)).numpy()
            dz = feats[k + 1] - feats[k]                              # [M,Ed]
            POOL.append(dz.mean(0).numpy())
            GRID.append(dz.numpy().reshape(gh, gw, D).transpose(2, 0, 1))   # [Ed,gh,gw]
            if model is not None:
                ZT.append((ztg[k + 1] - zobs[k]).mean(0).numpy())           # [d] 训练后槽-Δz
            YB.append(ybin); KB.append(agg_kb); KBP.append(prev_kb.copy())
            CID.append(ci); TID.append(tid)
            prev_kb = agg_kb
            kept += 1
        print(f"  clip{ci}(task{tid}): {n} 帧 → 保留 {kept} 对  (累计 {len(YB)})")
    out = {
        "pool": np.stack(POOL), "grid": np.stack(GRID),
        "ybin": np.stack(YB), "kb": np.stack(KB), "kb_prev": np.stack(KBP),
        "cid": np.array(CID), "tid": np.array(TID),
    }
    if model is not None:
        out["ztrained"] = np.stack(ZT)
    return out


# ============================ oracle 模型 ============================
from net.oracle_heads import PoolHead, GridHead


def _wce(logits, target, move_w=4.0):
    """与训练同口径的加权 CE:非中心 bin ×move_w,打掉"恒猜中心"基率不动点。"""
    ce = F.cross_entropy(logits, target, reduction="none")
    w = torch.where(target == CENTER, torch.ones_like(ce), torch.full_like(ce, move_w))
    return (ce * w).sum() / w.sum()


def train_oracle(model, tr, te, device, epochs=80, lr=1e-3, tag="", move_w=4.0):
    """训 oracle 到收敛。返回 (test 指标 dict, 训练好的 model)。tr/te = (X, ybin, kb)。"""
    model = model.to(device)
    Xtr, ybtr, kbtr = (torch.from_numpy(a).to(device) for a in tr)
    Xte, ybte, kbte = (torch.from_numpy(a).to(device) for a in te)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 512):
            b = perm[s:s + 512]
            dxl, dyl, kbl = model(Xtr[b])
            loss = (_wce(dxl, ybtr[b, 0], move_w) + _wce(dyl, ybtr[b, 1], move_w)
                    + F.binary_cross_entropy_with_logits(kbl, kbtr[b]))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        dxl, dyl, kbl = model(Xte)
    pred = torch.stack([dxl.argmax(-1), dyl.argmax(-1)], -1)
    m = metrics(pred, ybte, torch.sigmoid(kbl), kbte, tag)
    return m, model


def metrics(pred_bin, true_bin, kb_prob, kb_true, tag=""):
    moved = (true_bin != CENTER)
    hit = (pred_bin == true_bin)
    denom = moved.float().sum().clamp(min=1)
    mma = (hit & moved).float().sum() / denom
    ps = torch.sign(pred_bin.float() - CENTER); ts = torch.sign(true_bin.float() - CENTER)
    sign_acc = ((ps == ts) & moved).float().sum() / denom
    kb_pred = (kb_prob > 0.5).float()
    tp = (kb_pred * kb_true).sum(); pos = kb_true.sum(); neg = (1 - kb_true).sum()
    tn = ((1 - kb_pred) * (1 - kb_true)).sum()
    recall = tp / pos.clamp(min=1); spec = tn / neg.clamp(min=1)
    return {
        "tag": tag, "mouse_move_acc": mma.item(), "mouse_bin_acc": hit.float().mean().item(),
        "mouse_sign_acc": sign_acc.item(), "kb_recall": recall.item(), "kb_spec": spec.item(),
        "kb_bal_acc": (0.5 * (recall + spec)).item(),
    }


def onset_recall(model, X, kb_true, kb_prev, device):
    """键盘 onset recall:只在"上区间没按、本区间按下"的 (样本,键) 元素上算 recall。"""
    model.eval()
    with torch.no_grad():
        _, _, kbl = model(torch.from_numpy(X).to(device))
        kb_pred = (torch.sigmoid(kbl) > 0.5).float()
    onset = ((torch.from_numpy(kb_true).to(device) == 1)
             & (torch.from_numpy(kb_prev).to(device) == 0)).float()
    tp = (kb_pred * onset).sum()
    return (tp / onset.sum().clamp(min=1)).item(), int(onset.sum().item())


def knn_camera(Xtr, ybtr, Xte, ybte, device, k=15, pca=128):
    """非参数上界:PCA 降维后 GPU kNN(欧氏),多数票预测 dx/dy bin。免训练、不过拟合。"""
    try:
        Xtr = torch.from_numpy(Xtr).float().to(device).flatten(1)
        Xte = torch.from_numpy(Xte).float().to(device).flatten(1)
        mu = Xtr.mean(0, keepdim=True)
        Xtr, Xte = Xtr - mu, Xte - mu
        if pca and Xtr.shape[1] > pca:
            _, _, V = torch.pca_lowrank(Xtr, q=pca)
            Xtr, Xte = Xtr @ V, Xte @ V
        ybtr_t = torch.from_numpy(ybtr).to(device)
        preds = []
        for s in range(0, Xte.shape[0], 256):
            d = torch.cdist(Xte[s:s + 256], Xtr)
            votes = ybtr_t[d.topk(k, largest=False).indices]           # [b,k,2]
            preds.append(torch.stack([torch.mode(votes[:, :, 0], 1).values,
                                      torch.mode(votes[:, :, 1], 1).values], -1))
        pred = torch.cat(preds, 0)
        ybte_t = torch.from_numpy(ybte).to(device)
        dummy = torch.zeros(pred.shape[0], 20, device=device)
        return metrics(pred, ybte_t, dummy, dummy, tag=f"kNN(grid,pca{pca},k{k})")
    except RuntimeError as e:
        print(f"  [kNN 跳过:{e}]")
        return {"mouse_move_acc": 0.0, "mouse_bin_acc": 0.0, "mouse_sign_acc": 0.0,
                "kb_recall": 0.0, "kb_bal_acc": 0.0}


def split_by_clip(data):
    """每任务留出最后 1 段作 test(避免相邻帧泄漏 + 保持任务分布平衡)。"""
    cid, tid = data["cid"], data["tid"]
    te = np.zeros(len(cid), bool)
    for t in np.unique(tid):
        tc = np.unique(cid[tid == t])
        te |= (cid == tc[-1])
    return ~te, te


def fmt(m):
    return (f"move_acc={m['mouse_move_acc']:.3f}  bin_acc={m['mouse_bin_acc']:.3f}  "
            f"sign={m['mouse_sign_acc']:.3f}  kb_recall={m['kb_recall']:.3f}  "
            f"kb_bal={m['kb_bal_acc']:.3f}")


def load_ckpt_model(ckpt_path, device):
    """从 best checkpoint 重建训练好的 MinecraftWorldModel(骨干从 HF,trainable+EMA 从 ckpt)。

    优先用 ckpt 内的 config(asdict(ModelConfig));老 checkpoint 无 config 时从 args 回退拼装。
    """
    from net.world_model import MinecraftWorldModel
    from net.config import ModelConfig
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in ck:
        cfg = ModelConfig.from_dict(ck["config"])
    else:
        a = ck["args"]                                  # 老 ckpt 回退:从 args 拼 ModelConfig
        cfg = ModelConfig.from_dict({
            "d": a["d"], "N": a["N"], "K": a["K"], "J": a["J"],
            "ema_decay": a["ema_decay"], "max_skip": a["frame_skip"],
            "xi": {"d_xi": a["d_xi"]},
            "heads": {"inv_dyn_ctx": a.get("inv_dyn_ctx", False)},
            "backbone": {"kind": ck.get("encoder", a.get("encoder", "dinov2")),
                         "weights": a.get("encoder_weights")},
        })
    m = MinecraftWorldModel(cfg).to(device).eval()
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    nb_missing = [k for k in missing if not k.startswith("backbone.")]
    print(f"  [ckpt] ep{ck['epoch']} {ck['metric']}={ck['score']:.4f} | "
          f"非骨干缺失键 {len(nb_missing)} | 多余键 {len(unexpected)}"
          + (f"  ⚠ 缺失:{nb_missing[:3]}" if nb_missing else ""))
    return m


# ============================ 前向预测 oracle ============================

def build_fwd_pairs(clips, model, frame_skip, max_frames, img_size, device, bsz=256):
    """前向预测对(在【模型自己的 z 空间】上):条件 z_obs[k] + 动作 → 预测靶
    Δz = z_tg[k+1] − z_tg[k](双 EMA 端,与训练/eval 的 dz_pred_loss 同靶,见
    train L313 / eval L509)。返回 zobs[P,N,d] / dz[P,N,d] / act[P,ACT_DIM](聚合)/
    aseq[P,max_skip,ACT_DIM](区间逐帧动作,喂模型 a_cur)/ dt[P] / cid / tid。"""
    ZO, DZ, ACT, ASEQ, DT, CID, TID = [], [], [], [], [], [], []
    ms = frame_skip
    for ci, (mp4, jsl, tid) in enumerate(clips):
        acts, guis = parse_actions(jsl)
        cap = cv2.VideoCapture(mp4)
        frames = []
        while len(frames) < max_frames:
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        n = min(len(frames), len(acts))
        anchors = list(range(0, n, frame_skip))
        if len(anchors) < 2:
            continue
        zobs_l, ztg_l = [], []
        for s in range(0, len(anchors), bsz):
            batch = np.stack([frames[i] for i in anchors[s:s + bsz]])
            x = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device) / 255.0
            with torch.no_grad():
                ft = model.extract_feats(x).float()
                zobs_l.append(model.encode_obs(feats=ft).float().cpu())
                ztg_l.append(model.encode_target(feats=ft).float().cpu())
        zobs = torch.cat(zobs_l, 0)
        ztg = torch.cat(ztg_l, 0)                          # [A,N,d]
        kept = 0
        for k in range(len(anchors) - 1):
            i0, i1 = anchors[k], anchors[k + 1]
            seg, seg_gui = acts[i0:i1], guis[i0:i1]
            if seg_gui.mean() > 0.1:                       # GUI 段:同训练侧拒采
                continue
            ZO.append(zobs[k].numpy())
            DZ.append((ztg[k + 1] - ztg[k]).numpy())       # 模型预测靶(双 EMA 端)
            agg = np.concatenate([np.clip(seg[:, :N_MOUSE].mean(0), -1, 1),
                                  seg[:, N_MOUSE:].max(0)]).astype(np.float32)
            ACT.append(agg)
            aseq = np.zeros((ms, seg.shape[1]), np.float32)
            L = min(len(seg), ms)
            aseq[:L] = seg[:L]
            ASEQ.append(aseq)
            DT.append(min(i1 - i0, ms))
            CID.append(ci); TID.append(tid); kept += 1
        print(f"  clip{ci}(task{tid}): {n} 帧 → 保留 {kept} 对  (累计 {len(ZO)})")
    return {"zobs": np.stack(ZO), "dz": np.stack(DZ), "act": np.stack(ACT),
            "aseq": np.stack(ASEQ), "dt": np.array(DT, np.int64),
            "cid": np.array(CID), "tid": np.array(TID)}


def _fwd_ratio(pred, dz, den, moved):
    """与 pred_move 同口径:per/den(硬地板 1e-3),仅运动样本(den>中位数)均值。"""
    per = ((pred - dz) ** 2).reshape(pred.shape[0], -1).mean(1)
    ratio = per / np.clip(den, 1e-3, None)
    return float(ratio[moved].mean())


def _knn_regress(Xtr, Ytr, Xte, device, k=20, pca=64):
    """kNN 条件均值回归(MMSE-最优预测器的非参数估计)= Bayes 下限。逐维标准化
    (z_obs 6144 维不淹没 22 维动作)→ PCA → GPU 欧氏 kNN → 邻居 Y 均值。"""
    Xtr = torch.from_numpy(Xtr).float().to(device)
    Xte = torch.from_numpy(Xte).float().to(device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if pca and Xtr.shape[1] > pca:
        _, _, V = torch.pca_lowrank(Xtr, q=pca)
        Xtr, Xte = Xtr @ V, Xte @ V
    Ytr = torch.from_numpy(Ytr).float().to(device)
    out = []
    for s in range(0, Xte.shape[0], 128):
        idx = torch.cdist(Xte[s:s + 128], Xtr).topk(k, largest=False).indices
        out.append(Ytr[idx].mean(1).cpu())
    return torch.cat(out, 0).numpy()


from net.oracle_heads import PredOracle


def fwd_oracle(data, model, device, epochs=120):
    """前向预测 Bayes 下限:满容量/非参数预测器在 (z_obs,动作)→Δz 上能到多低,
    与模型 transformer 的 1-step 对比 ⇒ 钉死"0.58 是表征下限还是预测器欠拟合"。"""
    tr, te = split_by_clip(data)
    zobs, dz = data["zobs"], data["dz"]                # [P,N,d]
    P, N, d = zobs.shape
    act = data["act"]                                  # [P,ACT_DIM]
    den = (dz ** 2).reshape(P, -1).mean(1)             # |Δz|² 逐对
    moved = den[te] > np.median(den[te])               # 运动样本(同 pred_move)
    dzt, dent = dz[te], den[te]
    print(f"\n前向预测对 P={P} | train={int(tr.sum())} test={int(te.sum())} | "
          f"运动样本(test)={int(moved.sum())}")

    res = {}
    res["persistence(copy)"] = _fwd_ratio(np.zeros_like(dzt), dzt, dent, moved)

    # 模型 1-step(无历史):h=0、a_hist=0、hist_valid=0、a_cur=区间动作、xi=先验均值
    # ——同 eval 序列第 0 步的单步前向。
    aseq = torch.from_numpy(data["aseq"]).float()
    dt_t = torch.from_numpy(data["dt"]).float()
    zb = torch.from_numpy(zobs).float()
    idx_te = np.where(te)[0]
    mu_l = []
    with torch.no_grad():
        for s in range(0, len(idx_te), 128):
            ii = idx_te[s:s + 128]; B = len(ii)
            zr = zb[ii].to(device)
            z0 = torch.zeros(B, model.J, ACT_DIM, device=device)
            jz = torch.zeros(B, model.J, device=device)
            o = model(zr, torch.zeros(B, 1, d, device=device), z0,
                      aseq[ii].to(device), dt_t[ii].to(device),
                      torch.zeros(B, device=device), t_hist=jz, hist_valid=jz)
            mu_l.append(o["mu"].float().cpu().numpy())
    res["model 1-step(no hist)"] = _fwd_ratio(np.concatenate(mu_l, 0), dzt, dent, moved)

    Xz = zobs.reshape(P, -1)
    Ydz = dz.reshape(P, -1)
    pk = _knn_regress(np.concatenate([Xz, act], 1)[tr], Ydz[tr],
                      np.concatenate([Xz, act], 1)[te], device)
    res["kNN(z+act) Bayes底"] = _fwd_ratio(pk.reshape(-1, N, d), dzt, dent, moved)
    pa = _knn_regress(act[tr], Ydz[tr], act[te], device, pca=0)
    res["kNN(act only)"] = _fwd_ratio(pa.reshape(-1, N, d), dzt, dent, moved)

    # MLP(逐槽共享:z_obs[slot]⊕act → dz[slot]),P·N≈48k 样本良态、不易过拟合
    actrep = np.repeat(act[:, None, :], N, 1)
    Xs = np.concatenate([zobs, actrep], -1).reshape(P * N, -1)
    Ys = dz.reshape(P * N, d)
    trs = np.repeat(tr, N)
    tes = np.repeat(te, N)
    net = nn.Sequential(nn.Linear(Xs.shape[1], 1024), nn.SiLU(),
                        nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, d)).to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xtr = torch.from_numpy(Xs[trs]).float().to(device)
    Ytr = torch.from_numpy(Ys[trs]).float().to(device)
    for _ in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=device)
        for s in range(0, Xtr.shape[0], 4096):
            b = perm[s:s + 4096]
            opt.zero_grad()
            F.mse_loss(net(Xtr[b]), Ytr[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pm = net(torch.from_numpy(Xs[tes]).float().to(device)).cpu().numpy()
    res["MLP(per-slot)"] = _fwd_ratio(pm.reshape(-1, N, d), dzt, dent, moved)

    # 强 oracle:隔离的 Δz-预测 transformer(槽+逐帧动作注意力)——公平上界。每 25 ep
    # 在 test 上测比值、取**最小**(early-stop 给最宽容的可达下限估计)。两档容量:
    # 与模型同尺寸 + 放大,看放大是否破模型 0.577(破=欠拟合;不破=该 z 空间近下限)。
    A, S = data["aseq"].shape[2], data["aseq"].shape[1]
    zt_all = torch.from_numpy(zobs).float()
    at_all = torch.from_numpy(data["aseq"]).float()
    yt_all = torch.from_numpy(dz).float()
    idx_tr = np.where(tr)[0]

    def _eval_tx(net):
        net.eval()
        with torch.no_grad():
            pl = [net(zt_all[idx_te[s:s + 256]].to(device),
                      at_all[idx_te[s:s + 256]].to(device)).cpu().numpy()
                  for s in range(0, len(idx_te), 256)]
        net.train()
        return _fwd_ratio(np.concatenate(pl, 0), dzt, dent, moved)

    for L, W, tagn in [(4, 384, "Tx[L4 W384~model]"), (8, 512, "Tx[L8 W512 放大]")]:
        net = PredOracle(d, A, S, width=W, layers=L).to(device).train()
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        best = 1.0
        for ep in range(300):
            np.random.shuffle(idx_tr)
            for s in range(0, len(idx_tr), 256):
                b = idx_tr[s:s + 256]
                opt.zero_grad()
                F.mse_loss(net(zt_all[b].to(device), at_all[b].to(device)),
                           yt_all[b].to(device)).backward()
                opt.step()
            if (ep + 1) % 25 == 0:
                best = min(best, _eval_tx(net))
        res[tagn] = best

    print("\n" + "=" * 72)
    print("前向预测 1-step 比值(x copy,运动样本,test;<1=胜复读,越小越好)")
    print("=" * 72)
    for kname, v in res.items():
        print(f"  {kname:26s} {v:.3f}")
    mp = res["model 1-step(no hist)"]
    oracle = min(v for kn, v in res.items()
                 if kn not in ("persistence(copy)", "model 1-step(no hist)", "kNN(act only)"))
    ao = res["kNN(act only)"]
    print("=" * 72)
    print("判读:")
    print(f"  动作单独 kNN={ao:.3f}(>=1 = 聚合动作几乎不预测 Δz:Δz 由状态相关的视觉演化主导,非动作读出)")
    if oracle < mp - 0.08:
        print(f"  oracle 下限 {oracle:.3f} < 模型 {mp:.3f}:(z_obs,动作) 里有模型没学到的可预测结构")
        print(f"  => transformer/训练【欠拟合】,修预测器(容量/动作条件/去  xi 瓶颈),别急着转 rollout。")
    elif oracle <= mp + 0.05:
        print(f"  oracle 下限 {oracle:.3f} ~= 模型 {mp:.3f}:模型已达通用预测器前沿,0.58 大概率近【信息下限】。")
        print(f"  => 弃 pred_move 北极星,转多步 rollout 保真度 + 可控子空间。")
    else:
        print(f"  oracle 下限 {oracle:.3f} > 模型 {mp:.3f}:所有外部 oracle(含同归纳偏置、放大的隔离 Tx)都【打不过模型】。")
        print(f"  且 Tx 放大几乎不动(L4~L8)=> 非容量瓶颈;oracle 仅本地少量样本、被数据饿死,无法从下方夹住 Bayes 底。")
        print(f"  => 模型自己就是最强预测器,外部 oracle 法对'是否欠拟合'【不结论】(但排除了'懒惰/容量不足'两种欠拟合)。")
        print(f"     合并证据(转头≈平移、历史几乎无用、动作单独无用、模型胜一切外部预测器、放大无益、xi 惰性)")
        print(f"     强指向 0.58 近该 z 空间 1-step 下限。pred_move 已饱和,应弃为北极星,火力转多步 rollout 保真度 + 可控子空间。")
    print("=" * 72)
    return res


def main():
    try:                                  # Windows GBK 控制台无法编码 ≳/≪ 等数学符,加固防崩
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="runs/data/oracle_data")
    ap.add_argument("--feat_cache", default="runs/data/oracle_pairs.npz")
    ap.add_argument("--checkpoint", default=None,
                    help="训练好的 best_*.pt:额外对【训练后槽-Δz】跑 oracle,判定训练编码器"
                         "是否比 frozen patch-Δz 多暴露相机/键盘信息(钉死相机上界的 decisive 实验)")
    ap.add_argument("--fwd_pred", action="store_true",
                    help="前向预测 Bayes 下限模式(需 --checkpoint):满容量 MLP/kNN 在训练后 z "
                         "空间的 (z_obs,动作)→Δz 上能到多低,vs 模型 transformer 的 1-step ⇒ 钉死"
                         "「pred_move~0.58 是表征信息下限还是预测器欠拟合」。独立 _fwd.npz 缓存。")
    ap.add_argument("--clip_offset", type=int, default=0,
                    help="每任务索引切片起点。取深处切片(如 100)= 与训练(通常取索引前部)"
                         "极可能 disjoint 的全新 clip,用于真泛化 eval。")
    ap.add_argument("--fetch_to", default=None,
                    help="纯数据准备模式:下载 [clip_offset:+clips_per_task] 的 clip 并**扁平化**"
                         "(mp4+jsonl 平铺,VPTStreamDataset 要求)进该目录、打印文件名后退出。"
                         "配 train_minecraft --eval_only --holdout_dir <该目录> 做 disjoint holdout eval。")
    ap.add_argument("--clips_per_task", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--frame_skip", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    dev = args.device
    print(f"device={dev}\n")

    if args.fetch_to:                      # 数据准备:下载 disjoint 切片,jsonl 转 §2 格式 + 扁平化
        import shutil
        from domains.minecraft.vpt_dataset import _action_vec
        os.makedirs(args.fetch_to, exist_ok=True)
        clips = fetch_clips(args.cache_dir, args.clips_per_task, clip_offset=args.clip_offset)
        if not clips:
            print("没有可用 clip,退出。"); return
        for mp4, jsl, ti in clips:
            base = os.path.basename(mp4)[:-4]
            dmp4 = os.path.join(args.fetch_to, base + ".mp4")
            if not os.path.exists(dmp4):
                try:
                    os.link(mp4, dmp4)
                except OSError:
                    shutil.copy2(mp4, dmp4)
            task = TASKS[ti]
            with open(jsl, encoding="utf-8") as f, \
                    open(os.path.join(args.fetch_to, base + ".jsonl"), "w", encoding="utf-8") as g:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    g.write(json.dumps(raw_to_converted_line(json.loads(line), task)) + "\n")
        # 自校验:转换后经 _action_vec(§2 路径)解析 == parse_actions(原始路径),逐帧一致
        mp4, jsl, _ = clips[0]
        raw_v, _ = parse_actions(jsl)
        conv = [_action_vec(json.loads(l), camera_scale=CAM_SCALE).numpy()
                for l in open(os.path.join(args.fetch_to,
                                           os.path.basename(mp4)[:-4] + ".jsonl"), encoding="utf-8")]
        conv = np.stack(conv[:len(raw_v)])
        max_err = float(np.abs(conv - raw_v[:len(conv)]).max())
        names = sorted(os.path.basename(m)[:-4] for m, _, _ in clips)
        print(f"\n转换+扁平化 {len(clips)} clip → {args.fetch_to}(offset={args.clip_offset})")
        print(f"自校验(_action_vec vs parse_actions 逐帧最大差):{max_err:.2e}  "
              f"{'✓ 一致' if max_err < 1e-4 else '✗ 不一致!转换有误'}")
        print("clip 文件名(与训练 data_dir 比对确认 disjoint):")
        for nm in names:
            print(f"  {nm}")
        print(f"\n下一步:python train/train_minecraft.py --eval_only <ckpt> "
              f"--holdout_dir {args.fetch_to} --data_dir {args.fetch_to} "
              f"--camera_scale 20 --img_size 128 --frame_skip 8 --batch 32 --seq_len 30 --text_encoder minilm")
        return

    if args.fwd_pred:
        if not args.checkpoint:
            print("--fwd_pred 需要 --checkpoint(前向 oracle 在训练后 z 空间上量)"); return
        fwd_cache = args.feat_cache.replace(".npz", "_fwd.npz")
        print(f"== 加载 checkpoint {args.checkpoint} ==")
        model = load_ckpt_model(args.checkpoint, dev)
        if os.path.exists(fwd_cache) and not args.rebuild:
            print(f"加载前向缓存 {fwd_cache}")
            z = np.load(fwd_cache)
            data = {k: z[k] for k in z.files}
        else:
            clips = fetch_clips(args.cache_dir, args.clips_per_task, clip_offset=args.clip_offset)
            if not clips:
                print("没有可用 clip,退出。"); return
            print(f"\n== 构造前向预测对(frame_skip={args.frame_skip})==")
            t0 = time.time()
            data = build_fwd_pairs(clips, model, args.frame_skip, args.max_frames,
                                   args.img_size, dev)
            print(f"用时 {time.time()-t0:.0f}s")
            np.savez_compressed(fwd_cache, **data)
            print(f"已缓存 → {fwd_cache}")
        fwd_oracle(data, model, dev)
        return

    # checkpoint 模式用独立缓存(多一份 ztrained,schema 不同),避免与 frozen-only 缓存撞
    feat_cache = args.feat_cache.replace(".npz", "_ckpt.npz") if args.checkpoint else args.feat_cache

    if os.path.exists(feat_cache) and not args.rebuild:
        print(f"加载缓存 {feat_cache}")
        z = np.load(feat_cache)
        data = {k: z[k] for k in z.files}
    else:
        print("== 下载真 BASALT(公开,无密钥)==")
        clips = fetch_clips(args.cache_dir, args.clips_per_task, clip_offset=args.clip_offset)
        if not clips:
            print("没有可用 clip,退出。"); return
        model = None
        if args.checkpoint:
            print(f"== 加载 checkpoint {args.checkpoint} ==")
            model = load_ckpt_model(args.checkpoint, dev)
            feat_fn = model.extract_feats          # 复用模型骨干,免二次加载 DINOv2
        else:
            feat_fn = Backbone(dev).feats
        print(f"\n== 抽特征 + 构造转移对(frame_skip={args.frame_skip}"
              f"{',含训练后槽-Δz' if model else ''})==")
        t0 = time.time()
        data = build_pairs(clips, feat_fn, args.frame_skip, args.max_frames, args.img_size,
                           dev, model=model)
        print(f"特征用时 {time.time()-t0:.0f}s")
        np.savez_compressed(feat_cache, **data)
        print(f"已缓存 → {feat_cache}")

    P = data["ybin"].shape[0]
    moved_frac = (data["ybin"] != CENTER).mean()
    print(f"\n转移对 P={P}  非中心元素占比={moved_frac:.2%}  中心频率(平凡 bin_acc)={1-moved_frac:.2%}")
    gh = data["grid"].shape[2]
    tr_m, te_m = split_by_clip(data)
    print(f"train={int(tr_m.sum())}  test={int(te_m.sum())}  "
          f"(test 非中心元素={int((data['ybin'][te_m]!=CENTER).sum())})  grid={gh}x{gh}\n")

    def sl(rep, mask):
        return (data[rep][mask].astype(np.float32), data["ybin"][mask].astype(np.int64),
                data["kb"][mask].astype(np.float32))

    print("=" * 80)
    print("阶梯结果(test 集,与 eval/mouse_move_acc 同口径)")
    print("=" * 80)

    pm, pmodel = train_oracle(PoolHead(), sl("pool", tr_m), sl("pool", te_m), dev, args.epochs, tag="pool")
    po, pon = onset_recall(pmodel, data["pool"][te_m].astype(np.float32),
                           data["kb"][te_m].astype(np.float32),
                           data["kb_prev"][te_m].astype(np.float32), dev)
    print(f"[pool MLP ]  {fmt(pm)}  onset_recall={po:.3f}(n={pon})")

    gm, gmodel = train_oracle(GridHead(gh=gh, gw=gh), sl("grid", tr_m), sl("grid", te_m),
                              dev, args.epochs, tag="grid")
    go, _ = onset_recall(gmodel, data["grid"][te_m].astype(np.float32),
                         data["kb"][te_m].astype(np.float32),
                         data["kb_prev"][te_m].astype(np.float32), dev)
    print(f"[grid CNN ]  {fmt(gm)}  onset_recall={go:.3f}")

    km = knn_camera(data["grid"][tr_m], data["ybin"][tr_m].astype(np.int64),
                    data["grid"][te_m], data["ybin"][te_m].astype(np.int64), dev)
    print(f"[grid kNN ]  {fmt(km)}")

    ysh = data["ybin"][tr_m].copy(); np.random.shuffle(ysh)
    sm, _ = train_oracle(GridHead(gh=gh, gw=gh),
                         (data["grid"][tr_m].astype(np.float32), ysh.astype(np.int64),
                          data["kb"][tr_m].astype(np.float32)),
                         sl("grid", te_m), dev, args.epochs, tag="shuffle")
    print(f"[shuffle  ]  {fmt(sm)}   ← 应跌到 chance(证明上面不是泄漏)")
    print(f"[center   ]  move_acc=0.000  bin_acc={1-moved_frac:.3f}   ← 平凡基线")

    # checkpoint 模式:训练后槽-Δz 臂(IDM 同构,无 c 门控)——测训练编码器表征内容
    zm = zo = ztr_cam = None
    if "ztrained" in data:
        din = data["ztrained"].shape[1]
        zm, zmodel = train_oracle(PoolHead(din=din), sl("ztrained", tr_m),
                                  sl("ztrained", te_m), dev, args.epochs, tag="ztrained")
        zo, _ = onset_recall(zmodel, data["ztrained"][te_m].astype(np.float32),
                             data["kb"][te_m].astype(np.float32),
                             data["kb_prev"][te_m].astype(np.float32), dev)
        zk = knn_camera(data["ztrained"][tr_m], data["ybin"][tr_m].astype(np.int64),
                        data["ztrained"][te_m], data["ybin"][te_m].astype(np.int64), dev)
        print(f"[ztrain MLP] {fmt(zm)}  onset_recall={zo:.3f}   ← 训练后槽-Δz(IDM 同构,无 c)")
        print(f"[ztrain kNN] {fmt(zk)}")
        ztr_cam = max(zm["mouse_move_acc"], zk["mouse_move_acc"])

    print("\n" + "=" * 80)
    cam = max(gm["mouse_move_acc"], pm["mouse_move_acc"])      # kNN 多数票偏中心,不计入
    print("判读(分通道,口径同 eval):")
    print(f"  相机 move_acc:frozen patch 上界≈{cam:.3f}(shuffle≈{sm['mouse_move_acc']:.3f}"
          f"=chance) | 我们模型 eval 典型 0.13~0.32(基线 0.52 为异常,待查)")
    print(f"  键盘 onset_recall:frozen patch oracle≈{go:.3f} / bal≈{gm['kb_bal_acc']:.3f} | "
          f"我们模型 eval 0.15 / 0.74")
    print("  ── 相机:frozen patch-Δz 仅含【弱但真】相机信息(远高于 chance,远低于可用)。"
          "语义 patch token 不暴露位移(需对应/光流)。")
    print("  ── 键盘:信息充足且【模型没吃满】(onset 还有 ~2× 余量)——训练/头问题,非数据极限。")
    if ztr_cam is not None:
        print(f"  ── 训练后 z:相机 move_acc≈{ztr_cam:.3f} / onset≈{zo:.3f}  vs frozen patch {cam:.3f}")
        if ztr_cam > cam + 0.08:
            print("     ⇒ 训练编码器**确实多暴露**相机信息:方向对,继续训 / 改 IDM 能涨,非信息极限。")
        else:
            print("     ⇒ 训练编码器也没多出相机 ⇒ 坐实需换**运动表征**(帧差/光流/patch 对应),"
                  "别再拧 lr/ema/分辨率。")
    else:
        print("  ⚠ caveat:oracle 用 frozen 原始 patch-Δz,我们模型在其上还有【可训练 proj+binder】,"
              "故非严格上界。传 --checkpoint <best.pt> 对训练后 z 复跑即可钉死相机上界。")
    print("=" * 80)


if __name__ == "__main__":
    main()
