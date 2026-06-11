"""MinecraftWorldModel 自监督训练(Δz-JEPA 版):在离线 VPT 视频(画面+动作)上学世界动力学。

目标:把"看着 Minecraft 录像 + 录像里的真实动作"变成自监督信号,让世界模型学到
"动作在画面里有什么效果",为后续 prompt-tuning 迁移打底座。

  L_total = L_pred + α·L_inv + β·L_sigreg

  L_pred (Δz 预测,归一化 MSE): 从 (z_obs(t), a_t, h, t) 预测潜表征**增量**
    Δz = sg[enc_ema(img_{t+1}) − enc_ema(img_t)],除以 E[Δz²](detach)归一。
    损失值直接可读:1.0 = persistence(预测 0)基线,<1 = 胜过复读。
    旧版预测绝对 z_{t+1} 时静态内容占目标能量 ~99.8%,动力学信号被淹没。
  L_inv  (逆动力学): 从 (z_tg(t+1) − z_obs(t)) ⊙ c 反推真实动作 a_t。
    鼠标 = mu-law 分箱分类(CE,堵死"恒 0"平凡解),键盘 = 20 键 BCE。
    z_obs 端带梯度 ⇒ 这是视觉编码器"必须让 Δ 编码动作"的唯一直接压力。
  L_sigreg (防坍缩): sliced 高斯正则,施加在**在线感知编码 z_obs** 上(坍缩
    发生的位置;旧版施加在 Z_out 上,Transformer 可放大时间 PE/动作 token 的
    方差来满足检验,视觉内容照样坍缩)。

目标编码器 = 在线权重的 EMA 副本(每个优化步后 model.ema_update()),平稳靶。
时间降采样 --frame_skip(默认 4,20fps→5Hz):单步动作效应放大 ~4×,鼠标 dx/dy
在跳过窗口内求和、键盘取 max(见 VPTStreamDataset)。

数据:utils.vpt_dataset.VPTStreamDataset —— 流式加载(不全量预载),每个 worker 维护
<=cache_size 个已解码序列的滚动缓存,多 worker 并行喂数据,每步随机取 batch 个序列。
吞吐优化:截断 BPTT、AMP 混合精度、损失张量累积(避免逐步 .item() 同步)。

先准备数据(Colab 见 colab_demo.ipynb 的转换;本地合成见 download_sample_data.py),然后:
    python train/train_minecraft.py --data_dir runs/vpt_sample --epochs 50 --device cuda
"""
import argparse
import itertools
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from net.minecraft_world_model import MinecraftWorldModel
from utils.vpt_dataset import VPTStreamDataset
from utils.vpt_action import CAMERA_SCALE, CAMERA_BINS, camera_to_bin
from utils.minecraft_viz import visualize_minecraft
from blocks.primitives import SIGReg

EPS = 1e-4          # I1
ACT_DIM = 22        # 2 鼠标 + 20 键盘(见 VPT_KEYS)
N_MOUSE = 2


# =====================================================================
# 损失函数
# =====================================================================

def dz_pred_loss(mu, dz_tg, eps=1e-8):
    """归一化 Δz 预测损失:MSE(μ, Δz) / E[Δz²](分母 detach)。

    归一化的两个作用:(1) 损失值直接可读——它就是 (model/persistence)² 比值,
    1.0 = 复读基线;(2) 对编码尺度不变——编码器把 Δz 幅度整体压扁不能降低本
    损失,堵死"通过 EMA 把变化抹平"的间接捷径(目标本身 no_grad,但在线权重
    的坍缩动作会经 EMA 低通渗入目标;尺度不变性让这条路无利可图)。
    """
    denom = dz_tg.square().mean().detach().clamp(min=eps)
    return (mu - dz_tg).square().mean() / denom


def minecraft_inv_dyn_loss(delta_z, c, true_action, inv_dyn_head):
    """逆动力学:从 (z_tg(t+1) − z_obs(t)) ⊙ c 反推混合动作。

    鼠标 = mu-law 分箱 CE(分箱见 utils.vpt_action;MSE 下"恒预测 0"是平凡解,
    分类下基率解的 CE = 边缘熵,真实信号可压过)。键盘 = 20 键 BCE。
    delta_z 的 z_obs 端带梯度:本损失是编码器"让 Δ 编码动作"的唯一直接压力,
    也是可控闸 c 的唯一梯度来源(动作可解释 → c↑;噪声槽稀释池化特征 → c↓)。
    """
    mouse_logits, kb_prob = inv_dyn_head(delta_z * c)
    mouse_bin = camera_to_bin(true_action[:, :N_MOUSE])               # [B,2] long
    l_mouse = F.cross_entropy(mouse_logits.reshape(-1, mouse_logits.shape[-1]),
                              mouse_bin.reshape(-1))
    l_kb = F.binary_cross_entropy(kb_prob.clamp(EPS, 1 - EPS),
                                  true_action[:, N_MOUSE:])
    mouse_acc = (mouse_logits.argmax(-1) == mouse_bin).float().mean()
    return l_kb + l_mouse, l_mouse.detach(), l_kb.detach(), mouse_acc.detach()


def _to_float_img(img):
    """uint8 [.,3,H,W] → float∈[0,1](归一化推迟到 GPU 上做,PCIe 传 uint8 省 4×)。"""
    return img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()


# =====================================================================
# GPU 利用率采样(优先 NVML,兜底 nvidia-smi)
# =====================================================================

_GPU_UTIL_FN = None


def _resolve_gpu_util():
    try:
        torch.cuda.utilization()                       # 需 nvidia-ml-py
        return torch.cuda.utilization
    except Exception:
        pass
    import shutil, subprocess
    smi = shutil.which("nvidia-smi")
    if smi:
        def _via_smi():
            try:
                out = subprocess.run(
                    [smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                return float(out.stdout.strip().splitlines()[0])
            except Exception:
                return None
        return _via_smi
    return lambda: None


def _gpu_util():
    """瞬时 GPU 利用率(%);CPU 或无法采样时返回 None。"""
    global _GPU_UTIL_FN
    if _GPU_UTIL_FN is None:
        _GPU_UTIL_FN = _resolve_gpu_util()
    return _GPU_UTIL_FN()


# =====================================================================
# 训练 / 评估
# =====================================================================

def roll_append(a_raw, action):
    """滚动追加当前动作到历史缓冲。a_raw:[B,J,ACT_DIM], action:[B,ACT_DIM]。"""
    return torch.cat([a_raw[:, 1:], action.unsqueeze(1)], dim=1)


def _run_sequence(model, sigreg, img, action, t_vec, device, k_bptt,
                  alpha_inv, beta_sigreg, amp_dev, use_amp, scaler, acc):
    """对一个 batch 的完整序列做截断 BPTT 前向+反向;损失以张量形式累加进 acc。

    时序 = teacher forcing:每步感知输入 z_obs(t) 都来自真实画面(批量编码),
    跨步记忆只走 h token(窗口内带梯度,窗口边界 detach)。开环推演(ẑ+μ 累积)
    只在可视化/推理里做,训练不做 scheduled sampling——先把一步动力学学对。

    吞吐:目标编码整条序列一次批量(EMA + no_grad);在线编码按 BPTT 窗口批量;
    只有 Transformer 递归本身逐步(固有串行)。
    """
    B, T = img.shape[0], img.shape[1]
    h = torch.zeros(B, 1, model.d, device=device)
    a_raw = torch.zeros(B, model.J, ACT_DIM, device=device)
    n_win = -(-(T - 1) // k_bptt)        # ceil:窗口数,用于 SIGReg 权重尺度归一
    last_c = None

    # JEPA 目标:整条序列一次批量过 EMA 编码器(平稳靶,no_grad)
    with torch.autocast(device_type=amp_dev, enabled=use_amp):
        z_tg = model.encode_target(
            img.reshape(B * T, *img.shape[2:])).view(B, T, model.N, model.d)
    z_tg = z_tg.float()

    for w0 in range(0, T - 1, k_bptt):
        w1 = min(w0 + k_bptt, T - 1)
        k = w1 - w0
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            # 在线感知:窗口 B·k 帧一次性编码(带梯度)
            z_obs = model.encode_obs(
                img[:, w0:w1].reshape(B * k, *img.shape[2:])
            ).view(B, k, model.N, model.d)

        accum = torch.zeros((), device=device)
        for i in range(k):
            t = w0 + i
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                a_raw = roll_append(a_raw, action[:, t])
                out = model(z_obs[:, i], h, a_raw, t_vec[:, t])

            # 损失在 fp32 计算(autocast 外:BCE 不可 autocast,CE/归一化 MSE 在 fp32 更稳)
            mu, c = out["mu"].float(), out["c"].float()
            dz = z_tg[:, t + 1] - z_tg[:, t]                    # Δz 目标(no_grad)
            l_pred = dz_pred_loss(mu, dz)
            l_inv, l_mouse, l_kb, m_acc = minecraft_inv_dyn_loss(
                z_tg[:, t + 1] - z_obs[:, i].float(), c, action[:, t], model.inv_dyn)

            accum = accum + (l_pred + alpha_inv * l_inv) / (T - 1)
            last_c = c
            h = out["h_next"]            # 窗口内保留梯度(截断 BPTT,跨步记忆只走 h)
            # 张量累加(不 .item(),epoch 末一次性取出 → 去掉逐步 GPU↔CPU 同步)
            acc["pred"] += l_pred.detach(); acc["inv"] += l_inv.detach()
            acc["mouse"] += l_mouse; acc["kb"] += l_kb; acc["mouse_acc"] += m_acc
            acc["pred_rms"] += (mu - dz).square().mean().sqrt().detach()
            acc["dz_rms"] += dz.square().mean().sqrt()
            acc["inner"] += 1

        # SIGReg:施加在**在线感知编码 z_obs** 上——防坍缩要钉在坍缩发生的张量上。
        # slot 为分组、(窗口时间 × batch) 为样本维(近似平稳假设,把样本数提到 k·B,
        # 小 batch 下 ECF 检验才有功效)。除以 n_win:总权重 = β·mean_w(sig),
        # 不随 rollout 长度/k_bptt 改变与 pred 项的相对比例。
        l_sig = sigreg(z_obs.float().permute(2, 1, 0, 3).reshape(model.N, k * B, model.d))
        win_loss = accum + (beta_sigreg / n_win) * l_sig
        scaler.scale(win_loss).backward()
        acc["loss"] += win_loss.detach()
        acc["sigreg"] += l_sig.detach(); acc["win"] += 1
        h = h.detach()                                           # 窗口边界截断
    return last_c


def train_epoch(model, sigreg, loader, opt, scaler, device, steps, k_bptt,
                alpha_inv, beta_sigreg, amp_dev, use_amp):
    model.train()
    acc = {k: torch.zeros((), device=device) for k in
           ["loss", "pred", "pred_rms", "dz_rms", "inv", "mouse", "mouse_acc", "kb", "sigreg"]}
    acc["inner"] = acc["win"] = 0
    gpu_samples, c_last, n_batches = [], None, 0

    for batch in itertools.islice(loader, steps):
        img = _to_float_img(batch["img"].to(device, non_blocking=True))
        action = batch["action"].to(device, non_blocking=True)
        t_vec = batch["t_vec"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        c_last = _run_sequence(model, sigreg, img, action, t_vec, device, k_bptt,
                               alpha_inv, beta_sigreg, amp_dev, use_amp, scaler, acc)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        model.ema_update()                  # 目标编码器跟踪在线权重(每优化步一次)
        n_batches += 1
        u = _gpu_util()
        if u is not None:
            gpu_samples.append(u)

    ic = max(acc["inner"], 1); wc = max(acc["win"], 1)
    cv = c_last.squeeze(-1).flatten()
    return {
        "loss": (acc["loss"] / wc).item(),
        "pred": (acc["pred"] / ic).item(),          # 1.0 = persistence 基线,<1 = 胜过复读
        "pred_rms": (acc["pred_rms"] / ic).item(),  # 真实预测误差(不可被 σ/归一化美化)
        "dz_rms": (acc["dz_rms"] / ic).item(),      # 目标本身的变化幅度(基线尺度)
        "inv": (acc["inv"] / ic).item(),
        "mouse": (acc["mouse"] / ic).item(), "mouse_acc": (acc["mouse_acc"] / ic).item(),
        "kb": (acc["kb"] / ic).item(),
        "sigreg": (acc["sigreg"] / wc).item(),
        "c_mean": cv.mean().item(), "c_std": cv.std().item(),
        "batches": n_batches,
        "gpu_util": (sum(gpu_samples) / len(gpu_samples)) if gpu_samples else None,
    }


@torch.no_grad()
def evaluate(model, loader, device, steps, amp_dev, use_amp):
    """逆动力学评估:能否从潜变化里反推出"做了什么操作"。

    键盘:平衡准确率 + **跳变**(onset/release recall)——VPT 数据里 w/attack 等键
    常整段按住,逐帧 balanced acc 会被"输出基率常数"的平凡解灌高;只有按下/松开
    瞬间的检出率才证明模型从 ΔZ 里读出了动作。
    鼠标:分箱准确率,分全帧 acc 与 **move_acc**(仅 GT 非中心 bin、即真动了
    鼠标的帧)——基率解(恒中心 bin)在 move_acc 上得 0,同 onset 逻辑。
    """
    model.eval()
    tp = fp = fn = tn = 0
    on_tp = on_n = off_tp = off_n = 0
    m_hit = m_n = mv_hit = mv_n = 0
    center = (CAMERA_BINS - 1) // 2
    for batch in itertools.islice(loader, steps):
        img = _to_float_img(batch["img"].to(device))
        action = batch["action"].to(device)
        t_vec = batch["t_vec"].to(device)
        B, T = img.shape[0], img.shape[1]
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            z_obs = model.encode_obs(
                img[:, :T - 1].reshape(B * (T - 1), *img.shape[2:])
            ).view(B, T - 1, model.N, model.d)
            z_tg = model.encode_target(
                img.reshape(B * T, *img.shape[2:])).view(B, T, model.N, model.d)
        z_obs, z_tg = z_obs.float(), z_tg.float()
        h = torch.zeros(B, 1, model.d, device=device)
        a_raw = torch.zeros(B, model.J, ACT_DIM, device=device)
        prev_true = None
        for t in range(T - 1):
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                a_raw = roll_append(a_raw, action[:, t])
                out = model(z_obs[:, t], h, a_raw, t_vec[:, t])
            mouse_logits, kb_prob = model.inv_dyn(
                (z_tg[:, t + 1] - z_obs[:, t]) * out["c"].float())
            kb_pred = (kb_prob > 0.5)
            kb_true = (action[:, t, N_MOUSE:] > 0.5)
            tp += (kb_pred & kb_true).sum().item();  fp += (kb_pred & ~kb_true).sum().item()
            fn += (~kb_pred & kb_true).sum().item();  tn += (~kb_pred & ~kb_true).sum().item()
            if prev_true is not None:
                onset = kb_true & ~prev_true; release = ~kb_true & prev_true
                on_tp += (kb_pred & onset).sum().item();    on_n += onset.sum().item()
                off_tp += (~kb_pred & release).sum().item(); off_n += release.sum().item()
            prev_true = kb_true
            mb_pred = mouse_logits.argmax(-1)
            mb_true = camera_to_bin(action[:, t, :N_MOUSE])
            hit = (mb_pred == mb_true)
            m_hit += hit.sum().item(); m_n += hit.numel()
            moved = (mb_true != center)
            mv_hit += (hit & moved).sum().item(); mv_n += moved.sum().item()
            h = out["h_next"]
    recall = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"kb_recall": recall, "kb_spec": spec, "kb_bal_acc": 0.5 * (recall + spec),
            "kb_onset_recall": on_tp / max(on_n, 1),
            "kb_release_recall": off_tp / max(off_n, 1),
            "kb_edges": on_n + off_n,
            "mouse_bin_acc": m_hit / max(m_n, 1),
            "mouse_move_acc": mv_hit / max(mv_n, 1),
            "mouse_moves": mv_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample", help="VPTDataset 数据目录(.mp4+.jsonl)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=50, help="每 epoch 迭代多少个 batch(流式)")
    ap.add_argument("--batch", type=int, default=16,
                    help="每步随机取几个序列(batch=8 时实测 L4 显存仅用 ~0.9GB/24GB、"
                         "利用率 ~16%%——显存余量极大,默认提到 16;SIGReg 检验功效同步受益)")
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame_skip", type=int, default=4,
                    help="时间降采样(每 skip 帧取 1 帧,动作在窗口内聚合)。20fps 下相邻"
                         "帧几乎相同,一步动力学信号被 persistence 淹没;默认 4 → 等效 5Hz")
    ap.add_argument("--img_size", type=int, default=128,
                    help="训练分辨率(数据集端 INTER_AREA 下采样;0=保持原始分辨率)")
    ap.add_argument("--camera_scale", type=float, default=None,
                    help="鼠标 dx/dy 归一化尺度,固定超参(None=用 vpt_action.CAMERA_SCALE 默认 10;"
                         "真 BASALT 转身 ±190 用 ~20)")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader 并行加载进程数;默认 None=自动取 CPU 核数-1(上限 8)。"
                         "窗口解码是 CPU 大头,旧默认 2 是 GPU 利用率 ~16%% 的主因")
    ap.add_argument("--cache_size", type=int, default=32, help="每个 worker 缓存的动作表文件数")
    ap.add_argument("--refresh_every", type=int, default=64, help="已废弃(窗口化解码无整段缓存),仅保留兼容")
    ap.add_argument("--viz_every", type=int, default=10,
                    help="每多少 epoch 输出一次可视化面板(0=关闭)")
    ap.add_argument("--viz_dir", default="runs/mc_viz")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--N", type=int, default=16, help="实体槽数")
    ap.add_argument("--K", type=int, default=5, help="动作查询数")
    ap.add_argument("--J", type=int, default=8, help="历史动作长度")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha_inv", type=float, default=1.0, help="逆动力学损失权重")
    ap.add_argument("--beta_sigreg", type=float, default=0.1, help="SIGReg 防坍缩权重(施加在 z_obs 上)")
    ap.add_argument("--ema_decay", type=float, default=0.99,
                    help="目标编码器 EMA 衰减(时间常数 ≈ 1/(1−τ) 个优化步;短跑程用 0.99,"
                         "长跑程可升 0.996+)")
    ap.add_argument("--k_bptt", type=int, default=4, help="截断 BPTT 窗口")
    ap.add_argument("--no_amp", action="store_true", help="关闭 AMP 混合精度(默认 cuda 上开启)")
    ap.add_argument("--wandb", action="store_true", help="开启 wandb 远程记录(key 从环境变量 WANDB_API_KEY 读)")
    ap.add_argument("--wandb_project", default="minecraft-world-model")
    ap.add_argument("--wandb_run", default=None, help="wandb run 名(默认自动生成)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_cuda = str(dev).startswith("cuda")
    amp_dev = "cuda" if is_cuda else "cpu"
    use_amp = is_cuda and not args.no_amp
    print(f"=== MINECRAFT WORLD MODEL (Δz-JEPA + InvDyn + SIGReg) | device={dev} | amp={use_amp} ===")

    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        except Exception as ex:
            print(f"[wandb] 初始化失败,关闭远程记录: {ex}")
            use_wandb = False

    if not os.path.isdir(args.data_dir) or not any(
            f.endswith(".mp4") for f in os.listdir(args.data_dir)):
        print(f"[!] 数据目录 '{args.data_dir}' 里没有 .mp4。先准备数据(download_sample_data.py / colab 转换)。")
        sys.exit(1)

    img_size = args.img_size if args.img_size > 0 else None
    cam_scale = args.camera_scale if args.camera_scale is not None else CAMERA_SCALE
    n_workers = args.workers if args.workers is not None \
        else max(2, min(8, (os.cpu_count() or 2) - 1))
    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                          cache_size=args.cache_size, refresh_every=args.refresh_every,
                          seed=args.seed, img_size=img_size, camera_scale=cam_scale,
                          frame_skip=args.frame_skip)
    # prefetch_factor=4:窗口解码耗时方差大(seek 距关键帧远近不一),更深的预取
    # 队列吸收抖动,避免 GPU 周期性空转等数据。
    loader = DataLoader(ds, batch_size=args.batch, num_workers=n_workers,
                        pin_memory=is_cuda,
                        persistent_workers=(n_workers > 0),
                        prefetch_factor=(4 if n_workers > 0 else None))

    viz_batch = None
    if args.viz_every > 0:
        # 固定一条可视化序列(独立 seed,跨 epoch 同一窗口 → 面板可前后对比)
        viz_ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                                  cache_size=4, seed=args.seed + 999, img_size=img_size,
                                  camera_scale=cam_scale, frame_skip=args.frame_skip)
        viz_batch = next(iter(DataLoader(viz_ds, batch_size=1)))
        os.makedirs(args.viz_dir, exist_ok=True)

    model = MinecraftWorldModel(d=args.d, N=args.N, K=args.K, J=args.J, act_dim=ACT_DIM,
                                n_cam_bins=CAMERA_BINS, ema_decay=args.ema_decay).to(dev)
    sigreg = SIGReg(knots=17, num_proj=512).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M (+EMA 目标副本不训练) | "
          f"batch={args.batch} workers={n_workers} steps/epoch={args.steps_per_epoch} "
          f"frame_skip={args.frame_skip}")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    util_hist = []

    for ep in range(args.epochs):
        r = train_epoch(model, sigreg, loader, opt, scaler, dev, args.steps_per_epoch,
                        args.k_bptt, args.alpha_inv, args.beta_sigreg, amp_dev, use_amp)
        if r.get("gpu_util") is not None:
            util_hist.append(r["gpu_util"])
        if use_wandb:
            _wb = {k: r[k] for k in ("loss", "pred", "pred_rms", "dz_rms", "inv",
                                     "mouse", "mouse_acc", "kb", "sigreg", "c_mean", "c_std")}
            if r.get("gpu_util") is not None:
                _wb["gpu_util"] = r["gpu_util"]
            wandb.log(_wb, step=ep)
        if ep % 5 == 0 or ep == args.epochs - 1:
            gpu = f" | gpu {r['gpu_util']:.0f}%" if r.get("gpu_util") is not None else ""
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | pred {r['pred']:.3f}×copy "
                  f"(rms {r['pred_rms']:.4f}/dz {r['dz_rms']:.4f}) | "
                  f"inv {r['inv']:.3f} (kb {r['kb']:.3f}/mouse {r['mouse']:.2f} "
                  f"acc {r['mouse_acc']:.2f}) | sig {r['sigreg']:.2f} | "
                  f"c mean={r['c_mean']:.3f} std={r['c_std']:.3f}{gpu}")
        if viz_batch is not None and (ep % args.viz_every == 0 or ep == args.epochs - 1):
            p = visualize_minecraft(model, viz_batch, dev,
                                    os.path.join(args.viz_dir, f"ep{ep:04d}.png"))
            if p:
                print(f"  [viz] {p}")
                if use_wandb:
                    wandb.log({"viz/panel": wandb.Image(p)}, step=ep)

    if util_hist:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[GPU] 训练平均利用率 {sum(util_hist) / len(util_hist):.0f}% "
              f"({len(util_hist)} epoch) | 峰值显存 {peak:.2f} GB")

    print("\n--- 最终评估:逆动力学(从画面变化反推操作)---")
    e = evaluate(model, loader, dev, max(args.steps_per_epoch, 20), amp_dev, use_amp)
    print(f"键盘 recall {e['kb_recall']:.3f} | spec {e['kb_spec']:.3f} | "
          f"平衡准确率 {e['kb_bal_acc']:.3f}(随机基线≈0.5)")
    print(f"跳变检出 onset {e['kb_onset_recall']:.3f} | release {e['kb_release_recall']:.3f} "
          f"({e['kb_edges']} 次跳变;常按键灌不高这两项,是更硬的证据)")
    print(f"鼠标 bin acc {e['mouse_bin_acc']:.3f} | move acc {e['mouse_move_acc']:.3f} "
          f"({e['mouse_moves']} 个运动帧;恒中心 bin 的基率解 move acc = 0)")
    ok = e["kb_bal_acc"] > 0.6 or e["mouse_move_acc"] > 0.2
    print(f"=> {'✅ 世界模型从画面里读出了动作信息' if ok else '⚠ 动作信息尚不显著(欠训练/调 α/数据太少)'}")

    if use_wandb:
        wandb.log({"eval/kb_recall": e["kb_recall"], "eval/kb_spec": e["kb_spec"],
                   "eval/kb_bal_acc": e["kb_bal_acc"],
                   "eval/kb_onset_recall": e["kb_onset_recall"],
                   "eval/kb_release_recall": e["kb_release_recall"],
                   "eval/mouse_bin_acc": e["mouse_bin_acc"],
                   "eval/mouse_move_acc": e["mouse_move_acc"]})
        wandb.summary["kb_bal_acc"] = e["kb_bal_acc"]
        wandb.summary["mouse_move_acc"] = e["mouse_move_acc"]
        wandb.finish()


if __name__ == "__main__":
    main()
