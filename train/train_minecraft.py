"""MinecraftWorldModel 自监督训练:在离线 VPT 视频(画面+动作)上学世界动力学。

目标:把"看着 Minecraft 录像 + 录像里的真实动作"变成自监督信号,让世界模型学到
"动作在画面里有什么效果",为后续 prompt-tuning 迁移打底座。

  L_total = L_pred + α·L_inv + β·L_sigreg

  L_pred (JEPA 前向预测): 从 (Z_t, a_t, t) 预测下一帧潜表征 μ_{t+1},
    目标为 stop-grad 的干净视觉编码 encode(img_{t+1})。受可控闸 c 与 σ 调节的加权 NLL。
  L_inv  (逆动力学): 从 (Z_{t+1}^target - Z_t) ⊙ c 反推真实动作 a_t。
    Minecraft 动作混合 —— 鼠标 (dx,dy) 回归(MSE)+ 20 个键盘按键二分类(BCE)。
  L_sigreg (防坍缩): sliced 高斯正则,确保潜序列不坍缩(blocks/primitives.SIGReg)。

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

from net.minecraft_world_model import MinecraftWorldModel, sinusoidal_time_encoding
from utils.vpt_dataset import VPTStreamDataset
from blocks.primitives import SIGReg

EPS = 1e-4          # I1
ACT_DIM = 22        # 2 鼠标 + 20 键盘(见 VPT_KEYS)
N_MOUSE = 2


# =====================================================================
# 损失函数
# =====================================================================

def jepa_pred_loss(mu, sigma, c, z_target):
    """JEPA 前向预测损失(加权 NLL)。可控槽(c→1)退化为 MSE;不可控槽 σ 兜底方差。"""
    diff_sq = (mu - z_target).square()
    sigma_sq = sigma.square().clamp(min=EPS)
    nll = c * diff_sq + (1 - c) * (diff_sq / sigma_sq + sigma_sq.log())
    return nll.mean()


def minecraft_inv_dyn_loss(z_t, z_target, c, true_action, inv_dyn_head, mouse_w=0.05):
    """Minecraft 逆动力学:从 (Z_next - Z_t)⊙c 反推混合动作。鼠标 MSE + 键盘 BCE。"""
    delta_z = (z_target - z_t) * c
    pred = inv_dyn_head(delta_z)                          # [B, ACT_DIM] = cat(mouse(2), kb_prob(20))
    mouse_pred, kb_pred = pred[:, :N_MOUSE], pred[:, N_MOUSE:]
    mouse_true, kb_true = true_action[:, :N_MOUSE], true_action[:, N_MOUSE:]
    l_mouse = F.mse_loss(mouse_pred, mouse_true)
    l_kb = F.binary_cross_entropy(kb_pred.clamp(EPS, 1 - EPS), kb_true)
    return l_kb + mouse_w * l_mouse, l_mouse.detach(), l_kb.detach()


def encode_target(model, img_next, t_next, Z_out):
    """干净(无空间掩码)的 stop-grad JEPA 目标:复刻 encode_vision 但跳过训练期掩码。"""
    with torch.no_grad():
        patch = model.vision_encoder(img_next)
        patch = patch + sinusoidal_time_encoding(t_next, model.d)
        return model.binder(Z_out.detach(), patch)


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
    """对一个 batch 的完整序列做截断 BPTT 前向+反向;损失以张量形式累加进 acc。"""
    B, T = img.shape[0], img.shape[1]
    Z = torch.zeros(B, model.N, model.d, device=device)
    h = torch.zeros(B, 1, model.d, device=device)
    a_raw = torch.zeros(B, model.J, ACT_DIM, device=device)
    accum = torch.zeros((), device=device)
    z_collect = []
    last_c = None

    for t in range(T - 1):
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            patch = model.vision_encoder(img[:, t])
            a_raw = roll_append(a_raw, action[:, t])
            out = model(patch, Z, h, a_raw, t_vec[:, t])
            z_target = encode_target(model, img[:, t + 1], t_vec[:, t + 1], out["Z_out"])

        # 损失在 fp32 计算(autocast 外:BCE 不可 autocast,NLL/逆动力学在 fp32 更稳)
        mu, sigma, c = out["mu"].float(), out["sigma"].float(), out["c"].float()
        zt = z_target.float()
        l_pred = jepa_pred_loss(mu, sigma, c, zt)
        l_inv, l_mouse, l_kb = minecraft_inv_dyn_loss(
            out["Z_out"].float(), zt, c, action[:, t], model.inv_dyn)

        accum = accum + (l_pred + alpha_inv * l_inv) / (T - 1)
        z_collect.append(out["Z_out"])
        last_c = c
        # 张量累加(不 .item(),epoch 末一次性取出 → 去掉逐步 GPU↔CPU 同步)
        acc["pred"] += l_pred.detach(); acc["inv"] += l_inv.detach()
        acc["mouse"] += l_mouse; acc["kb"] += l_kb; acc["inner"] += 1

        if (t + 1) % k_bptt == 0 or (t + 1) == (T - 1):
            z_stack = torch.stack(z_collect, dim=0).float()   # [G,B,N,d];SIGReg 全程 fp32
            l_sig = z_stack.new_zeros(())
            for si in range(z_stack.shape[2]):
                l_sig = l_sig + sigreg(z_stack[:, :, si, :])
            l_sig = l_sig / z_stack.shape[2]
            scaler.scale(accum + beta_sigreg * l_sig).backward()
            acc["loss"] += (accum + beta_sigreg * l_sig).detach()
            acc["sigreg"] += l_sig.detach(); acc["win"] += 1
            accum = torch.zeros((), device=device)
            z_collect = []
            Z, h = out["mu"].detach(), out["h_next"].detach()
        else:
            Z, h = out["mu"], out["h_next"]
    return last_c


def train_epoch(model, sigreg, loader, opt, scaler, device, steps, k_bptt,
                alpha_inv, beta_sigreg, amp_dev, use_amp):
    model.train()
    acc = {k: torch.zeros((), device=device) for k in ["loss", "pred", "inv", "mouse", "kb", "sigreg"]}
    acc["inner"] = acc["win"] = 0
    gpu_samples, c_last, n_batches = [], None, 0

    for batch in itertools.islice(loader, steps):
        img = batch["img"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        t_vec = batch["t_vec"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        c_last = _run_sequence(model, sigreg, img, action, t_vec, device, k_bptt,
                               alpha_inv, beta_sigreg, amp_dev, use_amp, scaler, acc)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        n_batches += 1
        u = _gpu_util()
        if u is not None:
            gpu_samples.append(u)

    ic = max(acc["inner"], 1); wc = max(acc["win"], 1)
    cv = c_last.squeeze(-1).flatten()
    return {
        "loss": (acc["loss"] / wc).item(),
        "pred": (acc["pred"] / ic).item(), "inv": (acc["inv"] / ic).item(),
        "mouse": (acc["mouse"] / ic).item(), "kb": (acc["kb"] / ic).item(),
        "sigreg": (acc["sigreg"] / wc).item(),
        "c_mean": cv.mean().item(), "c_std": cv.std().item(),
        "batches": n_batches,
        "gpu_util": (sum(gpu_samples) / len(gpu_samples)) if gpu_samples else None,
    }


@torch.no_grad()
def evaluate(model, loader, device, steps, amp_dev, use_amp):
    """逆动力学键盘平衡准确率:能否从隐变量变化里反推出"按了哪些键"。"""
    model.eval()
    tp = fp = fn = tn = 0
    for batch in itertools.islice(loader, steps):
        img = batch["img"].to(device); action = batch["action"].to(device)
        t_vec = batch["t_vec"].to(device)
        B, T = img.shape[0], img.shape[1]
        Z = torch.zeros(B, model.N, model.d, device=device)
        h = torch.zeros(B, 1, model.d, device=device)
        a_raw = torch.zeros(B, model.J, ACT_DIM, device=device)
        for t in range(T - 1):
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                patch = model.vision_encoder(img[:, t])
                a_raw = roll_append(a_raw, action[:, t])
                out = model(patch, Z, h, a_raw, t_vec[:, t])
                z_target = encode_target(model, img[:, t + 1], t_vec[:, t + 1], out["Z_out"])
                pred = model.inv_dyn((z_target - out["Z_out"]) * out["c"])
            kb_pred = (pred[:, N_MOUSE:] > 0.5)
            kb_true = (action[:, t, N_MOUSE:] > 0.5)
            tp += (kb_pred & kb_true).sum().item();  fp += (kb_pred & ~kb_true).sum().item()
            fn += (~kb_pred & kb_true).sum().item();  tn += (~kb_pred & ~kb_true).sum().item()
            Z, h = out["mu"], out["h_next"]
    recall = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"kb_recall": recall, "kb_spec": spec, "kb_bal_acc": 0.5 * (recall + spec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample", help="VPTDataset 数据目录(.mp4+.jsonl)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=50, help="每 epoch 迭代多少个 batch(流式)")
    ap.add_argument("--batch", type=int, default=2, help="每步随机取几个序列")
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader 并行加载进程数")
    ap.add_argument("--cache_size", type=int, default=32, help="每个 worker 缓存的已解码序列上限")
    ap.add_argument("--refresh_every", type=int, default=64, help="每多少样本换出一个旧片段")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--N", type=int, default=16, help="实体槽数")
    ap.add_argument("--K", type=int, default=5, help="动作查询数")
    ap.add_argument("--J", type=int, default=8, help="历史动作长度")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha_inv", type=float, default=1.0, help="逆动力学损失权重")
    ap.add_argument("--beta_sigreg", type=float, default=0.1, help="SIGReg 防坍缩权重")
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
    print(f"=== MINECRAFT WORLD MODEL (JEPA + InvDyn + SIGReg) | device={dev} | amp={use_amp} ===")

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

    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                          cache_size=args.cache_size, refresh_every=args.refresh_every, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=is_cuda,
                        persistent_workers=(args.workers > 0),
                        prefetch_factor=(2 if args.workers > 0 else None))

    model = MinecraftWorldModel(d=args.d, N=args.N, K=args.K, J=args.J, act_dim=ACT_DIM).to(dev)
    model.J = args.J            # 历史动作长度(MinecraftWorldModel 不强制,训练循环按此构造 a_raw)
    sigreg = SIGReg(knots=17, num_proj=512).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | "
          f"batch={args.batch} workers={args.workers} steps/epoch={args.steps_per_epoch}")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    util_hist = []

    for ep in range(args.epochs):
        r = train_epoch(model, sigreg, loader, opt, scaler, dev, args.steps_per_epoch,
                        args.k_bptt, args.alpha_inv, args.beta_sigreg, amp_dev, use_amp)
        if r.get("gpu_util") is not None:
            util_hist.append(r["gpu_util"])
        if use_wandb:
            _wb = {k: r[k] for k in ("loss", "pred", "inv", "mouse", "kb", "sigreg", "c_mean", "c_std")}
            if r.get("gpu_util") is not None:
                _wb["gpu_util"] = r["gpu_util"]
            wandb.log(_wb, step=ep)
        if ep % 5 == 0 or ep == args.epochs - 1:
            gpu = f" | gpu {r['gpu_util']:.0f}%" if r.get("gpu_util") is not None else ""
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | pred {r['pred']:.3f} "
                  f"inv {r['inv']:.3f} (kb {r['kb']:.3f}/mouse {r['mouse']:.2f}) "
                  f"sig {r['sigreg']:.2f} | c mean={r['c_mean']:.3f} std={r['c_std']:.3f}{gpu}")

    if util_hist:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[GPU] 训练平均利用率 {sum(util_hist) / len(util_hist):.0f}% "
              f"({len(util_hist)} epoch) | 峰值显存 {peak:.2f} GB")

    print("\n--- 最终评估:逆动力学(从画面变化反推按键)---")
    e = evaluate(model, loader, dev, max(args.steps_per_epoch, 20), amp_dev, use_amp)
    print(f"键盘 recall {e['kb_recall']:.3f} | spec {e['kb_spec']:.3f} | "
          f"平衡准确率 {e['kb_bal_acc']:.3f}(随机基线≈0.5)")
    ok = e["kb_bal_acc"] > 0.6
    print(f"=> {'✅ 世界模型从画面里读出了动作信息' if ok else '⚠ 动作信息尚不显著(欠训练/调 α/数据太少)'}")

    if use_wandb:
        wandb.log({"eval/kb_recall": e["kb_recall"], "eval/kb_spec": e["kb_spec"],
                   "eval/kb_bal_acc": e["kb_bal_acc"]})
        wandb.summary["kb_bal_acc"] = e["kb_bal_acc"]
        wandb.finish()


if __name__ == "__main__":
    main()
