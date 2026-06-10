"""MinecraftWorldModel 自监督训练:在离线 VPT 视频(画面+动作)上学世界动力学。

目标:把"看着 Minecraft 录像 + 录像里的真实动作"变成自监督信号,让世界模型学到
"动作在画面里有什么效果",为后续 prompt-tuning 迁移打底座。

  L_total = L_pred + α·L_inv + β·L_sigreg

  L_pred (JEPA 前向预测): 从 (Z_t, a_t, t) 预测下一帧潜表征 μ_{t+1},
    目标为 stop-grad 的干净视觉编码 encode(img_{t+1})。受可控闸 c 与 σ 调节的加权 NLL。
  L_inv  (逆动力学): 从 (Z_{t+1}^target - Z_t) ⊙ c 反推真实动作 a_t。
    Minecraft 动作是混合型 —— 鼠标 (dx,dy) 回归(MSE)+ 20 个键盘按键二分类(BCE)。
    梯度回流到 c_logit,驱动槽极化:与动作相关的槽 c→1,背景 c→0。
  L_sigreg (防坍缩): sliced 高斯正则,确保潜序列不坍缩到常数(见 blocks/primitives.SIGReg)。

数据来自 utils/vpt_dataset.VPTDataset(.mp4 + .jsonl)。先跑:
    python utils/download_sample_data.py          # 生成 runs/vpt_sample/
然后:
    python train/train_minecraft.py --data_dir runs/vpt_sample --epochs 50 --device cuda
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from net.minecraft_world_model import MinecraftWorldModel, sinusoidal_time_encoding
from utils.vpt_dataset import VPTDataset
from blocks.primitives import SIGReg

EPS = 1e-4          # I1
ACT_DIM = 22        # 2 鼠标 + 20 键盘(见 VPTDataset)
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
    """Minecraft 逆动力学损失:从 (Z_next - Z_t)⊙c 反推混合动作。

    inv_dyn_head 输出 cat([mouse(2, 回归), kb_prob(20, 已 sigmoid)])。
    鼠标用 MSE,键盘用 BCE。梯度回流到 c。
    """
    delta_z = (z_target - z_t) * c                       # [B,N,d],低 c 槽被压到近零
    pred = inv_dyn_head(delta_z)                          # [B, ACT_DIM]
    mouse_pred, kb_pred = pred[:, :N_MOUSE], pred[:, N_MOUSE:]
    mouse_true, kb_true = true_action[:, :N_MOUSE], true_action[:, N_MOUSE:]
    l_mouse = F.mse_loss(mouse_pred, mouse_true)
    l_kb = F.binary_cross_entropy(kb_pred.clamp(EPS, 1 - EPS), kb_true)
    return l_kb + mouse_w * l_mouse, l_mouse.detach(), l_kb.detach()


def encode_target(model, img_next, t_next, Z_out):
    """干净(无空间掩码)的 stop-grad JEPA 目标:复刻 encode_vision 但跳过训练期掩码。"""
    with torch.no_grad():
        patch = model.vision_encoder(img_next)                       # [B,M,d]
        patch = patch + sinusoidal_time_encoding(t_next, model.d)    # 注入绝对时间戳
        return model.binder(Z_out.detach(), patch)                   # 绑定到实体槽 -> [B,N,d]


# =====================================================================
# 训练 / 评估
# =====================================================================

def roll_append(a_raw, action):
    """滚动追加当前动作到历史动作缓冲。a_raw:[B,J,ACT_DIM], action:[B,ACT_DIM]。"""
    return torch.cat([a_raw[:, 1:], action.unsqueeze(1)], dim=1)


def train_epoch(model, sigreg, loader, opt, device, k_bptt, alpha_inv, beta_sigreg):
    model.train()
    agg = {k: 0.0 for k in ["loss", "pred", "inv", "mouse", "kb", "sigreg", "n"]}
    c_last = None

    for batch in loader:
        img = batch["img"].to(device)          # [B,T,3,H,W]
        action = batch["action"].to(device)    # [B,T,ACT_DIM]
        t_vec = batch["t_vec"].to(device)      # [B,T]
        B, T = img.shape[0], img.shape[1]

        Z = torch.zeros(B, model.N, model.d, device=device)
        h = torch.zeros(B, 1, model.d, device=device)
        a_raw = torch.zeros(B, model.J, ACT_DIM, device=device)   # 历史动作缓冲

        opt.zero_grad()
        accum = torch.zeros((), device=device)
        z_collect = []
        steps = 0

        for t in range(T - 1):
            patch = model.vision_encoder(img[:, t])            # [B,M,d](forward 内部会再做空间掩码增广)
            a_raw = roll_append(a_raw, action[:, t])
            out = model(patch, Z, h, a_raw, t_vec[:, t])
            mu, sigma, c = out["mu"], out["sigma"], out["c"]

            z_target = encode_target(model, img[:, t + 1], t_vec[:, t + 1], out["Z_out"])

            l_pred = jepa_pred_loss(mu, sigma, c, z_target)
            l_inv, l_mouse, l_kb = minecraft_inv_dyn_loss(
                out["Z_out"], z_target, c, action[:, t], model.inv_dyn)

            step_loss = l_pred + alpha_inv * l_inv
            accum = accum + step_loss / (T - 1)
            z_collect.append(out["Z_out"])
            steps += 1

            agg["pred"] += l_pred.item(); agg["inv"] += l_inv.item()
            agg["mouse"] += l_mouse.item(); agg["kb"] += l_kb.item()

            # 截断 BPTT:每 k_bptt 步反传一次并切断历史
            if (t + 1) % k_bptt == 0 or (t + 1) == (T - 1):
                z_stack = torch.stack(z_collect, dim=0)         # [G,B,N,d]
                l_sig = z_stack.new_zeros(())
                for slot_i in range(z_stack.shape[2]):
                    l_sig = l_sig + sigreg(z_stack[:, :, slot_i, :])  # 每槽独立做高斯检验
                l_sig = l_sig / z_stack.shape[2]
                (accum + beta_sigreg * l_sig).backward()
                agg["sigreg"] += l_sig.item()
                agg["loss"] += float(accum.detach()) + beta_sigreg * l_sig.item()
                accum = torch.zeros((), device=device)
                z_collect = []
                Z, h = out["mu"].detach(), out["h_next"].detach()
            else:
                Z, h = out["mu"], out["h_next"]

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        agg["n"] += steps
        c_last = c.detach()

    n = max(agg["n"], 1)
    cv = c_last.squeeze(-1).flatten()
    return {
        "loss": agg["loss"] / max(len(loader), 1),
        "pred": agg["pred"] / n, "inv": agg["inv"] / n,
        "mouse": agg["mouse"] / n, "kb": agg["kb"] / n,
        "sigreg": agg["sigreg"] / max(len(loader), 1),
        "c_mean": cv.mean().item(), "c_std": cv.std().item(),
        "c_min": cv.min().item(), "c_max": cv.max().item(),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """逆动力学键盘准确率(平衡):模型能否从隐变量变化里反推出"按了哪些键"。"""
    model.eval()
    tp = fp = fn = tn = 0
    for batch in loader:
        img = batch["img"].to(device); action = batch["action"].to(device)
        t_vec = batch["t_vec"].to(device)
        B, T = img.shape[0], img.shape[1]
        Z = torch.zeros(B, model.N, model.d, device=device)
        h = torch.zeros(B, 1, model.d, device=device)
        a_raw = torch.zeros(B, model.J, ACT_DIM, device=device)
        for t in range(T - 1):
            patch = model.vision_encoder(img[:, t])
            a_raw = roll_append(a_raw, action[:, t])
            out = model(patch, Z, h, a_raw, t_vec[:, t])
            z_target = encode_target(model, img[:, t + 1], t_vec[:, t + 1], out["Z_out"])
            pred = model.inv_dyn(((z_target - out["Z_out"]) * out["c"]))
            kb_pred = (pred[:, N_MOUSE:] > 0.5)
            kb_true = (action[:, t, N_MOUSE:] > 0.5)
            tp += (kb_pred & kb_true).sum().item()
            fp += (kb_pred & ~kb_true).sum().item()
            fn += (~kb_pred & kb_true).sum().item()
            tn += (~kb_pred & ~kb_true).sum().item()
            Z, h = out["mu"], out["h_next"]
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"kb_recall": recall, "kb_spec": spec, "kb_bal_acc": 0.5 * (recall + spec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample", help="VPTDataset 数据目录(.mp4+.jsonl)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--N", type=int, default=16, help="实体槽数")
    ap.add_argument("--K", type=int, default=5, help="动作查询数")
    ap.add_argument("--J", type=int, default=8, help="历史动作长度")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha_inv", type=float, default=1.0, help="逆动力学损失权重")
    ap.add_argument("--beta_sigreg", type=float, default=0.1, help="SIGReg 防坍缩权重")
    ap.add_argument("--k_bptt", type=int, default=4, help="截断 BPTT 窗口")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"=== MINECRAFT WORLD MODEL (JEPA + InvDyn + SIGReg) | device={dev} ===")

    if not os.path.isdir(args.data_dir) or not any(
            f.endswith(".mp4") for f in os.listdir(args.data_dir)):
        print(f"[!] 数据目录 '{args.data_dir}' 里没有 .mp4。先运行:")
        print(f"    python utils/download_sample_data.py --out {args.data_dir}")
        sys.exit(1)

    ds = VPTDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps)
    if len(ds) == 0:
        print(f"[!] VPTDataset 在 '{args.data_dir}' 没找到 >= seq_len({args.seq_len}) 的片段。"
              f"调小 --seq_len 或用 --frames 更长的数据重跑 download。")
        sys.exit(1)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = MinecraftWorldModel(d=args.d, N=args.N, K=args.K, J=args.J, act_dim=ACT_DIM).to(dev)
    model.J = args.J            # 历史动作长度(MinecraftWorldModel 不强制,训练循环按此构造 a_raw)
    sigreg = SIGReg(knots=17, num_proj=512).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | "
          f"clips≈{len(ds.videos)} | steps/epoch≈{len(loader)}")

    for ep in range(args.epochs):
        r = train_epoch(model, sigreg, loader, opt, dev,
                        args.k_bptt, args.alpha_inv, args.beta_sigreg)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | pred {r['pred']:.3f} "
                  f"inv {r['inv']:.3f} (kb {r['kb']:.3f}/mouse {r['mouse']:.2f}) "
                  f"sig {r['sigreg']:.2f} | c mean={r['c_mean']:.3f} std={r['c_std']:.3f}")

    print("\n--- 最终评估:逆动力学(从画面变化反推按键)---")
    e = evaluate(model, loader, dev)
    print(f"键盘 recall {e['kb_recall']:.3f} | spec {e['kb_spec']:.3f} | "
          f"平衡准确率 {e['kb_bal_acc']:.3f}(随机基线≈0.5)")
    ok = e["kb_bal_acc"] > 0.6
    print(f"=> {'✅ 世界模型从画面里读出了动作信息' if ok else '⚠ 动作信息尚不显著(欠训练/调 α/数据太少)'}")


if __name__ == "__main__":
    main()
