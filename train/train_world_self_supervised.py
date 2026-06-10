"""自监督世界模型训练：JEPA 潜空间预测 + 逆动力学 + SIGReg 防坍缩。

核心数学：
  L_total = L_pred + α·L_inv + β·L_sigreg

  L_pred（前向预测）：模型从 (Z_t, a_t) 预测 μ_{t+1}，目标为 stop-grad 的
    Z_{t+1}^target = encode(img_{t+1})。受 c_i 和 σ 调节的加权 NLL：
    可控槽（c→1）：严格 MSE；不可控槽（c→0）：σ 兜底方差。
  L_inv（逆动力学）：从 (Z_{t+1}^target - Z_t) ⊙ c 反推动作 a_t（BCE）。
    梯度回流到 c_logit，驱动极化：可解释的槽 c→1，否则 c→0。
  L_sigreg（防坍缩）：确保 Z 不坍缩到常数。

验证指标（训练时无监督标签，仅探针事后读出）：
  - c_i 极化：c 的方差（越大越好，说明网络在分离前景与背景）
  - 逆动力学准确率：能否从隐变量变化中反推出动作
  - 探针读出精度：冻结线性探针从 Z 中读出 lane 状态（与有监督版本对照）

用法：
    python train_world_self_supervised.py --epochs 200 --device cuda
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from net.tao_not_42 import TaoNot42Model
from utils.rhythm_action_env import RhythmActionEnv
from blocks.primitives import SIGReg, PreLNAttn

EPS = 1e-4  # I1


# =====================================================================
# 探针（纯评估用，不参与世界模型训练）
# =====================================================================

class EvalLaneProbe(nn.Module):
    """冻结的事后评估探针：验证 Z 是否学到了世界信息。

    与 train_action_world.py 中的 LaneProbe 结构相同，但在自监督训练中
    不参与主损失的反向传播，仅在评估阶段使用。
    """
    def __init__(self, d, n_lanes=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_lanes, d) * 0.02)
        self.attn = PreLNAttn(d, heads=4, mode="cross")
        self.trunk = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.SiLU())
        self.y = nn.Linear(128, 1)
        self.present = nn.Linear(128, 1)
        self.hittable = nn.Linear(128, 1)

    def forward(self, slots):
        q = self.queries.expand(slots.shape[0], -1, -1)
        f = self.trunk(self.attn(q, slots))
        return {"y_norm": torch.sigmoid(self.y(f).squeeze(-1)),
                "present": torch.sigmoid(self.present(f).squeeze(-1)),
                "hittable": torch.sigmoid(self.hittable(f).squeeze(-1))}


# =====================================================================
# 损失函数
# =====================================================================

Y_MIN, Y_MAX = -24.0, 280.0


def jepa_pred_loss(mu, sigma, c, z_target):
    """JEPA 前向预测损失（加权 NLL）。

    对可控槽（c→1），退化为严格的 MSE。
    对不可控槽（c→0），σ 兜底高方差，退化为预测分布的包络。

    ⚠️ c 在本损失中 detach（数学原因）：对 σ 取最优（σ²=diff²）时 NLL 支路的值为
    1+log diff²，由 log x ≤ x−1 得它 ≤ diff²（MSE 支路）对一切 diff² 恒成立——
    若让梯度流向 c，本损失会无条件把 c 推向 0，与"可控→c→1"的设计意图相反。
    c 的极化只由逆动力学损失驱动；这里 c 仅作固定权重。

    Args:
        mu: [B, N, d] 模型预测的未来状态均值
        sigma: [B, N, d] 模型预测的不确定性
        c: [B, N, 1] 可控性闸门（本损失内 stop-grad）
        z_target: [B, N, d] stop-grad 的编码目标

    Returns:
        标量损失
    """
    diff_sq = (mu - z_target).square()  # [B, N, d]
    sigma_sq = sigma.square().clamp(min=EPS)  # I1
    c = c.detach()

    # 可控项（c 大）：MSE 主导，要求 mu 精准
    # 不可控项（1-c 大）：σ 吸收方差，只要求标定不确定性
    nll = c * diff_sq + (1 - c) * (diff_sq / sigma_sq + sigma_sq.log())

    return nll.mean()


def inverse_dynamics_loss(z_t, z_target, c, true_action, inv_dyn_head):
    """逆动力学损失（BCE）。

    通过 (Z_next - Z_t) ⊙ c 反推动作。梯度回流到 c_logit。

    ⚠️ z_t 必须传纯感知编码 out["Z_enc"]（binder 输出，未经 Transformer）：
    a_raw 在 forward 前已滚入当前动作，Z_out 见过它——若用 Z_out，模型可把动作
    嵌进 Z_out 让 inv-dyn 不经视觉直接读回，护栏失效。

    Args:
        z_t: [B, N, d] 当前帧纯感知编码（Z_enc，带梯度）
        z_target: [B, N, d] 下一帧的 stop-grad 编码
        c: [B, N, 1] 可控性闸门
        true_action: [B, n_keys] 真实动作（0/1）
        inv_dyn_head: InverseDynamicsHead 模块

    Returns:
        标量损失
    """
    delta_z = (z_target - z_t) * c  # [B, N, d]，低 c 的 slot 被压到近零
    pred_action_logits = inv_dyn_head(delta_z)  # [B, n_keys]
    return F.binary_cross_entropy_with_logits(pred_action_logits, true_action)


def saliency_gaze_loss(out, img_t, img_next):
    """saliency / gaze 头的自监督信号（修复"注视头零梯度"缺陷）。

    训练循环里 gaze 输出一律 detach 后才喂下一步、且无任何直接损失 ⇒ gaze 头
    此前没有任何梯度来源，主动视觉等价于随机扫视；saliency 头同样闲置。

    代理目标 = 帧间运动能量（"预测误差惊奇图"的廉价稠密代理；信息增益最大的
    注视点 ≈ 变化最大处）：
      - saliency 头回归 8×8 运动能量图（与 PeripheralVision 输出网格对齐）；
      - gaze (g_x, g_y) 回归该图的 soft-argmax 期望坐标 ∈ [-1,1]
        （与 crop_fovea 的仿射平移坐标系一致）；g_s 不监督，保持自由。
    """
    with torch.no_grad():
        diff = (img_next - img_t).abs().mean(1, keepdim=True)            # [B,1,H,W]
        tgt = F.adaptive_avg_pool2d(diff, (8, 8)).squeeze(1)             # [B,8,8]
        tgt = tgt / tgt.amax(dim=(1, 2), keepdim=True).clamp(min=EPS)    # 逐样本归一到 [0,1]
        w = tgt.flatten(1)
        w = w / w.sum(1, keepdim=True).clamp(min=EPS)                    # 概率化
        coords = torch.linspace(-1.0, 1.0, 8, device=img_t.device)
        wg = w.view(-1, 8, 8)
        gx_t = (wg.sum(1) * coords).sum(1)   # 列边缘分布 × x 坐标 → [B]
        gy_t = (wg.sum(2) * coords).sum(1)   # 行边缘分布 × y 坐标 → [B]
    l_sal = F.mse_loss(out["saliency"], tgt)
    g_x, g_y, _ = out["gaze"]
    l_gaze = F.smooth_l1_loss(g_x, gx_t) + F.smooth_l1_loss(g_y, gy_t)
    return l_sal + l_gaze


def probe_lane_loss(pred, gt):
    """探针评估用的 lane loss（与有监督版本一致）。"""
    pres = gt["present"]
    y_t = ((gt["y"] - Y_MIN) / (Y_MAX - Y_MIN)).clamp(0, 1)
    l_y = (F.smooth_l1_loss(pred["y_norm"], y_t, reduction="none") * pres).sum() / pres.sum().clamp(min=1)
    l_p = F.binary_cross_entropy(pred["present"], pres)
    l_h = F.binary_cross_entropy(pred["hittable"], gt["hittable"])
    return l_y * 5.0 + l_p + l_h * 3.0


# =====================================================================
# 采样动作（复用 train_action_world.py 的探索性随机策略）
# =====================================================================

def sample_actions(env, hit_p=0.85, near_p=0.30, rand_p=0.03, near_win=0.15):
    """探索性随机动作采样。"""
    ls = env.get_lane_state()
    hittable, onset = ls["hittable"], ls["onset"]
    near = (onset.abs() <= near_win) & (hittable < 0.5)
    p = torch.full_like(onset, rand_p)
    p = torch.where(near, torch.full_like(p, near_p), p)
    p = torch.where(hittable > 0.5, torch.full_like(p, hit_p), p)
    return (torch.rand_like(p) < p).float()


# =====================================================================
# 初始化辅助
# =====================================================================

def make_inputs(B, model, device):
    Z = torch.zeros(B, model.N, model.d, device=device)
    h = torch.zeros(B, 1, model.d, device=device)
    t_hist = model.action_enc.net[0].in_features // model.n_keys
    a_raw = torch.zeros(B, t_hist, model.n_keys, device=device)
    g = (torch.zeros(B, device=device), torch.zeros(B, device=device),
         torch.ones(B, device=device) * 0.5)
    return Z, h, a_raw, g


# =====================================================================
# 训练循环
# =====================================================================

def train_step(env, model, sigreg, B, dt, rollout, device, opt, k_bptt=4,
               alpha_inv=1.0, beta_sigreg=0.1, gamma_sg=0.1):
    """一个 epoch 的自监督训练。

    核心循环：
      1. model.forward(img_t) → out_t（含 mu, sigma, c, Z_out, Z_enc）
      2. env.step(dt, action) → img_{t+1}
      3. model.encode(img_{t+1}, slot_anchor) → Z_target（stop-grad，固定锚查询）
      4. L = L_pred + α·L_inv + β·L_sigreg + γ·L_saliency_gaze
    """
    env.reset()
    Z, h, a_raw, g = make_inputs(B, model, device)
    dt_vec = torch.ones(B, device=device)  # τ 以帧为单位:每 step=1 帧(见 ContinuousTimeEncoding;env.step 仍吃秒 dt)

    # warmup：灌满管线
    for _ in range(80):
        env.step(dt)
    img = env.render()
    opt.zero_grad()

    total_loss = torch.zeros((), device=device)
    total_inv = torch.zeros((), device=device)
    total_sg = torch.zeros((), device=device)
    accum_loss = 0

    # 收集 Z 用于 SIGReg
    z_collect = []

    for t in range(rollout):
        # 采样动作
        action = sample_actions(env)
        a_raw = torch.cat([a_raw[:, 1:], action.unsqueeze(1)], dim=1)

        # 前向推演
        out = model(img, Z, h, a_raw, dt_vec, g)
        mu, sigma, c = out["mu"], out["sigma"], out["c"]

        # 执行动作，获取下一帧
        env.step(dt, action)
        img_next = env.render()

        # 生成 stop-grad 目标(固定锚查询,与递归状态解耦:
        # 旧版用 Z_out.detach() 做查询,binder 增益→0 时 z_target≈Z_out,
        # pred loss 存在平凡不动点,SIGReg 防不住这条信息坍缩通道)
        with torch.no_grad():
            anchor = model.slot_anchor.expand(B, -1, -1)
            z_target, _ = model.encode(img_next, anchor, g)

        # --- 损失 1：JEPA 前向预测 ---
        l_pred = jepa_pred_loss(mu, sigma, c, z_target)

        # --- 损失 2：逆动力学（吃纯感知编码 Z_enc，防动作直通作弊，见函数 docstring）---
        l_inv = inverse_dynamics_loss(
            out["Z_enc"], z_target, c, action, model.inv_dyn)

        # --- 损失 4：saliency / gaze 自监督（运动能量代理惊奇图）---
        l_sg = saliency_gaze_loss(out, img, img_next)

        # 收集 Z 用于 SIGReg（每步都收集）
        z_collect.append(out["Z_out"])

        step_loss = l_pred + alpha_inv * l_inv + gamma_sg * l_sg
        accum_loss = accum_loss + step_loss / rollout
        total_loss = total_loss + step_loss.detach()
        total_inv = total_inv + l_inv.detach()
        total_sg = total_sg + l_sg.detach()

        # 截断 BPTT
        if (t + 1) % k_bptt == 0 or (t + 1) == rollout:
            # --- 损失 3：SIGReg 防坍缩 ---
            if z_collect:
                # 把收集到的 Z 堆叠成 [G, B, d]（G=时间步数, B=batch, d=特征维）
                # 对每个 slot 独立做 SIGReg，取均值
                z_stack = torch.stack(z_collect, dim=0)  # [G, B, N, d]
                l_sigreg = 0
                N = z_stack.shape[2]
                for slot_i in range(N):
                    l_sigreg = l_sigreg + sigreg(z_stack[:, :, slot_i, :])
                l_sigreg = l_sigreg / N
                full_loss = accum_loss + beta_sigreg * l_sigreg
                z_collect = []
            else:
                full_loss = accum_loss

            full_loss.backward()
            accum_loss = 0
            Z = out["mu"].detach()
            h = out["h_next"].detach()
            g = tuple(x.detach() for x in out["gaze"])
        else:
            Z = out["mu"]
            h = out["h_next"]
            g = out["gaze"]

        img = img_next

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    # c_i 统计（现在是动态的）
    c_vals = c.squeeze(-1).flatten()  # [B*N]
    c_mean = c_vals.mean().item()
    c_std = c_vals.std().item()
    c_max = c_vals.max().item()
    c_min = c_vals.min().item()

    return {
        "loss": (total_loss / rollout).item(),
        "inv_loss": (total_inv / rollout).item(),
        "sg_loss": (total_sg / rollout).item(),
        "c_mean": c_mean, "c_std": c_std,
        "c_max": c_max, "c_min": c_min,
    }


# =====================================================================
# 探针评估（自监督训练完成后，单独训练一个小探针验证 Z 的质量）
# =====================================================================

@torch.no_grad()
def eval_with_probe(env, model, probe, B, dt, rollout, device):
    """用冻结的世界模型 + 训好的探针，评估 Z 的质量。"""
    env.reset()
    Z, h, a_raw, g = make_inputs(B, model, device)
    dt_vec = torch.ones(B, device=device)  # τ 以帧为单位:每 step=1 帧(见 ContinuousTimeEncoding;env.step 仍吃秒 dt)
    for _ in range(80):
        env.step(dt)
    img = env.render()

    total_probe_loss = 0
    m = {k: torch.zeros((), device=device) for k in
         ["ph", "nh", "ps", "ns", "accdec", "ndec"]}

    for t in range(rollout):
        gt_now = env.get_lane_state()
        action = sample_actions(env)
        a_raw = torch.cat([a_raw[:, 1:], action.unsqueeze(1)], dim=1)
        out = model(img, Z, h, a_raw, dt_vec, g)

        env.step(dt, action)
        img_next = env.render()
        gt_next = env.get_lane_state()

        # 探针读出下一帧
        pred_next = probe(out["mu"])
        total_probe_loss += probe_lane_loss(pred_next, gt_next).item()

        # 统计指标（与 train_action_world.py 一致）
        pred_h = pred_next["hittable"]
        hit = env.last_hit
        stay = ((gt_now["hittable"] > 0.5) & (action < 0.5)).float()
        m["ph"] += (pred_h * hit).sum();   m["nh"] += hit.sum()
        m["ps"] += (pred_h * stay).sum();  m["ns"] += stay.sum()
        dec = (gt_now["hittable"] > 0.5).float()
        correct = ((pred_h > 0.5) == (gt_next["hittable"] > 0.5)).float()
        m["accdec"] += (correct * dec).sum(); m["ndec"] += dec.sum()

        Z = out["mu"].detach()
        h = out["h_next"].detach()
        g = tuple(x.detach() for x in out["gaze"])
        img = img_next

    nh = m["nh"].clamp(min=1); ns = m["ns"].clamp(min=1)
    c_vals = out["c"].squeeze(-1).flatten()
    return {
        "probe_loss": total_probe_loss / rollout,
        "PredP_hit": (m["ph"] / nh).item(),
        "PredP_stay": (m["ps"] / ns).item(),
        "Gap": ((m["ps"] / ns) - (m["ph"] / nh)).item(),
        "DecAcc": (m["accdec"] / m["ndec"].clamp(min=1)).item(),
        "c_mean": c_vals.mean().item(),
        "c_std": c_vals.std().item(),
        "c_min": c_vals.min().item(),
        "c_max": c_vals.max().item(),
    }


def train_probe_on_frozen_model(env, model, probe, B, dt, rollout, device, epochs=50):
    """在冻结的世界模型上，训练一个小探针来验证 Z 的质量。"""
    model.eval()
    opt_probe = torch.optim.Adam(probe.parameters(), lr=3e-4)

    for ep in range(epochs):
        env.reset()
        Z, h, a_raw, g = make_inputs(B, model, device)
        dt_vec = torch.ones(B, device=device)  # τ 以帧为单位:每 step=1 帧(见 ContinuousTimeEncoding;env.step 仍吃秒 dt)
        for _ in range(80):
            env.step(dt)
        img = env.render()
        opt_probe.zero_grad()

        total = 0
        for t in range(rollout):
            gt_now = env.get_lane_state()
            action = sample_actions(env)
            a_raw = torch.cat([a_raw[:, 1:], action.unsqueeze(1)], dim=1)

            with torch.no_grad():
                out = model(img, Z, h, a_raw, dt_vec, g)
                Z_out = out["Z_out"].detach()
                mu = out["mu"].detach()

            # 探针读当前
            pred_now = probe(Z_out)
            l_now = probe_lane_loss(pred_now, gt_now)

            env.step(dt, action)
            img_next = env.render()
            gt_next = env.get_lane_state()

            pred_next = probe(mu)
            l_next = probe_lane_loss(pred_next, gt_next)

            total = total + (l_now + l_next) / rollout

            Z = mu
            h = out["h_next"].detach()
            g = tuple(x.detach() for x in out["gaze"])
            img = img_next

        total.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt_probe.step()

        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  probe ep {ep:3d} | loss {total.item():.3f}")

    return probe


# =====================================================================
# 主函数
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--rollout", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--spawn", type=float, default=2.5)
    ap.add_argument("--alpha_inv", type=float, default=1.0,
                    help="逆动力学损失权重")
    ap.add_argument("--beta_sigreg", type=float, default=0.1,
                    help="SIGReg 防坍缩权重")
    ap.add_argument("--gamma_sg", type=float, default=0.1,
                    help="saliency/gaze 自监督权重(运动能量代理惊奇图)")
    ap.add_argument("--probe_epochs", type=int, default=50,
                    help="事后探针训练轮数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"=== SELF-SUPERVISED WORLD MODEL (JEPA + InvDyn + SIGReg) | device={dev} ===")

    env = RhythmActionEnv(batch_size=args.batch, device=dev, spawn_prob=args.spawn)
    model = TaoNot42Model(d=args.d, N=16, M=2, J=2, n_keys=4, t_hist=10,
                          layers=args.layers, K=6).to(dev)
    sigreg = SIGReg(knots=17, num_proj=512).to(dev)

    # 世界模型的优化器（不含探针）
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f"\n--- Phase 1: Self-supervised world model training ({args.epochs} epochs) ---")
    for ep in range(args.epochs):
        model.train()
        r = train_step(env, model, sigreg, args.batch, args.dt, args.rollout,
                       dev, opt, alpha_inv=args.alpha_inv,
                       beta_sigreg=args.beta_sigreg, gamma_sg=args.gamma_sg)

        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d} | loss {r['loss']:6.3f} | inv {r['inv_loss']:.3f} | "
                  f"sg {r['sg_loss']:.3f} | "
                  f"c: mean={r['c_mean']:.3f} std={r['c_std']:.3f} "
                  f"[{r['c_min']:.3f}, {r['c_max']:.3f}]")

    print(f"\n--- Phase 2: Post-hoc probe training ({args.probe_epochs} epochs) ---")
    probe = EvalLaneProbe(args.d).to(dev)
    probe = train_probe_on_frozen_model(
        env, model, probe, args.batch, args.dt, args.rollout, dev,
        epochs=args.probe_epochs)

    print("\n--- Phase 3: Final evaluation ---")
    model.eval()
    probe.eval()
    with torch.no_grad():
        r = eval_with_probe(env, model, probe, args.batch, args.dt, args.rollout, dev)

    print("\n===== VERDICT (self-supervised world model) =====")
    print(f"Probe loss       : {r['probe_loss']:.3f}")
    print(f"PredP | hit      : {r['PredP_hit']:.3f}   (应→0: 按掉后下一刻不再可命中)")
    print(f"PredP | stay     : {r['PredP_stay']:.3f}   (应偏高: 没被打)")
    print(f"Gap (stay - hit) : {r['Gap']:+.3f}   (越大越说明学到'按→消除')")
    print(f"DecAcc           : {r['DecAcc']:.3f}   (决策点准确率; 随机基线≈0.5)")

    print(f"\nc_i mean={r['c_mean']:.3f} std={r['c_std']:.3f} [{r['c_min']:.3f}, {r['c_max']:.3f}]")

    success = r["Gap"] > 0.2 and r["DecAcc"] > 0.6
    print(f"\n=> {'✅ 自监督世界模型学到了动作因果效果' if success else '⚠ 尚需调参或更多训练'}")


if __name__ == "__main__":
    main()
