"""动作条件世界模型测试:模型能否预测"按键的效果"(命中→音符消失)。

数据流(每步):
  gt_now = env.get_lane_state()             # 当前每轨道(y, present)
  action = sample_actions(env)              # 探索性随机(近线多按,远线少按)
  a_raw  ← 滚动追加 current action          # 模型条件于动作
  out    = model(img, Z, h, a_raw, dt, g)
  loss   = lane_loss(probe(Z_out), gt_now)  # 读当前
         + lane_loss(probe(mu),    gt_next) # 预测下一帧(条件于动作)= 动作效果所在
监督只训世界预测;**准确率不是 loss**,只反馈:
  PredP|hit  : 命中事件上模型预测的 present(应→0)
  PredP|stay : 有音符且没按的轨道(应→1)
  HitAcc     : 命中事件中模型正确预测"消失"(pred<0.5)的比例
  shuffle    : 打乱动作后 loss_next 应变差(>0 才说明动作没被忽略)
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
from blocks.primitives import PreLNAttn

Y_MIN, Y_MAX = -24.0, 280.0


class LaneProbe(nn.Module):
    """结构化 per-lane 读出:N 个 slot [B,N,d] → 每轨道 (y_norm, present, hittable)。

    用 n_lanes 个可学 lane-query 对 slot 做 cross-attention(每条 lane 各自抽取相关 slot),
    取代会抹平 per-lane 结构、稀释单 lane 动作效果的 mean 池化。
    """
    def __init__(self, d, n_lanes=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_lanes, d) * 0.02)
        self.attn = PreLNAttn(d, heads=4, mode="cross")
        self.trunk = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.SiLU())
        self.y = nn.Linear(128, 1)
        self.present = nn.Linear(128, 1)
        self.hittable = nn.Linear(128, 1)

    def forward(self, slots):                          # slots: [B, N, d]
        q = self.queries.expand(slots.shape[0], -1, -1)   # [B, n_lanes, d]
        f = self.trunk(self.attn(q, slots))               # [B, n_lanes, 128]
        return {"y_norm": torch.sigmoid(self.y(f).squeeze(-1)),       # [B, n_lanes]
                "present": torch.sigmoid(self.present(f).squeeze(-1)),
                "hittable": torch.sigmoid(self.hittable(f).squeeze(-1))}


def lane_loss(pred, gt):
    pres = gt["present"]
    y_t = ((gt["y"] - Y_MIN) / (Y_MAX - Y_MIN)).clamp(0, 1)
    l_y = (F.smooth_l1_loss(pred["y_norm"], y_t, reduction="none") * pres).sum() / pres.sum().clamp(min=1)
    l_p = F.binary_cross_entropy(pred["present"], pres)
    l_h = F.binary_cross_entropy(pred["hittable"], gt["hittable"])   # 动作效果的关键监督
    return l_y * 5.0 + l_p + l_h * 3.0


def sample_actions(env, hit_p=0.85, near_p=0.30, rand_p=0.03, near_win=0.15):
    """按"真正可命中(±60ms)"的轨道高概率按下(保证频繁命中);近边界(±150ms 但不在窗内)
    中概率按(产生 miss 做对比);其余低概率随机。逼模型学 ±60ms 锐边界因果。"""
    ls = env.get_lane_state()
    hittable, onset = ls["hittable"], ls["onset"]
    near = (onset.abs() <= near_win) & (hittable < 0.5)
    p = torch.full_like(onset, rand_p)
    p = torch.where(near, torch.full_like(p, near_p), p)
    p = torch.where(hittable > 0.5, torch.full_like(p, hit_p), p)
    return (torch.rand_like(p) < p).float()


def make_inputs(B, model, device):
    Z = torch.zeros(B, model.N, model.d, device=device)
    h = torch.zeros(B, 1, model.d, device=device)
    t_hist = model.action_enc.net[0].in_features // model.n_keys
    a_raw = torch.zeros(B, t_hist, model.n_keys, device=device)
    g = (torch.zeros(B, device=device), torch.zeros(B, device=device),
         torch.ones(B, device=device) * 0.5)
    return Z, h, a_raw, g


def run(env, model, probe, B, dt, rollout, device, opt=None, shuffle_probe=False):
    env.reset()
    Z, h, a_raw, g = make_inputs(B, model, device)
    dt_vec = torch.ones(B, device=device)  # τ 以帧为单位:每 step=1 帧(见 ContinuousTimeEncoding;env.step 仍吃秒 dt)
    for _ in range(80):
        env.step(dt)                       # warmup 到稳态:慢音符也已落到判定线,管线灌满
    img = env.render()
    if opt is not None:
        opt.zero_grad()

    total = torch.zeros((), device=device)
    m = {k: torch.zeros((), device=device) for k in
         ["ph", "nh", "ps", "ns", "hitok", "lr", "ls", "accdec", "ndec"]}
    for _ in range(rollout):
        gt_now = env.get_lane_state()
        action = sample_actions(env)
        a_raw = torch.cat([a_raw[:, 1:], action.unsqueeze(1)], dim=1)   # 滚动追加当前动作
        out = model(img, Z, h, a_raw, dt_vec, g)
        loss_now = lane_loss(probe(out["Z_out"].mean(1)), gt_now)
        env.step(dt, action)
        img_next = env.render()
        gt_next = env.get_lane_state()
        pred_next = probe(out["mu"].mean(1))
        loss_next = lane_loss(pred_next, gt_next)
        # 聚焦损失:仅在"当前可命中"的 lane 上罚 next-hittable —— 那里按了→0、没按→还在,
        # 模型**必须用动作**才能预测对。破"易多数淹没动作信号"。
        hw = gt_now["hittable"]
        l_focus = (F.binary_cross_entropy(pred_next["hittable"], gt_next["hittable"],
                                          reduction="none") * hw).sum() / hw.sum().clamp(min=1)
        loss = loss_now + loss_next + 5.0 * l_focus
        if opt is not None:
            (loss / rollout).backward()
        total = total + loss.detach()

        with torch.no_grad():
            pred_h = probe(out["mu"].mean(1))["hittable"]            # [B,4] 预测下一帧 hittable(多音符下唯一干净信号)
            hit = env.last_hit                                       # 被命中(was hittable & pressed)
            stay = ((gt_now["hittable"] > 0.5) & (action < 0.5)).float()  # 在窗内但没按
            m["ph"] += (pred_h * hit).sum();   m["nh"] += hit.sum()
            m["ps"] += (pred_h * stay).sum();  m["ns"] += stay.sum()
            m["hitok"] += ((pred_h < 0.5).float() * hit).sum()
            # 决策点平衡准确率:仅在"当前可命中"lane(动作起作用处)上,阈值化预测 vs 真值
            dec = (gt_now["hittable"] > 0.5).float()
            correct = ((pred_h > 0.5) == (gt_next["hittable"] > 0.5)).float()
            m["accdec"] += (correct * dec).sum(); m["ndec"] += dec.sum()
            if shuffle_probe:
                perm = torch.randperm(B, device=device)
                out_s = model(img, Z, h, a_raw[perm], dt_vec, g)
                den = hw.sum().clamp(min=1)                          # 只在"当前可命中"lane 上测(去稀释)
                m["lr"] += (F.binary_cross_entropy(pred_h, gt_next["hittable"], reduction="none") * hw).sum() / den
                m["ls"] += (F.binary_cross_entropy(probe(out_s["mu"].mean(1))["hittable"], gt_next["hittable"], reduction="none") * hw).sum() / den

        Z = out["mu"].detach(); h = out["h_next"].detach()
        g = tuple(x.detach() for x in out["gaze"])
        img = img_next

    if opt is not None:
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0)
        opt.step()

    nh = m["nh"].clamp(min=1); ns = m["ns"].clamp(min=1)
    res = {
        "loss": (total / rollout).item(),
        "PredP_hit": (m["ph"] / nh).item(),       # → 0 好
        "PredP_stay": (m["ps"] / ns).item(),      # → 1 好
        "HitAcc": (m["hitok"] / nh).item(),       # → 1 好(命中事件预测消失,会被基率抬高)
        "DecAcc": (m["accdec"] / m["ndec"].clamp(min=1)).item(),   # 决策点平衡准确率(诚实)
        "hits_per_step": (m["nh"] / rollout).item(),
    }
    if shuffle_probe:
        lr = m["lr"].clamp(min=1e-6)
        res["shuffle_ratio"] = ((m["ls"] - m["lr"]) / lr).item()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--rollout", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--spawn", type=float, default=2.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"=== ACTION-CONDITIONED WORLD MODEL | device={dev} | ±60ms 命中消除 ===")
    env = RhythmActionEnv(batch_size=args.batch, device=dev, spawn_prob=args.spawn)
    model = TaoNot42Model(d=args.d, N=16, M=2, J=2, n_keys=4, t_hist=10,
                          layers=args.layers, K=6).to(dev)
    probe = LaneProbe(args.d).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(probe.parameters()), lr=3e-4)

    for ep in range(args.epochs):
        r = run(env, model, probe, args.batch, args.dt, args.rollout, dev, opt=opt)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d} | loss {r['loss']:5.3f} | PredHittable hit {r['PredP_hit']:.2f} "
                  f"stay {r['PredP_stay']:.2f} | Gap {r['PredP_stay']-r['PredP_hit']:+.2f} | "
                  f"HitAcc {r['HitAcc']:.2f} | hits/step {r['hits_per_step']:.1f}")

    model.eval(); probe.eval()
    with torch.no_grad():
        r = run(env, model, probe, args.batch, args.dt, args.rollout, dev, opt=None, shuffle_probe=True)
    print("\n===== VERDICT (action effect) =====")
    print(f"PredHittable | 命中(按掉) : {r['PredP_hit']:.3f}   (应→0:被打掉,下一刻不再可命中)")
    print(f"PredHittable | 窗内未按   : {r['PredP_stay']:.3f}   (应偏高:没被打,多半还在窗内)")
    print(f"判别 Gap (未按 - 按掉)    : {r['PredP_stay'] - r['PredP_hit']:+.3f}   (越大越说明学到'按→消除')")
    print(f"决策点准确率 DecAcc(诚实): {r['DecAcc']:.3f}   (仅在'当前可命中'lane 上,阈值化预测 vs 真值;随机基线≈0.5)")
    print(f"命中预测准确率 HitAcc    : {r['HitAcc']:.3f}   (命中事件预测'消失';会被低基率抬高,仅供参考)")
    print(f"动作打乱敏感度 ratio     : {r['shuffle_ratio']:+.3f}   (聚焦命中 lane;打乱动作→hittable 预测变差,>0.2 才算真用动作)")
    beat = (r['PredP_stay'] - r['PredP_hit']) > 0.4 and r['shuffle_ratio'] > 0.2
    print(f"=> {'✅ 模型学到了按键→消除的因果效果' if beat else '⚠ 动作效果尚不显著(欠训练/需调参)'}")


if __name__ == "__main__":
    main()
