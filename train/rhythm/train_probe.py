"""存/预测证明(加固版):证明 Z + transformer 真的在"存世界"且"预测未来"。

相比初版的三处加固
------------------
1. 随机逐音符速度(env 默认):音符各有自己的恒定速度,"加固定量"失效——
   模型必须从可见段推断该音符的速度,再带着它穿过盲区。这才是预测性推断。
2. 诚实基线 + skill score:不再只看绝对 MAE(会被"猜屏幕中位数"作弊)。
       persistence(冻结最后可见 y) = 0 分下界
       oracle(用真速度匀速外推)    = 满分上界(≈0 误差)
       skill = 1 − MAE_model / MAE_persistence   (>0 才算在预测)
3. 开环视野曲线:全屏涂黑 K 步,画 error(K)。温和增长=真前向模拟;立刻发散=假的。

用法(Colab)
-----------
    !python train_probe.py --epochs 400 --device cuda
    !python train_probe.py --epochs 400 --device cuda --no_memory   # 消融对照
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from net.tao_not_42 import TaoNot42Model
from net.world_probe import WorldProbeDecoder
from train.rhythm.rhythm_env import ProceduralRhythmEnv

Y_MIN, Y_MAX = -24.0, 280.0


def world_loss(pred, state, valid=None):
    active = state["active"] if valid is None else valid
    n = active.sum().clamp(min=1.0)
    y_t = ((state["y"] - Y_MIN) / (Y_MAX - Y_MIN)).clamp(0, 1)
    l_y = (F.smooth_l1_loss(pred["y_norm"], y_t, reduction="none") * active).sum() / n
    l_trk = (F.cross_entropy(pred["track_logits"], state["track"], reduction="none") * active).sum() / n
    l_clr = (F.cross_entropy(pred["color_logits"], state["color"], reduction="none") * active).sum() / n
    l_exist = F.binary_cross_entropy(pred["exist"], state["active"])
    return l_y * 5.0 + l_trk + l_clr + l_exist


def make_inputs(B, model, device):
    Z = torch.zeros(B, model.N, model.d, device=device)
    h = torch.zeros(B, 1, model.d, device=device)
    t_hist = model.action_enc.net[0].in_features // model.n_keys
    a_raw = torch.zeros(B, t_hist, model.n_keys, device=device)
    g = (torch.zeros(B, device=device), torch.zeros(B, device=device),
         torch.ones(B, device=device) * 0.5)
    return Z, h, a_raw, g


def train_rollout(env, model, probe, B, dt, rollout_len, device, memory, opt):
    """逐步 backward:显存 O(1) 不随 rollout 长度涨(递归 carry 本就 detach,每步图独立)。"""
    env.reset()
    Z, h, a_raw, g = make_inputs(B, model, device)
    dt_vec = torch.ones(B, device=device)  # τ 以帧为单位(ContinuousTimeEncoding 契约,传秒会退化);env.step 仍吃秒 dt
    env.step(dt); img = env.render()
    total = torch.zeros((), device=device)
    opt.zero_grad()
    for _ in range(rollout_len):
        state_now = env.get_tracer_state()
        out = model(img, Z, h, a_raw, dt_vec, g)
        loss_p = world_loss(probe(out["Z_out"][:, 0]), state_now)
        env.step(dt); img_next = env.render(); state_next = env.get_tracer_state()
        same = (state_now["active"] * state_next["active"] *
                (state_next["y"] > state_now["y"] - 1.0).float())
        loss_f = world_loss(probe(out["mu"][:, 0]), state_next, valid=same)
        step_loss = loss_p + 0.5 * loss_f
        (step_loss / rollout_len).backward()    # 每步反传并释放图
        total = total + step_loss.detach()
        Z = out["mu"].detach() if memory else torch.randn_like(Z) * 0.02
        h = out["h_next"].detach()
        g = tuple(x.detach() for x in out["gaze"])
        img = img_next
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(probe.parameters()), 1.0)
    opt.step()
    return (total / rollout_len).item()


@torch.no_grad()
def evaluate(env, model, probe, B, dt, rollout_len, device, memory):
    """对比 model / persistence / oracle,按遮挡-可见拆分,出 skill。收集 b0 轨迹画图。"""
    env.reset()
    Z, h, a_raw, g = make_inputs(B, model, device)
    dt_vec = torch.ones(B, device=device)  # τ 以帧为单位(契约同 train_rollout)
    env.step(dt); img = env.render()

    last_y = env.get_tracer_state()["y"].clone()   # 最后一次可见的 y
    last_t = torch.zeros(B, device=device)

    acc = {k: torch.zeros((), device=device) for k in
           ["mo", "po", "oo", "no", "mv", "pv", "ov", "nv"]}  # model/pers/oracle × occ/vis + 计数
    traj = {"true": [], "model": [], "pers": [], "oracle": [], "occ": []}

    for t in range(rollout_len):
        s = env.get_tracer_state()
        out = model(img, Z, h, a_raw, dt_vec, g)
        pred_y = probe(out["Z_out"][:, 0])["y_px"]

        pers = last_y
        oracle = last_y + s["speed"] * (t - last_t) * dt
        respawn = (s["active"] > 0.5) & (s["y"] < last_y - 5.0)  # 换了新音符,基线失效
        act = (s["active"] > 0.5) & (~respawn)
        occ = s["occluded"] & act
        vis = (~s["occluded"]) & act

        em = (pred_y - s["y"]).abs()
        ep = (pers - s["y"]).abs()
        eo = (oracle - s["y"]).abs()
        acc["mo"] += (em * occ).sum(); acc["po"] += (ep * occ).sum()
        acc["oo"] += (eo * occ).sum(); acc["no"] += occ.float().sum()
        acc["mv"] += (em * vis).sum(); acc["pv"] += (ep * vis).sum()
        acc["ov"] += (eo * vis).sum(); acc["nv"] += vis.float().sum()

        traj["true"].append(s["y"][0].item()); traj["model"].append(pred_y[0].item())
        traj["pers"].append(pers[0].item()); traj["oracle"].append(oracle[0].item())
        traj["occ"].append(bool(occ[0].item()))

        # 可见(或刚换音符)时刷新基线锚点
        fresh = ((s["active"] > 0.5) & (~s["occluded"])) | respawn
        last_y = torch.where(fresh, s["y"], last_y)
        last_t = torch.where(fresh, torch.full_like(last_t, t), last_t)

        env.step(dt); img = env.render()
        Z = out["mu"].detach() if memory else torch.randn_like(Z) * 0.02
        h = out["h_next"].detach()
        g = tuple(x.detach() for x in out["gaze"])

    no = acc["no"].clamp(min=1); nv = acc["nv"].clamp(min=1)
    r = {
        "occ_model": (acc["mo"] / no).item(), "occ_pers": (acc["po"] / no).item(),
        "occ_oracle": (acc["oo"] / no).item(),
        "vis_model": (acc["mv"] / nv).item(), "vis_pers": (acc["pv"] / nv).item(),
    }
    r["skill_occ"] = 1.0 - r["occ_model"] / max(r["occ_pers"], 1e-6)
    return r, traj


@torch.no_grad()
def eval_horizon(env, model, probe, B, dt, device, warmup=6, horizon=30):
    """开环视野曲线:可见 warmup 让 Z 锁定音符后,全屏涂黑,逐步记录 |pred_y - true_y| vs 视野。"""
    env.reset()
    Z, h, a_raw, g = make_inputs(B, model, device)
    dt_vec = torch.ones(B, device=device)  # τ 以帧为单位(契约同 train_rollout)
    for _ in range(warmup):
        env.step(dt); img = env.render()
        out = model(img, Z, h, a_raw, dt_vec, g)
        Z = out["mu"].detach(); h = out["h_next"].detach()
        g = tuple(x.detach() for x in out["gaze"])

    y0 = env.get_tracer_state()["y"].clone()
    black = torch.full_like(img, 0.05)   # 全屏涂黑,断绝一切观测
    errs = []
    for k in range(horizon):
        s = env.get_tracer_state()
        out = model(black, Z, h, a_raw, dt_vec, g)
        pred_y = probe(out["Z_out"][:, 0])["y_px"]
        valid = (s["active"] > 0.5) & (s["y"] >= y0 - 5.0)   # 仍是同一个、下落中的音符
        e = ((pred_y - s["y"]).abs() * valid.float()).sum() / valid.float().sum().clamp(min=1)
        errs.append(e.item())
        env.step(dt); img = env.render()
        Z = out["mu"].detach(); h = out["h_next"].detach()
        g = tuple(x.detach() for x in out["gaze"])
    return list(range(1, horizon + 1)), errs


def plot_all(traj, horizons, herrs, hit_line, fname):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped: {e}]"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    steps = list(range(len(traj["true"])))
    ax1.plot(steps, traj["true"], "k-", lw=2.5, label="true y")
    ax1.plot(steps, traj["model"], "r--", lw=2, label="model (from Z)")
    ax1.plot(steps, traj["pers"], "g:", lw=1.5, label="persistence (freeze)")
    ax1.plot(steps, traj["oracle"], "b-.", lw=1, label="oracle (true speed)")
    shaded = False
    in_occ = False
    for i, o in enumerate(traj["occ"] + [False]):
        if o and not in_occ:
            start = i; in_occ = True
        if not o and in_occ:
            ax1.axvspan(start, i, color="gray", alpha=0.22,
                        label=None if shaded else "OCCLUDED"); shaded = True; in_occ = False
    ax1.axhline(hit_line, color="orange", ls=":", label="hit line")
    ax1.set_xlabel("rollout step"); ax1.set_ylabel("note y (px)")
    ax1.set_title("Store + predict through occlusion (random velocity)")
    ax1.legend(fontsize=8); ax1.invert_yaxis()

    ax2.plot(horizons, herrs, "r.-", lw=2)
    ax2.set_xlabel("open-loop horizon (steps, full blackout)")
    ax2.set_ylabel("|pred y - true y| (px)")
    ax2.set_title("Prediction error vs eyes-closed horizon")
    fig.tight_layout(); fig.savefig(fname, dpi=110)
    print(f"[saved plot -> {fname}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_memory", action="store_true", help="消融:每帧重置 Z")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    memory = not args.no_memory
    print(f"=== {'NO-MEMORY (ablation)' if not memory else 'MEMORY (full)'} | "
          f"device={device} | random-velocity tracer ===")

    env = ProceduralRhythmEnv(batch_size=args.batch, device=device, tracer_mode=True)
    model = TaoNot42Model(d=args.d, N=16, M=2, J=2, n_keys=4, t_hist=10,
                          layers=args.layers, K=6).to(device)
    probe = WorldProbeDecoder(d=args.d, y_min=Y_MIN, y_max=Y_MAX).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(probe.parameters()), lr=3e-4)

    for ep in range(args.epochs):
        loss = train_rollout(env, model, probe, args.batch, args.dt,
                             args.rollout, device, memory, opt)
        if ep % 25 == 0 or ep == args.epochs - 1:
            r, _ = evaluate(env, model, probe, args.batch, args.dt,
                            args.rollout, device, memory)
            print(f"ep {ep:4d} | loss {loss:6.3f} | vis {r['vis_model']:5.1f}px | "
                  f"occ: model {r['occ_model']:5.1f} / pers {r['occ_pers']:5.1f} / "
                  f"oracle {r['occ_oracle']:4.1f} | skill {r['skill_occ']:+.2f}")

    model.eval(); probe.eval()
    r, traj = evaluate(env, model, probe, args.batch, args.dt, args.rollout, device, memory)
    horizons, herrs = eval_horizon(env, model, probe, args.batch, args.dt, device)

    print("\n===== VERDICT (random velocity) =====")
    print(f"visible    MAE: model {r['vis_model']:6.2f}px  (pers {r['vis_pers']:.2f})")
    print(f"OCCLUDED   MAE: model {r['occ_model']:6.2f}px | persistence {r['occ_pers']:6.2f} | "
          f"oracle {r['occ_oracle']:.2f}")
    print(f"SKILL (occ)   : {r['skill_occ']:+.3f}   "
          f"(>0 ⇒ 击败'冻结'、确在预测; ≤0 ⇒ 没在预测)")
    print(f"open-loop err : {herrs[0]:.1f}px @1step → {herrs[len(herrs)//2]:.1f}px "
          f"@{horizons[len(herrs)//2]} → {herrs[-1]:.1f}px @{horizons[-1]} steps")
    fname = "world_probe_nomem.png" if not memory else "world_probe.png"
    plot_all(traj, horizons, herrs, env.hit_line_y, fname)


if __name__ == "__main__":
    main()
