"""动作计划训练:一步预测多个带时长的定时动作(K 个 DETR 式动作查询)。

每个 forward,模型在多音符场景里一次性吐出一串"接下来要打的击打":
    (按哪条轨道, 多久后按下 onset, 按住多久 duration, 这一槽是否真有 exist)
GT 来自 env.get_upcoming_actions(K)(含长按音符的真实时长),用 Sinkhorn 集合匹配监督。

诚实标尺
--------
模型必须显著优于"朴素常数基线"(onset 全预测成均值、key 瞎猜)才算学会规划:
    OnsetMAE ≪ naive_onset_MAE,  KeyAcc ≫ 0.25(随机),  ExistOn↑1 / ExistOff↓0。

用法(Colab)
-----------
    !python train_action.py --epochs 400 --device cuda
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from net.tao_not_42 import TaoNot42Model
from utils.rhythm_env import ProceduralRhythmEnv
from utils.losses import action_plan_loss


def make_inputs(B, model, device):
    Z = torch.zeros(B, model.N, model.d, device=device)
    h = torch.zeros(B, 1, model.d, device=device)
    t_hist = model.action_enc.net[0].in_features // model.n_keys
    a_raw = torch.zeros(B, t_hist, model.n_keys, device=device)
    g = (torch.zeros(B, device=device), torch.zeros(B, device=device),
         torch.ones(B, device=device) * 0.5)
    return Z, h, a_raw, g


def naive_onset_mae(gt):
    """朴素基线:把每个 batch 的 onset 全预测成该 batch 有效 onset 的均值。返回 0 维 tensor。"""
    on, val = gt["onset"], gt["valid"]
    n = val.sum(dim=1).clamp(min=1)
    mean_on = (on * val).sum(dim=1) / n                    # [B]
    mae = ((on - mean_on.unsqueeze(1)).abs() * val).sum() / val.sum().clamp(min=1)
    return 1000.0 * mae


def run_rollout(env, model, K, B, dt, rollout_len, device, opt=None):
    """逐步 backward:显存 O(1) 不随 rollout 长度涨;指标累积为 tensor,热路径零同步。"""
    env.reset()
    Z, h, a_raw, g = make_inputs(B, model, device)
    dt_vec = torch.ones(B, device=device)  # τ 以帧为单位(ContinuousTimeEncoding 契约,传秒会退化);env.step 仍吃秒 dt
    env.step(dt); img = env.render()
    if opt is not None:
        opt.zero_grad()

    total = torch.zeros((), device=device)
    agg = {k: torch.zeros((), device=device)
           for k in ["OnsetMAEms", "DurMAEms", "KeyAcc", "ExistOn", "ExistOff", "naive"]}
    for _ in range(rollout_len):
        gt = env.get_upcoming_actions(K)
        out = model(img, Z, h, a_raw, dt_vec, g)
        loss, met = action_plan_loss(out["action_plan"], gt)
        if opt is not None:
            (loss / rollout_len).backward()   # 每步反传并释放图 ⇒ 显存不累积
        total = total + loss.detach()
        for k in ["OnsetMAEms", "DurMAEms", "KeyAcc", "ExistOn", "ExistOff"]:
            agg[k] = agg[k] + met[k]
        agg["naive"] = agg["naive"] + naive_onset_mae(gt)
        # 世界模型仍在后台滚动(动作计划以内部世界为条件)
        Z = out["mu"].detach(); h = out["h_next"].detach()
        g = tuple(x.detach() for x in out["gaze"])
        env.step(dt); img = env.render()

    if opt is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    agg = {k: v / rollout_len for k, v in agg.items()}
    return total / rollout_len, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--spawn", type=float, default=3.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    print(f"=== ACTION PLAN (K={args.K}, 多音符+长按) | device={device} ===")

    env = ProceduralRhythmEnv(batch_size=args.batch, device=device,
                              tracer_mode=False, spawn_prob=args.spawn)
    model = TaoNot42Model(d=args.d, N=16, M=2, J=2, n_keys=4, t_hist=10,
                          layers=args.layers, K=args.K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for ep in range(args.epochs):
        loss, agg = run_rollout(env, model, args.K, args.batch, args.dt,
                                args.rollout, device, opt=opt)
        if ep % 25 == 0 or ep == args.epochs - 1:
            a = {k: float(v) for k, v in agg.items()}
            print(f"ep {ep:4d} | loss {float(loss):6.3f} | onset {a['OnsetMAEms']:6.1f}ms "
                  f"(naive {a['naive']:6.1f}) | dur {a['DurMAEms']:6.1f}ms | "
                  f"key {a['KeyAcc']:.2f} | exist {a['ExistOn']:.2f}/{a['ExistOff']:.2f}")

    model.eval()
    with torch.no_grad():
        _, agg = run_rollout(env, model, args.K, args.batch, args.dt,
                             args.rollout, device, opt=None)
    a = {k: float(v) for k, v in agg.items()}
    print("\n===== VERDICT (action plan) =====")
    print(f"onset MAE : {a['OnsetMAEms']:6.1f} ms   (naive constant baseline {a['naive']:.1f} ms)")
    print(f"duration  : {a['DurMAEms']:6.1f} ms")
    print(f"key acc   : {a['KeyAcc']:.3f}   (random = 0.25)")
    print(f"exist     : on {a['ExistOn']:.2f} / off {a['ExistOff']:.2f}  (应分离 →1 / →0)")
    beat = a["OnsetMAEms"] < 0.6 * a["naive"] and a["KeyAcc"] > 0.5
    print(f"=> {'✅ 显著优于朴素基线,确在规划带时长动作' if beat else '⚠ 尚未明显击败基线(欠训练/需调参)'}")


if __name__ == "__main__":
    main()
