# -*- coding: utf-8 -*-
"""喂"教师 Mamba belief latent" vs "完成后的单帧向量":学生要学的内容是否减少/效果提升。

问题(用户):给学生塞慢塔(ContextTower/Mamba)的 belief latent,而不是完成后的向量,能否
减少学生自己要学的东西或提升效果?承 Step1 教训(播种通道可用但世界记忆真空),**必须带证伪对照**。

三条输入,同一小学生(逐帧,无自身时序 → 任何历史依赖只能来自输入):
  A  finished 单帧向量 = 该帧 81 个 DINO patch token 均值(仅当前帧,无历史)
  B  Mamba latent      = ContextTower.encode 后该帧动作位的 belief hidden(单帧 + Mamba 历史)
  Bshuf                = B 沿时间轴打乱(破坏 belief↔目标 的时序对齐)= 证伪对照

预登记判据:
  M2 减少学生要学的 = 固定小容量下 holdout 动作预测 R²(B) − R²(A);B 显著高 = 世界建模被 latent
     offload。同看 camera 转向方向 acc、按键 F1。
  M3 证伪(防真空)  = R²(B) − R²(Bshuf);若 ≈0,B 的收益不是"世界记忆内容"而是白喂维度/更好编码器。
  (M1 闭环成功率为终审,须活环境 + 慢塔在环,后续接;本探针是快速筛选,非终审。)

用法:PYTHONPATH=. ./.venv/bin/python tests/probe_mamba_seed.py --ckpt runs/ftt_w1/ckpt.pt --n 120
"""
import argparse
import glob
import json

import numpy as np
import torch

from net.fovea_twotower import ContextTower
from train.gaming500.dataset import N_MSG

P = 81 + 1 + 1                                          # 帧块周期(vis|msg|act)


@torch.no_grad()
def encode_clip(model, lat, act, msg, dev):
    """lat[L,81,384] act[L-1,24] msg[L,11] → (A[L-1,384], B[L-1,d], target act[L-1,24])。"""
    L = lat.shape[0]
    h, _ = model.encode(lat[None].to(dev), act[None].to(dev), msg[None].to(dev))
    # 消息位(t*83+81):看过帧 t 视觉 + 历史(含过去动作),但**未**看过 a_t → 无泄漏,是真"策略 belief"
    idx = torch.arange(L - 1, device=dev) * P + 81
    B = h[0, idx].float().cpu()                          # [L-1, d]
    A = lat[:L - 1].float().mean(1)                      # [L-1, 384] 单帧 pooled patch(无历史无动作)
    return A, B, act[:L - 1].float()


def fit_r2(Xtr, Ytr, Xte, Yte, hidden=0, dev="cuda", steps=400):
    """小学生(linear 或 1 隐层 MLP)回归 Y;返回 (整体 R², cam R², key R²)。"""
    din, dout = Xtr.shape[1], Ytr.shape[1]
    if hidden:
        net = torch.nn.Sequential(torch.nn.Linear(din, hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, dout)).to(dev)
    else:
        net = torch.nn.Linear(din, dout).to(dev)
    Xtr, Ytr, Xte, Yte = [t.to(dev) for t in (Xtr, Ytr, Xte, Yte)]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(); loss = ((net(Xtr) - Ytr) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xte)
    def r2(sl):
        se = ((pred[:, sl] - Yte[:, sl]) ** 2).sum()
        sst = ((Yte[:, sl] - Yte[:, sl].mean(0)) ** 2).sum()
        return float(1 - se / sst.clamp(min=1e-6))
    return r2(slice(None)), r2(slice(0, 2)), r2(slice(2, 22))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ftt_w1/ckpt.pt")
    p.add_argument("--data", default="runs/data/s8_traj_full")
    p.add_argument("--n", type=int, default=120, help="用多少条轨迹")
    p.add_argument("--hidden", type=int, default=0, help="学生隐层(0=纯线性,压容量测'少学')")
    p.add_argument("--out", default="runs/probe_mamba_seed.json")
    args = p.parse_args()
    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev)
    model = ContextTower(n_msg=N_MSG, aux_msg=ck.get("args", {}).get("aux_msg", 0.0)
                         ).to(dev).bfloat16().eval()
    model.load_state_dict(ck["model"])

    files = sorted(glob.glob(f"{args.data}/*.npz"))[:args.n]
    As, Bs, Ys = [], [], []
    for f in files:
        z = np.load(f)
        lat = torch.from_numpy(z["lat"].astype(np.float32)).bfloat16()
        act = torch.from_numpy(z["act"].astype(np.float32)).bfloat16()
        msg = torch.from_numpy(z["msg"].astype(np.float32)).bfloat16()
        A, B, Y = encode_clip(model, lat, act, msg, dev)
        As.append(A); Bs.append(B); Ys.append(Y)
    n_tr = int(len(files) * 0.7)
    cat = lambda xs: torch.cat(xs)
    Atr, Ate = cat(As[:n_tr]), cat(As[n_tr:])
    Btr, Bte = cat(Bs[:n_tr]), cat(Bs[n_tr:])
    Ytr, Yte = cat(Ys[:n_tr]), cat(Ys[n_tr:])
    # Bshuf:训练/测试各自沿时间打乱行(破坏 belief↔目标 对齐)
    g = torch.Generator().manual_seed(0)
    Bstr = Btr[torch.randperm(len(Btr), generator=g)]
    Bste = Bte[torch.randperm(len(Bte), generator=g)]
    print(f"[probe] n_clip={len(files)} tr={len(Atr)} te={len(Ate)} hidden={args.hidden}", flush=True)

    res = {"ckpt": args.ckpt, "n_clip": len(files), "n_tr": len(Atr), "n_te": len(Ate),
           "hidden": args.hidden}
    for name, Xtr, Xte in [("A_finished", Atr, Ate), ("B_mamba", Btr, Bte),
                           ("Bshuf", Bstr, Bste), ("AB", torch.cat([Atr, Btr], 1),
                                                   torch.cat([Ate, Bte], 1))]:
        r2, r2c, r2k = fit_r2(Xtr, Ytr, Xte, Yte, args.hidden, dev)
        res[name] = {"r2_all": round(r2, 4), "r2_cam": round(r2c, 4), "r2_key": round(r2k, 4)}
        print(f"  {name:12s} R²_all={r2:.4f} cam={r2c:.4f} key={r2k:.4f}", flush=True)
    m2 = res["B_mamba"]["r2_all"] - res["A_finished"]["r2_all"]
    m3 = res["B_mamba"]["r2_all"] - res["Bshuf"]["r2_all"]
    res["verdict_M2_reduce"] = f"{'提升' if m2 > 0.02 else '无益'} ΔR²(B-A)={m2:+.4f}"
    res["verdict_M3_falsify"] = f"{'内容有用' if m3 > 0.02 else '真空(收益非时序内容)'} ΔR²(B-Bshuf)={m3:+.4f}"
    print("[probe]", json.dumps({k: res[k] for k in ("verdict_M2_reduce", "verdict_M3_falsify")},
                                ensure_ascii=False), flush=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
