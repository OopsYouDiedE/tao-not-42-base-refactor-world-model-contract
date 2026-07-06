#!/usr/bin/env python3
"""C2 采矿闸门:在 C2 房间 rollout 快头 BC,测执行接口是否表达得了"挖铁+捡起"技能。

统一入口,两种口径(--cond 决定):
  无 --cond(默认,原 gate_c2):无条件 BCPolicy 单轮 rollout。预登记判据:易起点
      N 局中 score>0(挖到并捡到铁)占比 ≥15% = 执行接口 PASS;恒 0 = FAIL。
  有 --cond(原 gate_c2_cond,命题③):return-conditioned CondPolicy,对 --cmds 里每个
      命令回报各跑一轮,看行为是否随命令可控变化。判据:命令高回报 score>0 率 −
      命令低回报 ≥ +0.15 = "用信号能操纵执行"成立。

环境与 collect_s8 同款(superflat 定 seed + raycast + C2 参数化课程),episode 轮换
WALL_Z_VARIANTS 易起点;动作解码同 gate_fasthead(bin 采样→V2)。

用法(CPU 渲染 + GPU dinov3):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. python tests/integration/test_c2_gate.py \
      --ckpt runs/ftt_c2bc/best.pt --episodes 20 --steps 220 --out runs/ftt_c2bc_gate.json
  条件版:... test_c2_gate.py --cond --ckpt runs/ftt_c2cond/final.pt --cmds 2.0 0.0
"""
import argparse
import json
import os

import numpy as np
import torch

from net.bc import BCConfig, CondPolicy, build_bc_policy
from net.config import BackboneConfig
from tests.integration.collect_s8 import (WALL_Z_VARIANTS, build_c2_course,
                                          score_c2, _mined_iron)
from tests.integration.test_utils import crop128
from train.fovea_twotower.gate_fasthead import decode_action
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE


def build_env(port):
    """C2 环境(superflat 定 seed + raycast + iron_ore 统计)。返回 (env, noop)。"""
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        mined_stat_keys=["iron_ore"], initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=port, verbose=False)
    return env, no_op_v2()


@torch.no_grad()
def run_cell(policy, env, noop, episodes, steps, max_len, settle, device, rng,
             cmd_ret=None, ret_scale=2.0, greedy=False):
    """跑 episodes 局,返回每局 {ep,wall_z,score,mined_cum}。

    cmd_ret 非空 = 条件策略,命令目标回报 cmd_ret(归一化 /ret_scale)喂 CondPolicy。
    """
    ret_t = (torch.tensor([cmd_ret / ret_scale], device=device).float()
             if cmd_ret is not None else None)
    out = []
    for ep in range(episodes):
        wall_z = WALL_Z_VARIANTS[ep % len(WALL_Z_VARIANTS)]
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_c2_course(wall_z, pickaxe=True)})
        for _ in range(settle):
            obs, *_ = env.step(noop)
        feats_hist, act_hist = [], []
        prev_vec = np.zeros(ACTION_DIM, np.float32)
        for _t in range(steps):
            f = policy.encode_frames(crop128(obs["rgb"]).to(device))[:, 0]
            feats_hist.append(f)
            act_hist.append(torch.from_numpy(prev_vec).to(device).view(1, -1))
            fseq = torch.stack(feats_hist[-max_len:], 1)
            aseq = torch.stack(act_hist[-max_len:], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                if ret_t is not None:
                    cam_logits, key_logits = policy(fseq.float(), aseq.float(), ret_t)
                else:
                    cam_logits, key_logits = policy(fseq.float(), aseq.float())
            a, key_on = decode_action(cam_logits[0, -1], key_logits[0, -1],
                                      noop, greedy, rng)
            prev_vec = np.zeros(ACTION_DIM, np.float32)
            prev_vec[N_MOUSE:] = key_on              # 相机 prev 置 0(键主导记忆)
            obs, *_ = env.step(a)
        out.append({"ep": ep, "wall_z": wall_z, "score": score_c2(obs["full"]),
                    "mined_cum": _mined_iron(obs["full"])})
        print(f"[gate_c2] ep{ep} wall_z={wall_z} score={out[-1]['score']} "
              f"mined={out[-1]['mined_cum']}", flush=True)
    return out


def _pickup_rate(rows):
    return float((np.array([r["score"] for r in rows]) > 0).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ftt_c2bc/best.pt")
    p.add_argument("--cond", action="store_true", default=False,
                   help="return-conditioned CondPolicy:对 --cmds 每个命令回报各跑一轮")
    p.add_argument("--cmds", type=float, nargs="+", default=[2.0, 0.0],
                   help="仅 --cond:命令的目标回报(高在前);steer_delta=高−低 的 score>0 率差")
    p.add_argument("--episodes", type=int, default=20, help="(条件版=每个命令的局数)")
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--greedy", action="store_true", default=False)
    p.add_argument("--port", type=int, default=8565)
    p.add_argument("--out", default="runs/ftt_c2bc_gate.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cs = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=cs.get("d", 384),
                   heads=cs.get("heads", 6), layers=cs.get("layers", 4), dropout=0.0,
                   max_len=max(128, args.max_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = (CondPolicy(cfg) if args.cond else build_bc_policy(cfg)).to(device).eval()
    missing, unexpected = policy.load_state_dict(ck["policy"], strict=False)
    assert not [m for m in missing if not m.startswith("backbone.")], missing[:6]
    assert not unexpected, unexpected[:6]
    ret_scale = cs.get("cond_scale") or ck.get("ret_scale", 2.0)
    print(f"✅ 载入 {'条件化 ' if args.cond else ''}{args.ckpt}(step={ck.get('step')})",
          flush=True)

    env, noop = build_env(args.port)
    env.reset()
    rng = np.random.default_rng(args.seed)
    res = {"ckpt": args.ckpt, "ckpt_step": ck.get("step"), "steps": args.steps}

    if not args.cond:
        rows = run_cell(policy, env, noop, args.episodes, args.steps, args.max_len,
                        args.settle, device, rng, greedy=args.greedy)
        env.close()
        rate = _pickup_rate(rows)
        mined_per_ep = np.diff(np.concatenate(
            [[0], [r["mined_cum"] for r in rows]]))
        res.update(episodes=args.episodes, greedy=args.greedy,
                   pickup_rate=round(rate, 4),
                   mine_rate=round(float((mined_per_ep > 0).mean()), 4),
                   mean_score=round(float(np.mean([r["score"] for r in rows])), 3),
                   per_episode=rows,
                   verdict_interface=f"{'PASS' if rate >= 0.15 else 'FAIL'} "
                   f"(score>0 率={rate:.3f} 门 0.15;老师 0.42)")
        summary = {k: res[k] for k in ("pickup_rate", "mine_rate", "mean_score",
                                       "verdict_interface")}
    else:
        res["by_command"] = {}
        for cmd in args.cmds:
            rows = run_cell(policy, env, noop, args.episodes, args.steps, args.max_len,
                            args.settle, device, rng, cmd_ret=cmd, ret_scale=ret_scale)
            scores = np.array([r["score"] for r in rows])
            res["by_command"][f"{cmd}"] = {
                "pickup_rate": round(_pickup_rate(rows), 4),
                "mean_score": round(float(scores.mean()), 3),
                "per_ep_score": scores.tolist()}
            print(f"[cond] 命令回报={cmd}: score>0率={_pickup_rate(rows):.3f} "
                  f"均分={scores.mean():.2f}", flush=True)
        env.close()
        hi, lo = str(args.cmds[0]), str(args.cmds[-1])
        delta = (res["by_command"][hi]["pickup_rate"]
                 - res["by_command"][lo]["pickup_rate"])
        res["steer_delta"] = round(delta, 4)
        res["verdict_cond"] = (f"{'PASS' if delta >= 0.15 else 'FAIL'} "
                               f"(命令高{hi}−低{lo} 的 score>0率差={delta:+.3f} 门 +0.15)")
        summary = {"steer_delta": res["steer_delta"], "verdict": res["verdict_cond"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
