#!/usr/bin/env python3
"""C1b 全回路三步链原型:慢脑计划 → 快头执行 → 慢脑核查 → 注册表(视觉标记)回家。

集群就绪判据 C1 的系统闸门(docs/architectures/fovea-brain-division-scale-plan.md §4.5,
判据先登记:10 局成功率 ≥0.5)。链条:
  ① 慢脑(Qwen1.5B + runs/reason_delta_lora_v3)读任务"获得生铁"+库存(石镐)+配方卡
     → 差额计划 → 首个可执行项映射为感知类(raw_iron→iron_ore)→ 发指令;
  ② 快头(trackcmd 学生)goal 相对 token 追踪/逼近铁矿;**挖掘宏**(对准且近时锁相机
     持续攻击+吸拾)——宏=待学习技能的脚本占位(同 GUI 宏定位),诚实边界;
  ③ 慢脑用新库存复核("已齐备")→ 发"回家"指令;
  ④ 回家=指令切到 home 视觉标记类(出生点旁泥土柱),快头照常追踪导航——
     全程观测一致,位姿只用于评测(距出发点 ≤3 格判成功)。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. .venv/bin/python \
      tests/integration/fullloop_chain.py --episodes 10 --ckpt runs/trackcmd_bc_v15/best.pt
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

from tests.integration.collect_calib640 import _pose, _ray
from tests.integration.collect_track_cmd import CLASSES, TokenHead
from train.fovea_twotower.train_track_cmd import goal_relative

ITEM2CLS = {"raw_iron": "iron_ore", "生铁": "iron_ore"}
HOME_CLS = "dirt"                  # 出生点旁泥土柱=家的视觉标记


def build_course(wall_z=7):
    """铁矿十字+煤干扰上墙;泥土柱在出生点侧后方(墙上不放泥土,家标记唯一)。"""
    cmds = [
        "gamemode survival @p",
        "difficulty peaceful",
        "tp @p ~ ~ ~ 0 0",
        f"fill ~-4 ~-1 ~-4 ~4 ~4 ~{wall_z} minecraft:air",
        f"fill ~-4 ~-2 ~-4 ~4 ~-2 ~{wall_z} minecraft:stone",
        f"fill ~-4 ~-1 ~{wall_z} ~4 ~4 ~{wall_z} minecraft:stone",
        f"setblock ~0 ~ ~{wall_z} minecraft:iron_ore",
        f"setblock ~1 ~ ~{wall_z} minecraft:iron_ore",
        f"setblock ~0 ~1 ~{wall_z} minecraft:iron_ore",
        f"setblock ~-2 ~ ~{wall_z} minecraft:coal_ore",
        "setblock ~2 ~ ~-3 minecraft:dirt",            # 家标记(视线扫过可见)
        "setblock ~2 ~1 ~-3 minecraft:dirt",
        "setblock ~1 ~ ~-3 minecraft:dirt",
        "clear @p",
        "item replace entity @p weapon.mainhand with minecraft:stone_pickaxe 1",
    ]
    return cmds


class SlowBrain:
    """E2 产物:差额规划 + 库存复核。"""

    def __init__(self, adapter="runs/reason_delta_lora_v3",
                 base="Qwen/Qwen2.5-1.5B-Instruct", dev="cuda"):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from train.fovea_twotower.reason_delta_sft import prompt
        self.prompt = prompt
        self.tok = AutoTokenizer.from_pretrained(base)
        m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(m, adapter).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def plan(self, goal, inv):
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": self.prompt(goal, frozenset(inv), card=True)}],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        out = self.model.generate(enc["input_ids"].to(self.dev), max_new_tokens=384,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        body = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        steps = re.findall(r"\d+\.\s*(?:获得)?([\w一-鿿]+)", body)
        return steps, ("已齐备" in body), text


def env_inventory(full):
    inv = set()
    try:
        for it in full.inventory:
            if getattr(it, "count", 0) > 0 and it.translation_key:
                inv.add(it.translation_key.split(".")[-1])
    except Exception:  # noqa
        pass
    return inv


def run(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from train.fovea_twotower.eval_track_cmd import StudentPolicy

    tok_head = TokenHead(args.vectors, conv_head=args.conv_head)
    student = StudentPolicy(args.ckpt)
    brain = SlowBrain(args.adapter)

    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()

    results = []
    for ep in range(args.episodes):
        t0 = time.time()
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_course()})
        for _ in range(10):
            obs, *_ = env.step(noop)
        full = obs["full"]
        start = np.array([full.x, full.z])
        inv0 = env_inventory(full)

        # ① 慢脑计划
        steps, done0, _ = brain.plan("raw_iron", inv0 & {"stone_pickaxe", "raw_iron"})
        goal_item = steps[0] if steps else ""
        goal_cls = ITEM2CLS.get(goal_item, "")
        log = {"plan": steps, "goal_cls": goal_cls}
        if goal_cls != "iron_ore":
            results.append(dict(ok=False, why="plan", **log))
            print(f"[ep{ep}] ✗ 计划失败 {steps}", flush=True)
            continue

        # ② 执行:追踪逼近(学生) + 挖掘宏
        student.reset()
        rgb = np.asarray(obs["rgb"]).transpose(1, 2, 0) if np.asarray(obs["rgb"]).shape[0] == 3 else np.asarray(obs["rgb"])
        gcls = CLASSES.index("iron_ore")
        mined = False
        for t in range(args.max_steps):
            toks = tok_head(rgb)
            rel = goal_relative(toks[None], np.array([gcls]))[0]
            a = student(rel, noop)
            pg = toks[:, 6 + gcls]
            j = int(np.argmax(pg * toks[:, 5]))
            centered = (pg[j] > 0.4 and abs(toks[j, 0] - 0.5) < 0.07
                        and abs(toks[j, 1] - 0.5) < 0.09)
            key, dist = _ray(obs["full"])[1], _ray(obs["full"])[2]
            if centered and toks[j, 5] > 0.015:        # 挖掘宏:锁相机持续攻击
                a = dict(noop)
                a["attack"] = True
                if t % 8 == 0:
                    a["forward"] = True                # 吸拾轻触
            obs, *_ = env.step(a)
            rgb = np.asarray(obs["rgb"]).transpose(1, 2, 0) if np.asarray(obs["rgb"]).shape[0] == 3 else np.asarray(obs["rgb"])
            if "raw_iron" in env_inventory(obs["full"]):
                mined = True
                break
        log["mined"] = mined
        log["mine_steps"] = t + 1

        # ③ 慢脑复核
        confirm = False
        if mined:
            _, confirm, _ = brain.plan("raw_iron",
                                       env_inventory(obs["full"]) & {"stone_pickaxe", "raw_iron"})
        log["confirmed"] = confirm

        # ④ 回家(指令切到家标记类)
        home = False
        if mined:
            student.reset()
            hcls = CLASSES.index(HOME_CLS)
            for t2 in range(args.max_steps):
                toks = tok_head(rgb)
                rel = goal_relative(toks[None], np.array([hcls]))[0]
                a = student(rel, noop)
                obs, *_ = env.step(a)
                rgb = np.asarray(obs["rgb"]).transpose(1, 2, 0) if np.asarray(obs["rgb"]).shape[0] == 3 else np.asarray(obs["rgb"])
                full = obs["full"]
                if np.linalg.norm(np.array([full.x, full.z]) - start) <= 3.0:
                    home = True
                    break
            log["home_steps"] = t2 + 1
        ok = mined and home
        log.update(ok=ok, home=home)
        results.append(log)
        print(f"[ep{ep}] {'✓' if ok else '✗'} mined={mined}({log['mine_steps']}) "
              f"confirm={confirm} home={home} {time.time()-t0:.0f}s", flush=True)
    env.close()

    rate = float(np.mean([r["ok"] for r in results])) if results else 0.0
    out = dict(success_rate=rate, gate="≥0.5",
               verdict="PASS" if rate >= 0.5 else "FAIL",
               mine_rate=float(np.mean([r.get("mined", False) for r in results])),
               home_rate=float(np.mean([r.get("home", False) for r in results])),
               confirm_rate=float(np.mean([r.get("confirmed", False) for r in results])),
               n=len(results), episodes=results)
    os.makedirs("runs", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"[fullloop] 成功率={rate:.2f} (挖 {out['mine_rate']:.2f}/复核 "
          f"{out['confirm_rate']:.2f}/回家 {out['home_rate']:.2f}) → {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=220)
    p.add_argument("--ckpt", default="runs/trackcmd_bc_v15/best.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head_v4.pt")
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--adapter", default="runs/reason_delta_lora_v3")
    p.add_argument("--port", type=int, default=8570)
    p.add_argument("--out", default="runs/fullloop_chain.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
