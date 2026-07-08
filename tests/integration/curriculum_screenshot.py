# -*- coding: utf-8 -*-
"""教师课程可行性验证:用命令把世界构造成指定初始态,截图 + 读回库存为证。

回答用户命题"能否通过命令来设计课程":不只证明注入通道通(源码已证),
而是端到端把三套**过程导向**课程摆出来,以**截图**(obs["rgb"])为准。

三套课程(MC 1.21,全部 SURVIVAL,过程 > 结果):
  C1 hold_log_craft_table : 手持 1 原木 → 考合成工作台(过程:log→planks→table 两步)
  C2 cave_mine_iron       : 石室含铁矿脉 + 手持石镐 → 考生存中挖铁(过程:导航+正确工具破正确方块)
  C3 fill_junk_organize   : 36 格塞满杂物 → 考整理背包(过程:选择性丢弃/归并)

用法(W4 训练完、GPU 空出后跑):
    PYTHONPATH=. python tests/integration/curriculum_screenshot.py \
        --out runs/craftground_curriculum
产出:每套课程一张 PNG(构造后 + 若干沉降帧)+ 一份库存 json。
注:命令经原版 chat 派发器异步生效,构造后走几帧 no-op 让其沉降再截图。
"""
import argparse
import json
import os


from craftground import make
from craftground.initial_environment_config import InitialEnvironmentConfig
from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
from craftground.screen_encoding_modes import ScreenEncodingMode

from tests.integration.test_utils import dump_inventory, save_png


# ---- 三套课程的构造命令(不带前导 "/";mod 会补;避开保留词 respawn/fastreset/exit/random-summon)----
CURRICULA = {
    "C1_hold_log_craft_table": [
        "gamemode survival @p",
        "clear @p",
        # 主手放 1 个原木:1 log→4 planks→crafting_table(4 planks),过程恰好两步可拆
        "item replace entity @p weapon.mainhand with minecraft:oak_log 1",
    ],
    "C2_cave_mine_iron": [
        "gamemode survival @p",
        "clear @p",
        # 就地围一间 11x6x11 石室(hollow=只做壳),模拟矿洞封闭空间
        "fill ~-5 ~-1 ~-5 ~5 ~4 ~5 minecraft:stone hollow",
        # 在墙上镶几处铁矿脉(过程:得走过去、用对镐)
        "setblock ~4 ~ ~4 minecraft:iron_ore",
        "setblock ~4 ~1 ~4 minecraft:iron_ore",
        "setblock ~-4 ~ ~4 minecraft:iron_ore",
        "setblock ~4 ~ ~-4 minecraft:iron_ore",
        # 手持石镐(挖铁的最低工具;木镐挖不出铁,考"用对工具")
        "item replace entity @p weapon.mainhand with minecraft:stone_pickaxe 1",
    ],
    "C3_fill_junk_organize": [
        "gamemode survival @p",
        "clear @p",
        # 36 格逐格塞满杂物(container.0-8=快捷栏,9-35=主背包)
        # 用一组"垃圾"轮换填满,考整理(丢弃/归并同类)
    ],
}
# C3 程序化生成 36 格填充命令
_JUNK = ["dirt", "cobblestone", "gravel", "andesite", "diorite", "granite",
         "netherrack", "rotten_flesh", "dead_bush", "cobbled_deepslate",
         "tuff", "basalt"]
for _slot in range(36):
    CURRICULA["C3_fill_junk_organize"].append(
        f"item replace entity @p container.{_slot} with minecraft:{_JUNK[_slot % len(_JUNK)]} 64")


def run_curriculum(name, commands, out_dir, settle_frames, port):
    """单套课程:构造→沉降→截图+库存。用独立 env(construction 走 initial_extra_commands)。"""
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,  # CPU 读回,直接拿 numpy
        initial_extra_commands=list(commands),
    )
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=port, find_free_port=True, verbose=False)
    try:
        obs, _ = env.reset()
        noop = no_op_v2()
        for _ in range(settle_frames):  # 命令异步生效,走几帧 no-op 沉降
            obs, *_ = env.step(noop)
        shape = save_png(obs["rgb"], os.path.join(out_dir, f"{name}.png"))
        inv = dump_inventory(obs["full"])
        rec = {"curriculum": name, "n_commands": len(commands),
               "img_shape": list(shape), "inventory": inv,
               "commands": commands}
        json.dump(rec, open(os.path.join(out_dir, f"{name}.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(f"[{name}] png+json saved; inv={inv}", flush=True)
        return rec
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/craftground_curriculum")
    p.add_argument("--settle", type=int, default=5)
    p.add_argument("--base_port", type=int, default=8400)
    p.add_argument("--only", default="", help="只跑某套课程名(默认全跑)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = [args.only] if args.only else list(CURRICULA)
    summary = []
    for i, name in enumerate(names):
        summary.append(run_curriculum(name, CURRICULA[name], args.out,
                                      args.settle, args.base_port + i * 10))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[done] {len(summary)} curricula → {args.out}", flush=True)


if __name__ == "__main__":
    main()
