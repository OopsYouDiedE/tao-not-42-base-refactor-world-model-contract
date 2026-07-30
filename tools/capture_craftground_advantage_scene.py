"""创建相对优势演示场景并保存真实 CraftGround 起点图像。"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


SCENE_COMMANDS = (
    "gamemode survival @s",
    "gamerule doDaylightCycle false",
    "gamerule doWeatherCycle false",
    "gamerule doMobSpawning false",
    "time set 6000",
    "weather clear",
    "fill 0 62 0 8 69 10 minecraft:air",
    "fill 0 62 0 8 62 10 minecraft:smooth_stone",
    "fill 0 63 0 8 63 10 minecraft:oak_planks",
    "fill 0 64 0 8 67 0 minecraft:stone_bricks",
    "fill 0 64 10 8 67 10 minecraft:stone_bricks",
    "fill 0 64 0 0 67 10 minecraft:stone_bricks",
    "fill 8 64 0 8 67 10 minecraft:stone_bricks",
    "setblock 3 64 3 minecraft:chest[facing=south]",
    "item replace block 3 64 3 container.0 with minecraft:iron_ingot 3",
    "setblock 4 64 3 minecraft:crafting_table",
    "setblock 5 64 3 minecraft:furnace[facing=south]",
    "item replace block 5 64 3 container.0 with minecraft:raw_iron 2",
    "item replace block 5 64 3 container.1 with minecraft:coal 2",
    "setblock 2 64 3 minecraft:gold_block",
    "setblock 6 64 3 minecraft:diamond_block",
    "setblock 1 65 1 minecraft:torch",
    "setblock 7 65 1 minecraft:torch",
    "clear @s",
    "give @s minecraft:stick 2",
    "tp @s 4.5 64 8.5 180 12",
)


def build_environment(runtime: Path):
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.screen_encoding_modes import ScreenEncodingMode

    config = InitialEnvironmentConfig(
        image_width=640,
        image_height=360,
        seed="424242",
        render_distance=3,
        simulation_distance=5,
        screen_encoding_mode=ScreenEncodingMode.RAW,
    )
    return CraftGroundEnvironment(
        config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        env_path=str(runtime),
        port=18300,
        find_free_port=True,
        cleanup_world=False,
        verbose=False,
    )


def main() -> None:
    from craftground.environment.action_space import no_op_v2

    parser = argparse.ArgumentParser(description="捕获 CraftGround 相对优势场景")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    environment = build_environment(args.runtime.resolve())
    try:
        environment.reset(options={"fast_reset": False})
        environment.add_commands(list(SCENE_COMMANDS))
        observation = None
        for _ in range(20):
            observation = environment.step(no_op_v2())[0]
        assert observation is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observation["rgb"]).save(args.output)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
