"""启动 Godot 聚光灯环境并执行 PPO 训练。

对外接口：main（命令行训练入口）。
"""

import argparse
from pathlib import Path

from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

from rl_training_environments.godot.godot_process import (
    launch_godot,
    terminate_godot_process,
)
from rl_training_environments.godot.ppo_model_factory import build_ppo_model
from rl_training_environments.godot.vectorized_environment import (
    GodotVectorizedEnvironment,
    RolloutProgress,
)


def _parse_args():
    """解析训练参数。

    Returns
    -------
    argparse.Namespace
        不含张量的数据类命名空间。
    """
    parser = argparse.ArgumentParser(description="训练 Godot 聚光灯 PPO 策略")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/checkpoints/godot_ppo"))
    parser.add_argument("--godot-exe", default=None, help="Godot 4 .NET 可执行文件；默认读取 GODOT_EXE")
    parser.add_argument("--connect-timeout", type=float, default=60.0)
    parser.add_argument("--no-launch", action="store_true", help="连接已运行的 Godot，不启动子进程")
    return parser.parse_args()


def main():
    """运行 PPO 训练并保存模型。

    Returns
    -------
    None
        无张量返回值；模型保存到 ``--output`` 指定路径。
    """
    arguments = _parse_args()
    process = None
    environment = None
    try:
        if not arguments.no_launch:
            launch_arguments = (
                {} if arguments.godot_exe is None
                else {"godot_exe": arguments.godot_exe}
            )
            process = launch_godot(**launch_arguments)
        environment = VecTransposeImage(
            VecMonitor(GodotVectorizedEnvironment(arguments.connect_timeout)),
        )
        model = build_ppo_model(
            environment, device=arguments.device, verbose=1, with_null_logger=False,
        )
        model.learn(total_timesteps=arguments.total_timesteps, callback=RolloutProgress())
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(arguments.output)
    finally:
        if environment is not None:
            environment.close()
        terminate_godot_process(process)


if __name__ == "__main__":
    main()
