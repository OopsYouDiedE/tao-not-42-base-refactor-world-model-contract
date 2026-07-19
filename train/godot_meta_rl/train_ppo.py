"""启动 Godot 聚光灯环境并执行 PPO 训练。

对外接口：main（命令行训练入口）。
"""

import argparse
from pathlib import Path

from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

from train.godot_meta_rl.vec_env import GodotVecEnv, RolloutProgress
from utils.godot_rl.launch import kill_godot, launch_godot
from utils.godot_rl.ppo_factory import build_model


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
    args = _parse_args()
    proc = None
    env = None
    try:
        if not args.no_launch:
            launch_kwargs = {} if args.godot_exe is None else {"godot_exe": args.godot_exe}
            proc = launch_godot(**launch_kwargs)
        env = VecTransposeImage(VecMonitor(GodotVecEnv(args.connect_timeout)))
        model = build_model(env, device=args.device, verbose=1, with_null_logger=False)
        model.learn(total_timesteps=args.total_timesteps, callback=RolloutProgress())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
    finally:
        if env is not None:
            env.close()
        kill_godot(proc)


if __name__ == "__main__":
    main()
