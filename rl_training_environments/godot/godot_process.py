"""跨平台启动和终止 Godot 子进程。

对外接口：launch_godot（按平台拼命令启动 Godot 训练场景）、terminate_godot_process（终止进程树，
其它平台 terminate/kill）。
"""

import subprocess

from .shared_memory_environment import GODOT_EXE, PROJECT_DIR, TRAIN_SCENE, IS_WINDOWS


def launch_godot(scene=TRAIN_SCENE, project_dir=PROJECT_DIR, godot_exe=GODOT_EXE,
                 log=None, extra_env=None, base_env=None):
    """启动 Godot 训练场景子进程并返回 Popen。

    Parameters
    ----------
    scene : str          res:// 场景路径，默认 training_main.tscn。
    project_dir : str    Godot 工程目录（含 project.godot）。
    godot_exe : str      Godot 可执行文件路径（mono 版）。
    log : file or None   重定向 stdout/stderr 的文件句柄；None 则丢弃。
    extra_env : dict     额外环境变量（如 RL_FIXED_STEPS / RL_ASYNC），合并进进程环境。
    base_env : dict      基础环境（默认 os.environ.copy()）。
    """
    import os
    env = (base_env if base_env is not None else os.environ).copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    stdout = log if log is not None else subprocess.DEVNULL
    return subprocess.Popen([godot_exe, "--path", project_dir, scene],
                            stdout=stdout, stderr=subprocess.STDOUT, env=env)


def terminate_godot_process(process):
    """跨平台终止 Godot 进程：Windows 用 taskkill 杀整棵进程树，其它平台 terminate→kill。"""
    if process is None:
        return
    if IS_WINDOWS:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
