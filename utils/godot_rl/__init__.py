"""Godot 40 环境 RL 的跨平台基础设施（共享内存驱动 + 启动/工厂助手）。

显式 re-export，禁止桶式 `import *`（命名空间透明，承 code_conventions §2.5）。
"""

from .shared_memory_environment import (
    GodotTrainingEnvironment,
    shared_memory_path,
    set_timer_resolution,
    reset_timer_resolution,
    GODOT_EXE,
    PROJECT_DIR,
    TRAIN_SCENE,
    IS_WINDOWS,
    NUM_ENVS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    CHANNELS,
    CONT_DIM,
    DISC_DIM,
    META_PER_ENV,
    META_OFFSET,
    CONT_OFFSET,
    DISC_OFFSET,
    SEQ_OFFSET,
    OBS_SEQ_OFFSET,
    ACT_SEQ_OFFSET,
    TOTAL_SHM_SIZE,
    M_FRAME,
    M_STEPS,
    M_SIM_DT,
    M_REWARD,
    M_DONE,
)
from .godot_process import launch_godot, terminate_godot_process

__all__ = [
    "GodotTrainingEnvironment", "shared_memory_path", "set_timer_resolution", "reset_timer_resolution",
    "GODOT_EXE", "PROJECT_DIR", "TRAIN_SCENE", "IS_WINDOWS",
    "NUM_ENVS", "IMAGE_WIDTH", "IMAGE_HEIGHT", "CHANNELS", "CONT_DIM", "DISC_DIM",
    "META_PER_ENV", "META_OFFSET", "CONT_OFFSET", "DISC_OFFSET", "SEQ_OFFSET",
    "OBS_SEQ_OFFSET", "ACT_SEQ_OFFSET", "TOTAL_SHM_SIZE",
    "M_FRAME", "M_STEPS", "M_SIM_DT", "M_REWARD", "M_DONE",
    "launch_godot", "terminate_godot_process",
]
