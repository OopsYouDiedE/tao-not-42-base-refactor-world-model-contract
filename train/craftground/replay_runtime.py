"""把世界快照与真实 CraftGround 冷启动环境连接起来。

对外接口：CraftGroundReplayRuntime。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
from craftground.environment.action_space import ActionSpaceVersion
from craftground.screen_encoding_modes import ScreenEncodingMode

from train.craftground.trajectory_replay import ReplayTrajectory, restore_and_replay
from train.craftground.world_snapshot import WorldSnapshotStore


@dataclass(frozen=True)
class CraftGroundReplayRuntime:
    """真实 CraftGround 读档进程的启动配置。

    ``env_path`` 必须指向含 ``gradlew`` 与 ``run/saves`` 的独立 MinecraftEnv。
    每个并行进程必须使用不同目录，避免世界文件锁与工作副本互相覆盖。

    Attributes:
        env_path: MinecraftEnv 根目录；标量 Path。
        image_width: 图像宽度；标量 int，单位为像素。
        image_height: 图像高度；标量 int，单位为像素。
        port: IPC 起始端口；标量 int。
        screen_encoding_mode: 图像传输模式；标量枚举。
    """

    env_path: Path
    image_width: int = 640
    image_height: int = 360
    port: int = 8000
    screen_encoding_mode: ScreenEncodingMode = ScreenEncodingMode.RAW

    @property
    def saves_dir(self) -> Path:
        """返回工作世界目录；标量 Path。"""
        return self.env_path.resolve() / "run" / "saves"

    def build(self, display_name: str) -> CraftGroundEnvironment:
        """按显示名冷启动一个已有世界。

        Args:
            display_name: 快照清单记录的世界显示名；标量字符串，Dtype 为 str。

        Returns:
            未 reset 的 CraftGroundEnvironment 标量对象。
        """
        config = InitialEnvironmentConfig(
            image_width=self.image_width,
            image_height=self.image_height,
            screen_encoding_mode=self.screen_encoding_mode,
            level_display_name_to_play=display_name,
        )
        return CraftGroundEnvironment(
            config,
            action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
            env_path=str(self.env_path.resolve()),
            port=self.port,
            find_free_port=True,
            verbose=False,
        )

    def load_and_replay(
        self,
        store: WorldSnapshotStore,
        trajectory: ReplayTrajectory,
        *,
        slot_name: Optional[str] = None,
        replace: bool = False,
    ) -> Tuple[CraftGroundEnvironment, Any]:
        """恢复工作世界、启动 JVM 并校验重放完整轨迹。

        Args:
            store: 世界快照库；标量对象。
            trajectory: 完整 V2 轨迹；动作 Shape 为 [T]。
            slot_name: 可选工作目录名；标量字符串或 None。
            replace: 是否替换已有工作副本；标量 bool。

        Returns:
            ``(env, final_obs)``；标量环境与最终观测映射。
        """
        return restore_and_replay(
            store,
            trajectory,
            self.saves_dir,
            self.build,
            slot_name=slot_name,
            replace=replace,
        )
