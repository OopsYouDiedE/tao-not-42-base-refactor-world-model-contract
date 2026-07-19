"""离线验证 CraftGround 世界快照与逐 tick 动作重放。

对外接口：无。
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import pytest

from rl_training_environments.craftground.trajectory_replay import (
    ReplayDivergence,
    ReplayTrajectory,
    TrajectoryRecorder,
    WorldSnapshotStore,
    capture_running_world,
    replay_trajectory,
    restore_and_replay,
)


def _action(*, forward: bool = False, yaw: float = 0.0) -> Dict[str, Any]:
    """构造完整 V2 测试动作。

    Args:
        forward: 是否前进；标量 bool。
        yaw: 水平相机增量；标量 float，单位为度。

    Returns:
        完整动作映射；Shape 为 [22]，Dtype 为 bool/float。
    """
    keys = (
        "attack", "back", "forward", "jump", "left", "right", "sneak",
        "sprint", "use", "drop", "inventory", "hotbar.1", "hotbar.2",
        "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7",
        "hotbar.8", "hotbar.9",
    )
    action: Dict[str, Any] = {key: False for key in keys}
    action["forward"] = forward
    action["camera_pitch"] = 0.0
    action["camera_yaw"] = yaw
    return action


class _FakeEnv:
    """只用于离线集成测试的确定性 CraftGround 协议替身。"""

    def __init__(self, world_dir: Path | None = None):
        self.x = 0.0
        self.yaw = 0.0
        self.world_dir = world_dir
        self.commands = []
        self.closed = False

    def _full_obs(self) -> SimpleNamespace:
        """返回协议字段观测；Shape 为标量对象。"""
        return SimpleNamespace(
            x=self.x,
            y=64.0,
            z=0.0,
            yaw=self.yaw,
            pitch=0.0,
            health=20.0,
            food_level=20,
            selected_hotbar_slot=0,
            world_time=100,
            inventory=[],
        )

    def reset(self, *, options: Mapping[str, Any] | None = None):
        """返回初始观测；Shape 为标量映射。"""
        assert options == {"fast_reset": False}
        return {"full": self._full_obs()}, {}

    def step(self, action: Mapping[str, Any]):
        """执行一个确定性动作；输入 Shape 为 [22]。"""
        self.x += 1.0 if action["forward"] else 0.0
        self.yaw += float(action["camera_yaw"])
        return {"full": self._full_obs()}, 0.0, False, False, {}

    def add_command(self, command: str) -> None:
        """记录 Minecraft 命令；输入为标量字符串。"""
        self.commands.append(command)

    def close(self) -> None:
        """标记环境关闭；返回 None。"""
        self.closed = True


def _make_world(path: Path, *, level: bytes = b"level") -> None:
    """创建最小离线世界目录；输入为标量路径。"""
    (path / "region").mkdir(parents=True)
    (path / "playerdata").mkdir()
    (path / "level.dat").write_bytes(level)
    (path / "region" / "r.0.0.mca").write_bytes(b"region")
    (path / "playerdata" / "player.dat").write_bytes(b"player")
    (path / "session.lock").write_bytes(b"lock")


def test_snapshot_is_immutable_verified_and_restorable(tmp_path: Path) -> None:
    """完整快照应排除锁文件、不可覆盖并可恢复。"""
    source = tmp_path / "source" / "world-a"
    _make_world(source)
    store = WorldSnapshotStore(tmp_path / "snapshots")

    manifest = store.capture("start", source, display_name="Replay World")
    (source / "level.dat").write_bytes(b"mutated")
    restored, verified = store.restore("start", tmp_path / "work" / "saves")

    assert manifest == verified
    assert (restored / "level.dat").read_bytes() == b"level"
    assert (restored / "playerdata" / "player.dat").is_file()
    assert not (restored / "session.lock").exists()
    with pytest.raises(FileExistsError):
        store.capture("start", source, display_name="Replay World")


def test_trajectory_round_trip_replays_and_detects_divergence(tmp_path: Path) -> None:
    """轨迹 JSON 应可重放，且在首个不一致 tick 失败。"""
    env = _FakeEnv()
    initial, _ = env.reset(options={"fast_reset": False})
    recorder = TrajectoryRecorder("start", initial["full"])
    for action in (_action(forward=True), _action(yaw=15.0)):
        obs = env.step(action)[0]
        recorder.append(action, obs["full"])

    path = tmp_path / "trajectory.json"
    recorder.finish().save(path)
    loaded = ReplayTrajectory.load(path)
    replay_env = _FakeEnv()
    replay_initial, _ = replay_env.reset(options={"fast_reset": False})
    final = replay_trajectory(
        replay_env,
        loaded,
        initial_full_obs=replay_initial["full"],
    )

    assert final["full"].x == 1.0
    assert final["full"].yaw == 15.0
    divergent = _FakeEnv()
    divergent.x = 2.0
    with pytest.raises(ReplayDivergence, match="tick=0"):
        replay_trajectory(
            divergent,
            loaded,
            initial_full_obs=divergent._full_obs(),
        )


def test_capture_restore_and_replay_form_end_to_end_flow(tmp_path: Path) -> None:
    """运行中保存、冷启动恢复与重放应组成一个闭环。"""
    saves = tmp_path / "runtime" / "saves"
    source = saves / "world-a"
    _make_world(source)
    store = WorldSnapshotStore(tmp_path / "snapshots")
    saving_env = _FakeEnv(source)
    saving_env.reset(options={"fast_reset": False})

    manifest, replay_start = capture_running_world(
        store,
        saving_env,
        _action(),
        "start",
        saves,
        display_name="Replay World",
    )
    assert saving_env.commands == ["save-all flush"]

    recorder = TrajectoryRecorder("start", replay_start)
    action = _action(forward=True)
    obs = saving_env.step(action)[0]
    recorder.append(action, obs["full"])
    trajectory = recorder.finish()
    launched_names = []

    def factory(display_name: str) -> _FakeEnv:
        """记录冷启动显示名并返回环境；输入 Dtype 为 str。"""
        launched_names.append(display_name)
        return _FakeEnv()

    loaded_env, final = restore_and_replay(
        store,
        trajectory,
        tmp_path / "restored" / "saves",
        factory,
    )

    assert manifest.state_digest == trajectory.initial_digest
    assert launched_names == ["Replay World"]
    assert final["full"].x == 1.0
    assert not loaded_env.closed


def test_tampered_snapshot_is_rejected(tmp_path: Path) -> None:
    """快照任意文件变化都应在恢复前被拒绝。"""
    source = tmp_path / "source" / "world-a"
    _make_world(source)
    store = WorldSnapshotStore(tmp_path / "snapshots")
    store.capture("start", source, display_name="Replay World")
    (tmp_path / "snapshots" / "start" / "world" / "level.dat").write_bytes(b"bad")

    with pytest.raises(ValueError, match="校验失败"):
        store.restore("start", tmp_path / "work" / "saves")
