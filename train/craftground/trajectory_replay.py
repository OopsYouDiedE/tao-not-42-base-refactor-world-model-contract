"""CraftGround 可校验动作轨迹重放。

对外接口：WorldSnapshotStore、TrajectoryRecorder、ReplayTrajectory、replay_trajectory、
capture_running_world、restore_and_replay、state_fingerprint、ReplayDivergence。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from train.craftground.action_contract import V2_KEYS
from train.craftground.world_snapshot import (
    SnapshotManifest,
    WorldSnapshotStore,
    atomic_json_dump,
    discover_world_dir,
    validate_id,
)


TRAJECTORY_VERSION = 1
_BOOL_ACTION_KEYS = tuple(V2_KEYS)
_FLOAT_ACTION_KEYS = ("camera_pitch", "camera_yaw")
_ACTION_KEYS = frozenset(_BOOL_ACTION_KEYS + _FLOAT_ACTION_KEYS)


class ReplayDivergence(RuntimeError):
    """动作重放与记录状态首次不一致。

    Args:
        tick: 首个不一致动作后的 tick；标量整数，Dtype 为 int。
        expected: 记录轨迹中的 SHA-256 状态指纹；标量字符串，Dtype 为 str。
        actual: 重放得到的 SHA-256 状态指纹；标量字符串，Dtype 为 str。
    """

    def __init__(self, tick: int, expected: str, actual: str):
        super().__init__(
            f"轨迹在 tick={tick} 分叉：expected={expected}, actual={actual}"
        )
        self.tick = tick
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class ReplayTrajectory:
    """从一个世界快照出发的逐 tick 可逆动作日志。

    Attributes:
        snapshot_id: 起点快照标识；标量字符串，Dtype 为 str。
        actions: 完整 V2 动作序列；Shape 为 [T]，元素为 JSON 映射。
        checkpoints: tick 到状态指纹；Shape 为 [M]，键为十进制 tick 字符串。
        initial_digest: 动作前状态指纹；标量字符串。
        metadata: 调用方附加的 JSON 数据；标量映射。
    """

    snapshot_id: str
    actions: Tuple[Dict[str, Any], ...]
    checkpoints: Dict[str, str]
    initial_digest: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = TRAJECTORY_VERSION

    def save(self, path: os.PathLike[str] | str) -> None:
        """原子保存轨迹 JSON。

        Args:
            path: 输出文件路径；标量路径，Dtype 为 path-like。

        Returns:
            None。
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["actions"] = list(self.actions)
        atomic_json_dump(out, payload)

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "ReplayTrajectory":
        """读取并校验轨迹 JSON。

        Args:
            path: 输入文件路径；标量路径，Dtype 为 path-like。

        Returns:
            ReplayTrajectory 标量对象。
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != TRAJECTORY_VERSION:
            raise ValueError(f"不支持的轨迹版本：{data.get('version')}")
        actions = tuple(decode_v2_action(x) for x in data["actions"])
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            actions=actions,
            checkpoints={str(k): str(v) for k, v in data["checkpoints"].items()},
            initial_digest=str(data["initial_digest"]),
            metadata=dict(data.get("metadata", {})),
            version=int(data["version"]),
        )


class TrajectoryRecorder:
    """记录完整 V2 动作与动作后状态指纹。

    Args:
        snapshot_id: 起点快照标识；标量字符串，Dtype 为 str。
        initial_full_obs: 起点 protobuf/对象观测；无张量 Shape，Dtype 由 CraftGround 定义。
        metadata: 可选 JSON 元数据；标量映射。
    """

    def __init__(
        self,
        snapshot_id: str,
        initial_full_obs: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        validate_id(snapshot_id, "snapshot_id")
        self.snapshot_id = snapshot_id
        self.initial_digest = state_fingerprint(initial_full_obs)
        self.actions: List[Dict[str, Any]] = []
        self.checkpoints: Dict[str, str] = {"0": self.initial_digest}
        self.metadata = dict(metadata or {})

    def append(self, action: Mapping[str, Any], full_obs_after: Any) -> None:
        """追加一个逐 tick 动作及动作后状态。

        Args:
            action: 完整 CraftGround V2 动作；标量映射，布尔键 Dtype bool，
                `camera_pitch/yaw` Dtype float32/float64，单位为度。
            full_obs_after: 动作后完整观测；无张量 Shape，Dtype 由 CraftGround 定义。

        Returns:
            None。
        """
        self.actions.append(encode_v2_action(action))
        self.checkpoints[str(len(self.actions))] = state_fingerprint(full_obs_after)

    def finish(self) -> ReplayTrajectory:
        """冻结当前记录。

        Returns:
            ReplayTrajectory 标量对象；动作 Shape 为 [T]。
        """
        return ReplayTrajectory(
            snapshot_id=self.snapshot_id,
            actions=tuple(self.actions),
            checkpoints=dict(self.checkpoints),
            initial_digest=self.initial_digest,
            metadata=dict(self.metadata),
        )


def encode_v2_action(action: Mapping[str, Any]) -> Dict[str, Any]:
    """把完整 CraftGround V2 动作规范化为可逆 JSON 映射。

    Args:
        action: 标量动作映射；布尔键 Dtype bool，相机键 Dtype float，单位为度。

    Returns:
        完整动作标量映射；键集合固定，Dtype 为 bool/float。
    """
    unknown = set(action) - _ACTION_KEYS
    missing = _ACTION_KEYS - set(action)
    if unknown or missing:
        raise ValueError(f"V2 动作契约不匹配：missing={sorted(missing)}, unknown={sorted(unknown)}")
    out: Dict[str, Any] = {key: bool(action[key]) for key in _BOOL_ACTION_KEYS}
    out.update({key: float(action[key]) for key in _FLOAT_ACTION_KEYS})
    return out


def decode_v2_action(data: Mapping[str, Any]) -> Dict[str, Any]:
    """从 JSON 映射无损恢复 CraftGround V2 动作。

    Args:
        data: 标量 JSON 映射；键集合为 V2 协议。

    Returns:
        完整动作标量映射；相机单位为度。
    """
    return encode_v2_action(data)


def state_fingerprint(full_obs: Any) -> str:
    """计算与任务状态相关的稳定 SHA-256 指纹。

    指纹覆盖位置/朝向、生命与饥饿、经验、时间、天气、选中栏和逐槽库存。
    RGB 不参与，避免渲染器噪声导致物理状态误判。

    Args:
        full_obs: CraftGround `ObservationSpaceMessage` 或同字段对象；无张量 Shape。

    Returns:
        64 字符 SHA-256；标量字符串，Dtype 为 str。
    """
    payload = _canonical_state(full_obs)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def replay_trajectory(
    env: Any,
    trajectory: ReplayTrajectory,
    *,
    initial_full_obs: Any,
) -> Any:
    """从已加载快照逐 tick 重放并在首个分叉处失败。

    Args:
        env: 原生 CraftGround 环境；`step(V2)` 返回 gym 四/五元组。
        trajectory: 待重放轨迹；动作 Shape 为 [T]。
        initial_full_obs: 快照加载后的完整观测；无张量 Shape。

    Returns:
        最后一步 gym 观测映射；T=0 时返回 `{"full": initial_full_obs}`。
    """
    actual_initial = state_fingerprint(initial_full_obs)
    if actual_initial != trajectory.initial_digest:
        raise ReplayDivergence(0, trajectory.initial_digest, actual_initial)
    last_obs: Any = {"full": initial_full_obs}
    for tick, action in enumerate(trajectory.actions, start=1):
        result = env.step(decode_v2_action(action))
        last_obs = result[0]
        actual = state_fingerprint(last_obs["full"])
        expected = trajectory.checkpoints.get(str(tick))
        if expected is not None and actual != expected:
            raise ReplayDivergence(tick, expected, actual)
    return last_obs


def flush_world(env: Any, noop_action: Mapping[str, Any]) -> Any:
    """请求 Minecraft 手动落盘并推进一 tick 等待命令完成。

    CraftGround 只禁用了自动保存；未知命令会转交 Minecraft 命令执行器。调用后磁盘
    世界可交给 `WorldSnapshotStore.capture`。不要在命令完成前复制运行中目录。

    Args:
        env: 原生 CraftGround 环境，须提供 `add_command` 与 `step`。
        noop_action: 完整 V2 no-op；标量映射，相机单位为度。

    Returns:
        `env.step` 的 gym 四/五元组。
    """
    env.add_command("save-all flush")
    return env.step(decode_v2_action(noop_action))


def capture_running_world(
    store: WorldSnapshotStore,
    env: Any,
    noop_action: Mapping[str, Any],
    snapshot_id: str,
    saves_dir: os.PathLike[str] | str,
    *,
    display_name: str,
    preferred_folder: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[SnapshotManifest, Any]:
    """将运行中的 CraftGround 世界同步落盘并保存为不可变快照。

    Args:
        store: 世界快照库；标量对象。
        env: 原生 CraftGround 环境；标量对象。
        noop_action: 完整 V2 no-op；标量映射，相机单位为度。
        snapshot_id: 新快照标识；标量字符串，Dtype 为 str。
        saves_dir: 当前环境的 ``run/saves``；标量路径。
        display_name: 世界显示名；标量字符串，Dtype 为 str。
        preferred_folder: 可选世界目录名；标量字符串或 None。
        metadata: 可选 JSON 元数据；标量映射。

    Returns:
        ``(manifest, full_obs)``；新建清单与 flush 后轨迹起点观测。
    """
    result = flush_world(env, noop_action)
    obs = result[0]
    if "full" not in obs:
        raise KeyError("CraftGround step 结果缺少 obs['full']，无法记录保存点状态")
    world_dir = discover_world_dir(saves_dir, preferred_folder)
    manifest = store.capture(
        snapshot_id,
        world_dir,
        display_name=display_name,
        state_digest=state_fingerprint(obs["full"]),
        metadata=metadata,
    )
    return manifest, obs["full"]


def restore_and_replay(
    store: WorldSnapshotStore,
    trajectory: ReplayTrajectory,
    saves_dir: os.PathLike[str] | str,
    env_factory: Callable[[str], Any],
    *,
    slot_name: Optional[str] = None,
    replace: bool = False,
) -> Tuple[Any, Any]:
    """恢复世界工作副本、冷启动环境并逐 tick 重放轨迹。

    调用方必须先关闭使用同一 ``saves_dir`` 的旧 CraftGround 实例。工厂接收清单中的
    ``display_name``，并须构造设置了 ``level_display_name_to_play`` 的新环境。

    Args:
        store: 世界快照库；标量对象。
        trajectory: 从该快照开始的动作轨迹；动作 Shape 为 [T]。
        saves_dir: 新环境独占的 ``run/saves``；标量路径。
        env_factory: ``display_name -> env`` 工厂；输入 Dtype 为 str。
        slot_name: 可选工作副本目录名；标量字符串或 None。
        replace: 是否替换已有工作副本；标量 bool。

    Returns:
        ``(env, final_obs)``；标量环境对象与最后一步观测映射。

    Raises:
        ReplayDivergence: 冷启动状态或任一步状态与记录不一致。
    """
    _, manifest = store.restore(
        trajectory.snapshot_id,
        saves_dir,
        slot_name=slot_name,
        replace=replace,
    )
    env = env_factory(manifest.display_name)
    try:
        initial_obs, _ = env.reset(options={"fast_reset": False})
        full_obs = initial_obs["full"]
        if manifest.state_digest is not None:
            actual = state_fingerprint(full_obs)
            if actual != manifest.state_digest:
                raise ReplayDivergence(0, manifest.state_digest, actual)
        final_obs = replay_trajectory(
            env,
            trajectory,
            initial_full_obs=full_obs,
        )
        return env, final_obs
    except Exception:
        env.close()
        raise


def _canonical_state(full_obs: Any) -> Dict[str, Any]:
    scalar_names = (
        "x", "y", "z", "yaw", "pitch", "health", "food_level", "saturation_level",
        "experience", "experience_level", "selected_hotbar_slot", "world_time",
        "is_raining", "is_thundering", "is_dead",
    )
    state: Dict[str, Any] = {}
    for name in scalar_names:
        if hasattr(full_obs, name):
            state[name] = _normal_scalar(getattr(full_obs, name))

    inventory = []
    for index, item in enumerate(getattr(full_obs, "inventory", ())):
        row = {"index": index}
        for name in ("slot", "translation_key", "count", "damage", "max_damage"):
            if hasattr(item, name):
                row[name] = _normal_scalar(getattr(item, name))
        inventory.append(row)
    state["inventory"] = inventory
    return state


def _normal_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        return _normal_scalar(value.item())
    return str(value)
