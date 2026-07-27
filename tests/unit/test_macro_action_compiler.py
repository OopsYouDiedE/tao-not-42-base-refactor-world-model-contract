"""验证回合宏编译器：Turn 逐 tick 展开、latch 后台延续、观察点落点、相机度填充、mc 归属。"""

import pytest

from rl_training_environments.craftground.action_contract import V2_KEYS
from rl_training_environments.craftground.macro_action_compiler import (
    CAMERA_KEYS,
    CAM_TICKS,
    TAP_TICKS,
    build_noop_action,
    compile_macros,
    minecraft_command,
    turn,
)


def test_noop_action_has_full_v2_key_set():
    """空动作 dict 键集 = 全部二值键 + 2 相机字段，二值全 False、相机全 0。"""
    action = build_noop_action()
    assert set(action.keys()) == set(V2_KEYS) | set(CAMERA_KEYS)
    assert all(action[key] is False for key in V2_KEYS)
    assert action["camera_yaw"] == 0.0
    assert action["camera_pitch"] == 0.0


def test_clicks_are_two_ticks_and_combine():
    """离散键 click 固定 TAP_TICKS，组合键同帧一起为真、不做互斥。"""
    traj = compile_macros([turn(clicks=["jump", "attack"])])
    assert traj.total_ticks == TAP_TICKS
    for action in traj.tick_actions:
        assert action["jump"] is True
        assert action["attack"] is True
        assert action["forward"] is False


def test_within_turn_hold_uses_own_duration_and_advances_time():
    """非 latch 键的定时长按（如 jump）按各自时长为真并推进时间轴。"""
    traj = compile_macros([turn(holds={"jump": 3})])
    assert traj.total_ticks == 3
    assert all(a["jump"] is True for a in traj.tick_actions)


def test_short_within_turn_key_releases_early():
    """回合前台时长取最大值；较短的定时长按键提前松开。"""
    # jump 定时 1t（非 latch）+ clicks attack（2t）→ 前台 2t，jump 只前 1t 为真
    traj = compile_macros([turn(holds={"jump": 1}, clicks=["attack"])])
    assert traj.total_ticks == 2
    assert traj.tick_actions[0]["jump"] is True
    assert traj.tick_actions[1]["jump"] is False
    assert all(a["attack"] is True for a in traj.tick_actions)


def test_latch_hold_runs_as_background_countdown_trailing():
    """latch 键（forward）时长作为后台预算，回合后由尾段消费。"""
    traj = compile_macros([turn(holds={"forward": 40}, clicks=["attack"])], max_blind_ticks=1000)
    assert traj.total_ticks == 40
    assert all(a["forward"] is True for a in traj.tick_actions)
    # 前台 2t 由 attack 推进，其余为尾段
    assert traj.tick_actions[0]["attack"] is True
    assert traj.tick_actions[2]["attack"] is False
    assert traj.tick_sources[0] == 0
    assert traj.tick_sources[-1] == -1


def test_latch_default_carry_across_turns():
    """latch 键在后续回合不重设时沿用剩余预算（default-carry）。"""
    traj = compile_macros([turn(holds={"forward": 10}), turn(clicks=["attack"])], max_blind_ticks=1000)
    # 第二回合的 attack 帧里 forward 仍为真
    assert traj.tick_actions[0]["forward"] is True
    assert traj.total_ticks == 10


def test_latch_release_stops_carry():
    """release 停止 latch 延续。"""
    traj = compile_macros([
        turn(holds={"forward": 20}),
        turn(clicks=["attack"], release=["forward"]),
    ], max_blind_ticks=1000)
    # release 后 attack 帧的 forward 应为假
    assert traj.tick_actions[0]["forward"] is False


def test_camera_delta_fixed_two_ticks_and_distributes():
    """相机固定 CAM_TICKS，度数均摊，逐帧累加还原总量。"""
    traj = compile_macros([turn(camera_mode="delta", delta_yaw=30.0, delta_pitch=-10.0)])
    assert traj.total_ticks == CAM_TICKS
    assert sum(a["camera_yaw"] for a in traj.tick_actions) == pytest.approx(30.0)
    assert sum(a["camera_pitch"] for a in traj.tick_actions) == pytest.approx(-10.0)
    assert traj.tick_actions[0]["camera_yaw"] == pytest.approx(15.0)


def test_camera_over_36_degrees_raises():
    """相机单回合 |Δ| > 36°（CAM_TICKS×CAM_MAX_DEG）报错。"""
    with pytest.raises(ValueError):
        turn(camera_mode="delta", delta_yaw=40.0)


def test_camera_screen_mode_distributes_coordinates():
    """屏幕坐标模式把坐标均摊到 CAM_TICKS。"""
    traj = compile_macros([turn(camera_mode="screen", screen_x=20.0, screen_y=8.0)])
    assert traj.total_ticks == CAM_TICKS
    assert sum(a["camera_yaw"] for a in traj.tick_actions) == pytest.approx(20.0)
    assert sum(a["camera_pitch"] for a in traj.tick_actions) == pytest.approx(8.0)


def test_wait_ticks_advances_time_as_pure_wait():
    """纯等待回合用 wait_ticks 推进时间轴。"""
    traj = compile_macros([turn(wait_ticks=5)])
    assert traj.total_ticks == 5
    assert all(a["forward"] is False and a["camera_yaw"] == 0.0 for a in traj.tick_actions)


def test_empty_turn_raises():
    """空回合（无任何键/相机/等待）报错。"""
    with pytest.raises(ValueError):
        turn()


def test_minecraft_command_attaches_before_next_turn():
    """mc 命令不推进 tick，挂到下一条推进回合的第一个 tick 之前。"""
    traj = compile_macros([
        minecraft_command("give @p diamond 5"),
        turn(clicks=["attack"]),
    ])
    assert traj.total_ticks == TAP_TICKS
    assert traj.tick_commands.get(0) == ["give @p diamond 5"]


def test_trailing_minecraft_command_goes_to_tail():
    """排在所有 tick 之后的 mc 命令进 pending_tail_commands。"""
    traj = compile_macros([
        turn(clicks=["attack"]),
        minecraft_command("time set day"),
    ])
    assert traj.pending_tail_commands == ["time set day"]


def test_observation_points_at_turn_boundaries():
    """每条推进回合的结束边界都是观察点，含起点 0 与终点。"""
    traj = compile_macros([turn(holds={"jump": 10}), turn(holds={"use": 5})], max_blind_ticks=1000)
    assert traj.observation_tick_indices == [0, 10, 15]


def test_long_segment_auto_splits_by_max_blind_ticks():
    """超过 max_blind_ticks 的盲段自动等距补插观察点。"""
    traj = compile_macros([turn(holds={"forward": 100}, wait_ticks=100)], max_blind_ticks=30)
    idx = traj.observation_tick_indices
    assert idx[0] == 0
    assert idx[-1] == 100
    for a, b in zip(idx, idx[1:]):
        assert b - a <= 30


def test_manual_observation_points_merged():
    """手动插入的观察点 tick 与自动规则合并去重。"""
    traj = compile_macros([turn(wait_ticks=20)], max_blind_ticks=1000,
                          manual_observation_ticks=[7, 13])
    assert 7 in traj.observation_tick_indices
    assert 13 in traj.observation_tick_indices
    assert traj.observation_tick_indices[0] == 0
    assert traj.observation_tick_indices[-1] == 20


def test_manual_observation_out_of_range_raises():
    """手动观察点越界报错。"""
    with pytest.raises(ValueError):
        compile_macros([turn(wait_ticks=5)], manual_observation_ticks=[99])


def test_tick_sources_map_back_to_command_index():
    """每个前台 tick 能溯源到产生它的回合下标。"""
    traj = compile_macros([turn(holds={"jump": 3}), turn(holds={"use": 2})])
    assert traj.tick_sources == [0, 0, 0, 1, 1]


def test_segment_bounds_are_consecutive_observation_pairs():
    """段边界 = 相邻观察点对，覆盖全轨迹无缝。"""
    traj = compile_macros([turn(wait_ticks=10), turn(wait_ticks=10)], max_blind_ticks=1000)
    assert traj.segment_bounds() == [(0, 10), (10, 20)]


def test_invalid_key_raises():
    """未知二值键报错。"""
    with pytest.raises(ValueError):
        turn(clicks=["teleport"])


def test_release_non_latch_key_raises():
    """release 只能松开 latch 键，其余报错。"""
    with pytest.raises(ValueError):
        turn(clicks=["attack"], release=["jump"])


def test_invalid_max_blind_ticks_raises():
    """max_blind_ticks < 1 报错。"""
    with pytest.raises(ValueError):
        compile_macros([turn(wait_ticks=5)], max_blind_ticks=0)
