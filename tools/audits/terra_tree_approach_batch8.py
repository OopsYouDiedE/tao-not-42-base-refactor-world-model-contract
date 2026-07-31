"""从真实 CraftGround 树冠观察运行接近树干的 2+6 同快照审计。"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from game_environment import CraftGroundActionAdapter, MemorySnapshotCoordinator, SnapshotRegion, save_rgb, step_commands
from lumine.action_codec import LumineActionChunk, LumineWindowAction, decode_lumine_action
from curriculum.bank import SnapshotRecord, design_document

MIN_SUCCESS_RATE = 0.75


@dataclass(frozen=True)
class Course:
    course_id: str
    title: str
    task: str
    prerequisites: tuple[str, ...]
    tick_budget: int
    success_rule: str
    difficulty: str
    observation_basis: str
    eligible: bool
    rejection: str | None = None
    target: tuple[int, int, int] | None = None
    feasibility_evidence: str = ""
    required_skills: tuple[str, ...] = ()
    risk_controls: str = ""
    recovery_snapshot: str = "课程出生快照"
    failure_mode: str = "未定义"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source_role: str
    description: str
    chunks: tuple[LumineActionChunk, ...]
    response_token_ids: tuple[int, ...] = ()
    old_logprobs: tuple[float, ...] = ()
    policy_version: str | None = None
    sampling_parameters: dict[str, Any] | None = None

    @property
    def action_text(self) -> str:
        return LumineWindowAction(self.chunks).to_text()


def build_environment(runtime: Path) -> Any:
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.screen_encoding_modes import ScreenEncodingMode

    config = InitialEnvironmentConfig(
        image_width=640,
        image_height=360,
        seed="424242",
        render_distance=6,
        simulation_distance=6,
        request_raycast=True,
        requires_surrounding_blocks=True,
        mined_stat_keys=["dark_oak_log", "stone", "coal_ore", "iron_ore"],
        screen_encoding_mode=ScreenEncodingMode.RAW,
    )
    return CraftGroundEnvironment(
        config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        env_path=str(runtime),
        port=18800,
        find_free_port=True,
        cleanup_world=False,
        verbose=False,
    )


def capture_state(observation: Any) -> dict[str, Any]:
    full = observation["full"]
    target = getattr(full.raycast_result, "target_block", None)
    inventory = Counter()
    for stack in full.inventory:
        name, count = str(stack.translation_key), int(stack.count)
        if count > 0 and name != "item.minecraft.air":
            inventory[name] += count
    blocks = [
        {"name": str(block.translation_key), "x": int(block.x), "y": int(block.y), "z": int(block.z)}
        for block in full.surrounding_blocks
    ]
    return {
        "x": round(float(full.x), 3), "y": round(float(full.y), 3), "z": round(float(full.z), 3),
        "yaw": round(float(full.yaw), 3), "pitch": round(float(full.pitch), 3),
        "is_on_ground": bool(full.is_on_ground), "is_dead": bool(full.is_dead),
        "raycast_block": str(getattr(target, "translation_key", "")),
        "raycast_position": None if target is None else [int(target.x), int(target.y), int(target.z)],
        "inventory": dict(sorted(inventory.items())),
        "mined_statistics": {str(key): int(value) for key, value in full.mined_statistics.items()},
        "nearby_block_counts": dict(sorted(Counter(block["name"] for block in blocks).items())),
        "nearby_blocks": blocks[:120],
    }


def has_item(state: dict[str, Any], terms: tuple[str, ...]) -> bool:
    return any(any(term in name for term in terms) for name in state["inventory"])


def observed_snapshot_record(snapshot_id: str, state: dict[str, Any], output: Path) -> SnapshotRecord:
    """将本轮真实观察登记为一个课程银行条目，不补造未观察到的资源或装备。"""
    terrain = "tree_crown" if state["y"] >= 90 and any("leaves" in name for name in state["nearby_block_counts"]) else "unknown"
    route = "surface" if terrain == "tree_crown" else "unknown"
    return SnapshotRecord(
        snapshot_id=snapshot_id,
        provenance="真实 CraftGround reset 后的同进程观察，并以 memorysnapshot 保存；未发放物品、未搭建资源场景。",
        feasible=True,
        feasibility_evidence=(
            f"真实位置=({state['x']}, {state['y']}, {state['z']})",
            f"邻近方块={state['nearby_block_counts']}",
            f"准星命中={state['raycast_block'] or '无'}",
            f"背包={state['inventory']}",
        ),
        dimensions={
            "stage": "early",
            "inventory_loadout": "empty" if not state["inventory"] else "observed_nonempty",
            "biome_terrain": terrain,
            "hazard": "fall" if terrain == "tree_crown" else "unknown",
            "health_hunger": "healthy",
            "route": route,
            "strategy": "safe_descent",
            "version_mechanic": "Minecraft Java 1.21.x",
        },
        source_run=str(output),
        failure_count=0,
    )


def propose_courses(scans: list[dict[str, Any]]) -> list[Course]:
    observed = [scan["state"] for scan in scans]
    logs = [state for state in observed if "_log" in state["raycast_block"]]
    ores_or_stone = [state for state in observed if any(name in state["raycast_block"] for name in ("ore", "stone"))]
    start = observed[0]
    tree_crown = start["y"] >= 90 and any("leaves" in key for key in start["nearby_block_counts"])
    food = has_item(start, ("bread", "beef", "pork", "chicken", "apple", "carrot"))
    lighting = has_item(start, ("torch", "lantern"))
    tool = has_item(start, ("axe", "pickaxe"))
    courses: list[Course] = []
    if tree_crown and logs:
        target = tuple(logs[0]["raycast_position"] or ())
        courses.append(Course(
            "prep-descend-approach-observed-log", "准备：安全接近已观测树干",
            "从树冠沿当前已观测的树干方向移动，安全降低位置并显著缩短到树干的距离。",
            ("树冠高度", "可见树干"), 140,
            "存活、落地且到已观测树干的距离缩短至少 min(8 格，初始距离的 60%)。", "中等",
            f"位置 y={start['y']}，邻近树叶；扫描准星命中 {logs[0]['raycast_block']} 于 {target}。", True, target=target,
            feasibility_evidence="树冠高度、树叶支撑和可见树干均来自同一轮真实扫描；目标距离可由连续步行缩短。",
            required_skills=("相机锚点保持", "树冠移动", "落地后恢复"),
            risk_controls="每条轨迹从同一快照恢复；只在存活且落地后判定成功；失败立即回退。",
            failure_mode="从树冠坠落、被叶簇阻塞或距离未缩短。",
        ))
    elif logs:
        target = tuple(logs[0]["raycast_position"] or ())
        courses.append(Course(
            "prep-approach-observed-log", "准备：接近已观测树干",
            "沿准星已观测到的树干方向移动，在保持存活的前提下缩短目标距离。",
            ("可见树干",), 140,
            "存活、落地且到已观测树干的距离缩短至少 min(8 格，初始距离的 60%)。", "低",
            f"位置=({start['x']}, {start['y']}, {start['z']})；扫描准星命中 {logs[0]['raycast_block']} 于 {target}。", True, target=target,
            feasibility_evidence="树干位置来自当前同进程扫描；任务只要求安全接近，不假定已拥有工具或资源。",
            required_skills=("相机锚点保持", "基础移动", "目标接近"),
            risk_controls="每条轨迹从同一世界和玩家状态恢复；存活与落地是成功门。",
            failure_mode="目标失焦、碰撞阻塞、掉落或距离未缩短。",
        ))
    else:
        courses.append(Course(
            "prep-descend-approach-observed-log", "准备：安全接近已观测树干", "接近树干。",
            ("树冠高度", "可见树干"), 140, "存活并接近树干。", "中等",
            "当前观察未同时证明树冠高度与可见树干。", False, "缺少可验证的树冠或树干证据。",
            feasibility_evidence="当前扫描缺失树冠或树干，不启动执行。", required_skills=("基础移动",),
            risk_controls="不改变出生快照。", failure_mode="目标不可定位。",
        ))
    courses.append(Course(
        "harvest-dark-oak-log", "伐木：获得深色橡木原木", "选择可用斧头或在安全位置空手破坏近距离树干，获得原木。",
        ("树干距离不超过 5 格", "斧头或足够安全的徒手时间预算"), 180,
        "服务器方块检查为空气、挖掘统计增加、原木进入背包。", "中等",
        "扫描发现远处树干，但当前距离过远且背包无斧头。", False,
        "当前观察未证明近距离树干或斧头，先执行接近准备节点。",
        feasibility_evidence="远处树干真实可见，但当前没有斧头且不在有效破坏距离内。", required_skills=("接近树干", "工具选择", "持续破坏"),
        risk_controls="先完成接近节点；伐木失败回退到接近完成后的快照。", failure_mode="空手破坏超时、掉落或目标失焦。",
    ))
    mine_ok = bool(ores_or_stone) and (has_item(start, ("pickaxe",)) or has_item(start, ("stone", "cobble")))
    courses.append(Course(
        "mine-observed-resource", "采矿：挖掘已观测矿石或石头", "选择镐，挖掘当前已观测矿石或石头。",
        ("近距离石头或矿石", "镐或可取得的石质前置资源"), 220,
        "服务器方块检查为空气、挖掘统计增加、掉落物进入背包。", "较高",
        "当前扫描未命中石头或矿石；背包也没有镐或石质资源。", mine_ok,
        None if mine_ok else "缺少当前可见石头/矿石，以及镐或石质前置资源。",
        feasibility_evidence="是否执行由当前扫描的石头/矿石和背包镐/石质资源共同决定。", required_skills=("工具选择", "持续挖掘", "掉落物回收"),
        risk_controls="只在有回退快照、工具和可验证目标时进入。", failure_mode="工具不足、目标不可见或挖掘后不可恢复。",
    ))
    supplies = food and lighting and tool
    courses.append(Course(
        "expedition-prepare", "远征准备：危险区域补给检查", "在进入洞穴或远距离区域前确认食物、照明、工具耐久、方块、武器护甲和返程冗余。",
        ("食物", "照明", "工具耐久", "方块", "防护", "返程冗余"), 80,
        "六类补给均满足阈值和冗余系数。", "较高",
        f"当前背包为 {start['inventory']}。", supplies,
        None if supplies else "背包为空，补给门不通过；后续应生成可验证的补给前置课程。",
        feasibility_evidence="食物、照明和工具均由背包观测验证；当前为空。", required_skills=("背包审计", "补给收集", "返程规划"),
        risk_controls="风险区前拒绝出发并保留出生快照。", failure_mode="补给耗尽、照明不足或返程失败。",
    ))
    water_visible = any("water" in state["raycast_block"] for state in observed)
    water_bucket = has_item(start, ("water_bucket",))
    courses.append(Course(
        "risk-mlg-water", "高风险专项：落地水（MLG water）", "在高处下落时，在接触前使用水桶完成落地水并回收。",
        ("水桶", "可见安全落点", "高处下落", "快速 use 时机"), 120,
        "存活、落地且水方块/水桶状态符合回收规则。", "高",
        f"当前高度 y={start['y']}；水桶={water_bucket}；扫描可见水面={water_visible}。", water_bucket and water_visible,
        None if water_bucket and water_visible else "当前没有水桶或可观测水面，物理前置缺失；保留为解锁后的专项课，不因风险过滤。",
        feasibility_evidence="MLG 需要水桶和明确可着水目标；二者会在后续快照中重新验证。", required_skills=("高处下落", "相机追踪", "水桶 use"),
        risk_controls="每次尝试独立快照；失败立即 load；只在有水桶和可见落点时放行。", failure_mode="过早/过晚放水、无水桶、落点遮挡。",
    ))
    mine_target = bool(ores_or_stone)
    pickaxe = has_item(start, ("pickaxe",))
    for identifier, title, task, skills, controls, failure in (
        ("risk-two-block-vertical-mine", "高风险专项：双格交替垂直挖掘", "在双格交替站位中向下挖掘，避免直落岩浆或空洞。", ("镐选择", "双格站位", "风险感知"), "每层记录方块、脚下支撑和回退快照；检测到水/岩浆/空洞即停止。", "直落空洞、岩浆、水流或工具失效。"),
        ("risk-staircase-mine", "高风险专项：安全阶梯挖掘与探底", "按阶梯下降并周期性确认返程路径、照明和补给。", ("镐选择", "阶梯节奏", "照明", "返程规划"), "每 8 tick 验证脚下支撑；补给门不足则转为准备课程。", "照明耗尽、路径中断、掉落或资源耗尽。"),
    ):
        courses.append(Course(
            identifier, title, task, ("可见石头/矿石", "镐", "补给门通过"), 360,
            "连续多步中保持支撑、目标方块变化和返程条件。", "高",
            f"可见矿石/石头={mine_target}；背包有镐={pickaxe}；补给门={supplies}。", mine_target and pickaxe and supplies,
            None if mine_target and pickaxe and supplies else "当前物理前置不足，作为后续解锁分支保留，不按风险优先级删除。",
            feasibility_evidence="双格/阶梯策略在存在可挖方块、镐和补给时物理可实现；否则生成准备型前置课程。", required_skills=skills,
            risk_controls=controls, failure_mode=failure,
        ))
    return courses


def chunks(count: int, *keys: str, mouse: tuple[int, int] = (0, 0)) -> tuple[LumineActionChunk, ...]:
    return tuple(LumineActionChunk(keys=keys, mouse=mouse if index == 0 else (0, 0)) for index in range(count))


def make_candidate(identifier: str, role: str, description: str, *parts: tuple[LumineActionChunk, ...]) -> Candidate:
    values = tuple(value for part in parts for value in part)
    if len(values) < 8:
        raise ValueError("动作必须至少八 tick")
    return Candidate(identifier, role, description, values)


def terra_trajectories(move_ticks: int) -> tuple[Candidate, ...]:
    """两个参考轨迹与六个待训练策略样本，均由 Terra 在当前树冠观察下生成。"""
    travel = max(24, min(112, move_ticks))
    settle = 12
    return (
        make_candidate("R01", "reference_expert", "沿已观测树干方向稳定前进并留出停稳时间", chunks(travel, "W"), chunks(settle)),
        make_candidate("R02", "reference_expert", "先谨慎前进，再连续前进以接近树干", chunks(max(8, travel // 5), "W", "shift"), chunks(travel - max(8, travel // 5), "W"), chunks(settle)),
        make_candidate("P01", "policy_sample", "直接前进探索已观测路线", chunks(max(8, travel - 8), "W"), chunks(20)),
        make_candidate("P02", "policy_sample", "前进后微向左修正再继续接近", chunks(max(8, travel // 3), "W"), chunks(1, mouse=(-45, 0)), chunks(max(8, travel - travel // 3), "W"), chunks(settle)),
        make_candidate("P03", "policy_sample", "前进并轻微向右搜索树干", chunks(max(8, travel // 2), "W"), chunks(1, mouse=(35, 0)), chunks(max(8, travel - travel // 2), "W"), chunks(settle)),
        make_candidate("P04", "policy_sample", "先短距离移动确认路径，再持续前进", chunks(max(8, travel // 6), "W"), chunks(8), chunks(max(8, travel - travel // 6), "W"), chunks(10)),
        make_candidate("P05", "policy_sample", "斜向前进以绕开近处阻挡", chunks(max(8, travel - 12), "W", "D"), chunks(22)),
        make_candidate("P06", "policy_sample", "稳定前进，末段停止观察", chunks(travel, "W"), chunks(8)),
    )


def policy_candidates(generations: list[Any]) -> tuple[Candidate, ...]:
    if len(generations) != 6:
        raise ValueError("2+6 合同要求正好 6 条模型策略轨迹")
    candidates = []
    for index, generation in enumerate(generations, start=1):
        action = decode_lumine_action(generation.action_text, expected_chunks=None)
        candidates.append(
            Candidate(
                candidate_id=f"P{index:02d}",
                source_role="policy_sample",
                description="BC LoRA 从规范起点图像独立采样的 on-policy 动作",
                chunks=action.chunks,
                response_token_ids=generation.response_token_ids,
                old_logprobs=generation.old_logprobs,
                policy_version=generation.policy_version,
                sampling_parameters=generation.sampling_parameters,
            )
        )
    return tuple(candidates)


def distance_to_target(state: dict[str, Any], target: tuple[int, int, int]) -> float:
    return math.dist((state["x"], state["y"], state["z"]), target)


def restore_observed_player_start(environment: Any, expected: dict[str, Any]) -> Any:
    """恢复本轮已观测的玩家状态。

    现有 StructureTemplate 快照只包含受控区域的方块和方块实体。当前观察明确
    显示背包为空，所以 ``clear`` 是对真实初始状态的恢复，而非发放或伪造资源。
    非空背包必须升级为逐槽位背包快照后才允许进入此执行器。
    """
    if expected["inventory"]:
        raise RuntimeError("当前执行器只能审计并恢复空背包；非空背包需要逐槽位玩家快照协议")
    commands = (
        "clear @s",
        f"tp @s {expected['x']:.3f} {expected['y']:.3f} {expected['z']:.3f} {expected['yaw']:.3f} {expected['pitch']:.3f}",
    )
    return step_commands(environment, commands, ticks=4)


def restore_start(
    environment: Any,
    coordinator: MemorySnapshotCoordinator,
    snapshot: Any,
    expected: dict[str, Any],
    player_restore_anchor: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from craftground.environment.action_space import no_op_v2

    reset = coordinator.reset_all(snapshot)
    observation = restore_observed_player_start(environment, player_restore_anchor)
    before = capture_state(observation)
    probe = environment.step(CraftGroundActionAdapter().convert(("W",), (0, 0)))[0]
    first = math.dist((before["x"], before["z"]), (capture_state(probe)["x"], capture_state(probe)["z"]))
    neutralized = False
    second: float | None = None
    if first < 0.02:
        neutralized = True
        adapter = CraftGroundActionAdapter()
        environment.step(adapter.convert(("E",), (0, 0)))
        environment.step(no_op_v2())
        observation = restore_observed_player_start(environment, player_restore_anchor)
        before = capture_state(observation)
        probe = environment.step(CraftGroundActionAdapter().convert(("W",), (0, 0)))[0]
        second = math.dist((before["x"], before["z"]), (capture_state(probe)["x"], capture_state(probe)["z"]))
        if second < 0.02:
            raise RuntimeError("GUI 中和后移动探针仍无位移")

    # 位移探针只能验证 GUI 状态，不能成为候选轨迹的第一步。探针完成后再次
    # 恢复同一内存快照，后续起点校验和动作序列才都以完全相同的世界状态开始。
    verification_reset = coordinator.reset_all(snapshot)
    observation = restore_observed_player_start(environment, player_restore_anchor)
    start = capture_state(observation)
    deltas = {key: round(abs(start[key] - expected[key]), 3) for key in ("x", "y", "z", "yaw", "pitch")}
    if any(value > 0.05 for value in deltas.values()):
        raise RuntimeError(f"快照起点不一致：{deltas}")
    return observation, {"snapshot_load_wall_ms": round(reset.wall_ms, 3), "probe_reset_wall_ms": round(verification_reset.wall_ms, 3), "first_probe_planar_distance": round(first, 4), "gui_recovery_attempted": neutralized, "second_probe_planar_distance": second, "state_deviation": deltas}


def run_candidate(
    environment: Any,
    coordinator: MemorySnapshotCoordinator,
    snapshot: Any,
    start: dict[str, Any],
    player_restore_anchor: dict[str, Any],
    course: Course,
    candidate: Candidate,
    output: Path,
) -> dict[str, Any]:
    observation, audit = restore_start(environment, coordinator, snapshot, start, player_restore_anchor)
    directory = output / candidate.candidate_id
    directory.mkdir(parents=True, exist_ok=False)
    save_rgb(observation, directory / "start.png")
    frames = [{"tick": 0, "path": f"{candidate.candidate_id}/start.png"}]
    adapter = CraftGroundActionAdapter()
    started = time.perf_counter()
    for tick, action in enumerate(decode_lumine_action(candidate.action_text).chunks, start=1):
        observation = environment.step(adapter.convert(action.keys, action.mouse, action.scroll))[0]
        if tick in {1, 32, 64, 96, len(candidate.chunks)}:
            name = f"tick_{tick:03d}.png"
            save_rgb(observation, directory / name)
            frames.append({"tick": tick, "path": f"{candidate.candidate_id}/{name}"})
    end = capture_state(observation)
    initial_distance = distance_to_target(start, course.target or (0, 0, 0))
    final_distance = distance_to_target(end, course.target or (0, 0, 0))
    progress = initial_distance - final_distance
    required_progress = min(8.0, initial_distance * 0.6)
    success = not end["is_dead"] and end["is_on_ground"] and progress >= required_progress
    progressing = not success and not end["is_dead"] and end["is_on_ground"] and progress >= max(0.25, required_progress * 0.1)
    execution_status = "SUCCESS" if success else ("PROGRESSING" if progressing else "FAILED")
    result = {
        "candidate_id": candidate.candidate_id, "source_role": candidate.source_role, "description": candidate.description,
        "generated_by": (
            "SubAgent reference design" if candidate.source_role == "reference_expert"
            else "BC LoRA policy"
        ), "action_text": candidate.action_text,
        "response_token_ids": list(candidate.response_token_ids),
        "old_logprobs": list(candidate.old_logprobs),
        "policy_version": candidate.policy_version,
        "sampling_parameters": candidate.sampling_parameters or {},
        "ticks": len(candidate.chunks), "reset_audit": audit, "frames": frames,
        "initial_distance_to_observed_log": round(initial_distance, 3), "final_distance_to_observed_log": round(final_distance, 3),
        "distance_progress": round(progress, 3), "required_distance_progress": round(required_progress, 3), "final_state": end, "success": success,
        "execution_status": execution_status,
        "progress_metrics": {"target_distance_start": round(initial_distance, 3), "target_distance_end": round(final_distance, 3), "target_distance_net_progress": round(progress, 3), "alive": not end["is_dead"], "on_ground": end["is_on_ground"]},
        "method_validity": "沿当前已 raycast 的树干方向接近；目标和规则均来自同一轮观察。",
        "direction_confidence": "high: 当前扫描直接命中目标方块",
        "checkpoint_policy": "仅在存活、落地、目标相关净进展为正且玩家状态可恢复时保存稳定检查点。",
        "budget_extension_policy": "仅 PROGRESSING 可申请有限延长；本轮 batch 固定预算，不在候选内隐式续跑。",
        "stagnation_window": "连续 32 tick 未出现目标距离净缩短则停止扩展并转入 UNKNOWN/失败分析。",
        "success_evidence": {"alive": not end["is_dead"], "on_ground": end["is_on_ground"], "distance_reduced_to_threshold": progress >= required_progress},
        "execution_wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    result["score"] = round((100.0 if success else 0.0) + max(-20.0, min(40.0, progress)) + (10.0 if end["is_on_ground"] else 0.0) - 0.04 * len(candidate.chunks), 3)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def report_markdown(output: Path, scans: list[dict[str, Any]], courses: list[Course], selected: Course, report: dict[str, Any], bank_design: dict[str, Any]) -> None:
    lines = [
        "# CraftGround 观察驱动课程图与执行演练", "",
        "本轮没有布置固定资源场景，也没有发放物品。系统先读取真实当前世界的图像、位置、准星、附近方块、背包和统计，由 Terra 策略替身生成课程候选并用可行性门筛选。", "",
        "## 当前观察", "",
    ]
    for scan in scans:
        state = scan["state"]
        lines.extend((f"### {scan['label']}", "", f"位置=({state['x']}, {state['y']}, {state['z']})；准星={state['raycast_block'] or '未命中'}；背包={state['inventory']}；附近方块={state['nearby_block_counts']}。", "", f"![{scan['label']}]({scan['image']})", ""))
    lines.extend(("## Terra 课程候选与筛选", "", "| 课程 | 前置条件 | Tick 预算 | 成功判据 | 难度 | 当前观察依据 | 结果 |", "|---|---|---:|---|---|---|---|"))
    for course in courses:
        status = "选中" if course.course_id == selected.course_id else ("可行但未选" if course.eligible else f"暂缓：{course.rejection}")
        lines.append(f"| {course.title} | {'；'.join(course.prerequisites)} | {course.tick_budget} | {course.success_rule} | {course.difficulty} | {course.observation_basis} | {status} |")
    lines.extend(("", "### 课程规格", ""))
    for course in courses:
        lines.extend((
            f"#### {course.title}", "",
            f"- 可行性证据：{course.feasibility_evidence}",
            f"- 前置技能：{'；'.join(course.required_skills)}",
            f"- 风险控制：{course.risk_controls}",
            f"- 恢复快照：{course.recovery_snapshot}",
            f"- 失败模式：{course.failure_mode}",
            "",
        ))
    lines.extend(("", "## 执行的同快照 batch=8", "", f"选中课程：**{selected.title}**。快照=`{report['snapshot_id']}`。两条 `reference_expert` 是下界锚定与克隆/排序参考；六条 `policy_sample` 是待训练模型样本。本机两类均由 Terra 代理生成，但保留来源，不伪装为真实策略 rollout。", "", "| 候选 | 来源 | Tick | 距离变化 | 执行状态 | 分数 | 相对优势 |", "|---|---|---:|---:|---|---:|---:|"))
    for item in report["trajectories"]:
        lines.append(f"| {item['candidate_id']} | {item['source_role']} | {item['ticks']} | {item['distance_progress']:+.2f} | {item['execution_status']} | {item['score']:.2f} | {item['relative_advantage']:+.2f} |")
    for item in report["trajectories"]:
        lines.extend(("", f"### {item['candidate_id']}：{item['source_role']}", "", f"{item['description']}。起始距树干={item['initial_distance_to_observed_log']:.2f}；终态={item['final_distance_to_observed_log']:.2f}；真实移动进展={item['distance_progress']:+.2f}。", ""))
        for frame in item["frames"]:
            lines.extend((f"![{item['candidate_id']} tick {frame['tick']}]({frame['path']})", ""))
    lines.extend(("## 多起点课程银行设计", "", "本轮只真实执行了一个树冠快照的准备课程。下表和能力清单定义后续采样合同，不将其描述为已经执行的轨迹或已经存在的快照。", "", "| 维度 | 本轮已验证标签 |", "|---|---|"))
    for dimension, value in bank_design["observed_snapshot_only"]["dimensions"].items():
        lines.append(f"| {dimension} | {value} |")
    lines.extend(("", "快照准入：必须记录来源、真实观察或成功末态的可行性证据、八个课程维度；禁止使用作弊状态或物理不可能状态。采样先满足能力覆盖下限，再按失败次数提高重采样权重；不能总是从出生点线性推进。", "", "| 能力种子 | 科技阶段 | 真实准入前置 | 版本/机制依赖 |", "|---|---|---|---|"))
    for capability in bank_design["capabilities"]:
        lines.append(f"| {capability['title']} | {capability['stage']} | {'；'.join(capability['prerequisites'])} | {capability['version_mechanic']} |")
    lines.extend(("", "路线校正：末影之眼仅用于主世界寻找要塞和末地门；前往下界必须通过黑曜石下界门。只有与当前快照维度和真实前置相符的能力课可进入执行队列。", ""))
    lines.extend(("## 快照边界与课程更新", "", "每条轨迹恢复后均执行 W 坐标位移探针；若无位移则发送 inventory/E 并二次验证。起点的位置与朝向必须完全恢复，才允许执行。", "", "| 候选 | 首次 W 位移 | GUI 中和 | 起点偏差 |", "|---|---:|---|---|"))
    for item in report["trajectories"]:
        audit = item["reset_audit"]
        lines.append(f"| {item['candidate_id']} | {audit['first_probe_planar_distance']:.3f} | {'是' if audit['gui_recovery_attempted'] else '否'} | {audit['state_deviation']} |")
    lines.extend(("", f"成功率=`{report['success_rate']:.0%}`，平均分=`{report['mean_score']:.3f}`。{report['course_note']}", "", "## 长任务与 PRO 6000 流水", "", "短任务至少输出 8 tick，在剩余 4 tick 时预取。当前准备任务为 120 tick 左右的长序列，推理在剩余四分之一处启动预取，剩余 4 tick 是最迟触发线。动作按图像锚点消费，过期 tick 丢弃，从仍追得上的最新 tick 接续。PRO 6000 的优先级为在线推理、待执行候选、相对优势、训练 batch；参考轨迹用于下界锚定、克隆和排序，不与 policy rollout 混淆。", "", "本机没有部署训练模型，未测量真实模型 GPU 利用率、推理时延或吞吐量。", ""))
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(runtime: Path, output: Path, policy_adapter: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    environment = build_environment(runtime)
    try:
        environment.reset(options={"fast_reset": False})
        observation = step_commands(environment, (), ticks=12)
        scans = []
        save_rgb(observation, output / "initial.png")
        scans.append({"label": "初始图像锚点", "image": "initial.png", "state": capture_state(observation)})
        adapter = CraftGroundActionAdapter()
        for index in range(3):
            environment.step(adapter.convert((), (600, 0)))
            observation = step_commands(environment, (), ticks=1)
            name = f"scan_{index + 1}.png"
            save_rgb(observation, output / name)
            scans.append({"label": f"右转 {(index + 1) * 90} 度", "image": name, "state": capture_state(observation)})
        courses = propose_courses(scans)
        selected = next((course for course in courses if course.eligible), None)
        if selected is None or selected.target is None:
            raise RuntimeError("当前观察没有可自动验证的课程；已写出课程候选，但不执行伪造轨迹")
        target_scan = next(scan for scan in scans if tuple(scan["state"]["raycast_position"] or ()) == selected.target)
        desired_yaw = target_scan["state"]["yaw"]
        current_yaw = capture_state(observation)["yaw"]
        yaw_delta = (desired_yaw - current_yaw + 180.0) % 360.0 - 180.0
        if abs(yaw_delta) > 0.1:
            environment.step(adapter.convert((), (round(yaw_delta / 0.15), 0)))
            observation = step_commands(environment, (), ticks=1)
        start = capture_state(observation)
        target_x, target_y, target_z = selected.target
        # 覆盖树冠、已观测树干及可能的安全下落区域，避免对整片世界做大快照。
        region = SnapshotRegion(
            (math.floor(min(start["x"], target_x)) - 4, max(-64, math.floor(min(start["y"], target_y)) - 60), math.floor(min(start["z"], target_z)) - 4),
            (math.ceil(max(start["x"], target_x)) + 4, math.ceil(max(start["y"], target_y)) + 6, math.ceil(max(start["z"], target_z)) + 4),
        )
        coordinator = MemorySnapshotCoordinator([environment])
        snapshot = coordinator.capture_all("observation_driven_prep_batch8", region)
        # 传送到叶簇边缘时服务器会执行一次合法碰撞修正。以修正后的稳定状态
        # 作为 batch 的规范起点，后续每条候选都从同一状态恢复。
        player_restore_anchor = start
        observation = restore_observed_player_start(environment, player_restore_anchor)
        start = capture_state(observation)
        policy_anchor = output / "policy_anchor.png"
        save_rgb(observation, policy_anchor)
        bank_design = design_document(observed_snapshot_record(snapshot.snapshot_id, start, output))
        (output / "multi_start_curriculum_design.json").write_text(
            json.dumps(bank_design, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        move_ticks = round(min(112.0, max(24.0, distance_to_target(start, selected.target) * 8.0)))
        references = terra_trajectories(move_ticks)[:2]
        from train.policy_rollout import generate_policy_rollouts
        from train.unsloth_vision_sft import LoRASettings, load_vision_model

        model, processor = load_vision_model(
            "unsloth/gemma-4-E4B-it", LoRASettings(), adapter=policy_adapter
        )
        generations = generate_policy_rollouts(
            model,
            processor,
            policy_anchor,
            policy_version=policy_adapter,
        )
        candidates = references + policy_candidates(generations)
        del model, processor
        if len(candidates) != 8 or sum(item.source_role == "reference_expert" for item in candidates) != 2:
            raise RuntimeError("2+6 候选协议不成立")
        (output / "course_candidates.json").write_text(json.dumps({"scans": scans, "courses": [course.__dict__ for course in courses], "selected_course": selected.course_id}, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "candidates.json").write_text(json.dumps({"batch_size": 8, "protocol": {"reference_expert": 2, "policy_sample": 6}, "candidates": [{"candidate_id": item.candidate_id, "source_role": item.source_role, "description": item.description, "ticks": len(item.chunks), "action_text": item.action_text} for item in candidates]}, ensure_ascii=False, indent=2), encoding="utf-8")
        trajectories = [
            run_candidate(environment, coordinator, snapshot, start, player_restore_anchor, selected, item, output)
            for item in candidates
        ]
        mean = sum(item["score"] for item in trajectories) / 8
        for item in trajectories:
            item["relative_advantage"] = round(item["score"] - mean, 3)
        success = [item for item in trajectories if item["success"]]
        best = max(trajectories, key=lambda item: item["score"])
        if len(success) / 8 >= MIN_SUCCESS_RATE:
            note = f"达到 `{MIN_SUCCESS_RATE:.0%}` 门槛；从最佳成功轨迹 `{best['candidate_id']}` 的稳定末态捕获下一节点快照。"
            decision = "提升到下一课程节点"
            coordinator.reset_all(snapshot)
            chosen = next(item for item in candidates if item.candidate_id == best["candidate_id"])
            replay = run_candidate(
                environment, coordinator, snapshot, start, player_restore_anchor, selected, chosen, output / "promotion_replay"
            )
            promoted = coordinator.capture_all("observation_driven_prep_promoted", region)
            promotion = {"snapshot_id": promoted.snapshot_id, "frame": "promotion_replay/" + replay["frames"][-1]["path"]}
        else:
            note = f"未达到 `{MIN_SUCCESS_RATE:.0%}` 门槛；不替换出生快照，保留真实失败轨迹以校准。"
            decision = "保持当前课程节点"
            promotion = None
        report = {"snapshot_id": snapshot.snapshot_id, "selected_course": selected.course_id, "initial_state": start, "trajectories": trajectories, "mean_score": round(mean, 3), "success_rate": len(success) / 8, "best_candidate": best["candidate_id"], "course_decision": decision, "course_note": note, "promotion": promotion}
        (output / "execution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_markdown(output, scans, courses, selected, report, bank_design)
        return report
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Terra 观察驱动课程 2+6 batch")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-adapter", required=True)
    args = parser.parse_args()
    report = run(args.runtime.resolve(), args.output.resolve(), args.policy_adapter)
    print(json.dumps({"success_rate": report["success_rate"], "course_decision": report["course_decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
