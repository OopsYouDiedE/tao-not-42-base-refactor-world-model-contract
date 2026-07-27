# -*- coding: utf-8 -*-
"""手写轨迹录制器后端：持有常驻 CraftGround env + HTTP 接口 + 宏编译/步进/重置/导出。

设计（见记忆 craftground-recorder-capability-probe-2026-07-25 与 plans 文件）：
  - 冷启动**一个**常驻 CraftGroundEnvironment（≈30s，整个录制会话只付一次；绝不中途
    close/new，否则又冷启动）。之后所有"重开"走亚秒级 fast_reset。
  - CraftGround 是动作驱动器：env.step(V2 dict) 直接吃 22 键动作 = 模型动作空间本身。
  - 录制语义：作者用宏命令写轨迹 → 逐 tick 展开（macro_action_compiler）→「步进」执行到
    下一观察点并存该帧 →「重置」回退到轨迹初始态。观察点 = 一次推理边界 = 训练样本切点。
  - 分辨率对齐：与所有 craftground 入口一致的 640×360（从 observation_spaces.OBS_SHAPE_NATIVE 派生）。
  - mc 命令发送通道：fast_reset 用 extra_commands；轨迹内即时命令用 env.add_command()
    （queued_commands 随下一 step 发；实测 action dict 的 commands 键不生效，故不用它）。

对外接口：
    python -m rl_training_environments.craftground.trajectory_recorder_server
        [--http-port 8897] [--world normal|flat] [--gamemode survival|creative]
        [--max-blind-ticks 300] [--extra-command "give @p ..."]...

    浏览器打开 http://127.0.0.1:<port>/ 使用 recorder_ui.html 界面。
HTTP 端点见 _RecorderRequestHandler.do_GET / do_POST。
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit

import cv2
import numpy as np

from rl_training_environments.craftground.macro_action_compiler import (
    CAM_TICKS,
    CAMERA_DELTA,
    CAMERA_NONE,
    CAMERA_SCREEN,
    MacroCommand,
    TAP_TICKS,
    compile_macros,
)
from rl_training_environments.craftground.recorder_macros import macro_from_dict
from rl_training_environments.craftground.observation_spaces import OBS_SHAPE_NATIVE

# 分辨率对齐口径：OBS_SHAPE_NATIVE = (C, H, W) = (3, 360, 640)，与 solaris viewer 一致。
NATIVE_HEIGHT = OBS_SHAPE_NATIVE[1]
NATIVE_WIDTH = OBS_SHAPE_NATIVE[2]

# 录制器版本号：随录制语义/导出 schema 的破坏性变更递增（v3 = 回合模型 + md/json 导出）。
# 界面 header 展示，导出产物写入 provenance。
RECORDER_VERSION = 3

HERE = Path(__file__).resolve().parent
UI_HTML_PATH = HERE / "recorder_ui.html"


def _describe_macro_spec(spec: Dict) -> str:
    """把一条宏 spec 渲染成人读一行（导出 md 的回合表用）。"""
    if spec.get("kind") == "mc":
        return f"mc: `{spec.get('command', '')}`"
    parts: List[str] = []
    holds = spec.get("holds") or {}
    if holds:
        parts.append("长按 " + ", ".join(f"{key}×{ticks}t" for key, ticks in holds.items()))
    clicks = spec.get("clicks") or []
    if clicks:
        parts.append("点按 " + "+".join(clicks) + f"×{TAP_TICKS}t")
    release = spec.get("release") or []
    if release:
        parts.append("松开 " + ", ".join(release))
    camera_mode = spec.get("camera_mode", CAMERA_NONE)
    if camera_mode == CAMERA_DELTA:
        parts.append(f"相机Δ yaw={spec.get('delta_yaw', 0)}, pitch={spec.get('delta_pitch', 0)}"
                     f"×{CAM_TICKS}t" + ("(GUI光标)" if spec.get("gui_cursor") else ""))
    elif camera_mode == CAMERA_SCREEN:
        parts.append(f"相机→屏幕({spec.get('screen_x', 0)}, {spec.get('screen_y', 0)})"
                     f"×{CAM_TICKS}t")
    wait_ticks = spec.get("wait_ticks", 0)
    if wait_ticks:
        parts.append(f"等待 {wait_ticks}t")
    return " · ".join(parts) if parts else "（空）"


class TrajectoryRecorder:
    """持有常驻 CraftGround env 与录制状态；步进/重置/导出的实现在这里。

    线程安全：所有会改状态的操作用一把锁串行化（HTTP handler 多线程，但录制本质单会话）。
    """

    def __init__(self, world_type: str = "normal", gamemode: str = "survival",
                 seed: str = "42", extra_commands: Optional[List[str]] = None,
                 max_blind_ticks: int = 300, port: int = 8000,
                 difficulty: str = "normal", map_dir_path: str = "",
                 level_display_name: str = ""):
        self._lock = threading.Lock()
        self.world_type_name = world_type
        self.gamemode = gamemode
        self.seed = seed
        self.max_blind_ticks = max_blind_ticks
        self.difficulty = difficulty
        # 从存档加载（可选）：非空则冷启动时从该目录加载指定世界（见记忆
        # craftground-reset-and-save-facts；saveAll 被 mod 禁用，存档天然只读）。
        self.map_dir_path = map_dir_path
        self.level_display_name = level_display_name
        self.env_port = port
        # 初始 extra_commands：设定 gamemode + 用户预置（工作台/熔炉/酿造台/给料等）。
        self.initial_extra_commands = [f"gamemode {gamemode} @p"] + list(extra_commands or [])

        # 录制状态
        self.macros: List[MacroCommand] = []
        self.manual_observation_ticks: List[int] = []
        self.cursor_tick = 0                       # 已盲执行到的 tick
        self.observation_frames: Dict[int, np.ndarray] = {}   # 观察点 tick -> BGR 帧
        self.current_frame_bgr: Optional[np.ndarray] = None
        self._compiled = None

        # 冷启动常驻 env（仅一次）
        self._env = self._cold_start_env(port)
        self._reset_world_to_initial()

    # ── env 生命周期 ──────────────────────────────────────────────────────
    def _cold_start_env(self, port: int):
        """冷启动一个常驻 CraftGroundEnvironment（整个会话只调一次）。"""
        from craftground import CraftGroundEnvironment
        from craftground.initial_environment_config import (
            Difficulty,
            GameMode,
            InitialEnvironmentConfig,
            WorldType,
        )
        from craftground.environment.action_space import ActionSpaceVersion
        from craftground.screen_encoding_modes import ScreenEncodingMode

        world_type = WorldType.SUPERFLAT if self.world_type_name == "flat" else WorldType.DEFAULT
        game_mode = GameMode.SURVIVAL if self.gamemode == "survival" else GameMode.CREATIVE
        difficulty = {
            "peaceful": Difficulty.PEACEFUL, "easy": Difficulty.EASY,
            "normal": Difficulty.NORMAL, "hard": Difficulty.HARD,
        }.get(self.difficulty, Difficulty.NORMAL)

        config = InitialEnvironmentConfig(
            image_width=NATIVE_WIDTH, image_height=NATIVE_HEIGHT,
            world_type=world_type,
            gamemode=game_mode,
            difficulty=difficulty,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            seed=self.seed,
            initial_extra_commands=list(self.initial_extra_commands),
            map_dir_path=self.map_dir_path,
            level_display_name_to_play=self.level_display_name,
        )
        env = CraftGroundEnvironment(
            config, action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
            port=port, find_free_port=True, verbose=False,
        )
        obs, _ = env.reset()      # 冷启动
        self.current_frame_bgr = self._obs_to_bgr(obs)
        return env

    def _reset_world_to_initial(self):
        """死亡重置：fast_reset（/kill @p 重生）+ 重跑初始 extra_commands（亚秒级）。

        只重置玩家，不重置世界方块——若前面挖改过地形，方块改动仍在。要回到干净世界
        用 _cold_restart_world（完全重置）。
        """
        obs, _ = self._env.reset(options={
            "fast_reset": True,
            "extra_commands": list(self.initial_extra_commands),
        })
        # 读几帧让初始命令（gamemode/给料）同步
        for _ in range(4):
            obs = self._env.step(self._noop())[0]
        self.current_frame_bgr = self._obs_to_bgr(obs)

    def _cold_restart_world(self):
        """完全重置：fast_reset=False 冷重开（~30s），世界从 config（含 map_dir_path）重载。

        terminate 掉 MC 客户端再 start_server，世界回到干净初始态（会话内改动因 mod 的
        SaveWorldMixin 禁用 saveAll 而从未落盘，见记忆 craftground-reset-and-save-facts）。
        """
        obs, _ = self._env.reset(options={
            "fast_reset": False,
            "extra_commands": list(self.initial_extra_commands),
        })
        # 读几帧让初始命令（gamemode/给料）同步
        for _ in range(4):
            obs = self._env.step(self._noop())[0]
        self.current_frame_bgr = self._obs_to_bgr(obs)

    def _noop(self):
        from craftground.environment.action_space import no_op_v2
        return dict(no_op_v2())

    # ── 观测转换 ──────────────────────────────────────────────────────────
    @staticmethod
    def _obs_to_bgr(obs) -> np.ndarray:
        """CraftGround obs（dict，rgb 键）→ OpenCV BGR uint8 帧。"""
        rgb = obs["rgb"] if isinstance(obs, dict) else obs
        frame = np.ascontiguousarray(rgb)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── 录制操作 ──────────────────────────────────────────────────────────
    def add_macro(self, spec: Dict):
        with self._lock:
            self.macros.append(macro_from_dict(spec))
            self._compiled = None

    def remove_macro(self, index: int):
        with self._lock:
            if 0 <= index < len(self.macros):
                self.macros.pop(index)
                self._compiled = None

    def insert_observation_point(self, tick: int):
        """在指定 tick 手插观察点（界面时间线点击）。"""
        with self._lock:
            self.manual_observation_ticks.append(int(tick))
            self._compiled = None

    def _compile(self):
        if self._compiled is None:
            self._compiled = compile_macros(
                self.macros, max_blind_ticks=self.max_blind_ticks,
                manual_observation_ticks=self.manual_observation_ticks,
            )
        return self._compiled

    def _next_observation_after(self, tick: int) -> Optional[int]:
        """返回严格大于 tick 的最近观察点索引；没有则 None。"""
        compiled = self._compile()
        for obs_tick in compiled.observation_tick_indices:
            if obs_tick > tick:
                return obs_tick
        return None

    def step_to_next_observation(self) -> Dict:
        """执行从当前游标到下一观察点的 tick 段（盲执行），存到达帧。返回状态摘要。"""
        with self._lock:
            compiled = self._compile()
            target = self._next_observation_after(self.cursor_tick)
            if target is None:
                return {"done": True, "message": "已到轨迹末尾，无更多观察点",
                        **self._state_unlocked()}

            for tick_index in range(self.cursor_tick, target):
                # 附着在该 tick 之前的 mc 命令：走 add_command（queued，随下一 step 发）
                pending = compiled.tick_commands.get(tick_index)
                if pending:
                    for command in pending:
                        self._env.add_command(command)
                action = dict(compiled.tick_actions[tick_index])
                obs = self._env.step(action)[0]

            self.current_frame_bgr = self._obs_to_bgr(obs)
            self.observation_frames[target] = self.current_frame_bgr.copy()
            self.cursor_tick = target
            return {"done": target >= compiled.total_ticks, **self._state_unlocked()}

    def reset_death(self) -> Dict:
        """死亡重置：fast_reset 回初始态（亚秒）+ 游标归零 + 清观察帧。

        只重置玩家，不重置世界方块。适合"重来一遍这段操作"，前面挖改的地形保留。
        """
        with self._lock:
            self._reset_world_to_initial()
            self.cursor_tick = 0
            self.observation_frames.clear()
            return self._state_unlocked()

    def reset_full(self) -> Dict:
        """完全重置（从存档加载）：冷重开 env（~30s）+ 游标归零 + 清观察帧。

        世界回到干净初始态（config/存档定义），前面所有地形改动清空。
        """
        with self._lock:
            self._cold_restart_world()
            self.cursor_tick = 0
            self.observation_frames.clear()
            return self._state_unlocked()

    def replay_from_start_to(self, target_tick: int) -> Dict:
        """从初始态确定性重放到 target_tick（交互跳转到任意观察点用）。

        用 fast_reset（亚秒）保证交互流畅：玩家位置/视角准确重放，但世界方块沿用当前会话
        状态——survival 下前面挖改过的地形不会回滚，重放画面可能偏。要从干净世界精确重现，
        先点"完全重置"再重放；导出（export）已固定走冷重开保证最终产出正确。
        """
        with self._lock:
            self._reset_world_to_initial()
            self.cursor_tick = 0
            self.observation_frames.clear()
            compiled = self._compile()
            obs = None
            for tick_index in range(0, target_tick):
                pending = compiled.tick_commands.get(tick_index)
                if pending:
                    for command in pending:
                        self._env.add_command(command)
                obs = self._env.step(dict(compiled.tick_actions[tick_index]))[0]
                if (tick_index + 1) in compiled.observation_tick_indices:
                    self.observation_frames[tick_index + 1] = self._obs_to_bgr(obs).copy()
            if obs is not None:
                self.current_frame_bgr = self._obs_to_bgr(obs)
            self.cursor_tick = target_tick
            return self._state_unlocked()

    # ── 即时命令 ──────────────────────────────────────────────────────────
    def send_immediate_command(self, command: str) -> Dict:
        """发一条即时 Minecraft 命令到活 env（**只影响当前预览、不进导出、不可复现**）。

        env.add_command 是 queued（随下一次 step 才 flush），故主动 step 一个 noop 触发，
        再读几帧让结果反映到画面。这会推进活 env 若干 tick 但不动逻辑游标，故清空已存观察帧
        以示预览已失真——即时命令只用于步进前（cursor==0）搭建世界或调试，绝不进导出轨迹。
        序列内可复现的世界搭建请用 kind=="mc" 的宏（进 trajectory）。

        Parameters
        ----------
        command : str
            原始 Minecraft 命令（如 "time set day"、"setblock ~ ~ ~1 chest"）。

        Returns
        -------
        Dict
            状态摘要 + {"immediate_command": command}。
        """
        command = str(command).strip()
        if not command:
            return {"error": "即时命令不能为空"}
        with self._lock:
            self._env.add_command(command)
            # step 1 个 noop 触发 flush，再读 3 帧让结果反映到画面。
            obs = None
            for _ in range(4):
                obs = self._env.step(self._noop())[0]
            if obs is not None:
                self.current_frame_bgr = self._obs_to_bgr(obs)
            # 即时命令推进了活 env、施加了不在轨迹里的改动：已存观察帧作废。
            self.observation_frames.clear()
            return {"immediate_command": command, **self._state_unlocked()}

    # ── 导出 ─────────────────────────────────────────────────────────────
    def export(self, name: str) -> Dict:
        """导出轨迹：runs/craftground-trajectories/<name>/ 下 trajectory.md + .json + 观察点帧 + mp4。

        - trajectory.md：人读——元数据 + 回合表 + 观察点列表。
        - trajectory.json：机读——回合 spec 列表 + 观察点索引 + 段 + provenance，**不含逐 tick
          展开**（回合序列可确定性重算逐 tick，下游读 json 后自行跑 compile_macros）。
        - observation_frames/*.png + frames.mp4：从**干净初始态**（冷重开）完整重放存帧。
          用冷重开而非 fast_reset：survival 下若前面挖改过地形，fast_reset 不重置方块会导致
          重放偏差；冷重开保证导出轨迹从干净世界确定性重现。
        """
        with self._lock:
            compiled = self._compile()
            macro_specs = [self._macro_to_dict(m) for m in self.macros]
            out_dir = Path("runs") / "craftground-trajectories" / name
            frames_dir = out_dir / "observation_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # 从干净初始态完整重放，逐 tick 写 mp4，观察点存 png（仅用于产出帧，不写进 json）。
            self._cold_restart_world()
            writer = cv2.VideoWriter(
                str(out_dir / "frames.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                20.0, (NATIVE_WIDTH, NATIVE_HEIGHT),
            )
            obs_index_set = set(compiled.observation_tick_indices)
            saved_points = []
            obs = None
            for tick_index in range(compiled.total_ticks):
                pending = compiled.tick_commands.get(tick_index)
                if pending:
                    for command in pending:
                        self._env.add_command(command)
                obs = self._env.step(dict(compiled.tick_actions[tick_index]))[0]
                bgr = self._obs_to_bgr(obs)
                writer.write(bgr)
                if (tick_index + 1) in obs_index_set:
                    fname = f"obs_{tick_index + 1:06d}.png"
                    cv2.imwrite(str(frames_dir / fname), bgr)
                    saved_points.append({"observation_tick": tick_index + 1, "frame": fname})
            writer.release()

            trajectory = {
                "name": name,
                "recorder_version": RECORDER_VERSION,
                "resolution": [NATIVE_WIDTH, NATIVE_HEIGHT],
                "world_type": self.world_type_name,
                "gamemode": self.gamemode,
                "difficulty": self.difficulty,
                "seed": self.seed,
                "initial_extra_commands": self.initial_extra_commands,
                "max_blind_ticks": self.max_blind_ticks,
                "total_ticks": compiled.total_ticks,
                "observation_tick_indices": compiled.observation_tick_indices,
                "segments": compiled.segment_bounds(),
                "macros": macro_specs,
                "observation_frames": saved_points,
            }
            (out_dir / "trajectory.json").write_text(
                json.dumps(trajectory, ensure_ascii=False, indent=1), encoding="utf-8")
            (out_dir / "trajectory.md").write_text(
                self._render_trajectory_markdown(name, compiled, macro_specs, saved_points),
                encoding="utf-8")

            # 重放后把游标复位（导出不改变作者的当前编辑位）
            self.cursor_tick = 0
            self.observation_frames.clear()
            return {"exported_to": str(out_dir.resolve()),
                    "observation_points": len(saved_points),
                    "total_ticks": compiled.total_ticks}

    def _render_trajectory_markdown(self, name, compiled, macro_specs, saved_points) -> str:
        """把轨迹渲染成人读 Markdown（元数据 + 回合表 + 观察点列表；不逐 tick 展开）。"""
        lines = [
            f"# CraftGround 轨迹 `{name}`",
            "",
            f"- recorder_version: {RECORDER_VERSION}",
            f"- world: {self.world_type_name} | gamemode: {self.gamemode} | "
            f"difficulty: {self.difficulty} | seed: {self.seed}",
            f"- 分辨率: {NATIVE_WIDTH}x{NATIVE_HEIGHT} | max_blind_ticks: {self.max_blind_ticks}",
            f"- 总 tick: {compiled.total_ticks} | 观察点: {compiled.num_observation_points} 个 | "
            f"回合/命令: {len(macro_specs)} 条",
            "",
            "## 回合序列",
            "",
            "| # | 类型 | 描述 |",
            "|---|------|------|",
        ]
        for index, spec in enumerate(macro_specs):
            lines.append(f"| {index} | {spec.get('kind')} | {_describe_macro_spec(spec)} |")
        lines += [
            "",
            "## 观察点（tick 边界，每个 = 一次推理切点）",
            "",
            ", ".join(str(t) for t in compiled.observation_tick_indices),
            "",
            "## 观察点帧",
            "",
        ]
        for point in saved_points:
            lines.append(f"- tick {point['observation_tick']} → `observation_frames/{point['frame']}`")
        lines.append("")
        lines += [
            "> 逐 tick 动作未内联：回合序列（trajectory.json 的 `macros`）经 "
            "`macro_action_compiler.compile_macros` 可确定性重算逐 tick。",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _macro_to_dict(macro: MacroCommand) -> Dict:
        """MacroCommand → 可序列化 dict（导出/界面回显用）。"""
        base = {"kind": macro.kind, "label": macro.label}
        for attr in ("holds", "clicks", "release", "camera_mode", "delta_yaw", "delta_pitch",
                     "screen_x", "screen_y", "gui_cursor", "wait_ticks", "command"):
            if hasattr(macro, attr):
                base[attr] = getattr(macro, attr)
        return base

    # ── 状态查询 ──────────────────────────────────────────────────────────
    def _state_unlocked(self) -> Dict:
        """构造状态摘要（调用方需已持锁）。"""
        compiled = self._compile()
        return {
            "recorder_version": RECORDER_VERSION,
            "cursor_tick": self.cursor_tick,
            "total_ticks": compiled.total_ticks,
            "observation_tick_indices": compiled.observation_tick_indices,
            "num_macros": len(self.macros),
            "macros": [self._macro_to_dict(m) for m in self.macros],
            "world_type": self.world_type_name,
            "gamemode": self.gamemode,
            "difficulty": self.difficulty,
            "max_blind_ticks": self.max_blind_ticks,
        }

    def state(self) -> Dict:
        with self._lock:
            return self._state_unlocked()

    def current_frame_jpeg(self) -> bytes:
        with self._lock:
            frame = self.current_frame_bgr
        if frame is None:
            frame = np.zeros((NATIVE_HEIGHT, NATIVE_WIDTH, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else b""

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ── HTTP 层 ───────────────────────────────────────────────────────────────
class _RecorderRequestHandler(BaseHTTPRequestHandler):
    """路由：GET / (UI) /frame /state；
    POST /command /remove_command /step /reset /observe /replay /export /send_command。"""

    recorder: TrajectoryRecorder = None  # 由 serve() 注入

    def log_message(self, *args):        # 静音默认访问日志
        pass

    def _send_json(self, payload: Dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    @property
    def route(self) -> str:
        """请求路径去掉查询串（前端 /frame?ts=… 破缓存，路由须按纯路径匹配）。"""
        return urlsplit(self.path).path

    def do_GET(self):
        route = self.route
        if route in ("/", "/index.html"):
            html = UI_HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif route == "/frame":
            jpeg = self.recorder.current_frame_jpeg()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)
        elif route == "/state":
            self._send_json(self.recorder.state())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        route = self.route
        try:
            body = self._read_json_body()
            if route == "/command":
                self.recorder.add_macro(body)
                self._send_json(self.recorder.state())
            elif route == "/remove_command":
                self.recorder.remove_macro(int(body["index"]))
                self._send_json(self.recorder.state())
            elif route == "/step":
                self._send_json(self.recorder.step_to_next_observation())
            elif route == "/reset":
                # mode: "death"（fast_reset，亚秒）| "full"（冷重开，~30s，回干净世界）
                if body.get("mode", "death") == "full":
                    self._send_json(self.recorder.reset_full())
                else:
                    self._send_json(self.recorder.reset_death())
            elif route == "/observe":
                self.recorder.insert_observation_point(int(body["tick"]))
                self._send_json(self.recorder.state())
            elif route == "/replay":
                self._send_json(self.recorder.replay_from_start_to(int(body["tick"])))
            elif route == "/export":
                self._send_json(self.recorder.export(body.get("name", "trajectory")))
            elif route == "/send_command":
                # 即时命令：只影响当前预览、不进导出、不可复现（区别于 kind=="mc" 的序列内宏）。
                self._send_json(self.recorder.send_immediate_command(body.get("command", "")))
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exception:  # 把后端异常回传界面，便于调试
            self._send_json({"error": f"{type(exception).__name__}: {exception}"}, status=500)


def serve(recorder: TrajectoryRecorder, http_port: int):
    """启动 HTTP 服务（阻塞）。"""
    _RecorderRequestHandler.recorder = recorder
    server = ThreadingHTTPServer(("127.0.0.1", http_port), _RecorderRequestHandler)
    print(f"轨迹录制器界面: http://127.0.0.1:{http_port}/", flush=True)
    print(f"  分辨率 {NATIVE_WIDTH}x{NATIVE_HEIGHT} | world={recorder.world_type_name} "
          f"| gamemode={recorder.gamemode} | difficulty={recorder.difficulty}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="CraftGround 手写轨迹录制器")
    parser.add_argument("--http-port", type=int, default=8897, help="界面 HTTP 端口")
    parser.add_argument("--env-port", type=int, default=8000, help="CraftGround socket 起始端口")
    parser.add_argument("--world", choices=["flat", "normal"], default="normal",
                        help="世界类型：normal（真实地形，默认）/ flat（GUI 采集）")
    parser.add_argument("--gamemode", choices=["creative", "survival"], default="survival",
                        help="游戏模式：survival（真实生存示范，默认）/ creative")
    parser.add_argument("--difficulty", choices=["peaceful", "easy", "normal", "hard"],
                        default="normal", help="难度（survival 下影响饥饿/怪物）")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--max-blind-ticks", type=int, default=300,
                        help="单个盲执行段最大 tick（超过自动补插观察点）")
    parser.add_argument("--extra-command", action="append", default=[],
                        help="初始预置 Minecraft 命令（可多次；如给料/setblock 工作台）")
    parser.add_argument("--map-dir-path", type=str, default="",
                        help="从存档加载：指向 .minecraft 存档目录（非空则冷启动加载该世界）")
    parser.add_argument("--level-display-name", type=str, default="",
                        help="存档显示名（配合 --map-dir-path）")
    args = parser.parse_args()

    recorder = TrajectoryRecorder(
        world_type=args.world, gamemode=args.gamemode, seed=args.seed,
        extra_commands=args.extra_command, max_blind_ticks=args.max_blind_ticks,
        port=args.env_port, difficulty=args.difficulty,
        map_dir_path=args.map_dir_path, level_display_name=args.level_display_name,
    )
    serve(recorder, args.http_port)


if __name__ == "__main__":
    main()
