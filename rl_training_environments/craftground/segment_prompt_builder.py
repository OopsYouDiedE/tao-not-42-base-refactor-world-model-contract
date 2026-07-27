# -*- coding: utf-8 -*-
"""分区式动作段提示词构建：静态前缀 + 追加式事件日志 + 重写式真值 + 定额图像预算。

对外接口:
    SegmentRecord      — 一段的计划 / 回执 / 观察帧，事件日志的元素。
    FactLedger         — 已确认事实台账（[实测] 覆盖 [推断]，同键替换而非追加）。
    PromptBuilder      — 持有静态区与日志，产出 Anthropic messages 结构。

分区顺序按"追加式在前、重写式在后"排：静态格式 → 键表 → 常量 → 任务 → 事件日志
→ 已确认事实 → 当前观测 → 请求。日志是纯追加，不会让自身失效，因此可尽量靠前以
命中前缀缓存；真值/观测每轮重写，必须排在追加块之后。
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# 图像预算：当前帧 + 最多 (IMAGE_BUDGET - 1) 张历史观察帧。
IMAGE_BUDGET = 5
# 事件日志文本上限（字符数，粗略代理 token）。超出从最旧的链开始压缩。
LOG_CHARACTER_BUDGET = 12000
# [推断] 类事实上限，超出丢最旧。
MAX_INFERRED_FACTS = 6


@dataclass
class SegmentRecord:
    """事件日志里的一段。"""

    index: int
    canonical_text: str
    why_text: str
    executed_ticks: int
    planned_ticks: int
    tripped_guard: Optional[str] = None
    parse_warnings: List[str] = field(default_factory=list)
    truncation_notes: List[str] = field(default_factory=list)
    observation_frames: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    interrupt_frame: Optional[Tuple[int, np.ndarray]] = None
    state_note: str = ""

    def render_log_entry(self) -> str:
        """渲染成日志文本块（不含图像）。"""
        lines = [f"【第 {self.index} 段】", self.canonical_text]
        if self.tripped_guard is None:
            lines.append(f"  → 执行 {self.executed_ticks}/{self.planned_ticks} tick，完整跑完。")
        else:
            lines.append(
                f"  → 执行 {self.executed_ticks}/{self.planned_ticks} tick，"
                f"**{self.tripped_guard} 触发**被打断。"
            )
        for note in self.truncation_notes:
            lines.append(f"    {note}")
        for warning in self.parse_warnings:
            lines.append(f"    ⚠ 解析告警：{warning}")
        if self.state_note:
            lines.append(f"    {self.state_note}")
        return "\n".join(lines)


class FactLedger:
    """已确认事实台账。[实测] 与 [推断] 分列；同 key 替换而非追加。"""

    def __init__(self) -> None:
        self._measured: Dict[str, str] = {}
        self._inferred: Dict[str, str] = {}

    def record_measured(self, key: str, text: str) -> None:
        """记一条实测事实；同 key 覆盖，并撤销同 key 的推断。"""
        self._measured[key] = text
        self._inferred.pop(key, None)

    def record_inferred(self, key: str, text: str) -> None:
        """记一条模型自报的推断；已有同 key 实测时不覆盖实测。"""
        if key in self._measured:
            return
        self._inferred[key] = text
        while len(self._inferred) > MAX_INFERRED_FACTS:
            self._inferred.pop(next(iter(self._inferred)))

    def render(self) -> str:
        if not self._measured and not self._inferred:
            return "  （尚无。）"
        lines = [f"  [实测] {text}" for text in self._measured.values()]
        lines += [f"  [推断] {text}" for text in self._inferred.values()]
        return "\n".join(lines)

    def measured_count(self) -> int:
        return len(self._measured)


def encode_frame_to_base64_png(frame_rgb: np.ndarray) -> str:
    """RGB uint8 数组 → base64 PNG 字符串。"""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_rgb)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_block(frame_rgb: np.ndarray) -> Dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": encode_frame_to_base64_png(frame_rgb),
        },
    }


def _text_block(text: str) -> Dict[str, object]:
    return {"type": "text", "text": text}


ZONE_OUTPUT_FORMAT = """你在操作一个真实的 Minecraft 游戏。每次回答输出一段「动作程序」，它会被逐 tick
盲执行一段时间——执行期间你看不到画面，直到下一轮。

行清单（除 for 与 why 外都可省；顺序固定，一行一个）：
  for:      本段总长，必写
  hold:     按住的开关键，逗号分隔。可带窗口 `<键> <起>-<止>`
  tap:      点按，写「时刻 键」；同一时刻多键用 +
  Mouse:    视角旋转，写「时刻 +yaw,+pitch」，单位度，相对当前朝向，+yaw=右 +pitch=抬头
  Cursor:   GUI 光标要移动到的**绝对位置**，写「时刻 x,y」归一化 0..1，左上=0,0，
            仅背包打开时可用。写 0.4,0.7 = 移到那个点，不是"挪 0.4 屏"
  look:     观察点，逗号分隔时刻，最多 6 个。你会在下一轮看到这些时刻的画面
  stop_if:  中断条件，逗号分隔
  after:    段末处置 `<hold|stop|freeze> <租约>`
  learn:    可选，一行。你新发现的、值得长期记住的事实
  why:      必写。你这样做的理由

时间口径：**全部是绝对时刻**，从本段第一 tick 起算。写作 `<帧号>/20s`。
  20Hz 下 1 tick = 50ms。`30/20s` = 第 30 tick = 1.5 秒。同一行内不混写小数。
  Mouse/Cursor 的时刻是**截止时刻**：在这一刻之前把这个量转完。
  多个 Mouse 项 = 分段截止，前一项的截止时刻是后一项的起点。

键分两类：
  开关键 —— 只有按下/松开，进 hold（可带窗口）或 tap（单帧点按）。
  轴键   —— 自带数值，一个键占一行：Mouse、Cursor。

stop_if 可选值（只有这三个，别的写了会被丢弃）：
  scene_changed  画面大变（进/出 GUI、场景切换）
  flash          画面突然闪烁（多为受伤）
  stuck          按着移动键却走不动（撞墙/被挡住）。
                 站定挖矿不会触发它——不按移动键时它永不触发，所以挖掘段也可以放心带上。
after 可选值：
  hold    保持当前按住状态不变
  stop    松开全部键
  freeze  松开移动键，保留其余
  租约是死人开关：超过这个时长没有下一段，运行时会强制松开一切。

只输出动作行。不要输出解释、代码块标记或任何其他内容。"""


ZONE_KEY_TABLE = """开关键（书写顺序按此表分组，组内按表内顺序）：
  组1  W 前进 / A 左移 / S 后退 / D 右移
       互斥：W↔S、A↔D。同段同时按住会被双双丢弃，别写。
  组2  Space 跳跃 / Shift 潜行（可长按） / Ctrl 疾跑（需与 W 同按才有效）
  组3  1..9 直接选择快捷栏第 n 格
  组4  Mouse_L 挖掘·攻击（可长按，长按才能挖穿方块） / Mouse_R 放置·使用（可长按）
  组5  Q 丢弃手上物品 / E 开关背包（按一次开，再按一次关）

轴键：
  Mouse   世界视角。**每 tick 最多转 18°**，超了会被静默截断，不报错。
          转 N° 至少要给 ceil(N/18) 个 tick，例如转 90° 至少给 5 tick。
          pitch 范围 ±90°（+90 = 直视天空，-90 = 直视脚下）。
          ⚠ 方向：`Mouse: 5/20s +0,+30` = 向上转 30°；`+0,-30` = 向下转 30°。
            `+30,+0` = 向右转 30°；`-30,+0` = 向左转 30°。
            反馈里报的 pitch 与此同口径（正 = 抬头），不用做任何符号换算。
          yaw 在 ±180° 环绕，所以写 +180 和 -180 到达同一朝向，写 +360 等于没转。
  Cursor  背包 GUI 里的鼠标光标，写**绝对位置**（0..1，左上 0,0，右下 1,1）。
          速度上限每 tick 120 像素（0.19 屏宽 / 0.33 屏高），跨整屏给 6 tick 就够。
          语义是**到位后停住**：截止时刻到达目标，然后一直停在那儿，直到下一个
          Cursor 项的截止时刻才开始移动。所以两个 Cursor 项之间的点击打在前一个目标上。
          背包**每次打开**光标都复位到正中 (0.5,0.5)；背包不关，光标就停在上一段末尾，
          反馈每轮都会告诉你它现在在哪。
          ⚠ 按 E 之后 GUI 要 2 tick 才真正打开，这 2 tick 里光标不动（转的是世界视角）。
          ⚠ 点击要等光标到位：tap Mouse_L 的时刻必须**晚于**对应 Cursor 项的截止时刻。
          ⚠ 光标在画面里就是那个白色箭头，可以用 look 直接核对它在哪。

本设备不存在的键（写了会被丢弃）：Mouse_M、F、Tab、Esc。
本设备**不支持文本输入**——禁止使用聊天框或任何 / 指令。所有事都得靠键鼠做。
关背包用再按一次 E，没有 Esc。"""


ZONE_GAME_CONSTANTS = """  1 tick = 50ms，控制帧率 20Hz
  行走 4.3 米/秒（约 0.22 米/tick）      按住 Ctrl+W 疾跑 5.6 米/秒
  跳跃高度 1.25 格（能上 1 格台阶；走路撞到 1 格高的坎会被挡住）
  徒手挖橡木原木约 3 秒（60 tick）      拿石斧约 1.5 秒
  玩家宽 0.6 格、高 1.8 格，眼高 1.62 格
  一棵橡树通常 4-6 格高，砍掉底部一格后上面的原木**不会**自动掉落，
  需要逐格往上挖；掉落的原木会自动被捡起（走过去即可）。
  一棵树只有 4-6 格原木，砍完这一棵就没有了，森林里的树彼此相隔若干米。
  挖掘距离上限 4.5 格：更远的方块按 Mouse_L 打不到，得先走近。
  掉落物要走到它上面才会被捡起，光对准它不会进背包。
  各方块挖了给什么：原木→原木，草方块/泥土→泥土，树叶→偶尔掉树苗，石头→无（徒手挖不动）。
  石头徒手挖不掉；泥土和草方块徒手很快就能挖穿，连续挖会在地形上开洞。
  合成：1 原木 = 4 木板；4 木板 = 1 工作台；2 木板 = 4 木棍；
        木镐 = 3 木板 + 2 木棍，但木镐需要 3×3 合成格。
  背包自带的 2×2 合成格不需要工作台，够做木板/工作台/木棍；3×3 要把工作台放在地上
  （手持工作台、准心对着地面按 Mouse_R 放下），再对它按 Mouse_R 打开。
  石头必须用镐才有掉落，徒手挖石头挖穿了也什么都不给。
  GUI 里的鼠标操作（都是 vanilla 行为，实测确认）：
    左键点有物品的槽位 = 把**整叠**拿到光标上；再左键点空槽 = 整叠放下。
    右键点空槽 = 只放**1 个**（手上还剩其余）。要往 2×2 四格各放 1 个就用右键点四次。
    对输出格按住 Shift 左键点 = 把产物**直接送进背包**，且会尽可能重复合成。
    不按 Shift 点输出格只是把产物拿到光标上——此时关背包会把它**丢到地上**。
    Shift 送进背包是从**后往前**填空位，所以产物常出现在热键栏靠右的格子里，
    不是第 1 格；下一步要再拿它时先用 look 看清它到底在哪一格。
    光标上还拿着东西时，点另一个已占用的同类槽位会合并，点不同类的会交换。"""


def render_request_zone(
    suggested_minimum_ticks: int,
    suggested_maximum_ticks: int,
    latency_seconds: Optional[float],
    guard_required_above_ticks: int,
) -> str:
    """渲染请求区（S8）。"""
    if latency_seconds is None:
        latency_text = "推理耗时尚未实测，暂按 0.8s 估。"
    else:
        latency_text = f"上一轮推理耗时 {latency_seconds:.1f}s。"
    return (
        f"{latency_text}建议 for 落在 {suggested_minimum_ticks}/20s–"
        f"{suggested_maximum_ticks}/20s 之间。\n"
        f"超过 {guard_required_above_ticks}/20s 的段必须带 stop_if。\n"
        "只输出动作行，不要其他内容。"
    )


class PromptBuilder:
    """持有静态区、事件日志与真值台账，产出 Anthropic messages 结构。

    事件日志是追加式：已入库的段文本一字不改，只在链边界事件发生时整链压成一行。
    图像是定额的：当前帧永不丢，其余按「中间观察帧先丢、段末帧与中断帧最后丢」淘汰。
    """

    def __init__(self, task_text: str, image_budget: int = IMAGE_BUDGET) -> None:
        self.task_text = task_text
        self.image_budget = image_budget
        self.facts = FactLedger()
        self._compressed_lines: List[str] = []
        self._records: List[SegmentRecord] = []

    def append_record(self, record: SegmentRecord) -> None:
        self._records.append(record)

    def compress_chain(self, summary: str) -> None:
        """链边界：把当前未压缩的所有段压成一行摘要，图像一并丢弃。"""
        if not self._records:
            self._compressed_lines.append(summary)
            return
        total_ticks = sum(record.executed_ticks for record in self._records)
        self._compressed_lines.append(
            f"【已完成】{summary}（{len(self._records)} 段，共 {total_ticks / 20.0:.1f} 秒）"
        )
        self._records.clear()

    def force_compress_on_repeat(self, guard_name: str, repeat_count: int) -> None:
        """退化循环保护：同一守卫连续触发多次，强制压缩并标记为失败尝试。"""
        total_ticks = sum(record.executed_ticks for record in self._records)
        self._compressed_lines.append(
            f"【已废弃尝试】连续 {repeat_count} 次触发 {guard_name} 无进展"
            f"（{len(self._records)} 段，共 {total_ticks / 20.0:.1f} 秒）——换个办法"
        )
        self._records.clear()

    def _render_log_zone(self) -> str:
        if not self._compressed_lines and not self._records:
            return "  （尚无。这是本回合第一段。）"
        blocks = list(self._compressed_lines)
        blocks += [record.render_log_entry() for record in self._records]
        text = "\n\n".join(blocks)
        while len(text) > LOG_CHARACTER_BUDGET and self._compressed_lines:
            self._compressed_lines.pop(0)
            blocks = list(self._compressed_lines)
            blocks += [record.render_log_entry() for record in self._records]
            text = "\n\n".join(blocks)
        return text

    def _select_frames(self, current_frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """按预算与优先级挑出要塞的图像，返回 (标签, 帧) 列表，当前帧在最后。

        淘汰顺序：越早的段越先淘汰；同段内中间观察帧先丢，段末帧与中断帧最后丢。
        """
        candidates: List[Tuple[int, int, str, np.ndarray]] = []
        for record in self._records:
            frame_count = len(record.observation_frames)
            for order, (tick, frame) in enumerate(record.observation_frames):
                is_last = order == frame_count - 1
                # priority 小 = 先丢。段末帧优先级更高（更晚丢）。
                priority = 1 if is_last else 0
                label = f"  第 {record.index} 段 {tick}/20s" + ("（段末）" if is_last else "")
                candidates.append((record.index, priority, label, frame))
            if record.interrupt_frame is not None:
                tick, frame = record.interrupt_frame
                label = f"  第 {record.index} 段 {tick}/20s（**{record.tripped_guard} 触发瞬间**）"
                candidates.append((record.index, 2, label, frame))

        history_budget = max(0, self.image_budget - 1)
        if len(candidates) > history_budget:
            # 排序键：段序号大的、优先级高的留下。取末尾 history_budget 个。
            candidates.sort(key=lambda item: (item[0], item[1]))
            dropped = len(candidates) - history_budget
            candidates = candidates[dropped:]
        else:
            candidates.sort(key=lambda item: (item[0], item[1]))
        selected = [(label, frame) for _, _, label, frame in candidates]
        selected.append(("  当前帧：", current_frame))
        return selected

    def build_messages(
        self,
        current_frame: np.ndarray,
        control_state_text: str,
        request_zone_text: str,
        inventory_text: str = "",
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int]:
        """产出 (system_blocks, user_messages, image_count)。

        Parameters
        ----------
        current_frame : np.ndarray, [H,W,3] uint8 RGB
            当前观测帧。
        control_state_text : str
            当前按住的键与快捷栏状态。
        request_zone_text : str
            render_request_zone 的产物。
        inventory_text : str
            背包摘要；空串表示不提供。

        Returns
        -------
        system_blocks : List[Dict]
            静态区，标了 cache_control 断点。
        user_messages : List[Dict]
            单条 user 消息，内容为图文交错块列表。
        image_count : int
            本次实际塞入的图像张数。
        """
        system_blocks = [
            _text_block("── 输出格式 ──\n" + ZONE_OUTPUT_FORMAT),
            _text_block("── 键表与设备能力 ──\n" + ZONE_KEY_TABLE),
            _text_block("── 游戏常量 ──\n" + ZONE_GAME_CONSTANTS),
        ]
        # 静态三区整体作为缓存前缀断点：episode 内逐字节不变。
        system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        content: List[Dict[str, object]] = [
            _text_block(f"── 任务 ──\n  {self.task_text}"),
            _text_block("── 事件日志 ──\n" + self._render_log_zone()),
            _text_block("── 已确认事实 ──\n" + self.facts.render()),
            _text_block("── 当前观测 ──"),
        ]
        frames = self._select_frames(current_frame)
        for label, frame in frames:
            content.append(_text_block(label))
            content.append(_image_block(frame))
        if inventory_text:
            content.append(_text_block("  背包：" + inventory_text))
        content.append(_text_block("  控制状态：" + control_state_text))
        content.append(_text_block("── 请求 ──\n" + request_zone_text))
        return system_blocks, [{"role": "user", "content": content}], len(frames)

    def render_text_only_preview(self, control_state_text: str, request_zone_text: str) -> str:
        """渲染纯文本预览（不含图像），用于写进轨迹 md 做证据。"""
        return "\n\n".join([
            "── 输出格式 ──\n" + ZONE_OUTPUT_FORMAT,
            "── 键表与设备能力 ──\n" + ZONE_KEY_TABLE,
            "── 游戏常量 ──\n" + ZONE_GAME_CONSTANTS,
            f"── 任务 ──\n  {self.task_text}",
            "── 事件日志 ──\n" + self._render_log_zone(),
            "── 已确认事实 ──\n" + self.facts.render(),
            "── 当前观测 ──\n  （图像区，见轨迹 md 里的截图）\n  控制状态：" + control_state_text,
            "── 请求 ──\n" + request_zone_text,
        ])
