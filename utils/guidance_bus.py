"""异步 LLM 指导总线:慢系统(VLM/LLM)与 30Hz 快反环之间的无阻塞接口 (utils/guidance_bus.py)。

对外接口:
    Guidance    — 单条指导记录(当前子目标 + 冻结文本向量 + 计划全文)。
    GuidanceBus — 线程安全的最新值寄存器:LLM worker 按自身节拍 publish_plan,
                  30Hz 环每 tick 非阻塞 read;LLM 计划陈旧时自动降级到静态计划。

设计契约(见 [knowledge/design_llm_deep_integration.md] §3):
  - 快环 **永不等待** 慢系统——read() 无锁竞争之外零开销,延迟表现为陈旧度而非阻塞;
  - 断网/超时降级:publish 超过 stale_after_s 未刷新且存在静态计划 ⇒ 切到静态计划;
  - 文本编码器经依赖注入(encode_fn: list[str] -> [B, dim] fp32 CPU 张量),
    本模块不 import 任何模型/领域代码(utils 横向层)。
"""
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class Guidance:
    """一条指导记录(不可变快照)。

    Attributes:
        subgoal:  当前子目标文本。
        goal_vec: [goal_text_dim] fp32 CPU 张量(冻结句向量,L2 归一);无计划时为 None。
        plan:     完整子目标序列(供监控/重编译对照)。
        plan_idx: 当前子目标在 plan 中的下标。
        ts:       发布时刻(time.monotonic 秒)。
        source:   "llm"(异步编译)| "static"(降级静态计划)| "none"(无任何计划)。
    """
    subgoal: str
    goal_vec: Optional[torch.Tensor]
    plan: Tuple[str, ...]
    plan_idx: int
    ts: float
    source: str


class GuidanceBus:
    """慢-快双系统之间的最新值寄存器。

    Args:
        encode_fn:     子目标文本编码函数,list[str] -> [B, dim] fp32 CPU 张量(冻结,注入)。
        static_plan:   降级用静态子目标序列(断网/陈旧兜底);空则降级为 "none"。
        stale_after_s: LLM 计划陈旧阈值(秒)。

    线程模型:LLM worker 线程调 publish_plan / advance,采样线程调 read;
    全部状态变更持锁,read 返回不可变快照。
    """

    def __init__(self, encode_fn: Callable[[Sequence[str]], torch.Tensor],
                 static_plan: Sequence[str] = (), stale_after_s: float = 8.0):
        self._encode = encode_fn
        self._stale_after = stale_after_s
        self._lock = threading.Lock()
        self._static = tuple(static_plan)
        self._g = self._make(self._static, 0, "static") if self._static \
            else Guidance("", None, (), 0, time.monotonic(), "none")

    def _make(self, plan: Tuple[str, ...], idx: int, source: str) -> Guidance:
        idx = min(idx, len(plan) - 1)
        vec = self._encode([plan[idx]])[0].float().cpu()
        return Guidance(plan[idx], vec, plan, idx, time.monotonic(), source)

    def publish_plan(self, subgoals: Sequence[str], source: str = "llm") -> None:
        """LLM worker 发布(重编译)新计划,从其第 0 个子目标开始执行。"""
        plan = tuple(s for s in subgoals if s)
        if not plan:
            return
        g = self._make(plan, 0, source)
        with self._lock:
            self._g = g

    def advance(self) -> bool:
        """当前子目标完成(里程碑判定为真)→ 推进到计划的下一子目标。

        Returns:
            True = 已推进;False = 已是计划末项(等待下一次重编译)。
        """
        with self._lock:
            g = self._g
            if not g.plan or g.plan_idx + 1 >= len(g.plan):
                return False
            self._g = self._make(g.plan, g.plan_idx + 1, g.source)
            return True

    def read(self) -> Guidance:
        """非阻塞读取当前指导;LLM 计划陈旧且有静态计划时先降级再返回。"""
        with self._lock:
            g = self._g
            if (g.source == "llm" and self._static
                    and time.monotonic() - g.ts > self._stale_after):
                g = self._make(self._static, 0, "static")
                self._g = g
            return g

    def staleness(self) -> float:
        """当前指导的年龄(秒),监控用。"""
        with self._lock:
            return time.monotonic() - self._g.ts
