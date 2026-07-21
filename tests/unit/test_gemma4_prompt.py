"""验证 Gemma4 策略的多模态 prompt 构造契约（不加载模型权重）。"""

import pytest
from PIL import Image

from net.action_token_codec import ActionTokenFormat, StructuredAction
from net.gemma4_policy import HistoryContext, build_prompt_messages


def _context(history_frames: int = 3, past: int = 2) -> HistoryContext:
    return HistoryContext(
        frames=[Image.new("RGB", (32, 32)) for _ in range(history_frames)],
        task_text="obtain a diamond pickaxe",
        past_actions=[StructuredAction() for _ in range(past)],
    )


def test_prompt_interleaves_text_and_images():
    """消息含 system 与 user；user 内容交错：每张图像前有文本锚点。"""
    messages = build_prompt_messages(_context(3), ActionTokenFormat.COMPACT_TAG, 5, True)
    assert [message["role"] for message in messages] == ["system", "user"]
    # Gemma4 chat 模板要求 content 为块列表，system 也不能用裸字符串。
    assert isinstance(messages[0]["content"], list)
    assert messages[0]["content"][0]["type"] == "text"
    content = messages[1]["content"]
    # 首块为任务文本，末块为预测请求，中间图像总数等于帧数。
    assert content[0]["type"] == "text"
    assert "obtain a diamond pickaxe" in content[0]["text"]
    assert content[-1]["type"] == "text"
    assert sum(1 for item in content if item["type"] == "image") == 3
    # 每张图像的紧邻前一块必须是文本锚点（时间标签）。
    for index, item in enumerate(content):
        if item["type"] == "image":
            assert content[index - 1]["type"] == "text"


def test_prompt_labels_current_and_past_frames():
    """历史帧标注 t-N，当前帧标注 Current frame。"""
    content = build_prompt_messages(
        _context(3), ActionTokenFormat.COMPACT_TAG, 5, True,
    )[1]["content"]
    labels = [item["text"] for item in content if item["type"] == "text"]
    joined = "\n".join(labels)
    assert "Current frame:" in joined
    assert "Frame t-1:" in joined
    assert "Frame t-2:" in joined


def test_prompt_includes_horizon_and_format_hint():
    """末块含请求的帧数与格式说明。"""
    content = build_prompt_messages(_context(), ActionTokenFormat.JSON_LINE, 7, True)[1]["content"]
    tail = content[-1]["text"]
    assert "7" in tail
    assert "JSON" in tail or "json" in tail


def test_prompt_places_past_action_after_its_frame():
    """交错布局里每帧动作以 executed: 紧跟其图像之后；禁用时不出现。"""
    with_history = build_prompt_messages(
        _context(past=2), ActionTokenFormat.COMPACT_TAG, 5, True,
    )[1]["content"]
    without_history = build_prompt_messages(
        _context(past=2), ActionTokenFormat.COMPACT_TAG, 5, False,
    )[1]["content"]
    with_text = "\n".join(item["text"] for item in with_history if item["type"] == "text")
    without_text = "\n".join(item["text"] for item in without_history if item["type"] == "text")
    assert "executed:" in with_text
    assert "executed:" not in without_text
    # executed 锚点必须紧跟在某张图像之后。
    for index, item in enumerate(with_history):
        if item["type"] == "text" and item["text"].startswith("executed:"):
            assert with_history[index - 1]["type"] == "image"


def test_empty_frames_rejected():
    """空历史帧应报错，避免构造非法多模态请求。"""
    context = HistoryContext(frames=[], task_text="t", past_actions=[])
    with pytest.raises(ValueError):
        build_prompt_messages(context, ActionTokenFormat.COMPACT_TAG, 5, True)
