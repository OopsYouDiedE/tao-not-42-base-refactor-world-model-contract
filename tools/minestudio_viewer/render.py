"""帧标注的格式化与叠加绘制。"""
from __future__ import annotations

import numpy as np

# 动作键分组，便于结构化展示
_MOVE_KEYS = ["forward", "back", "left", "right", "jump", "sneak", "sprint"]
_INTERACT_KEYS = ["attack", "use", "drop", "inventory"]
_HOTBAR_KEYS = [f"hotbar.{i}" for i in range(1, 10)]


def _to_scalar(v):
    a = np.asarray(v)
    return a.item() if a.ndim == 0 else a.tolist()


def format_action_md(action: dict | None) -> str:
    """把单帧动作格式化为 Markdown。高亮被按下的键。"""
    if not action:
        return "_无动作数据_"
    lines = ["### 动作 (action)"]

    # camera：[pitch_delta, yaw_delta]，单位度
    if "camera" in action:
        cam = np.asarray(action["camera"]).reshape(-1)
        if cam.size >= 2:
            lines.append(f"- **camera**  pitch={cam[0]:+.2f}°  yaw={cam[1]:+.2f}°")

    def active(keys):
        return [k for k in keys if k in action and int(_to_scalar(action[k])) != 0]

    mv = active(_MOVE_KEYS)
    lines.append(f"- **移动**: {' '.join('`'+k+'`' for k in mv) if mv else '—'}")
    it = active(_INTERACT_KEYS)
    lines.append(f"- **交互**: {' '.join('`'+k+'`' for k in it) if it else '—'}")
    hb = active(_HOTBAR_KEYS)
    lines.append(f"- **快捷栏**: {' '.join('`'+k+'`' for k in hb) if hb else '—'}")
    return "\n".join(lines)


def format_meta_md(meta: dict | None) -> str:
    """把单帧 meta_info（状态真值）格式化为 Markdown。"""
    if not meta:
        return "_无状态真值数据_"
    lines = ["### 状态真值 (meta_info)"]
    # 常见字段优先展示
    preferred = [
        "pitch", "yaw", "cursor_x", "cursor_y",
        "xpos", "ypos", "zpos", "isGuiOpen", "isGuiInventory",
    ]
    shown = set()
    for k in preferred:
        if k in meta:
            lines.append(f"- **{k}**: {_fmt_val(meta[k])}")
            shown.add(k)
    # 其余字段（跳过体积大的，如 events / inventory 列表）
    for k, v in meta.items():
        if k in shown or k == "events":
            continue
        s = _fmt_val(v)
        if s is not None:
            lines.append(f"- {k}: {s}")
    return "\n".join(lines)


def _fmt_val(v):
    """格式化标量/短序列；过长或复杂对象返回简短描述或 None。"""
    if v is None:
        return "None"
    if isinstance(v, (bool, np.bool_)):
        return "✅" if v else "❌"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3f}"
    if isinstance(v, (list, tuple)):
        # 元素可能是 dict / 字符串等非数值（如 inventory 是一列物品 dict）
        n = len(v)
        if n == 0:
            return "[]"
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in v) and n <= 4:
            return "[" + ", ".join(_fmt_num(x) for x in v) + "]"
        # 含复杂元素：给出简短摘要
        if all(isinstance(x, dict) for x in v):
            head = ", ".join(_fmt_item(x) for x in v[:3])
            return f"[{head}{'…' if n > 3 else ''}] ({n})"
        return f"<列表 {n} 项>"
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return f"<对象数组 {v.shape}>"
        if v.size <= 4:
            return "[" + ", ".join(_fmt_num(x) for x in v.reshape(-1)) + "]"
        return f"<{v.shape} 数组>"
    if isinstance(v, dict):
        if not v:
            return None
        return "{" + ", ".join(list(v.keys())[:6]) + ("…" if len(v) > 6 else "") + "}"
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "…"


def _fmt_num(x):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return str(int(xf)) if xf == int(xf) else f"{xf:.2f}"


def _fmt_item(d):
    """把一个物品 dict 压成短串，如 {type:log, quantity:3} -> log×3。"""
    if not isinstance(d, dict):
        return str(d)[:20]
    name = d.get("type") or d.get("name") or d.get("item")
    qty = d.get("quantity") or d.get("count")
    if name is not None and qty is not None:
        return f"{name}×{qty}"
    if name is not None:
        return str(name)
    keys = list(d.keys())
    return "{" + ",".join(keys[:3]) + "}"


def format_events_md(events) -> str:
    """meta_info 里的逐帧 events 字段（dict 或 list）。"""
    if not events:
        return ""
    if isinstance(events, dict):
        nz = {k: v for k, v in events.items() if v}
        if not nz:
            return ""
        return "### 事件\n" + "\n".join(f"- **{k}**: {_fmt_val(v)}" for k, v in nz.items())
    return f"### 事件\n- {_fmt_val(events)}"


def overlay_annotations(img, action=None, meta=None, frame_idx=None, total=None):
    """在图像左上角叠加关键标注文字。img: (H,W,3) uint8 RGB。返回新图。"""
    import cv2

    if img is None:
        return None
    canvas = np.ascontiguousarray(img.copy())
    h, w = canvas.shape[:2]
    scale = max(0.4, min(w, h) / 640 * 0.5)
    y = int(18 * scale) + 4
    dy = int(20 * scale) + 4

    def put(text, color=(255, 255, 0)):
        nonlocal y
        cv2.putText(canvas, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), max(2, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(canvas, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, max(1, int(1.5 * scale)), cv2.LINE_AA)
        y += dy

    if frame_idx is not None:
        put(f"frame {frame_idx}" + (f"/{total-1}" if total else ""), (0, 255, 0))
    if action:
        if "camera" in action:
            cam = np.asarray(action["camera"]).reshape(-1)
            if cam.size >= 2:
                put(f"cam p={cam[0]:+.1f} y={cam[1]:+.1f}", (0, 255, 255))
        pressed = [k for k in _MOVE_KEYS + _INTERACT_KEYS
                   if k in action and int(_to_scalar(action[k])) != 0]
        if pressed:
            put(" ".join(pressed), (255, 200, 0))
    if meta:
        for k in ("pitch", "yaw"):
            if k in meta:
                put(f"{k}={_fmt_val(meta[k])}", (200, 200, 255))
    return canvas
