"""MineStudio 数据集帧级可视化 Gradio 应用。

功能：
  - 选择 part（分片）与 episode（片段），支持切换
  - 拖动滑条选取任意帧
  - 展示该帧画面，以及动作 / 状态真值 / 事件等标注
  - 可选在画面上叠加标注

用法：
  python app.py --data /content/minestudio-data-10xx-v110 [--share] [--port 7860]
"""
from __future__ import annotations

import argparse
import os

import gradio as gr

from reader import MineStudioDataset, compute_matches
from render import (
    format_action_md,
    format_events_md,
    format_meta_md,
    overlay_annotations,
)

DATASET: MineStudioDataset | None = None


def _episode_label(entry):
    return f"{entry.name}  ({entry.num_frames} 帧)"


def _label_to_name(label):
    return label.split("  (")[0] if label else None


def _safe(fn, *args, default=None):
    """调用 fn(*args)，异常时返回 default，避免单点错误拖垮整个回调。"""
    try:
        return fn(*args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return default


def list_episodes_for_part(part):
    """返回某 part 下的 episode 标签列表。"""
    eps = DATASET.episodes_in_part(part)
    return [_episode_label(e) for e in eps]


def load_frame(part, ep_label, frame_idx, do_overlay):
    """核心回调：给定 part/episode/帧号，返回 (图像, 动作md, 状态md, 事件md, 标题)。"""
    name = _label_to_name(ep_label)
    entry = DATASET.get(name) if name else None
    if entry is None:
        return None, "_未选择片段_", "", "", ""

    fi = int(frame_idx)
    fi = max(0, min(fi, entry.num_frames - 1))

    # 每个模态独立读取+格式化，任一环节出错都不影响其余（尤其是画面）
    img = _safe(DATASET.read_frame_image, entry, fi, default=None)
    action = _safe(DATASET.read_frame_action, entry, fi, default=None)
    meta = _safe(DATASET.read_frame_meta, entry, fi, default=None)

    action_md = _safe(format_action_md, action, default="_动作渲染出错_")
    meta_md = _safe(format_meta_md, meta, default="_状态渲染出错_")
    events_md = ""
    if isinstance(meta, dict) and meta.get("events"):
        events_md = _safe(format_events_md, meta["events"], default="")

    if img is None:
        img_out = None
    elif do_overlay:
        img_out = _safe(overlay_annotations, img, action, meta, fi,
                        entry.num_frames, default=img)
    else:
        img_out = img

    modals = "、".join(sorted(entry.locs.keys()))
    title = f"**{name}**  ·  帧 {fi} / {entry.num_frames - 1}  ·  模态: {modals}"
    return img_out, action_md, meta_md, events_md, title


def on_select_part(part):
    """切换 part 时刷新 episode 下拉，并定位到该 part 首个片段第 0 帧。"""
    labels = list_episodes_for_part(part)
    first = labels[0] if labels else None
    ep = DATASET.get(_label_to_name(first)) if first else None
    maxf = (ep.num_frames - 1) if ep else 0
    dd = gr.update(choices=labels, value=first)
    sl = gr.update(minimum=0, maximum=maxf, value=0, step=1)
    return (dd, sl) + load_frame(part, first, 0, True)


def on_select_episode(part, ep_label, do_overlay):
    """切换 episode 时把滑条重置到 0，并更新最大值。"""
    name = _label_to_name(ep_label)
    entry = DATASET.get(name) if name else None
    maxf = (entry.num_frames - 1) if entry else 0
    sl = gr.update(minimum=0, maximum=maxf, value=0, step=1)
    return (sl,) + load_frame(part, ep_label, 0, do_overlay)


def step_frame(part, ep_label, frame_idx, do_overlay, delta):
    name = _label_to_name(ep_label)
    entry = DATASET.get(name) if name else None
    maxf = (entry.num_frames - 1) if entry else 0
    fi = max(0, min(int(frame_idx) + delta, maxf))
    return (gr.update(value=fi),) + load_frame(part, ep_label, fi, do_overlay)


def on_refresh():
    """重新扫描数据目录（image 下载完成后可加载画面），刷新到首个 part。"""
    DATASET.build_index()
    parts = DATASET.parts()
    first_part = parts[0] if parts else None
    dd_part = gr.update(choices=parts, value=first_part)
    ep_dd, slider, *outs = on_select_part(first_part)
    return (dd_part, ep_dd, slider) + tuple(outs)


# 常用可筛选动作键（供多选）
FILTER_ACTION_KEYS = [
    "attack", "use", "jump", "forward", "back", "left", "right",
    "sneak", "sprint", "drop", "inventory",
]


def _event_choices(entry):
    """扫描片段并返回其事件词表（用于事件多选下拉）。"""
    if entry is None or "meta_info" not in entry.locs and "segmentation" not in entry.locs:
        return []
    scan = DATASET.scan_episode(entry)
    return sorted(scan["event_vocab"])


def apply_filter(part, ep_label, action_keys, logic, cam_min, gui, events):
    """执行筛选，返回 (匹配索引状态, 状态文字)。"""
    name = _label_to_name(ep_label)
    entry = DATASET.get(name) if name else None
    if entry is None:
        return [], "_未选择片段_"
    scan = DATASET.scan_episode(entry)
    spec = {
        "action_keys": action_keys or [],
        "action_any": (logic == "任一(OR)"),
        "cam_min": float(cam_min) if cam_min else 0.0,
        "gui": {"不限": "any", "GUI打开": "open", "物品栏": "inventory",
                "无GUI": "none"}.get(gui, "any"),
        "events": events or [],
    }
    matches = compute_matches(scan, spec).tolist()
    if not matches:
        return [], "**匹配 0 帧** — 没有符合条件的帧，试试放宽条件"
    txt = (f"**匹配 {len(matches)} 帧** / 共 {entry.num_frames} 帧　"
           f"（占 {100*len(matches)/entry.num_frames:.1f}%）　"
           f"首帧 {matches[0]}，末帧 {matches[-1]}　·　用 ⯇匹配 / 匹配⯈ 跳转")
    return matches, txt


def goto_match(part, ep_label, frame_idx, do_overlay, matches, direction):
    """跳到相对当前帧的上一个/下一个匹配帧。direction: -1/+1；0=首个。"""
    if not matches:
        return (gr.update(),) + load_frame(part, ep_label, frame_idx, do_overlay)
    cur = int(frame_idx)
    arr = matches
    if direction > 0:
        nxt = next((m for m in arr if m > cur), arr[0])  # 循环
    elif direction < 0:
        nxt = next((m for m in reversed(arr) if m < cur), arr[-1])
    else:
        nxt = arr[0]
    return (gr.update(value=nxt),) + load_frame(part, ep_label, nxt, do_overlay)


def build_ui():
    parts = DATASET.parts()
    init_part = parts[0] if parts else None
    init_eps = list_episodes_for_part(init_part) if init_part else []
    init_ep = init_eps[0] if init_eps else None
    init_entry = DATASET.get(_label_to_name(init_ep)) if init_ep else None
    init_max = (init_entry.num_frames - 1) if init_entry else 0

    with gr.Blocks(title="MineStudio 数据集帧查看器") as demo:
        gr.Markdown("# MineStudio 数据集帧查看器\n拖动滑条选帧，查看画面与标注（动作 / 状态真值 / 事件）。")
        with gr.Row():
            part_dd = gr.Dropdown(parts, value=init_part, label="Part（分片）", scale=1)
            ep_dd = gr.Dropdown(init_eps, value=init_ep, label="Episode（片段）", scale=3)
            refresh_btn = gr.Button("🔄 重建索引", scale=1)
        title_md = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=3):
                frame = gr.Image(label="画面", type="numpy", height=480)
                with gr.Row():
                    prev5 = gr.Button("⏪ -5")
                    prev1 = gr.Button("◀ -1")
                    next1 = gr.Button("+1 ▶")
                    next5 = gr.Button("+5 ⏩")
                slider = gr.Slider(0, init_max, value=0, step=1, label="帧号")
                overlay_ck = gr.Checkbox(True, label="在画面上叠加标注")

                with gr.Accordion("🔍 帧筛选", open=False):
                    matches_state = gr.State([])
                    filt_status = gr.Markdown("_设置条件后点“应用筛选”_")
                    with gr.Row():
                        f_actions = gr.Dropdown(
                            FILTER_ACTION_KEYS, value=[], multiselect=True,
                            label="动作键", scale=3)
                        f_logic = gr.Radio(
                            ["同时(AND)", "任一(OR)"], value="同时(AND)",
                            label="动作键逻辑", scale=2)
                    with gr.Row():
                        f_cam = gr.Slider(0, 30, value=0, step=0.5,
                                          label="视角移动幅度 ≥（0=不限）", scale=3)
                        f_gui = gr.Dropdown(
                            ["不限", "GUI打开", "物品栏", "无GUI"], value="不限",
                            label="GUI 状态", scale=2)
                    f_events = gr.Dropdown(
                        _event_choices(init_entry), value=[], multiselect=True,
                        label="事件 / 分割交互（任一出现即匹配）")
                    with gr.Row():
                        apply_btn = gr.Button("应用筛选", variant="primary")
                        clear_btn = gr.Button("清除")
                    with gr.Row():
                        m_first = gr.Button("⏮ 首个匹配")
                        m_prev = gr.Button("⯇ 匹配")
                        m_next = gr.Button("匹配 ⯈")

            with gr.Column(scale=2):
                action_md = gr.Markdown()
                meta_md = gr.Markdown()
                events_md = gr.Markdown()

        outs = [frame, action_md, meta_md, events_md, title_md]

        # 拖动滑条 -> 实时换帧
        slider.change(load_frame, [part_dd, ep_dd, slider, overlay_ck], outs)
        overlay_ck.change(load_frame, [part_dd, ep_dd, slider, overlay_ck], outs)
        # 切换 part / episode
        part_dd.change(on_select_part, [part_dd], [ep_dd, slider] + outs)
        ep_dd.change(on_select_episode, [part_dd, ep_dd, overlay_ck], [slider] + outs)
        # 步进按钮
        prev1.click(lambda p, e, f, o: step_frame(p, e, f, o, -1),
                    [part_dd, ep_dd, slider, overlay_ck], [slider] + outs)
        next1.click(lambda p, e, f, o: step_frame(p, e, f, o, 1),
                    [part_dd, ep_dd, slider, overlay_ck], [slider] + outs)
        prev5.click(lambda p, e, f, o: step_frame(p, e, f, o, -5),
                    [part_dd, ep_dd, slider, overlay_ck], [slider] + outs)
        next5.click(lambda p, e, f, o: step_frame(p, e, f, o, 5),
                    [part_dd, ep_dd, slider, overlay_ck], [slider] + outs)

        refresh_btn.click(on_refresh, [], [part_dd, ep_dd, slider] + outs)

        # ---- 帧筛选事件 ----
        filt_inputs = [part_dd, ep_dd, f_actions, f_logic, f_cam, f_gui, f_events]
        apply_btn.click(apply_filter, filt_inputs, [matches_state, filt_status])

        def _clear():
            return [], "_已清除筛选_", [], "同时(AND)", 0, "不限", []
        clear_btn.click(_clear, [],
                        [matches_state, filt_status, f_actions, f_logic,
                         f_cam, f_gui, f_events])

        nav_in = [part_dd, ep_dd, slider, overlay_ck, matches_state]
        m_first.click(lambda p, e, f, o, m: goto_match(p, e, f, o, m, 0),
                      nav_in, [slider] + outs)
        m_prev.click(lambda p, e, f, o, m: goto_match(p, e, f, o, m, -1),
                     nav_in, [slider] + outs)
        m_next.click(lambda p, e, f, o, m: goto_match(p, e, f, o, m, 1),
                     nav_in, [slider] + outs)

        # 切换 part/episode 时刷新事件下拉、清空匹配
        def _on_ep_extra(part, ep_label):
            name = _label_to_name(ep_label)
            entry = DATASET.get(name) if name else None
            return (gr.update(choices=_event_choices(entry), value=[]),
                    [], "_切换片段，筛选已重置_")
        ep_dd.change(_on_ep_extra, [part_dd, ep_dd],
                     [f_events, matches_state, filt_status])
        part_dd.change(lambda: ([], "_切换分片，筛选已重置_"), [],
                       [matches_state, filt_status])

        demo.load(load_frame, [part_dd, ep_dd, slider, overlay_ck], outs)
    return demo


def main():
    global DATASET
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/minestudio-data-10xx-v110",
                    help="数据集根目录")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"数据目录不存在: {args.data}")

    print(f"[+] 建立索引: {args.data}")
    DATASET = MineStudioDataset(args.data)
    print(f"[+] 共 {DATASET.num_episodes()} 个片段，{len(DATASET.parts())} 个 part")
    if DATASET.num_episodes() == 0:
        print("[!] 尚无可用片段（image 模态可能还没下载完）")

    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port,
                        share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
