"""Gradio 动作查看器：选一条轨迹、拖进度条，看该帧起若干帧的动作编码结果。

用法::

    python -m tools.action_inspector --dataset-dir runs/bc_datasets/minestudio-data-10xx-v110

image 模态可用时同时显示画面。尾部帧数不足一个窗口时不显示，不做补齐。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np

from bc_datasets.minestudio.lmdb_modality_reader import (
    TrajectoryReader,
    discover_part_directories,
)
from bc_datasets.minestudio.lumine_action_codec import (
    DEGREES_PER_PIXEL,
    MINECRAFT_KEYMAP,
    encode_lumine_action,
)

# 查看窗口的帧数。
WINDOW_FRAMES = 5
# 下拉框最多列出的 episode 数。10xx 全量 1846 条，全列会拖慢页面。
MAX_EPISODES = 200


class InspectorState:
    """持有读取器与 episode 列表。

    Attributes
    ----------
    reader : TrajectoryReader
        多模态读取器。image 模态缺失时只含 action。
    has_images : bool
        image 模态是否可用。
    episodes : list of str
        可查看的 episode 名，已按名称排序并截到 ``MAX_EPISODES``。
    """

    def __init__(self, dataset_directory: Path) -> None:
        modalities = ["action"]
        if discover_part_directories(dataset_directory, "image"):
            modalities.append("image")
        # TrajectoryReader 取各模态 episode 的交集，只有部分 image 分片时会自动落到
        # 那些分片对应的 episode 上。
        self.reader = TrajectoryReader([dataset_directory], modalities)
        self.has_images = "image" in modalities
        names = sorted(self.reader.episode_names())
        self.episodes = names[:MAX_EPISODES]

    def frame_count(self, episode: str) -> int:
        """episode 的帧数。"""
        return self.reader.episode_length(episode)

    def close(self) -> None:
        """关闭底层 LMDB 环境。"""
        self.reader.close()


def _frame_rows(actions: dict[str, np.ndarray], start: int) -> list[list[str]]:
    """逐帧拆出相机增量与按住的键，供表格显示。

    Parameters
    ----------
    actions : dict of str to numpy.ndarray
        一个窗口的 action 切片。
    start : int
        窗口起始帧号，用于标注绝对帧号。

    Returns
    -------
    list of list of str
        每帧一行：帧号、Δx 像素、Δy 像素、度数、按住的键。
    """
    camera = np.asarray(actions["camera"], dtype=np.float64)
    rows: list[list[str]] = []
    for offset in range(camera.shape[0]):
        pitch, yaw = camera[offset]
        held = [
            name
            for field, name in MINECRAFT_KEYMAP.items()
            if field in actions and bool(np.asarray(actions[field])[offset])
        ]
        rows.append(
            [
                str(start + offset),
                f"{yaw / DEGREES_PER_PIXEL:+.1f}",
                f"{pitch / DEGREES_PER_PIXEL:+.1f}",
                f"{yaw:+.3f} / {pitch:+.3f}",
                " ".join(held) if held else "—",
            ],
        )
    return rows


def _encoded_variants(actions: dict[str, np.ndarray]) -> str:
    """按不同 chunk 粒度编码同一窗口，便于对比控制密度。

    Returns
    -------
    str
        Markdown 文本。窗口帧数不能被某粒度整除时跳过该粒度。
    """
    num_frames = np.asarray(actions["camera"]).shape[0]
    lines = ["### 编码结果"]
    for frames_per_chunk in (1, 5):
        if num_frames % frames_per_chunk != 0:
            continue
        window = encode_lumine_action(actions, frames_per_chunk=frames_per_chunk)
        chunk_ms = frames_per_chunk * 50
        lines.append(
            f"**{frames_per_chunk} 帧/chunk（{chunk_ms}ms，{1000 // chunk_ms}Hz，"
            f"{len(window.chunks)} 个 chunk）**",
        )
        lines.append(f"```\n{window.to_text()}\n```")
    return "\n\n".join(lines)


def _summary(actions: dict[str, np.ndarray], start: int, total: int) -> str:
    """窗口的一行摘要。"""
    camera = np.asarray(actions["camera"], dtype=np.float64)
    pitch_sum, yaw_sum = camera.sum(axis=0)
    moving = int((np.abs(camera) > 1e-9).any(axis=1).sum())
    return (
        f"帧 **{start}** 到 **{start + camera.shape[0] - 1}**，共 {total} 帧。"
        f"窗口累计转角 yaw {yaw_sum:+.2f}°、pitch {pitch_sum:+.2f}°，"
        f"其中 {moving}/{camera.shape[0]} 帧有视角移动。"
    )


def build_interface(state: InspectorState) -> gr.Blocks:
    """组装 Gradio 界面。"""

    def inspect(episode: str, start: int) -> tuple:
        """读取一个窗口并渲染。帧数不足时各输出置空。"""
        if not episode:
            return "请选择一条轨迹。", [], "", None
        total = state.frame_count(episode)
        start = int(start)
        if start + WINDOW_FRAMES > total:
            message = (
                f"帧 {start} 起不足 {WINDOW_FRAMES} 帧（该轨迹共 {total} 帧，"
                f"最大可选 {max(0, total - WINDOW_FRAMES)}），不显示。"
            )
            return message, [], "", None
        window = state.reader.read_window(episode, start, WINDOW_FRAMES)
        actions = window["action"]
        frames = window.get("image") if state.has_images else None
        gallery = (
            [(frame, f"帧 {start + index}") for index, frame in enumerate(frames)]
            if frames is not None
            else None
        )
        return (
            _summary(actions, start, total),
            _frame_rows(actions, start),
            _encoded_variants(actions),
            gallery,
        )

    def on_episode_change(episode: str) -> gr.Slider:
        """切换轨迹时把进度条上限调整到该轨迹的可选范围。"""
        if not episode:
            return gr.Slider(maximum=0, value=0)
        maximum = max(0, state.frame_count(episode) - WINDOW_FRAMES)
        return gr.Slider(maximum=maximum, value=0)

    with gr.Blocks(title="Lumine 动作查看器") as interface:
        gr.Markdown(
            f"# Lumine 动作查看器\n"
            f"选一条轨迹，拖进度条查看该帧起 {WINDOW_FRAMES} 帧的动作编码。"
            + ("" if state.has_images else "\n\nimage 模态未下载，不显示画面。"),
        )
        with gr.Row():
            episode_input = gr.Dropdown(
                choices=state.episodes,
                value=state.episodes[0] if state.episodes else None,
                label=f"轨迹（共列出 {len(state.episodes)} 条）",
                scale=3,
            )
            start_input = gr.Slider(
                minimum=0,
                maximum=max(
                    0,
                    state.frame_count(state.episodes[0]) - WINDOW_FRAMES,
                )
                if state.episodes
                else 0,
                step=1,
                value=0,
                label="起始帧",
                scale=4,
            )
        summary_output = gr.Markdown()
        if state.has_images:
            gallery_output = gr.Gallery(label="画面", columns=WINDOW_FRAMES, height=200)
        else:
            gallery_output = gr.Gallery(visible=False)
        table_output = gr.Dataframe(
            headers=["帧", "Δx 像素", "Δy 像素", "yaw / pitch 度", "按住的键"],
            label="逐帧",
            wrap=True,
        )
        encoded_output = gr.Markdown()

        outputs = [summary_output, table_output, encoded_output, gallery_output]
        episode_input.change(on_episode_change, episode_input, start_input)
        episode_input.change(inspect, [episode_input, start_input], outputs)
        start_input.change(inspect, [episode_input, start_input], outputs)
        interface.load(inspect, [episode_input, start_input], outputs)
    return interface


def main() -> None:
    """命令行入口：启动查看器。"""
    parser = argparse.ArgumentParser(description="Lumine 动作编码查看器")
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("runs/bc_datasets/minestudio-data-10xx-v110"),
        help="MineStudio 数据集根目录",
    )
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    arguments = parser.parse_args()

    state = InspectorState(arguments.dataset_dir)
    if not state.episodes:
        raise SystemExit(f"{arguments.dataset_dir} 下没有可用 episode")
    interface = build_interface(state)
    try:
        # 仅监听本机回环。这个界面没有任何鉴权，不应暴露到网络上。
        interface.launch(
            server_name="127.0.0.1",
            server_port=arguments.port,
            share=arguments.share,
        )
    finally:
        state.close()


if __name__ == "__main__":
    main()
