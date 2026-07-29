"""开工前的环境体检：报告本机规格并对不达标项发出警告。

对外接口：
    MINIMUM_GRAPHICS_MEMORY_BYTES, MINIMUM_FREE_DISK_BYTES — 建议下限。
    PreflightWarning — 一条警告。
    check_graphics_memory — 显存是否达到建议下限。
    check_free_disk — 目标盘可用空间是否达到建议下限。
    check_network_reachability — 到 HuggingFace 的连通性。
    run_preflight — 跑完全部检查，返回警告列表。
    format_preflight — 渲染体检结论为文本。
    main — 命令行入口。

三项都是**建议性**阈值：不达标只警告，调用方照常继续。显存偏小可以调低
``--micro-batch``，存储偏小可以少下几个分片，网络不通只影响下载而不影响已下好的数据，
因此这里不替使用者做终止决定。

数据管线机器不装 CUDA 栈也能跑：检测不到的项按"未知"处理，并单独给出一条提示，
不猜测、不当作达标。
"""

from __future__ import annotations

import argparse
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from machine_environment.hardware_report import (
    MachineReport,
    collect_machine_report,
    format_bytes,
    format_machine_report,
)

_BYTES_PER_GIBIBYTE = 1024 ** 3

# 显存建议下限。低于此值时 26B-A4B 的 bf16 LoRA 几乎必然 OOM，需要显著调低
# micro-batch 或换更小的主干。
MINIMUM_GRAPHICS_MEMORY_BYTES = 24 * _BYTES_PER_GIBIBYTE

# 目标盘可用空间建议下限。单个 image 分片可达 29GB，全量下载远超此值，
# 100GiB 只是"能下几个分片试水"的底线。
MINIMUM_FREE_DISK_BYTES = 100 * _BYTES_PER_GIBIBYTE

# 连通性探测目标：下载数据集与模型权重都要经过它。
_NETWORK_HOST = "huggingface.co"
_NETWORK_PORT = 443
_NETWORK_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class PreflightWarning:
    """一条体检警告。

    Attributes
    ----------
    category : str
        检查项名称，如 ``"显存"``、``"存储"``、``"网络"``。
    message : str
        面向使用者的说明，含实测值、建议下限与应对方向。
    """

    category: str
    message: str


def check_graphics_memory(
    report: MachineReport,
    minimum_bytes: int = MINIMUM_GRAPHICS_MEMORY_BYTES,
) -> PreflightWarning | None:
    """检查最大单卡显存是否达到建议下限。

    Parameters
    ----------
    report : MachineReport
        已采集的环境报告。
    minimum_bytes : int
        建议下限，单位字节。

    Returns
    -------
    PreflightWarning or None
        达标时返回 None。检测不到 GPU 或显存读不到时也返回警告——未知不等于达标。

    Notes
    -----
    取**单卡最大值**而不是多卡合计：LoRA 训练跑在单卡上，两块 12GB 不等于一块 24GB。
    """
    candidates = [
        graphics.total_memory_bytes
        for graphics in report.gpus
        if graphics.total_memory_bytes is not None
    ]
    if not candidates:
        return PreflightWarning(
            category="显存",
            message=(
                "未检测到可用 NVIDIA 显存（nvidia-smi 不可用或驱动异常）。"
                "训练前请在装有 CUDA 栈的机器上复查；数据管线不受影响。"
            ),
        )
    largest = max(candidates)
    if largest >= minimum_bytes:
        return None
    return PreflightWarning(
        category="显存",
        message=(
            f"单卡最大显存 {format_bytes(largest)}，低于建议的 "
            f"{format_bytes(minimum_bytes)}。26B-A4B 的 bf16 LoRA 很可能 OOM："
            "请调低 --micro-batch，或换 gemma-4-E2B-it / E4B-it 这类更小的主干。"
        ),
    )


def check_free_disk(
    report: MachineReport,
    minimum_bytes: int = MINIMUM_FREE_DISK_BYTES,
) -> PreflightWarning | None:
    """检查各被检查路径中最大的可用空间是否达到建议下限。

    Returns
    -------
    PreflightWarning or None
        达标时返回 None。取各路径最大值：数据与 checkpoint 可能落在不同卷上，
        只要有一个卷够大就还有地方放。
    """
    candidates = [
        disk.free_bytes for disk in report.disks if disk.free_bytes is not None
    ]
    if not candidates:
        return PreflightWarning(
            category="存储",
            message="读不到任何卷的可用空间，下载前请手动确认容量。",
        )
    largest = max(candidates)
    if largest >= minimum_bytes:
        return None
    return PreflightWarning(
        category="存储",
        message=(
            f"可用空间最多的卷只剩 {format_bytes(largest)}，低于建议的 "
            f"{format_bytes(minimum_bytes)}。单个 image 分片可达 29GB："
            "请用 --maximum-parts 限制分片数，或把 --output-dir 指到更大的盘。"
        ),
    )


def check_network_reachability(
    host: str = _NETWORK_HOST,
    port: int = _NETWORK_PORT,
    timeout_seconds: float = _NETWORK_TIMEOUT_SECONDS,
) -> PreflightWarning | None:
    """探测到 HuggingFace 的 TCP 连通性。

    Returns
    -------
    PreflightWarning or None
        连通时返回 None。只建 TCP 连接，不发 HTTP 请求，也不携带任何凭据。

    Notes
    -----
    探测失败不代表机器不能用：数据已下载完时训练不需要外网。企业代理环境下
    直连常被挡，此时需要设 ``HF_ENDPOINT`` 或代理环境变量。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return None
    except OSError as error:
        return PreflightWarning(
            category="网络",
            message=(
                f"连不上 {host}:{port}（{type(error).__name__}）。"
                "下载数据集与模型权重会失败：请检查代理设置或 HF_ENDPOINT 镜像。"
                "若数据已下载完毕，训练不受影响。"
            ),
        )


def run_preflight(
    paths: list[Path] | None = None,
    minimum_graphics_memory_bytes: int = MINIMUM_GRAPHICS_MEMORY_BYTES,
    minimum_free_disk_bytes: int = MINIMUM_FREE_DISK_BYTES,
    check_network: bool = True,
) -> tuple[MachineReport, list[PreflightWarning]]:
    """采集环境报告并跑全部检查。

    Parameters
    ----------
    paths : list of Path or None
        要检查容量的路径，None 用 ``collect_machine_report`` 的默认值。
    minimum_graphics_memory_bytes, minimum_free_disk_bytes : int
        两项建议下限，单位字节。
    check_network : bool
        是否探测网络。离线环境可关掉以免每次多等一个超时。

    Returns
    -------
    tuple
        ``(报告, 警告列表)``。警告列表为空表示三项都达标。
    """
    report = collect_machine_report(paths)
    warnings = [
        check_graphics_memory(report, minimum_graphics_memory_bytes),
        check_free_disk(report, minimum_free_disk_bytes),
        check_network_reachability() if check_network else None,
    ]
    return report, [warning for warning in warnings if warning is not None]


def format_preflight(report: MachineReport, warnings: list[PreflightWarning]) -> str:
    """把体检结论渲染为文本：完整规格 + 警告清单。"""
    lines = [format_machine_report(report), "", "体检"]
    if not warnings:
        lines.append("  三项建议指标均达标（显存、存储、网络）。")
        return "\n".join(lines)
    for warning in warnings:
        lines.append(f"  [警告] {warning.category}：{warning.message}")
    lines.append("")
    lines.append("  以上均为建议性阈值，不阻止继续执行。")
    return "\n".join(lines)


def report_preflight(
    paths: list[Path] | None = None,
    check_network: bool = True,
    stream: object = None,
) -> list[PreflightWarning]:
    """跑体检并把结论打到 stderr，供下载与训练入口在开工前调用。

    Returns
    -------
    list of PreflightWarning
        警告列表，供调用方在最终结果里一并汇报。

    Notes
    -----
    写 stderr 而不是 stdout：这些入口的 stdout 是 JSON 统计，要能直接管道给 ``jq``。
    """
    report, warnings = run_preflight(paths, check_network=check_network)
    target = sys.stderr if stream is None else stream
    print(format_preflight(report, warnings), file=target)
    return warnings


def main() -> None:
    """命令行入口：打印环境报告与体检警告。"""
    # Windows 控制台默认代码页不是 UTF-8，中文会乱码；显式切到 UTF-8。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="开工前环境体检：显存 / 存储 / 网络三项建议指标",
    )
    parser.add_argument(
        "--path", type=Path, nargs="*", default=None,
        help="要检查容量的路径；默认当前目录与 runs/bc_datasets、runs/trains",
    )
    parser.add_argument(
        "--no-network-check", action="store_true", help="跳过网络连通性探测",
    )
    arguments = parser.parse_args()

    report, warnings = run_preflight(
        arguments.path, check_network=not arguments.no_network_check,
    )
    print(format_preflight(report, warnings))


if __name__ == "__main__":
    main()
