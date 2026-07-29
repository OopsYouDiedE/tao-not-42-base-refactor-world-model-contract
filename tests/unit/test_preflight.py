"""环境体检的阈值判定测试：不依赖真实硬件，也不碰网络。"""

from __future__ import annotations

from machine_environment.hardware_report import (
    DiskReport,
    GPUReport,
    MachineReport,
)
from machine_environment.preflight import (
    MINIMUM_FREE_DISK_BYTES,
    MINIMUM_GRAPHICS_MEMORY_BYTES,
    PreflightWarning,
    check_free_disk,
    check_graphics_memory,
    format_preflight,
)

_GIBIBYTE = 1024 ** 3


def _report_with_graphics(*memory_gibibytes: int) -> MachineReport:
    """造一个只填了 GPU 显存的报告。"""
    return MachineReport(
        gpus=[
            GPUReport(
                index=index, name=f"GPU{index}", total_memory_bytes=size * _GIBIBYTE,
            )
            for index, size in enumerate(memory_gibibytes)
        ],
    )


def test_graphics_memory_at_threshold_passes() -> None:
    """恰好达到 24GiB 视为达标，边界不算不足。"""
    assert check_graphics_memory(_report_with_graphics(24)) is None


def test_graphics_memory_below_threshold_warns() -> None:
    """显存低于建议下限时给出警告，并提示调低 micro-batch。"""
    warning = check_graphics_memory(_report_with_graphics(16))
    assert warning is not None
    assert warning.category == "显存"
    assert "micro-batch" in warning.message


def test_graphics_memory_uses_largest_single_card() -> None:
    """取单卡最大值：两块 16GB 不等于一块 32GB，但有一块 32GB 就算达标。"""
    assert check_graphics_memory(_report_with_graphics(16, 32)) is None
    assert check_graphics_memory(_report_with_graphics(16, 16)) is not None


def test_missing_graphics_is_warned_not_assumed_fine() -> None:
    """检测不到 GPU 时必须警告——未知不等于达标。"""
    warning = check_graphics_memory(MachineReport())
    assert warning is not None
    assert warning.category == "显存"


def test_free_disk_below_threshold_warns() -> None:
    """可用空间低于 100GiB 时警告，并提示分片体积。"""
    report = MachineReport(disks=[DiskReport(path="/data", free_bytes=50 * _GIBIBYTE)])
    warning = check_free_disk(report)
    assert warning is not None
    assert warning.category == "存储"
    assert "29GB" in warning.message


def test_free_disk_uses_largest_volume() -> None:
    """多个卷时取最大可用值：只要有一个卷够大就还有地方放。"""
    report = MachineReport(
        disks=[
            DiskReport(path="/small", free_bytes=10 * _GIBIBYTE),
            DiskReport(path="/big", free_bytes=500 * _GIBIBYTE),
        ],
    )
    assert check_free_disk(report) is None


def test_free_disk_at_threshold_passes() -> None:
    """恰好 100GiB 视为达标。"""
    report = MachineReport(
        disks=[DiskReport(path="/data", free_bytes=MINIMUM_FREE_DISK_BYTES)],
    )
    assert check_free_disk(report) is None


def test_unreadable_disk_is_warned() -> None:
    """读不到容量时警告，不当作达标。"""
    report = MachineReport(disks=[DiskReport(path="/data", free_bytes=None)])
    assert check_free_disk(report) is not None


def test_thresholds_match_documented_values() -> None:
    """阈值就是文档承诺的 24GiB 显存与 100GiB 存储。"""
    assert MINIMUM_GRAPHICS_MEMORY_BYTES == 24 * _GIBIBYTE
    assert MINIMUM_FREE_DISK_BYTES == 100 * _GIBIBYTE


def test_format_preflight_states_warnings_are_advisory() -> None:
    """有警告时必须写明不阻止执行，避免被误读为致命错误。"""
    text = format_preflight(
        MachineReport(),
        [PreflightWarning(category="网络", message="连不上")],
    )
    assert "[警告] 网络" in text
    assert "不阻止继续执行" in text


def test_format_preflight_reports_all_clear() -> None:
    """三项达标时明确说明，不留空白让人猜。"""
    assert "均达标" in format_preflight(MachineReport(), [])
