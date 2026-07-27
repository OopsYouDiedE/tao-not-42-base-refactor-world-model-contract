"""环境检测的解析与渲染测试：不依赖真实硬件，也不要求装 CUDA。"""

from __future__ import annotations

from pathlib import Path

from machine_environment.hardware_report import (
    CudaReport,
    MachineReport,
    ProcessorReport,
    _parse_graphics_line,
    collect_disk_reports,
    collect_machine_report,
    collect_processor_report,
    format_bytes,
    format_machine_report,
)


def test_format_bytes_uses_gibibytes() -> None:
    """字节按 1024 进制换算为 GiB，不是 1000 进制。"""
    assert format_bytes(8 * 1024 ** 3) == "8.0 GiB"


def test_format_bytes_renders_unknown_for_none() -> None:
    """检测不到的容量显示"未知"，不显示 0。"""
    assert format_bytes(None) == "未知"


def test_parse_graphics_line_converts_mebibytes() -> None:
    """nvidia-smi 的显存字段以 MiB 计，需换算成字节。"""
    report = _parse_graphics_line("0, NVIDIA GeForce RTX 3070, 8192, 6800, 595.79, 8.6")
    assert report is not None
    assert report.name == "NVIDIA GeForce RTX 3070"
    assert report.total_memory_bytes == 8192 * 1024 * 1024
    assert report.compute_capability == "8.6"


def test_parse_graphics_line_tolerates_unavailable_memory() -> None:
    """驱动异常时字段可能是 [N/A]，应降级为 None 而非抛错。"""
    report = _parse_graphics_line("1, Some GPU, [N/A], [N/A], 500.00, 9.0")
    assert report is not None
    assert report.total_memory_bytes is None
    assert report.free_memory_bytes is None


def test_parse_graphics_line_rejects_truncated_output() -> None:
    """字段不足时返回 None，不产生半填充的报告。"""
    assert _parse_graphics_line("0, NVIDIA GeForce RTX 3070") is None


def test_parse_graphics_line_rejects_non_numeric_index() -> None:
    """序号非法（如把表头当数据）时返回 None。"""
    assert _parse_graphics_line("index, name, 100, 50, 1.0, 2.0") is None


def test_disk_report_walks_up_to_existing_ancestor(tmp_path: Path) -> None:
    """路径尚未创建时上溯到已存在的祖先，仍能报出所在卷容量。"""
    reports = collect_disk_reports([tmp_path / "not" / "created" / "yet"])
    assert len(reports) == 1
    assert reports[0].total_bytes is not None
    assert reports[0].free_bytes is not None


def test_processor_report_always_has_logical_cores() -> None:
    """逻辑核心数来自标准库，任何平台都应拿到。"""
    assert (collect_processor_report().logical_cores or 0) >= 1


def test_format_report_marks_absent_gpu_and_torch() -> None:
    """无 GPU、无 torch 的机器上渲染不报错，并明确写出缺失原因。"""
    text = format_machine_report(
        MachineReport(
            platform_description="Linux-6.6",
            python_version="3.12.0",
            processor=ProcessorReport(model=None, logical_cores=8),
            cuda=CudaReport(),
        ),
    )
    assert "未检测到 NVIDIA GPU" in text
    assert "torch           未安装" in text
    assert "未知" in text  # CPU 型号缺失


def test_collect_machine_report_runs_on_this_machine() -> None:
    """端到端采集在本机不抛异常，且必填字段非空。"""
    report = collect_machine_report([Path.cwd()])
    assert report.platform_description
    assert report.python_version
    assert len(report.disks) == 1
