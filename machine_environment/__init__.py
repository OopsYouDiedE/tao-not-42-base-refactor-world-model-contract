"""本机硬件与 CUDA 环境检测。

对外接口：
    MachineReport — 完整检测结果。
    collect_machine_report — 采集报告。
    format_machine_report — 渲染为人读文本。
    run_preflight, report_preflight — 开工前体检：显存 / 存储 / 网络三项建议指标。

数据管线机器与训练机器规格差别很大，下载前要确认磁盘容量（单个 image 分片 29GB），
训练前要确认显存（决定 micro-batch），故本模块不依赖 torch 也能工作。
"""

from machine_environment.hardware_report import (
    CUDAReport,
    DiskReport,
    GPUReport,
    MachineReport,
    MemoryReport,
    ProcessorReport,
    collect_machine_report,
    format_bytes,
    format_machine_report,
)
from machine_environment.preflight import (
    MINIMUM_FREE_DISK_BYTES,
    MINIMUM_GRAPHICS_MEMORY_BYTES,
    PreflightWarning,
    format_preflight,
    report_preflight,
    run_preflight,
)

__all__ = [
    "MINIMUM_FREE_DISK_BYTES",
    "MINIMUM_GRAPHICS_MEMORY_BYTES",
    "CUDAReport",
    "DiskReport",
    "GPUReport",
    "MachineReport",
    "MemoryReport",
    "PreflightWarning",
    "ProcessorReport",
    "collect_machine_report",
    "format_bytes",
    "format_machine_report",
    "format_preflight",
    "report_preflight",
    "run_preflight",
]
