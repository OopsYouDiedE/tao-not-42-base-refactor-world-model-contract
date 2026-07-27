"""本机硬件与 CUDA 环境检测。

对外接口：
    MachineReport — 完整检测结果。
    collect_machine_report — 采集报告。
    format_machine_report — 渲染为人读文本。

数据管线机器与训练机器规格差别很大，下载前要确认磁盘容量（单个 image 分片 29GB），
训练前要确认显存（决定 micro-batch），故本模块不依赖 torch 也能工作。
"""

from machine_environment.hardware_report import (
    CudaReport,
    DiskReport,
    GraphicsProcessorReport,
    MachineReport,
    MemoryReport,
    ProcessorReport,
    collect_machine_report,
    format_bytes,
    format_machine_report,
)

__all__ = [
    "CudaReport",
    "DiskReport",
    "GraphicsProcessorReport",
    "MachineReport",
    "MemoryReport",
    "ProcessorReport",
    "collect_machine_report",
    "format_bytes",
    "format_machine_report",
]
