"""采集 CPU / 内存 / 磁盘 / GPU / CUDA 信息。

对外接口：
    ProcessorReport, MemoryReport, DiskReport, GraphicsProcessorReport, CudaReport
    MachineReport — 汇总以上各项。
    collect_machine_report — 采集。
    format_machine_report — 渲染为对齐文本。
    format_bytes — 字节数 → GiB 文本。
    main — 命令行入口。

只用标准库 + 可选的 ``nvidia-smi`` / ``torch``，在没装 CUDA 栈的数据管线机器上也能跑。
检测不到的项一律为 None 并在输出里显示"未知"，不猜测、不静默填默认值。

CUDA 有三个互不相同的版本号，混为一谈是环境排错时最常见的误判来源，故分别报告：
驱动支持的最高版本（``nvidia-smi``）、已装工具链版本（``nvcc``）、torch 编译时链接的版本。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 子进程超时秒数：nvidia-smi 在驱动异常时会长时间挂住，必须设上限。
_COMMAND_TIMEOUT = 10

_BYTES_PER_GIBIBYTE = 1024 ** 3


def format_bytes(value: int | None) -> str:
    """字节数渲染为 GiB 文本，None 渲染为"未知"。"""
    if value is None:
        return "未知"
    return f"{value / _BYTES_PER_GIBIBYTE:.1f} GiB"


def _run_command(command: list[str]) -> str | None:
    """执行命令并返回 stdout，不可用或失败时返回 None。"""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


@dataclass
class ProcessorReport:
    """CPU 信息。

    Attributes
    ----------
    model : str or None
        型号名，如 ``AMD Ryzen 9 7950X``。
    physical_cores : int or None
        物理核心数。
    logical_cores : int or None
        逻辑核心数（含超线程）。
    architecture : str
        指令集架构，如 ``x86_64``、``AMD64``。
    """

    model: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    architecture: str = ""


@dataclass
class MemoryReport:
    """系统内存，单位字节。

    Attributes
    ----------
    total_bytes : int or None
        物理内存总量。
    available_bytes : int or None
        当前可用量。Linux 取 ``MemAvailable``（已扣除不可回收的缓存），
        比 ``MemFree`` 更接近实际可分配量。
    """

    total_bytes: int | None = None
    available_bytes: int | None = None


@dataclass
class DiskReport:
    """一个路径所在卷的容量，单位字节。

    Attributes
    ----------
    path : str
        被检查的路径。
    total_bytes, used_bytes, free_bytes : int or None
        所在卷的总量、已用、可用。
    """

    path: str
    total_bytes: int | None = None
    used_bytes: int | None = None
    free_bytes: int | None = None


@dataclass
class GraphicsProcessorReport:
    """单块 GPU 的信息。

    Attributes
    ----------
    index : int
        ``nvidia-smi`` 中的设备序号。
    name : str
        型号名，如 ``NVIDIA RTX PRO 6000 Blackwell``。
    total_memory_bytes, free_memory_bytes : int or None
        显存总量与可用量。
    driver_version : str or None
        显卡驱动版本。
    compute_capability : str or None
        计算能力，如 ``12.0``。
    """

    index: int
    name: str
    total_memory_bytes: int | None = None
    free_memory_bytes: int | None = None
    driver_version: str | None = None
    compute_capability: str | None = None


@dataclass
class CudaReport:
    """CUDA 环境。三个版本号来源不同，通常不相等。

    Attributes
    ----------
    driver_maximum_version : str or None
        驱动支持的最高 CUDA 版本（``nvidia-smi`` 报告）。这是上限，不是已装版本。
    toolkit_version : str or None
        已安装的 CUDA 工具链版本（``nvcc --version``）。
    torch_version : str or None
        torch 版本；torch 未安装时为 None。
    torch_compiled_cuda_version : str or None
        torch 编译时链接的 CUDA 版本。与工具链版本不一致通常无害，torch 自带运行时。
    torch_sees_cuda : bool or None
        ``torch.cuda.is_available()``；torch 未安装时为 None。
    """

    driver_maximum_version: str | None = None
    toolkit_version: str | None = None
    torch_version: str | None = None
    torch_compiled_cuda_version: str | None = None
    torch_sees_cuda: bool | None = None


@dataclass
class MachineReport:
    """本机环境汇总。

    Attributes
    ----------
    platform_description : str
        操作系统描述。
    python_version : str
        运行本模块的 Python 版本。
    processor : ProcessorReport
        CPU 信息。
    memory : MemoryReport
        内存信息。
    disks : list of DiskReport
        各被检查路径的卷容量。
    graphics_processors : list of GraphicsProcessorReport
        各 GPU 信息；检测不到时为空列表。
    cuda : CudaReport
        CUDA 环境。
    """

    platform_description: str = ""
    python_version: str = ""
    processor: ProcessorReport = field(default_factory=ProcessorReport)
    memory: MemoryReport = field(default_factory=MemoryReport)
    disks: list[DiskReport] = field(default_factory=list)
    graphics_processors: list[GraphicsProcessorReport] = field(default_factory=list)
    cuda: CudaReport = field(default_factory=CudaReport)


def _read_processor_model_from_proc() -> tuple[str | None, int | None]:
    """从 ``/proc/cpuinfo`` 读 CPU 型号与物理核心数，非 Linux 或读不到时返回 (None, None)。"""
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None, None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    model = None
    cores = None
    for line in content.splitlines():
        if model is None and line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
        elif cores is None and line.startswith("cpu cores"):
            try:
                cores = int(line.split(":", 1)[1].strip())
            except ValueError:
                cores = None
        if model is not None and cores is not None:
            break
    return model, cores


def _read_physical_cores_from_windows() -> int | None:
    """用 PowerShell CIM 读 Windows 物理核心数。

    ``os.cpu_count()`` 只给逻辑核心数；混合架构 CPU（P 核 + E 核）上物理核心数不是
    简单折半，必须真查。
    """
    if platform.system() != "Windows":
        return None
    output = _run_command(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "(Get-CimInstance Win32_Processor |"
            " Measure-Object -Property NumberOfCores -Sum).Sum",
        ],
    )
    if output is None:
        return None
    try:
        return int(output.strip())
    except ValueError:
        return None


def _read_processor_model_from_registry() -> str | None:
    """从 Windows 注册表读 CPU 型号；``platform.processor()`` 在 Windows 上只给family/model 数字。"""
    if platform.system() != "Windows":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except OSError:
        return None
    return str(value).strip() or None


def collect_processor_report() -> ProcessorReport:
    """采集 CPU 信息。"""
    model, physical_cores = _read_processor_model_from_proc()
    if model is None:
        model = _read_processor_model_from_registry()
    if model is None:
        model = platform.processor().strip() or None
    if physical_cores is None:
        physical_cores = _read_physical_cores_from_windows()
    return ProcessorReport(
        model=model,
        physical_cores=physical_cores,
        logical_cores=os.cpu_count(),
        architecture=platform.machine(),
    )


def _read_memory_from_proc() -> MemoryReport | None:
    """从 ``/proc/meminfo`` 读内存，非 Linux 或读不到时返回 None。"""
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in content.splitlines():
        matched = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s*kB", line)
        if matched:
            values[matched.group(1)] = int(matched.group(2)) * 1024
    if "MemTotal" not in values:
        return None
    return MemoryReport(
        total_bytes=values["MemTotal"],
        available_bytes=values.get("MemAvailable"),
    )


def _read_memory_from_windows() -> MemoryReport | None:
    """用 ``GlobalMemoryStatusEx`` 读 Windows 内存，失败时返回 None。"""
    if platform.system() != "Windows":
        return None
    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    try:
        succeeded = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None
    if not succeeded:
        return None
    return MemoryReport(
        total_bytes=int(status.ullTotalPhys),
        available_bytes=int(status.ullAvailPhys),
    )


def collect_memory_report() -> MemoryReport:
    """采集内存信息，检测不到时各字段为 None。"""
    return _read_memory_from_proc() or _read_memory_from_windows() or MemoryReport()


def collect_disk_reports(paths: list[Path]) -> list[DiskReport]:
    """采集各路径所在卷的容量。

    路径不存在时逐级上溯到最近的已存在祖先——检查尚未创建的输出目录能落在哪个卷上，
    是下载前估容量的常见需求。
    """
    reports: list[DiskReport] = []
    for path in paths:
        resolved = Path(path).resolve()
        probe = resolved
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
        except OSError:
            reports.append(DiskReport(path=str(resolved)))
            continue
        reports.append(
            DiskReport(
                path=str(resolved),
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
            ),
        )
    return reports


# nvidia-smi 的 CSV 查询字段，顺序与 _parse_graphics_line 的解包一致。
_SMI_FIELDS = (
    "index,name,memory.total,memory.free,driver_version,compute_cap"
)


def _parse_graphics_line(line: str) -> GraphicsProcessorReport | None:
    """解析一行 nvidia-smi CSV 输出，字段不足或序号非法时返回 None。"""
    cells = [cell.strip() for cell in line.split(",")]
    if len(cells) < 6:
        return None
    try:
        index = int(cells[0])
    except ValueError:
        return None

    def megabytes_to_bytes(text: str) -> int | None:
        """``memory.total`` 等字段以 MiB 为单位，且驱动异常时可能是 [N/A]。"""
        try:
            return int(float(text)) * 1024 * 1024
        except ValueError:
            return None

    return GraphicsProcessorReport(
        index=index,
        name=cells[1],
        total_memory_bytes=megabytes_to_bytes(cells[2]),
        free_memory_bytes=megabytes_to_bytes(cells[3]),
        driver_version=cells[4] or None,
        compute_capability=cells[5] or None,
    )


def collect_graphics_processor_reports() -> list[GraphicsProcessorReport]:
    """用 ``nvidia-smi`` 采集各 GPU 信息；无 NVIDIA 驱动时返回空列表。"""
    output = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={_SMI_FIELDS}",
            "--format=csv,noheader,nounits",
        ],
    )
    if output is None:
        return []
    reports = [
        report
        for line in output.splitlines()
        if line.strip()
        if (report := _parse_graphics_line(line)) is not None
    ]
    return sorted(reports, key=lambda report: report.index)


def _read_driver_maximum_cuda_version() -> str | None:
    """从 ``nvidia-smi`` 首部读驱动支持的最高 CUDA 版本。"""
    output = _run_command(["nvidia-smi"])
    if output is None:
        return None
    matched = re.search(r"CUDA Version:\s*([\d.]+)", output)
    return matched.group(1) if matched else None


def _read_toolkit_version() -> str | None:
    """从 ``nvcc --version`` 读已装工具链版本；未装 nvcc 时返回 None。"""
    output = _run_command(["nvcc", "--version"])
    if output is None:
        return None
    matched = re.search(r"release\s+([\d.]+)", output)
    return matched.group(1) if matched else None


def collect_cuda_report() -> CudaReport:
    """采集 CUDA 环境。torch 未安装时相关字段为 None，不视为错误。"""
    report = CudaReport(
        driver_maximum_version=_read_driver_maximum_cuda_version(),
        toolkit_version=_read_toolkit_version(),
    )
    try:
        import torch
    except ImportError:
        return report
    report.torch_version = torch.__version__
    report.torch_compiled_cuda_version = torch.version.cuda
    try:
        report.torch_sees_cuda = bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError):
        # 驱动 / 运行时不匹配时 is_available 可能抛异常而非返回 False。
        report.torch_sees_cuda = False
    return report


def collect_machine_report(paths: list[Path] | None = None) -> MachineReport:
    """采集完整环境报告。

    Parameters
    ----------
    paths : list of Path or None
        要检查容量的路径。None 时检查当前目录、``runs/bc_datasets`` 与 ``runs/trains``
        ——数据与 checkpoint 可能落在不同卷上，分别报告才有意义。

    Returns
    -------
    MachineReport
        检测不到的项为 None 或空列表。
    """
    if paths is None:
        paths = [Path.cwd(), Path("runs/bc_datasets"), Path("runs/trains")]
    return MachineReport(
        platform_description=platform.platform(),
        python_version=platform.python_version(),
        processor=collect_processor_report(),
        memory=collect_memory_report(),
        disks=collect_disk_reports(paths),
        graphics_processors=collect_graphics_processor_reports(),
        cuda=collect_cuda_report(),
    )


def _format_optional(value: object) -> str:
    """None 渲染为"未知"，其余转字符串。"""
    return "未知" if value is None else str(value)


def format_machine_report(report: MachineReport) -> str:
    """把报告渲染为对齐文本。"""
    lines: list[str] = []

    lines.append("系统")
    lines.append(f"  操作系统        {report.platform_description}")
    lines.append(f"  Python          {report.python_version}")

    processor = report.processor
    lines.append("")
    lines.append("CPU")
    lines.append(f"  型号            {_format_optional(processor.model)}")
    lines.append(
        f"  核心            物理 {_format_optional(processor.physical_cores)} / "
        f"逻辑 {_format_optional(processor.logical_cores)}",
    )
    lines.append(f"  架构            {processor.architecture or '未知'}")

    lines.append("")
    lines.append("内存")
    lines.append(f"  总量            {format_bytes(report.memory.total_bytes)}")
    lines.append(f"  可用            {format_bytes(report.memory.available_bytes)}")

    lines.append("")
    lines.append("磁盘")
    for disk in report.disks:
        lines.append(f"  {disk.path}")
        lines.append(
            f"    总量 {format_bytes(disk.total_bytes)}  "
            f"已用 {format_bytes(disk.used_bytes)}  "
            f"可用 {format_bytes(disk.free_bytes)}",
        )

    lines.append("")
    lines.append("GPU")
    if not report.graphics_processors:
        lines.append("  未检测到 NVIDIA GPU（nvidia-smi 不可用）")
    for graphics in report.graphics_processors:
        lines.append(f"  [{graphics.index}] {graphics.name}")
        lines.append(
            f"    显存 {format_bytes(graphics.total_memory_bytes)}  "
            f"可用 {format_bytes(graphics.free_memory_bytes)}  "
            f"算力 {_format_optional(graphics.compute_capability)}  "
            f"驱动 {_format_optional(graphics.driver_version)}",
        )

    cuda = report.cuda
    lines.append("")
    lines.append("CUDA")
    lines.append(f"  驱动支持上限    {_format_optional(cuda.driver_maximum_version)}")
    lines.append(f"  工具链 (nvcc)   {_format_optional(cuda.toolkit_version)}")
    if cuda.torch_version is None:
        lines.append("  torch           未安装（数据管线无需 CUDA 栈）")
    else:
        lines.append(f"  torch           {cuda.torch_version}")
        lines.append(
            f"  torch 编译 CUDA {_format_optional(cuda.torch_compiled_cuda_version)}",
        )
        lines.append(
            f"  torch 可见 CUDA {'是' if cuda.torch_sees_cuda else '否'}",
        )
    return "\n".join(lines)


def main() -> None:
    """命令行入口：打印本机环境报告。"""
    # Windows 控制台默认代码页不是 UTF-8，中文会乱码；显式切到 UTF-8。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="检测本机 CPU / 内存 / 磁盘 / GPU / CUDA 环境")
    parser.add_argument(
        "--path", type=Path, nargs="*", default=None,
        help="要检查容量的路径；默认当前目录与 runs/bc_datasets、runs/trains",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非对齐文本")
    arguments = parser.parse_args()

    report = collect_machine_report(arguments.path)
    if arguments.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(format_machine_report(report))


if __name__ == "__main__":
    main()
