"""使用真实 CraftGround 验证 NVIDIA 渲染、共享内存与 CUDA IPC。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .runtime import create_environment, prepare_runtime_instance, prepare_runtime_template


def _run_checked(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def validate_gpu_host() -> dict[str, Any]:
    """验证当前 DISPLAY 由 NVIDIA GPU 渲染且 PyTorch CUDA 可执行。"""
    display = os.environ.get("DISPLAY")
    if not display:
        raise RuntimeError("DISPLAY 未设置，无法创建真实 OpenGL 渲染上下文")
    glxinfo = _run_checked(["glxinfo", "-B"])
    if "OpenGL vendor string: NVIDIA Corporation" not in glxinfo:
        raise RuntimeError("当前 OpenGL vendor 不是 NVIDIA Corporation")
    if "llvmpipe" in glxinfo.lower():
        raise RuntimeError("当前 DISPLAY 使用 llvmpipe 软件渲染")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False")
    probe = torch.arange(16, dtype=torch.float32, device="cuda").square().sum()
    if probe.item() != 1240.0:
        raise RuntimeError("PyTorch CUDA 运算结果不正确")
    renderer = next(
        line.split(":", 1)[1].strip()
        for line in glxinfo.splitlines()
        if line.startswith("OpenGL renderer string:")
    )
    return {
        "display": display,
        "opengl_renderer": renderer,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
    }


def _as_numpy(frame: Any) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        return np.ascontiguousarray(frame)
    return frame.detach().cpu().numpy()


def _frame_metrics(frame: Any) -> dict[str, Any]:
    array = _as_numpy(frame)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
    }


def _assert_frame_contract(frame: Any, mode: str, width: int, height: int) -> None:
    expected_shape = (height, width, 3)
    if tuple(frame.shape) != expected_shape:
        raise RuntimeError(f"{mode} 帧 shape 错误：{tuple(frame.shape)} != {expected_shape}")
    if mode == "raw":
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise RuntimeError("RAW 帧必须是 numpy uint8 数组")
    else:
        import torch

        if not isinstance(frame, torch.Tensor):
            raise RuntimeError("ZEROCOPY_TORCH 帧不是 torch.Tensor")
        if frame.dtype != torch.uint8 or frame.device.type != "cuda":
            raise RuntimeError(
                f"ZEROCOPY_TORCH 帧必须是 CUDA uint8，实际为 {frame.device} {frame.dtype}"
            )
    array = _as_numpy(frame)
    if int(array.max()) == int(array.min()):
        raise RuntimeError(f"{mode} 帧没有可见像素变化范围")


def _capture_nvidia_processes() -> str:
    return _run_checked(["nvidia-smi"])


def validate_mode(
    *,
    mode: str,
    template: Path,
    output: Path,
    width: int,
    height: int,
    steps: int,
    port: int,
) -> dict[str, Any]:
    """启动一个真实环境并验证指定屏幕编码模式。"""
    from craftground.environment.action_space import no_op_v2

    runtime_path = prepare_runtime_instance(
        f"cuda-validation-{mode}",
        template=template,
        instances_root=output / "instances",
    )
    environment = create_environment(
        runtime_path=runtime_path,
        image_width=width,
        image_height=height,
        port=port,
        find_free_port=False,
        use_shared_memory=True,
        cleanup_world=True,
        verbose=True,
        verbose_gradle=True,
        verbose_jvm=True,
        screen_encoding_mode=mode,  # type: ignore[arg-type]
    )
    frames: list[np.ndarray] = []
    frame_metrics: list[dict[str, Any]] = []
    public_data_pointers: list[int] = []
    transport_data_pointers: list[int] = []
    ipc_handle_sizes: list[int] = []
    process_handle = None
    try:
        observation, _ = environment.reset(options={"fast_reset": False})
        process_handle = environment.process
        for step_index in range(steps + 1):
            frame = observation["rgb"]
            _assert_frame_contract(frame, mode, width, height)
            frames.append(_as_numpy(frame).copy())
            frame_metrics.append(_frame_metrics(frame))
            ipc_handle_sizes.append(len(observation["full"].ipc_handle))
            if mode == "zerocopy_torch":
                public_data_pointers.append(frame.data_ptr())
                transport = environment.observation_converter.last_observations[0]
                transport_data_pointers.append(transport.data_ptr())
                if tuple(transport.shape) != (height, width, 4):
                    raise RuntimeError(
                        f"CUDA IPC live view shape 错误：{tuple(transport.shape)}"
                    )
                if transport.device.type != "cuda":
                    raise RuntimeError(f"CUDA IPC live view 不在 CUDA：{transport.device}")
            if step_index == steps:
                break
            action = no_op_v2()
            action["forward"] = step_index < max(1, steps // 2)
            action["camera_yaw"] = 12.0
            observation, _, _, _, _ = environment.step(action)

        changed_pairs = sum(
            not np.array_equal(previous, current)
            for previous, current in zip(frames, frames[1:])
        )
        if changed_pairs == 0:
            raise RuntimeError(f"{mode} 的所有相邻帧完全相同")
        if environment.use_shared_memory is not True:
            raise RuntimeError("环境未启用共享内存 IPC")
        if type(environment.ipc).__name__ != "BoostIPC":
            raise RuntimeError(f"共享内存 IPC 类型错误：{type(environment.ipc).__name__}")
        if mode == "zerocopy_torch":
            nonempty_handles = [size for size in ipc_handle_sizes if size > 0]
            if not nonempty_handles or nonempty_handles[0] != 68:
                raise RuntimeError(f"CUDA IPC handle 长度错误：{ipc_handle_sizes}")
            if len(set(transport_data_pointers)) != 1:
                raise RuntimeError("CUDA IPC live view 的 data pointer 在运行中发生变化")
            if any(
                public_pointer == transport_data_pointers[index]
                for index, public_pointer in enumerate(public_data_pointers)
            ):
                raise RuntimeError("公开 RGB 帧与 CUDA IPC live view 复用了同一 data pointer")

        Image.fromarray(frames[-1]).save(output / f"{mode}.png")
        return {
            "mode": mode,
            "shared_memory": True,
            "ipc_class": type(environment.ipc).__name__,
            "runtime_path": str(runtime_path),
            "frames": frame_metrics,
            "changed_adjacent_frame_pairs": changed_pairs,
            "ipc_handle_sizes": ipc_handle_sizes,
            "public_rgb_data_pointers": public_data_pointers,
            "transport_live_view_data_pointers": transport_data_pointers,
            "nvidia_smi_while_running": _capture_nvidia_processes(),
        }
    finally:
        environment.close()
        if process_handle is not None and process_handle.poll() is None:
            raise RuntimeError(f"{mode} 环境关闭后 Gradle/JVM 进程仍在运行")


def run_validation(
    output: Path,
    *,
    width: int,
    height: int,
    steps: int,
    port: int,
) -> dict[str, Any]:
    """执行 RAW 与 CUDA IPC zero-copy 的完整真实验收。"""
    if steps < 2:
        raise ValueError("steps 必须至少为 2")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    host = validate_gpu_host()
    template = prepare_runtime_template(output / "runtime-template")
    modes = [
        validate_mode(
            mode=mode,
            template=template,
            output=output,
            width=width,
            height=height,
            steps=steps,
            port=port + index,
        )
        for index, mode in enumerate(("raw", "zerocopy_torch"))
    ]
    result = {
        "status": "passed",
        "host": host,
        "packages": {
            "craftground": version("craftground"),
            "craftground-runtime-mc121": version("craftground-runtime-mc121"),
        },
        "template": str(template),
        "width": width,
        "height": height,
        "steps": steps,
        "modes": modes,
        "zero_copy_boundary": (
            "GL texture 到 CUDA IPC 共享 RGBA 缓冲区为 GPU-to-GPU；Python 内部张量是该缓冲区"
            "的 live view；公开 RGB 张量执行 clone、去 alpha 和垂直翻转，仍保留在 CUDA。"
        ),
    }
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="验证真实 CraftGround CUDA 渲染和 zero-copy")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/craftground_cuda_validation"),
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18300)
    arguments = parser.parse_args()
    result = run_validation(
        arguments.output,
        width=arguments.width,
        height=arguments.height,
        steps=arguments.steps,
        port=arguments.port,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
