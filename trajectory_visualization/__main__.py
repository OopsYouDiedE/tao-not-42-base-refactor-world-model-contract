"""启动控制台：拉起一个内核，把它的槽位挂到 HTTP 服务上。

在 WSL 中运行，`--host 0.0.0.0` 时 Windows 浏览器可以直接访问 `localhost:<port>`。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from online_interactive_environments.craftground import EnvironmentKernel

from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="CraftGround 实例控制台")
    parser.add_argument("--slots", type=int, default=2, help="CraftGround 实例数量")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 端口")
    parser.add_argument("--port-base", type=int, default=18300, help="CraftGround 起始端口")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--baseline-world", type=Path, default=None)
    parser.add_argument(
        "--socket-ipc",
        action="store_true",
        help="改用 socket IPC；无 GPU 的软件渲染环境下共享内存会阻塞在观察读取",
    )
    parser.add_argument(
        "--snapshot-radius",
        type=int,
        default=24,
        help="启动时保存根快照的水平半径；用于「重置到快照」",
    )
    arguments = parser.parse_args()

    with EnvironmentKernel.launch(
        slots=arguments.slots,
        port_base=arguments.port_base,
        baseline_world=arguments.baseline_world,
        image_width=arguments.image_width,
        image_height=arguments.image_height,
        use_shared_memory=not arguments.socket_ipc,
    ) as kernel:
        # 先存一份根快照，界面上的「重置到快照」才有目标；区域按实际玩家坐标计算。
        kernel.capture(
            "console-root",
            horizontal_radius=arguments.snapshot_radius,
            as_root=True,
        )
        print(f"控制台已就绪：http://localhost:{arguments.port}")
        uvicorn.run(create_app(kernel), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
