#!/usr/bin/env python3
"""
环境检测和自动依赖安装脚本。

功能：
  1. 检测运行环境（本机 / Colab / Godot）
  2. 检测 Python 版本（要求 >= 3.11）
  3. 自动安装系统依赖（apt-get）
  4. 推荐相应的 Python 包组合
  5. 使用 uv 进行高效安装

使用方法：
  python install_env.py                    # 交互式配置
  python install_env.py --colab            # 跳过检测，按 Colab 配置
  python install_env.py --godot            # Godot 环境
  python install_env.py --craftground      # 启用 craftground
  python install_env.py --full             # 安装所有可选依赖
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set


def is_colab() -> bool:
    """检测是否在 Google Colab 环境中运行。"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_in_venv() -> bool:
    """检测是否在虚拟环境中。"""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def check_python_version() -> bool:
    """检查 Python 版本 >= 3.11。"""
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        print(f"✅ Python {major}.{minor} OK")
        return True
    else:
        print(f"❌ Python {major}.{minor} 不满足（需要 >= 3.11）")
        return False


def check_uv_installed() -> bool:
    """检查 uv 是否已安装。"""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv():
    """安装 uv。"""
    print("📦 安装 uv...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv"],
            check=True,
        )
        print("✅ uv 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ uv 安装失败: {e}")
        return False


def install_system_deps_colab():
    """在 Colab 中安装系统依赖。"""
    print("📦 [Colab] 安装系统依赖...")
    deps = [
        "libgl1-mesa-dev",
        "libegl1-mesa-dev",
        "libglew-dev",
        "libglu1-mesa-dev",
        "xorg-dev",
        "libglfw3-dev",
        "xvfb",  # 虚拟显示
    ]
    try:
        subprocess.run(
            ["apt-get", "update"],
            check=True,
        )
        subprocess.run(
            ["apt-get", "install", "-y"] + deps,
            check=True,
        )
        print("✅ Colab 系统依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 系统依赖安装部分失败: {e}")
        return False


def install_system_deps_craftground():
    """安装 craftground 的系统依赖（Java 21）。"""
    print("📦 [Craftground] 安装系统依赖...")
    try:
        subprocess.run(
            ["apt-get", "update"],
            check=True,
        )
        subprocess.run(
            ["apt-get", "install", "-y", "openjdk-21-jdk"],
            check=True,
        )
        print("✅ Java 21 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Java 21 安装失败: {e}")
        return False


def install_system_deps_godot():
    """安装 Godot C# 的系统依赖（Mono / dotnet SDK）。"""
    print("📦 [Godot] 安装系统依赖...")
    try:
        subprocess.run(
            ["apt-get", "update"],
            check=True,
        )
        # Mono（通用 .NET 运行时）
        subprocess.run(
            ["apt-get", "install", "-y", "mono-complete"],
            check=True,
        )
        # .NET SDK（可选，用于 C# 开发）
        # subprocess.run(
        #     ["apt-get", "install", "-y", "dotnet-sdk-8.0"],
        #     check=True,
        # )
        print("✅ Mono 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Godot 系统依赖安装部分失败: {e}")
        return False


def install_python_deps(extras: List[str]) -> bool:
    """使用 uv 安装 Python 依赖。"""
    if not extras:
        print("⚠️ 未指定任何依赖组")
        return False

    extras_str = ",".join(extras)
    print(f"📦 使用 uv 安装 Python 依赖: [{extras_str}]...")

    try:
        subprocess.run(
            ["uv", "pip", "install", "-e", f".[{extras_str}]"],
            check=True,
        )
        print(f"✅ Python 依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False


def interactive_setup() -> tuple[Set[str], Set[str]]:
    """交互式询问用户的配置。"""
    sys_deps = set()
    py_extras = set()

    print("\n" + "=" * 60)
    print("🎯 环境检测和配置")
    print("=" * 60)

    is_in_colab = is_colab()
    print(f"\n📍 运行环境: {'Colab' if is_in_colab else '本机/服务器'}")

    # 询问是否在 Colab
    if not is_in_colab:
        colab_resp = input("\n你在 Colab 中运行吗？(y/n) [n]: ").strip().lower()
        if colab_resp == "y":
            is_in_colab = True

    if is_in_colab:
        sys_deps.add("colab")
        py_extras.add("colab")
        print("✅ 将安装 Colab 虚拟显示依赖")

    # 询问是否使用 Crafter
    crafter_resp = input("\n是否使用 Crafter 环境？(y/n) [y]: ").strip().lower()
    if crafter_resp != "n":
        py_extras.add("crafter")
        print("✅ 将安装 Crafter")

        # PPO+AD 或 DreamerV3
        algo_resp = (
            input("\n选择算法: (1) PPO+AD, (2) DreamerV3, (3) 两者, (4) 跳过 [1]: ")
            .strip()
            .lower()
        )
        if algo_resp in ("1", ""):
            py_extras.add("ppo-ad")
            print("✅ 将安装 PPO+AD")
        elif algo_resp == "2":
            py_extras.add("dreamer")
            print("✅ 将安装 DreamerV3")
        elif algo_resp == "3":
            py_extras.add("ppo-ad")
            py_extras.add("dreamer")
            print("✅ 将同时安装 PPO+AD 和 DreamerV3")

    # 询问是否使用 Craftground
    craftground_resp = input("\n是否使用 Craftground 环境？(y/n) [n]: ").strip().lower()
    if craftground_resp == "y":
        sys_deps.add("craftground")
        py_extras.add("craftground")
        print("✅ 将安装 Craftground（需要 Java 21）")

    # 询问是否使用 Godot
    godot_resp = input("\n是否使用 Godot 环境？(y/n) [n]: ").strip().lower()
    if godot_resp == "y":
        sys_deps.add("godot")
        py_extras.add("godot")
        print("✅ 将安装 Godot C# 支持（需要 Mono）")

    # 询问是否安装开发工具
    dev_resp = input("\n是否安装开发工具（pytest, black, mypy）？(y/n) [n]: ").strip().lower()
    if dev_resp == "y":
        py_extras.add("dev")
        print("✅ 将安装开发工具")

    return sys_deps, py_extras


def main():
    parser = argparse.ArgumentParser(
        description="环境检测和自动安装脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python install_env.py                    # 交互式配置
  python install_env.py --colab            # Colab 标准配置
  python install_env.py --colab --ppo-ad   # Colab + PPO+AD
  python install_env.py --full             # 安装全部
        """,
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="跳过检测，按 Colab 配置",
    )
    parser.add_argument(
        "--godot",
        action="store_true",
        help="启用 Godot 环境支持",
    )
    parser.add_argument(
        "--craftground",
        action="store_true",
        help="启用 Craftground 环境",
    )
    parser.add_argument(
        "--ppo-ad",
        action="store_true",
        help="安装 PPO+AD",
    )
    parser.add_argument(
        "--dreamer",
        action="store_true",
        help="安装 DreamerV3",
    )
    parser.add_argument(
        "--crafter",
        action="store_true",
        help="安装 Crafter",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="安装开发工具",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="安装所有可选依赖",
    )
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="跳过系统依赖安装（仅 apt）",
    )
    parser.add_argument(
        "--skip-python-deps",
        action="store_true",
        help="跳过 Python 依赖安装",
    )

    args = parser.parse_args()

    # ─── 检查 Python 版本 ──────────────────────────────────────
    if not check_python_version():
        sys.exit(1)

    # ─── 确定配置 ──────────────────────────────────────────────
    if args.full:
        # 全部安装
        sys_deps = {"colab", "craftground", "godot"}
        py_extras = {"crafter", "ppo-ad", "dreamer", "craftground", "godot", "dev"}
    elif any(vars(args).values()):
        # 命令行参数指定
        sys_deps = set()
        py_extras = set()

        if args.colab:
            sys_deps.add("colab")
            py_extras.add("colab")
        if args.godot:
            sys_deps.add("godot")
            py_extras.add("godot")
        if args.craftground:
            sys_deps.add("craftground")
            py_extras.add("craftground")

        if args.crafter or args.ppo_ad or args.dreamer:
            py_extras.add("crafter")
        if args.ppo_ad:
            py_extras.add("ppo-ad")
        if args.dreamer:
            py_extras.add("dreamer")
        if args.dev:
            py_extras.add("dev")
    else:
        # 交互式询问
        sys_deps, py_extras = interactive_setup()

    # ─── 检查 uv ──────────────────────────────────────────────
    if not check_uv_installed():
        print("\n⚠️ uv 未安装，正在安装...")
        if not install_uv():
            print("❌ 无法安装 uv，尝试使用 pip...")

    # ─── 安装系统依赖 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔧 安装系统依赖")
    print("=" * 60)

    if not args.skip_system_deps:
        if "colab" in sys_deps:
            install_system_deps_colab()
        if "craftground" in sys_deps:
            install_system_deps_craftground()
        if "godot" in sys_deps:
            install_system_deps_godot()
    else:
        print("⏭️ 跳过系统依赖安装")

    # ─── 安装 Python 依赖 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("🐍 安装 Python 依赖")
    print("=" * 60)

    if py_extras and not args.skip_python_deps:
        install_python_deps(sorted(py_extras))
    elif args.skip_python_deps:
        print("⏭️ 跳过 Python 依赖安装")
    else:
        print("⚠️ 未指定任何 Python 依赖，仅安装核心包...")
        install_python_deps([])

    # ─── 完成 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ 安装完成！")
    print("=" * 60)
    print(f"\n已安装模块: {', '.join(sorted(py_extras))}")
    print("\n接下来:")
    if "ppo-ad" in py_extras:
        print("  python -m train.crafter.train_ppo_ad --n-envs 16")
    if "dreamer" in py_extras:
        print("  python -m train.crafter.train_dreamerv3")
    if "craftground" in py_extras:
        print("  # Craftground 环境已就绪")


if __name__ == "__main__":
    main()
