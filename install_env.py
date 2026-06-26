#!/usr/bin/env python3
"""
智能依赖安装脚本 — 自动适配平台和环境。

核心原则：
  - 用户指定「功能模块」（ppo-ad, dreamer, godot）
  - 脚本自动检测「运行平台」（Colab, 本机, 服务器）
  - 根据模块 + 平台组合，自动装相应的系统依赖和 Python 包

使用方法：
  python install_env.py                  # 交互式
  python install_env.py --ppo-ad         # 指定 PPO+AD（自动适配平台）
  python install_env.py --dreamer        # DreamerV3
  python install_env.py --godot          # Godot RL
  python install_env.py --craftground    # Craftground
  python install_env.py --ppo-ad --dev   # 组合多个模块
  python install_env.py --full           # 全部
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple


# ────────────────────────────────────────────────────────────────
# 平台检测
# ────────────────────────────────────────────────────────────────


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


def is_headless() -> bool:
    """检测是否是无图形界面环境（Colab 或 SSH）。"""
    if is_colab():
        return True
    # 可以通过其他方式检测 SSH / 服务器环境
    # 例如检查 DISPLAY 环境变量
    import os
    return os.environ.get("DISPLAY") is None




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


# ────────────────────────────────────────────────────────────────
# 系统依赖安装（按模块）
# ────────────────────────────────────────────────────────────────


def install_system_deps_for_headless():
    """为 Headless 环境安装虚拟显示依赖（Colab）。"""
    print("📦 [Headless] 安装虚拟显示依赖...")
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
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + deps, check=True)
        print("✅ 虚拟显示依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 虚拟显示依赖安装部分失败: {e}")
        return False


def install_system_deps_for_craftground():
    """为 Craftground 安装 Java 21（apt-get + sudo）。"""
    print("📦 [Craftground] 安装 Java 21...")
    print("   使用 apt-get 安装 OpenJDK 21（需要 sudo，请输入密码）...")
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "openjdk-21-jdk"], check=True)
        print("✅ Java 21 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Java 21 安装失败: {e}")
        print("   手动安装: sudo apt-get install openjdk-21-jdk")
        return False


def install_system_deps_for_godot():
    """为 Godot 安装 Mono（C# 运行时，apt-get + sudo）。"""
    print("📦 [Godot] 安装 Mono...")
    print("   使用 apt-get 安装 Mono（需要 sudo，请输入密码）...")
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "mono-complete"], check=True)
        print("✅ Mono 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Mono 安装部分失败: {e}")
        print("   手动安装: sudo apt-get install mono-complete")
        return False


# ────────────────────────────────────────────────────────────────
# 依赖组合和 Python 包安装
# ────────────────────────────────────────────────────────────────


def resolve_extras(
    modules: Set[str], platform: str
) -> Tuple[Set[str], Set[str]]:
    """
    根据指定的功能模块和平台，自动解析 Python 和系统依赖。

    Args:
        modules: 用户指定的模块集合（ppo-ad, dreamer, godot 等）
        platform: 检测到的平台（colab, headless, godot, local）

    Returns:
        (py_extras, sys_deps) — Python extras 和系统依赖
    """
    py_extras = set()
    sys_deps = set()

    # ──────────────────────────────────────────────────────────
    # Python 依赖逻辑
    # ──────────────────────────────────────────────────────────

    # PPO+AD
    if "ppo-ad" in modules:
        py_extras.add("ppo-ad")
        print("✅ 将安装 PPO+AD")

    # DreamerV3
    if "dreamer" in modules:
        py_extras.add("dreamer")
        print("✅ 将安装 DreamerV3")

    # Craftground（待确认包名和依赖）
    if "craftground" in modules:
        py_extras.add("craftground")
        sys_deps.add("craftground")  # 需要 Java 21
        print("✅ 将安装 Craftground（需要 Java 21）")
        print("   注意：craftground 包的可用性待确认，可能需要手动安装或自编译")

    # Minecraft
    if "minecraft" in modules:
        py_extras.add("minecraft")
        print("✅ 将安装 Minecraft（VPT 数据处理）")

    # RL 工具
    if "rl" in modules:
        py_extras.add("rl")
        print("✅ 将安装 RL 工具集")

    # Godot（仅装 Mono，Python binding 待确认）
    if "godot" in modules:
        py_extras.add("godot")
        sys_deps.add("godot")  # 需要 Mono（C# 运行时）
        print("✅ 将安装 Godot 环境（Mono C# 支持）")
        print("   注意：Python binding（godot-python）包名待确认，当前仅装 Mono 系统依赖")

    # 开发工具
    if "dev" in modules:
        py_extras.add("dev")
        print("✅ 将安装开发工具")

    # Crafter（如果指定了依赖 crafter 的模块，自动添加）
    if any(m in modules for m in ["ppo-ad", "dreamer"]) and "crafter" not in py_extras:
        py_extras.add("crafter")

    # ──────────────────────────────────────────────────────────
    # 平台特定的依赖
    # ──────────────────────────────────────────────────────────

    if platform == "colab":
        sys_deps.add("headless")  # 虚拟显示
        py_extras.add("headless")
        print("✅ Colab 环境：自动添加虚拟显示支持")

    elif platform == "headless":
        # 服务器或无显示的环境（但不是 Colab）
        # 可选：询问用户是否需要虚拟显示
        pass

    return py_extras, sys_deps


def install_python_deps(extras: List[str]) -> bool:
    """使用 uv 安装 Python 依赖。"""
    if not extras:
        print("⚠️ 未指定任何 Python 依赖")
        return False

    extras_str = ",".join(sorted(extras))
    print(f"📦 使用 uv 安装 Python 依赖: [{extras_str}]...")

    try:
        subprocess.run(
            ["uv", "pip", "install", "-e", f".[{extras_str}]"],
            check=True,
        )
        print("✅ Python 依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False


# ────────────────────────────────────────────────────────────────
# 交互式配置
# ────────────────────────────────────────────────────────────────


def interactive_setup() -> Set[str]:
    """交互式询问用户要安装的模块。"""
    modules = set()

    print("\n" + "=" * 60)
    print("🎯 模块选择")
    print("=" * 60)

    # 检测平台
    if is_colab():
        print("\n📍 环境: Google Colab（自动检测）")
    elif is_headless():
        print("\n📍 环境: Headless / 服务器")
    else:
        print("\n📍 环境: 本地桌面环境")

    # Crafter 相关模块
    crafter_resp = input("\n是否使用 Crafter 环境？(y/n) [y]: ").strip().lower()
    if crafter_resp != "n":
        algo_resp = (
            input(
                "\n选择算法:\n"
                "  (1) PPO+AD\n"
                "  (2) DreamerV3\n"
                "  (3) 两者都装\n"
                "  (4) 跳过\n"
                "[1]: "
            )
            .strip()
            .lower()
        )
        if algo_resp in ("1", ""):
            modules.add("ppo-ad")
        elif algo_resp == "2":
            modules.add("dreamer")
        elif algo_resp == "3":
            modules.add("ppo-ad")
            modules.add("dreamer")

    # 其他环境
    if (
        input("\n是否使用 Craftground 游戏环境？(y/n) [n]: ").strip().lower()
        == "y"
    ):
        modules.add("craftground")

    if (
        input("\n是否使用 Minecraft / VPT 数据处理？(y/n) [n]: ").strip().lower()
        == "y"
    ):
        modules.add("minecraft")

    if input("\n是否使用 Godot RL 环境？(y/n) [n]: ").strip().lower() == "y":
        modules.add("godot")

    if input("\n是否使用通用 RL 工具（gymnasium, envpool）？(y/n) [n]: ").strip().lower() == "y":
        modules.add("rl")

    # 开发工具
    if (
        input(
            "\n是否安装开发工具（pytest, black, mypy）？(y/n) [n]: "
        )
        .strip()
        .lower()
        == "y"
    ):
        modules.add("dev")

    return modules


# ────────────────────────────────────────────────────────────────
# 主程序
# ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="智能依赖安装脚本 — 自动适配平台和环境",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python install_env.py                  # 交互式配置
  python install_env.py --ppo-ad         # 安装 PPO+AD（自动适配平台）
  python install_env.py --ppo-ad --dev   # 组合安装
  python install_env.py --full           # 安装全部

支持的模块：
  --ppo-ad          PPO + Achievement Distillation（需要 Crafter）
  --dreamer         DreamerV3 世界模型（需要 Crafter）
  --craftground     Craftground 游戏环境
  --minecraft       Minecraft VPT 数据处理
  --godot           Godot RL 环境支持
  --rl              通用 RL 工具（gymnasium, envpool）
  --dev             开发工具（pytest, black, mypy）
  --full            全部（包括所有可选）
        """,
    )

    # 功能模块选项（不涉及平台）
    parser.add_argument("--ppo-ad", action="store_true", help="安装 PPO+AD")
    parser.add_argument("--dreamer", action="store_true", help="安装 DreamerV3")
    parser.add_argument("--craftground", action="store_true", help="安装 Craftground")
    parser.add_argument("--minecraft", action="store_true", help="安装 Minecraft")
    parser.add_argument("--godot", action="store_true", help="安装 Godot RL")
    parser.add_argument("--rl", action="store_true", help="安装 RL 工具")
    parser.add_argument("--dev", action="store_true", help="安装开发工具")
    parser.add_argument("--full", action="store_true", help="安装全部")

    # 控制选项
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="跳过系统依赖安装",
    )
    parser.add_argument(
        "--skip-python-deps",
        action="store_true",
        help="跳过 Python 依赖安装",
    )

    args = parser.parse_args()

    # ─── 检查 Python 版本 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 环境检查")
    print("=" * 60 + "\n")

    if not check_python_version():
        sys.exit(1)

    # ─── 确定运行平台 ──────────────────────────────────────────
    is_in_colab = is_colab()
    is_in_headless = is_headless() and not is_in_colab

    if is_in_colab:
        platform = "colab"
        print("📍 环境: Google Colab")
    elif is_in_headless:
        platform = "headless"
        print("📍 环境: Headless / 服务器")
    else:
        platform = "local"
        print("📍 环境: 本地桌面")

    # ─── 确定要安装的模块 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("📦 模块选择")
    print("=" * 60)

    if args.full:
        modules = {"ppo-ad", "dreamer", "craftground", "minecraft", "godot", "rl", "dev"}
        print("\n🚀 全部安装模式")
    elif any(vars(args).get(m) for m in ["ppo_ad", "dreamer", "craftground", "minecraft", "godot", "rl", "dev"]):
        # 命令行参数指定
        modules = set()
        if args.ppo_ad:
            modules.add("ppo-ad")
        if args.dreamer:
            modules.add("dreamer")
        if args.craftground:
            modules.add("craftground")
        if args.minecraft:
            modules.add("minecraft")
        if args.godot:
            modules.add("godot")
        if args.rl:
            modules.add("rl")
        if args.dev:
            modules.add("dev")
        print(f"\n✅ 模块: {', '.join(sorted(modules))}")
    else:
        # 交互式询问
        modules = interactive_setup()

    # ─── 解析依赖 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔧 依赖解析")
    print("=" * 60 + "\n")

    py_extras, sys_deps = resolve_extras(modules, platform)

    if not py_extras and not sys_deps:
        print("⚠️ 未指定任何模块")
        return

    # ─── 检查 uv ──────────────────────────────────────────────
    if not check_uv_installed():
        print("\n⚠️ uv 未安装，正在安装...")
        if not install_uv():
            print("❌ 无法自动安装 uv，请手动运行: pip install uv")
            return

    # ─── 安装系统依赖 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔨 系统依赖安装")
    print("=" * 60)

    if not args.skip_system_deps:
        if "headless" in sys_deps:
            install_system_deps_for_headless()
        if "craftground" in sys_deps:
            install_system_deps_for_craftground()
        if "godot" in sys_deps:
            install_system_deps_for_godot()
    else:
        print("⏭️ 跳过系统依赖安装")

    # ─── 安装 Python 依赖 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("🐍 Python 依赖安装")
    print("=" * 60 + "\n")

    if not args.skip_python_deps:
        install_python_deps(sorted(py_extras))
    else:
        print("⏭️ 跳过 Python 依赖安装")

    # ─── 完成 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ 安装完成！")
    print("=" * 60)
    print(f"\n🎯 已安装模块: {', '.join(sorted(modules))}")
    print(f"📍 平台: {platform}")

    print("\n📝 接下来:")
    if "ppo-ad" in modules:
        print("  python -m train.crafter.train_ppo_ad --n-envs 16 --total-timesteps 3000000")
    if "dreamer" in modules:
        print("  python -m train.crafter.train_dreamerv3")
    if "godot" in modules:
        print("  # 打开 Godot 编辑器进行 RL 实验...")
    if "craftground" in modules:
        print("  # Craftground 环境已就绪")

    print("\n🔗 查看完整文档: INSTALL.md")


if __name__ == "__main__":
    main()
