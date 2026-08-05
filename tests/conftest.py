"""真实 CraftGround 测试的收集策略。

项目不接受环境替身：任何触碰 CraftGround 的测试都必须连真实 JVM。这类测试标记为
`craftground`，只有显式 `--craftground` 时才收集，从而让无 JVM 的机器上 `pytest`
仍然只跑纯逻辑测试，而不是静默地用假环境冒充通过。
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--craftground",
        action="store_true",
        default=False,
        help="收集需要真实 CraftGround JVM 的测试（需要 JDK 21 与 OpenGL）",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "craftground: 需要真实 CraftGround JVM；默认跳过，使用 --craftground 收集",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--craftground"):
        return
    skip = pytest.mark.skip(reason="需要真实 CraftGround JVM，使用 --craftground 运行")
    for item in items:
        if "craftground" in item.keywords:
            item.add_marker(skip)
