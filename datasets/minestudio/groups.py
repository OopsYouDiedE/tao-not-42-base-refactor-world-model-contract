"""映射 MineStudio 数据范围、仓库和训练任务文本。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MineStudioDatasetGroup:
    """一个可以完整下载并无限训练的数据范围。"""

    dataset_group: str
    repository_id: str
    task_text: str


MINESTUDIO_DATASET_GROUPS = (
    MineStudioDatasetGroup(
        dataset_group="7xx",
        repository_id="CraftJarvis/minestudio-data-7xx-v110",
        task_text="survive from a fresh Minecraft world and make early-game progress",
    ),
    MineStudioDatasetGroup(
        dataset_group="9xx",
        repository_id="CraftJarvis/minestudio-data-9xx-v110",
        task_text="build a useful house with the available materials",
    ),
    MineStudioDatasetGroup(
        dataset_group="10xx",
        repository_id="CraftJarvis/minestudio-data-10xx-v110",
        task_text="obtain a diamond pickaxe through the Minecraft technology tree",
    ),
)


def get_dataset_group(dataset_group: str) -> MineStudioDatasetGroup:
    """按 MineStudio 数据范围（例如 ``10xx``）返回训练配置。"""
    for configuration in MINESTUDIO_DATASET_GROUPS:
        if configuration.dataset_group == dataset_group:
            return configuration
    choices = ", ".join(
        configuration.dataset_group for configuration in MINESTUDIO_DATASET_GROUPS
    )
    raise ValueError(f"未知数据范围 {dataset_group!r}；可选值: {choices}")
