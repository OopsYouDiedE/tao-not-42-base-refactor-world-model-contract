"""定义 CraftJarvis MineStudio 数据课程及其规模估算接口。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MineStudioCurriculumStage:
    """一个可以独立下载、训练和续训的数据课程阶段。"""

    name: str
    dataset_group: str
    repository_id: str
    task_text: str
    purpose: str
    image_bytes: int
    action_bytes: int
    metadata_bytes: int


MAIN_CURRICULUM = (
    MineStudioCurriculumStage(
        name="foundation",
        dataset_group="7xx",
        repository_id="CraftJarvis/minestudio-data-7xx-v110",
        task_text="survive from a fresh Minecraft world and make early-game progress",
        purpose="通用移动、采集、合成与早期生存",
        image_bytes=368_636_000_000,
        action_bytes=17_660_000_000,
        metadata_bytes=46_066_000_000,
    ),
    MineStudioCurriculumStage(
        name="construction",
        dataset_group="9xx",
        repository_id="CraftJarvis/minestudio-data-9xx-v110",
        task_text="build a useful house with the available materials",
        purpose="方块放置、hotbar 使用与局部空间控制",
        image_bytes=178_950_000_000,
        action_bytes=8_049_000_000,
        metadata_bytes=19_617_000_000,
    ),
    MineStudioCurriculumStage(
        name="long_horizon",
        dataset_group="10xx",
        repository_id="CraftJarvis/minestudio-data-10xx-v110",
        task_text="obtain a diamond pickaxe through the Minecraft technology tree",
        purpose="长程采集、工具升级与技术树顺序",
        image_bytes=94_908_000_000,
        action_bytes=4_832_000_000,
        metadata_bytes=9_614_000_000,
    ),
)

_STAGES = {stage.name: stage for stage in MAIN_CURRICULUM}

# OpenAI VPT 报告承包商数据约 2,000 小时。MineStudio 五组图像 LMDB 的总大小
# 约 901.642 GB；这里按图像字节比例估算主课程所覆盖的小时数。磁盘压缩率并非
# 完全恒定，所以结果只用于选参数档位，实际训练覆盖率以 LMDB 帧计数为准。
ALL_GROUP_IMAGE_BYTES = 901_642_000_000
CONTRACTOR_HOURS = 2_000.0
FRAMES_PER_SECOND = 20.0
EFFECTIVE_TOKENS_PER_FRAME = 40.0
TOKENS_PER_PARAMETER = 20.0


@dataclass(frozen=True)
class ModelScaleEstimate:
    """由课程数据量得到的多模态 Chinchilla 启发式估算。"""

    estimated_hours: float
    estimated_frames: int
    effective_tokens: int
    recommended_parameters: int


def get_curriculum_stage(name: str) -> MineStudioCurriculumStage:
    """按公开名称返回课程阶段。"""
    try:
        return _STAGES[name]
    except KeyError as error:
        choices = ", ".join(_STAGES)
        raise ValueError(f"未知课程阶段 {name!r}；可选值: {choices}") from error


def estimate_main_curriculum_model_scale() -> ModelScaleEstimate:
    """估算三阶段课程适配的可训练参数量。

    这里不把 576 个高度相关的 DINO patch 当成 576 个独立文本 token，而把每帧
    折算为 40 个有效多模态 token，再采用 ``有效 token ≈ 20 × 参数`` 的数据侧
    标尺。该结果用于选择 100M/210M/320M 档位，不是视觉模型的普适缩放定律。
    """
    selected_image_bytes = sum(stage.image_bytes for stage in MAIN_CURRICULUM)
    estimated_hours = CONTRACTOR_HOURS * selected_image_bytes / max(
        ALL_GROUP_IMAGE_BYTES, 1,
    )
    estimated_frames = round(
        estimated_hours * 3600.0 * FRAMES_PER_SECOND,
    )
    effective_tokens = round(estimated_frames * EFFECTIVE_TOKENS_PER_FRAME)
    recommended_parameters = round(
        effective_tokens / max(TOKENS_PER_PARAMETER, 1e-4),
    )
    return ModelScaleEstimate(
        estimated_hours=estimated_hours,
        estimated_frames=estimated_frames,
        effective_tokens=effective_tokens,
        recommended_parameters=recommended_parameters,
    )


def curriculum_stage_names() -> tuple[str, ...]:
    """返回主课程的稳定命令行顺序。"""
    return tuple(stage.name for stage in MAIN_CURRICULUM)
