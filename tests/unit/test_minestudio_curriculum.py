"""验证 MineStudio 分阶段课程、分片选择和模型规模标尺。"""

import torch

from datasets.vpt.minestudio_curriculum import (
    curriculum_stage_names,
    estimate_main_curriculum_model_scale,
    get_curriculum_stage,
)
from datasets.vpt.minestudio_download import (
    image_shards_from_repository_files,
    local_complete_image_shards,
    prune_completed_stage,
    prune_other_image_shards,
    prune_unselected_image_shards,
    select_stage_modalities,
    select_stage_shard,
)
from net.latent_world_model import (
    LatentWorldModelConfiguration,
    build_latent_world_model,
)
from net.spatiotemporal_fast_tower import (
    SpatiotemporalFastTowerConfiguration,
    build_spatiotemporal_fast_tower,
)
from train.minecraft.autodl_curriculum import (
    build_curriculum_schedule,
    distribute_updates,
)


def test_main_curriculum_uses_three_complementary_groups():
    assert curriculum_stage_names() == (
        "foundation", "construction", "long_horizon",
    )
    assert get_curriculum_stage("foundation").dataset_group == "7xx"
    assert get_curriculum_stage("construction").dataset_group == "9xx"
    assert get_curriculum_stage("long_horizon").dataset_group == "10xx"


def test_model_scale_matches_combined_trainable_core():
    estimate = estimate_main_curriculum_model_scale()
    assert 1_400 <= estimate.estimated_hours <= 1_450
    assert 100_000_000 <= estimate.estimated_frames <= 105_000_000
    assert 195_000_000 <= estimate.recommended_parameters <= 215_000_000
    with torch.device("meta"):
        tower = build_spatiotemporal_fast_tower(SpatiotemporalFastTowerConfiguration())
        world_model = build_latent_world_model(LatentWorldModelConfiguration())
    actual_parameters = sum(
        parameter.numel()
        for model in (tower, world_model)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert 200_000_000 <= actual_parameters <= 220_000_000


def test_shard_selection_keeps_all_actions_and_one_image_database():
    files = [
        "action/part-3/data.mdb",
        "action/part-3/lock.mdb",
        "image/part-11/data.mdb",
        "image/part-11/lock.mdb",
        "image/part-2/data.mdb",
        "image/part-2/lock.mdb",
        "meta_info/part-1/data.mdb",
    ]
    selection = select_stage_shard(files, image_shard_index=1)
    assert selection.image_shard == "image/part-11"
    assert selection.image_shard_count == 2
    assert selection.allow_patterns == (
        "action/**", "meta_info/**", "image/part-11/**",
    )


def test_repository_image_shards_have_numeric_stable_order():
    files = [
        "image/part-11/data.mdb",
        "image/part-2/data.mdb",
        "image/part-1/lock.mdb",
    ]
    assert image_shards_from_repository_files(files) == (
        "image/part-2", "image/part-11",
    )


def test_modality_selection_supports_full_metadata_and_multiple_images():
    files = [
        "action/part-3/data.mdb",
        "meta_info/part-8/data.mdb",
        "image/part-2/data.mdb",
        "image/part-11/data.mdb",
        "event/data.mdb",
        "motion/data.mdb",
        "segmentation/part-2/data.mdb",
    ]
    selection = select_stage_modalities(
        files,
        modalities=("image", "action", "meta_info"),
        image_shard_indices=(0, 1),
    )
    assert selection.modalities == ("action", "meta_info", "image")
    assert selection.image_shards == ("image/part-2", "image/part-11")
    assert selection.allow_patterns == (
        "action/**", "meta_info/**", "image/part-2/**", "image/part-11/**",
    )

    metadata_only = select_stage_modalities(
        files,
        modalities=("action", "meta_info"),
    )
    assert metadata_only.image_shards == ()
    assert metadata_only.allow_patterns == ("action/**", "meta_info/**")


def test_autodl_schedule_covers_every_shard_and_has_exact_targets():
    allocations = distribute_updates(10, 3)
    assert allocations == (4, 3, 3)
    schedule = build_curriculum_schedule(
        stages=("foundation", "construction"),
        stage_image_shards={
            "foundation": ("image/part-0", "image/part-1"),
            "construction": ("image/part-0",),
        },
        stage_updates={"foundation": 7, "construction": 5},
        stage_learning_rates={"foundation": 1e-4, "construction": 7e-5},
    )
    assert [entry.updates for entry in schedule] == [4, 3, 5]
    assert [entry.target_step for entry in schedule] == [4, 7, 12]
    assert [entry.stage for entry in schedule] == [
        "foundation", "foundation", "construction",
    ]


def test_pruning_only_removes_complete_unselected_image_databases(tmp_path):
    image_root = tmp_path / "image"
    for name in ("part-1", "part-2"):
        directory = image_root / name
        directory.mkdir(parents=True)
        (directory / "data.mdb").write_bytes(b"database")
    partial = image_root / "part-downloading"
    partial.mkdir()
    (partial / "data.mdb.part").write_bytes(b"partial")

    removed = prune_other_image_shards(tmp_path, "image/part-2")

    assert removed == ["part-1"]
    assert not (image_root / "part-1").exists()
    assert (image_root / "part-2" / "data.mdb").is_file()
    assert (partial / "data.mdb.part").is_file()


def test_pruning_can_keep_multiple_prefetched_image_shards(tmp_path):
    image_root = tmp_path / "image"
    for name in ("part-1", "part-2", "part-3"):
        directory = image_root / name
        directory.mkdir(parents=True)
        (directory / "data.mdb").write_bytes(b"database")

    removed = prune_unselected_image_shards(
        tmp_path, ("image/part-1", "image/part-3"),
    )

    assert removed == ["part-2"]
    assert (image_root / "part-1").is_dir()
    assert (image_root / "part-3").is_dir()


def test_local_complete_image_shards_ignore_partial_downloads(tmp_path):
    image_root = tmp_path / "7xx" / "image"
    for name in ("part-11", "part-2"):
        directory = image_root / name
        directory.mkdir(parents=True)
        (directory / "data.mdb").write_bytes(b"database")
    partial = image_root / "part-3"
    partial.mkdir()
    (partial / "data.mdb.part").write_bytes(b"partial")

    assert local_complete_image_shards(tmp_path, "foundation") == (
        "image/part-2", "image/part-11",
    )


def test_completed_stage_pruning_stays_inside_data_root(tmp_path):
    completed = tmp_path / "7xx"
    completed.mkdir()
    (completed / "marker").write_text("data", encoding="utf-8")
    untouched = tmp_path / "9xx"
    untouched.mkdir()

    assert prune_completed_stage(tmp_path, "foundation") is True
    assert not completed.exists()
    assert untouched.is_dir()
    assert prune_completed_stage(tmp_path, "foundation") is False
