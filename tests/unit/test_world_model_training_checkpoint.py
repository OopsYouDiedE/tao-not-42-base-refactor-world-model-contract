"""验证世界模型训练 checkpoint 与轻量恢复状态同步发布。"""

from dataclasses import asdict
import json

import pytest
import torch

from net.latent_world_model import LatentWorldModelConfiguration
from net.spatiotemporal_fast_tower import SpatiotemporalFastTowerConfiguration
from train.minecraft.world_model_training import (
    CHECKPOINT_VERSION,
    _check_resume_compatibility,
    _save_checkpoint,
)


def _resume_configurations():
    tower_configuration = SpatiotemporalFastTowerConfiguration(
        visual_dim=2, text_dim=2, d=4, heads=1,
        spatial_layers=1, temporal_layers=1, grid_hw=(2, 2),
    )
    world_configuration = LatentWorldModelConfiguration(
        observation_dim=2, d=4, stochastic_variables=1,
        stochastic_classes=2, dynamics_layers=1,
    )
    return tower_configuration, world_configuration


def _resume_checkpoint(tower_configuration, world_configuration, dataset_group):
    return {
        "version": CHECKPOINT_VERSION,
        "tower_configuration": asdict(tower_configuration),
        "world_model_configuration": asdict(world_configuration),
        "dataset_group": dataset_group,
        "step": 42000,
    }


def test_resume_compatibility_same_dataset_group_is_not_transfer():
    tower_configuration, world_configuration = _resume_configurations()
    checkpoint = _resume_checkpoint(tower_configuration, world_configuration, "10xx")
    assert _check_resume_compatibility(
        checkpoint, tower_configuration, world_configuration,
        "10xx", allow_dataset_transfer=False,
    ) is False


def test_resume_compatibility_rejects_dataset_change_without_flag():
    tower_configuration, world_configuration = _resume_configurations()
    checkpoint = _resume_checkpoint(tower_configuration, world_configuration, "10xx")
    with pytest.raises(RuntimeError, match="allow-dataset-transfer"):
        _check_resume_compatibility(
            checkpoint, tower_configuration, world_configuration,
            "7xx", allow_dataset_transfer=False,
        )


def test_resume_compatibility_allows_dataset_change_with_flag():
    tower_configuration, world_configuration = _resume_configurations()
    checkpoint = _resume_checkpoint(tower_configuration, world_configuration, "10xx")
    assert _check_resume_compatibility(
        checkpoint, tower_configuration, world_configuration,
        "7xx", allow_dataset_transfer=True,
    ) is True


def test_resume_compatibility_keeps_architecture_strict_under_transfer():
    tower_configuration, world_configuration = _resume_configurations()
    checkpoint = _resume_checkpoint(tower_configuration, world_configuration, "10xx")
    checkpoint["tower_configuration"]["d"] = 8
    with pytest.raises(RuntimeError, match="快塔配置"):
        _check_resume_compatibility(
            checkpoint, tower_configuration, world_configuration,
            "7xx", allow_dataset_transfer=True,
        )


def test_checkpoint_publishes_matching_json_sidecar(tmp_path):
    tower = torch.nn.Linear(2, 2)
    world_model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        list(tower.parameters()) + list(world_model.parameters()),
    )
    checkpoint_path = tmp_path / "last.pt"

    _save_checkpoint(
        checkpoint_path,
        tower,
        world_model,
        SpatiotemporalFastTowerConfiguration(
            visual_dim=2, text_dim=2, d=4, heads=1,
            spatial_layers=1, temporal_layers=1, grid_hw=(2, 2),
        ),
        LatentWorldModelConfiguration(
            observation_dim=2, d=4, stochastic_variables=1,
            stochastic_classes=2, dynamics_layers=1,
        ),
        optimizer,
        step=17,
        dataset_group="10xx",
        image_shards=("part-0",),
    )

    metadata = json.loads((tmp_path / "last.json").read_text(encoding="utf-8"))
    assert checkpoint_path.is_file()
    assert metadata["version"] == CHECKPOINT_VERSION
    assert metadata["step"] == 17
    assert metadata["dataset_group"] == "10xx"
    assert metadata["image_shards"] == ["part-0"]
    assert metadata["checkpoint_size"] == checkpoint_path.stat().st_size
    assert metadata["checkpoint_modified_ns"] == checkpoint_path.stat().st_mtime_ns
