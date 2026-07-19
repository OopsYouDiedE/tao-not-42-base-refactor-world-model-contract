"""验证世界模型训练 checkpoint 与轻量恢复状态同步发布。"""

import json

import torch

from net.latent_world_model import LatentWorldModelConfiguration
from net.spatiotemporal_fast_tower import SpatiotemporalFastTowerConfiguration
from train.minecraft.world_model_warm_start import (
    CHECKPOINT_VERSION,
    _save_checkpoint,
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
        curriculum_stage="foundation",
        image_shards=("part-0",),
    )

    metadata = json.loads((tmp_path / "last.json").read_text(encoding="utf-8"))
    assert checkpoint_path.is_file()
    assert metadata["version"] == CHECKPOINT_VERSION
    assert metadata["step"] == 17
    assert metadata["curriculum_stage"] == "foundation"
    assert metadata["image_shards"] == ["part-0"]
    assert metadata["checkpoint_size"] == checkpoint_path.stat().st_size
    assert metadata["checkpoint_modified_ns"] == checkpoint_path.stat().st_mtime_ns
