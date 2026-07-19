"""验证重新设计的时空快塔契约。"""

import torch

from net.spatiotemporal_fast_tower import (
    NullMemory,
    SpatiotemporalFastTowerConfiguration,
    build_spatiotemporal_fast_tower,
)


def test_spatiotemporal_fast_tower_shapes_and_legacy_keys():
    """验证时空输入、结构化动作和 20 键展开。"""
    configuration = SpatiotemporalFastTowerConfiguration(
        visual_dim=24, text_dim=12, d=32, heads=4,
        spatial_layers=1, temporal_layers=1, grid_hw=(4, 6),
        max_history=4, max_text_tokens=8, action_horizon=3,
    )
    model = build_spatiotemporal_fast_tower(configuration)
    output, state = model.forward_with_state(
        current_patches=torch.zeros(2, 24, 24),
        history_patches=torch.zeros(2, 2, 24, 24),
        text_tokens=torch.zeros(2, 5, 12),
        text_mask=torch.ones(2, 5, dtype=torch.bool),
        past_actions=torch.zeros(2, 3, 22),
        dt=torch.full((2, 3, 1), 0.05),
    )
    assert output.camera_logits.shape == (2, 3, 2, 11)
    assert output.move_fb_logits.shape == (2, 3, 3)
    assert output.move_lr_logits.shape == (2, 3, 3)
    assert output.stance_logits.shape == (2, 3, 3)
    assert output.hotbar_logits.shape == (2, 3, 10)
    assert output.button_logits.shape == (2, 3, 5)
    assert state.shape == (2, 32)
    assert output.legacy_key_probabilities().shape == (2, 3, 20)
    camera, keys = output.sample_legacy(deterministic=True)
    assert camera.shape == (2, 3, 2)
    assert keys.shape == (2, 3, 20)
    assert not torch.any((keys[..., 0] > 0) & (keys[..., 1] > 0))
    assert not torch.any((keys[..., 2] > 0) & (keys[..., 3] > 0))
    assert torch.all(keys[..., 11:20].sum(dim=-1) <= 1)


def test_null_memory_is_empty():
    """验证默认记忆不注入 token。"""
    memory = NullMemory(32)
    assert memory(3, torch.device("cpu")).shape == (3, 0, 32)


def test_default_trainable_parameter_budget():
    """验证默认快塔核心处于约 100M 可训练参数预算。"""
    with torch.device("meta"):
        model = build_spatiotemporal_fast_tower(SpatiotemporalFastTowerConfiguration())
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert 90_000_000 <= count <= 120_000_000
