"""验证 Dreamer-lite 潜状态模型的数值与 Shape 契约。"""

import torch

from net.latent_world_model import (
    LatentWorldModelConfiguration,
    balanced_categorical_kl_loss,
    build_latent_world_model,
)


def test_world_model_observe_and_imagine_shapes():
    """真实观测初始化后可以执行动作条件化想象并校正后验。"""
    configuration = LatentWorldModelConfiguration(
        observation_dim=12, action_dim=6, d=32,
        stochastic_variables=4, stochastic_classes=5, dynamics_layers=1,
        event_dim=3, inventory_dim=7,
    )
    model = build_latent_world_model(configuration)
    state, posterior = model.initialize(torch.zeros(2, 12))
    prediction = model.imagine(state, torch.zeros(2, 6), torch.full((2, 1), 0.05))
    corrected, next_posterior = model.observe(
        prediction.next_state, torch.zeros(2, 12),
    )

    assert posterior.shape == (2, 4, 5)
    assert prediction.prior_logits.shape == (2, 4, 5)
    assert prediction.observation.shape == (2, 12)
    assert prediction.reward.shape == (2,)
    assert prediction.continuation_logits.shape == (2,)
    assert prediction.event_logits.shape == (2, 3)
    assert prediction.inventory_delta.shape == (2, 7)
    assert corrected.deterministic.shape == (2, 32)
    assert next_posterior.shape == (2, 4, 5)
    loss = balanced_categorical_kl_loss(next_posterior, prediction.prior_logits)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_world_model_rejects_implicit_time_shape():
    """时间跨度必须显式保留末维，避免帧数与秒静默广播。"""
    model = build_latent_world_model(LatentWorldModelConfiguration(
        observation_dim=4, action_dim=3, d=8,
        stochastic_variables=2, stochastic_classes=2, dynamics_layers=1,
    ))
    state, _ = model.initialize(torch.zeros(1, 4))
    try:
        model.imagine(state, torch.zeros(1, 3), torch.zeros(1))
    except ValueError as error:
        assert "dt" in str(error)
    else:
        raise AssertionError("一维 dt 应被拒绝")
