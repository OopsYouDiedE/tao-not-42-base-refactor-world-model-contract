"""验证结构化动作监督不会奖励互斥冲突。"""

import torch

from net.spatiotemporal_fast_tower import StructuredActionOutput
from train.minecraft.action_supervision import encode_targets, structured_action_loss


def test_structured_action_loss_is_finite_for_conflicting_raw_keys():
    """原始数据偶发前后同按时屏蔽对应组，其他动作仍参与训练。"""
    raw = torch.zeros(2, 3, 22)
    raw[..., 2] = 1.0
    raw[..., 4] = 1.0
    camera, keys, canonical = encode_targets(raw)
    output = StructuredActionOutput(
        camera_logits=torch.zeros(2, 3, 2, 11),
        move_fb_logits=torch.zeros(2, 3, 3),
        move_lr_logits=torch.zeros(2, 3, 3),
        stance_logits=torch.zeros(2, 3, 3),
        hotbar_logits=torch.zeros(2, 3, 10),
        button_logits=torch.zeros(2, 3, 5),
    )
    loss = structured_action_loss(output, camera, keys)
    assert canonical.shape == (2, 3, 22)
    assert torch.isfinite(loss)
