"""验证 Minecraft BC 像素快塔的前向契约。"""

import torch

from net.pixel_tower import PixelTowerConfig, build_pixel_tower


def test_pixel_tower_forward_shapes():
    """验证相机分类头和按键头的 Shape 与 Dtype。

    Returns
    -------
    None
        前向输出满足契约时无返回值。
    """
    cfg = PixelTowerConfig(
        img_hw=(32, 32), d=32, heads=2, layers=1, goal_dim=16,
        max_len=8, frame_stack=4,
    )
    model = build_pixel_tower(cfg)
    image = torch.zeros(2, 3, 12, 32, 32, dtype=torch.float32)
    goal = torch.zeros(2, 16, dtype=torch.float32)
    previous_action = torch.zeros(2, 3, 22, dtype=torch.float32)

    camera, keys = model(image, goal, previous_action)

    assert camera.shape == (2, 3, 1, 2, 11)
    assert keys.shape == (2, 3, 1, 20)
    assert camera.dtype == torch.float32
    assert keys.dtype == torch.float32
