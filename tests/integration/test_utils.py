# -*- coding: utf-8 -*-
"""tests/integration 内部共享的低层辅助(§8:仅供 tests/ 调用,不被生产依赖)。

对外接口:
    crop_square(rgb, size) — 中心正方裁剪 + 缩放到 size(HWC uint8)。
    crop128(rgb) — crop_square(128) 后转 [1,1,3,128,128] float 张量(骨干输入)。
    dump_inventory(full_obs) — 从 protobuf 观测读非空库存 [{key,count}]。
    save_png(rgb, path) — obs['rgb'] 存 PNG(兜底 CHW→HWC / 非 uint8)。
"""
import cv2
import numpy as np
import torch
from PIL import Image


def crop_square(rgb, size):
    """中心正方裁剪并缩放。rgb (H,W,3) 或 (3,H,W) → (size,size,3) uint8。"""
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    h, w = arr.shape[:2]
    s = min(h, w)
    crop = arr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def crop128(rgb):
    """crop_square(128) → [1,1,3,128,128] float32 [0,1](骨干 encode_frames 输入契约)。"""
    im = crop_square(rgb, 128)
    return torch.from_numpy(im.transpose(2, 0, 1)).float().view(1, 1, 3, 128, 128) / 255.0


def dump_inventory(full_obs):
    """从 obs['full'](protobuf ObservationSpaceMessage)读库存 translation_key+count。"""
    inv = []
    try:
        for it in full_obs.inventory:
            if getattr(it, "count", 0) > 0:
                inv.append({"key": it.translation_key, "count": it.count})
    except Exception as e:  # noqa
        inv = [{"error": repr(e)}]
    return inv


def save_png(rgb, path):
    """obs['rgb'] → PNG。RAW 编码下应为 (H,W,3) uint8;兜底 CHW→HWC 与非 uint8。"""
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    Image.fromarray(arr).save(path)
    return list(arr.shape)
