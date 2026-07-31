"""保持宽高比的视觉尺寸与 Gemma 4 patch 预算合同。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionGeometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale: float
    raw_patches: int
    soft_tokens: int


def plan_gemma4_geometry(width: int, height: int, scale: float = 1.0) -> VisionGeometry:
    """缩放后对齐 48 像素；16 patch 经 3x3 pooling 得到 soft token。"""
    if width <= 0 or height <= 0 or scale <= 0:
        raise ValueError("尺寸和 scale 必须大于零")
    resized_width = max(48, round(width * scale / 48) * 48)
    resized_height = max(48, round(height * scale / 48) * 48)
    raw = (resized_width // 16) * (resized_height // 16)
    soft = (resized_width // 48) * (resized_height // 48)
    if raw > 2520 or soft > 280:
        raise ValueError(f"Gemma 4 视觉预算超限：raw_patches={raw}, soft_tokens={soft}")
    return VisionGeometry(width, height, resized_width, resized_height, scale, raw, soft)


def camera_degrees_after_resize(pitch_degrees: float, yaw_degrees: float) -> tuple[float, float]:
    """Minecraft 相机标签是世界角度，图像 resize 不改变其数值。"""
    return float(pitch_degrees), float(yaw_degrees)
