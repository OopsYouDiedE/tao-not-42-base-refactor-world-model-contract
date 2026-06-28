"""YOLO26s 作为 Craftground 世界模型的 Backbone 编码器。

设计：
  - YOLOv8-s 的 Detect 头从第 15/18/21 层取 P3/P4/P5 多尺度特征
  - 用 forward hook 抓这三层的**真实**特征图（128/256/512 通道）
  - 通过 MultiScaleFusionHead 融合为固定大小的状态表征 (output_dim)
  - 兼容下游 PPO / RSSM 的输入

关键修复（相对旧版）：
  1. 用 forward hook 取真实特征，删除 torch.randn 噪声兜底——失败必须报错，
     绝不能静默地把随机噪声当观测喂给策略。
  2. 不再用 torch.no_grad() 包裹特征提取，梯度可以流到 backbone，
     使冻结/解冻计划真正生效。
  3. backbone 固定 eval 模式（冻结 BN running stats），保证特征确定性，
     这是预训练检测器做特征提取/微调的标准做法。

接口：
  YoloBackboneEncoder - 观测 (B, C, H, W) → 特征 (B, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# YOLOv8 DetectionModel.model 中，Detect 头取特征的来源层索引（P3, P4, P5）。
# 由 model.model.model[-1].f 确认为 [15, 18, 21]。
_FEATURE_LAYERS = (15, 18, 21)
# 对应通道数（yolov8s，输入 384×640 时实测 128/256/512）。
_FEATURE_CHANNELS = (128, 256, 512)


class YoloBackboneEncoder(nn.Module):
    """YOLOv8-s Backbone 编码器。

    将 YOLOv8-s 的多尺度特征 (P3, P4, P5) 融合为单一状态表征。

    Args:
        output_dim: 输出特征维度（default: 512）
        pretrained: 是否加载预训练权重（default: True）
        freeze_backbone: 是否冻结 backbone 权重（default: False）
        device: 设备（default: 'cpu'）
    """

    def __init__(
        self,
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.device_name = device

        # 加载 YOLOv8-s backbone
        try:
            from ultralytics import YOLO

            model = YOLO("runs/checkpoints/yolov8s.pt" if pretrained else "yolov8s.yaml", verbose=False)
            self.backbone = model.model  # DetectionModel (nn.Module)
        except ImportError:
            raise ImportError("ultralytics 未安装。请运行: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"YOLO 模型加载失败: {e}")

        # backbone 固定 eval（冻结 BN），保证同一输入→同一特征（确定性）。
        # 微调时仍可通过梯度更新卷积权重，只是 BN 用 running stats。
        self.backbone.eval()

        # 在 P3/P4/P5 来源层注册 forward hook，捕获真实多尺度特征。
        self._feature_cache: dict = {}
        layers = self.backbone.model  # nn.Sequential
        for idx in _FEATURE_LAYERS:
            layers[idx].register_forward_hook(self._make_hook(idx))

        # 冻结 backbone 权重（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 多尺度特征融合头：输入通道数取自实测 P3/P4/P5
        self.fusion_head = MultiScaleFusionHead(
            input_channels=list(_FEATURE_CHANNELS),
            output_dim=output_dim,
        )

        self.to(device)

    def _make_hook(self, idx: int):
        def hook(_module, _inp, out):
            self._feature_cache[idx] = out

        return hook

    def train(self, mode: bool = True):
        """覆盖 train()：保持 backbone 始终 eval（冻结 BN），其余子模块跟随 mode。"""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W) float32，值域 [0, 1] 或 uint8 [0, 255]
                 分辨率任意，会自动调整到 384×640

        Returns:
            state: (B, output_dim) 状态表征
        """
        # 值域归一化到 [0, 1]
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        elif obs.max() > 1.0:
            obs = obs / 255.0

        # 调整到目标分辨率 384×640
        if obs.shape[2:] != (384, 640):
            obs = F.interpolate(obs, size=(384, 640), mode="bilinear", align_corners=False)

        features = self._extract_yolo_features(obs)
        state = self.fusion_head(features)
        return state

    def _extract_yolo_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """通过 forward hook 从 YOLO 提取真实多尺度特征 [P3, P4, P5]。

        不使用 torch.no_grad()：当 backbone 解冻时梯度可正常回传。
        若任一层未捕获到 4D 特征，直接报错（绝不返回随机噪声）。
        """
        self._feature_cache.clear()
        # 跑一次 backbone（Detect 头输出被忽略，只为触发 hook）。
        self.backbone(x)

        features = []
        for idx in _FEATURE_LAYERS:
            feat = self._feature_cache.get(idx, None)
            if not (isinstance(feat, torch.Tensor) and feat.dim() == 4):
                raise RuntimeError(
                    f"YOLO 特征提取失败：第 {idx} 层未捕获到 4D 特征图，"
                    f"得到 {type(feat)}。请检查 ultralytics 版本与层索引 {_FEATURE_LAYERS}。"
                )
            features.append(feat)

        self._feature_cache.clear()
        return features


class MultiScaleFusionHead(nn.Module):
    """多尺度特征融合头。

    将 P3, P4, P5 三个不同尺度的特征融合为单一状态向量。

    Args:
        input_channels: [C_p3, C_p4, C_p5] 输入通道数
        output_dim: 输出向量维度
    """

    def __init__(self, input_channels: List[int], output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim

        # 每个尺度：全局池化 → 投影到 proj_dim
        proj_dim = output_dim // len(input_channels)
        self.projections = nn.ModuleList([
            nn.Linear(c, proj_dim) for c in input_channels
        ])
        concat_dim = proj_dim * len(input_channels)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [P3, P4, P5] 三个尺度的特征 (B, C, H, W)

        Returns:
            state: (B, output_dim)
        """
        projected = []
        for feat, proj in zip(features, self.projections):
            # 全局平均池化 (B, C, H, W) → (B, C)
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            projected.append(proj(pooled))

        fused = torch.cat(projected, dim=1)
        return self.fusion(fused)


def create_yolo_encoder(
    output_dim: int = 512,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cpu",
) -> YoloBackboneEncoder:
    """便捷函数：创建 YOLO 编码器。"""
    return YoloBackboneEncoder(
        output_dim=output_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device=device,
    )
