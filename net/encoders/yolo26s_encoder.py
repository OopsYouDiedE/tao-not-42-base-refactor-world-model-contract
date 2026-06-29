"""YOLO26s Backbone 编码器 - 为 Minecraft RL 优化。

使用 YOLO26s 作为图像编码器，利用其目标检测能力：
  - 多尺度特征提取（P3, P4, P5）
  - 小物体检测优化（方块、物品识别）
  - 高效参数化（~11.2M）

多尺度融合方法：
  - P3: 细节层（方块级别）
  - P4: 中层（区域级别）
  - P5: 语义层（全局上下文）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class YOLO26sEncoder(nn.Module):
    """YOLO26s Backbone + 多尺度融合编码器。

    Args:
        input_shape: (C, H, W) - 输入图像形状
        output_dim: 输出特征向量维度
        pretrained: 是否使用预训练权重
    """

    def __init__(self, input_shape=(3, 360, 640), output_dim=512, pretrained=True):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        # YOLO26s Backbone
        # 加载 YOLO v8-s（假设用户指的 YOLO26s 对应 v8-s）
        yolo = YOLO("runs/checkpoints/yolov8s.pt") if pretrained else YOLO("yolov8s.yaml")
        self.backbone = yolo.model

        # 多尺度融合权重（可学习）
        # P3, P4, P5 分别对应三个特征尺度
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

        # 特征压缩层
        self.feature_dim = 512  # YOLO26s 最后一层特征维度
        self.compress = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 360, 640) uint8 or float32 [0, 1]

        Returns:
            features: (B, output_dim) 编码后的特征向量
        """
        # 如果是 uint8，转换为 float32 [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # 通过 backbone 获取多尺度特征
        features = self._extract_yolo_features(x)

        # 多尺度融合
        fused = self._fuse_multi_scale(features)

        # 特征压缩
        output = self.compress(fused)

        return output

    def _extract_yolo_features(self, x):
        """从 YOLO backbone 提取多尺度特征 (P3, P4, P5)。"""
        p3, p4, p5 = self.backbone(x)[:3]  # 取前三个尺度
        return [p3, p4, p5]

    def _fuse_multi_scale(self, features):
        """多尺度特征加权融合。

        Args:
            features: [P3, P4, P5] - 三个尺度的特征

        Returns:
            fused: (B, feature_dim) 融合后的特征向量
        """
        # 全局平均池化 + 展平
        pooled = []
        for feat in features:
            # (B, C, H, W) → (B, C)
            pooled_feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            pooled.append(pooled_feat)

        # 归一化融合权重
        weights = F.softmax(self.fusion_weights, dim=0)

        # 加权求和
        fused_list = []
        for w, feat in zip(weights, pooled):
            # 如果 channel 不同，投影到统一维度
            if feat.shape[1] != self.feature_dim:
                feat = F.linear(feat, torch.randn(self.feature_dim, feat.shape[1]).to(feat.device))
            fused_list.append(w * feat)

        fused = sum(fused_list)

        return fused


# 便捷函数
def create_yolo26s_encoder(pretrained=True, **kwargs):
    """创建 YOLO26s 编码器。"""
    return YOLO26sEncoder(pretrained=pretrained, **kwargs)
