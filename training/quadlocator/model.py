"""Core ML / ONNX 友好的轻量多任务 QuadLocator-S。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _channels(value: int, width_multiplier: float) -> int:
    return max(8, int(round(value * width_multiplier / 8.0)) * 8)


class ConvNormAct(nn.Sequential):
    """只使用常规卷积、BN 与 SiLU，便于后续 Apple 后端转换。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseBlock(nn.Module):
    """MobileNet 风格深度可分离残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.pointwise(self.depthwise(inputs))
        return output + inputs if self.use_residual else output


class PredictionHead(nn.Sequential):
    """共享 1/4 特征上的轻量输出头。"""

    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__(
            DepthwiseBlock(channels, channels, stride=1),
            nn.Conv2d(channels, outputs, kernel_size=1),
        )


class QuadLocatorS(nn.Module):
    """输出 content/outer 四角热图、mask、boundary、presence 与类别。"""

    def __init__(self, width_multiplier: float = 1.0, class_count: int = 4) -> None:
        super().__init__()
        if not 0.35 <= width_multiplier <= 2.0:
            raise ValueError("width_multiplier 必须位于 0.35..2.0")
        channels = [_channels(value, width_multiplier) for value in (16, 24, 40, 80, 128)]
        self.width_multiplier = width_multiplier
        self.class_count = class_count
        self.stem = ConvNormAct(3, channels[0], stride=2)
        self.stage2 = nn.Sequential(
            DepthwiseBlock(channels[0], channels[1], stride=2),
            DepthwiseBlock(channels[1], channels[1], stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthwiseBlock(channels[1], channels[2], stride=2),
            DepthwiseBlock(channels[2], channels[2], stride=1),
        )
        self.stage4 = nn.Sequential(
            DepthwiseBlock(channels[2], channels[3], stride=2),
            DepthwiseBlock(channels[3], channels[3], stride=1),
        )
        self.stage5 = nn.Sequential(
            DepthwiseBlock(channels[3], channels[4], stride=2),
            DepthwiseBlock(channels[4], channels[4], stride=1),
        )
        fpn_channels = _channels(64, width_multiplier)
        self.lateral2 = nn.Conv2d(channels[1], fpn_channels, 1)
        self.lateral3 = nn.Conv2d(channels[2], fpn_channels, 1)
        self.lateral4 = nn.Conv2d(channels[3], fpn_channels, 1)
        self.lateral5 = nn.Conv2d(channels[4], fpn_channels, 1)
        self.fuse4 = DepthwiseBlock(fpn_channels, fpn_channels, stride=1)
        self.fuse3 = DepthwiseBlock(fpn_channels, fpn_channels, stride=1)
        self.fuse2 = DepthwiseBlock(fpn_channels, fpn_channels, stride=1)
        self.content_corner_head = PredictionHead(fpn_channels, 4)
        self.outer_corner_head = PredictionHead(fpn_channels, 4)
        self.content_mask_head = PredictionHead(fpn_channels, 1)
        self.boundary_head = PredictionHead(fpn_channels, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.presence_head = nn.Linear(channels[4], 1)
        self.class_head = nn.Linear(channels[4], class_count)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(image)
        feature2 = self.stage2(stem)
        feature3 = self.stage3(feature2)
        feature4 = self.stage4(feature3)
        feature5 = self.stage5(feature4)
        pyramid5 = self.lateral5(feature5)
        pyramid4 = self.fuse4(
            self.lateral4(feature4)
            + F.interpolate(pyramid5, size=feature4.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid3 = self.fuse3(
            self.lateral3(feature3)
            + F.interpolate(pyramid4, size=feature3.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid2 = self.fuse2(
            self.lateral2(feature2)
            + F.interpolate(pyramid3, size=feature2.shape[-2:], mode="bilinear", align_corners=False)
        )
        pooled = self.global_pool(feature5).flatten(1)
        return {
            "content_corner_heatmaps": self.content_corner_head(pyramid2),
            "outer_corner_heatmaps": self.outer_corner_head(pyramid2),
            "content_mask_logits": self.content_mask_head(pyramid2),
            "boundary_logits": self.boundary_head(pyramid2),
            "presence_logits": self.presence_head(pooled),
            "class_logits": self.class_head(pooled),
        }


class QuadLocatorExportWrapper(nn.Module):
    """将字典输出稳定转换为 ONNX/Core ML 可命名元组。"""

    def __init__(self, model: QuadLocatorS) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(image)
        return (
            output["content_corner_heatmaps"],
            output["outer_corner_heatmaps"],
            output["content_mask_logits"],
            output["boundary_logits"],
            output["presence_logits"],
            output["class_logits"],
        )
