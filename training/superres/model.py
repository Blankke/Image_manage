"""可导出的有界残差超分网络。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.layers(value))


class ConservativeSuperResolutionNet(nn.Module):
    """在 bicubic 观测上施加有限残差，避免把超分误作内容重绘。"""

    def __init__(self, scale: int, channels: int = 32, blocks: int = 6, max_delta: float = 0.08) -> None:
        super().__init__()
        if scale not in {2, 4}:
            raise ValueError("scale 必须是 2 或 4")
        if not 8 <= channels <= 128 or channels % 8 or not 2 <= blocks <= 16:
            raise ValueError("channels 或 blocks 超出受支持范围")
        if not 0.01 <= max_delta <= 0.25:
            raise ValueError("max-delta 必须位于 0.01..0.25")
        self.scale = scale
        self.channels = channels
        self.blocks = blocks
        self.max_delta = max_delta
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.SiLU(inplace=True))
        self.body = nn.Sequential(*(_Block(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export() and (image.ndim != 4 or image.shape[1] != 3):
            raise ValueError("超分模型需要 N×3×H×W RGB 输入")
        baseline = functional.interpolate(image, scale_factor=self.scale, mode="bicubic", align_corners=False)
        prediction = self.head(self.body(self.stem(baseline)))
        residual = torch.tanh(prediction[:, :3]) * self.max_delta
        alpha = torch.sigmoid(prediction[:, 3:4])
        return torch.clamp(baseline + alpha * residual, 0.0, 1.0)
