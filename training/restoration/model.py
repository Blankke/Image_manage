"""可导出的紧凑有界残差 Fidelity 恢复网络。"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    """只由常规卷积和 SiLU 组成，便于 ONNX / Apple 后端部署。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.activation(image + self.layers(image))


class BoundedResidualNet(nn.Module):
    """只允许在输入附近做有限幅像素修正的同尺寸模型。

    网络输出 RGB residual 和置信 alpha。``tanh`` 将每通道修正锁在 ``±max_delta``，
    alpha 进一步降低弱证据区域的变化幅度，因而它不能借整图输出重绘缺失内容。
    """

    def __init__(self, channels: int = 32, blocks: int = 6, max_delta: float = 0.06) -> None:
        super().__init__()
        if not 8 <= channels <= 128 or channels % 8:
            raise ValueError("channels 必须是 8..128 的 8 倍数")
        if not 2 <= blocks <= 16:
            raise ValueError("blocks 必须位于 2..16")
        if not 0.01 <= max_delta <= 0.25:
            raise ValueError("max_delta 必须位于 0.01..0.25")
        self.channels = channels
        self.blocks = blocks
        self.max_delta = max_delta
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.SiLU(inplace=True))
        self.body = nn.Sequential(*(_ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # 导出 ONNX 时 batch、空间维度是动态符号量，避免把 Python 形状判断固化进图。
        if not torch.onnx.is_in_onnx_export() and (image.ndim != 4 or image.shape[1] != 3):
            raise ValueError("BoundedResidualNet 需要 N×3×H×W RGB 输入")
        features = self.body(self.stem(image))
        prediction = self.head(features)
        residual = torch.tanh(prediction[:, :3]) * self.max_delta
        alpha = torch.sigmoid(prediction[:, 3:4])
        return torch.clamp(image + alpha * residual, 0.0, 1.0)
