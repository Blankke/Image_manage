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


class _SimpleGate(nn.Module):
    """NAF 风格无激活门，保持 ONNX/Core ML 友好。"""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second


class _NafLiteBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.project_in = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded
        )
        self.gate = _SimpleGate()
        self.project_out = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn_in = nn.Conv2d(channels, expanded, 1)
        self.ffn_out = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.project_out(self.gate(self.depthwise(self.project_in(self.norm1(value)))))
        value = value + self.beta * residual
        feed_forward = self.ffn_out(self.gate(self.ffn_in(self.norm2(value))))
        return value + self.gamma * feed_forward


class FidelityNetV2(nn.Module):
    """三尺度 NAF-lite 忠实恢复网络，输出受预算约束的 RGB residual。"""

    def __init__(
        self,
        channels: int = 48,
        blocks_per_scale: tuple[int, int, int] = (2, 2, 4),
        max_delta: float = 0.08,
    ) -> None:
        super().__init__()
        if channels < 24 or channels % 8:
            raise ValueError("FidelityNetV2 channels 必须是不小于 24 的 8 倍数")
        if len(blocks_per_scale) != 3 or any(value < 1 for value in blocks_per_scale):
            raise ValueError("FidelityNetV2 必须声明三个尺度的正 block 数")
        self.channels = channels
        self.blocks_per_scale = blocks_per_scale
        self.max_delta = max_delta
        c1, c2, c3 = channels, channels * 2, channels * 4
        self.stem = nn.Conv2d(3, c1, 3, padding=1)
        self.encoder1 = nn.Sequential(*(_NafLiteBlock(c1) for _ in range(blocks_per_scale[0])))
        self.down1 = nn.Conv2d(c1, c2, 2, stride=2)
        self.encoder2 = nn.Sequential(*(_NafLiteBlock(c2) for _ in range(blocks_per_scale[1])))
        self.down2 = nn.Conv2d(c2, c3, 2, stride=2)
        self.middle = nn.Sequential(*(_NafLiteBlock(c3) for _ in range(blocks_per_scale[2])))
        self.up2 = nn.Sequential(nn.Conv2d(c3, c2 * 4, 1), nn.PixelShuffle(2))
        self.decoder2 = nn.Sequential(*(_NafLiteBlock(c2) for _ in range(blocks_per_scale[1])))
        self.up1 = nn.Sequential(nn.Conv2d(c2, c1 * 4, 1), nn.PixelShuffle(2))
        self.decoder1 = nn.Sequential(*(_NafLiteBlock(c1) for _ in range(blocks_per_scale[0])))
        self.head = nn.Conv2d(c1, 5, 3, padding=1)
        # 在线 degradation trace 提供六类低层 artifact/severity 监督；该 head 不参与像素输出。
        self.artifact_head = nn.Linear(c3, 12)
        # 初始像素路径严格 identity，训练后再由 residual/alpha/budget 学习有限修正。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.artifact_head.weight)
        nn.init.zeros_(self.artifact_head.bias)

    def _features(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self.encoder1(self.stem(image))
        second = self.encoder2(self.down1(first))
        middle = self.middle(self.down2(second))
        return first, second, middle

    def forward_components(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second, middle = self._features(image)
        decoded_second = self.decoder2(self.up2(middle) + second)
        decoded_first = self.decoder1(self.up1(decoded_second) + first)
        prediction = self.head(decoded_first)
        residual = torch.tanh(prediction[:, :3]) * self.max_delta
        alpha = torch.sigmoid(prediction[:, 3:4])
        # budget 再约束整图改动，clean identity gate 可直接审计该通道。
        budget = torch.sigmoid(prediction[:, 4:5])
        restored = torch.clamp(image + residual * alpha * budget, 0.0, 1.0)
        return restored, alpha, budget

    def forward_training(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回恢复图、门控量和 trace 辅助预测，供 P3 训练使用。"""

        first, second, middle = self._features(image)
        decoded_second = self.decoder2(self.up2(middle) + second)
        decoded_first = self.decoder1(self.up1(decoded_second) + first)
        prediction = self.head(decoded_first)
        residual = torch.tanh(prediction[:, :3]) * self.max_delta
        alpha = torch.sigmoid(prediction[:, 3:4])
        budget = torch.sigmoid(prediction[:, 4:5])
        restored = torch.clamp(image + residual * alpha * budget, 0.0, 1.0)
        auxiliary = self.artifact_head(torch.mean(middle, dim=(2, 3)))
        artifact_logits, severity_raw = auxiliary.chunk(2, dim=1)
        return restored, alpha, budget, artifact_logits, torch.sigmoid(severity_raw)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export() and (image.ndim != 4 or image.shape[1] != 3):
            raise ValueError("FidelityNetV2 需要 N×3×H×W RGB 输入")
        return self.forward_components(image)[0]


__all__ = ["BoundedResidualNet", "FidelityNetV2"]
