"""P3 dewarp、光度、专项恢复与 artifact router 轻量模型。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _Encoder(nn.Module):
    def __init__(self, output_features: int, channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(channels * 4, output_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1))


class DewarpGridNet(nn.Module):
    """预测轻度弯曲的低分辨率 output→input 归一化位移网格。"""

    def __init__(self, grid_size: int = 17, max_displacement: float = 0.06) -> None:
        super().__init__()
        if grid_size not in (17, 33):
            raise ValueError("dewarp grid_size 必须为 17 或 33")
        self.grid_size = grid_size
        self.max_displacement = max_displacement
        self.encoder = _Encoder(grid_size * grid_size * 2, 32)
        nn.init.zeros_(self.encoder.head.weight)
        nn.init.zeros_(self.encoder.head.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(self.encoder(image)) * self.max_displacement
        return value.reshape(-1, self.grid_size, self.grid_size, 2)


class PhotometricNet(nn.Module):
    """只预测有界摄影参数，不直接生成 RGB 图。"""

    curve_knots = 8
    gain_rows = 4
    gain_columns = 4

    def __init__(self) -> None:
        super().__init__()
        # EV + RGB gains + CCM + curve increments + log gain grid
        self.parameter_count = 1 + 3 + 9 + 3 * self.curve_knots + 3 * 4 * 4
        self.encoder = _Encoder(self.parameter_count, 32)
        # 未训练模型必须是确定性 identity，避免随机参数改变 Archive 输出。
        nn.init.zeros_(self.encoder.head.weight)
        nn.init.zeros_(self.encoder.head.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def apply(self, image: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        offset = 0
        ev = torch.tanh(raw[:, offset : offset + 1]) * 1.0
        offset += 1
        gains = 1.0 + torch.tanh(raw[:, offset : offset + 3]) * 0.25
        offset += 3
        ccm_delta = torch.tanh(raw[:, offset : offset + 9]).reshape(-1, 3, 3) * 0.08
        offset += 9
        increments = F.softplus(
            raw[:, offset : offset + 3 * self.curve_knots].reshape(-1, 3, self.curve_knots)
        )
        offset += 3 * self.curve_knots
        curves = torch.cumsum(increments, dim=2)
        curves = (curves - curves[:, :, :1]) / (curves[:, :, -1:] - curves[:, :, :1]).clamp_min(1e-6)
        log_gain = torch.tanh(raw[:, offset:].reshape(-1, 3, 4, 4)) * 0.18

        value = image * gains[:, :, None, None] * torch.pow(2.0, ev[:, :, None, None])
        identity = torch.eye(3, device=image.device, dtype=image.dtype)[None]
        matrix = identity + ccm_delta
        value = torch.einsum("bij,bjhw->bihw", matrix, value)
        gain_field = torch.exp(F.interpolate(log_gain, image.shape[-2:], mode="bilinear", align_corners=True))
        value = torch.clamp(value * gain_field, 0.0, 1.0)
        # 每通道单调折线；索引只选区间，梯度仍通过区间内线性插值传到像素和曲线。
        scaled = value * (self.curve_knots - 1)
        lower = torch.floor(scaled).to(torch.int64).clamp(0, self.curve_knots - 2)
        fraction = scaled - lower.to(value.dtype)
        expanded_curves = curves[:, :, None, None, :].expand(-1, -1, value.shape[2], value.shape[3], -1)
        low = torch.gather(expanded_curves, 4, lower[..., None]).squeeze(4)
        high = torch.gather(expanded_curves, 4, (lower + 1)[..., None]).squeeze(4)
        return torch.clamp(low + (high - low) * fraction, 0.0, 1.0)


class _ResidualSpecialist(nn.Module):
    def __init__(self, output_channels: int, channels: int = 40, blocks: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, channels, 3, padding=1), nn.SiLU()]
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                    nn.Conv2d(channels, channels, 1),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Conv2d(channels, output_channels, 3, padding=1))
        self.layers = nn.Sequential(*layers)
        nn.init.zeros_(self.layers[-1].weight)  # type: ignore[union-attr]
        nn.init.zeros_(self.layers[-1].bias)  # type: ignore[union-attr]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class DemoireNet(nn.Module):
    """以空间高通残差为频率提示的多尺度去摩尔纹网络。"""

    def __init__(self, max_delta: float = 0.12) -> None:
        super().__init__()
        self.max_delta = max_delta
        self.full = _ResidualSpecialist(4, 40, 6)
        # 避免与 ``torch.nn.Module.half()`` 精度转换方法重名。
        self.half_scale = _ResidualSpecialist(4, 32, 4)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        half = F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False)
        half_prediction = F.interpolate(
            self.half_scale(half), image.shape[-2:], mode="bilinear", align_corners=False
        )
        prediction = self.full(image) + half_prediction
        residual = torch.tanh(prediction[:, :3]) * self.max_delta
        alpha = torch.sigmoid(prediction[:, 3:4])
        return torch.clamp(image + residual * alpha, 0.0, 1.0)


class ReflectionNet(nn.Module):
    """输出透射估计、reflection mask 与 unresolved mask。"""

    def __init__(self, max_delta: float = 0.18) -> None:
        super().__init__()
        self.max_delta = max_delta
        self.network = _ResidualSpecialist(5, 48, 8)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self.network(image)
        reflection_mask = torch.sigmoid(prediction[:, 3:4])
        unresolved = torch.sigmoid(prediction[:, 4:5]) * reflection_mask
        residual = torch.tanh(prediction[:, :3]) * self.max_delta
        transmission = torch.clamp(image + residual * reflection_mask * (1.0 - unresolved), 0.0, 1.0)
        return transmission, reflection_mask, unresolved


class ArtifactRouterNet(nn.Module):
    """多标签 artifact 存在性与 severity 预测。"""

    labels = ("noise", "blur", "jpeg", "photometric", "reflection", "moire", "dewarp")

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder(len(self.labels) * 2, 24)
        nn.init.zeros_(self.encoder.head.weight)
        nn.init.zeros_(self.encoder.head.bias)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.encoder(image)
        logits, severity_raw = values.chunk(2, dim=1)
        return logits, torch.sigmoid(severity_raw)


__all__ = [
    "ArtifactRouterNet",
    "DemoireNet",
    "DewarpGridNet",
    "PhotometricNet",
    "ReflectionNet",
]
