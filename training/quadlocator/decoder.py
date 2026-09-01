"""QuadLocator 训练侧局部可微角点解码。

离散全局峰仅决定局部窗口，不参与梯度；窗口内使用 sigmoid 权重求亚像素坐标。规范
和版本号直接复用产品运行时模块，防止训练/部署协议漂移。
"""

from __future__ import annotations

import torch

from screenrestore.geometry.decoder import DECODER_VERSION, CornerDecoderSpec


def local_softargmax_corners(
    logits: torch.Tensor,
    spec: CornerDecoderSpec | None = None,
) -> torch.Tensor:
    """把 ``B×4×H×W`` logits 解码为归一化 ``B×4×2`` 坐标。"""

    active_spec = spec or CornerDecoderSpec()
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("角点 logits 必须为 B×4×H×W")
    batch, corners, height, width = logits.shape
    radius = active_spec.local_window // 2
    peaks = logits.detach().reshape(batch, corners, -1).argmax(dim=-1)
    peak_y = torch.div(peaks, width, rounding_mode="floor")
    peak_x = peaks % width
    all_y = torch.arange(height, device=logits.device).view(1, 1, height, 1)
    all_x = torch.arange(width, device=logits.device).view(1, 1, 1, width)
    local_mask = (
        (torch.abs(all_y - peak_y[:, :, None, None]) <= radius)
        & (torch.abs(all_x - peak_x[:, :, None, None]) <= radius)
    )
    weights = torch.sigmoid(logits) * local_mask.to(logits.dtype)
    denominator = weights.sum(dim=(2, 3)).clamp_min(1e-8)
    x = (weights * all_x).sum(dim=(2, 3)) / denominator / max(1, width - 1)
    y = (weights * all_y).sum(dim=(2, 3)) / denominator / max(1, height - 1)
    return torch.stack((x, y), dim=-1)


__all__ = ["DECODER_VERSION", "CornerDecoderSpec", "local_softargmax_corners"]
