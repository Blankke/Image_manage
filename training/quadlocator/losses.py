"""QuadLocator-S 多任务监督损失。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def quadlocator_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """组合角点、mask、boundary、存在性、类别与坐标一致性损失。"""

    content_per_sample = _gaussian_focal_loss(
        outputs["content_corner_heatmaps"], targets["content_corner_heatmaps"]
    )
    content_weights = 0.15 + 0.85 * targets["presence"].reshape(-1)
    content_heatmap = (content_per_sample * content_weights).mean()
    outer_per_sample = _gaussian_focal_loss(
        outputs["outer_corner_heatmaps"], targets["outer_corner_heatmaps"]
    )
    outer_weights = targets["outer_present"].reshape(-1)
    # absent 样本保留低权重全零热图监督，防止 outer head 在负样本上完全失去梯度。
    outer_sample_weights = 0.10 + 0.90 * outer_weights
    outer_heatmap = (outer_per_sample * outer_sample_weights).mean()
    mask = F.binary_cross_entropy_with_logits(
        outputs["content_mask_logits"], targets["content_mask"]
    ) + _dice_loss(outputs["content_mask_logits"], targets["content_mask"])
    boundary = F.binary_cross_entropy_with_logits(
        outputs["boundary_logits"], targets["boundary"]
    )
    presence = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"], targets["presence"]
    )
    outer_presence = F.binary_cross_entropy_with_logits(
        outputs["outer_presence_logits"], targets["outer_present"]
    )
    classification = F.cross_entropy(outputs["class_logits"], targets["target_class"])
    predicted_corners = _softargmax_corners(outputs["content_corner_heatmaps"])
    corner_per_sample = torch.abs(predicted_corners - targets["content_corners"]).mean(dim=(1, 2))
    present_weights = targets["presence"].reshape(-1)
    corner_geometry = (corner_per_sample * present_weights).sum() / present_weights.sum().clamp_min(1.0)
    predicted_outer_corners = _softargmax_corners(outputs["outer_corner_heatmaps"])
    outer_corner_per_sample = torch.abs(
        predicted_outer_corners - targets["outer_corners"]
    ).mean(dim=(1, 2))
    outer_corner_geometry = (
        (outer_corner_per_sample * outer_weights).sum() / outer_weights.sum().clamp_min(1.0)
    )
    total = (
        2.0 * content_heatmap
        + 0.7 * outer_heatmap
        + 1.0 * mask
        + 0.8 * boundary
        + 0.8 * presence
        + 0.6 * outer_presence
        + 0.6 * classification
        + 1.2 * corner_geometry
        + 0.7 * outer_corner_geometry
    )
    metrics = {
        "total": float(total.detach()),
        "content_heatmap": float(content_heatmap.detach()),
        "outer_heatmap": float(outer_heatmap.detach()),
        "mask": float(mask.detach()),
        "boundary": float(boundary.detach()),
        "presence": float(presence.detach()),
        "outer_presence": float(outer_presence.detach()),
        "classification": float(classification.detach()),
        "corner_geometry": float(corner_geometry.detach()),
        "outer_corner_geometry": float(outer_corner_geometry.detach()),
    }
    return total, metrics


def _gaussian_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """返回逐样本 Gaussian focal loss，控制稠密负像素对正峰的影响。"""

    probabilities = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
    positive = target >= 1.0 - 1e-6
    negative_weight = torch.pow(1.0 - target, 4.0)
    positive_loss = -torch.log(probabilities) * torch.pow(1.0 - probabilities, 2.0) * positive
    negative_loss = (
        -torch.log(1.0 - probabilities)
        * torch.pow(probabilities, 2.0)
        * negative_weight
        * (~positive)
    )
    spatial_dims = (1, 2, 3)
    positive_count = positive.sum(dim=spatial_dims).clamp_min(1)
    positive_term = positive_loss.sum(dim=spatial_dims) / positive_count
    # 无正峰样本仍从全零监督获得梯度；按像素平均避免稠密负例淹没正峰。
    negative_term = negative_loss.mean(dim=spatial_dims)
    return positive_term + 0.25 * negative_term


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _softargmax_corners(logits: torch.Tensor) -> torch.Tensor:
    batch, corners, height, width = logits.shape
    probabilities = torch.softmax(logits.reshape(batch, corners, -1), dim=-1)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=logits.device),
        torch.linspace(0.0, 1.0, width, device=logits.device),
        indexing="ij",
    )
    x = (probabilities * xx.reshape(1, 1, -1)).sum(dim=-1)
    y = (probabilities * yy.reshape(1, 1, -1)).sum(dim=-1)
    return torch.stack((x, y), dim=-1)
