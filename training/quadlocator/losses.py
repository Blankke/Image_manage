"""QuadLocator-S 多任务监督损失。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def quadlocator_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """组合角点、mask、boundary、存在性、类别与坐标一致性损失。"""

    content_heatmap = _weighted_heatmap_bce(
        outputs["content_corner_heatmaps"], targets["content_corner_heatmaps"]
    )
    outer_per_sample = F.binary_cross_entropy_with_logits(
        outputs["outer_corner_heatmaps"],
        targets["outer_corner_heatmaps"],
        reduction="none",
    ).mean(dim=(1, 2, 3))
    outer_weights = targets["outer_present"].reshape(-1)
    outer_heatmap = (outer_per_sample * outer_weights).sum() / outer_weights.sum().clamp_min(1.0)
    mask = F.binary_cross_entropy_with_logits(
        outputs["content_mask_logits"], targets["content_mask"]
    ) + _dice_loss(outputs["content_mask_logits"], targets["content_mask"])
    boundary = F.binary_cross_entropy_with_logits(
        outputs["boundary_logits"], targets["boundary"]
    )
    presence = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"], targets["presence"]
    )
    classification = F.cross_entropy(outputs["class_logits"], targets["target_class"])
    predicted_corners = _softargmax_corners(outputs["content_corner_heatmaps"])
    corner_per_sample = torch.abs(predicted_corners - targets["content_corners"]).mean(dim=(1, 2))
    present_weights = targets["presence"].reshape(-1)
    corner_geometry = (corner_per_sample * present_weights).sum() / present_weights.sum().clamp_min(1.0)
    total = (
        2.0 * content_heatmap
        + 0.7 * outer_heatmap
        + 1.0 * mask
        + 0.8 * boundary
        + 0.8 * presence
        + 0.6 * classification
        + 1.2 * corner_geometry
    )
    metrics = {
        "total": float(total.detach()),
        "content_heatmap": float(content_heatmap.detach()),
        "outer_heatmap": float(outer_heatmap.detach()),
        "mask": float(mask.detach()),
        "boundary": float(boundary.detach()),
        "presence": float(presence.detach()),
        "classification": float(classification.detach()),
        "corner_geometry": float(corner_geometry.detach()),
    }
    return total, metrics


def _weighted_heatmap_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = 1.0 + 10.0 * target
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()


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
