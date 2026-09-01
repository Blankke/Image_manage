"""QuadLocator-S 多任务监督损失。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from training.quadlocator.decoder import local_softargmax_corners


def quadlocator_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    profile: str = "full",
) -> tuple[torch.Tensor, dict[str, float]]:
    """组合角点、mask、boundary、存在性、类别与坐标一致性损失。"""

    if profile not in {"p2", "boundary", "tail", "full"}:
        raise ValueError("loss profile 必须为 p2/boundary/tail/full")

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
    boundary = (
        F.binary_cross_entropy_with_logits(outputs["boundary_logits"], targets["boundary"])
        if profile in {"p2", "tail"}
        else _balanced_boundary_loss(outputs["boundary_logits"], targets["boundary"])
    )
    presence = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"], targets["presence"]
    )
    outer_presence = F.binary_cross_entropy_with_logits(
        outputs["outer_presence_logits"], targets["outer_present"]
    )
    classification = F.cross_entropy(outputs["class_logits"], targets["target_class"])
    predicted_corners = local_softargmax_corners(outputs["content_corner_heatmaps"])
    corner_per_sample = F.smooth_l1_loss(
        predicted_corners,
        targets["content_corners"],
        beta=0.01,
        reduction="none",
    ).mean(dim=(1, 2))
    present_weights = targets["presence"].reshape(-1)
    corner_geometry = (corner_per_sample * present_weights).sum() / present_weights.sum().clamp_min(1.0)
    predicted_outer_corners = local_softargmax_corners(outputs["outer_corner_heatmaps"])
    outer_corner_per_sample = F.smooth_l1_loss(
        predicted_outer_corners,
        targets["outer_corners"],
        beta=0.01,
        reduction="none",
    ).mean(dim=(1, 2))
    outer_corner_geometry = (
        (outer_corner_per_sample * outer_weights).sum() / outer_weights.sum().clamp_min(1.0)
    )
    positive_corner_errors = corner_per_sample[present_weights >= 0.5]
    cvar = _tail_mean(positive_corner_errors, fraction=0.25)
    ambiguity = _peak_ambiguity_penalty(outputs["content_corner_heatmaps"], present_weights)
    mask_consistency = _mask_quad_consistency(
        outputs["content_mask_logits"], predicted_corners, present_weights
    )
    boundary_consistency = _corner_boundary_consistency(
        outputs["boundary_logits"], predicted_corners, present_weights
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
    if profile in {"tail", "full"}:
        total = (
            total
            + 0.35 * cvar
            + 0.08 * ambiguity
            + 0.12 * mask_consistency
            + 0.12 * boundary_consistency
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
        "corner_cvar": float(cvar.detach()),
        "ambiguity": float(ambiguity.detach()),
        "mask_quad_consistency": float(mask_consistency.detach()),
        "corner_boundary_consistency": float(boundary_consistency.detach()),
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


def _balanced_boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """稀疏窄带使用动态正类权重的 BCE/focal 与 soft Dice。"""

    positive_fraction = target.mean().detach().clamp(1e-4, 0.5)
    positive_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(1.0, 30.0)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
        reduction="none",
    )
    probability = torch.sigmoid(logits)
    focal = torch.pow(torch.abs(target - probability), 2.0)
    return (bce * focal).mean() + _dice_loss(logits, target)


def _tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    count = max(1, int(round(values.numel() * fraction)))
    return torch.topk(values, count, largest=True).values.mean()


def _peak_ambiguity_penalty(logits: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """在抑制主峰 3px 邻域后惩罚第二强峰，避免把 shoulder 当候选。"""

    probabilities = torch.sigmoid(logits)
    batch, corners, height, width = probabilities.shape
    peaks = probabilities.detach().reshape(batch, corners, -1).argmax(dim=2)
    peak_y = torch.div(peaks, width, rounding_mode="floor")
    peak_x = peaks % width
    yy = torch.arange(height, device=logits.device).view(1, 1, height, 1)
    xx = torch.arange(width, device=logits.device).view(1, 1, 1, width)
    keep = (xx - peak_x[:, :, None, None]).square() + (
        yy - peak_y[:, :, None, None]
    ).square() > 9
    peak1 = probabilities.reshape(batch, corners, -1).amax(dim=2)
    peak2 = (probabilities * keep.to(probabilities.dtype)).reshape(batch, corners, -1).amax(dim=2)
    ratio = peak2 / peak1.clamp_min(1e-6)
    per_sample = ratio.mean(dim=1)
    return (per_sample * present).sum() / present.sum().clamp_min(1.0)


def _corner_boundary_consistency(
    boundary_logits: torch.Tensor,
    corners: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    """沿四条预测边可微采样 boundary，鼓励角点和边界头形成同一四边形。"""

    samples = torch.linspace(0.0, 1.0, 16, device=corners.device, dtype=corners.dtype)
    edge_points = []
    for index in range(4):
        start = corners[:, index : index + 1]
        end = corners[:, (index + 1) % 4 : (index + 1) % 4 + 1]
        if end.shape[1] == 0:
            end = corners[:, :1]
        edge_points.append(start + (end - start) * samples[None, :, None])
    grid = torch.cat(edge_points, dim=1) * 2.0 - 1.0
    sampled = F.grid_sample(
        torch.sigmoid(boundary_logits),
        grid[:, None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).flatten(1)
    per_sample = 1.0 - sampled.mean(dim=1)
    return (per_sample * present).sum() / present.sum().clamp_min(1.0)


def _mask_quad_consistency(
    mask_logits: torch.Tensor,
    corners: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    """以四角包围盒约束 mask 重心，提供便宜且稳定的一致性信号。"""

    probability = torch.sigmoid(mask_logits[:, 0])
    height, width = probability.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=probability.device),
        torch.linspace(0.0, 1.0, width, device=probability.device),
        indexing="ij",
    )
    total = probability.sum(dim=(1, 2)).clamp_min(1e-6)
    center = torch.stack(
        (
            (probability * xx).sum(dim=(1, 2)) / total,
            (probability * yy).sum(dim=(1, 2)) / total,
        ),
        dim=1,
    )
    quad_center = corners.mean(dim=1)
    per_sample = F.smooth_l1_loss(center, quad_center, beta=0.02, reduction="none").mean(1)
    return (per_sample * present).sum() / present.sum().clamp_min(1.0)


def _softargmax_corners(logits: torch.Tensor) -> torch.Tensor:
    """保留旧导入名，但实现已统一为局部 sigmoid soft-argmax。"""

    return local_softargmax_corners(logits)
