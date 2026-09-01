"""P3 轻度 dewarp 与专项模型的可审计约束损失。

使用示例：
    from training.p3.losses import dewarp_grid_loss
    loss, parts = dewarp_grid_loss(predicted_grid, target_grid)
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def bending_energy(grid: torch.Tensor) -> torch.Tensor:
    """惩罚控制网格二阶变化，保留平滑曲面先验。"""

    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid 必须为 B×rows×columns×2")
    second_x = grid[:, :, 2:] - 2.0 * grid[:, :, 1:-1] + grid[:, :, :-2]
    second_y = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
    return second_x.square().mean() + second_y.square().mean()


def jacobian_fold_loss(grid: torch.Tensor, minimum: float = 0.05) -> torch.Tensor:
    """对 output→input 位移网格的非正 Jacobian 施加软门。"""

    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid 必须为 B×rows×columns×2")
    rows, columns = grid.shape[1:3]
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, rows, device=grid.device, dtype=grid.dtype),
        torch.linspace(0.0, 1.0, columns, device=grid.device, dtype=grid.dtype),
        indexing="ij",
    )
    mapping = grid + torch.stack((xx, yy), dim=-1)[None]
    dx = (mapping[:, :, 1:] - mapping[:, :, :-1]) * max(1, columns - 1)
    dy = (mapping[:, 1:] - mapping[:, :-1]) * max(1, rows - 1)
    dx = dx[:, :-1]
    dy = dy[:, :, :-1]
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    return F.relu(minimum - determinant).mean()


def straight_line_loss(grid: torch.Tensor) -> torch.Tensor:
    """限制同一控制行/列的局部方向突变，保护直线结构。"""

    return bending_energy(grid)


def dewarp_grid_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    identity_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = F.smooth_l1_loss(predicted, target, beta=0.005)
    bending = bending_energy(predicted)
    fold = jacobian_fold_loss(predicted)
    straight = straight_line_loss(predicted)
    identity = predicted.abs().mean() if torch.count_nonzero(target).item() == 0 else predicted.new_zeros(())
    total = reconstruction + 0.04 * bending + 0.5 * fold + 0.03 * straight + identity_weight * identity
    return total, {
        "grid_reconstruction": float(reconstruction.detach()),
        "bending": float(bending.detach()),
        "fold": float(fold.detach()),
        "straight_line": float(straight.detach()),
        "identity": float(identity.detach()),
    }


__all__ = ["bending_energy", "dewarp_grid_loss", "jacobian_fold_loss", "straight_line_loss"]
