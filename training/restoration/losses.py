"""Fidelity 恢复的重建与干净输入 identity 约束。"""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def fidelity_loss(
    restored: torch.Tensor,
    target: torch.Tensor,
    identity_restored: torch.Tensor,
    clean_input: torch.Tensor,
    *,
    identity_weight: float = 0.35,
    edge_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    """约束模型清除退化的同时，让干净输入尽可能保持像素稳定。"""

    if not 0.0 <= identity_weight <= 2.0 or not 0.0 <= edge_weight <= 2.0:
        raise ValueError("损失权重必须位于 0..2")
    reconstruction = _charbonnier(restored - target)
    identity = _charbonnier(identity_restored - clean_input)
    edge = _charbonnier(_gradient(restored) - _gradient(target))
    total = reconstruction + identity_weight * identity + edge_weight * edge
    return total, {
        "total": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "identity": float(identity.detach()),
        "edge": float(edge.detach()),
    }


def identity_loss(restored: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """无标签观测的保守 identity 约束，不将其当作干净恢复真值。"""

    if restored.shape != observed.shape:
        raise ValueError("identity 约束的输入与输出形状必须相同")
    return _charbonnier(restored - observed)


def _charbonnier(values: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(values.square() + epsilon**2).mean()


def _gradient(image: torch.Tensor) -> torch.Tensor:
    """一阶前向差分足以惩罚明显软化和振铃，无需感知生成特征。"""

    horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
    vertical = image[:, :, 1:, :] - image[:, :, :-1, :]
    return functional.pad(horizontal, (0, 1, 0, 0)) + functional.pad(vertical, (0, 0, 0, 1))
