"""无需额外依赖的 Fidelity 恢复验证指标。"""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def fidelity_metrics(restored: torch.Tensor, target: torch.Tensor, clean_restored: torch.Tensor) -> dict[str, float]:
    """返回 PSNR、全局 SSIM、identity、梯度相关和 RGB 色差。"""

    return {
        "psnr": float(_psnr(restored, target).detach()),
        "ssim": float(_global_ssim(restored, target).detach()),
        "identity_mae": float(torch.abs(clean_restored - target).mean().detach()),
        "edge_correlation": float(_edge_correlation(restored, target).detach()),
        "color_error_255": float((torch.linalg.vector_norm(restored - target, dim=1).mean() * 255.0).detach()),
    }


def _psnr(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((first - second).square()).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def _global_ssim(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """逐通道全局 SSIM；验证时稳定且不引入 scikit-image 依赖。"""

    c1, c2 = 0.01**2, 0.03**2
    mean_first = first.mean(dim=(-2, -1), keepdim=True)
    mean_second = second.mean(dim=(-2, -1), keepdim=True)
    variance_first = ((first - mean_first) ** 2).mean(dim=(-2, -1), keepdim=True)
    variance_second = ((second - mean_second) ** 2).mean(dim=(-2, -1), keepdim=True)
    covariance = ((first - mean_first) * (second - mean_second)).mean(dim=(-2, -1), keepdim=True)
    score = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
        (mean_first.square() + mean_second.square() + c1) * (variance_first + variance_second + c2)
    )
    return score.mean()


def _edge_correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_edges = _gradient_magnitude(first).flatten(1)
    second_edges = _gradient_magnitude(second).flatten(1)
    first_edges = first_edges - first_edges.mean(dim=1, keepdim=True)
    second_edges = second_edges - second_edges.mean(dim=1, keepdim=True)
    numerator = (first_edges * second_edges).sum(dim=1)
    denominator = torch.linalg.vector_norm(first_edges, dim=1) * torch.linalg.vector_norm(second_edges, dim=1)
    return (numerator / denominator.clamp_min(1e-8)).mean()


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    kernel_x = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-2, -1)
    grad_x = functional.conv2d(gray, kernel_x, padding=1)
    grad_y = functional.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
