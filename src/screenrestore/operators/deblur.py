"""实验性 Wiener 反卷积。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import require_range, require_rgb_u8


class PsfType(StrEnum):
    """点扩散函数类型。"""

    MOTION = "motion"
    GAUSSIAN = "gaussian"


@dataclass
class DeblurParameters(ParameterModel):
    """Wiener 反卷积参数，默认强度为零。"""

    psf_type: PsfType = PsfType.MOTION
    motion_length: int = 9
    motion_angle: float = 0.0
    gaussian_sigma: float = 2.0
    noise_ratio: float = 0.01
    strength: float = 0.0

    def validate(self) -> None:
        if not 1 <= self.motion_length <= 99:
            raise ValueError("motion_length 必须位于 1..99")
        require_range("motion_angle", self.motion_angle, -180.0, 180.0)
        require_range("gaussian_sigma", self.gaussian_sigma, 0.3, 20.0)
        require_range("noise_ratio", self.noise_ratio, 1e-5, 1.0)
        require_range("strength", self.strength, 0.0, 1.0)


class DeblurOperator(ImageOperator[DeblurParameters]):
    """对 LAB 亮度做有正则项的 Wiener 反卷积并受限混合。"""

    id = "deblur"
    display_name = "去模糊（实验性）"
    parameter_type = DeblurParameters

    def default_parameters(self) -> DeblurParameters:
        return DeblurParameters()

    def apply(
        self,
        image: np.ndarray,
        params: DeblurParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_u8(image)
        self.validate(params)
        if params.strength == 0:
            return image.copy()
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness = lab[..., 0].astype(np.float32) / 255.0
        psf = _make_psf(params)
        context.report(0.15, "执行 Wiener 反卷积")
        transfer = np.fft.fft2(np.fft.ifftshift(psf), s=lightness.shape)
        denominator = np.square(np.abs(transfer)) + max(params.noise_ratio, 1e-6)
        restored_frequency = np.fft.fft2(lightness) * np.conj(transfer) / denominator
        restored = np.real(np.fft.ifft2(restored_frequency)).astype(np.float32)
        restored = np.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
        # 以稳健分位数限制振铃和数值爆炸，再映射回原亮度范围。
        low, high = np.percentile(restored, (0.2, 99.8))
        restored = np.clip(restored, low, max(low + 1e-5, high))
        restored = np.clip(restored, 0.0, 1.0)
        mixed = lightness * (1.0 - params.strength) + restored * params.strength
        output_lab = lab.copy()
        output_lab[..., 0] = np.clip(np.rint(mixed * 255), 0, 255).astype(np.uint8)
        context.report(1.0, "实验性去模糊完成")
        return cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)


def _make_psf(params: DeblurParameters) -> np.ndarray:
    if params.psf_type == PsfType.GAUSSIAN:
        radius = max(2, int(np.ceil(params.gaussian_sigma * 3)))
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel_1d = np.exp(-0.5 * np.square(axis / params.gaussian_sigma))
        psf = np.outer(kernel_1d, kernel_1d)
    else:
        size = max(3, params.motion_length | 1)
        psf = np.zeros((size, size), np.float32)
        center = (size - 1) / 2
        radius = (params.motion_length - 1) / 2
        radians = np.deg2rad(params.motion_angle)
        dx, dy = radius * np.cos(radians), radius * np.sin(radians)
        cv2.line(
            psf,
            (round(center - dx), round(center - dy)),
            (round(center + dx), round(center + dy)),
            1.0,
            1,
            cv2.LINE_AA,
        )
    return psf / max(float(psf.sum()), 1e-8)

