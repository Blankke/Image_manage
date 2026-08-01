"""多帧真实观测融合的对齐和质量测试。"""

from __future__ import annotations

import cv2
import numpy as np
from tests.synthetic.generators import checkerboard

from screenrestore.operators.multiframe_fusion import (
    AlignmentModel,
    MultiFrameFusionParameters,
    align_and_fuse,
)


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def test_other_frames_restore_saturated_reference_observation() -> None:
    clean = checkerboard(360, 220, 18)
    reference = clean.copy()
    reference[65:150, 115:245] = 255
    second = clean.copy()
    second[20:54, 25:100] = 0
    third = clean.copy()
    third[170:205, 270:340] = (248, 248, 248)

    result = align_and_fuse(
        [reference, second, third],
        MultiFrameFusionParameters(
            alignment=AlignmentModel.NONE,
            reference_index=0,
            exposure_compensation=False,
        ),
    )
    damaged_region = np.s_[65:150, 115:245]
    before = _mae(reference[damaged_region], clean[damaged_region])
    after = _mae(result.image_rgb[damaged_region], clean[damaged_region])

    assert after < before * 0.12
    assert float(result.recovered_observation_mask[damaged_region].mean()) > 0.7
    assert result.diagnostics["claim"] == "observed-multiframe-fusion"


def test_translation_alignment_reduces_multiframe_error() -> None:
    clean = checkerboard(320, 180, 15)
    transforms = [(0.0, 0.0), (4.0, -3.0), (-5.0, 2.0)]
    shifted = [
        cv2.warpAffine(
            clean,
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32),
            (clean.shape[1], clean.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        for dx, dy in transforms
    ]
    result = align_and_fuse(
        shifted,
        MultiFrameFusionParameters(
            alignment=AlignmentModel.TRANSLATION,
            reference_index=0,
            minimum_alignment_score=0.03,
            minimum_overlap=0.8,
        ),
    )
    unaligned_average = np.mean(np.stack(shifted).astype(np.float32), axis=0).astype(np.uint8)
    margin = np.s_[12:-12, 12:-12]
    assert _mae(result.image_rgb[margin], clean[margin]) < _mae(
        unaligned_average[margin], clean[margin]
    ) * 0.35
    assert len(result.aligned_indices) == 3


def test_local_detail_weight_prefers_sharper_observation() -> None:
    clean = checkerboard(300, 170, 13)
    blurred = cv2.GaussianBlur(clean, (0, 0), 2.4)
    result = align_and_fuse(
        [blurred, clean, clean],
        MultiFrameFusionParameters(
            alignment=AlignmentModel.NONE,
            reference_index=0,
        ),
    )
    plain_average = np.mean(np.stack((blurred, clean, clean)), axis=0).astype(np.uint8)
    assert _mae(result.image_rgb, clean) < _mae(plain_average, clean) * 0.9


def test_changed_screen_content_is_rejected_instead_of_ghosted() -> None:
    clean = checkerboard(300, 170, 13)
    changed = np.random.default_rng(42).integers(0, 256, clean.shape, np.uint8)
    result = align_and_fuse(
        [clean, clean.copy(), changed],
        MultiFrameFusionParameters(
            alignment=AlignmentModel.NONE,
            reference_index=0,
            minimum_alignment_score=0.0,
        ),
    )
    assert result.rejected_indices == (2,)
    assert _mae(result.image_rgb, clean) < 0.2
