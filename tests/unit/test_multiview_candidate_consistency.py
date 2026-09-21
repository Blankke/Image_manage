"""多视角照片特征审计的数值回归测试。"""

from __future__ import annotations

import cv2
import numpy as np
from scripts.audit_multiview_candidate_consistency import _features, _pair_evidence


def test_repeated_textured_quad_has_cross_view_support(tmp_path) -> None:
    rng = np.random.default_rng(1203)
    image = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    assert cv2.imwrite(str(first), image)
    assert cv2.imwrite(str(second), image)
    detector = cv2.SIFT_create(nfeatures=2500)
    quad = np.asarray([[0.08, 0.08], [0.92, 0.08], [0.92, 0.92], [0.08, 0.92]], np.float32)

    evidence = _pair_evidence(_features(first, detector), _features(second, detector), quad, "second")

    assert evidence["supported"]
    assert evidence["inliers"] >= 12
    assert evidence["seed_coverage"] >= 0.35


def test_unrelated_textures_do_not_gain_cross_view_support(tmp_path) -> None:
    rng = np.random.default_rng(1204)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    assert cv2.imwrite(str(first), rng.integers(0, 256, (512, 512), dtype=np.uint8))
    assert cv2.imwrite(str(second), rng.integers(0, 256, (512, 512), dtype=np.uint8))
    detector = cv2.SIFT_create(nfeatures=2500)
    quad = np.asarray([[0.08, 0.08], [0.92, 0.08], [0.92, 0.92], [0.08, 0.92]], np.float32)

    evidence = _pair_evidence(_features(first, detector), _features(second, detector), quad, "second")

    assert not evidence["supported"]
