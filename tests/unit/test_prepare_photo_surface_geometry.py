"""真实纸面合成的隔离与内容边界回归。

运行范例：source .venv/bin/activate && which python &&
XDG_STATE_HOME=/tmp/screenrestore-photo-surface-test pytest -q tests/unit/test_prepare_photo_surface_geometry.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts.audit_photo_surface_composite import _audit_record
from scripts.prepare_photo_surface_geometry import _development_groups, _make_cover, _replace_page


def test_development_groups_follow_document_identity() -> None:
    groups = {
        (background, model): [{"split": split}]
        for background in ("background01", "background02")
        for model, split in (("paper", "train"), ("letter", "validation"), ("card", "test"))
    }
    splits, selected = _development_groups(groups)
    assert splits == {"paper": "train", "letter": "validation", "card": "test"}
    assert len(selected) == 2
    assert all(model == "paper" for _, model in selected)

    groups[("background02", "paper")] = [{"split": "validation"}]
    with pytest.raises(ValueError, match="跨 split"):
        _development_groups(groups)


def test_replaced_page_stays_inside_labeled_quad() -> None:
    # 内部像素被替换，远离纸面的桌面像素保留；标签四角对应替换区域。
    photo = np.full((120, 160, 3), 160, np.uint8)
    cover = np.zeros((80, 60, 3), np.uint8)
    cover[:, :] = (20, 30, 220)
    quad = np.asarray([[35, 20], [95, 20], [95, 100], [35, 100]], np.float32)
    result = _replace_page(photo, quad, cover)
    assert np.array_equal(result[5, 5], photo[5, 5])
    assert result[60, 65, 2] > 180
    assert result[60, 65, 0] < 80
    assert np.array_equal(photo[60, 65], np.asarray([160, 160, 160], np.uint8))


def test_cover_keeps_observed_texture_at_top_and_bottom() -> None:
    # 上下区域应保留原画纹理，避免旧版宽色条形成错误的内容层边界。
    source = np.full((900, 620, 3), 135, np.uint8)
    source[:, :310] = (55, 85, 120)
    source[:, 310:] = (190, 150, 85)
    for seed in range(12):
        cover = _make_cover(source, np.random.default_rng(seed))
        assert cover.shape == (900, 620, 3)
        for y in (12, 135, 795, 887):
            assert np.array_equal(cover[y, 8], source[y, 8])
            assert np.array_equal(cover[y, 610], source[y, 610])


def test_pixel_audit_detects_changes_outside_labeled_paper(tmp_path: Path) -> None:
    root = tmp_path / "public-data"
    (root / "geometry").mkdir(parents=True)
    photo = np.full((240, 320, 3), 160, np.uint8)
    cover = np.full((160, 120, 3), (25, 55, 220), np.uint8)
    quad = np.array([[80, 40], [240, 40], [240, 200], [80, 200]], np.float32)
    composite = _replace_page(photo, quad, cover)
    assert cv2.imwrite(str(root / "geometry" / "source.jpg"), photo)
    assert cv2.imwrite(str(root / "geometry" / "composite.jpg"), composite)
    row = {
        "source": "smartdoc-photo-surface-composite",
        "source_capture_id": "geometry/source.jpg",
        "image": "geometry/composite.jpg",
        "content_quad": (quad / np.array([319, 239])).tolist(),
    }
    assert _audit_record(row, root)["status"] == "PASS"

    composite[:40, :40] = 0
    assert cv2.imwrite(str(root / "geometry" / "composite.jpg"), composite)
    assert _audit_record(row, root)["status"] == "FAIL"
