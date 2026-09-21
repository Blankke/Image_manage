"""Mendeley ZIP 按需抽样的索引、字节范围与 CRC 回归。

运行范例：source .venv/bin/activate && which python &&
XDG_STATE_HOME=/tmp/screenrestore-mendeley-test pytest -q tests/unit/test_sample_mendeley_corner_dataset.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from scripts import sample_mendeley_corner_dataset as sampler
from scripts.audit_mendeley_corner_sample import _read_quad


def test_zip_tail_offsets_and_member_crc(monkeypatch) -> None:
    # 小归档模拟远端 ZIP：尾部索引中的 header_offset 与整包坐标不同。
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("All My Dataset/(7).jpg", b"photo" * 300, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("All My Dataset/(7).csv", b"1,2,3,4\n", compress_type=zipfile.ZIP_STORED)
        archive.writestr("All My Dataset/(8).jpg", b"missing label")
        archive.writestr("All My Dataset/(7).jpg 2", b"duplicate suffix")
    complete = stream.getvalue()
    tail_start = len(complete) - 350
    with zipfile.ZipFile(io.BytesIO(complete[tail_start:])) as archive:
        pairs = sampler._paired_members(archive.infolist())
    assert list(pairs) == [7]

    def fake_range(spec: str) -> tuple[bytes, int, int]:
        start, end = map(int, spec.removeprefix("bytes=").split("-"))
        return complete[start : end + 1], len(complete), start

    monkeypatch.setattr(sampler, "_range_request", fake_range)
    image, label = pairs[7]
    assert sampler._read_member(image, tail_start, len(complete)) == b"photo" * 300
    assert sampler._read_member(label, tail_start, len(complete)) == b"1,2,3,4\n"


def test_annotation_is_checked_in_exif_oriented_image_coordinates(tmp_path: Path) -> None:
    label = tmp_path / "label.csv"
    label.write_text("10,20,90,20,90,180,10,180,device\n", encoding="utf-8")

    quad, reason = _read_quad(label, (100, 200))

    assert reason is None
    assert quad is not None
    assert quad.tolist() == [[10.0, 20.0], [90.0, 20.0], [90.0, 180.0], [10.0, 180.0]]
    _, reason = _read_quad(label, (200, 100))
    assert reason == "coordinates_outside_exif_oriented_image"
