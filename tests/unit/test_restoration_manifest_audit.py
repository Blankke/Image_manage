"""专项恢复清单的契约与泄漏回归测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_restoration_manifest.py"
_SPEC = importlib.util.spec_from_file_location("audit_restoration_manifest", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_manifest = _MODULE.audit_manifest


def test_audit_accepts_supervised_pairs_without_group_leakage(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_image(data_root / "noise" / "input.png", 16, 12)
    _write_image(data_root / "noise" / "target.png", 16, 12)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    report = audit_manifest(data_root, manifest)

    assert report["records"] == 1
    assert report["task_counts"] == {"denoise": 1}
    assert report["split_counts"] == {"train": 1}


def test_audit_rejects_group_split_leakage(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_image(data_root / "noise" / "input.png", 16, 12)
    _write_image(data_root / "noise" / "target.png", 16, 12)
    first = _record(sample_id="a", split="train")
    second = _record(sample_id="b", split="validation")
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in (first, second)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="数据泄漏"):
        audit_manifest(data_root, manifest)


def test_audit_requires_explicit_private_access(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_image(data_root / "private" / "input.png", 16, 12)
    _write_image(data_root / "private" / "target.png", 16, 12)
    manifest = tmp_path / "pairs.jsonl"
    record = _record(input_image="private/input.png", target_image="private/target.png")
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--allow-private"):
        audit_manifest(data_root, manifest)
    assert audit_manifest(data_root, manifest, allow_private=True)["records"] == 1


def _record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "sample_id": "denoise:001",
        "task": "denoise",
        "split": "train",
        "group_id": "scene:001",
        "capture_session": "session:001",
        "input_image": "noise/input.png",
        "target_image": "noise/target.png",
        "source": "test",
        "license": "test",
    }
    record.update(updates)
    return record


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
