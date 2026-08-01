"""模型清单和外部进程后端测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Timer
from types import SimpleNamespace

import numpy as np
import pytest

from screenrestore.core.cancellation import CancellationToken, ProcessingCancelled
from screenrestore.core.operator import ProcessingContext
from screenrestore.inference.external_process import ExternalProcessBackend
from screenrestore.inference.model_manifest import ModelManifest, discover_manifests, load_manifest
from screenrestore.inference.onnx_backend import OnnxBackend


def test_manifest_roundtrip_and_discovery(tmp_path: Path) -> None:
    manifest_path = tmp_path / "本地模型.json"
    manifest_path.write_text(
        """{
          "id": "copy", "name": "Copy", "type": "external_process",
          "executable": "/bin/true", "arguments": [], "license": "MIT"
        }""",
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    discovered, errors = discover_manifests(tmp_path)
    assert manifest.id == "copy"
    assert [item.id for item in discovered] == ["copy"]
    assert errors == []


def test_external_process_handles_image_io_without_shell() -> None:
    manifest = ModelManifest.from_dict(
        {
            "id": "python-copy",
            "name": "Python copy",
            "type": "external_process",
            "executable": sys.executable,
            "arguments": [
                "-c",
                "from PIL import Image; import sys; Image.open(sys.argv[1]).save(sys.argv[2])",
                "{input}",
                "{output}",
            ],
            "license": "test-only",
        }
    )
    image = np.full((24, 31, 3), (25, 80, 190), np.uint8)
    output = ExternalProcessBackend(manifest).run(image, ProcessingContext())
    assert np.array_equal(output, image)


def test_external_process_exposes_manifest_directory_placeholder(tmp_path: Path) -> None:
    manifest_path = tmp_path / "嵌套" / "model.json"
    manifest_path.parent.mkdir()
    manifest = ModelManifest.from_dict(
        {
            "id": "manifest-dir",
            "name": "Manifest dir",
            "type": "external_process",
            "executable": sys.executable,
            "arguments": [
                "-c",
                (
                    "from PIL import Image; import pathlib,sys; "
                    "assert pathlib.Path(sys.argv[3]).name == '嵌套'; "
                    "Image.open(sys.argv[1]).save(sys.argv[2])"
                ),
                "{input}",
                "{output}",
                "{manifest_dir}",
            ],
        },
        manifest_path,
    )
    image = np.full((7, 9, 3), 91, np.uint8)
    output = ExternalProcessBackend(manifest).run(image, ProcessingContext())
    assert np.array_equal(output, image)


def test_external_process_reports_missing_required_weight(tmp_path: Path) -> None:
    manifest_path = tmp_path / "model.json"
    manifest = ModelManifest.from_dict(
        {
            "id": "missing-weight",
            "name": "Missing weight",
            "type": "external_process",
            "executable": sys.executable,
            "required_files": ["weights/not-installed.pth"],
        },
        manifest_path,
    )
    available, reason = ExternalProcessBackend(manifest).is_available()
    assert not available
    assert "not-installed.pth" in reason


def test_external_process_python_placeholder_uses_active_venv() -> None:
    manifest = ModelManifest.from_dict(
        {
            "id": "active-python",
            "name": "Active Python",
            "type": "external_process",
            "executable": "{python}",
        }
    )
    available, reason = ExternalProcessBackend(manifest).is_available()
    assert available
    assert Path(reason) == Path(sys.executable).absolute()


def test_external_process_can_be_cancelled() -> None:
    manifest = ModelManifest.from_dict(
        {
            "id": "sleep",
            "name": "Sleep",
            "type": "external_process",
            "executable": sys.executable,
            "arguments": ["-c", "import time; time.sleep(3)", "{input}", "{output}"],
            "timeout_seconds": 5,
        }
    )
    token = CancellationToken()
    timer = Timer(0.15, token.cancel)
    timer.start()
    try:
        with pytest.raises(ProcessingCancelled):
            ExternalProcessBackend(manifest).run(
                np.zeros((8, 8, 3), np.uint8),
                ProcessingContext(cancellation=token),
            )
    finally:
        timer.cancel()


def test_onnx_manifest_tiling_runs_identity_session(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    model_path = tmp_path / "identity.onnx"
    model_path.write_bytes(b"test-placeholder")

    class IdentitySession:
        def __init__(self, path: str, providers: list[str]) -> None:
            assert path == str(model_path)
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):  # type: ignore[no-untyped-def]
            return [SimpleNamespace(name="input")]

        def run(self, outputs, feed):  # type: ignore[no-untyped-def]
            del outputs
            return [feed["input"]]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=IdentitySession),
    )
    manifest = ModelManifest.from_dict(
        {
            "id": "identity-onnx",
            "name": "Identity ONNX",
            "type": "onnx",
            "model_path": str(model_path),
            "supports_tiling": True,
            "tile_size": 32,
            "tile_overlap": 8,
            "tile_padding": 4,
        }
    )
    yy, xx = np.indices((43, 61))
    image = np.stack(((xx * 3) % 256, (yy * 5) % 256, (xx + yy) % 256), axis=2).astype(
        np.uint8
    )
    restored = OnnxBackend(manifest).run(image, ProcessingContext())
    assert np.array_equal(restored, image)
