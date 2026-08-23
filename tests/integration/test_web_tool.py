"""本地 Web 服务、上传边界和共享流水线的集成测试。"""

from __future__ import annotations

import base64
import http.client
import json
import sys
import threading
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from screenrestore.io.image_exporter import ExportFormat, encode_image_bytes
from screenrestore.web.server import create_server
from screenrestore.web.service import WebModelCatalog, WebRestoreService


def _test_image() -> np.ndarray:
    yy, xx = np.indices((84, 132))
    return np.stack(
        ((xx * 2) % 255, (yy * 3) % 255, ((xx + yy) * 2) % 255),
        axis=2,
    ).astype(np.uint8)


def _multipart(image_data: bytes, settings: dict[str, object]) -> tuple[str, bytes]:
    boundary = "ScreenRestoreBoundary7MA4YWxk"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="settings"\r\n')
    body.extend(b"Content-Type: application/json; charset=utf-8\r\n\r\n")
    body.extend(json.dumps(settings, ensure_ascii=False).encode("utf-8"))
    body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="files"; filename="screen.png"\r\n')
    body.extend(b"Content-Type: image/png\r\n\r\n")
    body.extend(image_data)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", bytes(body)


def test_web_health_static_and_restore_roundtrip() -> None:
    server = create_server("127.0.0.1", 0, max_upload_mb=8, max_jobs=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request("GET", "/api/v1/health")
        response = connection.getresponse()
        health = json.loads(response.read())
        assert response.status == 200
        assert health["privacy"] == "local-memory-only"

        connection.request("GET", "/api/v1/models")
        response = connection.getresponse()
        model_catalog = json.loads(response.read())
        assert response.status == 200
        assert model_catalog["status"] == "ok"
        assert all("manifest_path" not in item for item in model_catalog["models"])
        assert all("/home/" not in item["status"] for item in model_catalog["models"])

        connection.request("GET", "/")
        response = connection.getresponse()
        html = response.read().decode("utf-8")
        assert response.status == 200
        assert "ScreenRestore Web" in html
        assert 'id="demoirePolicy"' in html
        assert 'id="dehaloPolicy"' in html
        assert "default-src 'self'" in response.getheader("Content-Security-Policy")

        image_data = encode_image_bytes(_test_image(), ExportFormat.PNG)
        content_type, body = _multipart(
            image_data,
            {
                "preset": "display",
                "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "ratio_mode": "free",
            },
        )
        connection.request(
            "POST",
            "/api/v1/restore",
            body=body,
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        restored_bytes = response.read()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/png"
        with Image.open(BytesIO(restored_bytes)) as restored:
            assert restored.mode == "RGB"
            assert restored.width > 100 and restored.height > 60

        encoded_diagnostics = response.getheader("X-ScreenRestore-Diagnostics")
        padded = encoded_diagnostics + "=" * ((4 - len(encoded_diagnostics) % 4) % 4)
        diagnostics = json.loads(base64.urlsafe_b64decode(padded))
        assert diagnostics["status"] == "ok"
        assert diagnostics["fusion"]["claim"] == "single-observation"
        assert diagnostics["fusion"]["clipped_observation_fraction"] > 0
        assert diagnostics["fusion"]["unresolved_fraction"] == 0
        assert diagnostics["provenance"]["variant"] == "archive"
        assert diagnostics["provenance"]["pixel_origin_fraction"]["observed"] == 1.0
        assert "lens_distortion" not in diagnostics["operator_timings"]
        assert diagnostics["artifacts"]["demoire"]["mode"] == "joint_edge_aware"
        assert "activated" in diagnostics["artifacts"]["dehalo"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_rejects_unknown_restore_settings() -> None:
    server = create_server("127.0.0.1", 0, max_upload_mb=8, max_jobs=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        image_data = encode_image_bytes(_test_image(), ExportFormat.PNG)
        content_type, body = _multipart(image_data, {"unknown_switch": True})
        connection.request(
            "POST",
            "/api/v1/restore",
            body=body,
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 400
        assert "未知字段" in payload["error"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_ai_only_accepts_allowlisted_model_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = WebRestoreService(model_directories=[tmp_path])
    common = {
        "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "ratio_mode": "free",
        "processing_mode": "ai_enhanced",
        "output_variant": "ai_enhanced",
    }
    with pytest.raises(ValueError, match="未知字段"):
        service.restore(
            [_test_image()],
            {**common, "ai": {"enabled": True, "manifest_path": "/tmp/model.json"}},
        )
    with pytest.raises(ValueError, match="未获服务器允许"):
        service.restore(
            [_test_image()],
            {**common, "ai": {"enabled": True, "manifest_id": "not-allowed"}},
        )


def test_web_classic_artifact_overrides_are_reachable() -> None:
    service = WebRestoreService(model_directories=[])
    result = service.restore(
        [_test_image()],
        {
            "preset": "cinema",
            "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "ratio_mode": "free",
            "operator_overrides": {
                "demoire": {
                    "enabled": True,
                    "params": {"mode": "joint_edge_aware", "strength": 1.0},
                },
                "dehalo": {"enabled": False},
            },
        },
    )
    artifacts = result.diagnostics["artifacts"]
    assert artifacts["demoire"]["mode"] == "joint_edge_aware"  # type: ignore[index]
    assert artifacts["dehalo"] is None  # type: ignore[index]


def test_web_ai_enhancement_runs_allowlisted_external_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest_path = tmp_path / "copy-enhancement.json"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "copy-enhancement",
                "name": "测试增强模型",
                "type": "external_process",
                "role": "enhancement",
                "task": "perceptual_restoration",
                "executable": sys.executable,
                "arguments": [
                    "-c",
                    (
                        "from PIL import Image; import sys; "
                        "Image.open(sys.argv[1]).save(sys.argv[2])"
                    ),
                    "{input}",
                    "{output}",
                ],
                "license": "test-only",
            }
        ),
        encoding="utf-8",
    )
    service = WebRestoreService(model_directories=[tmp_path])
    result = service.restore(
        [_test_image()],
        {
            "preset": "display",
            "processing_mode": "ai_enhanced",
            "output_variant": "ai_enhanced",
            "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "ratio_mode": "free",
            "ai": {
                "enabled": True,
                "manifest_id": "copy-enhancement",
                "strength": 0.2,
                "denoise_strength": 0.2,
                "output_scale": 1,
            },
        },
    )
    assert result.image_rgb.dtype == np.float32
    assert result.diagnostics["fidelity_claim"] == "perceptual-generated-detail"
    assert result.diagnostics["ai"]["enhancement"]["manifest_id"] == "copy-enhancement"  # type: ignore[index]


def test_web_model_catalog_rejects_duplicate_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    directories = [tmp_path / "one", tmp_path / "two", tmp_path / "three"]
    for directory in directories:
        directory.mkdir()
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "id": "duplicate",
                    "name": "重复模型",
                    "type": "external_process",
                    "role": "enhancement",
                    "task": "super_resolution",
                    "executable": sys.executable,
                }
            ),
            encoding="utf-8",
        )
    response = WebModelCatalog(directories).response()
    assert response["models"] == []
    assert len(response["errors"]) == 2  # type: ignore[arg-type]
