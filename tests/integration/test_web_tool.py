"""本地 Web 服务、上传边界和共享流水线的集成测试。"""

from __future__ import annotations

import base64
import http.client
import json
import threading
from io import BytesIO

import numpy as np
from PIL import Image

from screenrestore.io.image_exporter import ExportFormat, encode_image_bytes
from screenrestore.web.server import create_server


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

        connection.request("GET", "/")
        response = connection.getresponse()
        html = response.read().decode("utf-8")
        assert response.status == 200
        assert "ScreenRestore Web" in html
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
        assert "lens_distortion" not in diagnostics["operator_timings"]
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
