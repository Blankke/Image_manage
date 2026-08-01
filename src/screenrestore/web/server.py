"""ScreenRestore 本地 Web 服务。

使用示例：
    screenrestore-web --open
    python -m screenrestore.web.server --host 127.0.0.1 --port 8765

默认仅监听本机回环地址，所有图像只在请求内存中处理，不写上传缓存或遥测日志。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from screenrestore.diagnostics.logging_config import configure_logging
from screenrestore.io.image_exporter import ExportFormat, encode_image_bytes
from screenrestore.io.image_loader import ImageLoadError

from .multipart import parse_multipart
from .service import WebRestoreService

LOGGER = logging.getLogger(__name__)
STATIC_DIRECTORY = Path(__file__).with_name("static")
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class ScreenRestoreHTTPServer(ThreadingHTTPServer):
    """持有有界上传和并发处理配置的线程 HTTP 服务。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: WebRestoreService | None = None,
        max_body_bytes: int = 256 * 1024 * 1024,
        max_concurrent_jobs: int = 2,
    ) -> None:
        if max_body_bytes <= 0 or max_concurrent_jobs <= 0:
            raise ValueError("Web 服务限制必须大于 0")
        self.restore_service = service or WebRestoreService()
        self.max_body_bytes = max_body_bytes
        self.job_slots = threading.BoundedSemaphore(max_concurrent_jobs)
        super().__init__(server_address, ScreenRestoreRequestHandler)


class ScreenRestoreRequestHandler(BaseHTTPRequestHandler):
    """同源静态前端和版本化 JSON/PNG API。"""

    server: ScreenRestoreHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/v1/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "ScreenRestore",
                    "api_version": 1,
                    "privacy": "local-memory-only",
                },
            )
            return
        route = STATIC_ROUTES.get(path)
        if route is None:
            self._send_error(HTTPStatus.NOT_FOUND, "资源不存在")
            return
        filename, content_type = route
        try:
            data = (STATIC_DIRECTORY / filename).read_bytes()
        except OSError:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "前端静态资源缺失")
            return
        self._send_bytes(HTTPStatus.OK, data, content_type, cache_control="no-cache")

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path not in {"/api/v1/detect", "/api/v1/restore", "/api/v1/calibrate"}:
            self._send_error(HTTPStatus.NOT_FOUND, "接口不存在")
            return
        if not self.server.job_slots.acquire(blocking=False):
            self._send_error(HTTPStatus.TOO_MANY_REQUESTS, "处理任务已满，请稍后重试")
            return
        try:
            form = self._read_form()
            settings = _parse_settings(form.first("settings", "{}"))
            field_name = "calibration_files" if path.endswith("calibrate") else "files"
            uploads = form.files.get(field_name, [])
            images = self.server.restore_service.decode_uploads(uploads)
            if path.endswith("detect"):
                lens = settings.get("lens")
                if lens is not None and not isinstance(lens, dict):
                    raise ValueError("lens 必须是 JSON 对象")
                self._send_json(
                    HTTPStatus.OK,
                    self.server.restore_service.detect(images[0], lens, include_preview=True),
                )
            elif path.endswith("calibrate"):
                self._send_json(
                    HTTPStatus.OK,
                    self.server.restore_service.calibrate(images, settings),
                )
            else:
                result = self.server.restore_service.restore(images, settings)
                encoded = encode_image_bytes(result.image_rgb, ExportFormat.PNG)
                diagnostics = _diagnostics_header(result.diagnostics)
                self._send_bytes(
                    HTTPStatus.OK,
                    encoded,
                    "image/png",
                    extra_headers={"X-ScreenRestore-Diagnostics": diagnostics},
                )
        except RequestTooLargeError as exc:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
        except (ValueError, ImageLoadError, json.JSONDecodeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Web 请求处理失败")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"处理失败：{exc}")
        finally:
            self.server.job_slots.release()

    def _read_form(self):  # type: ignore[no-untyped-def]
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("请求缺少 Content-Length")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length 无效") from exc
        if length <= 0:
            raise ValueError("请求体为空")
        if length > self.server.max_body_bytes:
            raise RequestTooLargeError(
                f"请求体超过 {self.server.max_body_bytes // (1024 * 1024)} MiB 限制"
            )
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("请求体未完整接收")
        return parse_multipart(self.headers.get("Content-Type", ""), body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, data, "application/json; charset=utf-8")

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"status": "error", "error": message})

    def _send_bytes(
        self,
        status: HTTPStatus,
        data: bytes,
        content_type: str,
        *,
        cache_control: str = "no-store",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'",
        )
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format_string: str, *args: Any) -> None:
        """只记录方法/路径/状态，不记录请求体或图片内容。"""

        LOGGER.info("Web %s", format_string % args)


class RequestTooLargeError(ValueError):
    """上传请求超过服务配置上限。"""


def build_parser() -> argparse.ArgumentParser:
    """构建本地 Web 服务命令行参数。"""

    parser = argparse.ArgumentParser(description="启动 ScreenRestore 本地网页工具")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--open", action="store_true", help="启动后打开默认浏览器")
    parser.add_argument("--allow-remote", action="store_true", help="明确允许非回环地址")
    parser.add_argument("--max-upload-mb", type=int, default=256, help="单次请求上限 MiB")
    parser.add_argument("--max-jobs", type=int, default=2, help="最大并行处理任务数")
    parser.add_argument("--verbose", action="store_true")
    return parser


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    max_upload_mb: int = 256,
    max_jobs: int = 2,
) -> ScreenRestoreHTTPServer:
    """创建可由测试或命令行控制生命周期的服务实例。"""

    if not 0 <= port <= 65535:
        raise ValueError("端口必须位于 0..65535")
    if max_upload_mb <= 0:
        raise ValueError("上传上限必须大于 0")
    return ScreenRestoreHTTPServer(
        (host, port),
        max_body_bytes=max_upload_mb * 1024 * 1024,
        max_concurrent_jobs=max_jobs,
    )


def main(argv: list[str] | None = None) -> int:
    """持续运行本地服务，Ctrl+C 安全停止。"""

    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        raise SystemExit("非回环监听必须显式添加 --allow-remote")
    server = create_server(
        args.host,
        args.port,
        max_upload_mb=args.max_upload_mb,
        max_jobs=args.max_jobs,
    )
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{display_host}:{actual_port}/"
    LOGGER.info("ScreenRestore Web 已启动：%s", url)
    print(f"ScreenRestore Web：{url}")
    if args.open:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parse_settings(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("settings 必须是 JSON 对象")
    return value


def _diagnostics_header(diagnostics: dict[str, object]) -> str:
    data = json.dumps(diagnostics, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


if __name__ == "__main__":
    raise SystemExit(main())
