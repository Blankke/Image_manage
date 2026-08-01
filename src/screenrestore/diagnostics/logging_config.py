"""不记录图片内容的跨平台滚动日志配置。"""

from __future__ import annotations

import logging
import os
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2

from screenrestore import __version__


def log_directory() -> Path:
    """Windows 使用 LOCALAPPDATA，其他系统使用 XDG state 约定。"""

    if local_app_data := os.getenv("LOCALAPPDATA"):
        return Path(local_app_data) / "ScreenRestore" / "logs"
    if state_home := os.getenv("XDG_STATE_HOME"):
        return Path(state_home) / "ScreenRestore" / "logs"
    return Path.home() / ".local" / "state" / "ScreenRestore" / "logs"


def configure_logging(verbose: bool = False) -> Path:
    """配置控制台和滚动文件日志，并记录运行环境版本。"""

    destination = log_directory()
    destination.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not any(isinstance(handler, RotatingFileHandler) for handler in root.handlers):
        file_handler = RotatingFileHandler(
            destination / "screenrestore.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(file_handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger(__name__).info(
        "ScreenRestore %s | OS=%s | Python=%s | OpenCV=%s | backend=CPU/OpenCV",
        __version__,
        platform.platform(),
        sys.version.split()[0],
        cv2.__version__,
    )
    return destination

