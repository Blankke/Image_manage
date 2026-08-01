"""GUI 应用入口。"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from screenrestore.diagnostics.logging_config import configure_logging
from screenrestore.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    """启动 ScreenRestore GUI，可从命令行附带初始图片。"""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("image", nargs="?")
    args, qt_args = parser.parse_known_args(argv)
    configure_logging()
    application = QApplication([sys.argv[0], *qt_args])
    application.setApplicationName("ScreenRestore")
    window = MainWindow(args.image)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
