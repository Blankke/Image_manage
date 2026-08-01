"""Qt offscreen 主窗口烟雾测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from screenrestore.ui.main_window import MainWindow


def test_main_window_opens_unicode_image_without_running_event_loop(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    path = tmp_path / "界面输入.png"
    Image.fromarray(np.full((60, 100, 3), 127, np.uint8), "RGB").save(path)
    window = MainWindow()
    window.schedule_preview = lambda: None  # type: ignore[method-assign]

    assert window.open_path(path)
    assert window.document is not None
    assert window.document.width == 100
    assert window.operator_list.count() >= 10
    window.close()
    application.processEvents()

