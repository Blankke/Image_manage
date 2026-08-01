"""单图、左右对比和临时原图显示容器。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QStackedLayout, QWidget

from .corner_editor import CornerEditor, InteractionMode
from .image_canvas import ImageCanvas


class CompareView(QWidget):
    """在编辑画布与可拖分割条的左右对比之间切换。"""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.editor = CornerEditor()
        self.original_canvas = ImageCanvas()
        self.result_canvas = ImageCanvas()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.original_canvas)
        self.splitter.addWidget(self.result_canvas)
        self.splitter.setChildrenCollapsible(False)
        self._layout = QStackedLayout(self)
        self._layout.addWidget(self.editor)
        self._layout.addWidget(self.splitter)
        self._original: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._temporary_original = False

    def set_images(self, original: np.ndarray, result: np.ndarray | None = None) -> None:
        """设置原图代理和当前结果。"""

        self._original = original
        self._result = result if result is not None else original
        self.original_canvas.set_image(original)
        self.result_canvas.set_image(self._result)
        self.editor.set_image(self._result)

    def update_result(self, result: np.ndarray) -> None:
        """只更新结果，保留原图和当前对比模式。"""

        self._result = result
        self.result_canvas.set_image(result, fit=False)
        if self.editor.mode == InteractionMode.BROWSE and not self._temporary_original:
            self.editor.set_image(result, fit=False)

    @property
    def result_image(self) -> np.ndarray | None:
        """当前结果代理，只读使用。"""

        return self._result

    def set_compare_enabled(self, enabled: bool) -> None:
        """启用左右并排；中间 QSplitter 手柄可拖动。"""

        self._layout.setCurrentIndex(1 if enabled else 0)
        if enabled:
            self.original_canvas.fit_image()
            self.result_canvas.fit_image()

    def set_corner_editing(self, enabled: bool) -> None:
        """编辑模式始终显示未变形原图，退出后恢复结果。"""

        self._layout.setCurrentIndex(0)
        if enabled:
            if self._original is not None:
                self.editor.set_image(self._original)
            self.editor.set_mode(InteractionMode.CORNERS)
        else:
            self.editor.set_mode(InteractionMode.BROWSE)
            if self._result is not None:
                self.editor.set_image(self._result)

    def show_temporary_original(self, active: bool) -> None:
        """按住快捷键时临时用原图替代单图结果。"""

        if self.editor.mode == InteractionMode.CORNERS:
            return
        self._temporary_original = active
        image = self._original if active else self._result
        if image is not None:
            self.editor.set_image(image, fit=False)

    def fit_image(self) -> None:
        """适配所有可见画布。"""

        if self._layout.currentIndex() == 1:
            self.original_canvas.fit_image()
            self.result_canvas.fit_image()
        else:
            self.editor.fit_image()

    def set_zoom_percent(self, percent: int) -> None:
        """设置所有画布的缩放百分比。"""

        for canvas in (self.editor, self.original_canvas, self.result_canvas):
            canvas.set_zoom_percent(percent)
