"""直方图和可交互频谱诊断对话框。"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from screenrestore.diagnostics.histogram import render_histogram
from screenrestore.operators.demoire import (
    detect_frequency_peaks,
    frequency_spectrum,
    moire_heatmap,
)

from .image_canvas import rgb_to_qimage


class SpectrumWidget(QWidget):
    """左键增加、右键删除最近 Gaussian notch 点。"""

    def __init__(self, spectrum_rgb: np.ndarray, points: list[list[float]], parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._pixmap = QPixmap.fromImage(rgb_to_qimage(spectrum_rgb))
        self.points = [list(point) for point in points]
        self.setMinimumSize(520, 360)

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        del event
        painter = QPainter(self)
        target = self._target_rect()
        painter.drawPixmap(target.toRect(), self._pixmap)
        painter.setPen(QPen(QColor(255, 70, 40), 2))
        for x, y in self.points:
            point = QPointF(target.left() + x * target.width(), target.top() + y * target.height())
            painter.drawEllipse(point, 7, 7)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        normalized = self._normalized(event.position())
        if normalized is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.points.append(normalized)
        elif event.button() == Qt.MouseButton.RightButton and self.points:
            distances = [np.linalg.norm(np.asarray(point) - normalized) for point in self.points]
            nearest = int(np.argmin(distances))
            if distances[nearest] < 0.08:
                self.points.pop(nearest)
        self.update()

    def _target_rect(self) -> QRectF:
        image_ratio = self._pixmap.width() / max(1, self._pixmap.height())
        widget_ratio = self.width() / max(1, self.height())
        if widget_ratio > image_ratio:
            height = float(self.height())
            width = height * image_ratio
        else:
            width = float(self.width())
            height = width / image_ratio
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _normalized(self, position: QPointF) -> list[float] | None:
        target = self._target_rect()
        if not target.contains(position):
            return None
        return [
            float((position.x() - target.left()) / target.width()),
            float((position.y() - target.top()) / target.height()),
        ]


class DiagnosticsDialog(QDialog):
    """显示直方图、摩尔纹热图和频谱，并返回用户陷波点。"""

    def __init__(self, image_rgb: np.ndarray, notches: list[list[float]], parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.image_rgb = image_rgb
        self.setWindowTitle("图像诊断：直方图 / 摩尔纹热图 / 频谱")
        self.resize(820, 600)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        histogram = QLabel()
        histogram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        histogram.setPixmap(QPixmap.fromImage(rgb_to_qimage(render_histogram(image_rgb))))
        tabs.addTab(histogram, "直方图")
        heat = (moire_heatmap(image_rgb) * 255).astype(np.uint8)
        heat_color_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        heat_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
        heat_label = QLabel()
        heat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heat_label.setPixmap(QPixmap.fromImage(rgb_to_qimage(heat_rgb)))
        tabs.addTab(heat_label, "摩尔纹热图")
        self.spectrum = SpectrumWidget(frequency_spectrum(image_rgb), notches)
        spectrum_page = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_page)
        spectrum_layout.addWidget(QLabel("左键增加陷波点，右键删除最近点；滤波会自动加入中心对称点。"))
        spectrum_layout.addWidget(self.spectrum, 1)
        controls = QHBoxLayout()
        auto_button = QPushButton("检测异常峰")
        auto_button.clicked.connect(self._auto_peaks)
        clear_button = QPushButton("清空陷波点")
        clear_button.clicked.connect(self._clear_points)
        controls.addWidget(auto_button)
        controls.addWidget(clear_button)
        controls.addStretch(1)
        spectrum_layout.addLayout(controls)
        tabs.addTab(spectrum_page, "频谱/陷波")
        layout.addWidget(tabs, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def notch_points(self) -> list[list[float]]:
        """用户当前设置的归一化陷波点。"""

        return [list(point) for point in self.spectrum.points]

    def _auto_peaks(self) -> None:
        luminance = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        self.spectrum.points = detect_frequency_peaks(luminance)
        self.spectrum.update()

    def _clear_points(self) -> None:
        self.spectrum.points.clear()
        self.spectrum.update()

