"""反光包含/排除多边形蒙版编辑器。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .image_canvas import rgb_to_qimage


class MaskDrawingWidget(QWidget):
    """在适配图像上以归一化坐标绘制包含/排除多边形。"""

    def __init__(
        self,
        image_rgb: np.ndarray,
        include_polygons: list[list[list[float]]],
        exclude_polygons: list[list[list[float]]],
        parent=None,  # type: ignore[no-untyped-def]
    ) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap.fromImage(rgb_to_qimage(image_rgb))
        self.include_polygons = [[list(point) for point in polygon] for polygon in include_polygons]
        self.exclude_polygons = [[list(point) for point in polygon] for polygon in exclude_polygons]
        self.current: list[list[float]] = []
        self.mode = "include"
        self.setMinimumSize(640, 420)

    def finish_polygon(self) -> None:
        """把至少三个点的当前路径提交到当前模式。"""

        if len(self.current) >= 3:
            target = self.include_polygons if self.mode == "include" else self.exclude_polygons
            target.append(self.current)
        self.current = []
        self.update()

    def undo_point(self) -> None:
        """撤销当前路径最后一个点。"""

        if self.current:
            self.current.pop()
            self.update()

    def clear_mode(self) -> None:
        """清空当前包含或排除多边形。"""

        target = self.include_polygons if self.mode == "include" else self.exclude_polygons
        target.clear()
        self.current = []
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        del event
        painter = QPainter(self)
        target = self._target_rect()
        painter.drawPixmap(target.toRect(), self._pixmap)
        self._draw_polygons(painter, self.include_polygons, QColor(255, 70, 40, 150), target)
        self._draw_polygons(painter, self.exclude_polygons, QColor(50, 210, 255, 150), target)
        if self.current:
            color = QColor(255, 70, 40) if self.mode == "include" else QColor(50, 210, 255)
            painter.setPen(QPen(color, 2))
            painter.drawPolyline(self._polygon(self.current, target))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        point = self._normalized(event.position())
        if point is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.current.append(point)
            if event.type() == QMouseEvent.Type.MouseButtonDblClick:
                self.finish_polygon()
        elif event.button() == Qt.MouseButton.RightButton:
            self.undo_point()
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        point = self._normalized(event.position())
        if point is not None and (not self.current or self.current[-1] != point):
            self.current.append(point)
        self.finish_polygon()
        event.accept()

    def _draw_polygons(
        self,
        painter: QPainter,
        polygons: list[list[list[float]]],
        color: QColor,
        target: QRectF,
    ) -> None:
        painter.setPen(QPen(color, 2))
        brush = QColor(color)
        brush.setAlpha(45)
        painter.setBrush(brush)
        for polygon in polygons:
            painter.drawPolygon(self._polygon(polygon, target))

    def _polygon(self, points: list[list[float]], target: QRectF) -> QPolygonF:
        return QPolygonF(
            [
                QPointF(target.left() + x * target.width(), target.top() + y * target.height())
                for x, y in points
            ]
        )

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
            float(np.clip((position.x() - target.left()) / target.width(), 0, 1)),
            float(np.clip((position.y() - target.top()) / target.height(), 0, 1)),
        ]


class ReflectionMaskEditor(QDialog):
    """让用户手工画入/画出反光蒙版。"""

    def __init__(
        self,
        image_rgb: np.ndarray,
        include_polygons: list[list[list[float]]],
        exclude_polygons: list[list[list[float]]],
        parent=None,  # type: ignore[no-untyped-def]
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("反光蒙版编辑")
        self.resize(860, 650)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("左键添加顶点，双击或“完成多边形”闭合；红色画入，蓝色画出，右键撤销点。")
        )
        self.canvas = MaskDrawingWidget(image_rgb, include_polygons, exclude_polygons)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("画入蒙版", "include")
        self.mode_combo.addItem("画出蒙版", "exclude")
        self.mode_combo.currentIndexChanged.connect(
            lambda _index: setattr(self.canvas, "mode", str(self.mode_combo.currentData()))
        )
        finish = QPushButton("完成多边形")
        finish.clicked.connect(self.canvas.finish_polygon)
        undo = QPushButton("撤销顶点")
        undo.clicked.connect(self.canvas.undo_point)
        clear = QPushButton("清空当前模式")
        clear.clicked.connect(self.canvas.clear_mode)
        controls.addWidget(self.mode_combo)
        controls.addWidget(finish)
        controls.addWidget(undo)
        controls.addWidget(clear)
        controls.addStretch(1)
        layout.addLayout(controls)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_with_polygon)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_with_polygon(self) -> None:
        self.canvas.finish_polygon()
        self.accept()

