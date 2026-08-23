"""可缩放画布上的手动四角编辑器。"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeyEvent, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPolygonItem, QLabel

from screenrestore.geometry import order_corners

from .image_canvas import ImageCanvas, rgb_to_qimage


class InteractionMode(StrEnum):
    """画布交互模式，防止浏览时误拖角点。"""

    BROWSE = "browse"
    CORNERS = "corners"


class CornerEditor(ImageCanvas):
    """提供四个可拖控制点、覆盖区域、键盘微调与局部放大镜。"""

    cornersChanged = Signal(object)
    modeChanged = Signal(str)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._mode = InteractionMode.BROWSE
        self._corners = np.zeros((4, 2), dtype=np.float32)
        self._selected: int | None = None
        self._polygon = QGraphicsPolygonItem()
        self._polygon.setPen(QPen(QColor(50, 180, 255), 2))
        self._polygon.setBrush(QBrush(QColor(50, 180, 255, 45)))
        self._polygon.setZValue(10)
        self._scene.addItem(self._polygon)
        self._handles: list[QGraphicsEllipseItem] = []
        for _ in range(4):
            handle = QGraphicsEllipseItem(-7, -7, 14, 14)
            handle.setPen(QPen(Qt.GlobalColor.white, 2))
            handle.setBrush(QBrush(QColor(30, 145, 255)))
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            handle.setZValue(11)
            self._scene.addItem(handle)
            self._handles.append(handle)
        self._magnifier = QLabel(self.viewport())
        self._magnifier.setFixedSize(148, 148)
        self._magnifier.setStyleSheet("border: 2px solid white; background: black;")
        self._magnifier.hide()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._set_overlay_visible(False)

    @property
    def mode(self) -> InteractionMode:
        """当前浏览或四角编辑模式。"""

        return self._mode

    @property
    def corners(self) -> np.ndarray:
        """返回当前图像像素坐标的四角副本。"""

        return self._corners.copy()

    def set_image(self, image_rgb: np.ndarray | None, fit: bool = True) -> None:
        """替换图像并在尺寸变化时重置为完整边界。"""

        super().set_image(image_rgb, fit)
        if image_rgb is not None:
            self.reset_corners(emit=False)
        else:
            self._set_overlay_visible(False)

    def set_mode(self, mode: InteractionMode) -> None:
        """明确切换浏览和四角编辑模式。"""

        self._mode = InteractionMode(mode)
        visible = self._mode == InteractionMode.CORNERS and self.image_rgb is not None
        self._set_overlay_visible(visible)
        self._magnifier.hide()
        self.modeChanged.emit(self._mode.value)
        self.setCursor(
            Qt.CursorShape.CrossCursor if visible else Qt.CursorShape.ArrowCursor
        )

    def set_corners(self, corners: np.ndarray, emit: bool = False) -> None:
        """设置、排序并约束图像像素坐标角点。"""

        if self.image_rgb is None:
            return
        height, width = self.image_rgb.shape[:2]
        ordered = order_corners(corners)
        ordered[:, 0] = np.clip(ordered[:, 0], 0, width - 1)
        ordered[:, 1] = np.clip(ordered[:, 1], 0, height - 1)
        self._corners = ordered
        self._update_overlay()
        if emit:
            self.cornersChanged.emit(self.corners)

    def reset_corners(self, emit: bool = True) -> None:
        """把角点重置到图像四边。"""

        if self.image_rgb is None:
            return
        height, width = self.image_rgb.shape[:2]
        self.set_corners(
            np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32),
            emit=emit,
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        if self._mode != InteractionMode.CORNERS or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        positions = np.array(
            [
                [self.mapFromScene(QPointF(float(point[0]), float(point[1]))).x(),
                 self.mapFromScene(QPointF(float(point[0]), float(point[1]))).y()]
                for point in self._corners
            ],
            dtype=np.float32,
        )
        cursor = np.array([event.position().x(), event.position().y()], dtype=np.float32)
        distances = np.linalg.norm(positions - cursor, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 18:
            self._selected = nearest
            self.setFocus()
            self._move_selected(self.mapToScene(event.position().toPoint()), emit=False)
            event.accept()
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        if self._mode == InteractionMode.CORNERS and self._selected is not None:
            scene_point = self.mapToScene(event.position().toPoint())
            self._move_selected(scene_point, emit=False)
            self._show_magnifier(self._corners[self._selected], event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        if self._mode == InteractionMode.CORNERS and self._selected is not None:
            self._selected = None
            self._magnifier.hide()
            self.set_corners(self._corners, emit=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self._mode == InteractionMode.CORNERS and self._selected is not None:
            delta = {
                Qt.Key.Key_Left: (-1, 0),
                Qt.Key.Key_Right: (1, 0),
                Qt.Key.Key_Up: (0, -1),
                Qt.Key.Key_Down: (0, 1),
            }.get(event.key())
            if delta is not None:
                step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
                point = self._corners[self._selected] + np.array(delta) * step
                self._move_selected(QPointF(float(point[0]), float(point[1])), emit=True)
                event.accept()
                return
        super().keyPressEvent(event)

    def _move_selected(self, point: QPointF, emit: bool) -> None:
        if self.image_rgb is None or self._selected is None:
            return
        height, width = self.image_rgb.shape[:2]
        self._corners[self._selected] = (
            np.clip(point.x(), 0, width - 1),
            np.clip(point.y(), 0, height - 1),
        )
        self._update_overlay()
        if emit:
            self.cornersChanged.emit(self.corners)

    def _update_overlay(self) -> None:
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in self._corners])
        self._polygon.setPolygon(polygon)
        for handle, point in zip(self._handles, self._corners, strict=True):
            handle.setPos(float(point[0]), float(point[1]))

    def _set_overlay_visible(self, visible: bool) -> None:
        self._polygon.setVisible(visible)
        for handle in self._handles:
            handle.setVisible(visible)

    def _show_magnifier(self, point: np.ndarray, viewport_position: QPointF) -> None:
        """显示角点附近 40×40 像素的放大预览。"""

        if self.image_rgb is None:
            return
        x, y = (int(round(value)) for value in point)
        radius = 20
        padded = cv2.copyMakeBorder(
            self.image_rgb,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_REFLECT_101,
        )
        crop = padded[y : y + radius * 2 + 1, x : x + radius * 2 + 1]
        enlarged = cv2.resize(crop, (144, 144), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(enlarged, (72, 72), (255, 60, 30), cv2.MARKER_CROSS, 18, 2)
        self._magnifier.setPixmap(QPixmap.fromImage(rgb_to_qimage(enlarged)))
        position = viewport_position.toPoint() + QPoint(18, 18)
        self._magnifier.move(
            min(position.x(), max(0, self.viewport().width() - 154)),
            min(position.y(), max(0, self.viewport().height() - 154)),
        )
        self._magnifier.show()
