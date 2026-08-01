"""支持缩放、平移和像素读取的 RGB 图像画布。"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def rgb_to_qimage(image_rgb: np.ndarray) -> QImage:
    """把连续 RGB uint8 数组复制为脱离 NumPy 生命周期的 QImage。"""

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("需要 H×W×3 的 RGB uint8 图像")
    contiguous = np.ascontiguousarray(image_rgb)
    height, width = contiguous.shape[:2]
    return QImage(
        contiguous.data,
        width,
        height,
        int(contiguous.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()


class ImageCanvas(QGraphicsView):
    """以场景坐标对应图像像素坐标的图片画布。"""

    pixelHovered = Signal(int, int, object, object)
    zoomChanged = Signal(float)
    imageClicked = Signal(int, int)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._image_rgb: np.ndarray | None = None
        self._zoom = 1.0
        self._panning = False
        self._pick_mode = False
        self._last_pan = QPointF()
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(Qt.GlobalColor.black)

    @property
    def image_rgb(self) -> np.ndarray | None:
        """当前显示的 RGB 图像；调用者不得原地修改。"""

        return self._image_rgb

    def set_image(self, image_rgb: np.ndarray | None, fit: bool = True) -> None:
        """替换显示图像，可选择立即适应窗口。"""

        self._image_rgb = image_rgb
        if image_rgb is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(QRectF())
            return
        qimage = rgb_to_qimage(image_rgb)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self._scene.setSceneRect(QRectF(0, 0, qimage.width(), qimage.height()))
        if fit:
            self.fit_image()

    def fit_image(self) -> None:
        """让完整图像适应当前视口。"""

        if self._pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoomChanged.emit(self._zoom)

    def set_zoom_percent(self, percent: int) -> None:
        """设置相对原始像素的绝对缩放百分比。"""

        factor = max(0.05, min(16.0, percent / 100.0))
        self.resetTransform()
        self.scale(factor, factor)
        self._zoom = factor
        self.zoomChanged.emit(factor)

    def set_pick_mode(self, enabled: bool) -> None:
        """启用下一次左键像素点选，点选后自动恢复浏览。"""

        self._pick_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif not self._panning:
            self.unsetCursor()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._image_rgb is None:
            return super().wheelEvent(event)
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        target = self._zoom * factor
        if 0.05 <= target <= 16.0:
            self.scale(factor, factor)
            self._zoom = target
            self.zoomChanged.emit(target)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        if self._pick_mode and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            x, y = int(point.x()), int(point.y())
            if self._image_rgb is not None:
                height, width = self._image_rgb.shape[:2]
                if 0 <= x < width and 0 <= y < height:
                    self.imageClicked.emit(x, y)
            self.set_pick_mode(False)
            event.accept()
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = True
            self._last_pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        if self._panning:
            delta = event.position() - self._last_pan
            self._last_pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
        self._emit_pixel(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _emit_pixel(self, viewport_pos: QPointF) -> None:
        """读取图像像素并同时发送 RGB/HSV，失败时静默跳过。"""

        if self._image_rgb is None:
            return
        point = self.mapToScene(viewport_pos.toPoint())
        x, y = int(point.x()), int(point.y())
        height, width = self._image_rgb.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        rgb = tuple(int(value) for value in self._image_rgb[y, x])
        pixel = np.array([[rgb]], dtype=np.uint8)
        hsv = tuple(int(value) for value in cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0])
        self.pixelHovered.emit(x, y, rgb, hsv)
