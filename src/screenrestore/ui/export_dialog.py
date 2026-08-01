"""全分辨率导出参数对话框。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from screenrestore.io.image_exporter import ExportFormat, ExportOptions


@dataclass(slots=True)
class ExportDialogValues:
    """导出对话框返回值。"""

    path: Path
    options: ExportOptions
    limit_resolution: bool
    max_width: int
    max_height: int
    add_black_border: bool
    black_border: int


class ExportDialog(QDialog):
    """设置编码、元数据、输出包围盒和可选黑边。"""

    def __init__(self, suggested_path: Path, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("导出全分辨率结果")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        path_container = QWidget()
        path_layout = QHBoxLayout(path_container)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit(str(suggested_path))
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse)
        form.addRow("输出文件", path_container)

        self.format_combo = QComboBox()
        for export_format in ExportFormat:
            self.format_combo.addItem(export_format.value, export_format.value)
        self.format_combo.setCurrentIndex(self.format_combo.findData("PNG"))
        form.addRow("格式", self.format_combo)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(92)
        form.addRow("JPEG/WebP 质量", self.quality_spin)
        self.keep_exif = QCheckBox("保留 EXIF")
        self.keep_exif.setChecked(True)
        form.addRow(self.keep_exif)
        self.remove_gps = QCheckBox("删除 GPS（推荐）")
        self.remove_gps.setChecked(True)
        form.addRow(self.remove_gps)

        self.limit_resolution = QCheckBox("限制输出分辨率（不放大）")
        form.addRow(self.limit_resolution)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 65535)
        self.width_spin.setValue(3840)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 65535)
        self.height_spin.setValue(2160)
        form.addRow("最大宽度", self.width_spin)
        form.addRow("最大高度", self.height_spin)
        self.add_border = QCheckBox("添加黑边")
        form.addRow(self.add_border)
        self.border_spin = QSpinBox()
        self.border_spin.setRange(0, 4096)
        self.border_spin.setValue(0)
        form.addRow("黑边（像素）", self.border_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> ExportDialogValues:
        """返回验证后的导出设置。"""

        path = Path(self.path_edit.text()).expanduser()
        export_format = ExportFormat(str(self.format_combo.currentData()))
        if not path.suffix:
            extension = {
                ExportFormat.PNG: ".png",
                ExportFormat.JPEG: ".jpg",
                ExportFormat.WEBP: ".webp",
                ExportFormat.TIFF: ".tiff",
            }[export_format]
            path = path.with_suffix(extension)
        return ExportDialogValues(
            path=path,
            options=ExportOptions(
                format=export_format,
                quality=self.quality_spin.value(),
                keep_exif=self.keep_exif.isChecked(),
                remove_gps=self.remove_gps.isChecked(),
                overwrite=False,
            ),
            limit_resolution=self.limit_resolution.isChecked(),
            max_width=self.width_spin.value(),
            max_height=self.height_spin.value(),
            add_black_border=self.add_border.isChecked(),
            black_border=self.border_spin.value(),
        )

    def _browse(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            self.path_edit.text(),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        self.path_edit.setText(path)
        format_by_filter = {
            "PNG (*.png)": ExportFormat.PNG,
            "JPEG (*.jpg *.jpeg)": ExportFormat.JPEG,
            "WebP (*.webp)": ExportFormat.WEBP,
            "TIFF (*.tif *.tiff)": ExportFormat.TIFF,
        }
        if export_format := format_by_filter.get(selected_filter):
            self.format_combo.setCurrentIndex(self.format_combo.findData(export_format.value))

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        # Enter 仅由按钮盒确认，避免正在编辑路径时误触发。
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            event.accept()
            return
        super().keyPressEvent(event)

