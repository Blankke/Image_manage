"""可选模型插件发现与状态界面。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from screenrestore.inference.external_process import ExternalProcessBackend
from screenrestore.inference.model_manifest import ModelManifest, discover_manifests
from screenrestore.inference.onnx_backend import OnnxBackend
from screenrestore.inference.openvino_backend import OpenVinoBackend


class SettingsDialog(QDialog):
    """扫描本地 JSON 清单并显示后端可用性，不下载任何文件。"""

    def __init__(self, models_directory: Path, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("设置与可选模型插件")
        self.resize(680, 420)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "模型插件完全可选。ScreenRestore 不联网下载权重；请按 MODEL_PLUGINS.md "
            "自行安装后把清单放入 models/。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)
        manifests, errors = discover_manifests(models_directory)
        for manifest in manifests:
            available, reason = _availability(manifest)
            status = "可用" if available else "不可用"
            item = QListWidgetItem(
                f"{manifest.name} [{manifest.role.value}/{manifest.task}/{manifest.type}] — {status}\n"
                f"许可证：{manifest.license}  {reason}"
            )
            item.setData(Qt.ItemDataRole.UserRole, manifest.id)
            self.list_widget.addItem(item)
        for error in errors:
            self.list_widget.addItem(QListWidgetItem(f"清单错误：{error}"))
        if not manifests and not errors:
            self.list_widget.addItem(QListWidgetItem("未安装可选模型；经典 CPU 流水线可正常使用。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _availability(manifest: ModelManifest) -> tuple[bool, str]:
    if manifest.type == "external_process":
        return ExternalProcessBackend(manifest).is_available()
    if manifest.type == "onnx":
        return OnnxBackend(manifest).is_available()
    return OpenVinoBackend(manifest).is_available()
