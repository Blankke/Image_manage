"""场景预设选择面板。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from screenrestore.core.presets import PRESET_NAMES, PresetId


class PresetPanel(QWidget):
    """只改变默认算子开关和参数，不锁定手动覆盖。"""

    presetChanged = Signal(str)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QLabel("场景预设"))
        self.combo = QComboBox()
        for preset, name in PRESET_NAMES.items():
            self.combo.addItem(name, preset.value)
        layout.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(
            lambda _index: self.presetChanged.emit(str(self.combo.currentData()))
        )

    def set_preset(self, preset: PresetId) -> None:
        """不发信号地同步当前预设。"""

        index = self.combo.findData(preset.value)
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(index)
        self.combo.blockSignals(False)

