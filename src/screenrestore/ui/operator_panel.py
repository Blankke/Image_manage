"""算子列表与基于 dataclass 参数模型生成的编辑面板。"""

from __future__ import annotations

from dataclasses import fields
from enum import StrEnum
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from screenrestore.core.pipeline import ImagePipeline, OperatorState

PARAMETER_LABELS = {
    "mode": "模式",
    "strength": "强度",
    "radius": "半径",
    "amount": "数量",
    "threshold": "阈值",
    "exposure": "曝光（EV）",
    "gamma": "Gamma",
    "contrast": "对比度",
    "highlights": "高光",
    "shadows": "阴影",
    "blacks": "黑色",
    "whites": "白色",
    "saturation": "饱和度",
    "temperature": "色温",
    "tint": "色调",
    "clip_limit": "Clip limit",
    "tile_grid_size": "Tile 网格",
    "ratio_mode": "输出比例",
    "custom_ratio": "自定义比例",
    "rotation": "旋转角度",
    "black_border": "黑边（像素）",
    "interpolation": "插值",
    "auto_crop": "自动裁无效边",
    "direction": "方向",
    "smooth_scale": "平滑尺度",
    "max_correction": "最大校正",
    "show_curve": "显示估计曲线",
    "show_field": "显示照明场",
    "show_mask": "显示蒙版",
    "auto_frequency": "自动频域峰（实验）",
    "notch_radius": "陷波半径",
    "notch_depth": "陷波深度",
    "edge_protection": "边缘保护",
    "heat_threshold": "热度阈值",
    "manifest_path": "模型清单路径",
    "broad_haze_strength": "宽光幕强度",
    "broad_haze_scale": "宽光幕尺度",
    "black_level_quantile": "黑位分位数",
    "max_haze_correction": "最大光幕校正",
    "gradient_threshold": "DCT 梯度阈值",
    "smoothness_lambda": "DCT 平滑项",
    "curvature_weight": "DCT 曲率项",
}


class OperatorList(QListWidget):
    """支持勾选和拖动排序的算子列表。"""

    enabledChanged = Signal(str, bool)
    orderChanged = Signal(object)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._refreshing = False
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.itemChanged.connect(self._on_item_changed)
        self.model().rowsMoved.connect(self._on_rows_moved)

    def set_pipeline(self, pipeline: ImagePipeline) -> None:
        """根据流水线重建列表。"""

        self._refreshing = True
        self.clear()
        for state in pipeline.states:
            item = QListWidgetItem(state.operator.display_name)
            item.setData(Qt.ItemDataRole.UserRole, state.operator.id)
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if not state.operator.reorderable:
                flags &= ~Qt.ItemFlag.ItemIsDragEnabled
            item.setFlags(flags)
            item.setCheckState(
                Qt.CheckState.Checked if state.enabled else Qt.CheckState.Unchecked
            )
            self.addItem(item)
        self._refreshing = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if not self._refreshing:
            self.enabledChanged.emit(
                str(item.data(Qt.ItemDataRole.UserRole)),
                item.checkState() == Qt.CheckState.Checked,
            )

    def _on_rows_moved(self, *args: object) -> None:
        if self._refreshing:
            return
        order = [
            str(self.item(index).data(Qt.ItemDataRole.UserRole)) for index in range(self.count())
        ]
        self.orderChanged.emit(order)


class OperatorPanel(QWidget):
    """根据参数模型创建类型安全的输入控件。"""

    parametersChanged = Signal(str, object)
    editCornersRequested = Signal()
    resetCornersRequested = Signal()
    redetectCornersRequested = Signal()
    pickNeutralRequested = Signal()
    editReflectionMaskRequested = Signal()

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._operator_id = ""
        self._values: dict[str, Any] = {}
        self._body = QWidget()
        self._form = QFormLayout(self._body)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._body)
        layout = QVBoxLayout(self)
        self.title = QLabel("选择处理步骤")
        self.title.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(self.title)
        layout.addWidget(scroll, 1)

    def set_state(self, state: OperatorState | None) -> None:
        """显示指定算子参数；不支持直接编辑的复合字段会给出说明。"""

        while self._form.rowCount():
            self._form.removeRow(0)
        if state is None:
            self._operator_id = ""
            self.title.setText("选择处理步骤")
            return
        self._operator_id = state.operator.id
        self.title.setText(state.operator.display_name)
        self._values = state.params.to_dict()
        if state.operator.id == "geometry":
            edit = QPushButton("进入/退出四角编辑")
            reset = QPushButton("重置四角")
            detect = QPushButton("重新自动检测")
            edit.clicked.connect(self.editCornersRequested)
            reset.clicked.connect(self.resetCornersRequested)
            detect.clicked.connect(self.redetectCornersRequested)
            self._form.addRow(edit)
            self._form.addRow(reset)
            self._form.addRow(detect)
        if state.operator.id == "white_balance":
            pick_neutral = QPushButton("在图像上点选中性灰")
            pick_neutral.clicked.connect(self.pickNeutralRequested)
            self._form.addRow(pick_neutral)
        if state.operator.id == "reflection":
            edit_mask = QPushButton("绘制反光包含/排除蒙版")
            edit_mask.clicked.connect(self.editReflectionMaskRequested)
            self._form.addRow(edit_mask)
        for model_field in fields(state.params):
            name = model_field.name
            if name in {"corners", "manual_notches", "include_polygons", "exclude_polygons"}:
                continue
            value = getattr(state.params, name)
            widget = self._make_widget(name, value)
            self._form.addRow(PARAMETER_LABELS.get(name, name.replace("_", " ")), widget)
        if state.operator.id == "demoire":
            self._form.addRow(QLabel("手工陷波点可在频谱窗口中增加或删除。"))
        if state.operator.id == "reflection":
            self._form.addRow(QLabel("大面积反光只能抑制，无法真实恢复已丢失内容。"))

    def _make_widget(self, name: str, value: Any) -> QWidget:
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            widget.toggled.connect(lambda checked, key=name: self._update(key, checked))
            return widget
        if isinstance(value, StrEnum):
            combo = QComboBox()
            for member in type(value):
                combo.addItem(member.value, member.value)
            combo.setCurrentIndex(combo.findData(value.value))
            combo.currentIndexChanged.connect(
                lambda _index, key=name, control=combo: self._update(key, control.currentData())
            )
            return combo
        if name == "rotation":
            combo = QComboBox()
            for angle in (0, 90, 180, 270):
                combo.addItem(str(angle), angle)
            combo.setCurrentIndex(combo.findData(value))
            combo.currentIndexChanged.connect(
                lambda _index, key=name, control=combo: self._update(key, control.currentData())
            )
            return combo
        if isinstance(value, int):
            spin = QSpinBox()
            minimum, maximum = _integer_range(name)
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.valueChanged.connect(lambda changed, key=name: self._update(key, changed))
            return spin
        if isinstance(value, float):
            spin = QDoubleSpinBox()
            minimum, maximum, step = _float_range(name)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(4 if abs(step) < 0.01 else 3)
            spin.setValue(value)
            spin.valueChanged.connect(lambda changed, key=name: self._update(key, changed))
            return spin
        if isinstance(value, str):
            edit = QLineEdit(value)
            edit.setClearButtonEnabled(True)
            edit.editingFinished.connect(
                lambda key=name, control=edit: self._update(key, control.text())
            )
            return edit
        label = QLabel(str(value))
        label.setWordWrap(True)
        return label

    def _update(self, name: str, value: Any) -> None:
        self._values[name] = value
        self.parametersChanged.emit(self._operator_id, dict(self._values))


def _integer_range(name: str) -> tuple[int, int]:
    return {
        "rotation": (0, 270),
        "black_border": (0, 4096),
        "tile_grid_size": (2, 32),
        "radius": (8, 600),
        "motion_length": (1, 99),
        "max_width": (1, 65535),
        "max_height": (1, 65535),
    }.get(name, (0, 1000))


def _float_range(name: str) -> tuple[float, float, float]:
    ranges = {
        "exposure": (-4.0, 4.0, 0.1),
        "gamma": (0.2, 5.0, 0.05),
        "contrast": (-1.0, 1.0, 0.05),
        "highlights": (-1.0, 1.0, 0.05),
        "shadows": (-1.0, 1.0, 0.05),
        "blacks": (-1.0, 1.0, 0.05),
        "whites": (-1.0, 1.0, 0.05),
        "saturation": (-1.0, 1.0, 0.05),
        "temperature": (-1.0, 1.0, 0.05),
        "tint": (-1.0, 1.0, 0.05),
        "strength": (0.0, 1.0, 0.05),
        "amount": (0.0, 3.0, 0.05),
        "threshold": (0.0, 0.9, 0.01),
        "custom_ratio": (0.2, 5.0, 0.01),
        "smooth_scale": (4.0, 400.0, 2.0),
        "max_correction": (0.01, 0.5, 0.01),
        "broad_haze_strength": (0.0, 2.0, 0.05),
        "broad_haze_scale": (12.0, 600.0, 2.0),
        "black_level_quantile": (0.01, 0.3, 0.01),
        "max_haze_correction": (0.01, 0.35, 0.01),
        "gradient_threshold": (0.0, 0.13, 0.005),
        "smoothness_lambda": (0.0, 1.0, 0.05),
        "curvature_weight": (0.05, 1.0, 0.05),
        "clip_limit": (0.1, 8.0, 0.1),
        "notch_radius": (1.0, 50.0, 1.0),
        "notch_depth": (0.0, 1.0, 0.05),
        "heat_threshold": (0.0, 0.9, 0.05),
    }
    return ranges.get(name, (-10.0, 1000.0, 0.1))
