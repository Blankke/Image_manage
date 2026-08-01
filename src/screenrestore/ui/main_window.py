"""ScreenRestore 主窗口与非阻塞预览协调器。"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from screenrestore.core.cancellation import CancellationToken
from screenrestore.core.image_document import ImageDocument
from screenrestore.core.pipeline import PipelineHistory
from screenrestore.core.presets import (
    PRESET_NAMES,
    PresetId,
    apply_preset,
    build_default_pipeline,
    build_registry,
)
from screenrestore.io.image_loader import ImageLoadError, load_image
from screenrestore.io.project_file import (
    PROJECT_SUFFIX,
    ProjectFileError,
    load_project,
    relocate_source,
    save_project,
    verify_project_source,
)
from screenrestore.operators.geometry import detect_quadrilaterals

from .compare_view import CompareView
from .diagnostics_dialog import DiagnosticsDialog
from .export_dialog import ExportDialog
from .operator_panel import OperatorList, OperatorPanel
from .preset_panel import PresetPanel
from .reflection_mask_editor import ReflectionMaskEditor
from .settings_dialog import SettingsDialog
from .workers import ExportWorker, PipelineWorker

LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """协调项目状态、后台预览和 PySide6 界面，不承载图像算法。"""

    def __init__(self, initial_path: str | None = None) -> None:
        super().__init__()
        self.document: ImageDocument | None = None
        self.registry = build_registry()
        self.pipeline = build_default_pipeline(self.registry)
        self.history = PipelineHistory(self.pipeline, self.registry)
        self.current_preset = PresetId.DISPLAY
        self._thread_pool = QThreadPool.globalInstance()
        self._token: CancellationToken | None = None
        self._worker: PipelineWorker | None = None
        self._generation = 0
        self._started_at: dict[int, float] = {}
        self._closing = False
        self._corner_editing = False
        self._project_path: Path | None = None
        self._last_metadata: dict[str, object] = {}
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._start_preview)
        self.setWindowTitle("ScreenRestore")
        self.resize(1440, 900)
        self.setAcceptDrops(True)
        self._build_ui()
        self._refresh_pipeline_ui()
        if initial_path:
            QTimer.singleShot(0, lambda: self.open_path(initial_path))

    def _build_ui(self) -> None:
        self.compare_view = CompareView()
        self.operator_list = OperatorList()
        self.operator_panel = OperatorPanel()
        self.preset_panel = PresetPanel()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(self.preset_panel)
        left_layout.addWidget(QLabel("处理步骤（可勾选、拖动排序）"))
        left_layout.addWidget(self.operator_list, 1)
        left.setMinimumWidth(230)
        left.setMaximumWidth(330)
        self.operator_panel.setMinimumWidth(285)
        self.operator_panel.setMaximumWidth(390)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.compare_view)
        splitter.addWidget(self.operator_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 900, 310])
        self.setCentralWidget(splitter)

        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        open_action = QAction("打开", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_dialog)
        toolbar.addAction(open_action)
        open_project_action = QAction("打开项目", self)
        open_project_action.triggered.connect(self.open_project_dialog)
        toolbar.addAction(open_project_action)
        save_project_action = QAction("保存项目", self)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_project_action.triggered.connect(self.save_project_dialog)
        toolbar.addAction(save_project_action)
        export_action = QAction("导出", self)
        export_action.triggered.connect(self.export_dialog)
        toolbar.addAction(export_action)
        settings_action = QAction("设置/模型插件", self)
        settings_action.triggered.connect(self.open_settings)
        toolbar.addAction(settings_action)
        diagnostics_action = QAction("直方图/频谱", self)
        diagnostics_action.triggered.connect(self.open_diagnostics)
        toolbar.addAction(diagnostics_action)
        toolbar.addSeparator()
        undo_action = QAction("撤销", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)
        redo_action = QAction("重做", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)
        toolbar.addSeparator()
        auto_action = QAction("自动恢复", self)
        auto_action.triggered.connect(self.auto_restore)
        toolbar.addAction(auto_action)
        compare_action = QAction("左右对比", self)
        compare_action.setCheckable(True)
        compare_action.toggled.connect(self.compare_view.set_compare_enabled)
        toolbar.addAction(compare_action)
        toolbar.addSeparator()
        fit_action = QAction("Fit", self)
        fit_action.triggered.connect(self.compare_view.fit_image)
        toolbar.addAction(fit_action)
        for label, percent in (("100%", 100), ("200%", 200)):
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked=False, value=percent: self.compare_view.set_zoom_percent(value)
            )
            toolbar.addAction(action)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(180)
        self.progress.hide()
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.hide()
        self.pixel_label = QLabel("RGB: -   HSV: -")
        self.dimension_label = QLabel("未打开图像")
        self.backend_label = QLabel("后端：CPU / OpenCV")
        self.statusBar().addWidget(self.progress)
        self.statusBar().addWidget(self.cancel_button)
        self.statusBar().addPermanentWidget(self.pixel_label)
        self.statusBar().addPermanentWidget(self.dimension_label)
        self.statusBar().addPermanentWidget(self.backend_label)

        self.operator_list.currentItemChanged.connect(self._select_operator)
        self.operator_list.enabledChanged.connect(self._set_operator_enabled)
        self.operator_list.orderChanged.connect(self._reorder_operators)
        self.operator_panel.parametersChanged.connect(self._update_parameters)
        self.operator_panel.editCornersRequested.connect(self.toggle_corner_editing)
        self.operator_panel.resetCornersRequested.connect(self.reset_corners)
        self.operator_panel.redetectCornersRequested.connect(self.detect_corners)
        self.operator_panel.pickNeutralRequested.connect(self.start_neutral_pick)
        self.operator_panel.editReflectionMaskRequested.connect(self.edit_reflection_mask)
        self.preset_panel.presetChanged.connect(self._apply_preset)
        self.compare_view.editor.cornersChanged.connect(self._on_corners_changed)
        for canvas in (
            self.compare_view.editor,
            self.compare_view.original_canvas,
            self.compare_view.result_canvas,
        ):
            canvas.pixelHovered.connect(self._show_pixel)
        self.compare_view.editor.imageClicked.connect(self._on_neutral_picked)

    def open_dialog(self) -> None:
        """显示受支持格式的文件选择器。"""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开屏幕照片",
            "",
            "图像 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        """加载指定图像并启动代理恢复。"""

        try:
            document = load_image(path)
        except ImageLoadError as exc:
            LOGGER.exception("加载图像失败：%s", path)
            QMessageBox.critical(self, "无法打开图像", str(exc))
            return False
        self.cancel_processing()
        self.document = document
        self._project_path = None
        self.pipeline = build_default_pipeline(self.registry)
        self.history = PipelineHistory(self.pipeline, self.registry)
        self.current_preset = PresetId.DISPLAY
        self.preset_panel.set_preset(self.current_preset)
        proxy = document.proxy()
        self.compare_view.set_images(proxy)
        self.dimension_label.setText(f"{document.width} × {document.height}")
        self.setWindowTitle(f"ScreenRestore — {document.path.name}")
        self._refresh_pipeline_ui()
        gib = document.estimated_working_bytes / (1024**3)
        if gib >= 1.0:
            QMessageBox.warning(
                self,
                "大图像内存提示",
                f"全分辨率处理预计至少需要 {gib:.1f} GiB 内存；预览仍使用代理图。",
            )
        self.statusBar().showMessage("图像已加载，正在生成离线预览", 3000)
        self.schedule_preview()
        return True

    def open_project_dialog(self) -> None:
        """打开项目文件，并在源图缺失时允许用户重新定位。"""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 ScreenRestore 项目",
            "",
            f"ScreenRestore 项目 (*{PROJECT_SUFFIX})",
        )
        if not path:
            return
        try:
            loaded = load_project(path, self.registry)
            if not loaded.source_path.is_file():
                relocated, _ = QFileDialog.getOpenFileName(
                    self,
                    "项目原图缺失，请重新定位",
                    str(loaded.path.parent),
                    "图像 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
                )
                if not relocated:
                    return
                relocate_source(loaded, relocated)
            document = load_image(loaded.source_path)
            warnings = verify_project_source(loaded, document)
        except (ProjectFileError, ImageLoadError) as exc:
            LOGGER.exception("打开项目失败：%s", path)
            QMessageBox.critical(self, "无法打开项目", str(exc))
            return
        self.cancel_processing()
        self.document = document
        self.pipeline = loaded.pipeline
        self.history = PipelineHistory(self.pipeline, self.registry)
        self.current_preset = loaded.preset
        self._project_path = loaded.path
        self.preset_panel.set_preset(loaded.preset)
        self.compare_view.set_images(document.proxy())
        self._refresh_pipeline_ui()
        self.dimension_label.setText(f"{document.width} × {document.height}")
        self.setWindowTitle(f"ScreenRestore — {loaded.path.name}")
        if warnings:
            QMessageBox.warning(self, "项目源图警告", "\n".join(warnings))
        self.schedule_preview()

    def save_project_dialog(self) -> None:
        """保存当前非破坏状态，不写入任何图像内容。"""

        if self.document is None:
            return
        suggested = self._project_path or self.document.path.with_name(
            self.document.path.stem + PROJECT_SUFFIX
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 ScreenRestore 项目",
            str(suggested),
            f"ScreenRestore 项目 (*{PROJECT_SUFFIX})",
        )
        if not path:
            return
        try:
            self._project_path = save_project(
                path,
                self.document,
                self.pipeline,
                self.current_preset,
            )
        except ProjectFileError as exc:
            LOGGER.exception("保存项目失败")
            QMessageBox.critical(self, "保存项目失败", str(exc))
            return
        self.statusBar().showMessage(f"项目已保存：{self._project_path}", 5000)

    def export_dialog(self) -> None:
        """收集导出设置并启动全分辨率后台任务。"""

        if self.document is None:
            return
        suggested = self.document.path.with_name(self.document.path.stem + "_restored.png")
        dialog = ExportDialog(suggested, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if values.path.exists():
            answer = QMessageBox.question(
                self,
                "确认覆盖",
                f"文件已存在，是否覆盖？\n{values.path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            values.options.overwrite = True
        pipeline_data = self.pipeline.to_dict()
        for state in pipeline_data["operators"]:
            if state["id"] == "resize":
                state["enabled"] = values.limit_resolution
                state["params"] = {
                    "mode": "fit" if values.limit_resolution else "original",
                    "scale": 1.0,
                    "max_width": values.max_width,
                    "max_height": values.max_height,
                }
            elif state["id"] == "geometry":
                state["params"]["black_border"] = (
                    values.black_border if values.add_black_border else 0
                )
        self.cancel_processing()
        self._generation += 1
        generation = self._generation
        token = CancellationToken()
        self._token = token
        worker = ExportWorker(
            generation,
            self.document.original_rgb,
            pipeline_data,
            self.registry,
            token,
            self.document.content_hash + ":full",
            str(values.path),
            values.options,
            str(self.document.path),
        )
        self._worker = worker  # type: ignore[assignment]
        worker.signals.progress.connect(self._on_progress)
        worker.signals.result.connect(self._on_export_result)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_worker_finished)
        self._started_at[generation] = time.perf_counter()
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        self.statusBar().showMessage("正在处理原始分辨率图像")
        self._thread_pool.start(worker)

    def open_settings(self) -> None:
        """显示本地模型清单和可用性，不访问网络。"""

        if getattr(sys, "frozen", False):
            models_directory = Path(sys.executable).resolve().parent / "models"
        else:
            models_directory = Path(__file__).resolve().parents[3] / "models"
        SettingsDialog(models_directory, self).exec()

    def open_diagnostics(self) -> None:
        """打开直方图、摩尔纹热图和可点击频谱陷波。"""

        image = self.compare_view.result_image
        if image is None:
            return
        params = self.pipeline.state("demoire").params.to_dict()
        dialog = DiagnosticsDialog(image, params.get("manual_notches", []), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params["manual_notches"] = dialog.notch_points
        if dialog.notch_points:
            params["mode"] = "frequency_experimental"
            params["auto_frequency"] = False
        self.pipeline.update_parameters("demoire", params)
        self.history.checkpoint()
        self._mark_custom()
        self._refresh_pipeline_ui()
        self.schedule_preview()

    def start_neutral_pick(self) -> None:
        """让下一次结果画布点击成为中性灰样本。"""

        if self.document is None:
            return
        if self._corner_editing:
            self.toggle_corner_editing(force=False)
        self.compare_view.editor.set_pick_mode(True)
        self.statusBar().showMessage("请在图像中点击应为中性灰的区域", 5000)

    def _on_neutral_picked(self, x: int, y: int) -> None:
        image = self.compare_view.editor.image_rgb
        if image is None:
            return
        params = self.pipeline.state("white_balance").params.to_dict()
        params.update(
            {
                "mode": "neutral_point",
                "neutral_x": x / max(1, image.shape[1] - 1),
                "neutral_y": y / max(1, image.shape[0] - 1),
            }
        )
        self.pipeline.update_parameters("white_balance", params)
        self.history.checkpoint()
        self._mark_custom()
        self._refresh_pipeline_ui()
        self.schedule_preview()

    def edit_reflection_mask(self) -> None:
        """编辑反光手工画入/画出多边形。"""

        image = self.compare_view.result_image
        if image is None:
            return
        params = self.pipeline.state("reflection").params.to_dict()
        dialog = ReflectionMaskEditor(
            image,
            params.get("include_polygons", []),
            params.get("exclude_polygons", []),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params["include_polygons"] = dialog.canvas.include_polygons
        params["exclude_polygons"] = dialog.canvas.exclude_polygons
        self.pipeline.update_parameters("reflection", params)
        self.pipeline.set_enabled("reflection", True)
        self.history.checkpoint()
        self._mark_custom()
        self._refresh_pipeline_ui()
        self.schedule_preview()

    def schedule_preview(self) -> None:
        """使用 debounce 合并连续参数修改。"""

        if self.document is not None and not self._corner_editing:
            self._preview_timer.start()

    def _start_preview(self) -> None:
        if self.document is None or self._closing:
            return
        self.cancel_processing()
        self._generation += 1
        generation = self._generation
        token = CancellationToken()
        self._token = token
        proxy = self.document.proxy()
        source_id = f"{self.document.content_hash}:preview:{proxy.shape[1]}x{proxy.shape[0]}"
        worker = PipelineWorker(
            generation,
            proxy,
            self.pipeline.to_dict(),
            self.registry,
            token,
            source_id,
            True,
            self.pipeline.cache,
        )
        self._worker = worker
        worker.signals.progress.connect(self._on_progress)
        worker.signals.result.connect(self._on_preview_result)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_worker_finished)
        self._started_at[generation] = time.perf_counter()
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        self._thread_pool.start(worker)

    def cancel_processing(self) -> None:
        """请求当前任务协作式取消。"""

        if self._token is not None:
            self._token.cancel()
            self._token = None

    def auto_restore(self) -> None:
        """自动检测四边形并运行当前预设经典流水线。"""

        if self.document is None:
            return
        self.detect_corners()
        self.schedule_preview()

    def detect_corners(self) -> None:
        """在代理图上寻找最佳四边形；失败时保留手工编辑能力。"""

        if self.document is None:
            return
        proxy = self.document.proxy()
        candidates = detect_quadrilaterals(proxy)
        if not candidates:
            self.statusBar().showMessage("未检测到可靠四边形，可进入四角编辑手动调整", 5000)
            self.toggle_corner_editing(force=True)
            return
        self.compare_view.editor.set_corners(candidates[0].corners, emit=True)
        self.statusBar().showMessage(
            f"检测到 {len(candidates)} 个候选，已采用置信度 {candidates[0].confidence:.0%} 的候选",
            5000,
        )

    def toggle_corner_editing(self, force: bool | None = None) -> None:
        """在普通浏览和手动四角编辑间切换。"""

        if self.document is None:
            return
        enabled = (not self._corner_editing) if force is None else force
        self._corner_editing = enabled
        self.compare_view.set_corner_editing(enabled)
        if enabled:
            params = self.pipeline.state("geometry").params.to_dict()
            normalized = np.asarray(params["corners"], dtype=np.float32)
            proxy = self.document.proxy()
            corners = normalized * np.array([proxy.shape[1] - 1, proxy.shape[0] - 1])
            self.compare_view.editor.set_corners(corners)
            self.statusBar().showMessage("四角编辑模式：拖动控制点，方向键微调，Shift 为 10 像素")
        else:
            self.schedule_preview()

    def reset_corners(self) -> None:
        """重置四角为完整代理图边界。"""

        if self.document is None:
            return
        if not self._corner_editing:
            self.toggle_corner_editing(force=True)
        self.compare_view.editor.reset_corners(emit=True)

    def undo(self) -> None:
        """撤销最近一次参数、开关、预设或排序修改。"""

        if self.history.undo():
            self._refresh_pipeline_ui()
            self.schedule_preview()

    def redo(self) -> None:
        """重做最近撤销的修改。"""

        if self.history.redo():
            self._refresh_pipeline_ui()
            self.schedule_preview()

    def _select_operator(self, current, previous) -> None:  # type: ignore[no-untyped-def]
        del previous
        if current is None:
            self.operator_panel.set_state(None)
            return
        operator_id = str(current.data(Qt.ItemDataRole.UserRole))
        self.operator_panel.set_state(self.pipeline.state(operator_id))

    def _set_operator_enabled(self, operator_id: str, enabled: bool) -> None:
        self.pipeline.set_enabled(operator_id, enabled)
        self.history.checkpoint()
        self._mark_custom()
        self.schedule_preview()

    def _update_parameters(self, operator_id: str, values: object) -> None:
        if not isinstance(values, dict):
            return
        try:
            self.pipeline.update_parameters(operator_id, values)
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(f"参数无效：{exc}", 4000)
            return
        self.history.checkpoint()
        self._mark_custom()
        self.schedule_preview()

    def _reorder_operators(self, order: object) -> None:
        if not isinstance(order, list):
            return
        current = {state.operator.id: state for state in self.pipeline.states}
        if set(order) != set(current):
            self._refresh_pipeline_ui()
            return
        fixed = {
            index: state.operator.id
            for index, state in enumerate(self.pipeline.states)
            if not state.operator.reorderable
        }
        if any(order[index] != operator_id for index, operator_id in fixed.items()):
            self.statusBar().showMessage("方向、镜头、几何、网格和输出步骤的位置固定", 4000)
            self._refresh_pipeline_ui()
            return
        self.pipeline.states = [current[operator_id] for operator_id in order]
        self.pipeline.cache.clear()
        self.history.checkpoint()
        self._mark_custom()
        self.schedule_preview()

    def _apply_preset(self, preset_value: str) -> None:
        preset = PresetId(preset_value)
        if preset == PresetId.CUSTOM:
            return
        apply_preset(self.pipeline, preset)
        self.current_preset = preset
        self.history.checkpoint()
        self._refresh_pipeline_ui()
        self.schedule_preview()
        self.statusBar().showMessage(f"已应用{PRESET_NAMES[preset]}预设；所有参数仍可手动覆盖", 3500)

    def _mark_custom(self) -> None:
        self.current_preset = PresetId.CUSTOM
        self.preset_panel.set_preset(PresetId.CUSTOM)

    def _on_corners_changed(self, corners: object) -> None:
        if self.document is None:
            return
        values = np.asarray(corners, dtype=np.float32)
        proxy = self.document.proxy()
        normalized = values / np.array([proxy.shape[1] - 1, proxy.shape[0] - 1])
        params = self.pipeline.state("geometry").params.to_dict()
        params["corners"] = np.clip(normalized, 0.0, 1.0).tolist()
        self.pipeline.update_parameters("geometry", params)
        self.history.checkpoint()
        self._mark_custom()

    def _refresh_pipeline_ui(self) -> None:
        selected_id = None
        if current := self.operator_list.currentItem():
            selected_id = current.data(Qt.ItemDataRole.UserRole)
        self.operator_list.set_pipeline(self.pipeline)
        if selected_id:
            for index in range(self.operator_list.count()):
                item = self.operator_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == selected_id:
                    self.operator_list.setCurrentItem(item)
                    break
        elif self.operator_list.count():
            self.operator_list.setCurrentRow(0)

    def _on_progress(self, generation: int, fraction: float, message: str) -> None:
        if generation == self._generation:
            self.progress.setValue(round(fraction * 1000))
            self.statusBar().showMessage(message)

    def _on_preview_result(self, generation: int, result: object, metadata: object) -> None:
        if generation != self._generation or not isinstance(result, np.ndarray):
            return
        self._last_metadata = metadata if isinstance(metadata, dict) else {}
        timings = self._last_metadata.get("timings", {})
        if isinstance(timings, dict):
            LOGGER.info("预览算子耗时：%s", timings)
        display_source = (
            np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)
            if result.dtype == np.float32
            else result
        )
        display_result = self._diagnostic_display(display_source, self._last_metadata)
        self.compare_view.update_result(display_result)
        self.dimension_label.setText(f"预览 {result.shape[1]} × {result.shape[0]}")
        elapsed = time.perf_counter() - self._started_at.get(generation, time.perf_counter())
        self.statusBar().showMessage(f"预览完成，用时 {elapsed:.2f} 秒", 4000)

    def _diagnostic_display(
        self,
        result: np.ndarray,
        metadata: dict[str, object],
    ) -> np.ndarray:
        """按显式参数把照明场、反光蒙版或条带曲线显示在预览中。"""

        illumination_params = self.pipeline.state("illumination").params.to_dict()
        field = metadata.get("illumination_field")
        if illumination_params.get("show_field") and isinstance(field, np.ndarray):
            resized = cv2.resize(field.astype(np.float32), (result.shape[1], result.shape[0]))
            gray = np.clip(np.rint(resized * 255), 0, 255).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        reflection_params = self.pipeline.state("reflection").params.to_dict()
        mask = metadata.get("reflection_mask")
        if reflection_params.get("show_mask") and isinstance(mask, np.ndarray):
            resized = cv2.resize(mask.astype(np.float32), (result.shape[1], result.shape[0]))
            overlay = result.astype(np.float32)
            color = np.zeros_like(overlay)
            color[..., 0] = 255
            alpha = np.clip(resized[..., None] * 0.65, 0, 0.65)
            return np.clip(overlay * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        banding_params = self.pipeline.state("banding").params.to_dict()
        banding = metadata.get("banding")
        if banding_params.get("show_curve") and isinstance(banding, dict):
            gain = np.asarray(banding.get("gain", []), dtype=np.float32)
            if gain.size >= 2:
                chart = np.full_like(result, 22)
                x_values = np.linspace(0, result.shape[1] - 1, gain.size)
                low, high = float(gain.min()), float(gain.max())
                normalized = (gain - low) / max(1e-6, high - low)
                y_values = (result.shape[0] - 20) - normalized * (result.shape[0] - 40)
                points = np.column_stack((x_values, y_values)).astype(np.int32)
                cv2.polylines(chart, [points], False, (80, 210, 255), 2, cv2.LINE_AA)
                return chart
        return result

    def _on_worker_error(self, generation: int, message: str, details: str) -> None:
        if generation != self._generation:
            return
        LOGGER.error("预览处理失败：%s\n%s", message, details)
        QMessageBox.critical(self, "处理失败", f"无法生成预览：{message}\n详细信息已写入日志。")

    def _on_export_result(self, generation: int, destination: str, metadata: object) -> None:
        if generation != self._generation:
            return
        elapsed = time.perf_counter() - self._started_at.get(generation, time.perf_counter())
        timings = metadata.get("timings", {}) if isinstance(metadata, dict) else {}
        LOGGER.info("导出完成 path=%s elapsed=%.3f timings=%s", destination, elapsed, timings)
        self.progress.setValue(1000)
        QMessageBox.information(self, "导出完成", f"已导出：\n{destination}\n用时 {elapsed:.2f} 秒")

    def _on_worker_finished(self, generation: int) -> None:
        self._started_at.pop(generation, None)
        if generation == self._generation:
            self.progress.hide()
            self.cancel_button.hide()
            self._worker = None

    def _show_pixel(self, x: int, y: int, rgb: object, hsv: object) -> None:
        self.pixel_label.setText(f"({x}, {y}) RGB {rgb}  HSV {hsv}")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.compare_view.show_temporary_original(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.compare_view.show_temporary_original(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def] # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())
            event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """关闭窗口前请求任务取消，worker 不接触已销毁 UI。"""

        self._closing = True
        self._preview_timer.stop()
        self.cancel_processing()
        event.accept()
