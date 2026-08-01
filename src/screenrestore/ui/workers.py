"""后台流水线 worker，禁止直接修改 QWidget。"""

from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from screenrestore.core.cache import PipelineCache
from screenrestore.core.cancellation import CancellationToken, ProcessingCancelled
from screenrestore.core.operator import ProcessingContext
from screenrestore.core.pipeline import ImagePipeline, OperatorRegistry
from screenrestore.io.image_exporter import ExportOptions, export_image


class WorkerSignals(QObject):
    """后台任务的线程安全信号。"""

    result = Signal(int, object, object)
    error = Signal(int, str, str)
    progress = Signal(int, float, str)
    cancelled = Signal(int)
    finished = Signal(int)


class PipelineWorker(QRunnable):
    """从序列化快照创建独立流水线并处理一张 RGB 图像。"""

    def __init__(
        self,
        generation: int,
        image_rgb: np.ndarray,
        pipeline_data: dict[str, object],
        registry: OperatorRegistry,
        token: CancellationToken,
        source_id: str,
        preview: bool,
        cache: PipelineCache | None = None,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.image_rgb = image_rgb
        self.pipeline_data = pipeline_data
        self.registry = registry
        self.token = token
        self.source_id = source_id
        self.preview = preview
        self.cache = cache
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """执行任务，所有异常经信号传回主线程。"""

        metadata: dict[str, object] = {}
        try:
            pipeline = ImagePipeline.from_dict(self.pipeline_data, self.registry, self.cache)
            context = ProcessingContext(
                cancellation=self.token,
                progress=lambda fraction, message: self.signals.progress.emit(
                    self.generation, fraction, message
                ),
                preview=self.preview,
                metadata=metadata,
            )
            result = pipeline.process(self.image_rgb, context, self.source_id)
            metadata["timings"] = pipeline.last_timings
            self.signals.result.emit(self.generation, result, metadata)
        except ProcessingCancelled:
            self.signals.cancelled.emit(self.generation)
        except Exception as exc:  # noqa: BLE001 - worker 边界必须收拢异常
            self.signals.error.emit(self.generation, str(exc), traceback.format_exc())
        finally:
            self.signals.finished.emit(self.generation)


class ExportSignals(QObject):
    """全分辨率导出任务信号。"""

    result = Signal(int, str, object)
    error = Signal(int, str, str)
    progress = Signal(int, float, str)
    cancelled = Signal(int)
    finished = Signal(int)


class ExportWorker(QRunnable):
    """后台从原图重新运行流水线并编码，不依赖预览缓存。"""

    def __init__(
        self,
        generation: int,
        image_rgb: np.ndarray,
        pipeline_data: dict[str, object],
        registry: OperatorRegistry,
        token: CancellationToken,
        source_id: str,
        output_path: str,
        options: ExportOptions,
        source_path: str,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.image_rgb = image_rgb
        self.pipeline_data = pipeline_data
        self.registry = registry
        self.token = token
        self.source_id = source_id
        self.output_path = output_path
        self.options = options
        self.source_path = source_path
        self.signals = ExportSignals()

    @Slot()
    def run(self) -> None:
        """处理原始分辨率并原子导出。"""

        metadata: dict[str, object] = {}
        try:
            pipeline = ImagePipeline.from_dict(self.pipeline_data, self.registry)
            context = ProcessingContext(
                cancellation=self.token,
                progress=lambda fraction, message: self.signals.progress.emit(
                    self.generation, fraction * 0.93, message
                ),
                preview=False,
                metadata=metadata,
            )
            result = pipeline.process(self.image_rgb, context, self.source_id)
            self.token.check()
            self.signals.progress.emit(self.generation, 0.96, "编码输出图像")
            destination = export_image(
                result,
                self.output_path,
                self.options,
                self.source_path,
            )
            self.token.check()
            metadata["timings"] = pipeline.last_timings
            self.signals.result.emit(self.generation, str(destination), metadata)
        except ProcessingCancelled:
            self.signals.cancelled.emit(self.generation)
        except Exception as exc:  # noqa: BLE001 - worker 边界必须收拢异常
            self.signals.error.emit(self.generation, str(exc), traceback.format_exc())
        finally:
            self.signals.finished.emit(self.generation)
