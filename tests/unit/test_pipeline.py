"""非破坏流水线、缓存、启用状态和历史测试。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel
from screenrestore.core.pipeline import (
    ImagePipeline,
    OperatorRegistry,
    OperatorState,
    PipelineHistory,
)


@dataclass
class OffsetParameters(ParameterModel):
    amount: int = 1


class CountingOffsetOperator(ImageOperator[OffsetParameters]):
    parameter_type = OffsetParameters

    def __init__(self, operator_id: str) -> None:
        self.id = operator_id
        self.display_name = operator_id
        self.calls = 0

    def default_parameters(self) -> OffsetParameters:
        return OffsetParameters()

    def apply(
        self,
        image: np.ndarray,
        params: OffsetParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        self.calls += 1
        context.report(0.5, f"{self.id} half")
        return np.clip(image + params.amount / 255.0, 0.0, 1.0).astype(np.float32)


def _pipeline() -> tuple[ImagePipeline, CountingOffsetOperator, CountingOffsetOperator]:
    first = CountingOffsetOperator("first")
    second = CountingOffsetOperator("second")
    pipeline = ImagePipeline(
        [
            OperatorState(first, True, OffsetParameters(1)),
            OperatorState(second, True, OffsetParameters(2)),
        ]
    )
    return pipeline, first, second


def test_pipeline_cache_and_downstream_invalidation() -> None:
    pipeline, first, second = _pipeline()
    source = np.zeros((10, 10, 3), np.uint8)

    initial = pipeline.process(source, source_id="stable")
    cached = pipeline.process(source, source_id="stable")
    pipeline.update_parameters("second", {"amount": 5})
    changed = pipeline.process(source, source_id="stable")

    assert round(float(initial[0, 0, 0]) * 255) == 3
    assert cached is initial
    assert round(float(changed[0, 0, 0]) * 255) == 6
    assert first.calls == 1
    assert second.calls == 2


def test_disabling_operator_skips_it() -> None:
    pipeline, first, second = _pipeline()
    pipeline.set_enabled("first", False)
    result = pipeline.process(np.zeros((4, 4, 3), np.uint8), source_id="disabled")
    assert round(float(result[0, 0, 0]) * 255) == 2
    assert first.calls == 0
    assert second.calls == 1


def test_operator_progress_is_mapped_to_pipeline_fraction() -> None:
    pipeline, _, _ = _pipeline()
    events: list[tuple[float, str]] = []
    pipeline.process(
        np.zeros((4, 4, 3), np.uint8),
        ProcessingContext(progress=lambda fraction, message: events.append((fraction, message))),
        source_id="progress",
    )
    fractions = [round(fraction, 2) for fraction, _ in events]
    assert fractions == [0.25, 0.5, 0.75, 1.0]


def test_pipeline_serialization_and_undo_redo() -> None:
    pipeline, first, second = _pipeline()
    registry = OperatorRegistry([first, second])
    history = PipelineHistory(pipeline, registry)

    pipeline.update_parameters("first", {"amount": 8})
    history.checkpoint()
    assert history.undo()
    assert pipeline.state("first").params.to_dict()["amount"] == 1
    assert history.redo()
    assert pipeline.state("first").params.to_dict()["amount"] == 8
