"""可序列化、可重排并带节点缓存的非破坏式流水线。"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache import CacheKey, PipelineCache
from .operator import ImageOperator, ProcessingContext
from .parameters import ParameterModel


@dataclass(slots=True)
class OperatorState:
    """流水线中一个算子的启用状态和参数。"""

    operator: ImageOperator[Any]
    enabled: bool
    params: ParameterModel

    def to_dict(self) -> dict[str, Any]:
        """序列化节点状态。"""

        return {
            "id": self.operator.id,
            "enabled": self.enabled,
            "params": self.params.to_dict(),
        }


class OperatorRegistry:
    """由算子 ID 查找唯一实现，项目加载和 CLI 共用。"""

    def __init__(self, operators: list[ImageOperator[Any]] | None = None) -> None:
        self._operators: dict[str, ImageOperator[Any]] = {}
        for operator in operators or []:
            self.register(operator)

    def register(self, operator: ImageOperator[Any]) -> None:
        """注册算子，重复 ID 会被拒绝。"""

        if operator.id in self._operators:
            raise ValueError(f"重复算子 ID：{operator.id}")
        self._operators[operator.id] = operator

    def get(self, operator_id: str) -> ImageOperator[Any]:
        """获取算子，不存在时给出可读错误。"""

        try:
            return self._operators[operator_id]
        except KeyError as exc:
            raise ValueError(f"未知算子：{operator_id}") from exc

    @property
    def ids(self) -> tuple[str, ...]:
        """所有已注册算子 ID。"""

        return tuple(self._operators)


class ImagePipeline:
    """按节点顺序处理 RGB 图像，并复用未受影响的中间结果。"""

    def __init__(
        self,
        states: list[OperatorState],
        cache: PipelineCache | None = None,
    ) -> None:
        ids = [state.operator.id for state in states]
        if len(ids) != len(set(ids)):
            raise ValueError("同一流水线不能包含重复算子")
        self.states = states
        self.cache = cache or PipelineCache()
        self.last_timings: dict[str, float] = {}

    def process(
        self,
        image_rgb: np.ndarray,
        context: ProcessingContext | None = None,
        source_id: str | None = None,
    ) -> np.ndarray:
        """把加载图转换一次 float32，再以该精度处理到末端。"""

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("流水线输入必须是 H×W×3 RGB 图像")
        if image_rgb.dtype == np.uint8:
            current = np.ascontiguousarray(image_rgb.astype(np.float32) / 255.0)
        elif image_rgb.dtype == np.float32:
            current = np.ascontiguousarray(image_rgb.copy())
        else:
            raise ValueError("流水线输入仅支持 RGB uint8 或 float32")
        _validate_working_image(current, "流水线输入")
        active_context = context or ProcessingContext()
        resolved_source = source_id or _image_signature(image_rgb)
        enabled_states = [state for state in self.states if state.enabled]
        total_cost = sum(state.operator.estimate_cost(image_rgb.shape) for state in enabled_states) or 1.0
        completed_cost = 0.0
        cumulative = resolved_source
        self.last_timings = {}

        for index, state in enumerate(self.states):
            active_context.cancellation.check()
            state_signature = _state_signature(state)
            cumulative = hashlib.sha256(f"{cumulative}:{state_signature}".encode()).hexdigest()
            if not state.enabled:
                continue
            key = CacheKey(resolved_source, index, cumulative)
            cached = self.cache.get(key)
            cost = state.operator.estimate_cost(current.shape)
            if cached is not None:
                current = cached
                completed_cost += cost
                active_context.report(
                    completed_cost / total_cost,
                    f"{state.operator.display_name}（缓存）",
                )
                self.last_timings[state.operator.id] = 0.0
                continue
            state.operator.validate(state.params)
            started = time.perf_counter()

            def report_node(
                fraction: float,
                message: str,
                node_start: float = completed_cost,
                node_cost: float = cost,
            ) -> None:
                """把算子内部进度映射到整条流水线，保持 UI 可响应。"""

                active_context.report(
                    (node_start + node_cost * float(np.clip(fraction, 0.0, 1.0)))
                    / total_cost,
                    message,
                )

            node_context = ProcessingContext(
                cancellation=active_context.cancellation,
                progress=report_node,
                preview=active_context.preview,
                metadata=active_context.metadata,
            )
            result = state.operator.apply(current, state.params, node_context)
            try:
                _validate_working_image(result, f"算子 {state.operator.id} 输出")
            except ValueError as exc:
                raise TypeError(
                    f"算子 {state.operator.id} 违反 RGB float32 [0,1] 输出契约"
                ) from exc
            active_context.cancellation.check()
            elapsed = time.perf_counter() - started
            self.last_timings[state.operator.id] = elapsed
            self.cache.put(key, result)
            current = self.cache.get(key)
            assert current is not None
            completed_cost += cost
            active_context.report(completed_cost / total_cost, state.operator.display_name)
        return current

    def state(self, operator_id: str) -> OperatorState:
        """按 ID 获取节点状态。"""

        for state in self.states:
            if state.operator.id == operator_id:
                return state
        raise ValueError(f"流水线中不存在算子：{operator_id}")

    def set_enabled(self, operator_id: str, enabled: bool) -> None:
        """启用/禁用算子，并仅淘汰当前及下游缓存。"""

        index = self._index(operator_id)
        self.states[index].enabled = bool(enabled)
        self.cache.invalidate_from(index)

    def update_parameters(self, operator_id: str, values: dict[str, Any]) -> None:
        """一次性替换算子参数并使下游缓存失效。"""

        index = self._index(operator_id)
        state = self.states[index]
        state.params = state.operator.parameter_type.from_dict(values)
        state.operator.validate(state.params)
        self.cache.invalidate_from(index)

    def move(self, operator_id: str, new_index: int) -> None:
        """把允许重排的算子移动到另一个可重排槽位。"""

        old_index = self._index(operator_id)
        state = self.states[old_index]
        if not state.operator.reorderable:
            raise ValueError(f"算子 {operator_id} 的位置固定")
        if not 0 <= new_index < len(self.states):
            raise IndexError("目标位置超出流水线")
        if not self.states[new_index].operator.reorderable:
            raise ValueError("不能占用固定算子的位置")
        self.states.pop(old_index)
        self.states.insert(new_index, state)
        self.cache.invalidate_from(min(old_index, new_index))

    def to_dict(self) -> dict[str, Any]:
        """序列化完整流水线。"""

        return {"operators": [state.to_dict() for state in self.states]}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry: OperatorRegistry,
        cache: PipelineCache | None = None,
    ) -> ImagePipeline:
        """从项目数据恢复流水线。"""

        raw_states = data.get("operators")
        if not isinstance(raw_states, list):
            raise ValueError("项目缺少 operators 列表")
        states: list[OperatorState] = []
        for raw in raw_states:
            if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
                raise ValueError("算子节点格式无效")
            operator = registry.get(raw["id"])
            params_data = raw.get("params", {})
            if not isinstance(params_data, dict):
                raise ValueError(f"算子 {operator.id} 的参数格式无效")
            params = operator.parameter_type.from_dict(params_data)
            operator.validate(params)
            states.append(OperatorState(operator, bool(raw.get("enabled", True)), params))
        return cls(states, cache)

    def replace_from_dict(self, data: dict[str, Any], registry: OperatorRegistry) -> None:
        """原位恢复流水线，供撤销/重做保持控制器引用。"""

        restored = self.from_dict(data, registry, self.cache)
        self.states = restored.states
        self.cache.clear()

    def _index(self, operator_id: str) -> int:
        for index, state in enumerate(self.states):
            if state.operator.id == operator_id:
                return index
        raise ValueError(f"流水线中不存在算子：{operator_id}")


class PipelineHistory:
    """以序列化快照实现有界撤销/重做。"""

    def __init__(
        self,
        pipeline: ImagePipeline,
        registry: OperatorRegistry,
        max_entries: int = 100,
    ) -> None:
        self.pipeline = pipeline
        self.registry = registry
        self.max_entries = max_entries
        self._entries = [pipeline.to_dict()]
        self._cursor = 0

    @property
    def can_undo(self) -> bool:
        """是否存在更早快照。"""

        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        """是否存在被撤销的快照。"""

        return self._cursor < len(self._entries) - 1

    def checkpoint(self) -> None:
        """记录当前流水线；新修改会丢弃旧 redo 分支。"""

        snapshot = self.pipeline.to_dict()
        if snapshot == self._entries[self._cursor]:
            return
        self._entries = self._entries[: self._cursor + 1]
        self._entries.append(snapshot)
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        self._cursor = len(self._entries) - 1

    def undo(self) -> bool:
        """恢复上一个快照。"""

        if not self.can_undo:
            return False
        self._cursor -= 1
        self.pipeline.replace_from_dict(self._entries[self._cursor], self.registry)
        return True

    def redo(self) -> bool:
        """恢复下一个快照。"""

        if not self.can_redo:
            return False
        self._cursor += 1
        self.pipeline.replace_from_dict(self._entries[self._cursor], self.registry)
        return True


def _state_signature(state: OperatorState) -> str:
    payload = {
        "id": state.operator.id,
        "version": state.operator.version,
        "enabled": state.enabled,
        "params": state.params.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _image_signature(image: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str((image.shape, image.dtype.str)).encode())
    digest.update(memoryview(np.ascontiguousarray(image)))
    return digest.hexdigest()


def _validate_working_image(image: np.ndarray, label: str) -> None:
    """验证内部唯一图像契约，防止某个算子重新引入量化或越界。"""

    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label}必须是 H×W×3 RGB float32")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{label}包含非有限值")
    if image.size and (float(image.min()) < 0.0 or float(image.max()) > 1.0):
        raise ValueError(f"{label}必须位于 [0,1]")
