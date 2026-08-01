"""算子参数的可序列化基类。"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T", bound="ParameterModel")


class ParameterModel:
    """所有算子参数的轻量、无额外依赖序列化协议。"""

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的字典。"""

        if not is_dataclass(self):
            raise TypeError("参数模型必须使用 @dataclass")
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """从字典构造参数，并恢复枚举字段。"""

        hints = get_type_hints(cls)
        values: dict[str, Any] = {}
        allowed = {item.name for item in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"未知参数：{', '.join(sorted(unknown))}")
        for key, value in data.items():
            hint = hints.get(key)
            if isinstance(hint, type) and issubclass(hint, Enum):
                values[key] = hint(value)
            else:
                values[key] = value
        return cls(**values)

    def validate(self) -> None:
        """参数自校验钩子；具体模型可覆盖。"""


def _json_value(value: Any) -> Any:
    """递归转换枚举等 JSON 不认识的值。"""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value

