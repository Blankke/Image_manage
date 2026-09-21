"""公开几何清单的作品、拍摄会话与场景家族隔离规则。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SMARTDOC_SCENE_FAMILY = "smartdoc2015-ch1-capture-environment"
MIDV500_SCENE_FAMILY = "midv500-shared-capture-environment"
MIDV_HOLO_SCENE_FAMILY = "midv-holo-shared-capture-environments"


def scene_group_id(record: Mapping[str, Any]) -> str | None:
    """对反复使用同一拍摄环境的公开来源赋予保守的场景家族。"""

    source = str(record.get("source", ""))
    image = str(record.get("image", ""))
    explicit = record.get("scene_group_id")
    known_family = None
    if source in {"smartdoc", "smartdoc-photo-surface-composite"} or image.startswith("geometry/smartdoc/frames/"):
        known_family = SMARTDOC_SCENE_FAMILY
    elif source == "midv500" or image.startswith("geometry/midv500/documents/"):
        known_family = MIDV500_SCENE_FAMILY
    elif source == "midv-holo" or image.startswith("geometry/midv-holo/subset/"):
        known_family = MIDV_HOLO_SCENE_FAMILY
    if known_family is not None:
        if explicit and str(explicit) != known_family:
            raise ValueError("公开来源的 scene_group_id 与共享拍摄环境不一致")
        return known_family
    return str(explicit) if explicit else None


def validate_geometry_split_isolation(records: Iterable[Mapping[str, Any]]) -> None:
    """拒绝同一作品、会话或场景家族跨 train/validation/test。"""

    assignments: dict[tuple[str, str], str] = {}
    for record in records:
        split = str(record["split"])
        identifiers = [("group_id", str(record["group_id"]))]
        for field in ("capture_session", "digital_source_id", "subject_id"):
            value = record.get(field)
            if value:
                identifiers.append((field, str(value)))
        scene = scene_group_id(record)
        if scene:
            identifiers.append(("scene_group_id", scene))
        for field, value in identifiers:
            key = (field, value)
            previous = assignments.setdefault(key, split)
            if previous != split:
                raise ValueError(
                    f"数据泄漏：{field}={value!r} 同时出现在 {previous} 与 {split}"
                )
