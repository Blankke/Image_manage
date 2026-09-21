#!/usr/bin/env python3
"""从公开 replay 隔离 SmartDoc 重复拍摄环境，生成新的训练清单。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/isolate_smartdoc_scenes.py \
      --data-root /Users/caozichen/screenrestore-data \
      --input /Users/caozichen/screenrestore-data/manifests/p11/p8-gallery-photo-docci-full-bleed-replay-20260918.geometry.jsonl \
      --output /Users/caozichen/screenrestore-data/manifests/p12/scene-isolated-replay.geometry.jsonl

SmartDoc 的五种背景在不同文档中复用，现有按文档划出的 validation/test
不能作为独立拍摄场景。仅保留其 train 原片和衍生图，其它公开来源沿用原 split。
输入、输出和删除计数用哈希追踪；脚本拒绝覆盖，不读取私人数据。
旧合成负样本的 visible=true 会统一纠正为 false，写盘前逐条校验 geometry schema。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from screenrestore.io.geometry_isolation import (
    SMARTDOC_SCENE_FAMILY,
    scene_group_id,
    validate_geometry_split_isolation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    source = _public_file(args.input, root)
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("输出须位于 data-root 下的新路径，且不能覆盖")
    provenance_path = output.with_suffix(".provenance.json")
    if provenance_path.exists():
        raise FileExistsError(f"来源记录已存在：{provenance_path}")
    rows, removed, normalized_negative_visibility = isolate_rows(source, root)
    _validate_schema(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "kind": "smartdoc_scene_isolated_public_replay",
        "input": str(source),
        "input_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "generator_sha256": _sha256(Path(__file__)),
        "smartdoc_scene_group_id": SMARTDOC_SCENE_FAMILY,
        "removed_by_source_split": {
            f"{source_name}:{split}": count
            for (source_name, split), count in sorted(removed.items())
        },
        "normalized_negative_visibility": normalized_negative_visibility,
        "retained_count": len(rows),
        "retained_splits": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"retained": len(rows), "removed": sum(removed.values()),
                      "output": str(output)}, ensure_ascii=False))
    return 0


def isolate_rows(path: Path, root: Path) -> tuple[list[dict], Counter[tuple[str, str]], int]:
    """只过滤已知共享场景的 SmartDoc 行，再统一核查作品与场景隔离。"""

    rows: list[dict] = []
    removed: Counter[tuple[str, str]] = Counter()
    normalized_negative_visibility = 0
    total = sum(1 for _ in path.open(encoding="utf-8"))
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"清单第 {index} 行必须为对象")
            image = _public_file(root / str(row["image"]), root)
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
                raise ValueError(f"清单第 {index} 行不是受支持图片")
            scene = scene_group_id(row)
            if scene == SMARTDOC_SCENE_FAMILY:
                if row["split"] != "train":
                    removed[(str(row.get("source", "smartdoc")), str(row["split"]))] += 1
                    _progress(index, total)
                    continue
                row["scene_group_id"] = scene
            # 旧合成器把“多目标、无可接受目标”误记为 visible=true；其标签仍是
            # present=false/none，统一将可见性改为 false，使拒绝语义与 schema 一致。
            if row.get("present") is False and row.get("visible") is True:
                row["visible"] = False
                normalized_negative_visibility += 1
            rows.append(row)
            _progress(index, total)
    retained_images = {str(row["image"]) for row in rows}
    for row in rows:
        if (
            row.get("scene_group_id") == SMARTDOC_SCENE_FAMILY
            and row.get("source_capture_id")
            and str(row["source_capture_id"]) not in retained_images
        ):
            raise ValueError("SmartDoc 衍生图缺少保留的原片")
    validate_geometry_split_isolation(rows)
    return rows, removed, normalized_negative_visibility


def _validate_schema(rows: list[dict]) -> None:
    """最终写盘前验证每条来源记录，避免将旧数据的语义错误带入新训练。"""

    schema_path = Path(__file__).resolve().parents[1] / "datasets/schemas/geometry.schema.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    for index, row in enumerate(rows, 1):
        error = next(validator.iter_errors(row), None)
        if error is not None:
            raise ValueError(f"输出第 {index} 行不符合 geometry schema：{error.message[:150]}")
        if bool(row["present"]) == (str(row["target_class"]) == "none"):
            raise ValueError(f"输出第 {index} 行 present 与 target_class 矛盾")
        if index % max(1, len(rows) // 40) == 0 or index == len(rows):
            _progress(index, len(rows), "验证清单 schema")


def _public_file(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root) or any(
        "private" in part.lower() for part in resolved.relative_to(root).parts
    ) or not resolved.is_file():
        raise ValueError(f"公开数据路径不存在或越界：{path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(done: int, total: int, message: str = "隔离 SmartDoc 场景") -> None:
    if done < total and done % max(1, total // 40):
        return
    filled = round(24 * done / max(1, total))
    print(f"\r{message} [{'#' * filled}{'-' * (24 - filled)}] {done}/{total}",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
