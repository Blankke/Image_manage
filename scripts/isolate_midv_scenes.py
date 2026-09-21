#!/usr/bin/env python3
"""从 P12 公开清单隔离 MIDV 共享拍摄环境，生成 P13 训练清单。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/isolate_midv_scenes.py \
      --data-root /Users/caozichen/screenrestore-data \
      --input /Users/caozichen/screenrestore-data/manifests/p12/scene-isolated-replay-20260918-v3.geometry.jsonl \
      --output /Users/caozichen/screenrestore-data/manifests/p13/scene-isolated-replay-20260918.geometry.jsonl

MIDV-500 和 MIDV-Holo 的不同证件组共享拍摄布景，因此各自只保留 train。
脚本逐条验证图片路径及 geometry schema，记录输入、输出和脚本摘要，拒绝覆盖。
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
    MIDV500_SCENE_FAMILY,
    MIDV_HOLO_SCENE_FAMILY,
    scene_group_id,
    validate_geometry_split_isolation,
)

MIDV_FAMILIES = {MIDV500_SCENE_FAMILY, MIDV_HOLO_SCENE_FAMILY}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    source = _public_file(args.input, root)
    output = args.output.expanduser().resolve()
    provenance_path = output.with_suffix(".provenance.json")
    if not output.is_relative_to(root) or output.exists() or provenance_path.exists():
        raise ValueError("输出须位于 data-root 下的新路径，且不能覆盖")

    rows, removed = isolate_rows(source, root)
    _validate_schema(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "kind": "midv_scene_isolated_public_replay",
        "input": str(source),
        "input_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "generator_sha256": _sha256(Path(__file__)),
        "scene_families": sorted(MIDV_FAMILIES),
        "removed_by_source_split": {
            f"{name}:{split}": count for (name, split), count in sorted(removed.items())
        },
        "retained_count": len(rows),
        "retained_splits": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"retained": len(rows), "removed": sum(removed.values()),
                      "output": str(output)}, ensure_ascii=False))
    return 0


def isolate_rows(path: Path, root: Path) -> tuple[list[dict], Counter[tuple[str, str]]]:
    """只排除 MIDV 相关验证/测试行，保留其它来源的原 split。"""

    rows: list[dict] = []
    removed: Counter[tuple[str, str]] = Counter()
    total = sum(1 for _ in path.open(encoding="utf-8"))
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"清单第 {index} 行必须为对象")
            _public_file(root / str(row["image"]), root)
            scene = scene_group_id(row)
            if scene in MIDV_FAMILIES:
                if row["split"] != "train":
                    removed[(str(row.get("source", "unknown")), str(row["split"]))] += 1
                    _progress(index, total)
                    continue
                row["scene_group_id"] = scene
            rows.append(row)
            _progress(index, total)
    validate_geometry_split_isolation(rows)
    return rows, removed


def _validate_schema(rows: list[dict]) -> None:
    """写盘前检查全部标签结构与 presence 语义。"""

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


def _progress(done: int, total: int, message: str = "隔离 MIDV 场景") -> None:
    if done < total and done % max(1, total // 40):
        return
    filled = round(24 * done / max(1, total))
    print(f"\r{message} [{'#' * filled}{'-' * (24 - filled)}] {done}/{total}",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
