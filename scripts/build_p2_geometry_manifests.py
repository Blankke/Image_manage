"""合并 P2 geometry 数据并生成 Stage A/B/C 与 calibration 固定清单。

使用范例：
    source .venv/bin/activate
    which python
    export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
    python scripts/build_p2_geometry_manifests.py \
        --data-root "$SCREENRESTORE_DATA_ROOT"

脚本只读取显式数据根下的 JSONL 与图片路径，不复制图片。Stage C 只让 private-train
进入梯度，同时混入 public/synthetic train replay；private-validation 用于选模，
private-test 保持 test。所有输出按 group/capture_session 再做一次跨 split 泄漏检查。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"image", "split", "group_id", "present", "target_class"}
VALID_SPLITS = {"train", "validation", "test"}
MULTI_TARGET_SCENE_TYPES = {
    "gallery_multi_target",
    "multiple_artworks",
    "multiple_equally_plausible",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--stage-a-synthetic-samples", type=int, default=2400)
    parser.add_argument("--stage-c-replay-samples", type=int, default=2400)
    args = parser.parse_args(argv)
    if args.stage_a_synthetic_samples < 1 or args.stage_c_replay_samples < 1:
        raise ValueError("synthetic/replay 样本数必须大于 0")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"data-root 不存在：{data_root}")
    output_directory = (
        args.output_directory.expanduser().resolve()
        if args.output_directory is not None
        else data_root / "manifests" / "p2"
    )
    if not output_directory.is_relative_to(data_root):
        raise ValueError("P2 manifests 必须写入 data-root")
    output_directory.mkdir(parents=True, exist_ok=True)

    smartdoc = _load_source(
        data_root,
        data_root / "manifests" / "smartdoc.geometry.jsonl",
        source="smartdoc",
    )
    synthetic = _load_source(
        data_root,
        data_root / "geometry" / "synthetic" / "manifest.jsonl",
        source="synthetic",
        image_prefix=Path("geometry/synthetic"),
    )
    optional_public = []
    for filename, source in (
        ("midv500.geometry.jsonl", "midv500"),
        ("midv-holo.geometry.jsonl", "midv-holo"),
    ):
        path = data_root / "manifests" / filename
        if path.is_file():
            optional_public.extend(_load_source(data_root, path, source=source))
        else:
            print(f"WARNING: 缺少可选公开清单：{path}", file=sys.stderr)
    private_path = data_root / "private" / "geometry.annotations.jsonl"
    private = (
        _load_source(data_root, private_path, source="private-labeled")
        if private_path.is_file()
        else []
    )
    if not private:
        print("WARNING: private 标注尚未完成，Stage C 清单不会生成", file=sys.stderr)

    stage_a_synthetic = _balanced_limit(synthetic, args.stage_a_synthetic_samples)
    stage_a = [*smartdoc, *stage_a_synthetic]
    stage_b = [*smartdoc, *optional_public, *synthetic]
    calibration = [record for record in stage_b if record["split"] == "validation"]
    if private:
        calibration.extend(record for record in private if record["split"] == "validation")

    outputs: dict[str, list[dict[str, Any]]] = {
        "stage-a.geometry.jsonl": stage_a,
        "stage-b.geometry.jsonl": stage_b,
        "calibration.geometry.jsonl": calibration,
        "all.geometry.jsonl": [*stage_b, *private],
    }
    if private:
        replay_candidates = [record for record in stage_b if record["split"] == "train"]
        replay = _balanced_limit(replay_candidates, args.stage_c_replay_samples)
        outputs["stage-c.geometry.jsonl"] = [*private, *replay]

    inventory: dict[str, object] = {"format_version": 1, "manifests": {}}
    for index, (filename, records) in enumerate(outputs.items(), start=1):
        _progress(index - 1, len(outputs), f"构建 {filename}")
        _validate_group_isolation(records)
        path = output_directory / filename
        _write_jsonl(path, records)
        inventory["manifests"][filename] = _statistics(records)  # type: ignore[index]
    _progress(len(outputs), len(outputs), "完成 P2 geometry manifests")
    inventory_path = output_directory / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(inventory_path)
    return 0


def _load_source(
    data_root: Path,
    manifest: Path,
    *,
    source: str,
    image_prefix: Path | None = None,
) -> list[dict[str, Any]]:
    if not manifest.is_file():
        raise ValueError(f"缺少必需清单：{manifest}")
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or REQUIRED_FIELDS - value.keys():
                raise ValueError(f"{manifest.name} 第 {line_number} 行缺少 geometry 字段")
            record = dict(value)
            image = Path(str(record["image"]))
            if image.is_absolute() or ".." in image.parts:
                raise ValueError(f"{manifest.name} 第 {line_number} 行 image 不安全")
            if image_prefix is not None:
                image = image_prefix / image
            resolved = (data_root / image).resolve()
            if not resolved.is_relative_to(data_root) or not resolved.is_file():
                raise ValueError(f"{manifest.name} 第 {line_number} 行图片不存在")
            record["image"] = image.as_posix()
            record["source"] = source
            record.setdefault("capture_session", str(record["group_id"]))
            record.setdefault("device", "unknown")
            record.setdefault("visible", bool(record["present"]))
            record.setdefault("occlusion", 0.0)
            record.setdefault("glare_level", "none")
            record.setdefault("ambiguous", False)
            record.setdefault("in_scope", bool(record["present"]))
            _validate_record_semantics(record, manifest, line_number)
            records.append(record)
    return records


def _balanced_limit(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """按 source/split/group 轮转选择，避免受长视频或单一 source 支配。"""

    if len(records) <= limit:
        return list(records)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("source", "unknown")), str(record["split"]), str(record["group_id"]))
        buckets.setdefault(key, []).append(record)
    for values in buckets.values():
        values.sort(key=lambda item: _stable_digest(str(item["image"])))
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda item: _stable_digest(":".join(item)))
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        values = buckets[key]
        if values:
            selected.append(values.pop())
        if not values:
            keys.remove(key)
            cursor = 0
        else:
            cursor += 1
    return sorted(selected, key=lambda item: (str(item["split"]), str(item["source"]), str(item["image"])))


def _validate_group_isolation(records: list[dict[str, Any]]) -> None:
    assignments: dict[tuple[str, str], str] = {}
    for record in records:
        split = str(record["split"])
        if split not in VALID_SPLITS:
            raise ValueError(f"非法 split：{split}")
        for kind in ("group_id", "capture_session"):
            key = (kind, str(record[kind]))
            previous = assignments.setdefault(key, split)
            if previous != split:
                raise ValueError(f"P2 清单泄漏：{kind}={key[1]!r} 跨 {previous}/{split}")


def _validate_record_semantics(record: dict[str, Any], manifest: Path, line_number: int) -> None:
    """阻止多目标歧义样本携带任意单一四角，避免给模型冲突监督。"""

    scene_type = str(record.get("scene_type", ""))
    multiple_targets = scene_type in MULTI_TARGET_SCENE_TYPES
    if multiple_targets and not bool(record.get("ambiguous")):
        raise ValueError(f"{manifest.name} 第 {line_number} 行多目标场景必须标记 ambiguous")
    if not multiple_targets:
        return
    invalid = (
        bool(record.get("present"))
        or str(record.get("target_class")) != "none"
        or record.get("content_quad") is not None
        or record.get("outer_quad") is not None
        or bool(record.get("visible"))
        or bool(record.get("in_scope"))
    )
    if invalid:
        raise ValueError(f"{manifest.name} 第 {line_number} 行多目标场景必须使用拒绝语义")


def _statistics(records: list[dict[str, Any]]) -> dict[str, object]:
    return {
        "samples": len(records),
        "groups": len({str(record["group_id"]) for record in records}),
        "splits": dict(sorted(Counter(str(record["split"]) for record in records).items())),
        "classes": dict(sorted(Counter(str(record["target_class"]) for record in records).items())),
        "sources": dict(sorted(Counter(str(record.get("source", "unknown")) for record in records).items())),
        "scene_types": dict(
            sorted(Counter(str(record.get("scene_type", "unknown")) for record in records).items())
        ),
        "ambiguous": dict(
            sorted(Counter("yes" if bool(record.get("ambiguous")) else "no" for record in records).items())
        ),
        "multi_target": dict(
            sorted(
                Counter(
                    "yes" if str(record.get("scene_type", "")) in MULTI_TARGET_SCENE_TYPES else "no"
                    for record in records
                ).items()
            )
        ),
        "content_presence": dict(
            sorted(Counter("present" if bool(record["present"]) else "absent" for record in records).items())
        ),
        "outer_presence": dict(
            sorted(Counter("present" if record.get("outer_quad") is not None else "absent" for record in records).items())
        ),
    }


def _stable_digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    filled = round(width * min(1.0, done / max(1, total)))
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>2}/{total:<2} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
