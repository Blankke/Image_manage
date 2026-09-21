#!/usr/bin/env python3
"""合并公开 replay 与目标域数据，保留同一原片衍生样本的分组身份。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/build_p5_target_manifest.py \
      --data-root /data/screenrestore \
      --base-manifest /data/screenrestore/manifests/p2-public/stage-b.geometry.jsonl \
      --target-manifest /data/screenrestore/geometry/synthetic-target-v4/manifest.jsonl \
      --target-root /data/screenrestore/geometry/synthetic-target-v4 \
      --target-namespace target-v4 \
      --output /data/screenrestore/manifests/p5/target-v4-replay.geometry.jsonl

脚本只接受公开路径，拒绝覆盖既有输出；独立新来源使用 target-namespace，引用 base
图片的衍生样本沿用原片 group 与 capture session。训练时可直接使用 ``--dataset-root``。
进度和最终分布写到终端，不生成额外报告文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from screenrestore.io.geometry_isolation import scene_group_id, validate_geometry_split_isolation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-namespace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = build_manifest(
        data_root=args.data_root,
        base_manifest=args.base_manifest,
        target_manifest=args.target_manifest,
        target_root=args.target_root,
        target_namespace=args.target_namespace,
        output=args.output,
    )
    splits = Counter(str(row["split"]) for row in rows)
    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    classes = Counter(str(row["target_class"]) for row in rows)
    print(
        json.dumps(
            {
                "samples": len(rows),
                "splits": dict(sorted(splits.items())),
                "sources": dict(sorted(sources.items())),
                "classes": dict(sorted(classes.items())),
                "output": str(args.output.expanduser().resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_manifest(
    *,
    data_root: Path,
    base_manifest: Path,
    target_manifest: Path,
    target_root: Path,
    target_namespace: str,
    output: Path,
) -> list[dict[str, Any]]:
    """返回合并行并写入唯一输出，过程中验证路径与 split 隔离。"""

    root = data_root.expanduser().resolve()
    base_path = _public_file(base_manifest, root)
    target_path = _public_file(target_manifest, root)
    target_directory = _public_directory(target_root, root)
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{destination}")
    if not target_namespace.strip():
        raise ValueError("target-namespace 不能为空")
    base_rows = _read_jsonl(base_path)
    target_rows = _read_jsonl(target_path)
    base_by_image = {str(row["image"]): row for row in base_rows}
    if len(base_by_image) != len(base_rows):
        raise ValueError("base 清单中存在重复图片路径")
    rows: list[dict[str, Any]] = []
    total = len(base_rows) + len(target_rows)
    for index, row in enumerate(base_rows):
        _validate_image(root / str(row["image"]), root)
        rows.append(row)
        _progress(index + 1, total, "合并公开 replay")
    for target_index, row in enumerate(target_rows, start=1):
        image = (target_directory / str(row["image"])).resolve()
        _validate_image(image, root)
        # 真实原片衍生图继承原拍摄身份；独立合成来源才分配新命名空间。
        parent = base_by_image.get(str(row.get("source_capture_id", "")))
        if parent is not None:
            if row["split"] != parent["split"] or row["group_id"] != parent["group_id"]:
                raise ValueError("衍生样本与原片的 split 或 group 不一致")
            group_id = str(parent["group_id"])
            capture_session = str(parent.get("capture_session", group_id))
        else:
            group_id = f"{target_namespace}:{row['group_id']}"
            capture_session = f"{target_namespace}:{row.get('capture_session', row['group_id'])}"
        rewritten = {
            **row,
            "image": image.relative_to(root).as_posix(),
            "group_id": group_id,
            "capture_session": capture_session,
        }
        if parent is not None and scene_group_id(parent):
            rewritten["scene_group_id"] = scene_group_id(parent)
        rows.append(rewritten)
        _progress(len(base_rows) + target_index, total, "合并目标域")
    validate_geometry_split_isolation(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _progress(total, total, "完成")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须为对象")
        for key in ("image", "split", "group_id", "target_class", "present"):
            if key not in value:
                raise ValueError(f"{path} 第 {line_number} 行缺少 {key}")
        rows.append(value)
    return rows


def _public_file(path: Path, data_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    _ensure_public_under_root(resolved, data_root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _public_directory(path: Path, data_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    _ensure_public_under_root(resolved, data_root)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_image(path: Path, data_root: Path) -> None:
    resolved = path.resolve()
    _ensure_public_under_root(resolved, data_root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)


def _ensure_public_under_root(path: Path, data_root: Path) -> None:
    try:
        relative = path.relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"路径必须位于 data-root：{path}") from error
    if any("private" in part.lower() for part in relative.parts):
        raise ValueError(f"P5 公开训练清单禁止读取 private 路径：{path}")


def _progress(done: int, total: int, message: str) -> None:
    # CI/代理终端会保留每次回车刷新；最多刷新约 100 次，避免数万行噪声。
    interval = max(1, total // 100)
    if done < total and done != 1 and done % interval != 0:
        return
    width = 28
    filled = round(width * min(1.0, done / max(1, total)))
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>6}/{total:<6} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
