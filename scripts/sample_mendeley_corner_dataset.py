#!/usr/bin/env python3
"""从 Mendeley 官方 ZIP 按需抽样真实文档照片与同编号四角 CSV。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/sample_mendeley_corner_dataset.py \
      --data-root /Users/caozichen/screenrestore-data \
      --output-directory /Users/caozichen/screenrestore-data/geometry/mendeley-corner-pilot-20260917 \
      --count 12 --seed 20260917

只读取 ZIP 尾部索引及被选文件的字节范围，避免提前下载约 2.6 GB 整包。
数据页标注 CC BY 4.0；图中原始文档、隐私和角点语义须逐张复核。
抽样结果仅作待审核数据，不能直接进入训练或自动验收。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

from PIL import Image

DATASET_URL = "https://data.mendeley.com/datasets/x3nm4cxr83/3"
ARCHIVE_URL = "https://data.mendeley.com/public-api/zip/x3nm4cxr83/download/3"
ARCHIVE_TAIL_BYTES = 1024 * 1024
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MEMBER_NAME = re.compile(r"All My Dataset/\((\d+)\)\.(jpg|csv)$")
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("输出须位于 data-root 内的新目录，且不能覆盖现有文件")
    if args.count < 1 or args.count > 100:
        raise ValueError("count 必须位于 1..100；扩大使用规模须先完成样本审核")

    tail, total_size, tail_start = _range_request(f"bytes=-{ARCHIVE_TAIL_BYTES}")
    with zipfile.ZipFile(io.BytesIO(tail)) as archive:
        members = _paired_members(archive.infolist())
    if args.count > len(members):
        raise ValueError(f"可配对编号只有 {len(members)} 个")
    chosen = sorted(random.Random(args.seed).sample(sorted(members), args.count))
    # 全部样本下载并校验后再写盘，避免失败时留下看似完整的审计目录。
    downloaded: list[tuple[int, zipfile.ZipInfo, zipfile.ZipInfo, bytes, bytes]] = []
    for position, number in enumerate(chosen, start=1):
        image_info, label_info = members[number]
        image_bytes = _read_member(image_info, tail_start, total_size)
        label_bytes = _read_member(label_info, tail_start, total_size)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        label_bytes.decode("utf-8-sig")
        downloaded.append((number, image_info, label_info, image_bytes, label_bytes))
        _progress(position, len(chosen))
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir()
    records: list[dict[str, object]] = []
    for number, image_info, label_info, image_bytes, label_bytes in downloaded:
        image_path = output / "images" / f"{number:04d}.jpg"
        label_path = output / "labels" / f"{number:04d}.csv"
        image_path.write_bytes(image_bytes)
        label_path.write_bytes(label_bytes)
        records.append(
            {
                "source_id": number,
                "image": image_path.relative_to(root).as_posix(),
                "annotation": label_path.relative_to(root).as_posix(),
                "archive_image_name": image_info.filename,
                "archive_annotation_name": label_info.filename,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "annotation_sha256": hashlib.sha256(label_bytes).hexdigest(),
                "review_status": "pending_visual_rights_and_geometry_review",
            }
        )
    (output / "sample-index.json").write_text(
        json.dumps(
            {
                "kind": "mendeley_document_corner_pilot",
                "dataset_url": DATASET_URL,
                "dataset_version": 3,
                "dataset_license": "CC-BY-4.0; underlying document rights require review",
                "archive_bytes": total_size,
                "archive_index_sha256": hashlib.sha256(tail).hexdigest(),
                "seed": args.seed,
                "eligible_pairs": len(members),
                "samples": records,
                "training_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sample_count": len(records), "output": str(output)}, ensure_ascii=False))
    return 0


def _range_request(spec: str) -> tuple[bytes, int, int]:
    # 官方站点拒绝 urllib 的请求；curl 可正确跟随临时下载地址并保留 Range。
    limit = MAX_MEMBER_BYTES + 131072
    with tempfile.TemporaryDirectory(prefix="screenrestore-range-") as temporary:
        data_path = Path(temporary) / "data"
        header_path = Path(temporary) / "headers"
        subprocess.run(
            [
                "curl", "--location", "--fail", "--silent", "--show-error",
                "--max-filesize", str(limit), "--range", spec.removeprefix("bytes="),
                "--dump-header", str(header_path), "--output", str(data_path),
                ARCHIVE_URL,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        headers = header_path.read_text(encoding="iso-8859-1").splitlines()
        if not any(line.startswith("HTTP/") and " 206 " in line for line in headers):
            raise OSError("服务器未遵守 Range 请求：缺少 HTTP 206")
        ranges = [line.split(":", 1)[1].strip() for line in headers if line.lower().startswith("content-range:")]
        match = CONTENT_RANGE.fullmatch(ranges[-1]) if ranges else None
        if match is None:
            raise OSError("下载响应缺少合法 Content-Range")
        start, end, total = (int(value) for value in match.groups())
        expected = end - start + 1
        if expected > limit:
            raise ValueError("单次下载范围过大")
        data = data_path.read_bytes()
        if len(data) != expected:
            raise OSError("下载字节数与 Content-Range 不一致")
        return data, total, start


def _paired_members(
    infos: list[zipfile.ZipInfo],
) -> dict[int, tuple[zipfile.ZipInfo, zipfile.ZipInfo]]:
    by_number: dict[int, dict[str, zipfile.ZipInfo]] = {}
    for info in infos:
        match = MEMBER_NAME.fullmatch(info.filename)
        if match is None:
            continue
        number, suffix = int(match.group(1)), match.group(2)
        pair = by_number.setdefault(number, {})
        if suffix in pair:
            raise ValueError(f"ZIP 中存在重复编号和后缀：{number}")
        pair[suffix] = info
    return {
        number: (pair["jpg"], pair["csv"])
        for number, pair in by_number.items()
        if "jpg" in pair and "csv" in pair
    }


def _read_member(info: zipfile.ZipInfo, tail_start: int, total_size: int) -> bytes:
    if info.file_size > MAX_MEMBER_BYTES or info.compress_size > MAX_MEMBER_BYTES:
        raise ValueError(f"ZIP 成员超出大小限制：{info.filename[:80]}")
    # 只拿到 ZIP 尾部时，zipfile 会把原始 header_offset 平移到尾部文件坐标系。
    offset = info.header_offset + tail_start
    size = 30 + len(info.filename.encode("utf-8")) + info.compress_size + 65535
    end = min(total_size - 1, offset + size - 1)
    payload, reported_total, reported_start = _range_request(f"bytes={offset}-{end}")
    if reported_total != total_size or reported_start != offset:
        raise OSError("ZIP 归档版本在下载期间发生变化")
    if len(payload) < 30:
        raise OSError("ZIP 成员本地头不完整")
    signature, *_, name_len, extra_len = struct.unpack_from("<IHHHHHIIIHH", payload)
    if signature != 0x04034B50:
        raise ValueError("ZIP 成员本地头签名错误")
    begin = 30 + name_len + extra_len
    compressed = payload[begin : begin + info.compress_size]
    if len(compressed) != info.compress_size:
        raise OSError("ZIP 成员压缩数据不完整")
    if info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise ValueError(f"不支持的 ZIP 压缩方法：{info.compress_type}")
    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
        raise ValueError(f"ZIP 成员长度或 CRC 校验失败：{info.filename[:80]}")
    return content


def _progress(done: int, total: int) -> None:
    filled = round(24 * done / total)
    print(
        f"\r[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} 文档数据抽样",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
