#!/usr/bin/env python3
"""从 Wikimedia Commons 官方分类抽取实拍平面目标的许可审核缩略图。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/sample_commons_planar_targets.py \
      --data-root /Users/caozichen/screenrestore-data \
      --category poster \
      --output-directory /Users/caozichen/screenrestore-data/geometry/commons-poster-review-20260918

脚本只下载预览图，不生成四角，也不将 Commons 文件页的照片许可推断为画面内作品许可。
逐图照片许可、内容权利、独立场景与 content_quad 审核通过前，候选不得进入训练或验收。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

API_URL = "https://commons.wikimedia.org/w/api.php"
CATEGORIES = {
    "poster": "Poster displays",
    "artwork": "Works of art hanging on the walls",
    # 该分类保留画框与少量拍摄环境，比展厅远景更适合检查 content/outer 层级。
    "framed-artwork": "Framed paintings in the National Gallery, London",
}
ALLOWED_LICENSES = {
    "Public domain", "CC0", "CC BY 2.0", "CC BY 2.5", "CC BY 3.0", "CC BY 4.0",
    "CC BY-SA 2.0", "CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0",
}
USER_AGENT = "ScreenRestore/0.1 (public image license audit; local research)"
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
NETWORK_ATTEMPTS = 3
VENUE_TITLE_PREFIXES = (
    "london - tate britain -",
    "museo dell'opera del duomo",
    "presentation by using poster in village of bangladesh",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", choices=tuple(CATEGORIES), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=36)
    parser.add_argument("--max-download-mib", type=float, default=32.0)
    parser.add_argument("--max-per-title-family", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("输出必须位于 data-root 下的新目录，且不能覆盖")
    if args.max_images < 1 or args.max_download_mib <= 0 or args.max_per_title_family < 1:
        raise ValueError("下载数量和上限必须为正数")

    category = CATEGORIES[args.category]
    titles = _category_titles(category)
    metadata = _image_metadata(titles)
    eligible = [record for record in metadata if _metadata_eligible(record)]
    eligible.sort(key=lambda record: hashlib.sha256(
        f"{args.seed}:{record['title']}".encode()
    ).digest())
    selected: list[dict] = []
    family_count: dict[str, int] = {}
    for record in eligible:
        family = _title_family(str(record["title"]))
        if family_count.get(family, 0) >= args.max_per_title_family:
            continue
        family_count[family] = family_count.get(family, 0) + 1
        record["sampling_title_family"] = family
        selected.append(record)
        if len(selected) == args.max_images:
            break
    if not selected:
        raise ValueError("分类中没有符合文件格式、尺寸和照片许可元数据的候选")

    output.mkdir(parents=True)
    image_dir = output / "thumbnails"
    image_dir.mkdir()
    downloaded = 0
    total_bytes = 0
    budget = int(args.max_download_mib * 1024**2)
    for index, record in enumerate(selected, 1):
        url = str(record["thumbnail_url"])
        record["thumbnail_status"] = "unavailable"
        if _allowed_thumbnail_url(url):
            try:
                payload = _download_thumbnail(url)
                if total_bytes + len(payload) > budget:
                    raise ValueError("total_download_budget_exceeded")
                relative = Path("thumbnails") / f"{record['page_id']}.jpg"
                (output / relative).write_bytes(payload)
                record["thumbnail_path"] = relative.as_posix()
                record["thumbnail_sha256"] = hashlib.sha256(payload).hexdigest()
                record["thumbnail_status"] = "downloaded"
                downloaded += 1
                total_bytes += len(payload)
            except (OSError, ValueError) as exc:
                record["thumbnail_error"] = type(exc).__name__
        _progress(index, len(selected), "下载 Commons 预览")
    board_path = output / "review-board.jpg"
    _write_board(selected, output, board_path)
    report = {
        "kind": "commons_planar_target_photo_candidates",
        "category": category,
        "category_url": "https://commons.wikimedia.org/wiki/Category:" + quote(category.replace(" ", "_")),
        "api_url": API_URL,
        "generator_sha256": _sha256(Path(__file__)),
        "seed": args.seed,
        "category_file_count": len(titles),
        "photo_license_metadata_candidates": len(eligible),
        "selected_count": len(selected),
        "downloaded_count": downloaded,
        "board_sha256": _sha256(board_path),
        "records": selected,
    }
    (output / "sample-index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected": len(selected), "downloaded": downloaded,
                      "index": str(output / "sample-index.json")}, ensure_ascii=False))
    return 0


def _category_titles(category: str) -> list[dict]:
    """用官方分类 API 枚举文件页，不继承分类页的许可结论。"""

    titles: list[dict] = []
    continuation: str | None = None
    while True:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": "file", "cmlimit": "500",
        }
        if continuation:
            params["cmcontinue"] = continuation
        response = _request_json(params)
        titles.extend(response.get("query", {}).get("categorymembers", []))
        continuation = response.get("continue", {}).get("cmcontinue")
        if continuation is None:
            break
        if len(titles) > 1000:
            raise ValueError("分类文件过多，请选择更具体的分类")
    return titles


def _image_metadata(titles: list[dict]) -> list[dict]:
    """文件页逐批查询照片许可；元数据只生成候选，内容权利仍待人工审查。"""

    records: list[dict] = []
    batches = [titles[index : index + 10] for index in range(0, len(titles), 10)]
    for index, batch in enumerate(batches, 1):
        response = _request_json({
            "action": "query", "format": "json", "prop": "imageinfo",
            "titles": "|".join(str(item["title"]) for item in batch),
            "iiprop": "url|size|mime|sha1|extmetadata", "iiurlwidth": "480",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Copyrighted|Artist|Credit|ImageDescription",
        })
        for page in response.get("query", {}).get("pages", {}).values():
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
            meta = info.get("extmetadata", {})
            title = str(page["title"])
            records.append({
                "page_id": int(page["pageid"]),
                "title": title,
                "file_page_url": "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_")),
                "mime": info.get("mime"),
                "width": info.get("width"),
                "height": info.get("height"),
                "file_sha1": info.get("sha1"),
                "original_url": info.get("url"),
                "thumbnail_url": info.get("thumburl", info.get("url")),
                "photo_license_metadata": _meta(meta, "LicenseShortName"),
                "photo_license_url": _meta(meta, "LicenseUrl"),
                "copyrighted_metadata": _meta(meta, "Copyrighted"),
                "artist_excerpt": _plain(_meta(meta, "Artist"), 180),
                "description_excerpt": _plain(_meta(meta, "ImageDescription"), 240),
                "photo_license_review": "pending_file_page_review",
                "depicted_content_rights_review": "pending",
                "scene_group_review": "pending",
                "content_quad_review": "pending",
                "training_eligible": False,
                "evaluation_eligible": False,
            })
        _progress(index, len(batches), "核对 Commons 文件元数据")
    return records


def _metadata_eligible(record: dict) -> bool:
    return (
        record.get("mime") in {"image/jpeg", "image/png"}
        and min(int(record.get("width") or 0), int(record.get("height") or 0)) >= 800
        and record.get("photo_license_metadata") in ALLOWED_LICENSES
        and _allowed_thumbnail_url(str(record.get("thumbnail_url") or ""))
    )


def _title_family(title: str) -> str:
    # 连拍常以末尾数字区分；这只控制预览抽样数量，正式 group 仍需视觉核查。
    stem = re.sub(r"\.[^.]+$", "", title.removeprefix("File:"), flags=re.IGNORECASE)
    lowered = stem.casefold()
    for prefix in VENUE_TITLE_PREFIXES:
        if lowered.startswith(prefix):
            return prefix
    stem = re.sub(r"(?:[ _-]+(?:view|img)?[ _-]*\d{1,3})+$", "", stem, flags=re.IGNORECASE)
    return stem.casefold()


def _allowed_thumbnail_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in {"thumb.wikimedia.org", "upload.wikimedia.org"}
        and parsed.path.startswith("/wikipedia/commons/")
    )


def _download_thumbnail(url: str) -> bytes:
    payload = _request_bytes(url, MAX_THUMBNAIL_BYTES, timeout=25)
    if len(payload) > MAX_THUMBNAIL_BYTES:
        raise ValueError("thumbnail_too_large")
    with Image.open(io.BytesIO(payload)) as photo:
        photo.verify()
    return payload


def _write_board(records: list[dict], output: Path, path: Path) -> None:
    cell_width, cell_height, columns = 240, 225, 6
    board = Image.new("RGB", (cell_width * columns, cell_height * ((len(records) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(board)
    for index, record in enumerate(records):
        x, y = index % columns * cell_width, index // columns * cell_height
        relative = record.get("thumbnail_path")
        if relative:
            with Image.open(output / str(relative)) as source:
                photo = source.convert("RGB")
                photo.thumbnail((cell_width - 12, cell_height - 52))
            board.paste(photo, (x + (cell_width - photo.width) // 2, y + 37))
        draw.text((x + 5, y + 5), str(record["title"])[5:36], fill="black")
        draw.text((x + 5, y + cell_height - 16), str(record["photo_license_metadata"])[:30], fill="black")
    board.save(path, quality=88)


def _meta(metadata: dict, name: str) -> str:
    return str(metadata.get(name, {}).get("value", ""))[:500]


def _plain(value: str, limit: int) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()[:limit]


def _request_json(params: dict[str, str]) -> dict:
    payload = _request_bytes(API_URL + "?" + urlencode(params), 5 * 1024 * 1024, timeout=40)
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError("Commons API 响应超过大小上限")
    return json.loads(payload)


def _request_bytes(url: str, limit: int, *, timeout: int) -> bytes:
    """对 Commons 的瞬时连接失败做有限重试，同时维持严格响应大小上限。"""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: OSError | None = None
    for attempt in range(NETWORK_ATTEMPTS):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError("Commons 响应超过大小上限")
            return payload
        except OSError as exc:
            last_error = exc
            if attempt + 1 < NETWORK_ATTEMPTS:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(done: int, total: int, message: str) -> None:
    filled = round(24 * done / max(1, total))
    print(f"\r{message} [{'#' * filled}{'-' * (24 - filled)}] {done}/{total}",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
