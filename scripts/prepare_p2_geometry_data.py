"""准备 P2 geometry 的受控公开数据，不读取 private，也不启动训练。

使用范例：
    source .venv/bin/activate
    which python
    export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
    python scripts/prepare_p2_geometry_data.py --dataset all \
        --data-root "$SCREENRESTORE_DATA_ROOT" --met-count 1500

数据策略：MIDV-500 固定选择 10 种文档并保留各自完整 capture condition；MIDV-Holo
从官方 14GB images.tar 按 lighting×device×document kind×sample kind 流式抽取固定子集，
成功后默认删除大归档；Met 只下载 isPublicDomain=true 且有 primary image 的对象；COCO
只下载 val2017。原始公开数据上限默认 14 GiB，为合成输出另留 6 GiB，总新增预算约
20 GiB；同时保留 10 GiB 文件系统可用空间。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

GIB = 1024**3
MIDV500_ROOT = "ftp://smartengines.com/midv-500/dataset"
MIDV500_MD5_URL = "ftp://smartengines.com/midv-500/md5.txt"
MIDV_HOLO_ROOT = "ftp://smartengines.com/midv-holo"
MET_API_ROOT = "https://collectionapi.metmuseum.org/public/collection/v1"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_BYTES = 815_585_330
MIDV_HOLO_IMAGES_BYTES = 14_166_804_480
MIDV500_SUBSET = (
    "01_alb_id.zip",
    "06_bra_passport.zip",
    "11_cze_passport.zip",
    "16_deu_passport_new.zip",
    "21_esp_id_old.zip",
    "26_hrv_drvlic.zip",
    "31_jpn_drvlic.zip",
    "36_pol_drvlic.zip",
    "41_srb_passport.zip",
    "46_ury_passport.zip",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("midv500", "midv-holo", "met", "coco", "all"),
        default="all",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--met-count", type=int, default=1500)
    parser.add_argument("--midv-holo-clips-per-cell", type=int, default=1)
    parser.add_argument("--max-new-gib", type=float, default=14.0)
    parser.add_argument("--minimum-free-gib", type=float, default=10.0)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)
    if not 100 <= args.met_count <= 3000:
        raise ValueError("met-count 必须位于 100..3000")
    if not 1 <= args.midv_holo_clips_per_cell <= 3:
        raise ValueError("midv-holo-clips-per-cell 必须位于 1..3")
    if args.max_new_gib <= 0 or args.minimum_free_gib < 0:
        raise ValueError("容量参数无效")
    data_root = args.data_root.expanduser().resolve()
    if "private" in data_root.parts:
        raise ValueError("公开数据准备器拒绝 private 路径")
    data_root.mkdir(parents=True, exist_ok=True)
    initial_public_bytes = _public_usage(data_root)
    context = {
        "data_root": data_root,
        "initial_public_bytes": initial_public_bytes,
        "max_new_bytes": int(args.max_new_gib * GIB),
        "minimum_free_bytes": int(args.minimum_free_gib * GIB),
        "keep_archives": args.keep_archives,
    }
    selected = (
        ("midv500", "midv-holo", "met", "coco")
        if args.dataset == "all"
        else (args.dataset,)
    )
    reports: list[dict[str, object]] = []
    for index, dataset in enumerate(selected, start=1):
        _progress(index - 1, len(selected), f"准备 {dataset}")
        if dataset == "midv500":
            reports.append(_prepare_midv500(context))
        elif dataset == "midv-holo":
            reports.append(
                _prepare_midv_holo(context, clips_per_cell=args.midv_holo_clips_per_cell)
            )
        elif dataset == "met":
            reports.append(_prepare_met(context, count=args.met_count))
        else:
            reports.append(_prepare_coco(context))
        _enforce_final_budget(context)
    _progress(len(selected), len(selected), "P2 geometry 数据准备完成")
    inventory = {
        "format_version": 1,
        "selection_rules": {
            "midv500": list(MIDV500_SUBSET),
            "midv_holo_clips_per_cell": args.midv_holo_clips_per_cell,
            "met_count": args.met_count,
            "coco": "val2017 only",
        },
        "initial_public_bytes": initial_public_bytes,
        "final_public_bytes": _public_usage(data_root),
        "new_public_bytes": _public_usage(data_root) - initial_public_bytes,
        "filesystem_free_bytes": shutil.disk_usage(data_root).free,
        "datasets": reports,
    }
    inventory_path = data_root / "manifests" / "p2-download-inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(inventory_path)
    return 0


def _prepare_midv500(context: dict[str, Any]) -> dict[str, object]:
    data_root: Path = context["data_root"]
    downloads = data_root / "downloads" / "p2" / "midv500"
    destination = data_root / "geometry" / "midv500" / "documents"
    downloads.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    checksum_path = downloads / "md5.txt"
    if not checksum_path.is_file():
        _download(MIDV500_MD5_URL, checksum_path)
    checksums = _read_midv500_checksums(checksum_path)
    for index, filename in enumerate(MIDV500_SUBSET, start=1):
        _progress(index - 1, len(MIDV500_SUBSET), f"MIDV-500 {filename}")
        archive = downloads / filename
        marker = destination / f".{filename}.ready"
        if not marker.is_file():
            _ensure_peak_capacity(context, _remote_size(f"{MIDV500_ROOT}/{filename}"))
            _download(f"{MIDV500_ROOT}/{filename}", archive)
            actual_md5 = _file_digest(archive, "md5")
            if actual_md5 != checksums[filename]:
                raise ValueError(
                    f"MIDV-500 {filename} MD5 不匹配：期望 {checksums[filename]}，实际 {actual_md5}"
                )
            _extract_zip_safely(archive, destination)
            marker.write_text("ready\n", encoding="utf-8")
        if archive.is_file() and not context["keep_archives"]:
            archive.unlink()
    _progress(len(MIDV500_SUBSET), len(MIDV500_SUBSET), "MIDV-500 子集完成")
    manifest = data_root / "manifests" / "midv500.geometry.jsonl"
    records = _build_midv500_manifest(data_root, destination)
    _write_jsonl(manifest, records)
    return {
        "dataset": "midv500",
        "license": "CC-BY-SA-2.5",
        "selection": "10 fixed document types; all capture conditions within selected types",
        "samples": len(records),
        "groups": len({record["group_id"] for record in records}),
        "manifest": str(manifest),
    }


def _build_midv500_manifest(data_root: Path, root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for truth in sorted(root.rglob("ground_truth/**/*.json")):
        relative_truth = truth.relative_to(root)
        parts = relative_truth.parts
        truth_index = parts.index("ground_truth")
        document_root = root.joinpath(*parts[:truth_index])
        relative = Path(*parts[truth_index + 1 :])
        image = _matching_image(document_root / "images" / relative.with_suffix(""))
        if image is None or relative.parent == Path("."):
            continue
        value = json.loads(truth.read_text(encoding="utf-8"))
        quad = _find_quad(value)
        if quad is None:
            continue
        with Image.open(image) as opened:
            width, height = opened.size
        normalized = np_normalized_quad(quad, width, height)
        visible = all(0.0 <= coordinate <= 1.0 for point in normalized for coordinate in point)
        document_code = document_root.name
        session = relative.parent.as_posix()
        group_id = f"midv500:{document_code}"
        records.append(
            {
                "image": image.resolve().relative_to(data_root).as_posix(),
                "split": _split_for_group(group_id),
                "group_id": group_id,
                "capture_session": f"midv500:{document_code}:{session}",
                "device": "midv500-mobile",
                "present": visible,
                "target_class": "postcard" if visible else "none",
                "content_quad": normalized if visible else None,
                "outer_quad": None,
                "visible": visible,
                "occlusion": 0.0,
                "glare_level": "none",
                "in_scope": visible,
                "scene_type": "mobile_document" if visible else "partial_document",
                "source": "midv500",
            }
        )
    if not records:
        raise ValueError("MIDV-500 子集没有构建出 geometry 样本")
    return records


def _prepare_midv_holo(
    context: dict[str, Any],
    *,
    clips_per_cell: int,
) -> dict[str, object]:
    data_root: Path = context["data_root"]
    downloads = data_root / "downloads" / "p2" / "midv-holo"
    destination = data_root / "geometry" / "midv-holo" / "subset"
    downloads.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("metadata.tar", "markup.tar"):
        archive = downloads / filename
        component_ready = destination / f".{filename.removesuffix('.tar')}-ready"
        if not component_ready.is_file():
            _ensure_peak_capacity(context, _remote_size(f"{MIDV_HOLO_ROOT}/{filename}"))
            _download(f"{MIDV_HOLO_ROOT}/{filename}", archive)
            _extract_tar_safely(archive, destination)
            component_ready.write_text("ready\n", encoding="utf-8")
        if archive.is_file() and not context["keep_archives"]:
            archive.unlink()
    selected_clips = _select_midv_holo_clips(destination / "metadata", clips_per_cell)
    selection_path = destination / "selection.json"
    selection_path.write_text(
        json.dumps({"clips": sorted(selected_clips)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    images_archive = downloads / "images.tar"
    ready = destination / ".images-subset-ready"
    selection_digest = hashlib.sha256("\n".join(sorted(selected_clips)).encode("utf-8")).hexdigest()
    if not ready.is_file() or ready.read_text(encoding="utf-8").strip() != selection_digest:
        _ensure_peak_capacity(context, MIDV_HOLO_IMAGES_BYTES)
        _download(f"{MIDV_HOLO_ROOT}/images.tar", images_archive)
        _extract_selected_tar_prefixes(images_archive, destination, selected_clips)
        ready.write_text(selection_digest + "\n", encoding="utf-8")
    if images_archive.is_file() and not context["keep_archives"]:
        images_archive.unlink()
    manifest = data_root / "manifests" / "midv-holo.geometry.jsonl"
    records = _build_midv_holo_manifest(data_root, destination, selected_clips)
    _write_jsonl(manifest, records)
    return {
        "dataset": "midv-holo",
        "license": "CC-BY-SA-2.5; Generated Photos attribution requested by source",
        "selection": "first N clips per sample-kind×lighting×device×ID/passport cell",
        "selected_clips": len(selected_clips),
        "samples": len(records),
        "groups": len({record["group_id"] for record in records}),
        "manifest": str(manifest),
    }


def _select_midv_holo_clips(metadata_root: Path, clips_per_cell: int) -> set[str]:
    selected: set[str] = set()
    for metadata in sorted(metadata_root.glob("*.json")):
        values = json.loads(metadata.read_text(encoding="utf-8"))
        for lighting in sorted(values):
            for device in sorted(values[lighting]):
                paths = [str(path).strip("/") for path in values[lighting][device]]
                for document_kind in ("/ID/", "/passport/"):
                    cell = sorted(path for path in paths if document_kind in path)
                    selected.update(cell[:clips_per_cell])
    if not selected:
        raise ValueError("MIDV-Holo metadata 没有选出 clip")
    return selected


def _extract_selected_tar_prefixes(
    archive_path: Path,
    destination: Path,
    selected_clips: set[str],
) -> None:
    prefixes = tuple(f"{clip.rstrip('/')}/" for clip in sorted(selected_clips))
    extracted = 0
    with tarfile.open(archive_path, "r") as archive:
        for member in archive:
            normalized_name = member.name.lstrip("./")
            if not normalized_name.startswith(prefixes):
                continue
            _validate_archive_member(member.name, destination, member.issym() or member.islnk())
            archive.extract(member, destination)
            extracted += int(member.isfile())
            if extracted % 500 == 0:
                print(f"MIDV-Holo 已流式抽取 {extracted} 帧", file=sys.stderr)
    if extracted == 0:
        raise ValueError("MIDV-Holo images.tar 中未匹配受控 clip")


def _build_midv_holo_manifest(
    data_root: Path,
    root: Path,
    selected_clips: set[str],
) -> list[dict[str, Any]]:
    condition_lookup = _midv_holo_conditions(root / "metadata")
    records: list[dict[str, Any]] = []
    for clip in sorted(selected_clips):
        image_directory = root / clip
        markup_directory = root / clip.replace("images/", "markup/", 1)
        if not image_directory.is_dir():
            continue
        clip_parts = Path(clip).parts
        document_kind_index = next(
            index for index, value in enumerate(clip_parts) if value in {"ID", "passport"}
        )
        sample_kind = "/".join(clip_parts[1:document_kind_index])
        clip_name = Path(clip).name
        document_instance = "_".join(clip_name.split("_")[:2])
        # 同一虚构文档的 original/fraud/多光照变体必须在同一 split。
        group_id = f"midv-holo:{document_instance}"
        lighting, device = condition_lookup.get(clip, ("unknown", "unknown"))
        glare = "none" if lighting in {"A", "C"} else "strong" if lighting == "D" else "medium"
        for image in sorted(path for path in image_directory.iterdir() if path.is_file()):
            # 官方 markup 文件名保留图片扩展名，例如 ``img_0001.jpg.json``。
            truth = markup_directory / f"{image.name}.json"
            if not truth.is_file():
                continue
            quad = _find_quad(json.loads(truth.read_text(encoding="utf-8")))
            if quad is None:
                continue
            with Image.open(image) as opened:
                width, height = opened.size
            normalized = np_normalized_quad(quad, width, height)
            visible = all(0.0 <= coordinate <= 1.0 for point in normalized for coordinate in point)
            records.append(
                {
                    "image": image.resolve().relative_to(data_root).as_posix(),
                    "split": _split_for_group(group_id),
                    "group_id": group_id,
                    "capture_session": f"midv-holo:{sample_kind}:{clip_name}",
                    "device": f"midv-holo-{device}",
                    "present": visible,
                    "target_class": "postcard" if visible else "none",
                    "content_quad": normalized if visible else None,
                    "outer_quad": None,
                    "visible": visible,
                    "occlusion": 0.0,
                    "glare_level": glare,
                    "in_scope": visible,
                    "scene_type": f"reflective_document_lighting_{lighting}",
                    "source": "midv-holo",
                }
            )
    if not records:
        raise ValueError("MIDV-Holo 子集没有构建出 geometry 样本；请检查 markup 路径")
    return records


def _midv_holo_conditions(metadata_root: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for metadata in sorted(metadata_root.glob("*.json")):
        values = json.loads(metadata.read_text(encoding="utf-8"))
        for lighting, devices in values.items():
            for device, paths in devices.items():
                for path in paths:
                    result[str(path).strip("/")] = (str(lighting), str(device))
    return result


def _prepare_met(context: dict[str, Any], *, count: int) -> dict[str, object]:
    data_root: Path = context["data_root"]
    root = data_root / "textures" / "met-open-access"
    image_directory = root / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.jsonl"
    existing = _read_jsonl(metadata_path)
    records_by_id = {int(record["object_id"]): record for record in existing}
    queries = ("painting", "print", "photograph")
    candidates: dict[str, list[int]] = {}
    for query in queries:
        response = _request_json(f"{MET_API_ROOT}/search?{urllib.parse.urlencode({'hasImages': 'true', 'q': query})}")
        candidates[query] = [int(value) for value in response.get("objectIDs") or []]
    attempted: set[int] = set(records_by_id)
    query_index = 0
    checkpoint_bucket = len(records_by_id) // 25
    # Met 对象元数据和公开缩略图彼此独立。限制为 4 个并发请求，显著缩短准备时间，
    # 同时避免无节制地冲击公开 API；每批最多等于剩余目标数，不产生超额孤儿图片。
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while len(records_by_id) < count:
            batch: list[tuple[int, str]] = []
            batch_limit = min(12, count - len(records_by_id))
            while len(batch) < batch_limit:
                query = queries[query_index % len(queries)]
                query_index += 1
                available = candidates[query]
                object_id = next((value for value in available if value not in attempted), None)
                if object_id is None:
                    if all(all(value in attempted for value in candidates[name]) for name in queries):
                        break
                    continue
                attempted.add(object_id)
                batch.append((object_id, query))
            if not batch:
                break
            futures = [
                executor.submit(
                    _prepare_met_object,
                    data_root,
                    image_directory,
                    object_id,
                    query,
                )
                for object_id, query in batch
            ]
            for future in futures:
                record = future.result()
                if record is not None:
                    records_by_id[int(record["object_id"])] = record
            current_bucket = len(records_by_id) // 25
            if current_bucket > checkpoint_bucket:
                checkpoint_bucket = current_bucket
                _write_jsonl(metadata_path, list(records_by_id.values()))
                _enforce_final_budget(context)
                print(f"Met Open Access {len(records_by_id)}/{count}", file=sys.stderr)
    _write_jsonl(metadata_path, list(records_by_id.values()))
    if len(records_by_id) < count:
        raise RuntimeError(f"Met 只获得 {len(records_by_id)}/{count} 个合格对象")
    return {
        "dataset": "met-open-access",
        "license": "CC0 for selected public-domain object images",
        "selection": "round-robin painting/print/photograph; isPublicDomain=true; primary image required",
        "images": len(records_by_id),
        "metadata": str(metadata_path),
    }


def _prepare_met_object(
    data_root: Path,
    image_directory: Path,
    object_id: int,
    query: str,
) -> dict[str, object] | None:
    try:
        value = _request_json(f"{MET_API_ROOT}/objects/{object_id}")
    except (RuntimeError, urllib.error.URLError):
        return None
    image_url = str(value.get("primaryImageSmall") or value.get("primaryImage") or "")
    if not value.get("isPublicDomain") or not image_url:
        return None
    suffix = Path(urllib.parse.urlparse(image_url).path).suffix.lower() or ".jpg"
    if suffix not in {".jpeg", ".jpg", ".png"}:
        suffix = ".jpg"
    image_path = image_directory / f"{object_id}{suffix}"
    try:
        _download_http(image_url, image_path)
        with Image.open(image_path) as opened:
            width, height = opened.size
    except (OSError, RuntimeError, urllib.error.URLError):
        image_path.unlink(missing_ok=True)
        return None
    return {
        "object_id": object_id,
        "query_bucket": query,
        "title": str(value.get("title") or "")[:200],
        "classification": str(value.get("classification") or "")[:120],
        "object_name": str(value.get("objectName") or "")[:120],
        "is_public_domain": True,
        "image": image_path.relative_to(data_root).as_posix(),
        "primary_image": str(value.get("primaryImage") or ""),
        "object_url": str(value.get("objectURL") or ""),
        "width": width,
        "height": height,
        "aspect_ratio": round(width / max(1, height), 6),
        "license": "CC0/Open Access",
    }


def _prepare_coco(context: dict[str, Any]) -> dict[str, object]:
    data_root: Path = context["data_root"]
    downloads = data_root / "downloads" / "p2" / "coco"
    destination = data_root / "backgrounds" / "coco"
    archive = downloads / "val2017.zip"
    ready = destination / ".val2017-ready"
    downloads.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    if not ready.is_file():
        _ensure_peak_capacity(context, COCO_VAL_BYTES)
        _download(COCO_VAL_URL, archive)
        _extract_zip_safely(archive, destination)
        if not (destination / "val2017").is_dir():
            raise ValueError("COCO val2017 解压结构异常")
        ready.write_text("ready\n", encoding="utf-8")
    if archive.is_file() and not context["keep_archives"]:
        archive.unlink()
    images = sum(1 for path in (destination / "val2017").glob("*.jpg") if path.is_file())
    return {
        "dataset": "coco-2017-val",
        "license": "image-level Flickr licenses and COCO terms apply",
        "selection": "official val2017 images only; train2017 excluded",
        "images": images,
        "directory": str(destination / "val2017"),
    }


def _request_json(url: str) -> dict[str, Any]:
    for attempt in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": "ScreenRestore-P2/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                value = json.load(response)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            if attempt == 4:
                raise RuntimeError(f"API 请求失败：{url}") from exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"API 响应不是对象：{url}")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # curl 的单进程 --retry 会沿用首次构造的参数：若文件起初不存在，连接在 99% 中断时
    # 下一次尝试仍会从 0 覆盖。每次失败后重建命令，才能根据当前部分文件真正续传。
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(9):
        command = [
            "curl",
            "--fail",
            "--location",
            "--connect-timeout",
            "45",
            # FTP 偶尔会在文件尾保持连接却不再发送字节；一分钟低于 1 KiB/s
            # 就主动失败，让外层循环按当前文件大小重新建立断点连接。
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
        ]
        if destination.is_file() and destination.stat().st_size:
            command.extend(["--continue-at", "-"])
        command.extend(["--output", str(destination), url])
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == 8:
                break
            print(
                f"下载连接中断，保留 {destination.stat().st_size if destination.exists() else 0} "
                f"字节并断点续传（{attempt + 1}/8）",
                file=sys.stderr,
            )
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"下载重试耗尽：{url}") from last_error


def _download_http(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    # Met 个别官方图片路径含空格；先只编码 URL 的 path/query，保留已有的百分号转义。
    request = urllib.request.Request(
        _encode_remote_url(url),
        headers={"User-Agent": "ScreenRestore-P2/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def _encode_remote_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe="/%:@")
    query = urllib.parse.quote(parts.query, safe="=&%:@/?+")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def _read_midv500_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 32:
            values[parts[1]] = parts[0].lower()
    missing = set(MIDV500_SUBSET) - values.keys()
    if missing:
        raise ValueError(f"MIDV-500 官方 md5.txt 缺少：{sorted(missing)}")
    return values


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_size(url: str) -> int:
    result = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--head", url],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"远端未提供 Content-Length：{url}")


def _extract_zip_safely(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP 校验失败：{archive_path.name}")
        for member in archive.infolist():
            _validate_archive_member(member.filename, destination, False)
            archive.extract(member, destination)


def _extract_tar_safely(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r") as archive:
        for member in archive:
            _validate_archive_member(member.name, destination, member.issym() or member.islnk())
            archive.extract(member, destination)


def _validate_archive_member(name: str, destination: Path, is_link: bool) -> None:
    if is_link:
        raise ValueError("公开数据归档不得包含链接")
    target = (destination / name).resolve()
    if not target.is_relative_to(destination.resolve()) or "private" in target.relative_to(destination.resolve()).parts:
        raise ValueError(f"归档包含不安全路径：{name!r}")


def _matching_image(stem: Path) -> Path | None:
    for suffix in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        candidate = stem.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _find_quad(value: Any) -> list[list[float]] | None:
    if isinstance(value, dict):
        for key in ("quad", "template_quad", "quadrangle", "points"):
            quad = value.get(key)
            if _is_quad(quad):
                return [[float(coordinate) for coordinate in point] for point in quad]
        for nested in value.values():
            found = _find_quad(nested)
            if found is not None:
                return found
    return None


def _is_quad(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(point, list) and len(point) == 2 for point in value)
    )


def np_normalized_quad(
    quad: list[list[float]],
    width: int,
    height: int,
) -> list[list[float]]:
    return [
        [float(point[0]) / max(1, width - 1), float(point[1]) / max(1, height - 1)]
        for point in quad
    ]


def _split_for_group(group_id: str) -> str:
    bucket = hashlib.sha256(group_id.encode("utf-8")).digest()[0] % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _ensure_peak_capacity(context: dict[str, Any], incoming_bytes: int) -> None:
    data_root: Path = context["data_root"]
    required_free = incoming_bytes + context["minimum_free_bytes"]
    if shutil.disk_usage(data_root).free < required_free:
        raise RuntimeError(
            f"磁盘空间不足：本步至少需要 {required_free / GIB:.2f} GiB 可用空间"
        )
    _enforce_final_budget(context)


def _enforce_final_budget(context: dict[str, Any]) -> None:
    current = _public_usage(context["data_root"])
    added = current - context["initial_public_bytes"]
    if added > context["max_new_bytes"]:
        raise RuntimeError(
            f"P2 新增公开数据 {added / GIB:.2f} GiB 已超过预算 "
            f"{context['max_new_bytes'] / GIB:.2f} GiB"
        )


def _public_usage(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "private":
            continue
        # downloads/p2 只保存可恢复的传输中归档，成功抽取后会删除；最终数据预算只核算
        # 持久化样本，峰值空间另由 _ensure_peak_capacity 按真实磁盘余量约束。
        if relative.parts[:2] == ("downloads", "p2"):
            continue
        total += path.stat().st_size
    return total


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
