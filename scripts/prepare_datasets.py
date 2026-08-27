"""下载、校验并解压官方公开训练数据；本脚本从不扫描 private。

使用范例：
    source .venv/bin/activate
    which python
    export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
    python scripts/prepare_datasets.py --dataset smartdoc
    python scripts/prepare_datasets.py --dataset div2k --include-lr-x2
    python scripts/prepare_datasets.py --dataset all --include-lr-x2 --build-manifests

下载固定写入 ``$SCREENRESTORE_DATA_ROOT/downloads``，支持 aria2c 或 curl 断点续传。
SmartDoc 始终先下载官方 sha256.chksum 并逐文件校验。压缩包仅在安全解压、文件数量
检查和可选清单构建完成后才会删除。为避免读取用户图片，容量计算只遍历公开目录，并把
private 保留空间作为显式预算参数处理。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SMARTDOC_RELEASE = "https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0"
DIV2K_ROOT = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
GIB = 1024**3


@dataclass(frozen=True, slots=True)
class DownloadAsset:
    """公开下载对象及其保守容量预估。"""

    dataset: str
    filename: str
    url: str
    destination: Path
    extract_to: Path
    archive_bytes: int
    extracted_bytes: int
    sha256: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--dataset", choices=("smartdoc", "div2k", "all"), default="all")
    parser.add_argument(
        "--smartdoc-assets",
        choices=("all", "frames", "models"),
        default="all",
        help="可在 frames 仍下载时独立校验、解压已完成的 models。",
    )
    parser.add_argument("--include-lr-x2", action="store_true", help="额外下载官方 DIV2K x2 bicubic LR。")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--build-manifests", action="store_true")
    parser.add_argument("--manifest-frame-stride", type=int, default=1)
    parser.add_argument("--data-budget-gib", type=float, default=30.0)
    parser.add_argument(
        "--private-reserve-gib",
        type=float,
        default=10.0,
        help="不读取 private 时为其预留的容量；默认 10 GiB，按实际已知大小提高。",
    )
    parser.add_argument("--minimum-free-gib", type=float, default=10.0)
    parser.add_argument("--download-only", action="store_true", help="仅下载校验，不解压。")
    args = parser.parse_args(argv)
    if args.data_budget_gib <= 0 or args.private_reserve_gib < 0 or args.minimum_free_gib < 0:
        raise ValueError("所有容量参数必须为非负数，data-budget-gib 必须大于 0")
    if args.private_reserve_gib >= args.data_budget_gib:
        raise ValueError("private-reserve-gib 必须小于 data-budget-gib")
    if args.manifest_frame_stride < 1:
        raise ValueError("manifest-frame-stride 必须大于 0")

    root = _public_root(args.data_root)
    _ensure_layout(root)
    assets = _select_assets(root, args.dataset, args.include_lr_x2, args.smartdoc_assets)
    expected = _expected_smartdoc_checksums(root) if args.dataset in {"smartdoc", "all"} else {}
    reports: list[dict[str, object]] = []
    for asset in assets:
        checksum = expected.get(asset.filename, asset.sha256)
        _check_budget(root, asset, args)
        # 有官方 hash 时严格按 hash 判断；DIV2K 未发布可用 hash，改用 archive
        # 结构校验识别已完成的本地文件，避免把用户已下载的完整归档重新下载一遍。
        complete = _is_complete_archive(asset.destination, checksum)
        if not complete:
            _download(asset, root / "downloads")
            if checksum is not None:
                _verify_sha256(asset.destination, checksum)
        if args.download_only:
            reports.append(_asset_report(asset, "downloaded"))
            continue
        _extract(asset)
        _check_extraction(asset)
        if not args.keep_archives:
            asset.destination.unlink()
        reports.append(_asset_report(asset, "ready"))
    if args.build_manifests and not args.download_only:
        _build_manifests(root, args.dataset, args.manifest_frame_stride)
    report = {
        "timestamp_unix": int(time.time()),
        "data_root": str(root),
        "assets": reports,
        "public_bytes": _public_usage(root),
        "private_reserve_bytes": int(args.private_reserve_gib * GIB),
        "data_budget_bytes": int(args.data_budget_gib * GIB),
        "filesystem_free_bytes": shutil.disk_usage(root).free,
    }
    report_path = root / "manifests" / "download_inventory.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def _select_assets(
    root: Path,
    dataset: str,
    include_lr_x2: bool,
    smartdoc_assets: str,
) -> list[DownloadAsset]:
    downloads = root / "downloads"
    smartdoc = root / "geometry" / "smartdoc"
    div2k = root / "superres" / "div2k"
    assets: list[DownloadAsset] = []
    if dataset in {"smartdoc", "all"}:
        smartdoc_assets_to_download = [
                DownloadAsset(
                    "smartdoc",
                    "frames.tar.gz",
                    f"{SMARTDOC_RELEASE}/frames.tar.gz",
                    downloads / "frames.tar.gz",
                    smartdoc / "frames",
                    archive_bytes=1_019_404_933,
                    extracted_bytes=4 * GIB,
                ),
                DownloadAsset(
                    "smartdoc",
                    "models.tar.gz",
                    f"{SMARTDOC_RELEASE}/models.tar.gz",
                    downloads / "models.tar.gz",
                    smartdoc / "models",
                    archive_bytes=409_034_297,
                    extracted_bytes=2 * GIB,
                ),
            ]
        if smartdoc_assets != "all":
            smartdoc_assets_to_download = [
                asset for asset in smartdoc_assets_to_download if asset.filename.startswith(smartdoc_assets)
            ]
        assets.extend(smartdoc_assets_to_download)
    if dataset in {"div2k", "all"}:
        assets.extend(
            [
                _div2k_asset(div2k, downloads, "DIV2K_train_HR.zip", 3_530_603_713, 4 * GIB),
                _div2k_asset(div2k, downloads, "DIV2K_valid_HR.zip", 475_000_000, 600 * 1024**2),
            ]
        )
        if include_lr_x2:
            assets.extend(
                [
                    _div2k_asset(div2k, downloads, "DIV2K_train_LR_bicubic_X2.zip", 1_500 * 1024**2, 2 * GIB),
                    _div2k_asset(div2k, downloads, "DIV2K_valid_LR_bicubic_X2.zip", 220 * 1024**2, 300 * 1024**2),
                ]
            )
    return assets


def _div2k_asset(
    extract_to: Path,
    downloads: Path,
    filename: str,
    archive_bytes: int,
    extracted_bytes: int,
) -> DownloadAsset:
    return DownloadAsset(
        "div2k",
        filename,
        f"{DIV2K_ROOT}/{filename}",
        downloads / filename,
        extract_to,
        archive_bytes=archive_bytes,
        extracted_bytes=extracted_bytes,
    )


def _expected_smartdoc_checksums(root: Path) -> dict[str, str]:
    """从同一官方 release 下载校验表，并拒绝缺失或格式异常的值。"""

    destination = root / "downloads" / "smartdoc-sha256.chksum"
    asset = DownloadAsset(
        "smartdoc",
        "sha256.chksum",
        f"{SMARTDOC_RELEASE}/sha256.chksum",
        destination,
        root / "downloads",
        archive_bytes=160,
        extracted_bytes=0,
    )
    _download(asset, root / "downloads")
    values: dict[str, str] = {}
    for line in destination.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64 or any(char not in "0123456789abcdef" for char in parts[0]):
            raise ValueError("官方 SmartDoc sha256.chksum 格式异常")
        values[parts[1]] = parts[0]
    required = {"frames.tar.gz", "models.tar.gz"}
    if required - set(values):
        raise ValueError("官方 SmartDoc sha256.chksum 缺少必需文件")
    return values


def _download(asset: DownloadAsset, downloads: Path) -> None:
    downloads.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c"):
        command = [
            "aria2c",
            "-x",
            "8",
            "-s",
            "8",
            "-k",
            "8M",
            "--continue=true",
            "--file-allocation=none",
            "--dir",
            str(asset.destination.parent),
            "--out",
            asset.destination.name,
            asset.url,
        ]
    elif shutil.which("curl"):
        command = [
            "curl",
            "-4",
            "--fail",
            "--location",
            "--retry",
            "10",
            "--retry-delay",
            "3",
            "--retry-all-errors",
        ]
        # 某些 GitHub release 首次请求会拒绝 ``Range: bytes=0-``；仅对已有部分文件续传。
        if asset.destination.exists() and asset.destination.stat().st_size > 0:
            command.extend(["--continue-at", "-"])
        command.extend(["--output", str(asset.destination), asset.url])
    else:
        raise RuntimeError("需要 aria2c 或 curl 以支持可恢复的公开数据下载")
    subprocess.run(command, check=True)


def _verify_sha256(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 不匹配：{path.name}，期望 {expected}，实际 {actual}")


def _sha256_matches(path: Path, expected: str) -> bool:
    return _sha256(path) == expected


def _is_complete_archive(path: Path, checksum: str | None) -> bool:
    """判断归档是否可安全复用，绝不把部分下载误作完整数据。"""

    if not path.is_file():
        return False
    if checksum is not None:
        return _sha256_matches(path, checksum)
    try:
        if path.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(path, "r:gz") as archive:
                # 读取到末尾会触发 gzip CRC 校验，避免仅凭文件存在跳过下载。
                for _member in archive:
                    pass
            return True
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                return archive.testzip() is None
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(asset: DownloadAsset) -> None:
    asset.extract_to.mkdir(parents=True, exist_ok=True)
    if asset.destination.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(asset.destination, "r:gz") as archive:
            _safe_extract_members(archive.getmembers(), asset.extract_to, lambda member: archive.extract(member, asset.extract_to))
    elif asset.destination.suffix.lower() == ".zip":
        with zipfile.ZipFile(asset.destination) as archive:
            _safe_extract_members(archive.infolist(), asset.extract_to, lambda member: archive.extract(member, asset.extract_to))
    else:
        raise ValueError(f"未知 archive 格式：{asset.destination}")


def _safe_extract_members(members: list[object], destination: Path, extract: Callable[[object], None]) -> None:
    """逐项验证路径，防止压缩包目录穿越覆盖 public 数据根外的文件。"""

    destination = destination.resolve()
    for member in members:
        is_symbolic_link = callable(getattr(member, "issym", None)) and member.issym()
        is_hard_link = callable(getattr(member, "islnk", None)) and member.islnk()
        if is_symbolic_link or is_hard_link:
            raise ValueError("压缩包不得包含符号链接或硬链接")
        name = getattr(member, "name", "") or getattr(member, "filename", "")
        target = (destination / name).resolve()
        if not target.is_relative_to(destination) or "private" in target.relative_to(destination).parts:
            raise ValueError(f"压缩包包含不安全路径：{name!r}")
        extract(member)


def _check_extraction(asset: DownloadAsset) -> None:
    if asset.dataset == "smartdoc" and asset.filename == "frames.tar.gz":
        metadata = asset.extract_to / "metadata.csv.gz"
        images = sum(1 for _ in asset.extract_to.glob("background*/*/*.jpeg"))
        if not metadata.is_file() or images < 1_000:
            raise ValueError("SmartDoc frames 解压不完整：缺少 metadata 或有效帧数量异常")
    elif asset.dataset == "smartdoc" and asset.filename == "models.tar.gz":
        images = sum(1 for _ in (asset.extract_to / "05-corrected-nexus-scaled33").glob("*.png"))
        if images < 30:
            raise ValueError("SmartDoc models 解压不完整：推荐参考模型数量异常")
    elif asset.dataset == "div2k":
        archive_stem = asset.filename.removesuffix(".zip")
        # 官方 x2 archive 文件名带 ``_X2``，但 zip 内目录是 ``..._bicubic/X2``。
        directory = (
            asset.extract_to / archive_stem.removesuffix("_X2") / "X2"
            if archive_stem.endswith("_LR_bicubic_X2")
            else asset.extract_to / archive_stem
        )
        if not directory.is_dir() or not any(directory.rglob("*.png")):
            raise ValueError(f"DIV2K 解压不完整：{directory.name}")


def _check_budget(root: Path, asset: DownloadAsset, args: argparse.Namespace) -> None:
    public_limit = int((args.data_budget_gib - args.private_reserve_gib) * GIB)
    current = _public_usage(root)
    archive_present = asset.destination.stat().st_size if asset.destination.exists() else 0
    # 同时保守预留 archive 与解压阶段峰值，压缩包删除前也不会突破公开额度。
    required_public = current + max(0, asset.archive_bytes - archive_present) + asset.extracted_bytes
    if required_public > public_limit:
        raise RuntimeError(
            f"公开数据预算不足：需要约 {required_public / GIB:.2f} GiB，"
            f"但 public 上限为 {public_limit / GIB:.2f} GiB；private 未被读取。"
        )
    free_required = max(0, asset.archive_bytes - archive_present) + asset.extracted_bytes + int(args.minimum_free_gib * GIB)
    if shutil.disk_usage(root).free < free_required:
        raise RuntimeError("磁盘可用空间不足，无法保留 minimum-free-gib 并安全下载/解压")


def _public_usage(root: Path) -> int:
    """只统计公开目录，显式跳过 private，避免触达用户未授权图片。"""

    total = 0
    for name in ("geometry", "reflection", "superres", "downloads", "manifests", ".download-cache"):
        directory = root / name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    return total


def _ensure_layout(root: Path) -> None:
    _public_root(root)
    for relative in (
        "geometry/smartdoc",
        "reflection/sir2",
        "superres/div2k",
        "superres/realsr",
        "downloads",
        "manifests",
        # 仅创建约定目录，不列举、读取或修改 private 下的任何文件。
        "private/gallery",
        "private/postcard",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _build_manifests(root: Path, dataset: str, frame_stride: int) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("build_dataset_manifests.py")),
        "--data-root",
        str(root),
        "--dataset",
        dataset,
        "--frame-stride",
        str(frame_stride),
    ]
    subprocess.run(command, check=True)


def _asset_report(asset: DownloadAsset, status: str) -> dict[str, object]:
    return {"dataset": asset.dataset, "filename": asset.filename, "status": status, "destination": str(asset.extract_to)}


def _default_data_root() -> Path:
    return Path(os.environ.get("SCREENRESTORE_DATA_ROOT", "~/screenrestore-data")).expanduser()


def _public_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if _contains_user_private_component(resolved):
        raise ValueError("数据准备工具拒绝读取或写入 private 目录")
    return resolved


def _contains_user_private_component(path: Path) -> bool:
    """拒绝数据根内的 private；仅放行 macOS 系统临时目录固定的 /private/var 前缀。"""

    parts = path.parts
    start = 3 if parts[:3] == ("/", "private", "var") else 0
    return "private" in parts[start:]


if __name__ == "__main__":
    raise SystemExit(main())
