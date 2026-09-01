"""训练 P3 dewarp、Photometric、Demoire、Reflection 或 artifact router。

使用示例：
    python -m training.p3.train_specialist --task dewarp --budget ABLATION \
      --hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR" \
      --output-directory "$SCREENRESTORE_RUN_ROOT/p3-dewarp-ablation" --device mps

FULL 的 Demoire 和 Reflection 必须提供已审计的 P3 restoration manifest；缺少真实配对时
会明确失败。Router 可由统一在线退化追踪监督。SMOKE/ABLATION 可用在线 synthetic 验证代码，
但不能形成真实能力声明。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from screenrestore.geometry import dense_grid_inverse_map, remap_original_once
from training.p3.degradations import (
    mild_dewarp_grid,
    synthetic_reflection,
    synthetic_screen_recapture,
)
from training.p3.losses import dewarp_grid_loss
from training.p3.models import (
    ArtifactRouterNet,
    DemoireNet,
    DewarpGridNet,
    PhotometricNet,
    ReflectionNet,
)
from training.restoration.degradation import factorized_degradation_graph

_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class P3TrainingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        task: str,
        *,
        data_root: Path | None,
        hr_directory: Path | None,
        manifest: Path | None,
        split: str,
        patch_size: int,
        samples: int,
        seed: int,
    ) -> None:
        self.task = task
        self.data_root = data_root.expanduser().resolve() if data_root else None
        self.patch_size = patch_size
        self.samples = samples
        self.seed = seed
        self.records = _read_manifest(manifest, split) if manifest else []
        self.paths = (
            [path for path in sorted(hr_directory.expanduser().resolve().rglob("*")) if path.suffix.lower() in _SUFFIXES]
            if hr_directory
            else []
        )
        if self.records and self.data_root is None:
            raise ValueError("使用 manifest 时必须提供 --data-root")
        if not self.records and not self.paths:
            raise ValueError("必须提供含有效样本的 --manifest 或 --hr-directory")
        if samples <= 0:
            self.samples = len(self.records) if self.records else len(self.paths)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, index)))
        if self.records:
            return self._from_record(self.records[index % len(self.records)], rng)
        clean = _load_crop(self.paths[index % len(self.paths)], self.patch_size, rng)
        if self.task == "dewarp":
            grid, _trace = mild_dewarp_grid(17, 17, rng)
            inverse = dense_grid_inverse_map(grid, clean.shape[:2])
            warped = remap_original_once(clean, inverse, interpolation=cv2.INTER_CUBIC)
            return {"input": _tensor(warped), "target": _tensor(clean), "grid": torch.from_numpy(grid)}
        if self.task == "photometric":
            sample = factorized_degradation_graph(
                clean,
                task="photometric",
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            return {"input": _tensor(sample.input_rgb), "target": _tensor(sample.target_rgb)}
        if self.task == "demoire":
            pair = synthetic_screen_recapture(clean, rng)
            return {"input": _tensor(pair.input_rgb), "target": _tensor(pair.target_rgb)}
        if self.task == "reflection":
            pair = synthetic_reflection(clean, rng)
            mask = pair.mask if pair.mask is not None else np.zeros(clean.shape[:2], np.float32)
            unresolved = (
                pair.unresolved_mask
                if pair.unresolved_mask is not None
                else (pair.input_rgb.max(axis=2) >= 0.995).astype(np.float32) * mask
            )
            return {
                "input": _tensor(pair.input_rgb),
                "target": _tensor(pair.target_rgb),
                "mask": torch.from_numpy(mask[None]),
                "unresolved": torch.from_numpy(unresolved[None]),
            }
        if self.task == "router":
            # 30% clean 样本显式监督 bypass；其余单 artifact 样本直接由在线追踪给标签。
            label = int(rng.integers(0, 10))
            input_rgb, severity_value = _router_degradation(clean, label, rng)
            labels = torch.zeros(len(ArtifactRouterNet.labels))
            if label < len(labels):
                labels[label] = 1.0
            severity = labels * severity_value
            return {"input": _tensor(input_rgb), "labels": labels, "severity": severity}
        raise ValueError(f"未知 task：{self.task}")

    def _from_record(
        self, record: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        assert self.data_root is not None
        input_rgb = _load_crop(_safe_path(self.data_root, record["input_image"]), self.patch_size, rng)
        target_rgb = _load_crop(_safe_path(self.data_root, record["target_image"]), self.patch_size, rng)
        if self.task == "router":
            names = ArtifactRouterNet.labels
            labels = torch.tensor([float(name in record["artifact_labels"]) for name in names])
            severity = torch.tensor([float(record["artifact_severity"].get(name, 0.0)) for name in names])
            return {"input": _tensor(input_rgb), "labels": labels, "severity": severity}
        result = {"input": _tensor(input_rgb), "target": _tensor(target_rgb)}
        if self.task == "reflection":
            masks = record.get("artifact_masks", {})
            mask_path = masks.get("reflection") if isinstance(masks, dict) else None
            unresolved_path = masks.get("unresolved") if isinstance(masks, dict) else None
            result["mask"] = _mask_tensor(self.data_root, mask_path, self.patch_size)
            result["unresolved"] = _mask_tensor(self.data_root, unresolved_path, self.patch_size)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("dewarp", "photometric", "demoire", "reflection", "router"), required=True)
    parser.add_argument("--budget", choices=("SMOKE", "ABLATION", "FULL"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--hr-directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)
    if args.budget == "FULL" and args.device != "mps":
        raise RuntimeError("P3 FULL 正式训练必须显式使用 --device mps")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前 PyTorch MPS 不可用")
    if args.budget == "FULL" and args.task in {"demoire", "reflection"} and args.manifest is None:
        raise RuntimeError(f"{args.task} FULL 缺少已审计真实 P3 manifest，当前状态 BLOCKED")
    output = args.output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有 run：{output}")
    output.mkdir(parents=True, exist_ok=True)
    _seed(args.seed)
    device = torch.device(args.device)
    train_data = P3TrainingDataset(
        args.task,
        data_root=args.data_root,
        hr_directory=args.hr_directory,
        manifest=args.manifest,
        split="train",
        patch_size=args.patch_size,
        samples=args.train_samples,
        seed=args.seed,
    )
    validation_data = P3TrainingDataset(
        args.task,
        data_root=args.data_root,
        hr_directory=args.hr_directory,
        manifest=args.manifest,
        split="validation",
        patch_size=args.patch_size,
        samples=args.validation_samples,
        seed=args.seed + 1,
    )
    model = _model(args.task).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    started = time.monotonic()
    history: list[dict[str, float | int]] = []
    best = float("inf")
    steps = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, step_count = _epoch(model, train_loader, args.task, device, optimizer, epoch, args.epochs)
        validation_loss, _ = _epoch(model, validation_loader, args.task, device, None, epoch, args.epochs)
        steps += step_count
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        history.append(record)
        checkpoint = _checkpoint(model, args, epoch, record)
        torch.save(checkpoint, output / "last.pt")
        if validation_loss < best:
            best = validation_loss
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)
    run = {
        "format_version": 1,
        "kind": "p3_specialist_training",
        "task": args.task,
        "budget": args.budget,
        "device": str(device),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "seed": args.seed,
        "epochs": args.epochs,
        "samples": {"train": len(train_data), "validation": len(validation_data)},
        "steps": steps,
        "wall_time_seconds": round(time.monotonic() - started, 4),
        "peak_memory_bytes": _peak_memory(device),
        "git_commit": _git_commit(),
        "manifest_sha256": _sha256(args.manifest),
        "best_validation_loss": best,
    }
    (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    evaluation = _evaluate_frozen(model, validation_loader, args.task, device)
    (output / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "slices.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "kind": "p3_specialist_slices",
                "task": args.task,
                "slices": {"synthetic_or_manifest_validation": evaluation["metrics"]},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    task: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    epochs: int,
) -> tuple[float, int]:
    model.train(optimizer is not None)
    total = 0.0
    for index, batch in enumerate(loader, 1):
        _progress(index - 1, len(loader), f"epoch {epoch}/{epochs} {'train' if optimizer else 'val'}")
        batch = {key: value.to(device) for key, value in batch.items()}
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, task, batch)
        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach())
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} {'train' if optimizer else 'val'}")
    return total / max(1, len(loader)), len(loader)


def _loss(model: torch.nn.Module, task: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if task == "dewarp":
        predicted = model(batch["input"])
        return dewarp_grid_loss(predicted, batch["grid"])[0]
    if task == "photometric":
        assert isinstance(model, PhotometricNet)
        restored = model.apply(batch["input"], model(batch["input"]))
        return F.l1_loss(restored, batch["target"]) + 0.1 * F.l1_loss(_gradient(restored), _gradient(batch["target"]))
    if task == "reflection":
        transmission, mask, unresolved = model(batch["input"])
        localized = torch.mean(torch.abs(transmission - batch["input"]) * (1.0 - batch["mask"]))
        return F.l1_loss(transmission, batch["target"]) + F.binary_cross_entropy(mask, batch["mask"]) + F.binary_cross_entropy(unresolved, batch["unresolved"]) + 2.0 * localized
    if task == "router":
        logits, severity = model(batch["input"])
        return F.binary_cross_entropy_with_logits(logits, batch["labels"]) + F.smooth_l1_loss(severity, batch["severity"])
    restored = model(batch["input"])
    identity = model(batch["target"])
    return F.l1_loss(restored, batch["target"]) + 0.3 * F.l1_loss(identity, batch["target"]) + 0.1 * F.l1_loss(_gradient(restored), _gradient(batch["target"]))


def _model(task: str) -> torch.nn.Module:
    return {
        "dewarp": DewarpGridNet,
        "photometric": PhotometricNet,
        "demoire": DemoireNet,
        "reflection": ReflectionNet,
        "router": ArtifactRouterNet,
    }[task]()


def _evaluate_frozen(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    task: str,
    device: torch.device,
) -> dict[str, object]:
    """生成无需可选指标依赖的冻结验证；LPIPS 缺失会明确记录。"""

    model.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            _progress(index - 1, len(loader), f"{task} frozen evaluation")
            batch = {key: value.to(device) for key, value in batch.items()}
            if task == "dewarp":
                prediction = model(batch["input"])
                loss, parts = dewarp_grid_loss(prediction, batch["grid"])
                values = {"loss": float(loss), **parts}
            elif task == "router":
                logits, severity = model(batch["input"])
                probability = torch.sigmoid(logits)
                clean = batch["labels"].sum(dim=1) == 0
                values = {
                    "bce": float(F.binary_cross_entropy_with_logits(logits, batch["labels"])),
                    "severity_mae": float(torch.abs(severity - batch["severity"]).mean()),
                    "micro_accuracy_at_0_5": float(
                        ((probability >= 0.5) == (batch["labels"] >= 0.5)).float().mean()
                    ),
                    "clean_false_positive_rate_at_0_5": float(
                        (probability[clean] >= 0.5).any(dim=1).float().mean()
                    )
                    if bool(clean.any())
                    else 0.0,
                    "artifact_false_negative_rate_at_0_5": float(
                        ((probability < 0.5) & (batch["labels"] >= 0.5)).any(dim=1).float().mean()
                    ),
                }
            else:
                restored = _restore_for_evaluation(model, task, batch["input"])
                target = batch["target"]
                clean_restored = _restore_for_evaluation(model, task, target)
                values = _image_metrics(restored, target, clean_restored)
                if task == "demoire":
                    values["frequency_residual"] = float(
                        torch.mean(
                            torch.abs(
                                torch.fft.rfft2(restored, norm="ortho").abs()
                                - torch.fft.rfft2(target, norm="ortho").abs()
                            )
                        )
                    )
                    values["chroma_error"] = float(
                        torch.abs(
                            (restored - restored.mean(dim=1, keepdim=True))
                            - (target - target.mean(dim=1, keepdim=True))
                        ).mean()
                    )
                if task == "reflection":
                    outside = 1.0 - batch["mask"]
                    values["outside_mask_modification"] = float(
                        (torch.abs(restored - batch["input"]) * outside).mean()
                    )
                    values["unresolved_coverage"] = float(batch["unresolved"].mean())
                    values["reflection_residual"] = float(
                        (torch.abs(restored - target) * batch["mask"]).mean()
                    )
                if task == "photometric":
                    values["luminance_error"] = float(
                        torch.abs(restored.mean(dim=1) - target.mean(dim=1)).mean()
                    )
                    values["chroma_error"] = float(
                        torch.abs(
                            (restored - restored.mean(dim=1, keepdim=True))
                            - (target - target.mean(dim=1, keepdim=True))
                        ).mean()
                    )
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value)
    _progress(len(loader), len(loader), f"{task} frozen evaluation")
    metrics = {name: value / max(1, len(loader)) for name, value in totals.items()}
    return {
        "format_version": 1,
        "kind": "p3_specialist_evaluation",
        "task": task,
        "samples": len(loader.dataset),
        "metrics": metrics,
        "lpips": "NOT_AVAILABLE: optional metric dependency is not installed by default",
    }


def _restore_for_evaluation(
    model: torch.nn.Module,
    task: str,
    image: torch.Tensor,
) -> torch.Tensor:
    if task == "photometric":
        assert isinstance(model, PhotometricNet)
        return model.apply(image, model(image))
    if task == "reflection":
        assert isinstance(model, ReflectionNet)
        return model(image)[0]
    return model(image)  # type: ignore[no-any-return]


def _image_metrics(
    restored: torch.Tensor,
    target: torch.Tensor,
    clean_restored: torch.Tensor,
) -> dict[str, float]:
    mse = torch.mean((restored - target).square()).clamp_min(1e-12)
    mean_restored = restored.mean(dim=(-2, -1), keepdim=True)
    mean_target = target.mean(dim=(-2, -1), keepdim=True)
    variance_restored = ((restored - mean_restored) ** 2).mean(dim=(-2, -1))
    variance_target = ((target - mean_target) ** 2).mean(dim=(-2, -1))
    covariance = ((restored - mean_restored) * (target - mean_target)).mean(dim=(-2, -1))
    ssim = (
        (2 * mean_restored.flatten(1) * mean_target.flatten(1) + 0.01**2)
        * (2 * covariance + 0.03**2)
        / (
            (mean_restored.flatten(1).square() + mean_target.flatten(1).square() + 0.01**2)
            * (variance_restored + variance_target + 0.03**2)
        )
    ).mean()
    return {
        "psnr": float(-10.0 * torch.log10(mse)),
        "ssim": float(ssim),
        "mae": float(torch.abs(restored - target).mean()),
        "gradient_error": float(torch.abs(_gradient(restored) - _gradient(target)).mean()),
        "identity_drift": float(torch.abs(clean_restored - target).mean()),
    }


def _gradient(value: torch.Tensor) -> torch.Tensor:
    horizontal = value[:, :, :-1, 1:] - value[:, :, :-1, :-1]
    vertical = value[:, :, 1:, :-1] - value[:, :, :-1, :-1]
    return torch.cat((horizontal, vertical), dim=2)


def _read_manifest(path: Path | None, split: str) -> list[dict[str, Any]]:
    if path is None:
        return []
    return [json.loads(line) for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("split") == split]


def _load_crop(path: Path, size: int, rng: np.random.Generator) -> np.ndarray:
    with Image.open(path) as opened:
        image = np.asarray(opened.convert("RGB"), np.uint8)
    height, width = image.shape[:2]
    if min(height, width) < size:
        scale = size / min(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
    y = int(rng.integers(0, height - size + 1))
    x = int(rng.integers(0, width - size + 1))
    return np.ascontiguousarray(image[y : y + size, x : x + size].astype(np.float32) / 255.0)


def _safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("manifest 路径越出 data-root 或不存在")
    return path


def _mask_tensor(root: Path, value: object, size: int) -> torch.Tensor:
    if not isinstance(value, str):
        return torch.zeros((1, size, size))
    image = cv2.imread(str(_safe_path(root, value)), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((resized.astype(np.float32) / 255.0)[None])


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float()


def _router_degradation(
    clean: np.ndarray,
    label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """生成与 Router 七个输出一一对应的单 artifact 样本。"""

    if label == 0:  # noise
        standard_deviation = float(rng.uniform(0.004, 0.035))
        value = np.clip(
            clean + rng.normal(0.0, standard_deviation, clean.shape).astype(np.float32),
            0.0,
            1.0,
        )
        return value, min(1.0, standard_deviation / 0.035)
    if label == 1:  # blur
        sigma = float(rng.uniform(0.45, 2.5))
        return cv2.GaussianBlur(clean, (0, 0), sigma), min(1.0, sigma / 2.5)
    if label == 2:  # jpeg
        quality = int(rng.integers(35, 86))
        bgr = cv2.cvtColor(np.rint(clean * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Router synthetic JPEG 编码失败")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        value = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return value, (100.0 - quality) / 65.0
    if label == 3:  # photometric
        sample = factorized_degradation_graph(
            clean,
            task="photometric",
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        severities = sample.trace.to_dict()["severity"]
        assert isinstance(severities, dict)
        return sample.input_rgb, float(max(severities.values(), default=0.0))
    if label == 4:  # reflection
        pair = synthetic_reflection(clean, rng)
        return pair.input_rgb, float(pair.trace["severity"])
    if label == 5:  # moire
        pair = synthetic_screen_recapture(clean, rng)
        return pair.input_rgb, float(pair.trace["severity"])
    if label == 6:  # dewarp
        grid, trace = mild_dewarp_grid(17, 17, rng)
        inverse = dense_grid_inverse_map(grid, clean.shape[:2])
        value = remap_original_once(clean, inverse, interpolation=cv2.INTER_CUBIC)
        return value, min(1.0, float(trace["amplitude"]) / 0.06)
    return clean.copy(), 0.0


def _checkpoint(model: torch.nn.Module, args: argparse.Namespace, epoch: int, metrics: dict[str, float | int]) -> dict[str, object]:
    return {"format_version": 1, "task": args.task, "model": type(model).__name__, "epoch": epoch, "metrics": metrics, "state_dict": model.state_dict()}


def _sha256(path: Path | None) -> str | None:
    return hashlib.sha256(path.expanduser().resolve().read_bytes()).hexdigest() if path else None


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _peak_memory(device: torch.device) -> int | None:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return None


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    filled = round(width * done / max(1, total))
    print(f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} {message}", end="\n" if done >= total else "\r", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
