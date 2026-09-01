"""为 P3 faithful restoration 生成非 cherry-pick contact sheets。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/render_p3_restoration_contact_sheets.py \
      --task demoire --checkpoint "$SCREENRESTORE_RUN_ROOT/<run>/demoire/best.pt" \
      --hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_valid_HR" \
      --output-directory "$SCREENRESTORE_RUN_ROOT/<run>/demoire/contact-sheets"

脚本使用固定 synthetic validation seed，按数值误差自动选择 best/median/worst、最大 identity
regression 和 strong residual；Router 另外输出 false-positive/false-negative。每行固定显示 input、
output、GT、absolute difference 与可选 artifact mask，不手选成功案例。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# 支持按文档直接执行脚本，不依赖 editable install 是否刷新到当前工作树。
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from training.p3.models import (  # noqa: E402
    ArtifactRouterNet,
    PhotometricNet,
    ReflectionNet,
)
from training.p3.train_specialist import P3TrainingDataset, _model  # noqa: E402
from training.restoration.dataset import Div2kHrDataset  # noqa: E402
from training.restoration.degradation import CameraDegradationConfig  # noqa: E402
from training.restoration.model import FidelityNetV2  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("fidelity", "photometric", "demoire", "reflection", "router"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hr-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)
    if args.samples < 3 or args.patch_size < 64 or args.patch_size % 8:
        raise ValueError("samples 至少为 3，patch-size 必须不小于 64 且为 8 的倍数")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前 MPS 不可用")
    output = args.output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖 contact sheets：{output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    model = _load_model(args.task, args.checkpoint).to(device).eval()
    loader = DataLoader(_dataset(args), batch_size=1, shuffle=False)
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            _progress(index - 1, len(loader))
            image = batch["input"].to(device)
            if args.task == "router":
                assert isinstance(model, ArtifactRouterNet)
                logits, severity = model(image)
                probabilities = torch.sigmoid(logits)
                labels = batch["labels"].to(device)
                false_positive = float((probabilities * (1.0 - labels)).max())
                false_negative = float(((1.0 - probabilities) * labels).max())
                output_image = image
                target = image
                identity_error = 0.0
                score = float(torch.nn.functional.binary_cross_entropy(probabilities, labels))
                router = {
                    "probabilities": probabilities[0].cpu().tolist(),
                    "severity": severity[0].cpu().tolist(),
                    "labels": labels[0].cpu().tolist(),
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                }
            else:
                target = batch["target"].to(device)
                output_image = _restore(model, args.task, image)
                clean_output = _restore(model, args.task, target)
                score = float(torch.mean(torch.abs(output_image - target)))
                identity_error = float(torch.mean(torch.abs(clean_output - target)))
                router = None
            residual = float(torch.mean(torch.abs(output_image - image)))
            records.append(
                {
                    "index": index - 1,
                    "score": score,
                    "identity_error": identity_error,
                    "residual": residual,
                    "router": router,
                    "row": _row(batch, image, output_image, target),
                }
            )
    _progress(len(loader), len(loader))
    categories = _categories(records, args.task)
    for name, selected in categories.items():
        Image.fromarray(selected["row"]).save(output / f"{name}.png")  # type: ignore[arg-type]
    report = {
        "format_version": 1,
        "kind": "p3_restoration_contact_sheets",
        "task": args.task,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "samples": len(records),
        "selection": {
            name: {
                key: value
                for key, value in record.items()
                if key != "row"
            }
            for name, record in categories.items()
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def _dataset(args: argparse.Namespace) -> torch.utils.data.Dataset[dict[str, torch.Tensor]]:
    if args.task == "fidelity":
        return Div2kHrDataset(
            args.hr_directory,
            patch_size=args.patch_size,
            degradation=CameraDegradationConfig(),
            seed=args.seed,
            max_samples=args.samples,
            preserve_photometric_nuisance=True,
        )
    return P3TrainingDataset(
        args.task,
        data_root=None,
        hr_directory=args.hr_directory,
        manifest=None,
        split="validation",
        patch_size=args.patch_size,
        samples=args.samples,
        seed=args.seed,
    )


def _load_model(task: str, path: Path) -> torch.nn.Module:
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if task == "fidelity":
        if checkpoint.get("model") != "FidelityNetV2":
            raise ValueError("Fidelity contact sheet 要求 FidelityNetV2 checkpoint")
        model: torch.nn.Module = FidelityNetV2(
            int(checkpoint["channels"]),
            max_delta=float(checkpoint["max_delta"]),
        )
    else:
        model = _model(task)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def _restore(model: torch.nn.Module, task: str, image: torch.Tensor) -> torch.Tensor:
    if task == "photometric":
        assert isinstance(model, PhotometricNet)
        return model.apply(image, model(image))
    if task == "reflection":
        assert isinstance(model, ReflectionNet)
        return model(image)[0]
    return model(image)  # type: ignore[no-any-return]


def _row(
    batch: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> np.ndarray:
    input_rgb = _rgb(input_tensor)
    output_rgb = _rgb(output_tensor)
    target_rgb = _rgb(target_tensor)
    difference = np.abs(output_rgb.astype(np.int16) - target_rgb.astype(np.int16)).astype(np.uint8)
    difference = np.clip(difference * 4, 0, 255).astype(np.uint8)
    mask = batch.get("mask")
    if mask is None:
        mask_rgb = np.zeros_like(input_rgb)
    else:
        value = np.rint(mask[0, 0].numpy() * 255).astype(np.uint8)
        mask_rgb = np.repeat(value[..., None], 3, axis=2)
    tiles = []
    for label, value in (
        ("input", input_rgb),
        ("output", output_rgb),
        ("GT", target_rgb),
        ("abs diff x4", difference),
        ("artifact mask", mask_rgb),
    ):
        tile = value.copy()
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 20), 1)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def _rgb(value: torch.Tensor) -> np.ndarray:
    return np.rint(
        value[0].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
    ).astype(np.uint8)


def _categories(
    records: list[dict[str, object]], task: str
) -> dict[str, dict[str, object]]:
    ordered = sorted(records, key=lambda item: float(item["score"]))
    output = {
        "best": ordered[0],
        "median": ordered[len(ordered) // 2],
        "worst": ordered[-1],
        "identity-regression": max(records, key=lambda item: float(item["identity_error"])),
        "strong-residual": max(records, key=lambda item: float(item["residual"])),
    }
    if task == "router":
        output["router-false-positive"] = max(
            records,
            key=lambda item: float((item["router"] or {}).get("false_positive", 0.0)),  # type: ignore[union-attr]
        )
        output["router-false-negative"] = max(
            records,
            key=lambda item: float((item["router"] or {}).get("false_negative", 0.0)),  # type: ignore[union-attr]
        )
    return output


def _progress(done: int, total: int) -> None:
    width = 24
    filled = round(width * done / max(1, total))
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} contact sheets",
        end="\n" if done >= total else "\r",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
