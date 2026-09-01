"""P3 模型、统一解码与 ONNX Runtime 的小型 smoke。

使用示例：
    source .venv/bin/activate
    which python
    python -m training.p3.smoke --output-directory /tmp/screenrestore-p3-smoke --device cpu

该入口只使用随机小张量，执行一次前向/反向和 ONNX 数值核对，不读取正式训练清单，
也不会启动耗时训练。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from screenrestore.geometry.decoder import CornerDecoderSpec, decode_corner_logits
from training.p3.models import (
    ArtifactRouterNet,
    DemoireNet,
    DewarpGridNet,
    PhotometricNet,
    ReflectionNet,
)
from training.quadlocator.decoder import local_softargmax_corners
from training.restoration.model import FidelityNetV2


class _ReflectionExport(torch.nn.Module):
    def __init__(self, model: ReflectionNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(image)


class _RouterExport(torch.nn.Module):
    def __init__(self, model: ArtifactRouterNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(image)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--baseline-onnx", type=Path)
    args = parser.parse_args(argv)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("请求了 MPS smoke，但当前 PyTorch MPS 不可用")
    output = args.output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有 smoke 目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    started = time.monotonic()

    decoder_report = _decoder_parity()
    model_specs: list[tuple[str, torch.nn.Module, tuple[str, ...]]] = [
        ("dewarp", DewarpGridNet(), ("grid",)),
        ("fidelity", FidelityNetV2(), ("restored",)),
        ("photometric", PhotometricNet(), ("parameters",)),
        ("demoire", DemoireNet(), ("restored",)),
        (
            "reflection",
            _ReflectionExport(ReflectionNet()),
            ("transmission", "reflection_mask", "unresolved_mask"),
        ),
        ("router", _RouterExport(ArtifactRouterNet()), ("artifact_logits", "severity")),
    ]
    reports: dict[str, object] = {}
    for index, (name, model, output_names) in enumerate(model_specs, start=1):
        print(f"[{'#' * (index * 24 // len(model_specs))}{'-' * (24 - index * 24 // len(model_specs))}] {index}/{len(model_specs)} {name}")
        reports[name] = _smoke_model(name, model, output_names, output, device)
    baseline_report = _baseline_onnx(args.baseline_onnx) if args.baseline_onnx else None
    report = {
        "kind": "p3_smoke",
        "device": str(device),
        "seed": args.seed,
        "decoder": decoder_report,
        "models": reports,
        "baseline_onnx": baseline_report,
        "wall_time_seconds": round(time.monotonic() - started, 4),
    }
    (output / "smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[########################] 完成：{output / 'smoke.json'}")
    return 0


def _decoder_parity() -> dict[str, object]:
    logits = torch.full((1, 4, 16, 16), -8.0)
    for index, (x, y) in enumerate(((2, 3), (13, 2), (12, 13), (3, 12))):
        logits[0, index, y, x] = 7.0
        logits[0, index, y, min(15, x + 1)] = 5.0
    torch_coordinates = local_softargmax_corners(logits)[0].numpy()
    numpy_decoded = decode_corner_logits(logits.numpy(), CornerDecoderSpec())
    assert numpy_decoded.coordinates is not None
    numpy_coordinates = numpy_decoded.coordinates / 15.0
    maximum_error = float(np.max(np.abs(torch_coordinates - numpy_coordinates)))
    if maximum_error > 1e-6:
        raise AssertionError(f"Torch/NumPy decoder 漂移：{maximum_error}")
    return {
        "version": numpy_decoded.spec.version,
        "maximum_normalized_coordinate_error": maximum_error,
        "status": "PASS",
    }


def _smoke_model(
    name: str,
    model: torch.nn.Module,
    output_names: tuple[str, ...],
    output_directory: Path,
    device: torch.device,
) -> dict[str, object]:
    model = model.to(device).train()
    image = torch.rand((1, 3, 64, 64), device=device, requires_grad=True)
    outputs = model(image)
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    loss = sum(value.float().mean() for value in tensors)
    loss.backward()
    if image.grad is None or not torch.isfinite(image.grad).all():
        raise AssertionError(f"{name} backward 未产生有限梯度")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if name == "fidelity" and not 1_000_000 <= parameter_count <= 3_000_000:
        raise AssertionError(f"FidelityNetV2 参数量不在 1–3M：{parameter_count}")

    cpu_model = model.to("cpu").eval()
    example = image.detach().to("cpu")
    onnx_path = output_directory / f"{name}.onnx"
    torch.onnx.export(
        cpu_model,
        example,
        onnx_path,
        input_names=["image"],
        output_names=list(output_names),
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_output = cpu_model(example)
    expected = torch_output if isinstance(torch_output, tuple) else (torch_output,)
    actual = session.run(None, {"image": example.numpy()})
    maximum_error = max(
        float(np.max(np.abs(value.detach().numpy() - observed)))
        for value, observed in zip(expected, actual, strict=True)
    )
    if maximum_error > 2e-4:
        raise AssertionError(f"{name} ONNX 数值漂移：{maximum_error}")
    return {
        "parameter_count": parameter_count,
        "output_shapes": [list(value.shape) for value in expected],
        "onnx_max_abs_error": maximum_error,
        "onnx": str(onnx_path),
        "status": "PASS",
    }


def _baseline_onnx(path: Path) -> dict[str, object]:
    import onnxruntime as ort

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"B0 ONNX 不存在：{resolved}")
    session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    size = int(input_meta.shape[-1])
    outputs = session.run(None, {input_meta.name: np.zeros((1, 3, size, size), np.float32)})
    return {
        "path": str(resolved),
        "input_size": size,
        "output_count": len(outputs),
        "status": "PASS" if len(outputs) == 7 else "FAIL",
    }


if __name__ == "__main__":
    raise SystemExit(main())
