"""把本地 NAFNet 官方权重转换为可选 ONNX CPU 插件。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/export_nafnet_onnx.py \
      --weights models/weights/NAFNet-GoPro-width32.pth \
      --output models/onnx/nafnet-gopro-width32.onnx \
      --manifest models/manifests/nafnet-gopro-width32-onnx.json

说明：脚本不下载权重；需要用户已安装可选 PyTorch、ONNX 和 ONNX Runtime。输出
清单默认启用 CPU 分块推理，权重与 ONNX 文件均位于 Git 忽略的 ``models/`` 目录。
网络结构由本项目已记录许可证的 ``nafnet_torch_plugin.py`` 构造。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """构建转换参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="本地 NAFNet .pth 权重")
    parser.add_argument("--output", type=Path, required=True, help="输出 .onnx 路径")
    parser.add_argument("--manifest", type=Path, required=True, help="输出插件清单 JSON")
    parser.add_argument("--width", type=int, choices=(32, 64), default=32)
    parser.add_argument("--opset", type=int, choices=range(17, 22), default=17)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用 ONNX Runtime 对比 PyTorch 数值",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """加载权重、导出动态分辨率 ONNX、数值核对并写入清单。"""

    args = build_parser().parse_args(argv)
    if not args.weights.is_file():
        raise SystemExit(f"权重不存在：{args.weights}")
    if args.output.suffix.lower() != ".onnx":
        raise SystemExit("--output 必须使用 .onnx 扩展名")
    if args.manifest.suffix.lower() != ".json":
        raise SystemExit("--manifest 必须使用 .json 扩展名")
    if args.tile < 64 or not 0 <= args.overlap < args.tile:
        raise SystemExit("tile/overlap 参数无效")
    if not 0 <= args.padding < args.tile // 2:
        raise SystemExit("padding 参数无效")

    _progress(0.02, "加载 NAFNet 结构与权重")
    from nafnet_torch_plugin import _build_nafnet  # noqa: PLC0415

    torch, model = _build_nafnet(args.width)
    checkpoint = torch.load(str(args.weights), map_location="cpu", weights_only=True)
    parameters = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(parameters, strict=True)
    model.eval()
    dummy = torch.zeros((1, 3, 128, 160), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.stem}.tmp.onnx")
    _progress(0.18, "导出动态分辨率 ONNX")
    try:
        torch.onnx.export(
            model,
            dummy,
            str(temporary_output),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        os.replace(temporary_output, args.output)
    finally:
        temporary_output.unlink(missing_ok=True)

    verification: dict[str, float] | None = None
    if args.verify:
        _progress(0.72, "对比 ONNX Runtime 与 PyTorch")
        verification = _verify_export(torch, model, args.output)
        if verification["max_absolute_error"] > 2e-3:
            raise RuntimeError(
                "ONNX 数值误差过大："
                f"{verification['max_absolute_error']:.6g}，未写入可用清单"
            )
    _progress(0.88, "写入本地插件清单")
    manifest = build_manifest_payload(
        args.weights,
        args.output,
        args.manifest,
        width=args.width,
        tile=args.tile,
        overlap=args.overlap,
        padding=args.padding,
        opset=args.opset,
        verification=verification,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_name(f".{args.manifest.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, args.manifest)
    _progress(1.0, f"完成：{args.output} / {args.manifest}")
    return 0


def build_manifest_payload(
    weights_path: Path,
    onnx_path: Path,
    manifest_path: Path,
    *,
    width: int,
    tile: int,
    overlap: int,
    padding: int,
    opset: int,
    verification: dict[str, float] | None,
) -> dict[str, object]:
    """构造包含来源摘要和分块策略的严格 ONNX 清单。"""

    relative_model = os.path.relpath(onnx_path.resolve(), manifest_path.parent.resolve())
    payload: dict[str, object] = {
        "id": f"nafnet-gopro-width{width}-onnx-cpu",
        "name": f"NAFNet GoPro width{width} ONNX CPU（本地转换）",
        "type": "onnx",
        "model_path": Path(relative_model).as_posix(),
        "required_files": [Path(relative_model).as_posix()],
        "supports_tiling": True,
        "tile_size": tile,
        "tile_overlap": overlap,
        "tile_padding": padding,
        "license": "MIT（NAFNet）/ Apache-2.0（BasicSR 部分）；权重许可按上游发布说明",
        "homepage": "https://github.com/megvii-research/NAFNet",
        "timeout_seconds": 3600,
        "conversion": {
            "source_weight_sha256": _sha256(weights_path),
            "onnx_sha256": _sha256(onnx_path),
            "opset": opset,
            "width": width,
            "verification": verification,
        },
    }
    return payload


def _verify_export(torch, model, output_path: Path) -> dict[str, float]:  # type: ignore[no-untyped-def]
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("--verify 需要安装 onnxruntime") from exc
    generator = np.random.default_rng(20260801)
    sample = generator.random((1, 3, 96, 112), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(sample.copy())).cpu().numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    difference = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    return {
        "max_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(fraction: float, message: str) -> None:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    filled = round(fraction * 28)
    print(
        f"\r[{'#' * filled}{'-' * (28 - filled)}] {fraction * 100:5.1f}% {message}",
        end="\n" if fraction >= 1.0 else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
