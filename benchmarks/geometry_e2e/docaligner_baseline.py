"""独立运行 Apache-2.0 DocAligner 热图模型的 photo-only 几何基线。

使用范例：
    source benchmarks/geometry_e2e/.venv-docaligner/bin/activate
    which python
    python -m pip install -r benchmarks/geometry_e2e/requirements-docaligner.txt
    python -m benchmarks.geometry_e2e.docaligner_baseline --model-config lcnet100

DocAligner 只生成粗四角且没有本项目所需的内容/外框层级与可靠置信度头，因此本入口
报告 ``forced_prediction``，不把结果视为无人值守接受决策。实拍图定位完成后才读取
``oracle_corners`` 打分；脚本始终显示逐样本进度条。
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from screenrestore.geometry import EdgeRefinement, refine_quad_edges
from screenrestore.io.image_loader import load_image
from screenrestore.validation import corner_metrics

from .run import CASES, Case


def _prepare_turbojpeg_fallback() -> None:
    """在缺少系统 libturbojpeg 时，为上游图像工具提供 OpenCV 兼容实现。

    DocAligner 本身接收已经解码的 ``numpy.ndarray``，但它的通用工具依赖会在导入时
    无条件初始化 TurboJPEG。本基准不因此要求用户修改系统环境；回退类只覆盖该依赖
    导入时需要的 encode/decode 接口，不参与模型推理。
    """

    try:
        from turbojpeg import TurboJPEG

        TurboJPEG()
        return
    except (ImportError, RuntimeError):
        pass

    class OpenCvTurboJpeg:
        def encode(self, image: np.ndarray, quality: int = 90, **_: object) -> bytes:
            ok, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            )
            if not ok:
                raise RuntimeError("OpenCV JPEG 编码失败")
            return encoded.tobytes()

        def decode(self, data: bytes, **_: object) -> np.ndarray:
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("OpenCV JPEG 解码失败")
            return decoded

    module = types.ModuleType("turbojpeg")
    module.TurboJPEG = OpenCvTurboJpeg  # type: ignore[attr-defined]
    sys.modules["turbojpeg"] = module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-directory", type=Path, default=root / "测试数据")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=root / "benchmarks" / "ground_truth" / "targets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "evaluation" / "docaligner_baseline.json",
    )
    parser.add_argument(
        "--model-config",
        choices=("lcnet100", "fastvit_t8", "fastvit_sa24"),
        default="lcnet100",
    )
    args = parser.parse_args(argv)
    _prepare_turbojpeg_fallback()
    try:
        from docaligner import DocAligner, ModelType
    except ImportError as exc:
        raise RuntimeError(
            "请先安装 benchmarks/geometry_e2e/requirements-docaligner.txt"
        ) from exc
    _progress(0, len(CASES), "加载 DocAligner 热图模型（首次运行会下载官方权重）")
    model = DocAligner(model_type=ModelType.heatmap, model_cfg=args.model_config)
    data_directory = args.data_directory.expanduser().resolve()
    predictions: list[tuple[Case, np.ndarray, EdgeRefinement | None]] = []
    for index, case in enumerate(CASES):
        _progress(index, len(CASES), f"DocAligner {case.name}")
        photo = load_image(data_directory / case.photo).original_rgb
        coarse = np.asarray(model(cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)), dtype=np.float32)
        refinement = refine_quad_edges(photo, coarse) if coarse.shape == (4, 2) else None
        predictions.append((case, coarse, refinement))

    # 与产品 e2e 协议一致：全部模型输出冻结后才加载人工四角用于打分。
    gt_values = json.loads(args.ground_truth.expanduser().resolve().read_text(encoding="utf-8"))
    reports: list[dict[str, object]] = []
    for case, coarse, refinement in predictions:
        if case.name not in gt_values:
            raise ValueError(f"ground truth 缺少场景：{case.name}")
        expected = np.asarray(gt_values[case.name]["oracle_corners"], dtype=np.float32)
        if coarse.shape == (4, 2) and refinement is not None:
            coarse_nce, coarse_iou, coarse_max = corner_metrics(coarse, expected)
            refined_nce, refined_iou, refined_max = corner_metrics(refinement.corners, expected)
            metrics: dict[str, object] = {
                "detected": True,
                "coarse_corner_nce": round(coarse_nce, 8),
                "coarse_quad_iou": round(coarse_iou, 8),
                "coarse_max_corner_error_px": round(coarse_max, 4),
                "refinement_accepted": refinement.accepted,
                "refined_corner_nce": round(refined_nce, 8),
                "refined_quad_iou": round(refined_iou, 8),
                "refined_max_corner_error_px": round(refined_max, 4),
            }
        else:
            metrics = {"detected": False, "returned_shape": list(coarse.shape)}
        reports.append({"case": case.name, "photo": case.photo, "metrics": metrics})
    _progress(len(CASES), len(CASES), "写入 DocAligner baseline")
    detected = [report["metrics"] for report in reports if report["metrics"]["detected"]]  # type: ignore[index]
    report = {
        "protocol": "e2e_auto_baseline",
        "protocol_version": 1,
        "backend": f"docaligner_heatmap_{args.model_config}",
        "prediction_policy": "forced_prediction_without_product_confidence",
        "oracle_loaded_after_all_predictions": True,
        "upstream_license": "Apache-2.0",
        "upstream_commit_reviewed": "3275b0f07f8e99d8c01cb0774dea2549be1416b6",
        "sample_count": len(reports),
        "detected_count": len(detected),
        "mean_coarse_nce": (
            round(float(np.mean([item["coarse_corner_nce"] for item in detected])), 8)
            if detected
            else None
        ),
        "mean_coarse_iou": (
            round(float(np.mean([item["coarse_quad_iou"] for item in detected])), 8)
            if detected
            else None
        ),
        "mean_refined_nce": (
            round(float(np.mean([item["refined_corner_nce"] for item in detected])), 8)
            if detected
            else None
        ),
        "mean_refined_iou": (
            round(float(np.mean([item["refined_quad_iou"] for item in detected])), 8)
            if detected
            else None
        ),
        "cases": reports,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    fraction = min(1.0, done / max(1, total))
    filled = round(width * fraction)
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total:<3} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
