"""迭代验证脚本：处理四种场景测试图像，保存输出版本并生成数值对比报告。

v2 改进：
  - 直接使用 SIFT 单应矩阵将照片 warp 到精确参考图尺寸，
    绕过 GeometryOperator 以避免 output_size ≠ reference_size 的二次插值。
  - warp 后关闭所有几何相关算子（orientation/lens/geometry/mesh），
    只运行色彩/去噪/反光等恢复算子。
  - 新增 ECC 平移精修，矫正 SIFT 的亚像素残余偏差。

用法:
    source .venv/bin/activate
    python scripts/iterate_validation.py                # 默认 v2
    python scripts/iterate_validation.py --version v3   # 指定版本
    python scripts/iterate_validation.py --mode v1      # 旧版对比

输出:
    output/<version>/
        <场景>_恢复.png          处理结果
        <场景>_直接warp.png      SIFT 单应直接 warp（不经流水线几何）
        <场景>_差异热图.png      可视化差异
        <场景>_内容定位.png      检测到的内容区域
        metrics.json             数值指标汇总
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.core.pipeline import ImagePipeline, OperatorRegistry
from screenrestore.core.presets import (
    PresetId,
    ProcessingMode,
    apply_preset,
    apply_processing_mode,
    build_default_pipeline,
    build_registry,
)
from screenrestore.io.image_loader import load_image
from screenrestore.validation import (
    ReferenceRegistration,
    compare_images,
    difference_heatmap,
    register_reference,
)

# ── 场景定义 ──────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SceneCase:
    name: str
    photo: Path
    reference: Path
    preset: PresetId


def discover_scenes(data_dir: Path) -> list[SceneCase]:
    """发现四种测试场景及其原图。"""
    ref_dir = data_dir / "原图"
    mapping = [
        ("后台芭蕾", "后台芭蕾.jpg", "后台芭蕾原图.png", PresetId.ARTWORK),
        ("复古街头", "复古街头.jpg", "复古街头原图.png", PresetId.GLOSSY_ARTWORK),
        ("电脑屏幕", "电脑屏幕.jpg", "电脑屏幕原图.jpg", PresetId.DISPLAY),
        ("红发女子", "红发女子.jpg", "红发女子原图.png", PresetId.CINEMA),
    ]
    cases = []
    for name, photo_fn, ref_fn, preset in mapping:
        photo_path = data_dir / photo_fn
        ref_path = ref_dir / ref_fn
        if not photo_path.is_file():
            print(f"⚠ 跳过 {name}：找不到 {photo_path}", file=sys.stderr)
            continue
        if not ref_path.is_file():
            print(f"⚠ 跳过 {name}：找不到 {ref_path}", file=sys.stderr)
            continue
        cases.append(SceneCase(name, photo_path, ref_path, preset))
    return cases


# ── ECC 亚像素精修 ────────────────────────────────────────

def ecc_refine(
    candidate_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    max_iterations: int = 100,
    epsilon: float = 1e-8,
    max_shift_px: float = 20.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用 ECC 在亚像素级别精修对齐。

    策略：先尝试 MOTION_HOMOGRAPHY，若位移过大（>max_shift_px 说明收敛到
    错误局部极值）则回退到 MOTION_TRANSLATION。两张图视觉差异大（如照片 vs
    数字原图）时 homography 自由度太高易跑偏，translation 更稳健。
    """
    rh, rw = reference_rgb.shape[:2]

    if candidate_rgb.shape[:2] != (rh, rw):
        candidate_resized = cv2.resize(
            candidate_rgb, (rw, rh), interpolation=cv2.INTER_LANCZOS4,
        )
    else:
        candidate_resized = candidate_rgb

    candidate_gray = cv2.cvtColor(
        (candidate_resized.astype(np.float32) / 255.0).clip(0, 1), cv2.COLOR_RGB2GRAY,
    )
    reference_gray = cv2.cvtColor(
        (reference_rgb.astype(np.float32) / 255.0).clip(0, 1), cv2.COLOR_RGB2GRAY,
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)

    # 第一优先：homography
    try:
        H_init = np.eye(3, 3, dtype=np.float32)
        score, H = cv2.findTransformECC(
            reference_gray, candidate_gray, H_init,
            cv2.MOTION_HOMOGRAPHY, criteria,
            inputMask=None, gaussFiltSize=3,
        )
        dx, dy = H[0, 2], H[1, 2]
        # 检查位移是否合理——过大说明 ECC 跑偏到错误局部极值
        if abs(dx) < max_shift_px and abs(dy) < max_shift_px:
            aligned = cv2.warpPerspective(
                candidate_resized, H, (rw, rh),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned, {
                "ecc_score": round(float(score), 6),
                "ecc_method": "homography",
                "dx_px": round(float(dx), 4),
                "dy_px": round(float(dy), 4),
            }
    except cv2.error:
        pass

    # 第二优先：translation（更稳健）
    try:
        T_init = np.eye(2, 3, dtype=np.float32)
        score, T = cv2.findTransformECC(
            reference_gray, candidate_gray, T_init,
            cv2.MOTION_TRANSLATION, criteria,
            inputMask=None, gaussFiltSize=3,
        )
        dx, dy = T[0, 2], T[1, 2]
        if abs(dx) < max_shift_px and abs(dy) < max_shift_px:
            aligned = cv2.warpAffine(
                candidate_resized, T, (rw, rh),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned, {
                "ecc_score": round(float(score), 6),
                "ecc_method": "translation",
                "dx_px": round(float(dx), 4),
                "dy_px": round(float(dy), 4),
            }
    except cv2.error:
        pass

    return candidate_resized, {
        "ecc_score": 0.0,
        "ecc_method": "fallback_identity",
        "dx_px": 0.0,
        "dy_px": 0.0,
    }


# ── 处理逻辑 ──────────────────────────────────────────────

def process_scene(
    scene: SceneCase,
    registry: OperatorRegistry,
) -> dict:
    """处理单个场景：SIFT 配准 → 直接单应 warp 到参考尺寸 → 关闭几何 → 恢复。"""
    print(f"  📷 加载 {scene.name} ...")
    photo_doc = load_image(scene.photo)
    ref_doc = load_image(scene.reference)
    photo = photo_doc.original_rgb
    reference = ref_doc.original_rgb
    rh, rw = reference.shape[:2]

    # 1) SIFT 配准：参考图 → 拍摄图
    print(f"  🎯 SIFT 配准 {scene.name} ...")
    registration = register_reference(photo, reference)
    H_ref_to_photo = registration.homography_reference_to_photo

    # 2) 直接用单应的逆 warp 到参考图精确尺寸（绕过 GeometryOperator）
    H_photo_to_ref = np.linalg.inv(H_ref_to_photo)
    H_photo_to_ref /= H_photo_to_ref[2, 2]
    direct_warp = cv2.warpPerspective(
        photo, H_photo_to_ref, (rw, rh),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )
    print(f"  📐 直接 warp: {photo.shape[1]}x{photo.shape[0]} → {rw}x{rh}")

    # 3) 构建恢复流水线（关闭所有几何算子）
    print(f"  ⚙ 构建流水线 preset={scene.preset.value} ...")
    pipeline = build_restoration_pipeline(registry, scene.preset)
    context = ProcessingContext(preview=False)
    restored = pipeline.process(
        direct_warp, context,
        source_id=f"iterate:{photo_doc.content_hash}",
    )
    # pipeline 输出是 float32 [0,1]，转回 uint8
    if restored.dtype != np.uint8:
        restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)

    # 4) ECC 精修
    print(f"  📊 ECC 精修 {scene.name} ...")
    aligned_warp, warp_alignment = ecc_refine(direct_warp, reference)
    aligned_restored, restored_alignment = ecc_refine(restored, reference)

    # 5) 评分
    warp_metrics = compare_images(aligned_warp, reference)
    restored_metrics = compare_images(aligned_restored, reference)

    # 6) 差异热图 + 定位可视化
    diff_warp = difference_heatmap(aligned_warp, reference)
    diff_restored = difference_heatmap(aligned_restored, reference)
    corners_viz = draw_reference_corners(photo, registration.corners_photo)

    return {
        "scene": scene.name,
        "preset": scene.preset.value,
        "photo_size": [photo.shape[1], photo.shape[0]],
        "reference_size": [rw, rh],
        "warp_output_size": [direct_warp.shape[1], direct_warp.shape[0]],
        "restored_output_size": [restored.shape[1], restored.shape[0]],
        "registration": registration.to_dict(),
        "warp_alignment": warp_alignment,
        "restored_alignment": restored_alignment,
        "direct_warp_metrics": warp_metrics,
        "restored_metrics": restored_metrics,
        "images": {
            "direct_warp": direct_warp,
            "restored": restored,
            "aligned_warp": aligned_warp,
            "aligned_restored": aligned_restored,
            "diff_warp": diff_warp,
            "diff_restored": diff_restored,
            "corners_viz": corners_viz,
        },
    }


def build_restoration_pipeline(
    registry: OperatorRegistry,
    preset: PresetId,
) -> ImagePipeline:
    """构建仅含恢复算子的流水线（跳过 orientation/lens/geometry/mesh）。"""
    pipeline = build_default_pipeline(registry)
    apply_preset(pipeline, preset)
    apply_processing_mode(pipeline, ProcessingMode.FIDELITY)

    # 关闭所有几何算子 — 已通过直接单应 warp 完成几何校正
    for geo_op in ("orientation", "lens_distortion", "geometry", "mesh_warp"):
        pipeline.set_enabled(geo_op, False)

    return pipeline


def draw_reference_corners(photo: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """在拍摄图上画出检测到的内容四角。"""
    img = photo.copy()
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, (255, 0, 0), 3)
    for i, pt in enumerate(pts):
        cv2.putText(img, str(i + 1), (pt[0][0] + 8, pt[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    return img


# ── 主入口 ────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="迭代验证四种场景的恢复质量")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "测试数据",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "output",
    )
    parser.add_argument(
        "--version", default="v2",
        help="输出版本号 (如 v2, v3)",
    )
    parser.add_argument("--only", help="只处理名称包含该文本的场景")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    output_version_dir = (args.output_dir / args.version).expanduser().resolve()
    output_version_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(data_dir)
    if args.only:
        scenes = [s for s in scenes if args.only in s.name]
    if not scenes:
        print("❌ 没有找到测试场景", file=sys.stderr)
        return 2

    print(f"\n{'='*60}")
    print(f"🚀 ScreenRestore 迭代验证 — {args.version}")
    print(f"   数据目录: {data_dir}")
    print(f"   输出目录: {output_version_dir}")
    print(f"   场景数量: {len(scenes)}")
    print(f"   策略: SIFT单应直接warp → 关闭几何算子 → ECC精修")
    print(f"{'='*60}\n")

    registry = build_registry()
    reports = []

    for i, scene in enumerate(scenes):
        print(f"[{i+1}/{len(scenes)}] {scene.name} (preset={scene.preset.value})")
        try:
            result = process_scene(scene, registry)
            # 保存图像
            _save(result["images"]["direct_warp"],
                  output_version_dir / f"{scene.name}_直接warp.png")
            _save(result["images"]["restored"],
                  output_version_dir / f"{scene.name}_恢复.png")
            _save(result["images"]["diff_warp"],
                  output_version_dir / f"{scene.name}_warp差异热图.png")
            _save(result["images"]["diff_restored"],
                  output_version_dir / f"{scene.name}_恢复差异热图.png")
            _save(result["images"]["corners_viz"],
                  output_version_dir / f"{scene.name}_内容定位.png")
            # 去图像，只保留指标
            metrics = {k: v for k, v in result.items() if k != "images"}
            metrics["output_dir"] = str(output_version_dir)
            reports.append(metrics)

            r = result["restored_metrics"]
            w = result["direct_warp_metrics"]
            reg = result["registration"]
            print(f"  📐 SIFT: {reg['feature_matches']}匹配/{reg['inliers']}内点 "
                  f"({reg['inlier_ratio']:.1%}) 误差{reg['median_reprojection_error_px']:.1f}px")
            print(f"  🔧 ECC: {result['restored_alignment'].get('ecc_method','?')} "
                  f"score={result['restored_alignment'].get('ecc_score',0):.4f} "
                  f"dx={result['restored_alignment'].get('dx_px',0):.2f}px "
                  f"dy={result['restored_alignment'].get('dy_px',0):.2f}px")
            print(f"  ✅ warp → PSNR={w['psnr_db']:.2f}dB SSIM={w['luminance_ssim']:.4f} ΔE={w['delta_e_mean']:.1f}")
            print(f"  ✅ 恢复 → PSNR={r['psnr_db']:.2f}dB SSIM={r['luminance_ssim']:.4f} ΔE={r['delta_e_mean']:.1f}")
        except Exception as exc:
            print(f"  ❌ 失败: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            reports.append({"scene": scene.name, "error": str(exc)})

    # 汇总报告
    metrics_path = output_version_dir / "metrics.json"
    summary = {
        "version": args.version,
        "scenes": len(scenes),
        "reports": reports,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n📄 报告已保存: {metrics_path}")

    # 打印汇总表
    _print_summary(reports)
    return 0


def _print_summary(reports: list[dict]) -> None:
    header = f"{'场景':<12} {'SIFT内点率':>9} {'warpPSNR':>8} {'恢复PSNR':>8} {'SSIM':>8} {'ΔE':>7} {'梯度相关':>8} {'ECC方法':>10}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for r in reports:
        if "error" in r:
            print(f"{r['scene']:<12} {'ERROR':>8}")
            continue
        reg = r["registration"]
        wm = r["direct_warp_metrics"]
        rm = r["restored_metrics"]
        al = r["restored_alignment"]
        print(f"{r['scene']:<12} {reg['inlier_ratio']:>8.1%} "
              f"{wm['psnr_db']:>8.2f} {rm['psnr_db']:>8.2f} "
              f"{rm['luminance_ssim']:>8.4f} {rm['delta_e_mean']:>6.1f} "
              f"{rm['gradient_correlation']:>8.4f} "
              f"{al.get('ecc_method','?'):>10}")
    print(f"{'='*len(header)}\n")


def _save(image: np.ndarray, path: Path) -> None:
    """保存 uint8 RGB 图像为 PNG。"""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    raise SystemExit(main())
