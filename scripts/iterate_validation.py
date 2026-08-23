"""ScreenRestore 迭代验证 — Oracle + E2E 双 benchmark, v9"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.core.presets import (
    PresetId,
    ProcessingMode,
    apply_preset,
    apply_processing_mode,
    build_default_pipeline,
    build_registry,
)
from screenrestore.geometry import AspectRatioMode, InterpolationMode, warp_perspective
from screenrestore.io.image_loader import load_image
from screenrestore.semantic import RestorationPlanner, SemanticAnalyzer
from screenrestore.semantic.target_localizer import TargetLocalizer
from screenrestore.validation import compare_images, difference_heatmap, register_reference


@dataclass(frozen=True, slots=True)
class SceneCase:
    name: str
    photo: Path
    reference: Path
    preset: PresetId


def discover_scenes(data_dir):
    ref_dir = data_dir / "原图"
    return [
        SceneCase(n, data_dir / pn, ref_dir / rn, ps)
        for n, pn, rn, ps in [
            ("后台芭蕾", "后台芭蕾.jpg", "后台芭蕾原图.png", PresetId.ARTWORK),
            ("复古街头", "复古街头.jpg", "复古街头原图.png", PresetId.GLOSSY_ARTWORK),
            ("电脑屏幕", "电脑屏幕.jpg", "电脑屏幕原图.jpg", PresetId.DISPLAY),
            ("红发女子", "红发女子.jpg", "红发女子原图.png", PresetId.CINEMA),
        ]
        if (data_dir / pn).is_file() and (ref_dir / rn).is_file()
    ]


def ecc_refine(c, r, max_iter=100, eps=1e-8, max_shift=20.0):
    rh, rw = r.shape[:2]
    if c.shape[:2] != (rh, rw):
        c = cv2.resize(c, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
    cg = cv2.cvtColor((c.astype(np.float32) / 255).clip(0, 1), cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor((r.astype(np.float32) / 255).clip(0, 1), cv2.COLOR_RGB2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    for motion, name, init in [
        (cv2.MOTION_HOMOGRAPHY, "homography", np.eye(3, 3, dtype=np.float32)),
        (cv2.MOTION_TRANSLATION, "translation", np.eye(2, 3, dtype=np.float32)),
    ]:
        try:
            sc, M = cv2.findTransformECC(rg, cg, init, motion, crit, None, 3)
            dx, dy = M[0, 2], M[1, 2]
            if abs(dx) < max_shift and abs(dy) < max_shift:
                a = (
                    cv2.warpPerspective(
                        c, M, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                    )
                    if motion == cv2.MOTION_HOMOGRAPHY
                    else cv2.warpAffine(
                        c, M, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                    )
                )
                return a, {
                    "ecc_score": round(float(sc), 6),
                    "ecc_method": name,
                    "dx_px": round(float(dx), 4),
                    "dy_px": round(float(dy), 4),
                }
        except cv2.error:
            pass
    return c, {"ecc_score": 0, "ecc_method": "fallback", "dx_px": 0, "dy_px": 0}


def apply_plan(pipeline, plan, preset):
    apply_preset(pipeline, preset)
    apply_processing_mode(pipeline, ProcessingMode.FIDELITY)
    for g in ("orientation", "lens_distortion", "geometry", "mesh_warp"):
        pipeline.set_enabled(g, False)
    for oid, rec in plan.operators.items():
        try:
            st = pipeline.state(oid)
            if rec.enabled != st.enabled:
                pipeline.set_enabled(oid, rec.enabled)
            if rec.params:
                cur = st.params.to_dict()
                cur.update(rec.params)
                pipeline.update_parameters(oid, cur)
        except ValueError:
            pass
    return pipeline


def process_oracle(scene, registry, analyzer, planner):
    pd = load_image(scene.photo)
    rd = load_image(scene.reference)
    photo, ref = pd.original_rgb, rd.original_rgb
    rh, rw = ref.shape[:2]
    ctx = analyzer.analyze(photo)
    plan = planner.plan(ctx, scene_hint=scene.preset)
    ops = [k for k, v in plan.operators.items() if v.enabled]
    print(
        f"  🧠 {ctx.scene_type}({ctx.scene_confidence:.2f}) blur={ctx.properties.get('blur_estimate', 0):.2f} "
        f"noise={ctx.properties.get('noise_estimate', 0):.2f} lattice={ctx.properties.get('screen_lattice_confidence', 0):.2f}"
    )
    print(f"     ops={ops} notes={plan.notes}")
    reg = register_reference(photo, ref)
    H = np.linalg.inv(reg.homography_reference_to_photo)
    H /= H[2, 2]
    dw = cv2.warpPerspective(
        photo, H, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )
    pipe = build_default_pipeline(registry)
    apply_plan(pipe, plan, scene.preset)
    pc = ProcessingContext(preview=False)
    pc.metadata["scene_context"] = ctx
    restored = pipe.process(dw, pc, source_id=f"oracle:{pd.content_hash}")
    if restored.dtype != np.uint8:
        restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
    aw, _ = ecc_refine(dw, ref)
    ar, _ = ecc_refine(restored, ref)
    return {
        "scene": scene.name,
        "benchmark": "oracle",
        "preset": scene.preset.value,
        "semantic_scene": ctx.scene_type,
        "semantic_confidence": ctx.scene_confidence,
        "registration": reg.to_dict(),
        "scene_context": ctx.to_dict(),
        "restoration_plan": plan.to_dict(),
        "direct_warp_metrics": compare_images(aw, ref),
        "restored_metrics": compare_images(ar, ref),
        "images": {
            "direct_warp": dw,
            "restored": restored,
            "diff_restored": difference_heatmap(ar, ref),
            "corners_viz": _draw_corners(photo, reg.corners_photo),
        },
    }


def process_e2e(scene, registry, analyzer, planner):
    pd = load_image(scene.photo)
    photo = pd.original_rgb
    ctx = analyzer.analyze(photo)
    loc = TargetLocalizer()
    ctx = loc.localize(photo, ctx)
    plan = planner.plan(ctx, scene_hint=scene.preset)
    if ctx.has_target() and ctx.target_polygon is not None:
        geo, _ = warp_perspective(
            photo,
            ctx.target_polygon,
            ratio_mode=AspectRatioMode.FREE,
            interpolation=InterpolationMode.LANCZOS,
            auto_crop=True,
        )
    else:
        geo = photo
    pipe = build_default_pipeline(registry)
    apply_plan(pipe, plan, scene.preset)
    pc = ProcessingContext(preview=False)
    pc.metadata["scene_context"] = ctx
    restored = pipe.process(geo, pc, source_id=f"e2e:{pd.content_hash}")
    if restored.dtype != np.uint8:
        restored = np.clip(restored * 255, 0, 255).astype(np.uint8)

    # 定位、几何和恢复全部冻结后，才加载 reference 评分。
    rd = load_image(scene.reference)
    ref = rd.original_rgb
    rm = compare_images(restored, ref)
    reg = register_reference(photo, ref)
    if ctx.target_polygon is not None:
        diag = math.sqrt(photo.shape[0] ** 2 + photo.shape[1] ** 2)
        ce = float(
            np.mean(np.linalg.norm(ctx.target_polygon - reg.corners_photo, axis=1)) / max(diag, 1)
        )
    else:
        ce = 1.0
    return {
        "scene": scene.name,
        "benchmark": "e2e",
        "preset": scene.preset.value,
        "has_target": ctx.has_target(),
        "corner_error_normalized": round(ce, 6),
        "scene_context": ctx.to_dict(),
        "restoration_plan": plan.to_dict(),
        "restored_metrics": rm,
        "images": {"restored": restored},
    }


def _draw_corners(photo, corners):
    img = photo.copy()
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, (255, 0, 0), 3)
    for i, pt in enumerate(pts):
        cv2.putText(
            img,
            str(i + 1),
            (pt[0][0] + 8, pt[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 0, 0),
            3,
        )
    return img


def _save(img, path):
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "测试数据"
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    p.add_argument("--version", default="v9")
    p.add_argument("--only")
    a = p.parse_args(argv)
    dd = a.data_dir.expanduser().resolve()
    od = (a.output_dir / a.version).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)
    scenes = discover_scenes(dd)
    if a.only:
        scenes = [s for s in scenes if a.only in s.name]
    if not scenes:
        print("no scenes", file=sys.stderr)
        return 2
    print(f"\n{'=' * 60}\n🚀 ScreenRestore v9  {len(scenes)} scenes\n{'=' * 60}\n")
    reg = build_registry()
    ana = SemanticAnalyzer()
    pln = RestorationPlanner()
    reps = []
    for i, s in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {s.name} ({s.preset.value})")
        try:
            r = process_oracle(s, reg, analyzer=ana, planner=pln)
            for k, fn in [
                ("direct_warp", "oracle_warp"),
                ("restored", "oracle_恢复"),
                ("diff_restored", "oracle_差异热图"),
                ("corners_viz", "oracle_定位"),
            ]:
                _save(r["images"][k], od / f"{s.name}_{fn}.png")
            (od / f"{s.name}_oracle_语义分析.json").write_text(
                json.dumps(r["scene_context"], ensure_ascii=False, indent=2)
            )
            (od / f"{s.name}_oracle_恢复计划.json").write_text(
                json.dumps(r["restoration_plan"], ensure_ascii=False, indent=2)
            )
            reps.append({k: v for k, v in r.items() if k != "images"})
            wm, rm = r["direct_warp_metrics"], r["restored_metrics"]
            print(
                f"  ✅ Oracle {wm['psnr_db']:.1f}→{rm['psnr_db']:.1f}dB SSIM={rm['luminance_ssim']:.4f} ΔE={rm['delta_e_mean']:.1f} sharp={rm['sharpness_ratio']:.3f} spec={rm['spectral_peak_excess_db']:.3f}"
            )
            e = process_e2e(s, reg, analyzer=ana, planner=pln)
            _save(e["images"]["restored"], od / f"{s.name}_e2e_恢复.png")
            reps.append({k: v for k, v in e.items() if k != "images"})
            m = e["restored_metrics"]
            print(
                f"  ✅ E2E   target={'✓' if e['has_target'] else '✗'} corner_err={e['corner_error_normalized']:.4f} "
                f"PSNR={m['psnr_db']:.1f}dB SSIM={m['luminance_ssim']:.4f} ΔE={m['delta_e_mean']:.1f}"
            )
        except Exception as ex:
            print(f"  ❌ {ex}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            reps.append({"scene": s.name, "error": str(ex)})
    (od / "metrics.json").write_text(
        json.dumps(
            {"version": a.version, "scenes": len(scenes), "reports": reps},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n📄 {od}/metrics.json")
    hdr = f"{'scene':<12} {'bench':>7} {'PSNR':>7} {'SSIM':>7} {'ΔE':>6} {'sharp':>6} {'spec':>7} {'target':>6}"
    print(f"\n{'=' * len(hdr)}\n{hdr}\n{'-' * len(hdr)}")
    for r in reps:
        if "error" in r:
            continue
        b = r.get("benchmark", "?")
        m = r.get("restored_metrics", {})
        has = "✓" if r.get("has_target") else ("-" if b == "oracle" else "✗")
        print(
            f"{r['scene']:<12} {b:>7} {m.get('psnr_db', 0):>7.2f} {m.get('luminance_ssim', 0):>7.4f} "
            f"{m.get('delta_e_mean', 0):>5.1f} {m.get('sharpness_ratio', 0):>6.3f} "
            f"{m.get('spectral_peak_excess_db', 0):>7.3f} {has:>6}"
        )
    print(f"{'=' * len(hdr)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
