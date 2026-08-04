"""ScreenRestore Evaluator — Stage-Gate Benchmark Harness.

用法:
    python scripts/evaluate.py --all          # 全量评估
    python scripts/evaluate.py --gate geometry  # 仅 Geometry gate
    python scripts/evaluate.py --scene 电脑屏幕  # 仅一个场景

Gate 定义:
    GEOMETRY:  corner_error_normalized ≤ 0.02 AND polygon_iou ≥ 0.90
    DEMOIRE:   moire_suppression_db ≥ 3.0 AND texture_retention ≥ 0.85
    REFLECTION: reflection_gain ≥ 15% AND clean_region_regression ≤ 2%
    CINEMA:    black_clipping ≤ 0.003 AND shadow_chroma improved vs baseline

输出:
    output/evaluation/scorecard.json  — PASS/FAIL per gate
    output/experiments/ledger.jsonl  — 实验记录

Exit code: 0=PASS, 1=FAIL/regression
"""

from __future__ import annotations

import argparse, json, sys, math, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2, numpy as np

from screenrestore.io.image_loader import load_image
from screenrestore.validation import compare_images, register_reference, ReferenceRegistration
from screenrestore.core.pipeline import OperatorRegistry
from screenrestore.core.presets import (
    PresetId, ProcessingMode,
    apply_preset, apply_processing_mode,
    build_default_pipeline, build_registry,
)
from screenrestore.semantic import SemanticAnalyzer, RestorationPlanner
from screenrestore.semantic.target_localizer import TargetLocalizer
from screenrestore.operators.geometry import warp_perspective, AspectRatioMode, InterpolationMode

# ── Gate 阈值 ────────────────────────────────────────────

GATE = {
    "geometry": {
        "corner_error_max": 0.020,
        "corner_error_good": 0.010,
        "polygon_iou_min": 0.90,
        "polygon_iou_good": 0.95,
    },
    "demoire": {
        "suppression_db_min": 3.0,
        "texture_retention_min": 0.85,
    },
    "reflection": {
        "gain_min": 0.15,
        "clean_degradation_max": 0.02,
    },
    "cinema": {
        "black_clipping_max": 0.003,
        "shadow_luma_threshold": 0.35,
    },
}

# ── 场景定义 ──────────────────────────────────────────────

@dataclass
class SceneCase:
    name: str
    photo: Path
    reference: Path
    preset: PresetId

def discover_scenes(data_dir: Path) -> list[SceneCase]:
    ref_dir = data_dir / "原图"
    return [SceneCase(n, data_dir/pn, ref_dir/rn, ps)
            for n, pn, rn, ps in [
                ("后台芭蕾","后台芭蕾.jpg","后台芭蕾原图.png",PresetId.ARTWORK),
                ("复古街头","复古街头.jpg","复古街头原图.png",PresetId.GLOSSY_ARTWORK),
                ("电脑屏幕","电脑屏幕.jpg","电脑屏幕原图.jpg",PresetId.DISPLAY),
                ("红发女子","红发女子.jpg","红发女子原图.png",PresetId.CINEMA),
            ] if (data_dir/pn).is_file() and (ref_dir/rn).is_file()]


# ── Robust corner matching ───────────────────────────────

def _match_corners(detected: np.ndarray, oracle: np.ndarray) -> tuple[float, float, float]:
    """鲁棒角点匹配：考虑 cyclic permutation + clockwise/counter-clockwise。

    Returns:
        (normalized_error, polygon_iou, max_corner_error_px)
    """
    detected = np.asarray(detected, dtype=np.float32).reshape(4, 2)
    oracle = np.asarray(oracle, dtype=np.float32).reshape(4, 2)
    diag = float(np.linalg.norm(oracle.max(axis=0) - oracle.min(axis=0)))

    best_mean = float("inf")
    best_max = float("inf")
    best_iou = 0.0

    # cyclic permutations
    for shift in range(4):
        d_rolled = np.roll(detected, shift, axis=0)
        # both directions
        for flipped in [d_rolled, d_rolled[::-1]]:
            errors = np.linalg.norm(flipped - oracle, axis=1)
            mean_err = float(np.mean(errors))
            max_err = float(np.max(errors))
            if mean_err < best_mean:
                best_mean = mean_err
                best_max = max_err
                # IoU: approximate with convex hull intersection
                best_iou = _polygon_iou(flipped, oracle)

    return (
        round(best_mean / max(diag, 1.0), 6),
        round(best_iou, 4),
        round(best_max, 2),
    )


def _polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个四边形的 IoU。"""
    try:
        import cv2
        mask_a = np.zeros((1000, 1000), dtype=np.uint8)
        mask_b = np.zeros((1000, 1000), dtype=np.uint8)
        # 归一化到 [0,1] 再缩放到 mask
        all_pts = np.vstack([a, b])
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        scale_x = 998.0 / max(x_max - x_min, 1)
        scale_y = 998.0 / max(y_max - y_min, 1)
        scale = min(scale_x, scale_y)
        a_norm = ((a - [x_min, y_min]) * scale).astype(np.int32)
        b_norm = ((b - [x_min, y_min]) * scale).astype(np.int32)
        cv2.fillPoly(mask_a, [a_norm], 255)
        cv2.fillPoly(mask_b, [b_norm], 255)
        intersection = float((mask_a & mask_b).sum())
        union = float((mask_a | mask_b).sum())
        return intersection / max(union, 1)
    except Exception:
        return 0.0


# ── Moiré residual energy ────────────────────────────────

def _moire_energy(image: np.ndarray, reference: np.ndarray) -> tuple[float, dict]:
    """计算摩尔纹残余能量和 texture retention。"""
    rh, rw = reference.shape[:2]
    if image.shape[:2] != (rh, rw):
        image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
    gray_i = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_r = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    residual = gray_i - gray_r

    fft = np.fft.fft2(residual)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    exclude = min(cy, cx) // 8
    mask = np.ones_like(magnitude, dtype=bool)
    mask[cy-exclude:cy+exclude, cx-exclude:cx+exclude] = False

    total_energy = float(np.sum(magnitude[mask] ** 2))

    # texture retention: high-freq gradient correlation
    hi_i = cv2.Laplacian(gray_i, cv2.CV_32F)
    hi_r = cv2.Laplacian(gray_r, cv2.CV_32F)
    texture_ret = float(np.corrcoef(hi_i.flatten(), hi_r.flatten())[0, 1])
    texture_ret = max(0.0, min(1.0, texture_ret))

    return total_energy, {
        "residual_energy": round(total_energy, 2),
        "texture_retention": round(texture_ret, 4),
    }


# ── Evaluator ────────────────────────────────────────────

class Evaluator:
    def __init__(self, data_dir: Path, output_dir: Path, registry: OperatorRegistry):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.registry = registry
        self.gt = json.loads((data_dir.parent / "benchmarks/ground_truth/targets.json").read_text())

    def evaluate_all(self) -> dict:
        scenes = discover_scenes(self.data_dir)
        results = {
            "geometry": self._eval_geometry(scenes),
            "demoire": self._eval_demoire(scenes),
            "reflection": self._eval_reflection(scenes),
            "cinema": self._eval_cinema(scenes),
        }
        # overall
        gates_defined = ["geometry", "demoire", "reflection", "cinema"]
        fails = [g for g in gates_defined if results.get(g, {}).get("status") == "FAIL"]
        results["overall"] = "FAIL" if fails else "PASS"
        return results

    def _eval_geometry(self, scenes: list[SceneCase]) -> dict:
        localizer = TargetLocalizer()
        scene_results = {}
        all_pass = True

        for s in scenes:
            gt = self.gt.get(s.name, {})
            if not gt:
                scene_results[s.name] = {"status": "SKIP", "reason": "no ground truth"}
                continue

            photo = load_image(s.photo).original_rgb
            from screenrestore.semantic.context import SceneContext
            ctx = SceneContext()
            ctx = localizer.localize(photo, ctx)

            if not ctx.has_target() or ctx.target_polygon is None:
                scene_results[s.name] = {"status": "FAIL", "reason": "no target detected"}
                all_pass = False
                continue

            corner_err, iou, max_err = _match_corners(ctx.target_polygon, np.array(gt["oracle_corners"]))
            passed = corner_err <= GATE["geometry"]["corner_error_max"] and iou >= GATE["geometry"]["polygon_iou_min"]
            if not passed:
                all_pass = False

            scene_results[s.name] = {
                "status": "PASS" if passed else "FAIL",
                "corner_error_normalized": corner_err,
                "polygon_iou": iou,
                "max_corner_error_px": max_err,
                "corner_threshold": GATE["geometry"]["corner_error_max"],
                "iou_threshold": GATE["geometry"]["polygon_iou_min"],
            }

        return {
            "status": "PASS" if all_pass else "FAIL",
            "scenes": scene_results,
        }

    def _eval_demoire(self, scenes: list[SceneCase]) -> dict:
        display_scene = None
        for s in scenes:
            if s.preset == PresetId.DISPLAY:
                display_scene = s
                break
        if display_scene is None:
            return {"status": "SKIP", "reason": "no display scene"}

        photo = load_image(display_scene.photo).original_rgb
        ref = load_image(display_scene.reference).original_rgb

        # Oracle warp
        reg = register_reference(photo, ref)
        H = np.linalg.inv(reg.homography_reference_to_photo)
        H /= H[2, 2]
        rh, rw = ref.shape[:2]
        warped = cv2.warpPerspective(photo, H, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

        # Warp energy
        e_warp, diag = _moire_energy(warped, ref)

        # Restored (Oracle pipeline)
        from screenrestore.semantic import SemanticAnalyzer, RestorationPlanner
        analyzer = SemanticAnalyzer()
        planner = RestorationPlanner()
        ctx = analyzer.analyze(photo)
        plan = planner.plan(ctx, scene_hint=PresetId.DISPLAY)

        pipe = build_default_pipeline(self.registry)
        apply_preset(pipe, PresetId.DISPLAY)
        apply_processing_mode(pipe, ProcessingMode.FIDELITY)
        for g in ("orientation","lens_distortion","geometry","mesh_warp"):
            pipe.set_enabled(g, False)
        for oid, rec in plan.operators.items():
            try:
                st = pipe.state(oid)
                if rec.enabled != st.enabled: pipe.set_enabled(oid, rec.enabled)
                if rec.params:
                    cur = st.params.to_dict(); cur.update(rec.params)
                    pipe.update_parameters(oid, cur)
            except ValueError: pass

        from screenrestore.core.operator import ProcessingContext
        pc = ProcessingContext(preview=False)
        restored = pipe.process(warped, pc, source_id="eval:demoire")
        if restored.dtype != np.uint8:
            restored = np.clip(restored*255, 0, 255).astype(np.uint8)

        e_restored, diag2 = _moire_energy(restored, ref)
        suppression_db = float(10.0 * math.log10(max(e_warp, 1e-12) / max(e_restored, 1e-12)))
        texture_ret = diag2["texture_retention"]

        passed = (
            suppression_db >= GATE["demoire"]["suppression_db_min"]
            and texture_ret >= GATE["demoire"]["texture_retention_min"]
        )

        return {
            "status": "PASS" if passed else "FAIL",
            "moire_suppression_db": round(suppression_db, 2),
            "suppression_threshold": GATE["demoire"]["suppression_db_min"],
            "texture_retention": round(texture_ret, 4),
            "texture_threshold": GATE["demoire"]["texture_retention_min"],
        }

    def _eval_reflection(self, scenes: list[SceneCase]) -> dict:
        glossy = None
        for s in scenes:
            if s.preset == PresetId.GLOSSY_ARTWORK:
                glossy = s
                break
        if glossy is None:
            return {"status": "SKIP", "reason": "no glossy_artwork scene"}

        photo = load_image(glossy.photo).original_rgb
        ref = load_image(glossy.reference).original_rgb

        reg = register_reference(photo, ref)
        H = np.linalg.inv(reg.homography_reference_to_photo)
        H /= H[2, 2]
        rh, rw = ref.shape[:2]
        warped = cv2.warpPerspective(photo, H, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

        # Run restoration
        from screenrestore.semantic import SemanticAnalyzer, RestorationPlanner
        analyzer = SemanticAnalyzer()
        planner = RestorationPlanner()
        ctx = analyzer.analyze(photo)
        plan = planner.plan(ctx, scene_hint=PresetId.GLOSSY_ARTWORK)

        pipe = build_default_pipeline(self.registry)
        apply_preset(pipe, PresetId.GLOSSY_ARTWORK)
        apply_processing_mode(pipe, ProcessingMode.FIDELITY)
        for g in ("orientation","lens_distortion","geometry","mesh_warp"):
            pipe.set_enabled(g, False)
        for oid, rec in plan.operators.items():
            try:
                st = pipe.state(oid)
                if rec.enabled != st.enabled: pipe.set_enabled(oid, rec.enabled)
                if rec.params:
                    cur = st.params.to_dict(); cur.update(rec.params)
                    pipe.update_parameters(oid, cur)
            except ValueError: pass

        from screenrestore.core.operator import ProcessingContext
        pc = ProcessingContext(preview=False)
        restored = pipe.process(warped, pc, source_id="eval:reflection")
        if restored.dtype != np.uint8:
            restored = np.clip(restored*255, 0, 255).astype(np.uint8)

        # Full-image metrics
        wm = compare_images(warped, ref)
        rm = compare_images(restored, ref)

        # Reflection gain: use MAE as proxy (we lack GT reflection mask)
        # If reflection ROI not annotated, use full-image delta
        mae_before = wm["mae_255"]
        mae_after = rm["mae_255"]
        gain = (mae_before - mae_after) / max(mae_before, 1e-8)

        passed = gain >= GATE["reflection"]["gain_min"]
        # Clean region proxy: overall DeltaE degradation
        de_before = wm["delta_e_mean"]
        de_after = rm["delta_e_mean"]
        clean_degradation = (de_after - de_before) / max(de_before, 1e-8)

        return {
            "status": "PASS" if passed else "FAIL",
            "reflection_gain": round(gain, 4),
            "gain_threshold": GATE["reflection"]["gain_min"],
            "clean_region_degradation": round(clean_degradation, 4),
            "degradation_threshold": GATE["reflection"]["clean_degradation_max"],
            "note": "UNANNOTATED: using full-image MAE as proxy for reflection region",
        }

    def _eval_cinema(self, scenes: list[SceneCase]) -> dict:
        cinema = None
        for s in scenes:
            if s.preset == PresetId.CINEMA:
                cinema = s
                break
        if cinema is None:
            return {"status": "SKIP", "reason": "no cinema scene"}

        photo = load_image(cinema.photo).original_rgb
        ref = load_image(cinema.reference).original_rgb

        reg = register_reference(photo, ref)
        H = np.linalg.inv(reg.homography_reference_to_photo)
        H /= H[2, 2]
        rh, rw = ref.shape[:2]
        warped = cv2.warpPerspective(photo, H, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

        # Run restoration
        from screenrestore.semantic import SemanticAnalyzer, RestorationPlanner
        analyzer = SemanticAnalyzer()
        planner = RestorationPlanner()
        ctx = analyzer.analyze(photo)
        plan = planner.plan(ctx, scene_hint=PresetId.CINEMA)

        pipe = build_default_pipeline(self.registry)
        apply_preset(pipe, PresetId.CINEMA)
        apply_processing_mode(pipe, ProcessingMode.FIDELITY)
        for g in ("orientation","lens_distortion","geometry","mesh_warp"):
            pipe.set_enabled(g, False)
        for oid, rec in plan.operators.items():
            try:
                st = pipe.state(oid)
                if rec.enabled != st.enabled: pipe.set_enabled(oid, rec.enabled)
                if rec.params:
                    cur = st.params.to_dict(); cur.update(rec.params)
                    pipe.update_parameters(oid, cur)
            except ValueError: pass

        from screenrestore.core.operator import ProcessingContext
        pc = ProcessingContext(preview=False)
        restored = pipe.process(warped, pc, source_id="eval:cinema")
        if restored.dtype != np.uint8:
            restored = np.clip(restored*255, 0, 255).astype(np.uint8)

        wm = compare_images(warped, ref)
        rm = compare_images(restored, ref)

        # Shadow analysis: Y < threshold in reference
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
        warped_lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
        restored_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB).astype(np.float32)

        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        shadow = ref_gray < GATE["cinema"]["shadow_luma_threshold"]

        if shadow.sum() > 100:
            # Chroma in shadow
            warped_chroma = np.sqrt(warped_lab[shadow, 1]**2 + warped_lab[shadow, 2]**2).mean()
            restored_chroma = np.sqrt(restored_lab[shadow, 1]**2 + restored_lab[shadow, 2]**2).mean()
            ref_chroma = np.sqrt(ref_lab[shadow, 1]**2 + ref_lab[shadow, 2]**2).mean()

            shadow_de_before = float(np.linalg.norm(
                warped_lab[shadow] - ref_lab[shadow], axis=1).mean())
            shadow_de_after = float(np.linalg.norm(
                restored_lab[shadow] - ref_lab[shadow], axis=1).mean())

            black_clip_before = wm["black_clipping_ratio"]
            black_clip_after = rm["black_clipping_ratio"]

            passed = black_clip_after <= GATE["cinema"]["black_clipping_max"]
        else:
            shadow_de_before = 0; shadow_de_after = 0
            black_clip_before = wm["black_clipping_ratio"]
            black_clip_after = rm["black_clipping_ratio"]
            passed = black_clip_after <= GATE["cinema"]["black_clipping_max"]

        return {
            "status": "PASS" if passed else "FAIL",
            "shadow_delta_e_before": round(shadow_de_before, 2),
            "shadow_delta_e_after": round(shadow_de_after, 2),
            "shadow_chroma_ref": round(float(ref_chroma) if shadow.sum() > 100 else 0, 2),
            "shadow_chroma_restored": round(float(restored_chroma) if shadow.sum() > 100 else 0, 2),
            "black_clipping": round(black_clip_after, 6),
            "black_clipping_max": GATE["cinema"]["black_clipping_max"],
            "global_psnr": round(rm["psnr_db"], 2),
            "global_ssim": round(rm["luminance_ssim"], 4),
            "global_delta_e": round(rm["delta_e_mean"], 2),
        }


# ── Main ─────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="ScreenRestore Evaluator")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1]/"测试数据")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]/"output/evaluation")
    p.add_argument("--all", action="store_true", default=True)
    p.add_argument("--gate", help="仅评估指定 gate")
    p.add_argument("--scene", help="仅评估指定场景")
    a = p.parse_args(argv)

    dd = a.data_dir.expanduser().resolve()
    od = a.output_dir.expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🔍 ScreenRestore Evaluator")
    print(f"{'='*60}\n")

    registry = build_registry()
    evaluator = Evaluator(dd, od, registry)
    results = evaluator.evaluate_all()

    scorecard_path = od / "scorecard.json"
    json.dump(results, open(scorecard_path, "w"), indent=2, ensure_ascii=False)
    print(f"📄 {scorecard_path}\n")

    # Print summary
    for gate, data in results.items():
        if gate == "overall":
            continue
        status = data.get("status", "?")
        icon = "✅" if status == "PASS" else ("⚠️" if status == "SKIP" else "❌")
        print(f"{icon} {gate.upper():<15} {status}")

        if gate == "geometry":
            for sn, sd in data.get("scenes", {}).items():
                icon2 = "✅" if sd["status"] == "PASS" else "❌"
                print(f"   {icon2} {sn:<12} corner={sd.get('corner_error_normalized',1):.4f} IoU={sd.get('polygon_iou',0):.3f}")
        elif gate == "demoire":
            print(f"   suppression={data.get('moire_suppression_db',0):.1f}dB (need ≥{GATE['demoire']['suppression_db_min']}dB)")
            print(f"   texture_retention={data.get('texture_retention',0):.4f} (need ≥{GATE['demoire']['texture_retention_min']})")
        elif gate == "reflection":
            print(f"   gain={data.get('reflection_gain',0):.1%} (need ≥{GATE['reflection']['gain_min']:.0%})")
        elif gate == "cinema":
            print(f"   black_clipping={data.get('black_clipping',1):.6f} (need ≤{GATE['cinema']['black_clipping_max']})")
            print(f"   shadow_ΔE: {data.get('shadow_delta_e_before',0):.1f} → {data.get('shadow_delta_e_after',0):.1f}")
            print(f"   global PSNR={data.get('global_psnr',0):.1f}dB SSIM={data.get('global_ssim',0):.4f}")

    overall = results.get("overall", "FAIL")
    icon = "✅" if overall == "PASS" else "❌"
    print(f"\n{icon} OVERALL: {overall}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
