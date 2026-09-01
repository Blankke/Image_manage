"""P3 shared decoder、映射、退化图、忠实模型与路由的数值契约。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from training.p3.degradations import synthetic_reflection, synthetic_screen_recapture
from training.restoration.degradation import factorized_degradation_graph

from screenrestore.geometry import (
    InverseMap,
    RadialLensParameters,
    compose_inverse_maps,
    dense_grid_inverse_map,
    identity_inverse_map,
    radial_inverse_map,
    remap_original_once,
    safe_radial_inverse_map,
)
from screenrestore.geometry.decoder import CornerDecoderSpec, decode_corner_logits
from screenrestore.restoration import RestorationRoute, route_artifacts


def _corner_logits(size: int = 16) -> np.ndarray:
    logits = np.full((1, 4, size, size), -8.0, np.float32)
    for index, (x, y) in enumerate(((2, 3), (13, 2), (12, 13), (3, 12))):
        logits[0, index, y, x] = 8.0
        logits[0, index, y, x + (1 if x < size - 1 else -1)] = 5.0
        logits[0, index, size // 2, size // 2] = 2.0
    return logits


def test_shared_decoder_torch_numpy_parity() -> None:
    torch = pytest.importorskip("torch")
    from training.quadlocator.decoder import local_softargmax_corners

    logits = _corner_logits()
    expected = decode_corner_logits(logits, CornerDecoderSpec())
    assert expected.coordinates is not None
    actual = local_softargmax_corners(torch.from_numpy(logits))[0].numpy()
    assert np.allclose(actual, expected.coordinates / 15.0, atol=1e-6)


def test_local_peak_decoder_does_not_drift_to_remote_peak() -> None:
    logits = _corner_logits(17)
    logits[:, :, 8, 8] = 7.5
    decoded = decode_corner_logits(logits)
    assert decoded.coordinates is not None
    assert np.linalg.norm(decoded.coordinates[0] - np.array([2.0, 3.0])) < 1.0


def test_nms_second_peak_suppresses_primary_shoulder() -> None:
    logits = np.full((1, 4, 15, 15), -9.0, np.float32)
    logits[:, :, 4, 4] = 9.0
    logits[:, :, 4, 5] = 8.5
    logits[:, :, 11, 11] = 5.0
    decoded = decode_corner_logits(logits, CornerDecoderSpec(nms_radius=3))
    assert all(item.peak2 == pytest.approx(1.0 / (1.0 + np.exp(-5.0))) for item in decoded.diagnostics)
    assert all(item.peak_distance > 8.0 for item in decoded.diagnostics)


def test_nms_second_peak_suppresses_diagonal_shoulder() -> None:
    logits = np.full((1, 4, 17, 17), -9.0, np.float32)
    logits[:, :, 5, 5] = 9.0
    # 该点位于半径 3 的圆外、方形邻域内，仍属于同一个宽峰的肩部。
    logits[:, :, 8, 8] = 8.0
    logits[:, :, 14, 14] = 5.0
    decoded = decode_corner_logits(logits, CornerDecoderSpec(nms_radius=3))
    expected = 1.0 / (1.0 + np.exp(-5.0))
    assert all(item.peak2 == pytest.approx(expected) for item in decoded.diagnostics)
    assert all(item.peak_distance > 12.0 for item in decoded.diagnostics)


def test_candidate_margin_and_entropy_are_data_dependent() -> None:
    ambiguous = _corner_logits()
    clear = ambiguous.copy()
    clear[:, :, 8, 8] = -8.0
    first = decode_corner_logits(ambiguous).diagnostics[0]
    second = decode_corner_logits(clear).diagnostics[0]
    assert first.peak_difference != second.peak_difference
    assert 0.0 <= first.normalized_entropy <= 1.0
    assert first.normalized_entropy > second.normalized_entropy


def test_boundary_distance_target_and_balanced_loss() -> None:
    torch = pytest.importorskip("torch")
    from training.quadlocator.dataset import _boundary_target
    from training.quadlocator.losses import _balanced_boundary_loss

    quad = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], np.float32)
    target = _boundary_target(quad, 64, sigma=1.5)
    assert target.max() > 0.9
    assert 0 < np.count_nonzero(target > 0.1) < target.size // 2
    logits = torch.zeros((1, 1, 64, 64), requires_grad=True)
    loss = _balanced_boundary_loss(logits, torch.from_numpy(target[None, None]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad) > 0


def test_inverse_map_composition_and_single_remap() -> None:
    first_y, first_x = np.indices((8, 8), dtype=np.float32)
    first = InverseMap(first_x + 1.0, first_y + 2.0, "first")
    second_y, second_x = np.indices((4, 4), dtype=np.float32)
    second = InverseMap(second_x * 2.0, second_y * 2.0, "second")
    composed = compose_inverse_maps([first, second])
    assert np.allclose(composed.map_x, second_x * 2.0 + 1.0)
    assert np.allclose(composed.map_y, second_y * 2.0 + 2.0)
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    output = remap_original_once(image, composed, interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(output[0, 0], image[2, 1])


def test_radial_identity_and_monotonicity_rejection() -> None:
    identity = radial_inverse_map((31, 47, 3))
    expected = identity_inverse_map((31, 47, 3))
    assert np.array_equal(identity.map_x, expected.map_x)
    assert np.array_equal(identity.map_y, expected.map_y)
    with pytest.raises(ValueError, match="单调/双射"):
        RadialLensParameters(k1=-0.6, k2=-0.4)
    bypass, report = safe_radial_inverse_map(
        (16, 16, 3), RadialLensParameters(confidence=0.2), minimum_confidence=0.8
    )
    assert report["applied"] is False
    assert np.array_equal(bypass.map_x, identity_inverse_map((16, 16, 3)).map_x)


def test_dense_grid_identity_and_fold_rejection() -> None:
    identity = dense_grid_inverse_map(np.zeros((17, 17, 2), np.float32), (40, 60))
    expected = identity_inverse_map((40, 60, 3))
    assert np.allclose(identity.map_x, expected.map_x)
    folded = np.zeros((3, 3, 2), np.float32)
    folded[:, 0, 0] = 0.25
    folded[:, 1, 0] = -0.25
    folded[:, 2, 0] = 0.25
    with pytest.raises(ValueError, match="折叠"):
        dense_grid_inverse_map(folded, (32, 32))


def test_dense_grid_no_fold_loss_is_zero_for_identity() -> None:
    torch = pytest.importorskip("torch")
    from training.p3.losses import jacobian_fold_loss

    identity = torch.zeros((1, 17, 17, 2))
    folded = identity.clone()
    folded[:, :, 8, 0] = -0.8
    assert jacobian_fold_loss(identity).item() == pytest.approx(0.0)
    assert jacobian_fold_loss(folded).item() > 0.0


def test_factorized_fidelity_target_preserves_photometric_state() -> None:
    clean = np.full((48, 48, 3), 0.4, np.float32)
    first = factorized_degradation_graph(clean, task="fidelity", seed=123)
    second = factorized_degradation_graph(clean, task="fidelity", seed=123)
    assert np.array_equal(first.input_rgb, second.input_rgb)
    assert np.array_equal(first.target_rgb, first.artifact_rgb)
    assert first.trace.target_stage == "artifact"
    assert first.trace.to_dict()["seed"] == 123


def test_fidelity_and_photometric_start_as_exact_identity() -> None:
    torch = pytest.importorskip("torch")
    from training.p3.models import PhotometricNet
    from training.restoration.model import FidelityNetV2

    image = torch.rand((1, 3, 32, 32))
    fidelity = FidelityNetV2().eval()
    photometric = PhotometricNet().eval()
    with torch.no_grad():
        assert torch.equal(fidelity(image), image)
        assert torch.allclose(photometric.apply(image, photometric(image)), image, atol=2e-6)


def test_photometric_tone_curve_is_monotonic() -> None:
    torch = pytest.importorskip("torch")
    from training.p3.models import PhotometricNet

    ramp = torch.linspace(0.0, 1.0, 128).view(1, 1, 1, 128).repeat(1, 3, 8, 1)
    model = PhotometricNet().eval()
    raw = torch.randn((1, model.parameter_count)) * 0.2
    output = model.apply(ramp, raw)
    assert torch.all(output[..., 1:] - output[..., :-1] >= -1e-6)


def test_demoire_and_reflection_generators_are_seed_deterministic() -> None:
    clean = np.random.default_rng(9).random((48, 64, 3), dtype=np.float32)
    demoire_a = synthetic_screen_recapture(clean, np.random.default_rng(10))
    demoire_b = synthetic_screen_recapture(clean, np.random.default_rng(10))
    reflection_a = synthetic_reflection(clean, np.random.default_rng(11))
    reflection_b = synthetic_reflection(clean, np.random.default_rng(11))
    assert np.array_equal(demoire_a.input_rgb, demoire_b.input_rgb)
    assert demoire_a.trace == demoire_b.trace
    assert np.array_equal(reflection_a.input_rgb, reflection_b.input_rgb)
    assert np.array_equal(reflection_a.mask, reflection_b.mask)


def test_reflection_modification_is_mask_localized() -> None:
    clean = np.random.default_rng(12).random((64, 64, 3), dtype=np.float32)
    pair = synthetic_reflection(clean, np.random.default_rng(13))
    assert pair.mask is not None and pair.unresolved_mask is not None
    outside = pair.mask < 1e-3
    difference = np.abs(pair.input_rgb - pair.target_rgb).max(axis=2)
    assert float(difference[outside].max(initial=0.0)) < 1e-3
    assert np.all(pair.unresolved_mask <= (pair.mask >= 0.5))


def test_router_clean_bypass_and_mixed_review() -> None:
    clean = route_artifacts({}, {})
    assert clean.route == RestorationRoute.CLEAN_BYPASS
    mixed = route_artifacts(
        {"moire": 0.9, "reflection": 0.9},
        {"moire": 0.8, "reflection": 0.8},
    )
    assert mixed.route == RestorationRoute.REVIEW


def test_router_balanced_loss_prioritizes_sparse_positive() -> None:
    torch = pytest.importorskip("torch")
    from training.p3.train_specialist import _router_loss

    logits = torch.zeros((1, 7), requires_grad=True)
    labels = torch.zeros((1, 7))
    labels[0, 0] = 1.0
    severity = torch.full((1, 7), 0.5, requires_grad=True)
    target = torch.zeros((1, 7))
    target[0, 0] = 0.8
    loss = _router_loss(logits, severity, labels, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert abs(float(logits.grad[0, 0])) > abs(float(logits.grad[0, 1])) * 5.0


def test_restoration_schema_declares_p3_reference_types() -> None:
    path = Path(__file__).resolve().parents[2] / "datasets" / "schemas" / "restoration.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["properties"]["reference_type"]["enum"] == [
        "exact_pair",
        "registered_pair",
        "synthetic",
        "identity",
        "weak_unpaired",
    ]


def test_mps_model_smoke_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("当前环境没有 MPS")
    from training.p3.models import DemoireNet

    model = DemoireNet().to("mps")
    output = model(torch.rand((1, 3, 32, 32), device="mps"))
    assert torch.isfinite(output).all()


def test_full_archive_pipeline_identity_smoke() -> None:
    torch = pytest.importorskip("torch")
    from training.p3.models import PhotometricNet
    from training.restoration.model import FidelityNetV2

    original = np.random.default_rng(14).random((32, 48, 3), dtype=np.float32)
    geometry = remap_original_once(original, identity_inverse_map(original.shape))
    image = torch.from_numpy(geometry).permute(2, 0, 1)[None]
    fidelity = FidelityNetV2().eval()
    photometric = PhotometricNet().eval()
    with torch.no_grad():
        output = photometric.apply(fidelity(image), photometric(fidelity(image)))
    assert torch.allclose(output, image, atol=2e-6)
