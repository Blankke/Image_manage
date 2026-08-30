"""可选 QuadLocator-S 训练栈 smoke test。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_model_heads_and_loss_are_trainable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from training.quadlocator.dataset import QuadDataset
    from training.quadlocator.generate_synthetic import main as generate_synthetic
    from training.quadlocator.losses import quadlocator_loss
    from training.quadlocator.model import QuadLocatorS

    data_directory = tmp_path / "quad-data"
    assert (
        generate_synthetic(
            [
                "--output-directory",
                str(data_directory),
                "--count",
                "10",
                "--size",
                "128",
                "--negative-ratio",
                "0.2",
            ]
        )
        == 0
    )
    dataset = QuadDataset(data_directory / "manifest.jsonl", split="train", image_size=128)
    sample = dataset[0]
    assert sample["ambiguous"].shape == (1,)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = QuadLocatorS(width_multiplier=0.5)

    outputs = model(batch["image"])
    loss, metrics = quadlocator_loss(outputs, batch)
    loss.backward()

    assert outputs["content_corner_heatmaps"].shape == (1, 4, 32, 32)
    assert outputs["outer_corner_heatmaps"].shape == (1, 4, 32, 32)
    assert outputs["content_mask_logits"].shape == (1, 1, 32, 32)
    assert outputs["boundary_logits"].shape == (1, 1, 32, 32)
    assert outputs["presence_logits"].shape == (1, 1)
    assert outputs["outer_presence_logits"].shape == (1, 1)
    assert outputs["class_logits"].shape == (1, 4)
    assert torch.isfinite(loss)
    assert metrics["total"] > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dataset_rejects_capture_session_split_leakage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from training.quadlocator.dataset import QuadDataset

    manifest = tmp_path / "manifest.jsonl"
    base = {
        "present": False,
        "target_class": "none",
        "content_quad": None,
        "outer_quad": None,
        "capture_session": "same-burst",
    }
    records = [
        {**base, "image": "a.jpg", "split": "train", "group_id": "a"},
        {**base, "image": "b.jpg", "split": "validation", "group_id": "b"},
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="capture_session"):
        QuadDataset(manifest, split="train", image_size=128)


def test_training_letterbox_preserves_non_square_quad_geometry() -> None:
    from training.quadlocator.dataset import _letterbox_image, _transform_quad

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas, transform = _letterbox_image(image, 128)
    full_quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    transformed = _transform_quad(full_quad, transform)

    assert canvas.shape == (128, 128, 3)
    assert transformed is not None
    assert np.allclose(
        transformed,
        np.array(
            [[0, 32 / 127], [1, 32 / 127], [1, 95 / 127], [0, 95 / 127]],
            dtype=np.float32,
        ),
    )


def test_dataset_can_limit_samples_for_reproducible_smoke_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from training.quadlocator.dataset import QuadDataset
    from training.quadlocator.generate_synthetic import main as generate_synthetic

    data_directory = tmp_path / "quad-data"
    assert generate_synthetic(["--output-directory", str(data_directory), "--count", "12", "--size", "128"]) == 0

    dataset = QuadDataset(
        data_directory / "manifest.jsonl",
        split="train",
        image_size=128,
        max_samples=3,
    )

    assert len(dataset) == 3


def test_synthetic_texture_sources_are_isolated_by_split(tmp_path) -> None:
    from training.quadlocator.generate_synthetic import _partition_paths_by_split

    paths = [tmp_path / "textures" / f"texture-{index:03d}.jpg" for index in range(100)]
    partitions = _partition_paths_by_split(paths)

    assert set(partitions) == {"train", "validation", "test"}
    assert sum(len(values) for values in partitions.values()) == len(paths)
    assert not (set(partitions["train"]) & set(partitions["validation"]))
    assert not (set(partitions["train"]) & set(partitions["test"]))
    assert not (set(partitions["validation"]) & set(partitions["test"]))

    moved = _partition_paths_by_split(
        [tmp_path / "moved-root" / "textures" / path.name for path in paths]
    )
    original_assignment = {
        path.name: split for split, values in partitions.items() for path in values
    }
    moved_assignment = {path.name: split for split, values in moved.items() for path in values}
    assert moved_assignment == original_assignment


def _loss_tensors(*, outer_present: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = {
        "content_corner_heatmaps": torch.zeros(1, 4, 8, 8, requires_grad=True),
        "outer_corner_heatmaps": torch.zeros(1, 4, 8, 8, requires_grad=True),
        "content_mask_logits": torch.zeros(1, 1, 8, 8, requires_grad=True),
        "boundary_logits": torch.zeros(1, 1, 8, 8, requires_grad=True),
        "presence_logits": torch.zeros(1, 1, requires_grad=True),
        "outer_presence_logits": torch.zeros(1, 1, requires_grad=True),
        "class_logits": torch.zeros(1, 4, requires_grad=True),
    }
    content_heatmaps = torch.zeros(1, 4, 8, 8)
    content_heatmaps[:, :, 3, 3] = 1.0
    outer_heatmaps = torch.zeros(1, 4, 8, 8)
    if outer_present:
        outer_heatmaps[:, :, 2, 2] = 1.0
    targets = {
        "content_corner_heatmaps": content_heatmaps,
        "outer_corner_heatmaps": outer_heatmaps,
        "content_mask": torch.zeros(1, 1, 8, 8),
        "boundary": torch.zeros(1, 1, 8, 8),
        "presence": torch.ones(1, 1),
        "outer_present": torch.tensor([[float(outer_present)]]),
        "target_class": torch.tensor([0]),
        "content_corners": torch.full((1, 4, 2), 3 / 7),
        "outer_corners": torch.full((1, 4, 2), 2 / 7) if outer_present else torch.zeros(1, 4, 2),
    }
    return outputs, targets


def test_outer_absent_has_presence_and_heatmap_negative_gradients() -> None:
    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=False)
    loss, _ = quadlocator_loss(outputs, targets)
    loss.backward()

    assert torch.count_nonzero(outputs["outer_presence_logits"].grad) > 0
    assert torch.count_nonzero(outputs["outer_corner_heatmaps"].grad) > 0


def test_outer_present_trains_presence_heatmap_and_coordinates() -> None:
    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    loss, metrics = quadlocator_loss(outputs, targets)
    loss.backward()

    assert torch.count_nonzero(outputs["outer_presence_logits"].grad) > 0
    assert torch.count_nonzero(outputs["outer_corner_heatmaps"].grad) > 0
    assert metrics["outer_corner_geometry"] > 0


def test_validation_selection_prefers_rejecting_ambiguous_target() -> None:
    from training.quadlocator.metrics import ValidationMetrics

    accepted_outputs, targets = _loss_tensors(outer_present=False)
    targets["presence"] = torch.zeros(1, 1)
    targets["target_class"] = torch.tensor([3])
    targets["ambiguous"] = torch.ones(1, 1)
    with torch.no_grad():
        accepted_outputs["presence_logits"].fill_(10.0)
        accepted_outputs["outer_presence_logits"].fill_(-10.0)
        accepted_outputs["content_corner_heatmaps"].fill_(-10.0)
        accepted_outputs["content_corner_heatmaps"][:, :, 3, 3] = 10.0
        accepted_outputs["class_logits"].fill_(-10.0)
        accepted_outputs["class_logits"][:, 0] = 10.0

    accepted_metrics = ValidationMetrics()
    accepted_metrics.update(accepted_outputs, targets)
    accepted_result = accepted_metrics.compute()

    rejected_outputs = {key: value.detach().clone() for key, value in accepted_outputs.items()}
    rejected_outputs["presence_logits"].fill_(-10.0)
    rejected_metrics = ValidationMetrics()
    rejected_metrics.update(rejected_outputs, targets)
    rejected_result = rejected_metrics.compute()

    assert accepted_result["ambiguous_target_count"] == 1
    assert accepted_result["ambiguous_rejection_rate"] == 0.0
    assert rejected_result["ambiguous_rejection_rate"] == 1.0
    assert rejected_result["selection_score"] > accepted_result["selection_score"]


def test_init_checkpoint_loads_p1_compatible_parameters_and_keeps_new_head(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _load_compatible_checkpoint

    old_model = QuadLocatorS(width_multiplier=0.5)
    old_state = {
        name: value
        for name, value in old_model.state_dict().items()
        if not name.startswith("outer_presence_head.")
    }
    checkpoint = tmp_path / "p1.pt"
    torch.save({"format_version": 1, "state_dict": old_state}, checkpoint)
    new_model = QuadLocatorS(width_multiplier=0.5)
    new_head_before = new_model.outer_presence_head.weight.detach().clone()

    report = _load_compatible_checkpoint(new_model, checkpoint)

    assert "outer_presence_head.weight" in report["missing_new_parameters"]
    assert torch.equal(new_model.outer_presence_head.weight, new_head_before)
    assert torch.equal(new_model.stem[0].weight, old_model.stem[0].weight)


def test_exported_onnx_uses_seven_output_contract(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ort = pytest.importorskip("onnxruntime")
    from training.quadlocator.export_onnx import OUTPUT_NAMES
    from training.quadlocator.export_onnx import main as export_onnx
    from training.quadlocator.model import QuadLocatorS

    model = QuadLocatorS(width_multiplier=0.5)
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "model.onnx"
    torch.save(
        {
            "format_version": 2,
            "width_multiplier": 0.5,
            "image_size": 128,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    assert export_onnx(["--checkpoint", str(checkpoint), "--output", str(output)]) == 0
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert [value.name for value in session.get_outputs()] == OUTPUT_NAMES


class _FakeOnnxSession:
    def __init__(self, _path: str, *, providers: list[str], outer_logit: float = -8.0, old: bool = False) -> None:
        _ = providers
        self.outer_logit = outer_logit
        names = [
            "content_corner_heatmaps",
            "outer_corner_heatmaps",
            "content_mask_logits",
            "boundary_logits",
            "presence_logits",
            "class_logits",
        ]
        if not old:
            names.insert(5, "outer_presence_logits")
        self._outputs = [SimpleNamespace(name=name) for name in names]

    def get_inputs(self):  # type: ignore[no-untyped-def]
        return [SimpleNamespace(name="image", shape=[1, 3, 128, 128])]

    def get_outputs(self):  # type: ignore[no-untyped-def]
        return self._outputs

    def run(self, _names, _inputs):  # type: ignore[no-untyped-def]
        heatmaps = np.zeros((1, 4, 32, 32), np.float32)
        for corner, point in enumerate(((5, 5), (26, 5), (26, 26), (5, 26))):
            heatmaps[0, corner, point[1], point[0]] = 10.0
        values = [
            heatmaps,
            heatmaps.copy(),
            np.ones((1, 1, 32, 32), np.float32),
            np.ones((1, 1, 32, 32), np.float32),
            np.array([[8.0]], np.float32),
            np.array([[self.outer_logit]], np.float32),
            np.array([[8.0, 0.0, 0.0, 0.0]], np.float32),
        ]
        return values


def test_runtime_low_outer_presence_never_decodes_outer(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import onnxruntime

    import screenrestore.geometry.detector as detector_module

    model_path = tmp_path / "fake.onnx"
    model_path.touch()
    monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeOnnxSession)
    original_decode = detector_module._decode_corner_heatmaps
    calls = 0

    def counted_decode(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(detector_module, "_decode_corner_heatmaps", counted_decode)
    prediction = detector_module.OnnxQuadDetector(model_path).predict(
        np.zeros((128, 128, 3), np.uint8)
    )

    assert calls == 1
    assert prediction.outer_quad is None
    assert len(prediction.candidates) == 1
    assert prediction.outer_presence_confidence < 0.5


def test_runtime_high_outer_presence_wrong_quad_triggers_layer_ambiguity(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    import onnxruntime

    import screenrestore.geometry.detector as detector_module
    from screenrestore.geometry.confidence import ConfidencePolicy
    from screenrestore.geometry.types import RejectionReason

    model_path = tmp_path / "fake.onnx"
    model_path.touch()
    monkeypatch.setattr(
        onnxruntime,
        "InferenceSession",
        lambda path, providers: _FakeOnnxSession(path, providers=providers, outer_logit=8.0),
    )
    decoded = iter(
        (
            (np.array([[15, 15], [75, 15], [75, 75], [15, 75]], np.float32), (0.9,) * 4),
            (np.array([[90, 90], [120, 90], [120, 120], [90, 120]], np.float32), (0.9,) * 4),
        )
    )
    monkeypatch.setattr(detector_module, "_decode_corner_heatmaps", lambda *_args: next(decoded))
    prediction = detector_module.OnnxQuadDetector(model_path).predict(
        np.zeros((128, 128, 3), np.uint8)
    )
    _score, reasons, _diagnostics = ConfidencePolicy().assess(
        prediction, None, (128, 128, 3)
    )

    assert prediction.outer_quad is not None
    assert prediction.layer_confidence == 0.0
    assert RejectionReason.LAYER_AMBIGUOUS in reasons


def test_old_six_output_onnx_contract_fails_explicitly(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import onnxruntime

    from screenrestore.geometry.detector import OnnxQuadDetector

    model_path = tmp_path / "p1.onnx"
    model_path.touch()
    monkeypatch.setattr(
        onnxruntime,
        "InferenceSession",
        lambda path, providers: _FakeOnnxSession(path, providers=providers, old=True),
    )

    with pytest.raises(RuntimeError, match="7-output P2"):
        OnnxQuadDetector(model_path)


def test_augmentation_updates_image_and_quad_with_same_homography(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import training.quadlocator.dataset as dataset_module

    image = np.zeros((128, 128, 3), np.uint8)
    quad = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], np.float32)
    cv2.fillConvexPoly(image, np.rint(quad * 127).astype(np.int32), (255, 0, 0))
    matrix = np.array([[0.9, 0.0, 10.0], [0.0, 0.9, 4.0], [0.0, 0.0, 1.0]], np.float32)
    monkeypatch.setattr(dataset_module, "_sample_homography", lambda *_args: matrix)
    monkeypatch.setattr(dataset_module, "_photometric_augmentation", lambda value, _rng: value)

    augmented, augmented_quad, _ = dataset_module._augment_sample(
        image, quad, None, np.random.default_rng(7)
    )

    assert augmented_quad is not None
    red_mask = (augmented[:, :, 0] > 128).astype(np.uint8)
    contour = max(cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    predicted_mask = np.zeros((128, 128), np.uint8)
    cv2.fillConvexPoly(predicted_mask, np.rint(augmented_quad * 127).astype(np.int32), 1)
    observed_mask = np.zeros((128, 128), np.uint8)
    cv2.drawContours(observed_mask, [contour], -1, 1, -1)
    intersection = np.logical_and(predicted_mask, observed_mask).sum()
    union = np.logical_or(predicted_mask, observed_mask).sum()
    assert intersection / union > 0.96


def test_source_group_balanced_sampler_only_uses_dataset_split(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from PIL import Image
    from training.quadlocator.dataset import QuadDataset, SourceGroupBalancedSampler

    Image.fromarray(np.zeros((128, 128, 3), np.uint8)).save(tmp_path / "image.jpg")
    records = []
    for index in range(8):
        records.append(
            {
                "image": "image.jpg",
                "split": "train" if index < 6 else "validation",
                "present": False,
                "target_class": "none",
                "content_quad": None,
                "outer_quad": None,
                "group_id": f"group-{index // 3}" if index < 6 else f"validation-{index}",
                "source": "source-a" if index < 3 else "source-b",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    dataset = QuadDataset(manifest, split="train", image_size=128, augment=False)
    sampler = SourceGroupBalancedSampler(dataset, seed=17, samples_per_epoch=100)

    sampled = list(sampler)
    assert all(0 <= index < len(dataset) for index in sampled)
    assert {dataset.records[index]["split"] for index in sampled} == {"train"}


def test_calibration_never_lowers_default_and_respects_precision() -> None:
    from training.quadlocator.calibrate import _calibrate_threshold

    result = _calibrate_threshold(
        [0.55, 0.60, 0.70, 0.80, 0.90],
        [False, False, False, True, True],
        minimum=0.58,
        minimum_precision=0.99,
    )

    assert result["threshold"] >= 0.58
    assert result["precision"] >= 0.99
    assert result["accepted_count"] == 2
