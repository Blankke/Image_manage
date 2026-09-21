"""可选 QuadLocator-S 训练栈 smoke test。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("split", ["train", "validation"])
def test_training_rejects_subset_that_omits_target_source(split: str) -> None:
    from training.quadlocator.train import _check_required_sources

    rows = [
        {"source": "smartdoc"},
        {"source": "smartdoc"},
        {"source": "syngallery-reviewed"},
    ]
    _check_required_sources(rows, ["syngallery-reviewed=1"], split=split)
    with pytest.raises(ValueError, match="仅有 1 张"):
        _check_required_sources(rows, ["syngallery-reviewed=10"], split=split)
    with pytest.raises(ValueError, match="SOURCE=正整数"):
        _check_required_sources(rows, ["syngallery-reviewed=abc"], split=split)
    with pytest.raises(ValueError, match="须唯一"):
        _check_required_sources(rows, ["smartdoc=1", "smartdoc=1"], split=split)


def test_warm_start_coverage_rejects_wrong_model_width() -> None:
    from training.quadlocator.train import _assert_warm_start_coverage

    with pytest.raises(ValueError, match="模型宽度"):
        _assert_warm_start_coverage({"loaded_parameter_count": 27}, 145051, 0.95)
    _assert_warm_start_coverage({"loaded_parameter_count": 310000}, 311643, 0.95)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_sampling_reserves_rare_sources(tmp_path, split: str) -> None:  # type: ignore[no-untyped-def]
    from training.quadlocator.dataset import QuadDataset

    manifest = tmp_path / "manifest.jsonl"
    records = [
        {
            "image": f"{index}.jpg",
            "split": split,
            "group_id": f"group-{index}",
            "present": True,
            "target_class": "artwork",
            "source": "rare" if index < 4 else "common",
        }
        for index in range(100)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    options = dict(split=split, image_size=128, max_samples=10,
                   min_source_samples={"rare": 4}, seed=17)
    first = QuadDataset(manifest, **options)
    second = QuadDataset(manifest, **options)
    assert len(first) == 10
    assert sum(row["source"] == "rare" for row in first.records) == 4
    assert [row["image"] for row in first.records] == [row["image"] for row in second.records]
    with pytest.raises(ValueError, match="超过 max_samples"):
        QuadDataset(manifest, split=split, image_size=128, max_samples=3,
                    min_source_samples={"rare": 4})


def _state_dict_equal(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[name], second[name]) for name in first
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "private-labeled"),
        ("group_id", "private-dev:object-01"),
        ("capture_session", "private-dev:session-01"),
        ("subject_id", "private-dev:object-01"),
        ("source_capture", "private-dev:capture-01"),
        ("image", "private-validation/frame.jpg"),
    ],
)
def test_training_entry_rejects_private_manifest_rows(tmp_path, field, value) -> None:
    from training.quadlocator.train import _assert_public_training_manifest

    manifest = tmp_path / "public.geometry.jsonl"
    row = {
        "image": "geometry/public/frame.jpg",
        "source": "smartdoc",
        "group_id": "public:one",
        "split": "train",
    }
    manifest.write_text(json.dumps({**row, field: value}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="引用私人数据"):
        _assert_public_training_manifest(manifest)


def test_calibration_entry_rejects_private_manifest_before_checkpoint(tmp_path) -> None:
    from training.quadlocator.calibrate import main as calibrate

    manifest = tmp_path / "calibration.geometry.jsonl"
    manifest.write_text(
        json.dumps({"source": "private-development", "split": "validation"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="引用私人数据"):
        calibrate([
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--manifest", str(manifest),
            "--dataset-root", str(tmp_path),
            "--output", str(tmp_path / "calibration.json"),
        ])


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
    assert (
        generate_synthetic(
            ["--output-directory", str(data_directory), "--count", "12", "--size", "128"]
        )
        == 0
    )

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


def _loss_tensors(
    *, outer_present: bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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


def test_content_only_loss_does_not_backpropagate_other_heads() -> None:
    """G1 的 total 只能连接 content heatmap，禁止其它任务偷偷影响共享优化。"""

    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    loss, metrics = quadlocator_loss(outputs, targets, profile="content_only")
    loss.backward()

    assert metrics.keys() == {"total", "content_heatmap", "corner_geometry"}
    assert outputs["content_corner_heatmaps"].grad is not None
    for name in (
        "outer_corner_heatmaps",
        "content_mask_logits",
        "boundary_logits",
        "presence_logits",
        "outer_presence_logits",
        "class_logits",
    ):
        assert outputs[name].grad is None


def test_content_confidence_loss_isolates_heatmap_and_detaches_baseline() -> None:
    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    outputs["content_corner_base_heatmaps"] = torch.zeros(
        1, 4, 8, 8, requires_grad=True
    )
    loss, metrics = quadlocator_loss(outputs, targets, profile="content_confidence")
    loss.backward()

    assert metrics.keys() == {
        "total",
        "content_heatmap",
        "corner_geometry",
        "corner_cvar",
        "ambiguity",
        "residual_distillation",
    }
    assert outputs["content_corner_heatmaps"].grad is not None
    assert outputs["content_corner_base_heatmaps"].grad is None
    for name in (
        "outer_corner_heatmaps",
        "content_mask_logits",
        "boundary_logits",
        "presence_logits",
        "outer_presence_logits",
        "class_logits",
    ):
        assert outputs[name].grad is None


def test_decision_only_loss_only_backpropagates_presence_and_class() -> None:
    """决策头消融不能通过其它输出间接改写几何任务。"""

    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    loss, metrics = quadlocator_loss(outputs, targets, profile="decision_only")
    loss.backward()

    assert metrics.keys() == {"total", "presence", "classification"}
    assert outputs["presence_logits"].grad is not None
    assert outputs["class_logits"].grad is not None
    for name in (
        "content_corner_heatmaps",
        "outer_corner_heatmaps",
        "content_mask_logits",
        "boundary_logits",
        "outer_presence_logits",
    ):
        assert outputs[name].grad is None


def test_decision_correction_preserves_confident_base_without_backpropagating_teacher() -> None:
    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    outputs["presence_base_logits"] = torch.full((1, 1), 5.0, requires_grad=True)
    outputs["class_base_logits"] = torch.tensor(
        [[5.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    loss, metrics = quadlocator_loss(outputs, targets, profile="decision_correction")
    loss.backward()

    assert metrics.keys() == {
        "total",
        "presence_correction",
        "classification_correction",
        "presence_preserved_fraction",
        "class_preserved_fraction",
    }
    assert outputs["presence_logits"].grad is not None
    assert outputs["class_logits"].grad is not None
    assert outputs["presence_base_logits"].grad is None
    assert outputs["class_base_logits"].grad is None


def test_decision_head_scope_freezes_backbone_and_geometry_heads() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5)
    frozen = _configure_trainable_scope(model, "decision_heads")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all(
        name.startswith(
            (
                "presence_head.",
                "class_head.",
                "presence_local_head.",
                "class_local_head.",
            )
        )
        for name in trainable
    )
    assert "stem" in frozen
    assert "content_corner_head" in frozen


def test_local_decision_residual_is_zero_initialized_and_isolated() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5)
    assert torch.count_nonzero(model.presence_local_head.weight) == 0
    assert torch.count_nonzero(model.class_local_head.weight) == 0

    _configure_trainable_scope(model, "decision_residual_heads")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith(("presence_local_head.", "class_local_head.")) for name in trainable
    )


def test_class_context_branch_is_zero_initialized_and_isolated() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5).eval()
    image = torch.rand(2, 3, 128, 128)
    with torch.inference_mode():
        outputs = model(image)

    assert torch.count_nonzero(model.class_context_head.weight) == 0
    assert torch.equal(outputs["class_logits"], outputs["class_pre_context_logits"])
    with torch.no_grad():
        model.class_context_head.bias.fill_(0.75)
        adjusted = model(image)
    delta = adjusted["class_logits"] - adjusted["class_pre_context_logits"]
    assert torch.equal(delta[:, (0, 1, 3)], torch.zeros_like(delta[:, (0, 1, 3)]))
    assert torch.allclose(delta[:, 2], torch.full_like(delta[:, 2], 0.75))
    _configure_trainable_scope(model, "class_context_branch")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith(("class_context_encoder.", "class_context_head."))
        for name in trainable
    )


def test_artwork_context_branch_is_zero_initialized_and_isolated() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5).eval()
    image = torch.rand(2, 3, 128, 128)
    with torch.inference_mode():
        outputs = model(image)

    assert torch.count_nonzero(model.artwork_context_head[-1].weight) == 0
    assert torch.equal(outputs["class_logits"], outputs["class_pre_artwork_context_logits"])
    with torch.no_grad():
        model.artwork_context_head[-1].bias.fill_(0.75)
        adjusted = model(image)
    delta = adjusted["class_logits"] - adjusted["class_pre_artwork_context_logits"]
    assert torch.allclose(delta[:, 0], torch.full_like(delta[:, 0], 0.75))
    assert torch.equal(delta[:, (1, 2, 3)], torch.zeros_like(delta[:, (1, 2, 3)]))

    _configure_trainable_scope(model, "artwork_context_branch")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "artwork_context_head.0.weight",
        "artwork_context_head.0.bias",
        "artwork_context_head.2.weight",
        "artwork_context_head.2.bias",
    }


def test_old_checkpoint_migrates_only_zero_artwork_context_head() -> None:
    from training.quadlocator.model import QuadLocatorS, load_quadlocator_state_dict

    source = QuadLocatorS(width_multiplier=0.5).eval()
    legacy_state = {
        name: tensor.detach().clone()
        for name, tensor in source.state_dict().items()
        if not name.startswith("artwork_context_head.")
    }
    target = QuadLocatorS(width_multiplier=0.5).eval()
    migrated = load_quadlocator_state_dict(target, legacy_state)

    assert migrated == (
        "artwork_context_head.0.bias",
        "artwork_context_head.0.weight",
        "artwork_context_head.2.bias",
        "artwork_context_head.2.weight",
    )
    assert torch.count_nonzero(target.artwork_context_head[-1].weight) == 0
    with pytest.raises(RuntimeError, match="checkpoint 不兼容"):
        load_quadlocator_state_dict(
            QuadLocatorS(width_multiplier=0.5),
            {name: tensor for name, tensor in legacy_state.items() if name != "class_head.bias"},
        )


def test_content_confidence_residual_is_zero_initialized_and_isolated() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5).eval()
    image = torch.rand(2, 3, 128, 128)
    with torch.inference_mode():
        outputs = model(image)

    assert torch.count_nonzero(model.content_corner_residual_head[-1].weight) == 0
    assert torch.equal(
        outputs["content_corner_heatmaps"], outputs["content_corner_base_heatmaps"]
    )
    _configure_trainable_scope(model, "content_residual_head")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all(name.startswith("content_corner_residual_head.") for name in trainable)


def test_screen_context_loss_preserves_correct_non_screen_and_corrects_missed_screen() -> None:
    from training.quadlocator.losses import _screen_context_correction_loss

    base_logits = torch.tensor(
        [[3.0, 0.0, -1.0, 0.0], [3.0, 0.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    logits = base_logits.clone().requires_grad_(True)
    targets = torch.tensor([0, 2], dtype=torch.long)

    loss, preserved = _screen_context_correction_loss(logits, base_logits, targets)
    loss.backward()

    assert preserved.item() == pytest.approx(0.5)
    assert logits.grad is not None
    assert logits.grad[0, 2] > 0  # 正确 artwork 的 screen logit 应继续降低。
    assert logits.grad[1, 2] < 0  # 漏判 screen 的 screen logit 应提高。


def test_artwork_context_loss_preserves_correct_rest_and_corrects_missed_artwork() -> None:
    from training.quadlocator.losses import _one_vs_rest_context_correction_loss

    base_logits = torch.tensor(
        [[-1.0, 3.0, 0.0, 0.0], [-1.0, 3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    logits = base_logits.clone().requires_grad_(True)
    targets = torch.tensor([1, 0], dtype=torch.long)

    loss, preserved = _one_vs_rest_context_correction_loss(
        logits,
        base_logits,
        targets,
        target_index=0,
    )
    loss.backward()

    assert preserved.item() == pytest.approx(0.5)
    assert logits.grad is not None
    assert logits.grad[0, 0] > 0  # 正确 postcard 的 artwork logit 应继续降低。
    assert logits.grad[1, 0] < 0  # 漏判 artwork 的 artwork logit 应提高。


def test_content_head_scope_freezes_decision_residual_heads() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5)
    _configure_trainable_scope(model, "content_head")

    assert not any(parameter.requires_grad for parameter in model.presence_local_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.class_local_head.parameters())


def test_target_v7_contains_small_offcenter_closeup_and_cropped_screen_samples() -> None:
    from training.quadlocator.generate_synthetic import _sample_center, _target_scale_range

    assert _target_scale_range("far_screen") == (0.14, 0.34)
    assert _target_scale_range("monitor_photo_viewer") == (0.68, 0.90)
    assert _target_scale_range("cropped_monitor_photo_viewer") == (1.04, 1.28)
    rng = np.random.default_rng(9)
    center = _sample_center(rng, 640, 128, 96, 38.0, "offcenter_screen")

    assert 102 <= center[0] <= 537
    assert 86 <= center[1] <= 553
    assert abs(center[0] - 320) > 100 or abs(center[1] - 320) > 100


def test_target_v7_screen_uses_dark_bezel_without_artwork_mat() -> None:
    from training.quadlocator.generate_synthetic import _nested_patch

    screen, screen_quad, has_screen_outer = _nested_patch(
        np.random.default_rng(11), 220, 150, "screen", "monitor_screen", []
    )
    artwork, artwork_quad, has_artwork_outer = _nested_patch(
        np.random.default_rng(11), 220, 150, "artwork", "mat_artwork", []
    )
    screen_x = int(screen_quad[0, 0]) - 1
    screen_y = int(screen_quad[0, 1]) - 1
    artwork_x = int(artwork_quad[0, 0]) - 1
    artwork_y = int(artwork_quad[0, 1]) - 1

    assert has_screen_outer and has_artwork_outer
    assert float(screen[screen_y, screen_x].mean()) < 80.0
    assert float(artwork[artwork_y, artwork_x].mean()) > 180.0


def test_target_v7_screen_viewer_labels_inner_image_instead_of_entire_display() -> None:
    from training.quadlocator.generate_synthetic import _nested_patch

    patch, content_quad, has_outer = _nested_patch(
        np.random.default_rng(21),
        320,
        210,
        "screen",
        "monitor_photo_viewer",
        [],
    )

    assert has_outer
    assert content_quad[0, 0] > 5
    assert content_quad[0, 1] > 5
    assert content_quad[2, 0] < patch.shape[1] - 5
    assert content_quad[2, 1] < patch.shape[0] - 5
    assert cv2.contourArea(content_quad) < patch.shape[0] * patch.shape[1] * 0.70


def test_target_v7_cropped_screen_keeps_content_complete_and_omits_incomplete_outer() -> None:
    from training.quadlocator.generate_synthetic import _sample

    rng = np.random.default_rng(37)
    record = None
    for index in range(600):
        _image, candidate = _sample(rng, 256, index, False)
        if str(candidate["scene_type"]).startswith("cropped_monitor_"):
            record = candidate
            break

    assert record is not None
    assert record["target_class"] == "screen"
    assert record["outer_quad"] is None
    content_quad = np.asarray(record["content_quad"], dtype=np.float32)
    assert np.all(content_quad > 0.005)
    assert np.all(content_quad < 0.995)


def test_target_v7_mounted_poster_has_paper_outer_layer() -> None:
    from training.quadlocator.generate_synthetic import _nested_patch

    patch, content_quad, has_outer = _nested_patch(
        np.random.default_rng(29), 180, 260, "artwork", "mounted_wall_poster", []
    )
    border_x = max(0, int(content_quad[0, 0]) - 1)
    border_y = max(0, int(content_quad[0, 1]) - 1)

    assert has_outer
    assert cv2.contourArea(content_quad) < patch.shape[0] * patch.shape[1] * 0.92
    assert float(patch[border_y, border_x].mean()) > 180.0


def test_target_v8_neighbor_poster_uses_textured_content_and_paper_border(tmp_path) -> None:
    from PIL import Image
    from training.quadlocator.generate_synthetic import _draw_textured_neighbor_poster

    texture = np.zeros((96, 96, 3), np.uint8)
    texture[:, :48] = (225, 30, 30)
    texture[:, 48:] = (25, 40, 220)
    texture_path = tmp_path / "public-texture.png"
    Image.fromarray(texture).save(texture_path)
    canvas = np.full((256, 256, 3), 240, np.uint8)
    _draw_textured_neighbor_poster(
        canvas,
        np.random.default_rng(11),
        (52, 39),
        (120, 150),
        [texture_path],
    )
    region = canvas[45:185, 58:168]
    # 邻近目标必须同时呈现明亮纸边和不同颜色的图像区域；纯色块无法通过。
    assert np.any(np.all(region > 190, axis=2))
    assert np.any((region[:, :, 0] > 170) & (region[:, :, 2] < 90))
    assert np.any((region[:, :, 2] > 170) & (region[:, :, 0] < 90))


def test_target_v8_neighbor_posters_do_not_cover_reserved_target(monkeypatch) -> None:
    from training.quadlocator import generate_synthetic

    placements: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def record_placement(_image, _rng, origin, shape, _content_paths) -> None:
        placements.append((origin, shape))

    monkeypatch.setattr(generate_synthetic, "_draw_textured_neighbor_poster", record_placement)
    canvas = np.full((640, 640, 3), 240, np.uint8)
    reserved = np.asarray([[155, 125], [480, 125], [480, 480], [155, 480]], np.float32)
    generate_synthetic._draw_wall_poster_context(
        canvas,
        np.random.default_rng(38),
        clustered=True,
        content_paths=[],
        reserved_quad=reserved,
    )
    assert placements
    for (x0, y0), (width, height) in placements:
        overlap_width = max(0, min(x0 + width, 480) - max(x0, 155))
        overlap_height = max(0, min(y0 + height, 480) - max(y0, 125))
        assert overlap_width * overlap_height <= width * height * 0.08


def test_target_v9_wood_desk_has_texture_without_changing_quad() -> None:
    from training.quadlocator.generate_synthetic import _draw_wood_desk_context, _sample

    quad = np.asarray([[70, 55], [190, 55], [190, 220], [70, 220]], np.float32)
    image = np.full((256, 256, 3), 180, np.uint8)
    _draw_wood_desk_context(image, np.random.default_rng(11), reserved_quad=quad)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1),
    )
    assert float(np.median(gradient)) > 10.0

    # 新远拍类型保持完整可见的杂志内容层，没有误加画框层。
    rng = np.random.default_rng(29)
    samples = (_sample(rng, 256, index, False)[1] for index in range(600))
    far = next(row for row in samples if row["scene_type"] == "desk_magazine_far")
    assert far["target_class"] == "artwork"
    assert far["outer_quad"] is None
    content = np.asarray(far["content_quad"], np.float32)
    assert np.all((content >= 0.0) & (content <= 1.0))


def test_target_v9_wall_tape_stays_near_paper_corners() -> None:
    from training.quadlocator.generate_synthetic import _draw_poster_tape

    image = np.full((256, 256, 3), 150, np.uint8)
    quad = np.asarray([[60, 55], [190, 55], [190, 220], [60, 220]], np.float32)
    _draw_poster_tape(image, np.random.default_rng(42), quad)
    assert np.any(image != 150)
    assert np.array_equal(image[128, 128], np.asarray([150, 150, 150], np.uint8))
    assert np.array_equal(image[10, 10], np.asarray([150, 150, 150], np.uint8))


def test_target_v9_rejected_targets_are_not_marked_visible(monkeypatch) -> None:
    from training.quadlocator import generate_synthetic

    _image, hard_negative = generate_synthetic._sample(np.random.default_rng(3), 128, 0, True)
    assert hard_negative["present"] is False
    assert hard_negative["visible"] is False

    monkeypatch.setattr(generate_synthetic, "_scene_type", lambda _rng, _class: "multiple_artworks")
    _image, ambiguous = generate_synthetic._sample(np.random.default_rng(4), 128, 1, False)
    assert ambiguous["present"] is False
    assert ambiguous["visible"] is False
    assert ambiguous["ambiguous"] is True


def test_target_v7_postcard_uses_card_edge_instead_of_artwork_frame() -> None:
    from training.quadlocator.generate_synthetic import _nested_patch

    postcard, postcard_quad, has_outer = _nested_patch(
        np.random.default_rng(12), 240, 160, "postcard", "desk_postcard", []
    )
    x = int(postcard_quad[0, 0]) - 1
    y = int(postcard_quad[0, 1]) - 1

    assert has_outer
    assert float(postcard[y, x].mean()) > 190.0


def test_target_v7_patch_shapes_encode_common_object_ratios_without_fixing_orientation() -> None:
    from training.quadlocator.generate_synthetic import _sample_patch_size

    rng = np.random.default_rng(13)
    screen_ratios = []
    postcard_ratios = []
    for _ in range(100):
        width, height = _sample_patch_size(rng, 640, 0.4, 0.78, "screen")
        screen_ratios.append(width / height)
        width, height = _sample_patch_size(rng, 640, 0.4, 0.78, "postcard")
        postcard_ratios.append(width / height)

    assert sum(ratio > 1.35 for ratio in screen_ratios) >= 65
    assert any(ratio < 0.8 for ratio in screen_ratios)
    assert sum(ratio > 1.3 for ratio in postcard_ratios) >= 65
    assert any(ratio < 0.8 for ratio in postcard_ratios)


def test_p5_manifest_builder_namespaces_target_groups(tmp_path) -> None:
    from scripts.build_p5_target_manifest import build_manifest

    data_root = tmp_path / "data"
    base_root = data_root / "geometry" / "base"
    target_root = data_root / "geometry" / "target"
    base_root.mkdir(parents=True)
    target_root.mkdir(parents=True)
    (base_root / "base.jpg").write_bytes(b"base")
    (target_root / "target.jpg").write_bytes(b"target")
    base = {
        "image": "geometry/base/base.jpg",
        "split": "train",
        "group_id": "synthetic-1",
        "capture_session": "session-1",
        "target_class": "none",
        "present": False,
    }
    target = {
        "image": "target.jpg",
        "split": "validation",
        "group_id": "synthetic-1",
        "capture_session": "session-1",
        "target_class": "screen",
        "present": True,
    }
    base_manifest = base_root / "manifest.jsonl"
    target_manifest = target_root / "manifest.jsonl"
    base_manifest.write_text(json.dumps(base) + "\n", encoding="utf-8")
    target_manifest.write_text(json.dumps(target) + "\n", encoding="utf-8")
    output = data_root / "manifests" / "p5.jsonl"

    rows = build_manifest(
        data_root=data_root,
        base_manifest=base_manifest,
        target_manifest=target_manifest,
        target_root=target_root,
        target_namespace="target-v4",
        output=output,
    )

    assert rows[1]["image"] == "geometry/target/target.jpg"
    assert rows[1]["group_id"] == "target-v4:synthetic-1"
    assert rows[1]["capture_session"] == "target-v4:session-1"


def test_p5_manifest_builder_inherits_real_capture_identity(tmp_path) -> None:
    from scripts.build_p5_target_manifest import build_manifest

    root = tmp_path / "data"
    source = root / "geometry" / "smartdoc" / "frame.jpg"
    composite = root / "geometry" / "surface" / "cover.jpg"
    source.parent.mkdir(parents=True)
    composite.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    composite.write_bytes(b"composite")
    base = {
        "image": "geometry/smartdoc/frame.jpg", "split": "train", "group_id": "smartdoc:book",
        "capture_session": "smartdoc:desk:book", "target_class": "postcard", "present": True,
    }
    derived = {
        "image": "geometry/surface/cover.jpg", "split": "train", "group_id": "smartdoc:book",
        "capture_session": "surface:desk:book", "source_capture_id": base["image"],
        "target_class": "postcard", "present": True,
    }
    base_path = root / "base.jsonl"
    target_path = root / "target.jsonl"
    base_path.write_text(json.dumps(base) + "\n")
    target_path.write_text(json.dumps(derived) + "\n")
    result = build_manifest(
        data_root=root, base_manifest=base_path, target_manifest=target_path,
        target_root=root, target_namespace="surface", output=root / "merged.jsonl",
    )
    assert result[1]["group_id"] == base["group_id"]
    assert result[1]["capture_session"] == base["capture_session"]

    derived["split"] = "validation"
    target_path.write_text(json.dumps(derived) + "\n")
    with pytest.raises(ValueError, match="原片的 split"):
        build_manifest(
            data_root=root, base_manifest=base_path, target_manifest=target_path,
            target_root=root, target_namespace="surface", output=root / "rejected.jsonl",
        )


def test_p5_manifest_builder_rejects_private_training_paths(tmp_path) -> None:
    from scripts.build_p5_target_manifest import build_manifest

    data_root = tmp_path / "data"
    private_root = data_root / "private-validation"
    target_root = data_root / "geometry" / "target"
    private_root.mkdir(parents=True)
    target_root.mkdir(parents=True)
    (private_root / "sample.jpg").write_bytes(b"private")
    (target_root / "target.jpg").write_bytes(b"target")
    base_manifest = private_root / "manifest.jsonl"
    target_manifest = target_root / "manifest.jsonl"
    base_manifest.write_text("{}\n", encoding="utf-8")
    target_manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="禁止读取 private 路径"):
        build_manifest(
            data_root=data_root,
            base_manifest=base_manifest,
            target_manifest=target_manifest,
            target_root=target_root,
            target_namespace="target-v4",
            output=data_root / "manifests" / "p5.jsonl",
        )


@pytest.mark.parametrize(
    ("profile", "expected_metrics"),
    (
        ("content_heatmap_only", {"total", "content_heatmap"}),
        ("content_coordinate_only", {"total", "corner_geometry"}),
    ),
)
def test_content_loss_components_are_independently_auditable(
    profile: str, expected_metrics: set[str]
) -> None:
    """G3.5 必须能将 heatmap 与 local-softargmax 坐标梯度独立复现。"""

    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    loss, metrics = quadlocator_loss(outputs, targets, profile=profile)
    loss.backward()

    assert metrics.keys() == expected_metrics
    assert outputs["content_corner_heatmaps"].grad is not None
    for name in (
        "outer_corner_heatmaps",
        "content_mask_logits",
        "boundary_logits",
        "presence_logits",
        "outer_presence_logits",
        "class_logits",
    ):
        assert outputs[name].grad is None


def test_content_mask_loss_only_backpropagates_content_and_mask() -> None:
    """G2 只允许 content corner 与 content mask 共同影响共享优化。"""

    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    loss, metrics = quadlocator_loss(outputs, targets, profile="content_mask")
    loss.backward()

    assert metrics.keys() == {"total", "content_heatmap", "mask", "corner_geometry"}
    assert outputs["content_corner_heatmaps"].grad is not None
    assert outputs["content_mask_logits"].grad is not None
    for name in (
        "outer_corner_heatmaps",
        "boundary_logits",
        "presence_logits",
        "outer_presence_logits",
        "class_logits",
    ):
        assert outputs[name].grad is None


def test_content_boundary_loss_only_backpropagates_content_and_boundary() -> None:
    """G3 只允许 content corner 与 boundary 共同影响共享优化。"""

    from training.quadlocator.losses import quadlocator_loss

    outputs, targets = _loss_tensors(outer_present=True)
    loss, metrics = quadlocator_loss(outputs, targets, profile="content_boundary")
    loss.backward()

    assert metrics.keys() == {"total", "content_heatmap", "boundary", "corner_geometry"}
    assert outputs["content_corner_heatmaps"].grad is not None
    assert outputs["boundary_logits"].grad is not None
    for name in (
        "outer_corner_heatmaps",
        "content_mask_logits",
        "presence_logits",
        "outer_presence_logits",
        "class_logits",
    ):
        assert outputs[name].grad is None


def test_content_backbone_scope_freezes_all_non_content_heads() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5)
    frozen = _configure_trainable_scope(model, "content_backbone")

    assert "content_corner_head" not in frozen
    assert all(parameter.requires_grad for parameter in model.content_corner_head.parameters())
    assert all(parameter.requires_grad for parameter in model.stem.parameters())
    for name in frozen:
        assert not any(parameter.requires_grad for parameter in getattr(model, name).parameters())


def test_content_mask_backbone_scope_keeps_only_content_and_mask_heads() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5)
    frozen = _configure_trainable_scope(model, "content_mask_backbone")

    assert all(parameter.requires_grad for parameter in model.content_corner_head.parameters())
    assert all(parameter.requires_grad for parameter in model.content_mask_head.parameters())
    assert all(parameter.requires_grad for parameter in model.stem.parameters())
    for name in frozen:
        assert not any(parameter.requires_grad for parameter in getattr(model, name).parameters())


def test_content_boundary_backbone_scope_keeps_only_content_and_boundary_heads() -> None:
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _configure_trainable_scope

    model = QuadLocatorS(width_multiplier=0.5)
    frozen = _configure_trainable_scope(model, "content_boundary_backbone")

    assert all(parameter.requires_grad for parameter in model.content_corner_head.parameters())
    assert all(parameter.requires_grad for parameter in model.boundary_head.parameters())
    assert all(parameter.requires_grad for parameter in model.stem.parameters())
    for name in frozen:
        assert not any(parameter.requires_grad for parameter in getattr(model, name).parameters())


def test_p4_loss_profiles_publish_only_their_participating_losses() -> None:
    from training.quadlocator.train import _participating_losses

    assert _participating_losses("content_only") == [
        "content_heatmap",
        "content_corner_geometry",
    ]
    assert _participating_losses("content_heatmap_only") == ["content_heatmap"]
    assert _participating_losses("content_coordinate_only") == ["content_corner_geometry"]
    assert _participating_losses("content_mask") == [
        "content_heatmap",
        "content_mask",
        "content_corner_geometry",
    ]
    assert _participating_losses("content_boundary") == [
        "content_heatmap",
        "balanced_boundary",
        "content_corner_geometry",
    ]


def test_geometry_selection_rejects_iou_tail_collapse_for_tiny_nce_gain() -> None:
    from training.quadlocator.train import _geometry_selection_key

    stable = {
        "content_corner_nce_median": 0.031,
        "content_corner_nce_p95": 0.154,
        "content_iou_p05": 0.088,
        "content_iou_median": 0.677,
        "content_strict_correct_rate": 0.011,
    }
    collapsed = {
        "content_corner_nce_median": 0.030,
        "content_corner_nce_p95": 0.151,
        "content_iou_p05": 0.0,
        "content_iou_median": 0.652,
        "content_strict_correct_rate": 0.007,
    }

    assert _geometry_selection_key(stable) > _geometry_selection_key(collapsed)


def test_geometry_tail_watchdog_requires_both_tail_regressions() -> None:
    from training.quadlocator.train import _is_geometry_tail_collapse

    reference = {"content_corner_nce_p95": 0.154, "content_iou_p05": 0.153}
    assert _is_geometry_tail_collapse(
        {"content_corner_nce_p95": 0.21, "content_iou_p05": 0.09},
        reference,
        nce_p95_ratio=1.30,
        iou_p05_ratio=0.65,
    )
    assert not _is_geometry_tail_collapse(
        {"content_corner_nce_p95": 0.21, "content_iou_p05": 0.20},
        reference,
        nce_p95_ratio=1.30,
        iou_p05_ratio=0.65,
    )
    assert not _is_geometry_tail_collapse(
        {"content_corner_nce_p95": 0.16, "content_iou_p05": 0.09},
        reference,
        nce_p95_ratio=1.30,
        iou_p05_ratio=0.65,
    )


def test_best_geometry_eligibility_keeps_tail_safe_warm_start() -> None:
    from training.quadlocator.train import _is_geometry_eligible

    reference = {
        "content_corner_nce_median": 0.030,
        "content_corner_nce_p95": 0.154,
        "content_iou_median": 0.700,
        "content_iou_p05": 0.153,
    }
    assert _is_geometry_eligible(
        {
            "content_corner_nce_median": 0.031,
            "content_corner_nce_p95": 0.160,
            "content_iou_median": 0.695,
            "content_iou_p05": 0.170,
        },
        reference,
        nce_p95_ratio=1.10,
        iou_p05_ratio=0.90,
        nce_median_tolerance=0.002,
        iou_median_tolerance=0.01,
    )
    assert not _is_geometry_eligible(
        {
            "content_corner_nce_median": 0.031,
            "content_corner_nce_p95": 0.201,
            "content_iou_median": 0.695,
            "content_iou_p05": 0.170,
        },
        reference,
        nce_p95_ratio=1.10,
        iou_p05_ratio=0.90,
        nce_median_tolerance=0.002,
        iou_median_tolerance=0.01,
    )
    assert not _is_geometry_eligible(
        {
            "content_corner_nce_median": 0.031,
            "content_corner_nce_p95": 0.160,
            "content_iou_median": 0.695,
            "content_iou_p05": 0.099,
        },
        reference,
        nce_p95_ratio=1.10,
        iou_p05_ratio=0.90,
        nce_median_tolerance=0.002,
        iou_median_tolerance=0.01,
    )
    assert not _is_geometry_eligible(
        {
            "content_corner_nce_median": 0.04,
            "content_corner_nce_p95": 0.160,
            "content_iou_median": 0.695,
            "content_iou_p05": 0.170,
        },
        reference,
        nce_p95_ratio=1.10,
        iou_p05_ratio=0.90,
        nce_median_tolerance=0.002,
        iou_median_tolerance=0.01,
    )
    assert not _is_geometry_eligible(
        {
            "content_corner_nce_median": 0.031,
            "content_corner_nce_p95": 0.160,
            "content_iou_median": 0.68,
            "content_iou_p05": 0.170,
        },
        reference,
        nce_p95_ratio=1.10,
        iou_p05_ratio=0.90,
        nce_median_tolerance=0.002,
        iou_median_tolerance=0.01,
    )


def test_validation_nce_uses_target_bbox_diagonal_and_is_scale_invariant() -> None:
    from training.quadlocator.metrics import _corner_nce

    target = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], np.float32)
    predicted = target + np.array([0.06, 0.0], np.float32)
    expected = 0.06 / np.sqrt(0.6**2 + 0.6**2)

    assert _corner_nce(predicted, target) == pytest.approx(expected)
    assert _corner_nce(predicted * 256.0, target * 256.0) == pytest.approx(expected)


def test_milestone_checkpoint_keeps_epoch_state_and_refuses_overwrite(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from training.quadlocator.model import QuadLocatorS
    from training.quadlocator.train import _save_milestone_checkpoint

    directory = tmp_path / "checkpoints"
    directory.mkdir()
    model = QuadLocatorS(width_multiplier=0.5)
    checkpoint = {
        "epoch": 4,
        "state_dict": model.state_dict(),
        "validation_metrics": {"content_corner_nce_p95": 0.2},
    }
    torch.save(checkpoint, tmp_path / "last.pt")

    identity = _save_milestone_checkpoint(checkpoint, directory, 4)
    loaded = torch.load(directory / "epoch-004.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)

    assert identity["epoch"] == 4
    assert identity["path"] == str(directory / "epoch-004.pt")
    assert len(str(identity["sha256"])) == 64
    assert loaded["epoch"] == checkpoint["epoch"]
    assert loaded["validation_metrics"] == checkpoint["validation_metrics"]
    assert _state_dict_equal(loaded["state_dict"], checkpoint["state_dict"])
    assert loaded["epoch"] == last["epoch"]
    assert loaded["validation_metrics"] == last["validation_metrics"]
    assert _state_dict_equal(loaded["state_dict"], last["state_dict"])
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        _save_milestone_checkpoint(checkpoint, directory, 4)


def test_checkpoint_epoch_parser_and_scheduler_horizon_are_explicit() -> None:
    from training.quadlocator.train import _checkpoint_epochs, _resolve_scheduler_t_max

    assert _checkpoint_epochs("8,1,4,4") == (1, 4, 8)
    assert _resolve_scheduler_t_max(8, 16) == 16
    assert _resolve_scheduler_t_max(8, 0) == 8
    with pytest.raises(ValueError, match="不小于"):
        _resolve_scheduler_t_max(16, 8)


def test_explicit_scheduler_horizon_preserves_first_eight_epoch_trajectory() -> None:
    def learning_rates(actual_epochs: int, horizon: int) -> list[float]:
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon)
        result = []
        for _ in range(actual_epochs):
            optimizer.step()
            scheduler.step()
            result.append(float(scheduler.get_last_lr()[0]))
        return result

    full = learning_rates(16, 16)
    prefix_only = learning_rates(8, 16)
    changed_horizon = learning_rates(8, 8)

    assert prefix_only == full[:8]
    assert changed_horizon != full[:8]


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


@pytest.mark.parametrize("format_version", [2, 3, 4, 5])
def test_exported_onnx_uses_seven_output_contract(
    tmp_path,  # type: ignore[no-untyped-def]
    format_version: int,
) -> None:
    ort = pytest.importorskip("onnxruntime")
    from training.quadlocator.export_onnx import OUTPUT_NAMES
    from training.quadlocator.export_onnx import main as export_onnx
    from training.quadlocator.model import QuadLocatorS

    model = QuadLocatorS(width_multiplier=0.5)
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "model.onnx"
    torch.save(
        {
            "format_version": format_version,
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
    def __init__(
        self, _path: str, *, providers: list[str], outer_logit: float = -8.0, old: bool = False
    ) -> None:
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
    assert (
        prediction.decoder_diagnostics["content"]["coherence"]["version"]
        == "quad-coherent-evidence-guarded-v1"
    )


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
    monkeypatch.setattr(
        detector_module,
        "_decode_corner_heatmaps",
        lambda *_args, **_kwargs: next(decoded),
    )
    prediction = detector_module.OnnxQuadDetector(model_path).predict(
        np.zeros((128, 128, 3), np.uint8)
    )
    _score, reasons, _diagnostics = ConfidencePolicy().assess(prediction, None, (128, 128, 3))

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
    contour = max(
        cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        key=cv2.contourArea,
    )
    predicted_mask = np.zeros((128, 128), np.uint8)
    cv2.fillConvexPoly(predicted_mask, np.rint(augmented_quad * 127).astype(np.int32), 1)
    observed_mask = np.zeros((128, 128), np.uint8)
    cv2.drawContours(observed_mask, [contour], -1, 1, -1)
    intersection = np.logical_and(predicted_mask, observed_mask).sum()
    union = np.logical_or(predicted_mask, observed_mask).sum()
    assert intersection / union > 0.96


def test_augmentation_modes_separate_geometry_and_photometry(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import training.quadlocator.dataset as dataset_module

    image = np.full((128, 128, 3), 100, np.uint8)
    cv2.rectangle(image, (32, 32), (95, 95), (180, 20, 20), -1)
    quad = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], np.float32)
    matrix = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]], np.float32)
    monkeypatch.setattr(dataset_module, "_sample_homography", lambda *_args: matrix)
    monkeypatch.setattr(
        dataset_module,
        "_photometric_augmentation",
        lambda value, _rng: np.clip(value.astype(np.int16) + 10, 0, 255).astype(np.uint8),
    )

    photo, photo_quad, _ = dataset_module._augment_sample(
        image, quad, None, np.random.default_rng(7), mode="photometric"
    )
    geometric, geometric_quad, _ = dataset_module._augment_sample(
        image, quad, None, np.random.default_rng(7), mode="geometric"
    )

    assert np.array_equal(photo_quad, quad)
    assert np.array_equal(photo[0, 0], np.array([110, 110, 110], np.uint8))
    assert geometric_quad is not None
    assert not np.array_equal(geometric_quad, quad)
    assert not np.array_equal(geometric, image)


def test_dataset_geometric_augmentation_updates_model_input_and_quad(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """回归：几何增强后的模型输入与监督四角必须使用同一变换。"""

    import training.quadlocator.dataset as dataset_module
    from PIL import Image
    from training.quadlocator.dataset import QuadDataset

    image = np.full((128, 128, 3), 30, np.uint8)
    quad = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], np.float32)
    cv2.fillConvexPoly(image, np.rint(quad * 127).astype(np.int32), (240, 20, 20))
    Image.fromarray(image).save(tmp_path / "image.png")
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "image": "image.png",
                "split": "train",
                "present": True,
                "target_class": "artwork",
                "content_quad": quad.tolist(),
                "outer_quad": None,
                "group_id": "group-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = np.array([[1.0, 0.0, 16.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]], np.float32)
    monkeypatch.setattr(dataset_module, "_sample_homography", lambda *_args: matrix)

    baseline = QuadDataset(
        tmp_path / "manifest.jsonl", split="train", image_size=128, augmentation_mode="none"
    )[0]
    augmented = QuadDataset(
        tmp_path / "manifest.jsonl", split="train", image_size=128, augmentation_mode="geometric"
    )[0]

    assert not torch.equal(augmented["image"], baseline["image"])
    assert not torch.equal(augmented["content_corners"], baseline["content_corners"])
    observed = augmented["image"][0].numpy() > 0.7
    yy, xx = np.where(observed)
    observed_center = np.array([xx.mean() / 127.0, yy.mean() / 127.0], np.float32)
    assert np.allclose(
        observed_center, augmented["content_corners"].numpy().mean(axis=0), atol=0.03
    )


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


def test_class_balanced_sampler_keeps_minority_classes_visible(tmp_path) -> None:
    from PIL import Image
    from training.quadlocator.dataset import QuadDataset, SourceGroupBalancedSampler

    Image.fromarray(np.zeros((128, 128, 3), np.uint8)).save(tmp_path / "image.jpg")
    records = []
    for index in range(100):
        target_class = "postcard" if index < 94 else "none"
        records.append(
            {
                "image": "image.jpg",
                "split": "train",
                "present": target_class != "none",
                "target_class": target_class,
                "content_quad": None,
                "outer_quad": None,
                "group_id": f"group-{index}",
                "source": "source-a",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    dataset = QuadDataset(manifest, split="train", image_size=128, augment=False)
    sampler = SourceGroupBalancedSampler(
        dataset,
        seed=17,
        samples_per_epoch=1000,
        class_balancing=True,
    )

    sampled_classes = [dataset.records[index]["target_class"] for index in sampler]
    none_fraction = sampled_classes.count("none") / len(sampled_classes)
    assert 0.45 <= none_fraction <= 0.55


def test_focus_taxonomy_sampler_reserves_requested_fraction(tmp_path) -> None:
    from PIL import Image
    from training.quadlocator.dataset import QuadDataset, SourceGroupBalancedSampler

    Image.fromarray(np.zeros((128, 128, 3), np.uint8)).save(tmp_path / "image.jpg")
    records = []
    for index in range(100):
        records.append(
            {
                "image": "image.jpg",
                "split": "train",
                "present": True,
                "target_class": "artwork",
                "content_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "outer_quad": None,
                "group_id": f"group-{index}",
                "source": "source-a",
                "scene_type": "printed_poster" if index < 5 else "artwork",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    dataset = QuadDataset(manifest, split="train", image_size=128, augment=False)
    sampler = SourceGroupBalancedSampler(
        dataset,
        seed=17,
        samples_per_epoch=4000,
        focus_taxonomies=("printed_poster",),
        focus_probability=0.5,
    )

    sampled = [dataset.records[index]["scene_type"] for index in sampler]
    # 常规 replay 本身还会抽中约 5% poster，所以总比例略高于保留的 50%。
    focused_fraction = sampled.count("printed_poster") / len(sampled)
    assert 0.50 <= focused_fraction <= 0.56


def test_focus_taxonomy_sampler_rejects_unmatched_scene(tmp_path) -> None:
    from PIL import Image
    from training.quadlocator.dataset import QuadDataset, SourceGroupBalancedSampler

    Image.fromarray(np.zeros((128, 128, 3), np.uint8)).save(tmp_path / "image.jpg")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image": "image.jpg",
                "split": "train",
                "present": False,
                "target_class": "none",
                "content_quad": None,
                "outer_quad": None,
                "group_id": "group-0",
                "source": "source-a",
                "scene_type": "negative",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = QuadDataset(manifest, split="train", image_size=128, augment=False)
    with pytest.raises(ValueError, match="没有匹配"):
        SourceGroupBalancedSampler(
            dataset,
            seed=17,
            focus_taxonomies=("printed_poster",),
            focus_probability=0.5,
        )


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


def test_checkpoint_blend_only_changes_selected_module() -> None:
    from scripts.blend_quadlocator_checkpoints import blend_state_dict

    baseline = {
        "content_corner_head.weight": torch.tensor([0.0, 2.0]),
        "content_corner_head.counter": torch.tensor(2, dtype=torch.int64),
        "class_head.weight": torch.tensor([4.0]),
    }
    challenger = {
        "content_corner_head.weight": torch.tensor([4.0, 6.0]),
        "content_corner_head.counter": torch.tensor(8, dtype=torch.int64),
        "class_head.weight": torch.tensor([9.0]),
    }

    blended, names = blend_state_dict(
        baseline,
        challenger,
        alpha=0.25,
        prefixes=("content_corner_head.",),
    )

    assert names == ["content_corner_head.weight", "content_corner_head.counter"]
    assert torch.equal(blended["content_corner_head.weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(blended["content_corner_head.counter"], torch.tensor(2))
    assert torch.equal(blended["class_head.weight"], baseline["class_head.weight"])


def test_checkpoint_blend_keeps_new_baseline_parameters_missing_from_old_challenger() -> None:
    from scripts.blend_quadlocator_checkpoints import blend_state_dict

    baseline = {
        "content_corner_head.weight": torch.tensor([0.0]),
        "class_local_head.weight": torch.tensor([3.0]),
    }
    challenger = {"content_corner_head.weight": torch.tensor([2.0])}

    blended, _ = blend_state_dict(
        baseline,
        challenger,
        alpha=0.5,
        prefixes=("content_corner_head.",),
    )

    assert torch.equal(blended["content_corner_head.weight"], torch.tensor([1.0]))
    assert torch.equal(blended["class_local_head.weight"], torch.tensor([3.0]))


def test_checkpoint_blend_decision_metrics_include_negative_recall() -> None:
    from scripts.blend_quadlocator_checkpoints import _decision_metrics

    metrics = _decision_metrics(
        {
            "class_confusion": [
                [8, 1, 0, 1],
                [0, 9, 1, 0],
                [0, 0, 10, 0],
                [0, 0, 1, 9],
            ],
            "class_recall": {
                "artwork": 0.8,
                "postcard": 0.9,
                "screen": 1.0,
                "none": 0.9,
            },
            "no_candidate_rate": 0.25,
        }
    )

    assert metrics["class_accuracy"] == 0.9
    assert metrics["none_recall"] == 0.9
    assert metrics["target_macro_recall"] == pytest.approx(0.9)


def test_context_branch_prefix_enables_decision_gate() -> None:
    from scripts.blend_quadlocator_checkpoints import _is_decision_blend

    assert _is_decision_blend(("class_context_encoder.", "class_context_head."))
    assert _is_decision_blend(("artwork_context_head.",))
    assert not _is_decision_blend(("content_corner_head.",))
