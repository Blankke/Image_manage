"""可选 QuadLocator-S 训练栈 smoke test。"""

from __future__ import annotations

import json

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
