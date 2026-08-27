"""固定恢复退化切片的回归测试。"""

from __future__ import annotations

from training.restoration.evaluate_slices import evaluation_slices


def test_evaluation_slices_include_identity_and_major_camera_degradations() -> None:
    slices = evaluation_slices()

    assert set(slices) == {
        "clean_identity",
        "noise_light",
        "noise_heavy",
        "defocus",
        "motion",
        "jpeg",
        "exposure",
        "white_balance",
        "illumination",
        "compound_camera",
    }
    assert slices["clean_identity"].clean_probability == 1.0
    assert slices["defocus"].defocus_probability == 1.0
    assert slices["motion"].motion_probability == 1.0
    assert slices["jpeg"].jpeg_probability == 1.0
    assert slices["noise_heavy"].max_noise_std > slices["noise_light"].max_noise_std
