"""项目文件、中文路径导出和 CLI 烟雾测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from screenrestore.cli import main
from screenrestore.core.presets import PresetId, build_default_pipeline, build_registry
from screenrestore.io.image_exporter import (
    ExportFormat,
    ExportOptions,
    ImageExportError,
    export_image,
)
from screenrestore.io.image_loader import load_image
from screenrestore.io.project_file import load_project, save_project, verify_project_source


def _write_test_image(path: Path) -> None:
    yy, xx = np.indices((72, 112))
    image = np.stack(
        (
            (xx * 2) % 256,
            (yy * 3) % 256,
            ((xx + yy) * 2) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(image, "RGB").save(path)


def test_project_roundtrip_uses_relative_unicode_source(tmp_path: Path) -> None:
    source = tmp_path / "输入屏幕.png"
    project_path = tmp_path / "项目.screenrestore.json"
    _write_test_image(source)
    document = load_image(source)
    pipeline = build_default_pipeline()
    pipeline.update_parameters("exposure", {**pipeline.state("exposure").params.to_dict(), "gamma": 1.3})

    saved = save_project(project_path, document, pipeline, PresetId.CUSTOM)
    loaded = load_project(saved, build_registry())

    raw = json.loads(saved.read_text(encoding="utf-8"))
    assert raw["format_version"] == 2
    operator_ids = {item["id"] for item in raw["pipeline"]["operators"]}
    assert {"lens_distortion", "mesh_warp"}.issubset(operator_ids)
    assert not Path(raw["source"]["path"]).is_absolute()
    assert loaded.source_path == source.resolve()
    assert loaded.pipeline.state("exposure").params.to_dict()["gamma"] == 1.3
    assert verify_project_source(loaded, document) == []


def test_exporter_writes_unicode_png_and_refuses_overwrite(tmp_path: Path) -> None:
    image = np.full((30, 40, 3), (20, 80, 160), np.uint8)
    output = tmp_path / "恢复结果.png"
    export_image(image, output, ExportOptions(ExportFormat.PNG))
    with pytest.raises(ImageExportError, match="文件已存在"):
        export_image(image, output, ExportOptions(ExportFormat.PNG))
    with Image.open(output) as restored:
        assert restored.size == (40, 30)
        assert restored.mode == "RGB"


def test_cli_smoke_uses_shared_pipeline_and_json_diagnostics(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "命令行输入.png"
    output = tmp_path / "命令行输出.webp"
    _write_test_image(source)

    exit_code = main(
        [
            str(source),
            "--output",
            str(output),
            "--preset",
            "cinema",
            "--json-diagnostics",
        ]
    )

    diagnostics = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert diagnostics["status"] == "ok"
    assert diagnostics["backend"] == "CPU/OpenCV"
    assert output.is_file()
