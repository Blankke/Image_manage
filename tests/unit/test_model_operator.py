"""可选模型算子与清单后端的流水线集成测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.operators.model_plugin import (
    ModelPluginOperator,
    ModelPluginParameters,
)


def test_model_operator_runs_external_manifest_and_blends(tmp_path: Path) -> None:
    manifest_path = tmp_path / "复制模型.json"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "copy",
                "name": "Copy",
                "type": "external_process",
                "executable": sys.executable,
                "arguments": [
                    "-c",
                    "from PIL import Image; import sys; Image.open(sys.argv[1]).save(sys.argv[2])",
                    "{input}",
                    "{output}",
                ],
            }
        ),
        encoding="utf-8",
    )
    source = np.full((19, 27, 3), (15, 90, 170), np.uint8)
    output = ModelPluginOperator().apply(
        source,
        ModelPluginParameters(manifest_path=str(manifest_path), strength=0.6),
        ProcessingContext(),
    )
    assert np.array_equal(output, source)
