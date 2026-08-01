"""协作式取消测试。"""

from __future__ import annotations

import numpy as np
import pytest

from screenrestore.core.cancellation import CancellationToken, ProcessingCancelled
from screenrestore.core.operator import ProcessingContext
from screenrestore.core.presets import build_default_pipeline


def test_cancelled_pipeline_stops_before_first_operator() -> None:
    token = CancellationToken()
    token.cancel()
    context = ProcessingContext(cancellation=token)
    with pytest.raises(ProcessingCancelled):
        build_default_pipeline().process(np.zeros((20, 20, 3), np.uint8), context)

