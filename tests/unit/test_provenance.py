"""Archive / Enhanced 像素来源标签测试。"""

from __future__ import annotations

import numpy as np

from screenrestore.provenance import PixelOrigin, ProvenanceMap


def test_multiframe_masks_keep_recovered_and_unresolved_distinct() -> None:
    recovered = np.zeros((20, 30), dtype=bool)
    unresolved = np.zeros((20, 30), dtype=bool)
    recovered[2:6, 3:9] = True
    unresolved[10:15, 12:18] = True

    provenance = ProvenanceMap.from_fusion_masks((20, 30, 3), recovered, unresolved)
    summary = provenance.summary()

    assert summary["recovered_observation"] == 24 / 600
    assert summary["unresolved"] == 30 / 600
    assert np.all(provenance.labels[recovered] == int(PixelOrigin.RECOVERED_OBSERVATION))
    assert np.all(provenance.labels[unresolved] == int(PixelOrigin.UNRESOLVED))


def test_provenance_map_copies_inputs() -> None:
    labels = np.zeros((8, 9), dtype=np.uint8)
    provenance = ProvenanceMap(labels)
    labels[:] = int(PixelOrigin.GENERATED)
    assert np.all(provenance.labels == int(PixelOrigin.OBSERVED))
