"""轻量 pairwise 排序器应学会提高已知更优候选的分数。"""

import numpy as np
from training.quadlocator.train_classic_candidate_ranker import fit_pairwise_ranker


def test_pairwise_ranker_learns_positive_margin() -> None:
    pairs = np.asarray([[1.0, 0.0], [0.8, 0.1], [1.2, -0.1]], np.float64)
    validation = [
        {
            "items": [
                {"features": np.asarray([0.0, 0.0]), "iou": 0.2, "nce": 0.2},
                {"features": np.asarray([1.0, 0.0]), "iou": 0.96, "nce": 0.008},
            ]
        }
    ]
    mean = np.zeros(2, np.float64)
    scale = np.ones(2, np.float64)

    weights, selection = fit_pairwise_ranker(
        pairs,
        validation,
        mean,
        scale,
        steps=80,
        learning_rate=0.05,
        weight_decay=0.0,
    )

    assert weights[0] > 0
    assert selection["best_step"] > 0
