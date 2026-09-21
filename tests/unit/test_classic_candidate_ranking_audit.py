"""传统候选排序审核必须分开报告运行时 top-1 与 oracle 召回。"""

from scripts.audit_classic_candidate_ranking import balanced_group_limit, summarize_cases


def test_candidate_sample_limit_round_robins_groups() -> None:
    rows = [
        {"group_id": "a", "image": f"a-{index}"} for index in range(5)
    ] + [{"group_id": "b", "image": f"b-{index}"} for index in range(2)]

    selected = balanced_group_limit(rows, 4)

    assert [row["image"] for row in selected] == ["a-0", "b-0", "a-1", "b-1"]


def test_candidate_summary_does_not_treat_oracle_as_runtime_top1() -> None:
    cases = [
        {
            "group_id": "group-a",
            "runtime_top1": {"runtime_rank": 1, "quad_iou": 0.4, "corner_nce": 0.2},
            "reranked_top1": {"runtime_rank": 5, "quad_iou": 0.98, "corner_nce": 0.006},
            "model_agreement_top1": {"runtime_rank": 5, "quad_iou": 0.98, "corner_nce": 0.006},
            "oracle_best": {"runtime_rank": 5, "quad_iou": 0.98, "corner_nce": 0.006},
        },
        {
            "group_id": "group-b",
            "runtime_top1": {"runtime_rank": 1, "quad_iou": 0.94, "corner_nce": 0.009},
            "reranked_top1": {"runtime_rank": 1, "quad_iou": 0.94, "corner_nce": 0.009},
            "model_agreement_top1": {"runtime_rank": 1, "quad_iou": 0.94, "corner_nce": 0.009},
            "oracle_best": {"runtime_rank": 1, "quad_iou": 0.94, "corner_nce": 0.009},
        },
    ]

    summary = summarize_cases(cases)

    assert summary["top1_iou_0_9_count"] == 1
    assert summary["oracle_iou_0_9_count"] == 2
    assert summary["top1_strict_count"] == 1
    assert summary["reranked_iou_0_9_count"] == 2
    assert summary["reranked_strict_count"] == 2
    assert summary["model_agreement_iou_0_9_count"] == 2
    assert summary["model_agreement_strict_count"] == 2
    assert summary["oracle_strict_count"] == 2
    assert summary["oracle_best_runtime_rank_counts"] == {"1": 1, "5": 1}
