# P3 结果

run：`/Users/caozichen/screenrestore-runs/p3-full-20260830-171326`

本文件只汇总实际存在的产物。未执行的正式训练保持 PENDING，外部数据缺失保持 BLOCKED。

## 阶段状态

| stage | status |
|---|---|
| preflight | COMPLETED |
| smoke-cpu | PENDING |
| smoke-mps | COMPLETED |
| geometry-b0 | COMPLETED |
| geometry-b1 | COMPLETED |
| geometry-b2 | COMPLETED |
| geometry-b3 | COMPLETED |
| geometry-b4 | COMPLETED |
| geometry-b5 | COMPLETED |
| geometry-b6 | PENDING |
| dewarp | COMPLETED |
| fidelity | COMPLETED |
| photometric | COMPLETED |
| demoire | BLOCKED |
| demoire-synthetic | COMPLETED |
| reflection | BLOCKED |
| reflection-synthetic | COMPLETED |
| superres | COMPLETED |
| router | COMPLETED |
| evaluate | COMPLETED |
| demoire-real | BLOCKED |
| reflection-real | BLOCKED |

## 训练运行

| stage | task | budget | device | parameters | wall seconds |
|---|---|---|---|---:|---:|
| demoire-synthetic | demoire | ABLATION | mps | 22360 | 8260.5096 |
| dewarp | dewarp | ABLATION | mps | 169346 | 3990.2945 |
| fidelity | None | None | mps | 1395233 | 7352.074 |
| geometry-b1 | None | None | mps | 99632 | 937.2953 |
| geometry-b3 | None | None | mps | 99632 | 1416.2513 |
| geometry-b5 | None | None | mps | 99632 | 75400.067 |
| photometric | photometric | FULL | mps | 105749 | 16354.288 |
| reflection-synthetic | reflection | ABLATION | mps | 26165 | 7567.306 |
| router | router | FULL | mps | 55166 | 7898.257 |

## 模型产物

| path | bytes | SHA-256 |
|---|---:|---|
| demoire-synthetic/best.pt | 103793 | `b574606ea9676e097f96607f12a2627595feb6031fa3031134ad32d5849cddf5` |
| dewarp/best.pt | 680995 | `7a48caa861e94741fd9dd0d9353c55beb4019e7e84ab76e33ea7c1acbb8d99bc` |
| fidelity/best.pt | 5635879 | `23b047cc1c58bda88fbb12a77d86e2c938b7ad4922e396b3925a9b4160e7f753` |
| geometry-b1/best.pt | 475421 | `9b51485e4494797a9f2027e68cc88391e5ff4b2f542fb99b9bf8e0cb4246db60` |
| geometry-b3/best.pt | 475421 | `23419ba8219407e888eb378190f44e66a076159e17ed1b9604ba6fd02470d5f7` |
| geometry-b5/best.pt | 475421 | `5d9c2f4ea1682964957965a334a9ddb6c099dd8ba8fa1befcdb66c5a270cc2a1` |
| photometric/best.pt | 426595 | `573ea7a526f34101ab67774748a96a855b403363f6d4096d927d5b259c6ae399` |
| reflection-synthetic/best.pt | 115129 | `0a2fd8c7341ebe250bd192756aff9060c67fce7ec71f62f475f74af11d7299ed` |
| router/best.pt | 224291 | `2c7a87b41d990d70b221318454a2cc3709a33ed04b704c67b2f0c6ec4bff6f21` |
| geometry-b1/quadlocator-s.onnx | 428860 | `a07db753eec6dc687e33daf3c0e0f8fa4bda3d59102213ad802c879b64d51fb2` |
| geometry-b3/quadlocator-s.onnx | 428860 | `be153fae2656c41a5087eec8c9c79effb4e325f1e4632b8459c39c6b4563a64d` |
| geometry-b5/quadlocator-s.onnx | 428860 | `5539e3aec28bd3435e264fb2a120e04b7eb52c07cbfb6fad60d3a7c5c92642d0` |
| smoke-mps/demoire.onnx | 104572 | `196fdf9fbbd830f489785fd1a46865640497d69fda8a714eca14eac39cbda08e` |
| smoke-mps/dewarp.onnx | 680464 | `3afa11ef27157533dc52ed07c23fe673d461314d4fdf392c8baac4ef1e41b343` |
| smoke-mps/fidelity.onnx | 5672120 | `049f0f7aacb89f987f4787ebdec3a4f2fe5b5a8a80e6b938afe7c45b231d53bb` |
| smoke-mps/photometric.onnx | 425622 | `12a5403b053a09203200763ee84f356f011195ce1beb2bd45f01c6dbd2932b87` |
| smoke-mps/reflection.onnx | 117730 | `263cda9225aa1430a1fdf723a9212cc4884ba7fa2a82266ff269f7e08cc26105` |
| smoke-mps/router.onnx | 225151 | `07401773a7b4db33c48279e082d8e6fe9e04d399424a1aee53e2d1c28ca506c0` |
| geometry-b2/correctness-calibrator.json | 2709 | `f94acf1c7213feab7db5af8c42b0d5a1682c2cebce394565549875a5f6d4b315` |
| geometry-b5/calibration/correctness-calibrator.json | 2710 | `7f191a8ea97bda2b559fb7a8ee0b00cf3e10471533e4f1556ae1155362eb9fba` |

## 数据与阻塞

```json
{
  "preflight": {
    "status": "PASS",
    "data_root": "/Users/caozichen/screenrestore-data",
    "run_directory": "/Users/caozichen/screenrestore-runs/p3-full-20260830-171326",
    "data_kib": 24085884,
    "hard_cap_kib": 31457280,
    "free_kib": 55935324,
    "device": "mps",
    "baseline": {
      "best.pt": {
        "path": "/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/best.pt",
        "sha256": "3344ad62709abf9e413a1cdebbdb82b1c0add0655ffdaeadcd9ddcc6bac86746"
      },
      "quadlocator-s.onnx": {
        "path": "/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/quadlocator-s.onnx",
        "sha256": "e73ce6912205c210fbbbdbd66ddefc0c9fba27cdd5c41853019badec98c1ab48"
      }
    },
    "automatic_downloads": false,
    "augmentation_cache": false
  },
  "blockers": [
    {
      "path": "demoire-real-blocked.json",
      "status": "BLOCKED",
      "missing_dataset": "FHDMi/真实 paired 去摩尔纹",
      "expected_directory_or_manifest": "/Users/caozichen/screenrestore-data/demoire/fhdmi 或 P3_DEMOIRE_MANIFEST",
      "license_status": "需要人工审计数据许可、训练用途与再分发限制",
      "impact": "synthetic 训练继续；真实 paired 指标与泛化结论不可用"
    },
    {
      "path": "reflection-real-blocked.json",
      "status": "BLOCKED",
      "missing_dataset": "真实 paired reflection",
      "expected_directory_or_manifest": "/Users/caozichen/screenrestore-data/reflection/paired 或 P3_REFLECTION_MANIFEST",
      "license_status": "需要人工审计数据许可、训练用途与再分发限制",
      "impact": "synthetic 训练继续；真实 paired 指标与泛化结论不可用"
    }
  ]
}
```

## 评估摘要

```json
[
  {
    "stage": "demoire-synthetic/contact-sheets",
    "kind": "p3_restoration_contact_sheets",
    "summary": null
  },
  {
    "stage": "demoire-synthetic",
    "kind": "p3_specialist_evaluation",
    "summary": {
      "psnr": 24.188939056396485,
      "ssim": 0.9108794575929642,
      "mae": 0.040258450494147835,
      "gradient_error": 0.026300213802605867,
      "identity_drift": 0.04461570802144706,
      "frequency_residual": 0.018449652304407208,
      "chroma_error": 0.016610396136529745
    }
  },
  {
    "stage": "dewarp",
    "kind": "p3_specialist_evaluation",
    "summary": {
      "loss": 0.003179282312048599,
      "grid_reconstruction": 0.0031792550571844913,
      "bending": 3.892444826192332e-07,
      "fold": 0.0,
      "straight_line": 3.892444826192332e-07,
      "identity": 0.0
    }
  },
  {
    "stage": "evaluate",
    "kind": "e2e_auto",
    "summary": {
      "status": "FAIL",
      "sample_count": 1679,
      "independent_group_count": 2,
      "accepted_count": 0,
      "independent_accepted_group_count": 0,
      "independent_accepted_group_failures": 0,
      "accepted_group_error_rate_95ci": {
        "estimate": 0.0,
        "lower": 0.0,
        "upper": 1.0,
        "one_sided_upper": 1.0,
        "confidence": 0.95
      },
      "accepted_group_error_rate_95_one_sided_upper": 1.0,
      "supports_99_percent_precision_at_95_percent_confidence": false,
      "accepted_precision": 0.0,
      "in_scope_coverage": 0.0,
      "wrong_layer_rate": 0.0,
      "corner_nce_p95": 1.0,
      "quad_iou_median": 0.0,
      "quad_iou_p05": 0.0,
      "gates": {
        "minimum_samples": false,
        "accepted_precision": false,
        "in_scope_coverage": false,
        "wrong_layer_rate": true,
        "nce_p95": false,
        "iou_median": false,
        "iou_p05": false
      },
      "thresholds": {
        "accepted_precision_min": 0.99,
        "in_scope_coverage_min": 0.9,
        "wrong_layer_rate_max": 0.005,
        "nce_p95_max": 0.01,
        "iou_median_min": 0.97,
        "iou_p05_min": 0.93,
        "minimum_samples": 100
      }
    }
  },
  {
    "stage": "fidelity/contact-sheets",
    "kind": "p3_restoration_contact_sheets",
    "summary": null
  },
  {
    "stage": "fidelity",
    "kind": "fidelity_restoration_evaluation",
    "summary": {
      "total": 0.03282138705253601,
      "reconstruction": 0.025953149423003197,
      "identity": 0.0015302237141161011,
      "edge": 0.042217730186306514,
      "loss": 0.03282138705253601,
      "psnr": 26.636049857506386,
      "ssim": 0.9460281087802007,
      "identity_mae": 0.0007821227168628516,
      "edge_correlation": 0.875494911120488,
      "color_error_255": 12.203198469602144
    }
  },
  {
    "stage": "geometry-b0",
    "kind": "e2e_auto",
    "summary": {
      "status": "FAIL",
      "sample_count": 1679,
      "independent_group_count": 2,
      "accepted_count": 0,
      "independent_accepted_group_count": 0,
      "independent_accepted_group_failures": 0,
      "accepted_group_error_rate_95ci": {
        "estimate": 0.0,
        "lower": 0.0,
        "upper": 1.0,
        "one_sided_upper": 1.0,
        "confidence": 0.95
      },
      "accepted_group_error_rate_95_one_sided_upper": 1.0,
      "supports_99_percent_precision_at_95_percent_confidence": false,
      "accepted_precision": 0.0,
      "in_scope_coverage": 0.0,
      "wrong_layer_rate": 0.0,
      "corner_nce_p95": 1.0,
      "quad_iou_median": 0.0,
      "quad_iou_p05": 0.0,
      "gates": {
        "minimum_samples": false,
        "accepted_precision": false,
        "in_scope_coverage": false,
        "wrong_layer_rate": true,
        "nce_p95": false,
        "iou_median": false,
        "iou_p05": false
      },
      "thresholds": {
        "accepted_precision_min": 0.99,
        "in_scope_coverage_min": 0.9,
        "wrong_layer_rate_max": 0.005,
        "nce_p95_max": 0.01,
        "iou_median_min": 0.97,
        "iou_p05_min": 0.93,
        "minimum_samples": 100
      }
    }
  },
  {
    "stage": "geometry-b2",
    "kind": "e2e_auto",
    "summary": {
      "status": "FAIL",
      "sample_count": 5624,
      "independent_group_count": 2411,
      "accepted_count": 0,
      "independent_accepted_group_count": 0,
      "independent_accepted_group_failures": 0,
      "accepted_group_error_rate_95ci": {
        "estimate": 0.0,
        "lower": 0.0,
        "upper": 1.0,
        "one_sided_upper": 1.0,
        "confidence": 0.95
      },
      "accepted_group_error_rate_95_one_sided_upper": 1.0,
      "supports_99_percent_precision_at_95_percent_confidence": false,
      "accepted_precision": 0.0,
      "in_scope_coverage": 0.0,
      "wrong_layer_rate": 0.0,
      "corner_nce_p95": 1.0,
      "quad_iou_median": 0.0,
      "quad_iou_p05": 0.0,
      "gates": {
        "minimum_samples": true,
        "accepted_precision": false,
        "in_scope_coverage": false,
        "wrong_layer_rate": true,
        "nce_p95": false,
        "iou_median": false,
        "iou_p05": false
      },
      "thresholds": {
        "accepted_precision_min": 0.99,
        "in_scope_coverage_min": 0.9,
        "wrong_layer_rate_max": 0.005,
        "nce_p95_max": 0.01,
        "iou_median_min": 0.97,
        "iou_p05_min": 0.93,
        "minimum_samples": 100
      }
    }
  },
  {
    "stage": "geometry-b4",
    "kind": "e2e_auto",
    "summary": {
      "status": "FAIL",
      "sample_count": 1679,
      "independent_group_count": 2,
      "accepted_count": 0,
      "independent_accepted_group_count": 0,
      "independent_accepted_group_failures": 0,
      "accepted_group_error_rate_95ci": {
        "estimate": 0.0,
        "lower": 0.0,
        "upper": 1.0,
        "one_sided_upper": 1.0,
        "confidence": 0.95
      },
      "accepted_group_error_rate_95_one_sided_upper": 1.0,
      "supports_99_percent_precision_at_95_percent_confidence": false,
      "accepted_precision": 0.0,
      "in_scope_coverage": 0.0,
      "wrong_layer_rate": 0.0,
      "corner_nce_p95": 1.0,
      "quad_iou_median": 0.0,
      "quad_iou_p05": 0.0,
      "gates": {
        "minimum_samples": false,
        "accepted_precision": false,
        "in_scope_coverage": false,
        "wrong_layer_rate": true,
        "nce_p95": false,
        "iou_median": false,
        "iou_p05": false
      },
      "thresholds": {
        "accepted_precision_min": 0.99,
        "in_scope_coverage_min": 0.9,
        "wrong_layer_rate_max": 0.005,
        "nce_p95_max": 0.01,
        "iou_median_min": 0.97,
        "iou_p05_min": 0.93,
        "minimum_samples": 100
      }
    }
  },
  {
    "stage": "geometry-b5/calibration",
    "kind": "e2e_auto",
    "summary": {
      "status": "FAIL",
      "sample_count": 5624,
      "independent_group_count": 2411,
      "accepted_count": 0,
      "independent_accepted_group_count": 0,
      "independent_accepted_group_failures": 0,
      "accepted_group_error_rate_95ci": {
        "estimate": 0.0,
        "lower": 0.0,
        "upper": 1.0,
        "one_sided_upper": 1.0,
        "confidence": 0.95
      },
      "accepted_group_error_rate_95_one_sided_upper": 1.0,
      "supports_99_percent_precision_at_95_percent_confidence": false,
      "accepted_precision": 0.0,
      "in_scope_coverage": 0.0,
      "wrong_layer_rate": 0.0,
      "corner_nce_p95": 1.0,
      "quad_iou_median": 0.0,
      "quad_iou_p05": 0.0,
      "gates": {
        "minimum_samples": true,
        "accepted_precision": false,
        "in_scope_coverage": false,
        "wrong_layer_rate": true,
        "nce_p95": false,
        "iou_median": false,
        "iou_p05": false
      },
      "thresholds": {
        "accepted_precision_min": 0.99,
        "in_scope_coverage_min": 0.9,
        "wrong_layer_rate_max": 0.005,
        "nce_p95_max": 0.01,
        "iou_median_min": 0.97,
        "iou_p05_min": 0.93,
        "minimum_samples": 100
      }
    }
  },
  {
    "stage": "photometric/contact-sheets",
    "kind": "p3_restoration_contact_sheets",
    "summary": null
  },
  {
    "stage": "photometric",
    "kind": "p3_specialist_evaluation",
    "summary": {
      "psnr": 26.034206218719483,
      "ssim": 0.9706113225221634,
      "mae": 0.04039562182035297,
      "gradient_error": 0.0031729831709526478,
      "identity_drift": 0.02245596640743315,
      "luminance_error": 0.035550496983341874,
      "chroma_error": 0.019205742506310342
    }
  },
  {
    "stage": "reflection-synthetic/contact-sheets",
    "kind": "p3_restoration_contact_sheets",
    "summary": null
  },
  {
    "stage": "reflection-synthetic",
    "kind": "p3_specialist_evaluation",
    "summary": {
      "psnr": 21.769840326309204,
      "ssim": 0.7879272070527077,
      "mae": 0.03970530839171261,
      "gradient_error": 0.002424967629776802,
      "identity_drift": 0.00010124962271220284,
      "outside_mask_modification": 8.293835144286277e-05,
      "unresolved_coverage": 0.03637123107910156,
      "reflection_residual": 0.023584289932623504
    }
  },
  {
    "stage": "router/contact-sheets",
    "kind": "p3_restoration_contact_sheets",
    "summary": null
  },
  {
    "stage": "router",
    "kind": "p3_specialist_evaluation",
    "summary": {
      "bce": 0.3207891143858433,
      "severity_mae": 0.10805103257298469,
      "micro_accuracy_at_0_5": 0.9010714221000672,
      "clean_false_positive_rate_at_0_5": 0.0,
      "artifact_false_negative_rate_at_0_5": 0.6925
    }
  },
  {
    "stage": "superres",
    "kind": "p3_superres_comparison",
    "summary": {
      "bicubic": {
        "psnr": 36.9139328956604,
        "ssim": 0.9626564174890518,
        "overshoot": 6.179529405869743e-06,
        "ringing": 0.0
      },
      "p1": {
        "psnr": 37.17829179763794,
        "ssim": 0.9675484478473664,
        "overshoot": 0.0,
        "ringing": -0.0014779510449443479
      }
    }
  }
]
```

## Release gate

```json
{
  "development": {
    "status": "FAIL",
    "sample_count": 1679,
    "independent_group_count": 2,
    "accepted_count": 0,
    "independent_accepted_group_count": 0,
    "independent_accepted_group_failures": 0,
    "accepted_group_error_rate_95ci": {
      "estimate": 0.0,
      "lower": 0.0,
      "upper": 1.0,
      "one_sided_upper": 1.0,
      "confidence": 0.95
    },
    "accepted_group_error_rate_95_one_sided_upper": 1.0,
    "supports_99_percent_precision_at_95_percent_confidence": false,
    "accepted_precision": 0.0,
    "in_scope_coverage": 0.0,
    "wrong_layer_rate": 0.0,
    "corner_nce_p95": 1.0,
    "quad_iou_median": 0.0,
    "quad_iou_p05": 0.0,
    "gates": {
      "minimum_samples": false,
      "accepted_precision": false,
      "in_scope_coverage": false,
      "wrong_layer_rate": true,
      "nce_p95": false,
      "iou_median": false,
      "iou_p05": false
    },
    "thresholds": {
      "accepted_precision_min": 0.95,
      "in_scope_coverage_min": 0.5,
      "wrong_layer_rate_max": 0.02,
      "nce_p95_max": 0.03,
      "iou_median_min": 0.9,
      "iou_p05_min": 0.85,
      "minimum_samples": 30
    }
  },
  "release": {
    "status": "FAIL",
    "sample_count": 1679,
    "independent_group_count": 2,
    "accepted_count": 0,
    "independent_accepted_group_count": 0,
    "independent_accepted_group_failures": 0,
    "accepted_group_error_rate_95ci": {
      "estimate": 0.0,
      "lower": 0.0,
      "upper": 1.0,
      "one_sided_upper": 1.0,
      "confidence": 0.95
    },
    "accepted_group_error_rate_95_one_sided_upper": 1.0,
    "supports_99_percent_precision_at_95_percent_confidence": false,
    "accepted_precision": 0.0,
    "in_scope_coverage": 0.0,
    "wrong_layer_rate": 0.0,
    "corner_nce_p95": 1.0,
    "quad_iou_median": 0.0,
    "quad_iou_p05": 0.0,
    "gates": {
      "minimum_samples": false,
      "accepted_precision": false,
      "in_scope_coverage": false,
      "wrong_layer_rate": true,
      "nce_p95": false,
      "iou_median": false,
      "iou_p05": false
    },
    "thresholds": {
      "accepted_precision_min": 0.99,
      "in_scope_coverage_min": 0.9,
      "wrong_layer_rate_max": 0.005,
      "nce_p95_max": 0.01,
      "iou_median_min": 0.97,
      "iou_p05_min": 0.93,
      "minimum_samples": 100
    }
  }
}
```

## 结论

release gate 以机器报告为准；FAIL 保持 FAIL，不调整阈值。
