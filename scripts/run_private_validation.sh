#!/usr/bin/env bash
# ScreenRestore 私人开发验证复现入口。
#
# 使用范例：
#   source .venv/bin/activate
#   which python
#   PRIVATE_VALIDATION_RUN=b0-20260912 bash scripts/run_private_validation.sh prepare
#   PRIVATE_VALIDATION_RUN=b0-20260912 bash scripts/run_private_validation.sh freeze
#   PRIVATE_VALIDATION_RUN=b0-20260912 bash scripts/run_private_validation.sh label
#   PRIVATE_VALIDATION_RUN=b0-20260912 bash scripts/run_private_validation.sh score
#   PRIVATE_VALIDATION_RUN=b0-20260912 bash scripts/run_private_validation.sh review
#   PRIVATE_VALIDATION_RUN=b0-20260912 bash scripts/run_private_validation.sh restore
#   PRIVATE_VALIDATION_RUN=<新模型>-private-dev \
#     PRIVATE_VALIDATION_INDEX=output/private-validation/b0-20260912/prepare/dataset-index.json \
#     bash scripts/run_private_validation.sh restore

set -euo pipefail

stage="${1:?需要指定 prepare/freeze/label/score/review/restore}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
data_root="${SCREENRESTORE_DATA_ROOT:-/Users/caozichen/screenrestore-data}"
image_directory="${PRIVATE_VALIDATION_IMAGES:-$data_root/private-set-extract/private_set}"
run_name="${PRIVATE_VALIDATION_RUN:-b0-20260912}"
output_root="${PRIVATE_VALIDATION_OUTPUT:-$repo/output/private-validation/$run_name}"
annotations="${PRIVATE_VALIDATION_ANNOTATIONS:-$data_root/private-validation/annotations.jsonl}"
quad_model="${PRIVATE_VALIDATION_QUAD_MODEL:-/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/quadlocator-s.onnx}"
calibrator="${PRIVATE_VALIDATION_CALIBRATOR:-}"
classic_agreement="${PRIVATE_VALIDATION_CLASSIC_AGREEMENT:-0}"
# 新模型复用同一批私人照片时，必须复用最初冻结的数据索引。这样既避免重复 prepare，
# 也让 prediction_sha256 与同一 dataset_sha256 保持可追溯绑定。
index="${PRIVATE_VALIDATION_INDEX:-$output_root/prepare/dataset-index.json}"
predictions="$output_root/frozen-predictions/predictions.json"

if [[ -z "${VIRTUAL_ENV:-}" || "$VIRTUAL_ENV" != "$repo/.venv" ]]; then
  echo "请先执行：source $repo/.venv/bin/activate" >&2
  exit 2
fi
which python

case "$stage" in
  prepare)
    python "$repo/scripts/private_development_validation.py" prepare \
      --data-root "$data_root" \
      --image-directory "$image_directory" \
      --output-directory "$output_root/prepare" \
      --annotations "$annotations"
    ;;
  freeze)
    freeze_arguments=(
      freeze
      --data-root "$data_root"
      --index "$index"
      --quad-model "$quad_model"
      --output-directory "$output_root/frozen-predictions"
    )
    if [[ -n "$calibrator" ]]; then
      freeze_arguments+=(--calibrator "$calibrator")
    fi
    if [[ "$classic_agreement" == "1" ]]; then
      freeze_arguments+=(--classic-agreement-snap)
    fi
    python "$repo/scripts/private_development_validation.py" "${freeze_arguments[@]}"
    ;;
  label)
    python "$repo/scripts/private_development_validation.py" label \
      --data-root "$data_root" \
      --index "$index" \
      --annotations "$annotations"
    ;;
  score)
    python "$repo/scripts/private_development_validation.py" score \
      --index "$index" \
      --predictions "$predictions" \
      --annotations "$annotations" \
      --output-directory "$output_root/score"
    ;;
  review)
    python "$repo/scripts/private_development_validation.py" review \
      --data-root "$data_root" \
      --index "$index" \
      --predictions "$predictions" \
      --annotations "$annotations" \
      --output-directory "$output_root/annotation-review"
    ;;
  restore)
    python "$repo/scripts/private_development_validation.py" restore \
      --data-root "$data_root" \
      --index "$index" \
      --predictions "$predictions" \
      --annotations "$annotations" \
      --output-directory "$output_root/restored"
    ;;
  *)
    echo "未知阶段：$stage" >&2
    exit 2
    ;;
esac
