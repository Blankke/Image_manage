#!/usr/bin/env bash
# 用途：在已修复同步增强的代码上，从 P2 B0 开始进行受保护的几何恢复微调。
#
# 使用范例：
#   source .venv/bin/activate
#   which python
#   P5_RUN_NAME=p5-geometry-aligned-20260912 bash scripts/train_p5_geometry_recovery.sh preflight
#   P5_RUN_NAME=p5-geometry-aligned-20260912 bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-geometry-aligned-20260912 bash scripts/train_p5_geometry_recovery.sh private
#   P5_RUN_NAME=p5-content-head-aligned-20260913 P5_LOSS_PROFILE=content_only \
#     P5_TRAINABLE_SCOPE=content_head P5_LEARNING_RATE=1e-4 \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-content-tail-recovery-20260913 \
#     P5_BASELINE_CHECKPOINT=/path/to/content-head/epoch-002.pt \
#     P5_LOSS_PROFILE=tail P5_TRAINABLE_SCOPE=content_head P5_LEARNING_RATE=2e-5 \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-decision-target-v4-20260913 \
#     P5_MANIFEST=/data/target-v4/manifest.jsonl P5_LOSS_PROFILE=decision_only \
#     P5_TRAINABLE_SCOPE=decision_heads P5_CHECKPOINT_KIND=product \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-local-decision-replay-v4-20260913 \
#     P5_MANIFEST=/data/manifests/p5/target-v4-replay.geometry.jsonl \
#     P5_LOSS_PROFILE=decision_correction P5_TRAINABLE_SCOPE=decision_residual_heads \
#     P5_CLASS_BALANCED_SAMPLING=1 P5_CHECKPOINT_KIND=product \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-content-confidence-v3-20260913 \
#     P5_LOSS_PROFILE=content_confidence P5_TRAINABLE_SCOPE=content_residual_head \
#     P5_CLASS_BALANCED_SAMPLING=1 P5_LEARNING_RATE=1e-5 \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-class-context-v4-20260913 \
#     P5_MANIFEST=/data/manifests/p5/target-v4-replay.geometry.jsonl \
#     P5_LOSS_PROFILE=screen_context_correction P5_TRAINABLE_SCOPE=class_context_branch \
#     P5_CLASS_BALANCED_SAMPLING=1 P5_LEARNING_RATE=3e-4 \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-wide-v5-stage-a P5_INITIALIZATION=scratch \
#     P5_WIDTH_MULTIPLIER=1.5 P5_MANIFEST=/data/manifests/p5/target-v5-replay.geometry.jsonl \
#     P5_LOSS_PROFILE=full P5_TRAINABLE_SCOPE=all \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p5-wide-w15-v5-long P5_EPOCHS=14 P5_SCHEDULER_T_MAX=14 \
#     P5_CHECKPOINT_EPOCHS=1,2,4,6,8,10,12,14 P5_TRAIN_SAMPLES=0 \
#     P5_VALIDATION_SAMPLES=0 bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p9-public-artwork P5_MANIFEST=/data/manifests/p9/replay.geometry.jsonl \
#     P5_VALIDATION_SAMPLES=0 P5_REQUIRE_VALIDATION_SOURCES=syngallery-reviewed=10 \
#     bash scripts/train_p5_geometry_recovery.sh train
#   P5_RUN_NAME=p11-public-pilot P5_TRAIN_SAMPLES=5000 P5_SAMPLES_PER_EPOCH=1200 \
#     P5_REQUIRE_TRAIN_SOURCES='smartdoc-photo-surface-composite=375,syngallery-reviewed=104' \
#     P5_REQUIRE_VALIDATION_SOURCES='smartdoc-photo-surface-composite=45,syngallery-reviewed=10' \
#     bash scripts/train_p5_geometry_recovery.sh train
#
# 私人集只在 checkpoint 和 ONNX 冻结后评分，不进入训练、阈值拟合或 checkpoint 选择。

set -euo pipefail

stage="${1:?需要指定 preflight/train/private}"
project_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${SCREENRESTORE_DATA_ROOT:-/Users/caozichen/screenrestore-data}"
run_root="${SCREENRESTORE_RUN_ROOT:-/Users/caozichen/screenrestore-runs}"
run_name="${P5_RUN_NAME:-p5-geometry-aligned-20260912}"
run_directory="$run_root/$run_name/geometry"
baseline_checkpoint="${P5_BASELINE_CHECKPOINT:-$run_root/p2-geometry-w1-20260829-110658/stage-b/best.pt}"
initialization="${P5_INITIALIZATION:-warm_start}"
manifest="${P5_MANIFEST:-$data_root/manifests/p13/scene-isolated-replay-20260918.geometry.jsonl}"
dataset_root="${P5_DATASET_ROOT:-$data_root}"
device="${P5_DEVICE:-mps}"
epochs="${P5_EPOCHS:-2}"
scheduler_t_max="${P5_SCHEDULER_T_MAX:-$epochs}"
checkpoint_epochs="${P5_CHECKPOINT_EPOCHS:-1,$epochs}"
train_samples="${P5_TRAIN_SAMPLES:-5000}"
samples_per_epoch="${P5_SAMPLES_PER_EPOCH:-0}"
validation_samples="${P5_VALIDATION_SAMPLES:-1000}"
required_validation_sources="${P5_REQUIRE_VALIDATION_SOURCES:-}"
required_train_sources="${P5_REQUIRE_TRAIN_SOURCES:-}"
batch_size="${P5_BATCH_SIZE:-4}"
learning_rate="${P5_LEARNING_RATE:-1e-5}"
width_multiplier="${P5_WIDTH_MULTIPLIER:-1.0}"
warm_start_minimum_ratio="${P5_MIN_WARM_START_PARAMETER_RATIO:-0.95}"
image_size="${P5_IMAGE_SIZE:-512}"
loss_profile="${P5_LOSS_PROFILE:-p2}"
trainable_scope="${P5_TRAINABLE_SCOPE:-all}"
checkpoint_kind="${P5_CHECKPOINT_KIND:-geometry}"
class_balanced_sampling="${P5_CLASS_BALANCED_SAMPLING:-0}"
workers="${P5_WORKERS:-0}"
seed="${P5_SEED:-20260912}"
early_stopping_patience="${P5_EARLY_STOPPING_PATIENCE:-0}"
geometry_collapse_patience="${P5_GEOMETRY_COLLAPSE_PATIENCE:-0}"
geometry_collapse_nce_p95_ratio="${P5_GEOMETRY_COLLAPSE_NCE_P95_RATIO:-1.35}"
geometry_collapse_iou_p05_ratio="${P5_GEOMETRY_COLLAPSE_IOU_P05_RATIO:-0.70}"
private_run="${P5_PRIVATE_RUN:-$run_name-private-dev}"
private_index="${P5_PRIVATE_INDEX:-$project_directory/output/private-validation/b0-20260912/prepare/dataset-index.json}"
private_annotations="${PRIVATE_VALIDATION_ANNOTATIONS:-$data_root/private-validation/annotations.jsonl}"
private_calibrator="${P5_PRIVATE_CALIBRATOR:-}"
private_output="$project_directory/output/private-validation/$private_run"

cd "$project_directory"
if [[ -z "${VIRTUAL_ENV:-}" || "$VIRTUAL_ENV" != "$project_directory/.venv" ]]; then
  echo "请先执行：source $project_directory/.venv/bin/activate" >&2
  exit 2
fi
which python
python -c 'import sys; print(sys.executable)'
export XDG_STATE_HOME="${XDG_STATE_HOME:-/tmp/screenrestore-p5-state}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "缺少必需文件：$1" >&2
    exit 1
  fi
}

case "$stage" in
  preflight)
    require_file "$manifest"
    if [[ "$initialization" == "warm_start" ]]; then
      require_file "$baseline_checkpoint"
    elif [[ "$initialization" != "scratch" ]]; then
      echo "P5_INITIALIZATION 必须为 warm_start 或 scratch" >&2
      exit 2
    fi
    python - "$device" "$baseline_checkpoint" "$manifest" "$initialization" <<'PY'
import hashlib
import sys
from pathlib import Path

from training.quadlocator.dataset import _read_manifest
from training.quadlocator.train import _device

device = _device(sys.argv[1])
checkpoint = Path(sys.argv[2]).resolve()
manifest = Path(sys.argv[3]).resolve()
initialization = sys.argv[4]
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if initialization == "warm_start" else None
records = _read_manifest(manifest)
splits = {str(row["split"]) for row in records}
if not {"train", "validation"} <= splits:
    raise SystemExit("训练清单缺少 train 或 validation")
print(
    f"device={device} initialization={initialization} checkpoint_sha256={digest} "
    f"samples={len(records)} splits={sorted(splits)}"
)
PY
    ;;
  train)
    require_file "$manifest"
    if [[ -e "$run_directory" ]]; then
      echo "训练目录已存在，拒绝覆盖：$run_directory" >&2
      exit 1
    fi
    if [[ "$checkpoint_kind" != "geometry" && "$checkpoint_kind" != "product" ]]; then
      echo "P5_CHECKPOINT_KIND 必须为 geometry 或 product" >&2
      exit 2
    fi
    training_arguments=(
      --manifest "$manifest"
      --dataset-root "$dataset_root"
      --output-directory "$run_directory"
    )
    if [[ "$initialization" == "warm_start" ]]; then
      require_file "$baseline_checkpoint"
      training_arguments+=(--init-checkpoint "$baseline_checkpoint")
      training_arguments+=(--min-warm-start-parameter-ratio "$warm_start_minimum_ratio")
    elif [[ "$initialization" == "scratch" ]]; then
      if [[ "$trainable_scope" != "all" ]]; then
        echo "scratch 初始化必须使用 P5_TRAINABLE_SCOPE=all" >&2
        exit 2
      fi
    else
      echo "P5_INITIALIZATION 必须为 warm_start 或 scratch" >&2
      exit 2
    fi
    if [[ "$class_balanced_sampling" == "1" ]]; then
      training_arguments+=(--class-balanced-sampling)
    elif [[ "$class_balanced_sampling" != "0" ]]; then
      echo "P5_CLASS_BALANCED_SAMPLING 必须为 0 或 1" >&2
      exit 2
    fi
    if [[ -n "$required_validation_sources" ]]; then
      IFS=',' read -r -a validation_source_pairs <<< "$required_validation_sources"
      for source_pair in "${validation_source_pairs[@]}"; do
        training_arguments+=(--require-validation-source "$source_pair")
      done
    fi
    if [[ -n "$required_train_sources" ]]; then
      IFS=',' read -r -a train_source_pairs <<< "$required_train_sources"
      for source_pair in "${train_source_pairs[@]}"; do
        training_arguments+=(--require-train-source "$source_pair")
      done
    fi
    training_arguments+=(
      --epochs "$epochs"
      --scheduler-t-max "$scheduler_t_max"
      --checkpoint-epochs "$checkpoint_epochs"
      --learning-rate "$learning_rate"
      --loss-profile "$loss_profile"
      --trainable-scope "$trainable_scope"
      --train-augmentation full
      --image-size "$image_size"
      --width-multiplier "$width_multiplier"
      --batch-size "$batch_size"
      --train-samples "$train_samples"
      --samples-per-epoch "$samples_per_epoch"
      --validation-samples "$validation_samples"
      --workers "$workers"
      --device "$device"
      --seed "$seed"
      --evaluate-init
      --early-stopping-patience "$early_stopping_patience"
      --early-stopping-criterion geometry
      --geometry-collapse-patience "$geometry_collapse_patience"
      --geometry-collapse-nce-p95-ratio "$geometry_collapse_nce_p95_ratio"
      --geometry-collapse-iou-p05-ratio "$geometry_collapse_iou_p05_ratio"
      --best-geometry-nce-p95-ratio 1.05
      --best-geometry-iou-p05-ratio 0.95
      --best-geometry-nce-median-tolerance 0.002
      --best-geometry-iou-median-tolerance 0.01
    )
    python -m training.quadlocator.train "${training_arguments[@]}"
    python -m training.quadlocator.export_onnx \
      --checkpoint "$run_directory/best_${checkpoint_kind}.pt" \
      --output "$run_directory/quadlocator-s.onnx"
    ;;
  private)
    require_file "$run_directory/quadlocator-s.onnx"
    require_file "$private_index"
    require_file "$private_annotations"
    if [[ -e "$private_output" ]]; then
      echo "私人验证输出已存在，拒绝覆盖：$private_output" >&2
      exit 1
    fi
    private_freeze_arguments=(
      freeze
      --data-root "$data_root"
      --index "$private_index"
      --quad-model "$run_directory/quadlocator-s.onnx"
      --output-directory "$private_output/frozen-predictions"
    )
    if [[ -n "$private_calibrator" ]]; then
      require_file "$private_calibrator"
      private_freeze_arguments+=(--calibrator "$private_calibrator")
    fi
    python scripts/private_development_validation.py "${private_freeze_arguments[@]}"
    set +e
    python scripts/private_development_validation.py score \
      --index "$private_index" \
      --predictions "$private_output/frozen-predictions/predictions.json" \
      --annotations "$private_annotations" \
      --output-directory "$private_output/score"
    score_status=$?
    set -e
    python scripts/private_development_validation.py review \
      --data-root "$data_root" \
      --index "$private_index" \
      --predictions "$private_output/frozen-predictions/predictions.json" \
      --annotations "$private_annotations" \
      --output-directory "$private_output/annotation-review"
    echo "私人开发验证 score exit=${score_status}；FAIL 报告仍为有效实验结果。" >&2
    ;;
  *)
    echo "未知阶段：$stage" >&2
    exit 2
    ;;
esac
