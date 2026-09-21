#!/usr/bin/env bash
# 用途：复现 P4-G3.6 coordinate trajectory、validation-only 冻结、acceptance audit 与单次 test。
# 使用示例：
#   source .venv/bin/activate
#   which python
#   P4_G36_RUN_DIRECTORY=/Users/me/runs/p4-g36 bash scripts/run_p4_g36.sh all
#
# 可选环境变量：SCREENRESTORE_DATA_ROOT、SCREENRESTORE_RUN_ROOT、P4_G36_RUN_DIRECTORY、P4_DEVICE。
# all 严格按 train → select → acceptance → test 执行；任何阶段已有输出都会拒绝覆盖。

set -euo pipefail

stage="${1:-}"
if [[ "$stage" != "train" && "$stage" != "select" && "$stage" != "acceptance" && "$stage" != "test" && "$stage" != "all" ]]; then
  echo "用法：bash scripts/run_p4_g36.sh [train|select|acceptance|test|all]" >&2
  exit 2
fi

project_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${SCREENRESTORE_DATA_ROOT:-/Users/caozichen/screenrestore-data}"
run_root="${SCREENRESTORE_RUN_ROOT:-/Users/caozichen/screenrestore-runs}"
run_directory="${P4_G36_RUN_DIRECTORY:-$run_root/p4-g36-geometry-freeze}"
device="${P4_DEVICE:-mps}"
python_bin="$project_directory/.venv/bin/python"
source "$project_directory/.venv/bin/activate"
which python
b0_checkpoint="$run_root/p2-geometry-w1-20260829-110658/stage-b/best.pt"
trajectory_directory="$run_directory/coordinate-trajectory"
selection_directory="$run_directory/trajectory-selection"
acceptance_directory="$run_directory/acceptance-validation"
test_directory="$run_directory/frozen-smartdoc-test"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "缺少文件：$1" >&2
    exit 2
  fi
}

require_absent() {
  if [[ -e "$1" ]]; then
    echo "拒绝覆盖已有输出：$1" >&2
    exit 2
  fi
}

preflight() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "缺少 jq：selection 在无合格 challenger 时需要读取冻结状态。" >&2
    exit 2
  fi
  require_file "$python_bin"
  require_file "$b0_checkpoint"
  require_file "$data_root/manifests/p2-public/stage-b.geometry.jsonl"
  require_file "$data_root/manifests/p2-public/calibration-public.geometry.jsonl"
  require_file "$data_root/manifests/smartdoc.geometry.jsonl"
  "$python_bin" - "$data_root/manifests/p2-public/stage-b.geometry.jsonl" \
    "$data_root/manifests/p2-public/calibration-public.geometry.jsonl" <<'PY'
import sys
from pathlib import Path

from training.quadlocator.train import _assert_public_training_manifest

for manifest in sys.argv[1:]:
    _assert_public_training_manifest(Path(manifest))
print("公开训练与校准清单检查=PASS")
PY
  local data_kib
  data_kib="$(du -sk "$data_root" | awk '{print $1}')"
  if (( data_kib > 30 * 1024 * 1024 )); then
    echo "数据根超过 30 GiB：$data_kib KiB" >&2
    exit 2
  fi
  "$python_bin" -c 'import sys; print(sys.executable)'
}

run_train() {
  require_absent "$trajectory_directory"
  mkdir -p "$run_directory"
  "$python_bin" -m training.quadlocator.train \
    --manifest "$data_root/manifests/p2-public/stage-b.geometry.jsonl" \
    --dataset-root "$data_root" \
    --output-directory "$trajectory_directory" \
    --init-checkpoint "$b0_checkpoint" \
    --device "$device" --image-size 256 --batch-size 16 \
    --epochs 16 --scheduler-t-max 16 \
    --checkpoint-epochs 1,2,4,8,12,14,16 \
    --train-samples 20000 --validation-samples 1000 \
    --learning-rate 1e-5 --seed 20260902 --workers 0 \
    --loss-profile content_coordinate_only --trainable-scope content_head \
    --train-augmentation full --evaluate-init \
    --early-stopping-criterion geometry --early-stopping-patience 3 \
    --best-geometry-nce-p95-ratio 1.05 --best-geometry-iou-p05-ratio 0.9 \
    --best-geometry-nce-median-tolerance 0.002 \
    --best-geometry-iou-median-tolerance 0.01 \
    --geometry-collapse-patience 2 \
    --geometry-collapse-nce-p95-ratio 1.2 \
    --geometry-collapse-iou-p05-ratio 0.8
}

run_select() {
  require_file "$trajectory_directory/checkpoints/epoch-001.pt"
  require_file "$trajectory_directory/checkpoints/epoch-016.pt"
  require_absent "$selection_directory"
  local checkpoints=("$trajectory_directory"/checkpoints/epoch-*.pt)
  if "$python_bin" scripts/evaluate_p4_geometry_trajectory.py \
    --baseline "$b0_checkpoint" \
    --checkpoints "${checkpoints[@]}" \
    --internal-manifest "$data_root/manifests/p2-public/stage-b.geometry.jsonl" \
    --calibration-manifest "$data_root/manifests/p2-public/calibration-public.geometry.jsonl" \
    --smartdoc-manifest "$data_root/manifests/smartdoc.geometry.jsonl" \
    --dataset-root "$data_root" --internal-max-samples 1000 --internal-seed 20260902 \
    --evaluation-image-size 512 \
    --device "$device" --batch-size 8 --output-directory "$selection_directory"; then
    return 0
  else
    local status=$?
    if [[ "$status" -eq 3 ]]; then
      echo "没有 challenger 通过 eligibility；明确保留 incumbent B0 供 acceptance audit。" >&2
      return 0
    fi
    return "$status"
  fi
}

acceptance_checkpoint() {
  if [[ -f "$selection_directory/frozen-geometry.pt" ]]; then
    echo "$selection_directory/frozen-geometry.pt"
    return 0
  fi
  require_file "$selection_directory/trajectory-summary.json"
  local selection_status
  selection_status="$(jq -r '.selection.status' "$selection_directory/trajectory-summary.json")"
  if [[ "$selection_status" != "NO_ELIGIBLE_CHECKPOINT" ]]; then
    echo "selection 没有 frozen checkpoint，且状态不是 NO_ELIGIBLE_CHECKPOINT" >&2
    return 2
  fi
  echo "$b0_checkpoint"
}

run_acceptance() {
  local checkpoint
  checkpoint="$(acceptance_checkpoint)"
  require_file "$checkpoint"
  require_absent "$acceptance_directory"
  "$python_bin" scripts/audit_p4_acceptance.py --mode validation \
    --checkpoint "$checkpoint" \
    --manifest "$data_root/manifests/p2-public/calibration-public.geometry.jsonl" \
    --dataset-root "$data_root" --device "$device" --batch-size 8 \
    --output-directory "$acceptance_directory"
}

run_test() {
  local checkpoint
  checkpoint="$(acceptance_checkpoint)"
  require_file "$checkpoint"
  require_file "$acceptance_directory/correctness-calibrator.json"
  require_absent "$test_directory"
  "$python_bin" scripts/audit_p4_acceptance.py --mode test \
    --checkpoint "$checkpoint" \
    --manifest "$data_root/manifests/smartdoc.geometry.jsonl" \
    --dataset-root "$data_root" \
    --calibrator "$acceptance_directory/correctness-calibrator.json" \
    --device "$device" --batch-size 8 --output-directory "$test_directory"
}

cd "$project_directory"
preflight
case "$stage" in
  train) run_train ;;
  select) run_select ;;
  acceptance) run_acceptance ;;
  test) run_test ;;
  all)
    run_train
    run_select
    run_acceptance
    run_test
    ;;
esac
