#!/usr/bin/env bash
# ScreenRestore P3 分阶段训练、评估与报告唯一入口。
#
# 使用范例：
#   cd /Users/caozichen/Image_manage
#   source .venv/bin/activate
#   export SCREENRESTORE_DATA_ROOT=/Users/caozichen/screenrestore-data
#   export SCREENRESTORE_RUN_ROOT=/Users/caozichen/screenrestore-runs
#   export SCREENRESTORE_RUN_NAME="p3-$(date +%Y%m%d-%H%M%S)"
#   export P3_DEVICE=mps
#   bash scripts/train_p3.sh preflight
#   bash scripts/train_p3.sh smoke
#
# 该脚本不会下载数据，不覆盖已有 checkpoint/ONNX/evaluation，也不会启动未显式请求的阶段。

set -euo pipefail

STAGE="${1:-}"
VALID_STAGES=" preflight smoke geometry-b0 geometry-b1 geometry-b2 geometry-b3 geometry-b4 geometry-b5 geometry-b6 dewarp fidelity photometric demoire reflection superres router evaluate report "
if [[ "$VALID_STAGES" != *" $STAGE "* ]]; then
  echo "用法：bash scripts/train_p3.sh [preflight|smoke|geometry-b0|geometry-b1|geometry-b2|geometry-b3|geometry-b4|geometry-b5|geometry-b6|dewarp|fidelity|photometric|demoire|reflection|superres|router|evaluate|report]" >&2
  exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SCREENRESTORE_DATA_ROOT:-/Users/caozichen/screenrestore-data}"
RUN_ROOT="${SCREENRESTORE_RUN_ROOT:-/Users/caozichen/screenrestore-runs}"
RUN_NAME="${SCREENRESTORE_RUN_NAME:-}"
DEVICE="${P3_DEVICE:-mps}"
SEED="${P3_SEED:-20260830}"
B0_ROOT="/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b"
B0_CHECKPOINT="$B0_ROOT/best.pt"
B0_ONNX="$B0_ROOT/quadlocator-s.onnx"
GEOMETRY_MANIFEST="$DATA_ROOT/manifests/p2/stage-b.geometry.jsonl"
CALIBRATION_MANIFEST="$DATA_ROOT/manifests/p2/calibration.geometry.jsonl"
SMARTDOC_MANIFEST="$DATA_ROOT/manifests/smartdoc.geometry.jsonl"
SMARTDOC_DIRECTORY="$DATA_ROOT/geometry/smartdoc/frames"
TRAIN_HR="$DATA_ROOT/superres/div2k/DIV2K_train_HR"
VALIDATION_HR="$DATA_ROOT/superres/div2k/DIV2K_valid_HR"

if [[ -z "$RUN_NAME" ]]; then
  echo "必须显式设置唯一 SCREENRESTORE_RUN_NAME，避免覆盖历史 run" >&2
  exit 1
fi
RUN_DIRECTORY="$RUN_ROOT/$RUN_NAME"

if [[ ! -x "$REPOSITORY_ROOT/.venv/bin/python" ]]; then
  echo "未找到项目虚拟环境：$REPOSITORY_ROOT/.venv" >&2
  exit 1
fi
source "$REPOSITORY_ROOT/.venv/bin/activate"
cd "$REPOSITORY_ROOT"
which python
python -c 'import sys; print(sys.executable)'
export XDG_STATE_HOME="${XDG_STATE_HOME:-/tmp/screenrestore-p3-state}"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "缺少必需输入：$1" >&2
    exit 1
  fi
}

fresh_directory() {
  if [[ -e "$1" ]]; then
    echo "拒绝覆盖已有阶段目录：$1" >&2
    exit 1
  fi
  mkdir -p "$1"
}

require_mps() {
  if [[ "$DEVICE" != "mps" ]]; then
    echo "正式 P3 训练必须设置 P3_DEVICE=mps，当前为 $DEVICE" >&2
    exit 1
  fi
  python - <<'PY'
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("Apple Silicon 正式训练要求 MPS；当前不可用")
print("MPS preflight=PASS")
PY
}

write_blocked() {
  local output="$1"
  local dataset="$2"
  local expected="$3"
  mkdir -p "$(dirname "$output")"
  python - "$output" "$dataset" "$expected" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
if output.exists():
    raise SystemExit(f"拒绝覆盖已有阻塞报告：{output}")
output.write_text(
    json.dumps(
        {
            "status": "BLOCKED",
            "missing_dataset": sys.argv[2],
            "expected_directory_or_manifest": sys.argv[3],
            "license_status": "需要人工审计数据许可、训练用途与再分发限制",
            "impact": "synthetic 训练继续；真实 paired 指标与泛化结论不可用",
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

preflight() {
  require_path "$B0_CHECKPOINT"
  require_path "$B0_ONNX"
  require_path "$GEOMETRY_MANIFEST"
  require_path "$CALIBRATION_MANIFEST"
  require_path "$SMARTDOC_MANIFEST"
  require_path "$SMARTDOC_DIRECTORY"
  require_path "$TRAIN_HR"
  require_path "$VALIDATION_HR"
  if [[ "$DEVICE" == "mps" ]]; then
    require_mps
  fi
  local data_kib free_kib output
  data_kib="$(du -sk "$DATA_ROOT" | awk '{print $1}')"
  free_kib="$(df -Pk "$DATA_ROOT" | awk 'NR == 2 {print $4}')"
  if (( data_kib > 30 * 1024 * 1024 )); then
    echo "数据根超过 30 GiB 硬上限：$data_kib KiB" >&2
    exit 1
  fi
  output="$RUN_DIRECTORY/preflight"
  fresh_directory "$output"
  python - "$output/preflight.json" "$DATA_ROOT" "$RUN_DIRECTORY" "$data_kib" "$free_kib" "$DEVICE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, data_root, run_directory, data_kib, free_kib, device = sys.argv[1:]
baseline = Path("/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b")
files = [baseline / "best.pt", baseline / "quadlocator-s.onnx"]
report = {
    "status": "PASS",
    "data_root": data_root,
    "run_directory": run_directory,
    "data_kib": int(data_kib),
    "hard_cap_kib": 30 * 1024 * 1024,
    "free_kib": int(free_kib),
    "device": device,
    "baseline": {
        path.name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in files
    },
    "automatic_downloads": False,
    "augmentation_cache": False,
}
Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  echo "preflight PASS：data=${data_kib} KiB，free=${free_kib} KiB，run=$RUN_DIRECTORY" >&2
}

geometry_evaluate() {
  local model="$1"
  local output_directory="$2"
  local calibrator="${3:-}"
  fresh_directory "$output_directory"
  local arguments=(
    --data-directory "$SMARTDOC_DIRECTORY"
    --manifest "$SMARTDOC_MANIFEST"
    --dataset-root "$DATA_ROOT"
    --split test
    --quad-model "$model"
    --output "$output_directory/evaluation.json"
  )
  if [[ -n "$calibrator" ]]; then
    require_path "$calibrator"
    arguments+=(--correctness-calibrator "$calibrator")
  fi
  set +e
  python -m benchmarks.geometry_e2e.run "${arguments[@]}"
  local status=$?
  set -e
  echo "geometry evaluate exit=${status}（FAIL 报告会保留，不降低 gate）" >&2
  local overlay_arguments=(
    --manifest "$SMARTDOC_MANIFEST"
    --dataset-root "$DATA_ROOT"
    --split test
    --quad-model "$model"
    --output-directory "$output_directory/contact-sheets"
    --max-images "${P3_GEOMETRY_OVERLAY_SAMPLES:-80}"
  )
  if [[ -n "$calibrator" ]]; then
    overlay_arguments+=(--correctness-calibrator "$calibrator")
  fi
  python scripts/render_geometry_overlays.py "${overlay_arguments[@]}"
}

geometry_calibrate() {
  local model="$1"
  local output_directory="$2"
  fresh_directory "$output_directory"
  set +e
  python -m benchmarks.geometry_e2e.run \
    --data-directory "$DATA_ROOT" \
    --manifest "$CALIBRATION_MANIFEST" \
    --dataset-root "$DATA_ROOT" \
    --split validation \
    --quad-model "$model" \
    --output "$output_directory/evaluation.json"
  local evaluation_status=$?
  set -e
  echo "geometry calibration evaluation exit=${evaluation_status}（validation gate 可为 FAIL）" >&2
  python scripts/prepare_geometry_calibration.py \
    --evaluation "$output_directory/evaluation.json" \
    --output "$output_directory/features.jsonl" \
    --split validation
  local manifest_hash
  manifest_hash="$(python - "$CALIBRATION_MANIFEST" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
  python -m training.quadlocator.correctness_calibrator \
    --input "$output_directory/features.jsonl" \
    --output "$output_directory/correctness-calibrator.json" \
    --manifest-sha256 "$manifest_hash" \
    --minimum-precision 0.99
}

geometry_train() {
  local stage_name="$1"
  local epochs="$2"
  local train_samples="$3"
  local validation_samples="$4"
  local learning_rate="$5"
  local loss_profile="${6:-full}"
  local hard_sampling="${7:-1}"
  require_mps
  local output="$RUN_DIRECTORY/$stage_name"
  fresh_directory "$output"
  local arguments=(
    --manifest "$GEOMETRY_MANIFEST" \
    --dataset-root "$DATA_ROOT" \
    --output-directory "$output" \
    --init-checkpoint "$B0_CHECKPOINT" \
    --epochs "$epochs" \
    --learning-rate "$learning_rate" \
    --image-size "${P3_GEOMETRY_IMAGE_SIZE:-512}" \
    --batch-size "${P3_GEOMETRY_BATCH_SIZE:-4}" \
    --train-samples "$train_samples" \
    --validation-samples "$validation_samples" \
    --workers "${P3_WORKERS:-0}" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --loss-profile "$loss_profile"
  )
  if [[ "$hard_sampling" == "1" ]]; then
    arguments+=(--hard-sampling)
  fi
  python -m training.quadlocator.train "${arguments[@]}"
  python -m training.quadlocator.export_onnx \
    --checkpoint "$output/best.pt" \
    --output "$output/quadlocator-s.onnx"
}

specialist_train() {
  local task="$1"
  local stage_name="$2"
  local budget="$3"
  local epochs="$4"
  local manifest="${5:-}"
  require_mps
  local output="$RUN_DIRECTORY/$stage_name"
  local arguments=(
    --task "$task"
    --budget "$budget"
    --data-root "$DATA_ROOT"
    --hr-directory "$TRAIN_HR"
    --output-directory "$output"
    --epochs "$epochs"
    --batch-size "${P3_RESTORATION_BATCH_SIZE:-4}"
    --patch-size "${P3_RESTORATION_PATCH_SIZE:-128}"
    --train-samples "${P3_SPECIALIST_TRAIN_SAMPLES:-8000}"
    --validation-samples "${P3_SPECIALIST_VALIDATION_SAMPLES:-800}"
    --device "$DEVICE"
    --seed "$SEED"
  )
  if [[ -n "$manifest" ]]; then
    arguments+=(--manifest "$manifest")
  fi
  python -m training.p3.train_specialist "${arguments[@]}"
  if [[ "$task" != "dewarp" ]]; then
    python scripts/render_p3_restoration_contact_sheets.py \
      --task "$task" \
      --checkpoint "$output/best.pt" \
      --hr-directory "$VALIDATION_HR" \
      --output-directory "$output/contact-sheets" \
      --samples "${P3_CONTACT_SHEET_SAMPLES:-24}" \
      --patch-size "${P3_RESTORATION_PATCH_SIZE:-128}" \
      --device "$DEVICE" \
      --seed "$SEED"
  fi
}

case "$STAGE" in
  preflight)
    preflight
    ;;
  smoke)
    output="$RUN_DIRECTORY/smoke-$DEVICE"
    fresh_directory "$output"
    python -m training.p3.smoke \
      --output-directory "$output" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --baseline-onnx "$B0_ONNX"
    ;;
  geometry-b0)
    geometry_evaluate "$B0_ONNX" "$RUN_DIRECTORY/geometry-b0"
    ;;
  geometry-b1)
    geometry_train geometry-b1 2 "${P3_ABLATION_TRAIN_SAMPLES:-5000}" "${P3_ABLATION_VALIDATION_SAMPLES:-1000}" 1e-4 boundary 0
    ;;
  geometry-b2)
    geometry_calibrate "$B0_ONNX" "$RUN_DIRECTORY/geometry-b2"
    ;;
  geometry-b3)
    geometry_train geometry-b3 3 "${P3_ABLATION_TRAIN_SAMPLES:-5000}" "${P3_ABLATION_VALIDATION_SAMPLES:-1000}" 1e-4 tail 1
    ;;
  geometry-b4)
    geometry_evaluate \
      "$B0_ONNX" \
      "$RUN_DIRECTORY/geometry-b4" \
      "$RUN_DIRECTORY/geometry-b2/correctness-calibrator.json"
    ;;
  geometry-b5)
    geometry_train geometry-b5 "${P3_B5_EPOCHS:-18}" 0 0 "${P3_B5_LEARNING_RATE:-2e-4}" full 1
    geometry_calibrate \
      "$RUN_DIRECTORY/geometry-b5/quadlocator-s.onnx" \
      "$RUN_DIRECTORY/geometry-b5/calibration"
    ;;
  geometry-b6)
    if [[ "${P3_ENABLE_B6:-0}" != "1" ]]; then
      echo "B6 未启用：只有 B5 量化地板证据成立后才设置 P3_ENABLE_B6=1" >&2
      exit 3
    fi
    echo "B6 当前 BLOCKED：本轮未运行 B5，尚无量化地板证据，不得无依据启动 offset head 训练" >&2
    exit 3
    ;;
  dewarp)
    specialist_train dewarp dewarp ABLATION "${P3_DEWARP_EPOCHS:-6}"
    ;;
  fidelity)
    require_mps
    output="$RUN_DIRECTORY/fidelity"
    if [[ -e "$output" ]]; then echo "拒绝覆盖已有阶段目录：$output" >&2; exit 1; fi
    python -m training.restoration.train \
      --train-hr-directory "$TRAIN_HR" \
      --validation-hr-directory "$VALIDATION_HR" \
      --output-directory "$output" \
      --architecture fidelity_v2 \
      --channels 48 \
      --max-delta 0.08 \
      --preserve-photometric-nuisance \
      --epochs "${P3_FIDELITY_EPOCHS:-30}" \
      --batch-size "${P3_FIDELITY_BATCH_SIZE:-4}" \
      --patch-size "${P3_FIDELITY_PATCH_SIZE:-192}" \
      --device "$DEVICE" \
      --seed "$SEED"
    python -m training.restoration.evaluate \
      --checkpoint "$output/best.pt" \
      --hr-directory "$VALIDATION_HR" \
      --output "$output/evaluation.json" \
      --samples "${P3_FIDELITY_EVALUATION_SAMPLES:-100}" \
      --device "$DEVICE" \
      --seed "$SEED"
    python -m training.restoration.evaluate_slices \
      --checkpoint "$output/best.pt" \
      --hr-directory "$VALIDATION_HR" \
      --output "$output/slices.json" \
      --samples "${P3_FIDELITY_SLICE_SAMPLES:-100}" \
      --device "$DEVICE" \
      --seed "$SEED"
    python scripts/render_p3_restoration_contact_sheets.py \
      --task fidelity \
      --checkpoint "$output/best.pt" \
      --hr-directory "$VALIDATION_HR" \
      --output-directory "$output/contact-sheets" \
      --samples "${P3_CONTACT_SHEET_SAMPLES:-24}" \
      --patch-size "${P3_FIDELITY_PATCH_SIZE:-192}" \
      --device "$DEVICE" \
      --seed "$SEED"
    ;;
  photometric)
    specialist_train photometric photometric FULL "${P3_PHOTOMETRIC_EPOCHS:-24}"
    ;;
  demoire)
    manifest="${P3_DEMOIRE_MANIFEST:-}"
    if [[ -n "$manifest" && -f "$manifest" ]]; then
      python scripts/audit_restoration_manifest.py --data-root "$DATA_ROOT" --manifest "$manifest"
      specialist_train demoire demoire FULL "${P3_DEMOIRE_EPOCHS:-20}" "$manifest"
    else
      specialist_train demoire demoire-synthetic ABLATION "${P3_DEMOIRE_SYNTHETIC_EPOCHS:-10}"
      write_blocked "$RUN_DIRECTORY/demoire-real-blocked.json" "FHDMi/真实 paired 去摩尔纹" "$DATA_ROOT/demoire/fhdmi 或 P3_DEMOIRE_MANIFEST"
    fi
    ;;
  reflection)
    manifest="${P3_REFLECTION_MANIFEST:-}"
    if [[ -n "$manifest" && -f "$manifest" ]]; then
      python scripts/audit_restoration_manifest.py --data-root "$DATA_ROOT" --manifest "$manifest" --allow-private
      specialist_train reflection reflection FULL "${P3_REFLECTION_EPOCHS:-20}" "$manifest"
    else
      specialist_train reflection reflection-synthetic ABLATION "${P3_REFLECTION_SYNTHETIC_EPOCHS:-10}"
      write_blocked "$RUN_DIRECTORY/reflection-real-blocked.json" "真实 paired reflection" "$DATA_ROOT/reflection/paired 或 P3_REFLECTION_MANIFEST"
    fi
    ;;
  superres)
    output="$RUN_DIRECTORY/superres"
    fresh_directory "$output"
    sr_arguments=(
      --data-root "$DATA_ROOT"
      --output "$output/evaluation.json"
      --p1-run /Users/caozichen/screenrestore-runs/p1-full-20260828-133513
    )
    if [[ -n "${P3_PIPELINE_SR_CHECKPOINT:-}" ]]; then
      sr_arguments+=(--p3-checkpoint "$P3_PIPELINE_SR_CHECKPOINT")
    fi
    python scripts/evaluate_p3_superres.py "${sr_arguments[@]}"
    ;;
  router)
    specialist_train router router FULL "${P3_ROUTER_EPOCHS:-12}"
    ;;
  evaluate)
    model="$B0_ONNX"
    calibrator="$RUN_DIRECTORY/geometry-b2/correctness-calibrator.json"
    if [[ -f "$RUN_DIRECTORY/geometry-b5/quadlocator-s.onnx" ]]; then
      model="$RUN_DIRECTORY/geometry-b5/quadlocator-s.onnx"
      calibrator="$RUN_DIRECTORY/geometry-b5/calibration/correctness-calibrator.json"
    fi
    geometry_evaluate "$model" "$RUN_DIRECTORY/evaluate" "$calibrator"
    ;;
  report)
    python scripts/generate_p3_report.py \
      --run-directory "$RUN_DIRECTORY" \
      --docs-directory "$REPOSITORY_ROOT/docs"
    ;;
esac

echo "P3 $STAGE 完成：$RUN_DIRECTORY" >&2
