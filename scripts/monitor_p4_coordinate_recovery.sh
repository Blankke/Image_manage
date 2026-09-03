#!/usr/bin/env zsh
# 用途：两小时后（或指定延迟）检查 P4 coordinate-only 长训；异常中断时从 last.pt 启动剩余轮数的恢复训练。
# 使用示例：
#   scripts/monitor_p4_coordinate_recovery.sh \
#     --stage-directory /Users/caozichen/screenrestore-runs/.../content-coordinate-head \
#     --pid 12345 --delay-seconds 7200
#
# 说明：正常完成、geometry tail watchdog 停止和 early stopping 都会留下 history.json，
# 它们是有效实验终点，脚本不会重启。仅在训练进程消失且未留下完成记录时才恢复。

set -euo pipefail

if [[ $# -ne 6 || "$1" != "--stage-directory" || "$3" != "--pid" || "$5" != "--delay-seconds" ]]; then
  print -u2 "用法：$0 --stage-directory <目录> --pid <PID> --delay-seconds <秒数>"
  exit 2
fi

stage_directory="$2"
training_pid="$4"
delay_seconds="$6"
project_directory="${0:A:h:h}"
python_bin="$project_directory/.venv/bin/python"
status_file="$stage_directory/monitor-2h-status.json"

if [[ ! -x "$python_bin" ]]; then
  print -u2 "找不到项目虚拟环境解释器：$python_bin"
  exit 2
fi
if [[ ! -f "$stage_directory/run.json" ]]; then
  print -u2 "找不到训练元数据：$stage_directory/run.json"
  exit 2
fi

sleep "$delay_seconds"

if kill -0 "$training_pid" 2>/dev/null; then
  "$python_bin" - "$status_file" "$training_pid" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "status": "running",
    "pid": int(sys.argv[2]),
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  exit 0
fi

# Trainer 只有在完成训练循环（包含正常 early stop/watchdog）后才写 history.json。
if [[ -f "$stage_directory/history.json" ]]; then
  "$python_bin" - "$status_file" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "status": "completed_normally",
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  exit 0
fi

last_checkpoint="$stage_directory/last.pt"
if [[ ! -f "$last_checkpoint" ]]; then
  "$python_bin" - "$status_file" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "status": "interrupted_before_first_checkpoint",
    "action": "manual_diagnosis_required",
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  exit 1
fi

# 当前 trainer 的 checkpoint 未保存优化器状态，因此恢复会新建优化器；保留 remaining epochs，
# 并以 last.pt 作为模型 warm start，避免把已完成轮次重复跑一遍。
read -r manifest dataset_root device image_size batch_size learning_rate loss_profile trainable_scope train_augmentation train_samples validation_samples seed total_epochs early_patience early_criterion collapse_patience collapse_nce collapse_iou eligibility_nce eligibility_iou completed_epochs <<EOF
$($python_bin - "$stage_directory" <<'PY'
import json
import sys
from pathlib import Path

stage = Path(sys.argv[1])
run = json.loads((stage / "run.json").read_text(encoding="utf-8"))
history_path = stage / "history.json"
history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
watchdog = run["geometry_collapse_watchdog"]
eligibility = run["best_geometry_eligibility"]
values = [
    run["dataset_manifest"], run["dataset_root"], run["device"], run["image_size"],
    run["batch_size"], run["learning_rate"], run["loss_profile"], run["trainable_scope"],
    run["train_augmentation"], run["train_samples"], run["validation_samples"], run["seed"],
    run["epochs"], run["early_stopping_patience"], run["early_stopping_criterion"],
    watchdog["patience"], watchdog["nce_p95_ratio"], watchdog["iou_p05_ratio"],
    eligibility["nce_p95_ratio"], eligibility["iou_p05_ratio"], len(history),
]
print(" ".join(map(str, values)))
PY
)
EOF

remaining_epochs=$(( total_epochs - completed_epochs ))
if (( remaining_epochs < 1 )); then
  print -u2 "训练已完成全部轮次但未写入 history.json，保留现场等待人工检查。"
  exit 1
fi

recovery_directory="$stage_directory/recovery-after-interruption-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$recovery_directory"

"$python_bin" - "$status_file" "$recovery_directory" "$completed_epochs" "$remaining_epochs" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "status": "interrupted_restarting",
    "recovery_directory": sys.argv[2],
    "completed_epochs_before_interruption": int(sys.argv[3]),
    "remaining_epochs": int(sys.argv[4]),
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY

cd "$project_directory"
"$python_bin" -m training.quadlocator.train \
  --manifest "$manifest" \
  --dataset-root "$dataset_root" \
  --output-directory "$recovery_directory" \
  --init-checkpoint "$last_checkpoint" \
  --device "$device" \
  --image-size "$image_size" \
  --batch-size "$batch_size" \
  --epochs "$remaining_epochs" \
  --train-samples "$train_samples" \
  --validation-samples "$validation_samples" \
  --learning-rate "$learning_rate" \
  --seed "$seed" \
  --workers 0 \
  --loss-profile "$loss_profile" \
  --trainable-scope "$trainable_scope" \
  --train-augmentation "$train_augmentation" \
  --evaluate-init \
  --early-stopping-criterion "$early_criterion" \
  --early-stopping-patience "$early_patience" \
  --best-geometry-nce-p95-ratio "$eligibility_nce" \
  --best-geometry-iou-p05-ratio "$eligibility_iou" \
  --geometry-collapse-patience "$collapse_patience" \
  --geometry-collapse-nce-p95-ratio "$collapse_nce" \
  --geometry-collapse-iou-p05-ratio "$collapse_iou"
