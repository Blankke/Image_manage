#!/usr/bin/env bash
# 用途：逐阶段运行 P4-G3.7；无 all/test，不下载、不覆盖输出。
# 范例：P4_G37_RUN_DIRECTORY="$HOME/screenrestore-runs/p4-g37" bash scripts/run_p4_g37.sh preflight
set -euo pipefail
project_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_directory"
source .venv/bin/activate
which python
stage="${1:?需要指定 preflight/audit-data/baseline/sanity512/prepare-target/acceptance-dev/report}"
shift
python scripts/run_p4_g37.py "$stage" \
  --data-root "${SCREENRESTORE_DATA_ROOT:-$HOME/screenrestore-data}" \
  --run-root "${SCREENRESTORE_RUN_ROOT:-$HOME/screenrestore-runs}" \
  --output-root "${P4_G37_RUN_DIRECTORY:-$HOME/screenrestore-runs/p4-g37}" \
  --device "${P4_DEVICE:-mps}" "$@"
