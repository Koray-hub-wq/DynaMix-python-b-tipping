#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
check_runtime "$GPU_ID"
"$PYTHON" "$SCRIPT_DIR/evaluate.py" trajectories \
  --data-root "$DATA_ROOT" --run-root "$RUN_ROOT" \
  --output "${EXPERIMENT_ROOT}/trajectories_${EVAL_CONTEXT:-trained}" \
  --device "$DEVICE" --gpu-id "$GPU_ID" --threads "$THREADS" --seed "$SEED" \
  --eval-context "${EVAL_CONTEXT:-trained}" --forecast-steps "${TRAJECTORY_STEPS:-0}" \
  --batch-size "${EVAL_BATCH_SIZE:-8}" "$@"
