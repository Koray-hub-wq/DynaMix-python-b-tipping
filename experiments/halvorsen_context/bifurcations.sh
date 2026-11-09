#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
check_runtime "$GPU_ID"
"$PYTHON" "$SCRIPT_DIR/evaluate.py" bifurcations \
  --data-root "$DATA_ROOT" --run-root "$RUN_ROOT" \
  --analysis-dir "$ANALYSIS_DIR" --output "${EXPERIMENT_ROOT}/bifurcations_${EVAL_CONTEXT:-trained}" \
  --device "$DEVICE" --gpu-id "$GPU_ID" --threads "$THREADS" --seed "$SEED" \
  --eval-context "${EVAL_CONTEXT:-trained}" --forecast-steps "${BIFURCATION_STEPS:-15000}" \
  --num-parameters "${NUM_PARAMETERS:-320}" --rollouts "${ROLLOUTS:-1}" \
  --burn-in-steps "${BURN_IN_STEPS:-5000}" --discard-steps "${DISCARD_STEPS:-1000}" \
  --components "${COMPONENT:-z}" "$@"
