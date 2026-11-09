#!/usr/bin/env bash
# Shared server paths and runtime. Source from the experiment scripts.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    source "${REPO_ROOT}/venv/bin/activate"
  elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
  fi
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
SEED="${SEED:-8}"
REFERENCE_PHI="${REFERENCE_PHI:--0.75}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/results/halvorsen_context_phi_${REFERENCE_PHI}/seed_${SEED}}"
DATA_ROOT="${DATA_ROOT:-${EXPERIMENT_ROOT}/datasets}"
RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_ROOT}/models}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
DATA_SOURCE="${DATA_SOURCE:-/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/export/home/klenkeit/diffrentbutconst_prams}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-500}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-200}"
TEST_LENGTH="${TEST_LENGTH:-10000}"
WINDOWS_PER_TRAJECTORY="${WINDOWS_PER_TRAJECTORY:-16}"

check_runtime() {
  "$PYTHON" -c 'import sys, torch; device=sys.argv[1]; gpus=list(map(int,sys.argv[2:])); assert device != "cuda" or (torch.cuda.is_available() and all(0 <= gpu < torch.cuda.device_count() for gpu in gpus)), "Requested CUDA GPUs are unavailable: " + str(gpus)' "$DEVICE" "$@"
}
