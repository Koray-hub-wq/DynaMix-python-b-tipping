#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# GPUS distributes six independent runs over a GPU pool. Otherwise run pairs.
GPU_FIXED="${GPU_FIXED:-${GPU_ID}}"
GPU_MATCHED="${GPU_MATCHED:-${GPU_ID}}"
PARALLEL="${PARALLEL:-1}"
gpu_pool=()
if [[ -n "${GPUS:-}" ]]; then
  read -r -a gpu_pool <<< "${GPUS//,/ }"
  if (( ${#gpu_pool[@]} < 1 || ${#gpu_pool[@]} > 6 )); then
    echo 'GPUS must contain between one and six GPU IDs, e.g. GPUS="0,1,2,3,4,5".' >&2
    exit 1
  fi
  seen_gpus=' '
  for gpu in "${gpu_pool[@]}"; do
    if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ || "$seen_gpus" == *" $gpu "* ]]; then
      echo "GPUS must contain unique non-negative integer IDs: $GPUS" >&2
      exit 1
    fi
    seen_gpus+="$gpu "
  done
  check_runtime "${gpu_pool[@]}"
else
  check_runtime "$GPU_FIXED" "$GPU_MATCHED"
fi
cd "$REPO_ROOT"
mkdir -p "$EXPERIMENT_ROOT/logs"
if [[ ! -f "${DATA_ROOT}/manifest.json" ]]; then
  "$PYTHON" "$SCRIPT_DIR/prepare.py" \
    --source "$DATA_SOURCE" --output "$DATA_ROOT" --reference-phi "$REFERENCE_PHI" \
    --context-length "$CONTEXT_LENGTH" --prediction-length "$PREDICTION_LENGTH" \
    --test-length "$TEST_LENGTH" --windows-per-trajectory "$WINDOWS_PER_TRAJECTORY" --seed "$SEED"
fi
# Refuse reuse of data from a differently configured experiment.
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); keys=["reference_phi","context_length","prediction_length","test_length","windows_per_trajectory","seed"]; expected=[float(sys.argv[2]),*map(int,sys.argv[3:])]; assert [m[k] for k in keys]==expected, "Dataset manifest does not match requested settings; choose a new DATA_ROOT"' \
  "$DATA_ROOT/manifest.json" "$REFERENCE_PHI" "$CONTEXT_LENGTH" "$PREDICTION_LENGTH" "$TEST_LENGTH" "$WINDOWS_PER_TRAJECTORY" "$SEED"

run_training() {
  local split="$1" mode="$2" gpu="$3"
  local data="$DATA_ROOT/$split/$mode" run="$RUN_ROOT/$split/$mode"
  if [[ -f "$run/checkpoints/final_model.safetensors" ]]; then
    "$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); keys=["epochs","batch_size","batches_per_epoch","ssi","seed","experts","latent_dim","pwl_units"]; expected=list(map(int,sys.argv[2:])); assert [c[k] for k in keys]==expected, "Completed run has different hyperparameters; choose a new EXPERIMENT_ROOT"' \
      "$run/config.json" "${EPOCHS:-500}" "${BATCH_SIZE:-8}" "${BATCHES_PER_EPOCH:-20}" "${SSI:-25}" "$SEED" 30 10 2
    echo "Already complete: $split/$mode"
    return
  fi
  if [[ -f "$run/config.json" ]]; then
    echo "Incomplete existing run: $run. Choose a new EXPERIMENT_ROOT or move this run before retrying." >&2
    return 1
  fi
  echo "Training $split/$mode on GPU $gpu; log: $EXPERIMENT_ROOT/logs/${split}_${mode}.log"
  exec "$PYTHON" -m dynamix.training.training_setup \
    --data_path "$data/data.npy" --context_path "$data/context.npy" \
    --phi_path "$data/phi.npy" --context_phi_path "$data/context_phi.npy" \
    --test_path "$DATA_ROOT/evaluation/test.npy" --test_phi_path "$DATA_ROOT/evaluation/test_phi.npy" \
    --test_context_path "$DATA_ROOT/evaluation/${mode}_context.npy" \
    --test_initial_index "$CONTEXT_LENGTH" --no_eval_standardize --clean_initial_state \
    --save_path "$run" --device "$DEVICE" --gpu_id "$gpu" --threads "$THREADS" --seed "$SEED" \
    --experts 30 --latent_dim 10 --pwl_units 2 --expert_type almost_linear_rnn --phi_dim 1 \
    --epochs "${EPOCHS:-500}" --batch_size "${BATCH_SIZE:-8}" --batches_per_epoch "${BATCHES_PER_EPOCH:-20}" \
    --alpha 1.0 --n_interleave 10 --start_lr 0.004 --end_lr 0.00001 \
    --noise_level 0.02 --reg_strength 0.1 --ssi "${SSI:-25}" \
    --n_bins 30 --ps_smoothing 20 --mase_steps 10 \
    > "$EXPERIMENT_ROOT/logs/${split}_${mode}.log" 2>&1
}

pids=()
training_parent_pid="$BASHPID"
cleanup_training() {
  if [[ "$BASHPID" != "$training_parent_pid" ]]; then return; fi
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup_training INT TERM EXIT
if (( ${#gpu_pool[@]} > 0 )); then
  job_splits=(negative3 negative3 interleaved3 interleaved3 spanning5 spanning5)
  job_modes=(fixed matched fixed matched fixed matched)
  # Each worker runs its assigned jobs in order, at most one training per GPU.
  run_gpu_worker() {
    local slot="$1" job worker_child=''
    trap 'if [[ -n "$worker_child" ]]; then kill "$worker_child" 2>/dev/null || true; fi' INT TERM
    for ((job=slot; job<6; job+=${#gpu_pool[@]})); do
      run_training "${job_splits[$job]}" "${job_modes[$job]}" "${gpu_pool[$slot]}" & worker_child=$!
      wait "$worker_child" || return 1
      worker_child=''
    done
    trap - INT TERM
  }
  if [[ "$PARALLEL" == 1 ]]; then
    for slot in "${!gpu_pool[@]}"; do
      run_gpu_worker "$slot" & pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    pids=()
    if [[ "$failed" == 1 ]]; then
      echo "Training failed; inspect $EXPERIMENT_ROOT/logs." >&2
      exit 1
    fi
  else
    for ((job=0; job<6; job++)); do
      slot=$((job % ${#gpu_pool[@]}))
      (run_training "${job_splits[$job]}" "${job_modes[$job]}" "${gpu_pool[$slot]}")
    done
  fi
else
for split in negative3 interleaved3 spanning5; do
  if [[ "$PARALLEL" == 1 ]]; then
    run_training "$split" fixed "$GPU_FIXED" & pids=("$!")
    run_training "$split" matched "$GPU_MATCHED" & pids+=("$!")
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    pids=()
    if [[ "$failed" == 1 ]]; then
      echo "Training failed; inspect $EXPERIMENT_ROOT/logs/${split}_*.log" >&2
      exit 1
    fi
  else
    (run_training "$split" fixed "$GPU_FIXED")
    (run_training "$split" matched "$GPU_MATCHED")
  fi
done
fi
echo "Six trainings complete: $RUN_ROOT"
