#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$script_dir/train.sh"
bash "$script_dir/trajectories.sh"
bash "$script_dir/bifurcations.sh"
