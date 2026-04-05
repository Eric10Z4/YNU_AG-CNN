#!/usr/bin/env bash
set -euo pipefail

# Example server launcher for dual-socket NUMA machines.
# Usage:
#   bash gat_cpp/tools/run_train_server.sh
# Optional overrides:
#   CPP_THREADS=64 N_PLAYOUT=256 TORCH_CPU_THREADS=4 TORCH_INTEROP_THREADS=1 \ 
#   SELFPLAY_BACKEND=cpp EVAL_BACKEND=cpp AUTO_TUNE=1 \
#   bash gat_cpp/tools/run_train_server.sh

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/Scripts/python.exe}"

SELFPLAY_BACKEND="${SELFPLAY_BACKEND:-cpp}"
EVAL_BACKEND="${EVAL_BACKEND:-cpp}"
CPP_THREADS="${CPP_THREADS:-64}"
N_PLAYOUT="${N_PLAYOUT:-256}"
TORCH_CPU_THREADS="${TORCH_CPU_THREADS:-4}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
AUTO_TUNE="${AUTO_TUNE:-0}"
TUNE_PLAYOUT="${TUNE_PLAYOUT:-128}"
TUNE_STEPS="${TUNE_STEPS:-12}"
CPP_THREAD_CANDIDATES="${CPP_THREAD_CANDIDATES:-16,24,32,48,64,96}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/gat_cpp/cpp_train.py"
  --selfplay-backend "$SELFPLAY_BACKEND"
  --eval-backend "$EVAL_BACKEND"
  --cpp-threads "$CPP_THREADS"
  --n-playout "$N_PLAYOUT"
  --torch-cpu-threads "$TORCH_CPU_THREADS"
  --torch-interop-threads "$TORCH_INTEROP_THREADS"
)

if [[ "$AUTO_TUNE" == "1" ]]; then
  CMD+=(
    --auto-tune-threads
    --cpp-thread-candidates "$CPP_THREAD_CANDIDATES"
    --tune-playout "$TUNE_PLAYOUT"
    --tune-steps "$TUNE_STEPS"
  )
fi

if command -v numactl >/dev/null 2>&1; then
  echo "[INFO] Launch with numactl interleave (NUMA-friendly)."
  exec numactl --interleave=all "${CMD[@]}"
else
  echo "[WARN] numactl not found, run without NUMA pinning."
  exec "${CMD[@]}"
fi
