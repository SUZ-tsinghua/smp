#!/usr/bin/env bash
# Run csv_to_npz.py in parallel by sharding the CSV file list across N workers.
# Each worker initialises mjlab Sim once and processes its shard sequentially.
#
# Usage:
#   ./scripts/batch_csv_to_npz.sh                              # 4 workers, defaults
#   N_PARALLEL=8 ./scripts/batch_csv_to_npz.sh                 # 8 workers
#   INPUT_DIR=datasets/csv OUTPUT_DIR=datasets/npz \
#     WINDOW_SIZE=20 STRIDE=1 INPUT_FPS=30 OUTPUT_FPS=50 \
#     N_PARALLEL=4 ./scripts/batch_csv_to_npz.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

INPUT_DIR="${INPUT_DIR:-datasets/csv}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/npz}"
WINDOW_SIZE="${WINDOW_SIZE:-20}"
STRIDE="${STRIDE:-1}"
INPUT_FPS="${INPUT_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
N_PARALLEL="${N_PARALLEL:-4}"

# xargs/nohup spawn clean shells without the user's PATH; resolve uv up front.
UV="$(command -v uv || echo "${HOME}/.local/bin/uv")"

cd "${PROJECT_DIR}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi

total=$(find "${INPUT_DIR}" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')
if [[ "${total}" -eq 0 ]]; then
  echo "[ERROR] no CSV files in ${INPUT_DIR}" >&2
  exit 1
fi

LOG_DIR="logs/csv_to_npz/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[INFO] ${total} CSVs in ${INPUT_DIR}"
echo "[INFO] ${N_PARALLEL} parallel workers, logs -> ${LOG_DIR}"
echo "[INFO] window=${WINDOW_SIZE} stride=${STRIDE} fps ${INPUT_FPS} -> ${OUTPUT_FPS}"
echo

pids=()
for i in $(seq 0 $((N_PARALLEL - 1))); do
  log_file="${LOG_DIR}/worker_${i}.log"
  echo "[START] worker ${i}/${N_PARALLEL} (log: ${log_file})"
  "${UV}" run python "${SCRIPT_DIR}/csv_to_npz.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --window-size "${WINDOW_SIZE}" \
    --stride "${STRIDE}" \
    --input-fps "${INPUT_FPS}" \
    --output-fps "${OUTPUT_FPS}" \
    --shard-index "${i}" \
    --num-shards "${N_PARALLEL}" \
    > "${log_file}" 2>&1 &
  pids+=($!)
done

echo
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[DONE] worker ${i}"
  else
    echo "[FAIL] worker ${i} (see ${LOG_DIR}/worker_${i}.log)" >&2
    fail=1
  fi
done

if [[ ${fail} -ne 0 ]]; then
  echo "[INFO] some workers failed; partial outputs in ${OUTPUT_DIR}" >&2
  exit 1
fi

echo
echo "[INFO] all workers complete. ${total} CSVs -> ${OUTPUT_DIR}"
