#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DanMAX nano step0 Z1 parameter scan
# -----------------------------------------------------------------------------
# This wrapper reuses the current run_step0.sh so every parameter except Z1 and
# the output directory remains identical to the normal launcher.
#
# Default scan:
#   Z1 = 0.12300, 0.12320, ..., 0.12900 m  (31 runs)
#
# Usage:
#   bash run_step0_z1_scan.sh nfp
#   bash run_step0_z1_scan.sh sanity
#
# Existing run_step0.sh overrides are preserved, for example:
#   NITER=20 FRAME_IDS=0-99:2 NGPUS=1 bash run_step0_z1_scan.sh nfp
#
# Optional output root override:
#   Z1_SCAN_OUT_ROOT=/path/to/z1_scan bash run_step0_z1_scan.sh nfp
# -----------------------------------------------------------------------------

MODE="${1:-nfp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STEP0="${SCRIPT_DIR}/run_step0.sh"

if [[ ! -f "${RUN_STEP0}" ]]; then
  echo "ERROR: run_step0.sh was not found at ${RUN_STEP0}" >&2
  exit 2
fi

# Root directory for all Z1 experiments. Each Z1 gets its own subdirectory.
Z1_SCAN_OUT_ROOT="${Z1_SCAN_OUT_ROOT:-/zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/CA_z1_scan_step0}"
mkdir -p "${Z1_SCAN_OUT_ROOT}"

# Integer representation in units of 1e-5 m avoids floating-point accumulation:
#   12300 -> 0.12300 m
#   step 20 -> 0.00020 m
#   12900 -> 0.12900 m
Z1_START_SCALED=14600
Z1_STOP_SCALED=15300
Z1_STEP_SCALED=50

TOTAL_RUNS=$(( (Z1_STOP_SCALED - Z1_START_SCALED) / Z1_STEP_SCALED + 1 ))
RUN_INDEX=0

printf "=== DanMAX step0 Z1 scan ===\n"
printf "mode            : %s\n" "${MODE}"
printf "Z1 range        : 0.${Z1_START_SCALED} ... 0.${Z1_STOP_SCALED} m\n"
printf "Z1 step         : 0.000${Z1_STEP_SCALED} m\n"
printf "number of runs  : %d\n" "${TOTAL_RUNS}"
printf "output root     : %s\n" "${Z1_SCAN_OUT_ROOT}"

for (( z1_scaled=Z1_START_SCALED; z1_scaled<=Z1_STOP_SCALED; z1_scaled+=Z1_STEP_SCALED )); do
  RUN_INDEX=$((RUN_INDEX + 1))

  # Fixed-width decimal value, e.g. 0.12660.
  printf -v Z1_VALUE "0.%05d" "${z1_scaled}"
  Z1_TAG="${Z1_VALUE/./p}"
  RUN_OUT_DIR="${Z1_SCAN_OUT_ROOT}/z1_${Z1_TAG}"

  printf "\n"
  printf "[%02d/%02d] Z1=%s m\n" "${RUN_INDEX}" "${TOTAL_RUNS}" "${Z1_VALUE}"
  printf "output          : %s\n" "${RUN_OUT_DIR}"

  # Only Z1 and OUT_DIR are changed here. All other defaults and any environment
  # overrides are handled by the current run_step0.sh.
  Z1="${Z1_VALUE}" \
  NITER=10 \
  OUT_DIR="${RUN_OUT_DIR}" \
  bash "${RUN_STEP0}" "${MODE}"
done

printf "\nCompleted all %d Z1 runs.\n" "${TOTAL_RUNS}"
printf "Results root: %s\n" "${Z1_SCAN_OUT_ROOT}"
