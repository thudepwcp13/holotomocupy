#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ID16B NFP step0 launcher
# -----------------------------------------------------------------------------
# Usage:
#   DARK_FILE=/data/dark.h5 FLAT_FILE=/data/flat.h5 SAMPLE_FILE=/data/sample.h5 \
#     bash run_step0.sh sanity
#   DARK_FILE=/data/dark.h5 FLAT_FILE=/data/flat.h5 SAMPLE_FILE=/data/sample.h5 \
#     bash run_step0.sh nfp
# -----------------------------------------------------------------------------

MODE="${1:-sanity}"
case "${MODE}" in
  sanity) DEFAULT_RUN_RECONSTRUCTION="false" ;;
  nfp|recon|reconstruction) DEFAULT_RUN_RECONSTRUCTION="true" ;;
  *)
    echo "ERROR: unknown mode '${MODE}'. Use 'sanity' or 'nfp'." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Input files and HDF5 key layout
DARK_FILE="${DARK_FILE:-/zhome/64/c/214423/bioconfert/raw_data_extern/2026_05_07_ESRF_ID16B/RAW_DATA/S2_C1/S2_C1_ht_55nm_06/S2_C1_ht_55nm_06.h5}"
DARK_KEY="${DARK_KEY:-/3.1/measurement/pco1}"
DARK_NFRAMES="${DARK_NFRAMES:-51}"

FLAT_FILE="${FLAT_FILE:-/zhome/64/c/214423/bioconfert/raw_data_extern/2026_05_07_ESRF_ID16B/RAW_DATA/ptycho_ref_ter/ptycho_ref_ter_0001/ptycho_ref_ter_0001.h5}"
FLAT_KEY='/{n}.1/measurement/pco1'
FLAT_SCAN_IDS="${FLAT_SCAN_IDS:-1:10}"

SAMPLE_FILE="${SAMPLE_FILE:-/zhome/64/c/214423/bioconfert/raw_data_extern/2026_05_07_ESRF_ID16B/RAW_DATA/ptycho_ht_55nm_06_ter/ptycho_ht_55nm_06_ter_0001/ptycho_ht_55nm_06_ter_0001.h5}"
SAMPLE_KEY='/{n}.1/measurement/pco1'
SAMPLE_SCAN_IDS="${SAMPLE_SCAN_IDS:-1:256:4}"

MOTOR_X_KEY='/{n}.1/instrument/positioners/sy'
MOTOR_Y_KEY='/{n}.1/instrument/positioners/sz'

FRAME_IDS="${FRAME_IDS:-all}"

# Geometry
ENERGY="${ENERGY:-29.63}"
Z1="${Z1:-0.059599}"
FOCUSTODETECTORDISTANCE="${FOCUSTODETECTORDISTANCE:-0.704433}"
DETECTOR_PIXELSIZE="${DETECTOR_PIXELSIZE:-6.5e-7}"

# Position conversion
POSITION_UNIT="${POSITION_UNIT:-mm}"
POS_ROW_SIGN="${POS_ROW_SIGN:--1.0}"
POS_COL_SIGN="${POS_COL_SIGN:-1.0}"
CENTER_POSITIONS="${CENTER_POSITIONS:-true}"

# Reconstruction parameters
N="${N:-2048}"
NITER="${NITER:-10}"
NCHUNK="${NCHUNK:-4}"
VIS_STEP="${VIS_STEP:-1}"
ERR_STEP="${ERR_STEP:-1}"
RHO="${RHO:-1,2,0.00001}"

# Output paths
OUT_DIR="${OUT_DIR:-/zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/ID16B_output/output_step0_0deg_gap4}"
CONFIG_FILE="${CONFIG_FILE:-${OUT_DIR}/config_step0.generated.conf}"
H5_OUT="${H5_OUT:-${OUT_DIR}/ID16B_nfp_results.h5}"
PATH_OUT="${PATH_OUT:-${OUT_DIR}}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/step0_${MODE}.log}"

# Execution and preprocessing
RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-${DEFAULT_RUN_RECONSTRUCTION}}"
FLAT_CORRECTION="${FLAT_CORRECTION:-false}"
WRITE_CORRECTED_PREVIEW="${WRITE_CORRECTED_PREVIEW:-true}"
PREVIEW_COUNT="${PREVIEW_COUNT:-8}"
WRITE_POSITION_BBOX_PLOT="${WRITE_POSITION_BBOX_PLOT:-true}"
POSITION_BBOX_GRID_SIZE="${POSITION_BBOX_GRID_SIZE:-5}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
NGPUS="${NGPUS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"

mkdir -p "${OUT_DIR}" "${PATH_OUT}"

for file in "${DARK_FILE}" "${FLAT_FILE}" "${SAMPLE_FILE}"; do
  if [[ "${file}" == /path/to/* || ! -f "${file}" ]]; then
    echo "ERROR: input file does not exist: ${file}" >&2
    exit 2
  fi
done

cat > "${CONFIG_FILE}" <<CONF
dark_file=${DARK_FILE}
dark_key=${DARK_KEY}
dark_nframes=${DARK_NFRAMES}

flat_file=${FLAT_FILE}
flat_key=${FLAT_KEY}
flat_scan_ids=${FLAT_SCAN_IDS}

sample_file=${SAMPLE_FILE}
sample_key=${SAMPLE_KEY}
sample_scan_ids=${SAMPLE_SCAN_IDS}
frame_ids=${FRAME_IDS}
motor_x_key=${MOTOR_X_KEY}
motor_y_key=${MOTOR_Y_KEY}

h5_out=${H5_OUT}
path_out=${PATH_OUT}

energy=${ENERGY}
z1=${Z1}
focustodetectordistance=${FOCUSTODETECTORDISTANCE}
detector_pixelsize=${DETECTOR_PIXELSIZE}

position_unit=${POSITION_UNIT}
pos_row_sign=${POS_ROW_SIGN}
pos_col_sign=${POS_COL_SIGN}
center_positions=${CENTER_POSITIONS}

n=${N}
niter=${NITER}
nchunk=${NCHUNK}
vis_step=${VIS_STEP}
err_step=${ERR_STEP}
rho=${RHO}

flat_correct=${FLAT_CORRECTION}
run_reconstruction=${RUN_RECONSTRUCTION}
write_corrected_preview=${WRITE_CORRECTED_PREVIEW}
preview_count=${PREVIEW_COUNT}
write_position_bbox_plot=${WRITE_POSITION_BBOX_PLOT}
position_bbox_grid_size=${POSITION_BBOX_GRID_SIZE}
log_level=${LOG_LEVEL}
CONF

echo "=== ID16B NFP step0 launcher ==="
echo "mode                    : ${MODE}"
echo "run_reconstruction      : ${RUN_RECONSTRUCTION}"
echo "config                  : ${CONFIG_FILE}"
echo "h5_out                  : ${H5_OUT}"
echo "sample scan IDs         : ${SAMPLE_SCAN_IDS}"
echo "selected frame IDs      : ${FRAME_IDS}"
echo "n                       : ${N}"
echo "ngpus                   : ${NGPUS}"

unset I_MPI_SHM_LMT
unset I_MPI_FABRICS_LIST
if [[ "${RUN_RECONSTRUCTION}" == "true" || "${RUN_RECONSTRUCTION}" == "True" || "${RUN_RECONSTRUCTION}" == "1" ]]; then
  if [[ "${NGPUS}" -gt 1 ]]; then
    "${MPIRUN_BIN}" -n "${NGPUS}" "${PYTHON_BIN}" step0.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
  else
    "${PYTHON_BIN}" step0.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
  fi
else
  "${PYTHON_BIN}" step0.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
fi

echo "Done. Output HDF5: ${H5_OUT}"
