#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DanMAX nano step0 launcher
# -----------------------------------------------------------------------------
# Usage:
#   bash run_step0.sh sanity   # validate HDF5 layout and write preview
#   bash run_step0.sh nfp      # run near-field ptychography reconstruction
#
# Example masked full-width run:
#   N=3712 USE_VALID_DETECTOR_MASK=true NGPUS=1 bash run_step0.sh nfp
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

# Input files
DATA_FOLDER="/dtu/3d-imaging-center/projects/2026_DANFIX_XHIST/raw_data_3DIM/DanMAX April 2026/NTT_multi_dist/"
DARK_FILE="${DARK_FILE:-${DATA_FOLDER}/scan-0076.h5}"
FLAT_FILE="${FLAT_FILE:-${DATA_FOLDER}/scan-0097.h5}"
SAMPLE_FILE="${SAMPLE_FILE:-${DATA_FOLDER}/scan-0096.h5}"

# Output paths
OUT_DIR="${OUT_DIR:-/zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/output_step0_v4_3712_pos_row_iter100_40frames}"
CONFIG_FILE="${CONFIG_FILE:-${OUT_DIR}/config_step0.generated.conf}"
H5_OUT="${H5_OUT:-${OUT_DIR}/DanMAX_nano_nfp_results.h5}"
PATH_OUT="${PATH_OUT:-${OUT_DIR}/nfp_work}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/step0_${MODE}.log}"

# DanMAX HDF5 paths
DETECTOR_PATH="${DETECTOR_PATH:-/entry/measurement/orca}"
X_PATH="${X_PATH:-/entry/measurement/tom_sam_x}"
Y_PATH="${Y_PATH:-/entry/measurement/tom_y}"

# Geometry
ENERGY="${ENERGY:-19.55}"
Z1="${Z1:-0.12669}"
FOCUSTODETECTORDISTANCE="${FOCUSTODETECTORDISTANCE:-1.55669}"
DETECTOR_PIXELSIZE="${DETECTOR_PIXELSIZE:-5.5e-7}"

# Position conversion
POSITION_UNIT="${POSITION_UNIT:-mm}"
POS_ROW_SIGN="${POS_ROW_SIGN:-1.0}"
POS_COL_SIGN="${POS_COL_SIGN:-1.0}"
CENTER_POSITIONS="${CENTER_POSITIONS:-true}"

# Reconstruction size and solver parameters
N="${N:-3712}"
NITER="${NITER:-100}"
NCHUNK="${NCHUNK:-4}"
VIS_STEP="${VIS_STEP:-1}"
ERR_STEP="${ERR_STEP:-1}"
RHO="${RHO:-1,2,0.1}"

# Execution and preprocessing
RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-${DEFAULT_RUN_RECONSTRUCTION}}"
WRITE_CORRECTED_PREVIEW="${WRITE_CORRECTED_PREVIEW:-true}"
WRITE_POSITION_BBOX_PLOT="${WRITE_POSITION_BBOX_PLOT:-true}"
POSITION_BBOX_GRID_SIZE="${POSITION_BBOX_GRID_SIZE:-5}"
PREVIEW_COUNT="${PREVIEW_COUNT:-8}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
FLAT_CORRECTION="${FLAT_CORRECTION:-false}"
USE_VALID_DETECTOR_MASK="${USE_VALID_DETECTOR_MASK:-true}"
NGPUS="${NGPUS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"

mkdir -p "${OUT_DIR}" "${PATH_OUT}"

for f in "${DARK_FILE}" "${FLAT_FILE}" "${SAMPLE_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: input file does not exist: ${f}" >&2
    exit 2
  fi
done

cat > "${CONFIG_FILE}" <<EOF
dark_file=${DARK_FILE}
flat_file=${FLAT_FILE}
sample_file=${SAMPLE_FILE}

h5_out=${H5_OUT}
path_out=${PATH_OUT}

detector_path=${DETECTOR_PATH}
x_path=${X_PATH}
y_path=${Y_PATH}

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

run_reconstruction=${RUN_RECONSTRUCTION}
write_corrected_preview=${WRITE_CORRECTED_PREVIEW}
write_position_bbox_plot=${WRITE_POSITION_BBOX_PLOT}
position_bbox_grid_size=${POSITION_BBOX_GRID_SIZE}
preview_count=${PREVIEW_COUNT}
log_level=${LOG_LEVEL}

flat_correct=${FLAT_CORRECTION}
use_valid_detector_mask=${USE_VALID_DETECTOR_MASK}
frame_ids=30-69
EOF

echo "=== DanMAX nano step0 launcher ==="
echo "mode                    : ${MODE}"
echo "run_reconstruction      : ${RUN_RECONSTRUCTION}"
echo "config                  : ${CONFIG_FILE}"
echo "h5_out                  : ${H5_OUT}"
echo "log                     : ${LOG_FILE}"
echo "n                       : ${N}"
echo "flat_correct            : ${FLAT_CORRECTION}"
echo "valid_detector_mask     : ${USE_VALID_DETECTOR_MASK}"
echo "ngpus                   : ${NGPUS}"

unset I_MPI_SHM_LMT
unset I_MPI_FABRICS_LIST
if [[ "${RUN_RECONSTRUCTION}" == "true" || "${RUN_RECONSTRUCTION}" == "True" || "${RUN_RECONSTRUCTION}" == "1" ]]; then
  if [[ "${NGPUS}" -gt 1 ]]; then
    echo "Running: ${MPIRUN_BIN} -n ${NGPUS} ${PYTHON_BIN} step0.py ${CONFIG_FILE}"
    "${MPIRUN_BIN}" -n "${NGPUS}" "${PYTHON_BIN}" step0.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
  else
    echo "Running: ${PYTHON_BIN} step0.py ${CONFIG_FILE}"
    "${PYTHON_BIN}" step0.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
  fi
else
  echo "Running sanity check only: ${PYTHON_BIN} step0.py ${CONFIG_FILE}"
  "${PYTHON_BIN}" step0.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
fi

echo "Done. Output HDF5: ${H5_OUT}"
