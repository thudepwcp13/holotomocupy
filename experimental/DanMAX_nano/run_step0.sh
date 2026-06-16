#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DanMAX nano step0 launcher
# -----------------------------------------------------------------------------
# Usage:
#   bash run_step0.sh sanity   # default: only validate HDF5 layout and write preview
#   bash run_step0.sh nfp      # run near-field ptychography reconstruction
#
# All variables below can be overridden from the command line, for example:
#   DARK_FILE=/data/dark.h5 FLAT_FILE=/data/flat.h5 SAMPLE_FILE=/data/sample.h5 bash run_step0.sh sanity
#   NGPUS=4 RUN_RECONSTRUCTION=true bash run_step0.sh nfp
# -----------------------------------------------------------------------------

MODE="${1:-sanity}"
case "${MODE}" in
  sanity)
    DEFAULT_RUN_RECONSTRUCTION="false"
    ;;
  nfp|recon|reconstruction)
    DEFAULT_RUN_RECONSTRUCTION="true"
    ;;
  *)
    echo "ERROR: unknown mode '${MODE}'. Use 'sanity' or 'nfp'." >&2
    exit 2
    ;;
esac

# Resolve this directory so the script can be launched from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
DARK_FILE="${DARK_FILE:-/path/to/dark.h5}"
FLAT_FILE="${FLAT_FILE:-/path/to/flat.h5}"
SAMPLE_FILE="${SAMPLE_FILE:-/path/to/sample.h5}"

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/output_step0}"
CONFIG_FILE="${CONFIG_FILE:-${OUT_DIR}/config_step0.generated.conf}"
H5_OUT="${H5_OUT:-${OUT_DIR}/DanMAX_nano_nfp_results.h5}"
PATH_OUT="${PATH_OUT:-${OUT_DIR}/nfp_work}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/step0_${MODE}.log}"

# -----------------------------------------------------------------------------
# DanMAX HDF5 paths
# -----------------------------------------------------------------------------
DETECTOR_PATH="${DETECTOR_PATH:-/entry/measurement/orca}"
X_PATH="${X_PATH:-/entry/measurement/tom_sam_x}"
Y_PATH="${Y_PATH:-/entry/measurement/tom_y}"

# -----------------------------------------------------------------------------
# Geometry parameters. Edit or override these before a production run.
# -----------------------------------------------------------------------------
ENERGY="${ENERGY:-20.0}"                         # keV
Z1="${Z1:-0.004}"                                # focus-to-sample distance [m]
FOCUSTODETECTORDISTANCE="${FOCUSTODETECTORDISTANCE:-1.20}" # focus-to-detector distance [m]
DETECTOR_PIXELSIZE="${DETECTOR_PIXELSIZE:-6.5e-6}"          # detector pixel size [m]

# -----------------------------------------------------------------------------
# Position conversion for tom_sam_x / tom_y
# -----------------------------------------------------------------------------
POSITION_UNIT="${POSITION_UNIT:-um}"             # m, mm, um, nm, or px
POS_ROW_SIGN="${POS_ROW_SIGN:--1.0}"             # row shift from tom_y
POS_COL_SIGN="${POS_COL_SIGN:-1.0}"              # column shift from tom_sam_x
CENTER_POSITIONS="${CENTER_POSITIONS:-true}"

# -----------------------------------------------------------------------------
# Reconstruction crop and NFP solver parameters
# -----------------------------------------------------------------------------
N="${N:-2048}"                                   # set N=0 for largest centered square crop
NITER="${NITER:-129}"
NCHUNK="${NCHUNK:-4}"
VIS_STEP="${VIS_STEP:-32}"
ERR_STEP="${ERR_STEP:-32}"
RHO="${RHO:-1,2,0.1}"

# -----------------------------------------------------------------------------
# Execution parameters
# -----------------------------------------------------------------------------
RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-${DEFAULT_RUN_RECONSTRUCTION}}"
WRITE_CORRECTED_PREVIEW="${WRITE_CORRECTED_PREVIEW:-true}"
PREVIEW_COUNT="${PREVIEW_COUNT:-8}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
NGPUS="${NGPUS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"

mkdir -p "${OUT_DIR}" "${PATH_OUT}"

# Basic path checks. Keep /path/to placeholders explicit to avoid silent wrong runs.
for f in "${DARK_FILE}" "${FLAT_FILE}" "${SAMPLE_FILE}"; do
  if [[ "${f}" == /path/to/* ]]; then
    echo "ERROR: please set DARK_FILE, FLAT_FILE, and SAMPLE_FILE before running." >&2
    echo "Example:" >&2
    echo "  DARK_FILE=/data/dark.h5 FLAT_FILE=/data/flat.h5 SAMPLE_FILE=/data/sample.h5 bash run_step0.sh sanity" >&2
    exit 2
  fi
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
preview_count=${PREVIEW_COUNT}
log_level=${LOG_LEVEL}
EOF

echo "=== DanMAX nano step0 launcher ==="
echo "mode                 : ${MODE}"
echo "run_reconstruction   : ${RUN_RECONSTRUCTION}"
echo "config               : ${CONFIG_FILE}"
echo "h5_out               : ${H5_OUT}"
echo "log                  : ${LOG_FILE}"
echo "n                    : ${N}"
echo "ngpus                : ${NGPUS}"

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
