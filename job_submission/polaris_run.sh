#!/bin/bash
#PBS -A 14347
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:59:00
#PBS -q debug
#PBS -N holotomo
#PBS -j oe

# --- user configuration ---
CONFIG=config_step6.conf
SCRIPT=step6.py
SAMPLE_DIR=../experimental/y350a_80um
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/${SAMPLE_DIR}" && pwd)"

# rec_dir is path_out from the config file
rec_dir=$(grep -E '^\s*path_out\s*=' "${SAMPLE_DIR}/${CONFIG}" | head -1 \
          | sed 's/[^=]*=\s*//' | sed 's/\s*#.*//' | tr -d ' ')

mkdir -p "${rec_dir}"

# scripts_dir is a dated snapshot folder inside rec_dir
scripts_dir="${rec_dir}/scripts$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${scripts_dir}"

# copy scripts, configs, affinity helper, and this job script into scripts_dir
cp "${SAMPLE_DIR}"/*.py   "${scripts_dir}/" 2>/dev/null || true
cp "${SAMPLE_DIR}"/*.conf "${scripts_dir}/" 2>/dev/null || true
cp "$0"                   "${scripts_dir}/"
cp "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" "${scripts_dir}/"

# patch SAMPLE_DIR in the copied job script so reruns from scripts_dir are self-contained
sed -i "s|^SAMPLE_DIR=.*|SAMPLE_DIR=${scripts_dir}|" "${scripts_dir}/$(basename "$0")"

cd "${rec_dir}"
exec > "${scripts_dir}/pbs.out" 2>&1

echo "Sample dir:  ${SAMPLE_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Scripts dir: ${scripts_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

module use /soft/modulefiles;  module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="/home/vvnikitin/venvs/${CONDA_NAME}"
source "${VENV_DIR}/bin/activate"

# mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh python step0.py config_step0.conf
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh python "${SCRIPT}" "${CONFIG}"
# mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh python steps15.py config_steps15.conf
