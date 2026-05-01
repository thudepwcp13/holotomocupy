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
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# rec_dir is path_out from the config file
rec_dir=$(grep -E '^\s*path_out\s*=' "${SCRIPT_DIR}/${CONFIG}" | head -1 \
          | sed 's/[^=]*=\s*//' | sed 's/\s*#.*//' | tr -d ' ')
mkdir -p "${rec_dir}"

# snapshot the current scripts folder into a dated copy
scripts_dir="${rec_dir}/scripts$(date +%Y-%m-%d_%H-%M-%S)"
cp -r "${SCRIPT_DIR}" "${scripts_dir}"

cd "${rec_dir}"
exec > >(tee "${scripts_dir}/pbs.out" "pbs.out") 2>&1

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

# mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${scripts_dir}/set_affinity_gpu_polaris.sh" python "${scripts_dir}/step0.py" "${scripts_dir}/config_step0.conf"
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${scripts_dir}/set_affinity_gpu_polaris.sh" python "${scripts_dir}/${SCRIPT}" "${scripts_dir}/${CONFIG}"
# mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${scripts_dir}/set_affinity_gpu_polaris.sh" python "${scripts_dir}/steps15.py" "${scripts_dir}/config_steps15.conf"
