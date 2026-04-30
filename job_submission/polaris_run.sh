#!/bin/bash
#PBS -A 14347
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:59:00
#PBS -q debug
#PBS -N holotomo
#PBS -j oe

SAMPLE=y350a_80um

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/$SAMPLE"
exec > "$SCRIPT_DIR/$SAMPLE/pbs.out" 2>&1

cd "$SCRIPT_DIR/../$SAMPLE"
echo "Working directory: $(pwd)"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

module use /soft/modulefiles;  module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="/home/vvnikitin/venvs/${CONDA_NAME}"
source "${VENV_DIR}/bin/activate"

# mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "$SCRIPT_DIR/set_affinity_gpu_polaris.sh" python step0.py config_step0.conf
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "$SCRIPT_DIR/set_affinity_gpu_polaris.sh" python steps15.py config_steps15.conf
# mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "$SCRIPT_DIR/set_affinity_gpu_polaris.sh" python step6.py config_step6.conf
