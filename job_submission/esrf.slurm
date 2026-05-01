#!/bin/bash
#SBATCH --job-name holo
#SBATCH --partition=gpuhop-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mem=0

##SBATCH --nodelist=gpbm18-[01-21]
##SBATCH --nodelist=frcrg-gpu5h200-01
##SBATCH --nodelist=vgpu5h200x4-01
##SBATCH --nodelist gpu4a100-37

# --- user configuration ---
CONFIG=config_step6.conf
SCRIPT=step6.py
SAMPLE_DIR=../experimental/y350a_80um
# --------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
cp "${BASH_SOURCE[0]}"    "${scripts_dir}/"
cp "${SCRIPT_DIR}/set_affinity_gpu.sh" "${scripts_dir}/"

# patch SAMPLE_DIR in the copied job script so reruns from scripts_dir are self-contained
sed -i "s|^SAMPLE_DIR=.*|SAMPLE_DIR=${scripts_dir}|" "${scripts_dir}/$(basename "${BASH_SOURCE[0]}")"

cd "${rec_dir}"
exec > "${scripts_dir}/slurm.out" 2>&1

echo "Sample dir:  ${SAMPLE_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Scripts dir: ${scripts_dir}"
hostname

module add mamba cuda
mamba activate /home/esrf/viktor17051990/.conda/holotomocupy
export MATHDX_ROOT=/home/esrf/viktor17051990/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12

# srun --mpi=pmix_v3 ./set_affinity_gpu.sh python step0.py config_step0.conf
srun --mpi=pmix_v3 ./set_affinity_gpu.sh python "${SCRIPT}" "${CONFIG}"
# srun --mpi=pmix_v3 ./set_affinity_gpu.sh python steps15.py config_steps15.conf
