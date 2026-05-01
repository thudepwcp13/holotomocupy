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
# --------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# rec_dir is path_out from the config file
rec_dir=$(grep -E '^\s*path_out\s*=' "${SCRIPT_DIR}/${CONFIG}" | head -1 \
          | sed 's/[^=]*=\s*//' | sed 's/\s*#.*//' | tr -d ' ')
mkdir -p "${rec_dir}"

# snapshot the current scripts folder into a dated copy
scripts_dir="${rec_dir}/scripts$(date +%Y-%m-%d_%H-%M-%S)"
cp -r "${SCRIPT_DIR}" "${scripts_dir}"

cd "${rec_dir}"
exec > >(tee "${scripts_dir}/slurm.out" "slurm.out") 2>&1

echo "Rec dir:     ${rec_dir}"
echo "Scripts dir: ${scripts_dir}"
hostname

module add mamba cuda
mamba activate /home/esrf/viktor17051990/.conda/holotomocupy
export MATHDX_ROOT=/home/esrf/viktor17051990/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12

# srun --mpi=pmix_v3 "${scripts_dir}/set_affinity_gpu.sh" python "${scripts_dir}/step0.py" "${scripts_dir}/config_step0.conf"
srun --mpi=pmix_v3 "${scripts_dir}/set_affinity_gpu.sh" python "${scripts_dir}/${SCRIPT}" "${scripts_dir}/${CONFIG}"
# srun --mpi=pmix_v3 "${scripts_dir}/set_affinity_gpu.sh" python "${scripts_dir}/steps15.py" "${scripts_dir}/config_steps15.conf"
