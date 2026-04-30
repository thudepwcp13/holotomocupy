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

SAMPLE=y350a_80um

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/$SAMPLE"
exec > "$SCRIPT_DIR/$SAMPLE/slurm.out" 2>&1

cd "$SCRIPT_DIR/../experimental/$SAMPLE"
echo "Working directory: $(pwd)"

hostname

module add mamba cuda
mamba activate /home/esrf/viktor17051990/.conda/holotomocupy
export MATHDX_ROOT=/home/esrf/viktor17051990/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12

# srun --mpi=pmix_v3 "$SCRIPT_DIR/set_affinity_gpu.sh" python step0.py config_step0.conf
srun --mpi=pmix_v3 "$SCRIPT_DIR/set_affinity_gpu.sh" python steps15.py config_steps15.conf
# srun --mpi=pmix_v3 "$SCRIPT_DIR/set_affinity_gpu.sh" python step6.py config_step6.conf
