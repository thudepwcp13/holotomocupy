### General options
### -- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J phasenet_l40s_2026061701_raw_unrolled

### -- ask for number of cores (default: 1) --
#BSUB -n 16

### -- specify that the cores must be on the same host --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

### -- specify that we need 20GB of memory per core/slot --
#BSUB -R "rusage[mem=20GB]"

### -- specify that we want the job to get killed if it exceeds 20 GB per core/slot --
#BSUB -M 20GB

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u chewu@dtu.dk

### -- send notification at start --
#BSUB -B
### -- send notification at completion --it
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./outputs/outputs/HPC_Logs/Output_2026061701_%J.out
#BSUB -e ./outputs/outputs/HPC_Logs/Output_2026061701_%J.err

. /zhome/64/c/214423/projects/BioComfert/codes/Tomocupy/Job_Scripts/miniconda_chengpeng.sh
conda activate dl_torch

cd /dtu/3d-imaging-center/projects/2022_DANFIX_08_BioComFert/raw_data_extern/Chengpeng/PycharmProjects/PhaseNet/

##bash scripts/shell/run_sample_unet_fullres_static.sh
#python scripts/run_sample_unet_fullres_static.py \
#  --config configs/experiment/single_final_big/S4_ratio_probe_amp_xy_static_loss_only_target.yaml

#python scripts/simulate_ct_phantom.py \
#  -c configs/simulation/real_probe_spheres_512_p10_a360.yaml

#python scripts/simulate_ct_phantom.py \
#  -c configs/simulation/real_probe_spheres_512_p10_a360_d050.yaml
#RAW_CONFIG="configs/experiment/distance_unrolled/real_probe_512_p30_d050_A_true_probe_unrolled2.yaml"
RAW_CONFIG="configs/experiment/distance_unrolled/real_probe_512_p30_d050_A_true_probe_unrolled.yaml"

bash scripts/run_distance_unrolled_real_probe_512_pipeline.sh "${RAW_CONFIG}"