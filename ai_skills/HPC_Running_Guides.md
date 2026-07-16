# How to run an experiment in the DTU HPC?
Updating date: June 19, 2026

## Before you start
You may run Python commands outside the sandbox, but follow these rules:
1. Do not modify or delete files without asking me first.
2. Do not modify or delete existing checkpoints/, outputs/, predictions/, or test_data/. Writing new outputs/logs is allowed only when the task explicitly specifies the output path.
3. Before running pytest or scripts, set temporary/cache directories to a safe location.
4. After each command, report what happened before doing the next step.
5. All long-running training or experiment commands must run inside a `tmux` session, with stdout/stderr saved to a log file.

## Instructions
You are now working in the DTU HPC cluster, and you need to run some experiments by using the shell scripts or python scripts.

The DTU HPC has several different nodes, and you are most probably working on the login node. 

Usually we need to run scripts on GPU devices. So please run the following shell command to check if you have any GPUs:
```bash
nvidia-smi
```
If you got some responses to show the available GPUs, then it means you are working on a GPU node.

If `nvidia-smi` fails with `command not found`, or no NVIDIA GPU is visible because you are on a login node, this is not a task failure. Switch to a GPU node first, then run `nvidia-smi` again. This login-node/GPU-node issue is an allowed exception, similar to OOM retry handling.

Also, from the response, you can check how many GPU memory left in each GPU device, and then switch to use the maximum memory GPU device.
For example, if it response the following message:
|   0  NVIDIA A100-PCIE-40GB          On  |   00000000:37:00.0 Off |                    0 |
| N/A   70C    P0             48W /  250W |   18416MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-PCIE-40GB          On  |   00000000:86:00.0 Off |                    0 |
| N/A   72C    P0             48W /  250W |   32006MiB /  40960MiB |     11%      Default |

Then you can see GPU 0 has more available memory than GPU 1, and you can run the following command to change the visible device:
```bash
export CUDA_VISIBLE_DEVICES=0
```

If you can't see any GPU devices or all GPUs have very few memory left in this device, you can run the following command to switch to other nodes:
```bash
# 2x A100 GPU (40 GB memory)
a100sh

# 4x V100 GPU (32GB memory)
sxm2sh

# 2x V100 GPU (16 GB mempry)
voltash
```
The priority should be "a100sh > sxm2sh > voltash".

After finding the right device, start or reuse a `tmux` session on the GPU node before running any long experiment or training command.

Before starting a new experiment, check whether a previous session is already running. If a Codex session was interrupted, do not start a duplicate job until you have inspected the existing `tmux` sessions and logs:
```bash
tmux ls
```
If `tmux ls` reports `no server running` or `failed to connect to server`, it simply means there is no active tmux session. This is not a fatal task error.

If a relevant session already exists, attach to it or inspect its logs instead of restarting:
```bash
tmux attach -t <session_name>
```

If no relevant session exists, create one:
```bash
tmux new -s <session_name>
```

Inside `tmux`, run the experiment command and save both stdout and stderr to a task-specific log file. For example:
```bash
python path/to/script.py <args> 2>&1 | tee path/to/task_run.log
```

If Codex loses access to command output, reconnect by checking `tmux ls`, attaching to the existing session, and reading the log with:
```bash
tail -n 100 path/to/task_run.log
```

Do not restart the command just because Codex timed out or temporarily stopped receiving output. Only start a new run after confirming the previous process has ended or failed.

After finding the right device and starting `tmux`, then you can try to run the requested shell scripts or python scripts. 

Before running any scripts, you need to check if you are working in the correct directory, otherwise you need to switch to the right one.
And please also check which conda and python environment you are using, make sure they are following the task instructions.
For example, in this project, you will use the following commands most probably:
```bash
# Switch to the working directory
cd /dtu/3d-imaging-center/projects/2022_DANFIX_08_BioComFert/raw_data_extern/Chengpeng/PycharmProjects/PhaseNet

# Activate the conda environment
. /zhome/64/c/214423/projects/BioComfert/codes/Tomocupy/Job_Scripts/miniconda_chengpeng.sh
conda activate dl_torch
```

If the "OOM (Out-of-memory)" error happens, you just need to switch to another device and try again.

If the error is caused by still being on a login node, such as `nvidia-smi: command not found` or no GPU devices visible before entering `a100sh`/`sxm2sh`/`voltash`, switch to a GPU node and retry the GPU check. This is not a fatal experiment error.

For all other errors, stop the task immediately. Report the full command, the full error/traceback, and a short cause analysis. Do not continue with alternative commands unless the task or user explicitly allows it.

When the script is finished, you need to check all the output and log data, and analyze if they are consistent with your expectation.
Sometimes, the connection to GPU devices will be interrupted and you just need to try again by running the command:
```bash
# 2x A100 GPU (40 GB memory)
a100sh

# 4x V100 GPU (32GB memory)
sxm2sh

# 2x V100 GPU (16 GB mempry)
voltash
```
Finially, you need to update the RECORDS.md with the experiment description, results, and conclusions.
The last thing is to git commit and push the latest repo to the remote repo.
