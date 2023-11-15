#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=debug.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="ens-patched-h4"
export WANDB_NOTES="patched ensemble"

# generic settings
CONDA_ENV=aifs_dev
GITDIR=/home/momc/AIFS/aifs-mono
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV

wandb offline
srun aifs-ens-train hardware=atos_slurm --config-name=debug-ens-dmp-h4
