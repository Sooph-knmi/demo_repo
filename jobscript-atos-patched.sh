#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=ens-kcrps-mp-h4-test.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="ens-patched-energy-mpar-mc"
export WANDB_NOTES="Patched energy score optimized ensemble forecasting"


# generic settings
CONDA_ENV=aifs_dev
GITDIR=/perm/momc/AIFS/aifs-mono
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
wandb online
srun aifs-ens-train hardware=atos_slurm --config-name=ens-dmp-h4-patched.yaml
