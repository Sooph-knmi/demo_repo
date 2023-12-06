#!/bin/bash

#SBATCH -A DestE_340
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=aifs-ens-kcrps-mp-h5-part2.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="2 gpu leo"
export WANDB_NOTES="2 gpu leo"

# generic settings
CONDA_ENV=aifs-dev
GITDIR=/leonardo/home/userexternal/mclare00/aifs-mono
WORKDIR=$GITDIR

cd $WORKDIR
source /leonardo/home/userexternal/mclare00/aifs_dev_mc/bin/activate
wandb offline
srun aifs-ens-train hardware=leo_slurm --config-name=ens-kcrps-h4
