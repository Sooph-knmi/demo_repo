#!/bin/bash

#SBATCH --account=ecaifs
#SBATCH --qos=ng
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=ens-kcrps-h5-atos-roll28.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="ens-kcrps-h5-atos-roll28"
export WANDB_NOTES="KCRPS ensemble (reweighted loss, rollout 2-8)"

# generic settings
CONDA_ENV=aifs-ens-dev
GITDIR=/home/syma/GNN/aifs-temp/aifs-mono
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun aifs-ens-train hardware=atos_slurm --config-name=ens-dmp-h5-roll28