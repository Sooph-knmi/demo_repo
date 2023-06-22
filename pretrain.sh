#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --account=ecaifs
#SBATCH --output=gan-pretrain-critic-6h.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="gan-pretrain-critic-6h"
export WANDB_NOTES="Graph-GAN - pretrain critic on 6-hourly data"

# generic settings
CONDA_ENV=gnn-pyg-2.3
GITDIR=/home/syma/GNN/gnn-era5.git
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun aifs-gan-pretrain hardware=atos_slurm --config-name=gan
