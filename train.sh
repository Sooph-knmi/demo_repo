#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --account=ecaifs
#SBATCH --output=gan-train-o96-h4-v2.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="gan-train-o96-h4-v2"
export WANDB_NOTES="Like gan-train-o96-h4 but with different loss hyperparameters and start LR; pred ens size = 7; newer (better?) critic checkpoint"

# generic settings
CONDA_ENV=gnn-pyg-2.3
GITDIR=/home/syma/GNN/gnn-era5.git
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun aifs-gan-train hardware=atos_slurm --config-name=gan
