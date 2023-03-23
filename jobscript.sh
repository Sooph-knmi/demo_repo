#!/bin/bash

#SBATCH --qos=ng
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=o160-unet-neptune.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
# export WANDB_NAME="mihai-o160-unet-gpu4-bs8"
# export WANDB_NOTES="o160 ddp test run"

# generic settings
CONDA_ENV=gnn-pyg-2.1
GITDIR=/home/syma/dask/codes/gnn-era5
WORKDIR=/home/syma/dask/codes/gnn-era5

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun gnn-era-train --config $GITDIR/gnn_era5/config/atos.yaml
