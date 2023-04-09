#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=o160-h3_3-msg-torch-swa.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="o160-h3_3-gpu4-bs2-acc2-r1-swa"
export WANDB_NOTES="TEST: SWA + gradient accumulation"

# generic settings
CONDA_ENV=gnn-pyg-2.3
GITDIR=/home/syma/GNN/gnn-era5.git
WORKDIR=/home/syma/GNN/gnn-era5.git

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun gnn-era-train --config $GITDIR/gnn_era5/config/atos.yaml
