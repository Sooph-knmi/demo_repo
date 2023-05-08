#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=aifs-multistep-test.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
# export WANDB_NAME="aifs-multistep-test"
# export WANDB_NOTES="TEST: multistep input"

# generic settings
CONDA_ENV=gnn-pyg-2.3
GITDIR=/home/syma/GNN/gnn-era5.git
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun aifs-train --config $GITDIR/aifs/config/atos.yaml
