#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/o160-h3_3-msg-torch-swa.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
# export WANDB_NAME="o160-h3_3-gpu4-bs2-acc2-r1-swa"
# export WANDB_NOTES="TEST: SWA + gradient accumulation"

# generic settings
CONDA_ENV=gnn-pyg-2.3
GITDIR=/perm/pamc/software/aifs
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
export WANDB_NAME="learnable-embedding-2"
srun aifs-train --config $GITDIR/aifs/config/atos.yaml
