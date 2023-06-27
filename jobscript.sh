#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --account=ecaifs
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
CONDA_ENV=aifs-dev
GITDIR=/perm/madj/software/aifs-dev
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun aifs-train hardware=atos_slurm
