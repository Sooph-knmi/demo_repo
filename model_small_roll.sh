#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclude=ac6-307
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=model_small.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo


# generic settings
CONDA_ENV=aifs_dev
GITDIR=/perm/momc/AIFS/aifs-mono
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
export WANDB_RESUME='must'
export WANDB_RUN_ID='ens-patched-energy-roll'
srun aifs-ens-train --config-name=ens-dmp-h4-patched-roll
