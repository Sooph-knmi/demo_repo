#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:59:59
#SBATCH --account=ecaifs
#SBATCH --output=outputs/aifs-mono.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

export SLURM_GPUS_PER_NODE=4
source pyenv/bin/activate

srun aifs-train hardware=atos_slurm
