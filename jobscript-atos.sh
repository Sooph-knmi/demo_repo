#!/bin/bash

#SBATCH --account=ecaifs
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=ens-kcrps-mp.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=TRACE
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="ens-kcrps-mpar"
export WANDB_NOTES="KCRPS optimized ensemble forecasting (updated code)"

# generic settings
CONDA_ENV=gnn-pyg-2.3
GITDIR=/home/syma/GNN/gnn-era5.git
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun aifs-ens-train hardware=atos_slurm --config-name=ens-ddmp
