#!/bin/bash

#SBATCH --account=ecaifs
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=new-ens-test-h4-atos.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="ens-kcrps-mpar-test-atos"
export WANDB_NOTES="KCRPS ensemble - quick test on Atos"

# generic settings
VENV=aifs-ens-score
GITDIR=/home/$USER/AIFS/aifs-ens-score/aifs-mono
WORKDIR=$GITDIR

cd $WORKDIR

module load python3/may23
source /perm/$USER/venvs/shared/$VENV/bin/activate

srun aifs-ens-train hardware=atos_slurm --config-name=ens-kcrps-h4
