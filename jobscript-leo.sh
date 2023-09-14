#!/bin/bash

#SBATCH -A DestE_340
#SBATCH -p boost_usr_prod
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=aifs-ens-kcrps-mp-h5-part2.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="ens-kcrps-mpar-h5"
export WANDB_NOTES="Model-parallel kcrps (H5, ens size = 8)"

# generic settings
CONDA_ENV=aifs-dev
GITDIR=/leonardo/home/userexternal/malexe00/aifs/aifs-mono
WORKDIR=$GITDIR

cd $WORKDIR
module load python
source $WORK/AIFS_env/aifs-dev/bin/activate
srun aifs-ens-train hardware=leo_slurm --config-name=ens-large-dmp-h5
