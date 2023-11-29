#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1 #2
#SBATCH --ntasks-per-node=2 #4 #2 #1 #4
#SBATCH --gpus-per-node=2 #4 #2 #1 #4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G #256G
#SBATCH --time=48:00:00
#SBATCH --account=ecaifs
#SBATCH --output=outputs/some_stuff.out.%j

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
CONDA_ENV=xformers-dev-ens
GITDIR=/home/nesl/res/ai/aifs-mono-ens/
WORKDIR=$GITDIR

cd $WORKDIR
module load conda
conda activate $CONDA_ENV

srun aifs-ens-gradtest  hardware=atos_slurm hardware.num_gpus_per_ensemble=2 --config-name=gradtest

# ValueError: You set `devices=2` in Lightning, but the number of tasks per node configured in SLURM `--ntasks-per-node=1` does not match. HINT: Set `devices=1`.