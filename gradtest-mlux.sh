#!/bin/bash -l

#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --time=00:10:00
#SBATCH --account=p200177
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=gradtest.out.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
# export WANDB_NAME="gradtest-meluxina"
# export WANDB_NOTES="Gradtest on MeluXina"

# generic settings
VENV=aifs-dev
GITDIR=/project/home/p200177/syma/aifs-code/aifs-mono
WORKDIR=$GITDIR

# on MeluXina, CUDA_VISIBLE_DEVICES needs to be set manually (!!)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

cd $WORKDIR

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7
module load cuDNN/8.4.1.50-CUDA-11.7.0
source /project/home/p200177/syma/aifs-env/$VENV/bin/activate

# srun aifs-ens-gradtest hardware=mlux_slurm --config-name=gradtest

aifs-ens-gradtest
