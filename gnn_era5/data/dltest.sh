#!/bin/bash -x

#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=dltest.out.%j

# generic settings
CONDA_ENV=gnn-pyg-2.1
GITDIR=/home/syma/dask/codes/gnn-era5
WORKDIR=/home/syma/dask/codes/gnn-era5

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun gnn-era-dltest --config $GITDIR/gnn_era5/config/atos.yaml
