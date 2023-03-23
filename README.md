# gnn-era5

GNN for the ERA5 dataset

## Pre-commit Etiquette

Until we've set up pre-commit hooks, please auto-format your code using `black` _before_ committing changes. Just run `make black` in this folder:

```shell
$> cd <... your local clone dir ...> 
$> make black
find . -type f -name "*.py" | xargs black -l 132
All done!
27 files left unchanged.
```

**Please don't push changes directly to `master`**. Instead, PR changes from your own branch into `origin/master` so they get peer-reviewed.

## How to run

```shell
$> cd <... your local clone dir ...>
$> pip install -e .
# this creates entry points for training and prediction
# see gnn-era5/config/atos.yaml for an example o160 input configuration
$> gnn-era-train --config <path-to-config-file>
```

