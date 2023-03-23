# gnn-era5
GNN for the ERA5 dataset

## Pre-commit Etiquette

Until we set up pre-commit hooks, please format your code using `black` before committing changes. Just run `make black` in this folder:

```shell
$> cd <... your local clone dir ...> 
$> make black
find . -type f -name "*.py" | xargs black -l 132
All done!
27 files left unchanged.
```

