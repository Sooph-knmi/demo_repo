# AIFS

GNN for for weather prediction trained on the ERA5 dataset

## Installation

We have the `setup.py` set up for three use-cases currently:

1. Inference
2. Training
3. Graph Surgery

You can install the smallest set of dependencies via

```
pip install .
```

for training, you need to install

```
pip install .[training]
```

Graph shenanigans need to have some extra packages here:
```
pip install .[graph]
```

## Pre-commit Etiquette

We are using pre-commit hooks. You can find the config in `.pre-commit-config.yaml`, which automatically format new code and check with tools like `black` and `flake8`.

When you first set up this repo, run:

```
pre-commit install
```

To enable these code formatters.

**Please don't push changes directly to `master`**. Instead, PR changes from your own branch into `origin/master` so they get peer-reviewed.

## How to run

```shell
$> cd <... your local clone dir ...>
$> pip install -e .
# this creates entry points for training and prediction
$> aifs-train
```
