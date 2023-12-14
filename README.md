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

Graph shenanigans (like running the graph generation notebooks) need to have some extra packages here:
```
pip install .[graph]
```

And the profiler uses some additional libraries for parsing the output.
```
pip install .[profile]
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

## How to test
We have written tests using the `pytest` functional interface.

They're stored in the tests/ directory. After installing `pytest` (`pip install pytest`) you can simply run

```shell
$> pytest
```

or if you just want to run a specific file, run:

```shell
$> pytest tests/test_<file>.py
```

Be aware that some tests like the `test_gnn.py` run a singular forward pass, which can be slow on CPU and runs better on GPU.

## How to Profile

We wrote a special profiler that uses Pytorch, Lightning, and memray to measure the performance of the code in it's current training state. Run

```shell
$> aifs-profile
```

This starts a short training run and creates different information:

- Time Profile: Duration of different operations
- Speed Profile: Throughput of dataloader and model
- Memory Profile: Memory of the "worst offenders"
- System Utilization: Overall system utilization (needs W&B online)
