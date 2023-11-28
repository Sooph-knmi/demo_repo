# AIFS

Ensemble forecasting with the AIFS.

## Installation

We have the `setup.py` set up for three use-cases currently:

1. Inference
2. Training
3. Graph construction
4. Profiling

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

For profiling, you also need a couple extra packages:
```
pip install .[profiling]
```

## Pre-commit Etiquette

We are using pre-commit hooks. You can find the config in `.pre-commit-config.yaml`, which automatically format new code and check with tools like `black` and `flake8`.

When you first set up this repo, run:

```
pre-commit install
```

to enable these code formatters.

## How to run

```shell
$> cd <... your local clone dir ...>
$> pip install -e .

# training, works the same as before
$> aifs-ens-train hardware=[atos|leo|mlux]-slurm --config-name=...

# all tests run on single nodes only (so up to 4 devices on Atos, Leo and MeluXina; Lumi has 8 GPUs/node)

# model gradient test (can enable various configurations from very simple to more complex, see the code)
$> srun gradtest-mlux.sh

# gradient test for the collective comunication functions
$> srun commstest-mlux.sh

# "step-test": forward + backward + optimizer step
# use this to look at how the parameters change in various device configurations
$> srun steptest-mlux.sh
```
