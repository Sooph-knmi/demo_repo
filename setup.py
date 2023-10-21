from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def read(fname):
    file_path = Path(Path(__file__).parent, fname)
    return open(file_path, encoding="utf-8").read()


version = None
for line in read(Path("aifs", "__init__.py")).split("\n"):
    if line.startswith("__version__"):
        version = line.split("=")[-1].strip()[1:-1]

assert version

setup(
    name="aifs",
    version=version,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache Lincense Version 2.0",
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="ecmwf-authors@ecmwf.int",
    description="ERA5 forecasting with Graph Neural Networks",
    url="https://github.com/ecmwf-lab/aifs-mono",
    install_requires=[
        "torch==2.0.1",
        "torch_geometric>=2.3.1",
        "einops>=0.6.1",
    ],
    extras_require={
        "training": [
            "pytorch-lightning==2.0.8",
            "timm>=0.9.2",
            "hydra-core>=1.3",
            "matplotlib>=3.7.1",
            "tqdm>=4.65.0",
            "wandb>=0.15.0",
            "zarr>=2.14.2",
            "pre-commit>=3.3.3",
        ],
        "graph": [
            "networkx>=3.1",
            "h3>=3.7.6",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    package_data={"": ["continents.json"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "aifs-train=aifs.train.train:main",
        ]
    },
)
