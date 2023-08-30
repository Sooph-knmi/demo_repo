from setuptools import find_packages
from setuptools import setup

setup(
    name="aifs",
    version="0.6",
    url="https://github.com/ecmwf-lab/aifs-model",
    license="Apache Lincense Version 2.0",
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="ecmwf-authors@ecmwf.int",
    description="ERA5 forecasting with Graph Neural Networks",
    install_requires=[
        "torch_geometric==2.3.1",
        "pytorch-lightning==2.0.7",
        "timm==0.9.2",
        "hydra-core==1.3",
        "einops==0.6.1",
    ],
    extras_require={
        "extras": [
            "matplotlib==3.7.1",
            "tqdm==4.65.0",
            "wandb==0.15.0",
            "zarr==2.14.2",
            "networkx==3.1",
            "h3==3.7.6",
            "pre-commit==3.3.3",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "aifs-ens-train=aifs.train.train:main",
        ]
    },
)
