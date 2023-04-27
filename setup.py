from setuptools import find_packages, setup

setup(
    name="aifs",
    version="0.6",
    url="https://github.com/ecmwf-lab/aifs-model",
    license="Apache Lincense Version 2.0",
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="ecmwf-authors@ecmwf.int",
    description="ERA5 forecasting with Graph Neural Networks",
    install_requires=[
        "torch_geometric==2.1.0",
        "torch-sparse=0.6.17"
        "pytorch-lightning==2.0.2",
    ],
    extras_require = {'extras': [
        "matplotlib==3.7.1",
        "cartopy==0.21.1",
        "tqdm==4.65.0",
        "wandb==0.15.0",
        "einops==0.6.1",
        "zarr==2.14.2",
        "networkx==3.1",
        "h3==4.1.0"
    ]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "aifs-train=aifs.train.train:main",
            "aifs-predict=aifs.predict.predict:main",
            "aifs-dltest=aifs.data.dltest:main",
        ]
    },
)
