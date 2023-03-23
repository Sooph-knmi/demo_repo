from setuptools import find_packages, setup

setup(
    name="gnn_era5",
    version="0.6",
    url="",
    license="MIT License",
    company="ECMWF",
    author_email="ecmwf-authors@ecmwf.int",
    description="ERA5 forecasting with Graph Neural Networks",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "gnn-era-train=gnn_era5.train.train:main",
            "gnn-era-predict=gnn_era5.predict.predict:main",
            "gnn-era-dltest=gnn_era5.data.dltest:main",
        ]
    },
)
