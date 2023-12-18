import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from aifs.data.dataindices import IndexCollection
from aifs.train.forecaster import GraphForecaster
from aifs.utils.config import DotConfig


@pytest.fixture
def data_indices(config):
    config = DotConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
            }
        }
    )
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    return IndexCollection(config=config, name_to_index=name_to_index)


@pytest.fixture
def config():
    config = DictConfig(
        {
            "hardware": {
                "paths": {"graph": "/home/mlx/data/graphs/"},
                "files": {"graph": "graph_mappings_normed_edge_attrs_ordered_desc_lat_lon_20231122093910_o96_h_0_1_2_3_4.pt"},
                "num_nodes": 1,
                "num_gpus_per_node": 1,
                "num_gpus_per_model": 1,
            },
            "diagnostics": {"log": {"wandb": {"enabled": True}}, "plot": {"enabled": True}},
            "training": {
                "loss_scaling": {"default": 1, "sfc": {"z": 0.1, "other": 100}, "pl": {"y": 0.5}},
                "metrics": ["other", "y_850"],
                "loss_gradient_scaling": True,
                "multistep_input": 3,
                "lr": {"rate": 0.001, "iterations": 1000, "min": 0.0001},
                "rollout": {"start": 5, "epoch_increment": 2, "max": 10},
                "zero_optimizer": False,
            },
            "data": {
                "normalizer": {"default": "mean-std"},
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
            },
            "model": {
                "activation": "SiLU",
                "trainable_parameters": {"era": 8, "hidden": 8, "era2hidden": 8, "hidden2era": 8, "hidden2hidden": 8},
                "num_channels": 128,
                "mlp": {"extra_layers": 1, "hidden": 128, "dropout": 0.0},
                "encoder": {"num_chunks": 1},
                "decoder": {"num_chunks": 1},
                "processor": {"num_layers": 16, "chunks": 2},
            },
        }
    )
    return config


@pytest.fixture
def statistics():
    return {
        "mean": np.array([1.0, 2.0, 3.0, 4.5, 3.0]),
        "stdev": np.array([0.5, 0.5, 0.5, 1, 14]),
        "minimum": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "maximum": np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    }


@pytest.fixture
def dummy_forecaster(config, data_indices, statistics):
    return GraphForecaster(config=config, data_indices=data_indices, statistics=statistics, metadata=None)


@pytest.fixture
def fake_data(data_indices, config):
    return torch.randn(
        4, config.training.multistep_input + config.training.rollout.max, 64, len(data_indices.data.input.name_to_index)
    )


def test_advance_input(dummy_forecaster, config, fake_data):
    """Checks the advance_input function to correctly roll the input tensor.

    This current checks the shape and the data content of the input tensor

    Parameters
    ----------
    dummy_forecaster : _type_
        _description_
    config : _type_
        _description_
    fake_data : _type_
        _description_
    """
    multi_step = config.training.multistep_input
    x = fake_data[:, :multi_step, ..., dummy_forecaster.data_indices.data.input.full]  # example input tensor
    for rollout_step in range(config.training.rollout.start):
        # Generate a "predicted" tensor
        y_pred = fake_data[:, multi_step + rollout_step, ..., dummy_forecaster.data_indices.data.output.full]
        # Generate a "forcing" tensor
        fake_data[:, multi_step + rollout_step, ..., dummy_forecaster.data_indices.data.input.forcing]
        # Run the function
        x = dummy_forecaster.advance_input(x, y_pred, fake_data, rollout_step)
        # Check the shape
        assert x.shape == (
            fake_data.shape[0],
            multi_step,
            fake_data.shape[2],
            len(dummy_forecaster.data_indices.data.input.full),
        ), f"Advance_input shape doesn't match in rollout step: {rollout_step}"
        # Check the data content
        assert torch.allclose(
            fake_data[:, rollout_step + 1 : rollout_step + multi_step + 1, ..., dummy_forecaster.data_indices.data.input.full], x
        ), f"Advance_input data doesn't match in rollout step: {rollout_step}"


def test_set_model_comm_group(dummy_forecaster):
    model_comm_group = "example_model_comm_group"
    dummy_forecaster.set_model_comm_group(model_comm_group)
    assert dummy_forecaster.model_comm_group == model_comm_group  # check if model_comm_group is set correctly
