import pytest
import torch

from aifs.data.dataindices import IndexCollection
from aifs.train.forecaster import GraphForecaster
from aifs.utils.config import DotConfig


@pytest.fixture
def fake_data():
    config = DotConfig(
        {
            "data": {
                "forcing": ["x"],
                "diagnostic": ["z", "q"],
            },
            "training": {
                "loss_scaling": {
                    "default": 1,
                    "sfc": {
                        "z": 0.1,
                        "other": 100,
                    },
                    "pl": {"y": 0.5},
                },
                "metrics": ["other", "y_850"],
            },
        }
    )
    name_to_index = {"x": 0, "y_50": 1, "y_500": 2, "y_850": 3, "z": 5, "q": 4, "other": 6}
    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    return config, data_indices


def test_loss_scaling_vals(fake_data):
    config, data_indices = fake_data

    _, loss_scaling = GraphForecaster.metrics_loss_scaling(config, data_indices)
    expected_scaling = torch.Tensor(
        [
            50 / 1000 * 0.5,  # y_50
            500 / 1000 * 0.5,  # y_500
            850 / 1000 * 0.5,  # y_850
            1,  # q
            0.1,  # z
            100,  # other
        ]
    ).to(torch.float32)

    assert torch.allclose(loss_scaling, expected_scaling)


def test_metric_range(fake_data):
    config, data_indices = fake_data

    metric_range, _ = GraphForecaster.metrics_loss_scaling(config, data_indices)

    expected_metric_range = {
        "pl_y": [
            data_indices.model.output.name_to_index["y_50"],
            data_indices.model.output.name_to_index["y_500"],
            data_indices.model.output.name_to_index["y_850"],
        ],
        "sfc_other": [data_indices.model.output.name_to_index["other"]],
        "sfc_q": [data_indices.model.output.name_to_index["q"]],
        "sfc_z": [data_indices.model.output.name_to_index["z"]],
        "other": [data_indices.model.output.name_to_index["other"]],
        "y_850": [data_indices.model.output.name_to_index["y_850"]],
    }

    assert dict(metric_range) == expected_metric_range
