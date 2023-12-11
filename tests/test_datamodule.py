import pytest
import torch
from hydra import compose
from hydra import initialize

from aifs.data.dataindices import BaseIndex
from aifs.data.dataindices import BaseTensorIndex
from aifs.data.dataindices import IndexCollection
from aifs.data.datamodule import ECMLDataModule


@pytest.fixture
def datamodule():
    with initialize(version_base=None, config_path="../aifs/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return ECMLDataModule(cfg)


def test_datamodule_datasets(datamodule):
    assert hasattr(datamodule, "ds_train")
    assert hasattr(datamodule, "ds_valid")
    assert hasattr(datamodule, "ds_test")


def test_datamodule_dataloaders(datamodule):
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")


def test_datamodule_metadata(datamodule):
    assert hasattr(datamodule, "metadata")
    assert isinstance(datamodule.metadata, dict)


def test_datamodule_statistics(datamodule):
    assert hasattr(datamodule, "statistics")
    assert isinstance(datamodule.statistics, dict)
    assert "mean" in datamodule.statistics
    assert "stdev" in datamodule.statistics
    assert "minimum" in datamodule.statistics
    assert "maximum" in datamodule.statistics


@pytest.mark.parametrize(
    "data_model,in_out,full_only_prognostic",
    [(a, b, c) for a in ["data", "model"] for b in ["input", "output"] for c in ["full", "forcing", "diagnostic", "prognostic"]],
)
def test_datamodule_api(datamodule, data_model, in_out, full_only_prognostic):
    assert hasattr(datamodule, "data_indices")
    assert isinstance(datamodule.data_indices, IndexCollection)
    assert hasattr(datamodule.data_indices, data_model)
    assert isinstance(datamodule.data_indices[data_model], BaseIndex)
    data_indices = getattr(datamodule.data_indices, data_model)
    assert isinstance(getattr(data_indices, in_out), BaseTensorIndex)
    assert hasattr(getattr(data_indices, in_out), full_only_prognostic)
    assert isinstance(getattr(getattr(data_indices, in_out), full_only_prognostic), torch.Tensor)


def test_datamodule_data_indices(datamodule):
    # Check that different indices are split correctly
    all_data = set(datamodule.data_indices.data.input.name_to_index.values())
    assert (
        set(datamodule.data_indices.data.input.full.numpy()).union(
            datamodule.data_indices.data.input.name_to_index[v] for v in datamodule.config.data.diagnostic
        )
        == all_data
    )
    assert len(datamodule.data_indices.data.input.prognostic) <= len(datamodule.data_indices.data.input.full)
    assert len(datamodule.data_indices.data.output.prognostic) <= len(datamodule.data_indices.data.output.full)
    assert len(datamodule.data_indices.data.output.prognostic) == len(datamodule.data_indices.data.input.prognostic)

    assert len(datamodule.data_indices.model.input.prognostic) <= len(datamodule.data_indices.model.input.full)
    assert len(datamodule.data_indices.model.output.prognostic) <= len(datamodule.data_indices.model.output.full)
    assert len(datamodule.data_indices.model.output.prognostic) == len(datamodule.data_indices.model.input.prognostic)


def test_datamodule_batch(datamodule):
    first_batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(first_batch, torch.Tensor)
    assert first_batch.shape[-1] == len(
        datamodule.data_indices.data.input.name_to_index.values()
    ), "Batch should have all variables"
    assert first_batch.shape[0] == datamodule.config.dataloader.batch_size.training, "Batch should have correct batch size"
    assert first_batch.shape[1] == datamodule.config.training.multistep_input + 1, "Batch needs correct sequence length (steps + 1)"
