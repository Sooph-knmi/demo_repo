import pytest
import torch
from hydra import compose
from hydra import initialize

from aifs.data.dataindices import BaseIndex
from aifs.data.dataindices import BaseTensorIndex
from aifs.data.dataindices import DataIndex
from aifs.data.dataindices import IndexCollection
from aifs.data.dataindices import InputTensorIndex
from aifs.data.dataindices import OutputTensorIndex
from aifs.data.datamodule import ECMLDataModule


@pytest.fixture
def datamodule():
    with initialize(version_base=None, config_path="../aifs/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return ECMLDataModule(cfg)


@pytest.mark.parametrize(
    "data_model,in_out,full_only_prognostic",
    [(a, b, c) for a in ["data", "model"] for b in ["input", "output"] for c in ["full", "forcing", "diagnostic", "prognostic"]],
)
def test_dataindex_types(datamodule, data_model, in_out, full_only_prognostic):
    assert hasattr(datamodule, "data_indices")
    assert isinstance(datamodule.data_indices, IndexCollection)
    data_indices = datamodule.data_indices

    assert hasattr(data_indices, data_model)
    assert isinstance(getattr(data_indices, data_model), BaseIndex)
    assert hasattr(getattr(data_indices, data_model), in_out)
    assert isinstance(getattr(getattr(data_indices, data_model), in_out), BaseTensorIndex)
    assert hasattr(getattr(getattr(data_indices, data_model), in_out), full_only_prognostic)
    assert isinstance(getattr(getattr(getattr(data_indices, data_model), in_out), full_only_prognostic), torch.Tensor)


@pytest.fixture
def fake_data():
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    forcing = ["x", "y"]
    diagnostic = ["z"]
    return forcing, diagnostic, name_to_index


@pytest.fixture
def input_tensor_index(fake_data):
    forcing, diagnostic, name_to_index = fake_data
    return InputTensorIndex(includes=forcing, excludes=diagnostic, name_to_index=name_to_index)


@pytest.fixture
def output_tensor_index(fake_data):
    forcing, diagnostic, name_to_index = fake_data
    return OutputTensorIndex(includes=diagnostic, excludes=forcing, name_to_index=name_to_index)


def test_dataindex_init(fake_data, input_tensor_index, output_tensor_index):
    forcing, diagnostic, name_to_index = fake_data
    data_index = DataIndex(forcing=forcing, diagnostic=diagnostic, name_to_index=name_to_index)
    print(data_index.input, input_tensor_index)
    print(data_index.output, output_tensor_index)
    assert data_index.input == input_tensor_index
    assert data_index.output == output_tensor_index


def test_output_tensor_index_full(output_tensor_index):
    expected_output = torch.Tensor([2, 3, 4]).to(torch.int)
    assert torch.allclose(output_tensor_index.full, expected_output)


def test_output_tensor_index_only(output_tensor_index):
    expected_output = torch.Tensor([2]).to(torch.int)
    assert torch.allclose(output_tensor_index._only, expected_output)


def test_output_tensor_index_prognostic(output_tensor_index):
    expected_output = torch.Tensor([3, 4]).to(torch.int)
    assert torch.allclose(output_tensor_index.prognostic, expected_output)


def test_output_tensor_index_todict(output_tensor_index):
    expected_output = {
        "full": torch.Tensor([2, 3, 4]).to(torch.int),
        "diagnostic": torch.Tensor([2]).to(torch.int),
        "forcing": torch.Tensor([0, 1]).to(torch.int),
        "prognostic": torch.Tensor([3, 4]).to(torch.int),
    }
    for key, value in output_tensor_index.todict().items():
        assert key in expected_output
        assert torch.allclose(value, expected_output[key])


def test_output_tensor_index_getattr(output_tensor_index):
    assert output_tensor_index.full is not None
    with pytest.raises(AttributeError):
        output_tensor_index.z


def test_output_tensor_index_build_idx_from_excludes(output_tensor_index):
    expected_output = torch.Tensor([2, 3, 4]).to(torch.int)
    assert torch.allclose(output_tensor_index._build_idx_from_excludes(), expected_output)


def test_output_tensor_index_build_idx_from_includes(output_tensor_index):
    expected_output = torch.Tensor([2]).to(torch.int)
    assert torch.allclose(output_tensor_index._build_idx_from_includes(), expected_output)


def test_output_tensor_index_build_idx_prognostic(output_tensor_index):
    expected_output = torch.Tensor([3, 4]).to(torch.int)
    assert torch.allclose(output_tensor_index._build_idx_prognostic(), expected_output)


def test_input_tensor_index_full(input_tensor_index):
    expected_output = torch.Tensor([0, 1, 3, 4]).to(torch.int)
    assert torch.allclose(input_tensor_index.full, expected_output)


def test_input_tensor_index_only(input_tensor_index):
    expected_output = torch.Tensor([0, 1]).to(torch.int)
    assert torch.allclose(input_tensor_index._only, expected_output)


def test_input_tensor_index_prognostic(input_tensor_index):
    expected_output = torch.Tensor([3, 4]).to(torch.int)
    assert torch.allclose(input_tensor_index.prognostic, expected_output)


def test_input_tensor_index_todict(input_tensor_index):
    expected_output = {
        "full": torch.Tensor([0, 1, 3, 4]).to(torch.int),
        "diagnostic": torch.Tensor([2]).to(torch.int),
        "forcing": torch.Tensor([0, 1]).to(torch.int),
        "prognostic": torch.Tensor([3, 4]).to(torch.int),
    }
    for key, value in input_tensor_index.todict().items():
        assert key in expected_output
        assert torch.allclose(value, expected_output[key])


def test_input_tensor_index_getattr(input_tensor_index):
    assert input_tensor_index.full is not None
    with pytest.raises(AttributeError):
        input_tensor_index.z


def test_input_tensor_index_build_idx_from_excludes(input_tensor_index):
    expected_output = torch.Tensor([0, 1, 3, 4]).to(torch.int)
    assert torch.allclose(input_tensor_index._build_idx_from_excludes(), expected_output)


def test_input_tensor_index_build_idx_from_includes(input_tensor_index):
    expected_output = torch.Tensor([0, 1]).to(torch.int)
    assert torch.allclose(input_tensor_index._build_idx_from_includes(), expected_output)


def test_input_tensor_index_build_idx_prognostic(input_tensor_index):
    expected_output = torch.Tensor([3, 4]).to(torch.int)
    assert torch.allclose(input_tensor_index._build_idx_prognostic(), expected_output)
