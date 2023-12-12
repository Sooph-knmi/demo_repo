from pathlib import Path

import pytest
import torch
from hydra import compose
from hydra import initialize

from aifs.data.datamodule import ECMLDataModule
from aifs.model.gnn import GraphMSG


# This example drives some user logic with the composed config.
# In this case it calls hydra_app.main.add(), passing it the composed config.


@pytest.fixture()
def config(request):
    overrides = request.param
    with initialize(version_base=None, config_path="../aifs/config"):
        # config is relative to a module
        config = compose(config_name="debug", overrides=overrides)
    return config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU or accelerator to run")
@pytest.mark.parametrize(
    "config",
    [
        [],
        ["data.forcing=[]"],
        ["data.forcing=null"],
        ["data.forcing=[lsm]"],
        ["data.diagnostic=[]"],
        ["data.diagnostic=null"],
        ["data.diagnostic=[msl]"],
        ["data.forcing=null", "data.diagnostic=null"],
        ["data.forcing=[]", "data.diagnostic=[]"],
        ["data.forcing=[lsm]", "data.diagnostic=[]"],
        ["data.forcing=[]", "data.diagnostic=[lsm]"],
        ["data.forcing=[lsm]", "data.diagnostic=[tp]"],
    ],
    indirect=True,
)
def test_graph_msg(config):
    device = torch.device("cuda")
    data_module = ECMLDataModule(config)
    data_indices = data_module.data_indices
    config.data.num_features = len(data_module.ds_train.data.variables)
    graph_data = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph))
    model = GraphMSG(
        config=config,
        data_indices=data_indices,
        graph_data=graph_data,
    ).to(device)

    input = torch.randn(
        config.dataloader.batch_size.training,
        config.training.multistep_input,
        40320,
        len(data_indices.model.input.full),
        dtype=torch.float32,
        device=device,
    )
    output = torch.randn([config.dataloader.batch_size.training, 40320, len(data_indices.model.output.full)])
    assert model.forward(input).shape == output.shape, "Output shape is not correct"
