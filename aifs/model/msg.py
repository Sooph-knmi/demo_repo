from pathlib import Path

import einops
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from aifs.model.layers import MessagePassingMapper
from aifs.model.layers import MessagePassingProcessor
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class GraphMSG(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        config: DictConfig,
        graph_data: HeteroData = None,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData, optional
            Graph definition, by default None
        """
        super().__init__()

        # create mappings
        if graph_data is None:
            self._graph_data = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph))
        else:
            self._graph_data = graph_data

        self.in_channels = config.data.num_features - config.data.num_aux_features
        self.multi_step = config.training.multistep_input
        self.aux_in_channels = config.data.num_aux_features

        LOGGER.debug("self.in_channels + self.aux_channels == %d", self.in_channels + self.aux_in_channels)

        self.activation = config.model.activation

        self.register_buffer("e2e_edge_index", self._graph_data[("era", "to", "era")].edge_index, persistent=False)
        self.register_buffer("h2e_edge_index", self._graph_data[("h", "to", "era")].edge_index, persistent=False)
        self.register_buffer("e2h_edge_index", self._graph_data[("era", "to", "h")].edge_index, persistent=False)
        self.register_buffer("h2h_edge_index", self._graph_data[("h", "to", "h")].edge_index, persistent=False)

        self.register_buffer("e2e_edge_attr", self._graph_data[("era", "to", "era")].edge_attr, persistent=False)
        self.register_buffer("h2e_edge_attr", self._graph_data[("h", "to", "era")].edge_attr, persistent=False)
        self.register_buffer("e2h_edge_attr", self._graph_data[("era", "to", "h")].edge_attr, persistent=False)
        self.register_buffer("h2h_edge_attr", self._graph_data[("h", "to", "h")].edge_attr, persistent=False)

        self._era_size = self._graph_data[("era", "to", "era")].ecoords_rad.shape[0]
        self._h_size = self._graph_data[("h", "to", "h")].hcoords_rad.shape[0]

        self.era_trainable_size = config.model.trainable_parameters.era
        self.era_trainable = (
            nn.Parameter(torch.zeros(self._era_size, self.era_trainable_size)) if self.era_trainable_size > 0 else None
        )

        self.h_trainable_size = config.model.trainable_parameters.hidden
        self.h_trainable = nn.Parameter(torch.zeros(self._h_size, self.h_trainable_size)) if self.h_trainable_size > 0 else None

        self.e2h_trainable_size = config.model.trainable_parameters.era2hidden
        self.e2h_trainable = (
            nn.Parameter(
                torch.zeros(
                    self._graph_data[("era", "to", "h")].edge_attr.shape[0],
                    self.e2h_trainable_size,
                )
            )
            if self.e2h_trainable_size > 0
            else None
        )

        self.h2e_trainable_size = config.model.trainable_parameters.hidden2era
        self.h2e_trainable = (
            nn.Parameter(
                torch.zeros(
                    self._graph_data[("h", "to", "era")].edge_attr.shape[0],
                    self.h2e_trainable_size,
                )
            )
            if self.h2e_trainable_size > 0
            else None
        )

        self.h2h_trainable_size = config.model.trainable_parameters.hidden2hidden
        self.h2h_trainable = (
            nn.Parameter(
                torch.zeros(
                    self._graph_data[("h", "to", "h")].edge_attr.shape[0],
                    self.h2h_trainable_size,
                )
            )
            if self.h2h_trainable_size > 0
            else None
        )
        self.register_buffer(
            "_e2h_edge_inc", torch.from_numpy(np.asarray([[self._era_size], [self._h_size]], dtype=np.int64)), persistent=False
        )
        self.register_buffer(
            "_h2e_edge_inc", torch.from_numpy(np.asarray([[self._h_size], [self._era_size]], dtype=np.int64)), persistent=False
        )
        self.register_buffer(
            "_h2h_edge_inc", torch.from_numpy(np.asarray([[self._h_size], [self._h_size]], dtype=np.int64)), persistent=False
        )

        self.register_buffer(
            "era_latlons",
            torch.cat(
                [
                    torch.as_tensor(np.sin(self._graph_data[("era", "to", "era")].ecoords_rad)),
                    torch.as_tensor(np.cos(self._graph_data[("era", "to", "era")].ecoords_rad)),
                ],
                dim=-1,
            ),
            persistent=True,
        )

        self.register_buffer(
            "h_latlons",
            torch.cat(
                [
                    torch.as_tensor(np.sin(self._graph_data[("h", "to", "h")].hcoords_rad)),
                    torch.as_tensor(np.cos(self._graph_data[("h", "to", "h")].hcoords_rad)),
                ],
                dim=-1,
            ),
            persistent=True,
        )

        encoder_out_channels = config.model.num_channels
        mlp_extra_layers = config.model.mlp.extra_layers

        # Encoder from ERA -> H
        self.forward_mapper = MessagePassingMapper(
            in_channels_src=self.multi_step * (self.in_channels + self.aux_in_channels)
            + self.era_latlons.shape[1]
            + self.era_trainable_size,
            in_channels_dst=self.h_latlons.shape[1] + self.h_trainable_size,
            hidden_dim=encoder_out_channels,
            hidden_layers=config.model.encoder.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.e2h_edge_attr.shape[1] + self.e2h_trainable_size,
            chunks=1,
            activation=self.activation,
            ptype="fmapper",
        )

        # Processor H -> H
        self.h_processor = MessagePassingProcessor(
            hidden_dim=encoder_out_channels,
            hidden_layers=config.model.hidden.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
            chunks=2,
            activation=self.activation,
        )

        # Decoder H -> ERA5
        self.backward_mapper = MessagePassingMapper(
            in_channels_src=encoder_out_channels,
            in_channels_dst=encoder_out_channels,
            out_channels_dst=self.in_channels,
            hidden_dim=config.model.num_channels,
            hidden_layers=config.model.decoder.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2e_edge_attr.shape[1] + self.h2e_trainable_size,
            backward_mapper=True,
            chunks=1,
            activation=self.activation,
            ptype="bmapper",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        # add ERA positional info (lat/lon)
        x_in = [
            einops.rearrange(x, "b m n f -> (b n) (m f)"),
            einops.repeat(self.era_latlons, "e f -> (repeat e) f", repeat=bs),
        ]
        if self.era_trainable is not None:
            x_in.append(einops.repeat(self.era_trainable, "e f -> (repeat e) f", repeat=bs))
        x_era_latent = torch.cat(
            x_in,
            dim=-1,  # feature dimension
        )

        x_h_latent = [einops.repeat(self.h_latlons, "e f -> (repeat e) f", repeat=bs)]
        if self.h_trainable is not None:
            x_h_latent.append(einops.repeat(self.h_trainable, "e f -> (repeat e) f", repeat=bs))
        x_h_latent = torch.cat(
            x_h_latent,
            dim=-1,  # feature dimension
        )

        edge_era_to_h_latent = [einops.repeat(self.e2h_edge_attr, "e f -> (repeat e) f", repeat=bs)]
        if self.e2h_trainable is not None:
            edge_era_to_h_latent.append(einops.repeat(self.e2h_trainable, "e f -> (repeat e) f", repeat=bs))
        edge_era_to_h_latent = torch.cat(
            edge_era_to_h_latent,
            dim=-1,  # feature dimension
        )  # copy edge attributes bs times

        (x_era_latent, x_latent) = checkpoint(
            self.forward_mapper,
            (x_era_latent, x_h_latent),
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self.e2h_edge_index + i * self._e2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_era_to_h_latent,
            use_reentrant=False,
        )

        edge_h_to_h_latent = [einops.repeat(self.h2h_edge_attr, "e f -> (repeat e) f", repeat=bs)]
        if self.h2h_trainable is not None:
            edge_h_to_h_latent.append(einops.repeat(self.h2h_trainable, "e f -> (repeat e) f", repeat=bs))
        edge_h_to_h_latent = torch.cat(
            edge_h_to_h_latent,
            dim=-1,  # feature dimension
        )
        x_latent_proc = self.h_processor(  # has skipped connections and checkpoints inside
            x=x_latent,
            edge_index=torch.cat(
                [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_h_to_h_latent,
        )

        # add skip connection (H -> H)
        x_latent_proc = x_latent_proc + x_latent

        edge_h_to_e_latent = [einops.repeat(self.h2e_edge_attr, "e f -> (repeat e) f", repeat=bs)]
        if self.h2e_trainable is not None:
            edge_h_to_e_latent.append(einops.repeat(self.h2e_trainable, "e f -> (repeat e) f", repeat=bs))
        edge_h_to_e_latent = torch.cat(
            edge_h_to_e_latent,
            dim=-1,  # feature dimension
        )
        _, x_out = checkpoint(
            self.backward_mapper,
            x=(x_latent_proc, x_era_latent),
            edge_index=torch.cat(
                [self.h2e_edge_index + i * self._h2e_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_h_to_e_latent,
            use_reentrant=False,
        )

        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=bs)

        # residual connection (just for the predicted variables)
        return x_out + x[:, -1, :, : self.in_channels]

if __name__ == "__main__":
    # you'll need to run this test on a worker node
    import os
    from torch_geometric import seed_everything
    from prettytable import PrettyTable
    from hydra import compose, initialize
    # from torch.profiler import profile, record_function, ProfilerActivity

    from timeit import default_timer as timer

    initialize(config_path="../config", job_name="test_msg")
    config = compose(config_name="config", overrides=[
        'model.trainable_parameters.era=8',
        'model.trainable_parameters.hidden=8',
        'model.trainable_parameters.era2hidden=8',
        'model.trainable_parameters.hidden2era=8',
        'model.trainable_parameters.hidden2hidden=8',
        'model.num_channels=768', #1024',
        'dataloader.batch_size.training=1',
        'dataloader.batch_size.validation=1',
        'data.num_features=98',
        'data.num_aux_features=13',
        'training.multistep_input=2',
        'hardware.paths.graph="/home/mlx/data/graphs/"',
        'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4_5.pt"',
        #'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5_6.pt"',
        # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5.pt"',
    ])
    
    seed_everything(1234)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters_pretty(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params/1.e6}")
        return total_params

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.debug("Running on device: %s ...", device)

    gnn = GraphMSG(
        config,
    ).to(device)

    # count_parameters_pretty(gnn)

    _ERA_SIZE = gnn._era_size
    x_input = torch.randn(config.dataloader.batch_size.training, config.training.multistep_input, 
                          _ERA_SIZE, config.data.num_features).to(device)  # input tensor
    LOGGER.debug("Input shape: %s", x_input.shape)
    start = timer()
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof:
    y_pred = gnn(x_input)
    LOGGER.debug("Output shape: %s", y_pred.shape)
    LOGGER.debug("Model parameter count: %d", count_parameters(gnn)/1.e6)
    loss = y_pred.sum()
    LOGGER.debug("Running backward on a dummy loss ...")
    loss.backward()
    end = timer()
    LOGGER.debug("Ran backward. All good!")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

    LOGGER.debug("Runtime %f", (end - start))
    for name, param in gnn.named_parameters():
        if param.grad is None:
            print(name)

    LOGGER.debug(f"max memory alloc:    {torch.cuda.max_memory_allocated(torch.device(0))/(1000.*1024)}")
    LOGGER.debug(f"max memory reserved: {torch.cuda.max_memory_reserved(torch.device(0))/(1000.*1024)}")
    

# aifs-train dataloader.num_workers.training=1 dataloader.num_workers.validation=1 diagnostics.logging.wandb=False model.num_channels=128 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=1 training.rollout.max=1 hardware.num_gpus_per_node=2 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4.pt
# aifs-train dataloader.num_workers.training=8 dataloader.num_workers.validation=4 diagnostics.logging.wandb=True model.num_channels=256 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.rollout.start=1 training.rollout.epoch_increment=1 training.rollout.max=1 hardware.num_gpus_per_node=2 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4.pt
# aifs-train data.resolution=o160 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 diagnostics.logging.wandb=False model.num_channels=1024 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=1 training.rollout.max=12 hardware.num_gpus_per_node=2 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5.pt
# aifs-train data.resolution=o160 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 diagnostics.logging.wandb=False model.num_channels=1024 dataloader.limit_batches.training=2 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=1 training.rollout.max=1 hardware.num_gpus_per_node=2 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5.pt