import os
import sys

import einops
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from aifs.diagnostics.logger import get_logger
from aifs.model.layers import MessagePassingMapper
from aifs.model.layers import MessagePassingProcessor, MessagePassingProcessorWraper
from aifs.utils.distributed import get_shape_shards1, change_channels_in_shape1

LOGGER = get_logger(__name__)


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
            self._graph_data = torch.load(os.path.join(config.hardware.paths.graph, config.hardware.files.graph))
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

        self.encoder_out_channels = config.model.num_channels
        mlp_extra_layers = config.model.mlp.extra_layers

        # Encoder from ERA -> H
        self.forward_mapper = MessagePassingMapper(
            in_channels_src=self.multi_step * (self.in_channels + self.aux_in_channels)
            + self.era_latlons.shape[1]
            + self.era_trainable_size,
            in_channels_dst=self.h_latlons.shape[1] + self.h_trainable_size,
            hidden_dim=self.encoder_out_channels,
            hidden_layers=config.model.encoder.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.e2h_edge_attr.shape[1] + self.e2h_trainable_size,
            chunks=1,
            activation=self.activation,
        )

        # Processor H -> H
        self.h_processor = MessagePassingProcessorWraper(
            hidden_dim=self.encoder_out_channels,
            hidden_layers=config.model.hidden.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
            chunks=1,
            activation=self.activation,
        )

        # self.h_processor0 = MessagePassingProcessor(
        #     hidden_dim=self.encoder_out_channels,
        #     hidden_layers=int(config.model.hidden.num_layers/2),
        #     mlp_extra_layers=mlp_extra_layers,
        #     edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
        #     chunks=1,
        #     activation=self.activation,
        #     emb_edges=True,
        # )

        # # Processor H -> H
        # self.h_processor1 = MessagePassingProcessor(
        #     hidden_dim=self.encoder_out_channels,
        #     hidden_layers=int(config.model.hidden.num_layers/2),
        #     mlp_extra_layers=mlp_extra_layers,
        #     edge_dim=0,
        #     chunks=1,
        #     activation=self.activation,
        #     emb_edges=False,
        # )

        # Decoder H -> ERA5
        self.backward_mapper = MessagePassingMapper(
            in_channels_src=self.encoder_out_channels,
            in_channels_dst=self.encoder_out_channels,
            out_channels_dst=self.in_channels,
            hidden_dim=config.model.num_channels,
            hidden_layers=config.model.decoder.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2e_edge_attr.shape[1] + self.h2e_trainable_size,
            backward_mapper=True,
            chunks=1,
            activation=self.activation,
        )

    def forward(self, x: torch.Tensor, mgroupdef) -> torch.Tensor:
        # mgroupdef[0] : comms group
        # mgroupdef[1] : lenght of cumms group -> not used .. get via dist.get_world_size(group=mgroupdef[0])
        # mgroupdef[2] : local rank in comms group -> not used .. get via dist.get_rank(group=mgroupdef[0])

        bs = x.shape[0]

        # add ERA positional info (lat/lon)
        x_in = [
            einops.rearrange(x, "b m n f -> (b n) (m f)"),
            einops.repeat(self.era_latlons, "e f -> (repeat e) f", repeat=bs),
        ]
        if self.era_trainable is not None:
            x_in.append(einops.repeat(self.era_trainable, "e f -> (repeat e) f", repeat=bs))
        x_era_in = torch.cat(
            x_in,
            dim=-1,  # feature dimension
        )

        x_h_in = [einops.repeat(self.h_latlons, "e f -> (repeat e) f", repeat=bs)]
        if self.h_trainable is not None:
            x_h_in.append(einops.repeat(self.h_trainable, "e f -> (repeat e) f", repeat=bs))
        x_h_in = torch.cat(
            x_h_in,
            dim=-1,  # feature dimension
        )

        edge_era_to_h_latent = [einops.repeat(self.e2h_edge_attr, "e f -> (repeat e) f", repeat=bs)]
        if self.e2h_trainable is not None:
            edge_era_to_h_latent.append(einops.repeat(self.e2h_trainable, "e f -> (repeat e) f", repeat=bs))
        edge_era_to_h_latent = torch.cat(
            edge_era_to_h_latent,
            dim=-1,  # feature dimension
        )  # copy edge attributes bs times

        # size for mappers:
        size_fwd = (x_era_in.shape[0], x_h_in.shape[0])
        size_bwd = (x_h_in.shape[0], x_era_in.shape[0])

        # shapes of node shards:
        shape_x_fwd = get_shape_shards1(x_era_in, 0, mgroupdef[0])
        shape_h_fwd = get_shape_shards1(x_h_in, 0, mgroupdef[0])
        shape_h_proc = change_channels_in_shape1(shape_h_fwd, self.encoder_out_channels)
        shape_h_bwd = shape_h_proc
        shape_x_bwd = change_channels_in_shape1(shape_x_fwd, self.encoder_out_channels)

        (x_era_latent, x_latent) = checkpoint(
            self.forward_mapper,
            (x_era_in, x_h_in),
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self.e2h_edge_index + i * self._e2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_era_to_h_latent,
            shape_nodes=(shape_x_fwd, shape_h_fwd),
            size=size_fwd,
            mgroupdef=mgroupdef,
            use_reentrant=False,
        )

        edge_h_to_h_latent = [einops.repeat(self.h2h_edge_attr, "e f -> (repeat e) f", repeat=bs)]
        if self.h2h_trainable is not None:
            edge_h_to_h_latent.append(einops.repeat(self.h2h_trainable, "e f -> (repeat e) f", repeat=bs))
        edge_h_to_h_latent = torch.cat(
            edge_h_to_h_latent,
            dim=-1,  # feature dimension
        )

        # x_latent_proc0, edge_attr_proc0, edge_index_proc0 = checkpoint(self.h_processor0,  # has skipped connections and checkpoints inside
        #     x=x_latent,
        #     edge_index=torch.cat(
        #         [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bs)],
        #         dim=1,
        #     ),
        #     edge_attr=edge_h_to_h_latent,
        #     shape_nodes=shape_h_proc,
        #     mgroupdef=mgroupdef,
        #     use_reentrant=False,
        # )

        # x_latent_proc, _, _ = checkpoint(self.h_processor1,  # has skipped connections and checkpoints inside
        #     x=x_latent_proc0,
        #     edge_index=edge_index_proc0,
        #     edge_attr=edge_attr_proc0,
        #     shape_nodes=shape_h_proc,
        #     mgroupdef=mgroupdef,
        #     use_reentrant=False,
        # )

        # x_latent_proc = self.h_processor(  # has skipped connections and checkpoints inside
        #     x=x_latent,
        #     edge_index=torch.cat(
        #         [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bs)],
        #         dim=1,
        #     ),
        #     edge_attr=edge_h_to_h_latent,
        #     shape_nodes=shape_h_proc,
        #     mgroupdef=mgroupdef,
        # )
        # x_latent_proc = checkpoint(self.h_processor,  # has skipped connections and checkpoints inside
        x_latent_proc = self.h_processor(  # has skipped connections and checkpoints inside
            x=x_latent,
            edge_index=torch.cat(
                [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_h_to_h_latent,
            shape_nodes=shape_h_proc,
            mgroupdef=mgroupdef,
            # use_reentrant=False,
        )

        # x_latent_proc = self.h_processor(  # has skipped connections and checkpoints inside
        #     x=x_latent,
        #     edge_index=torch.cat(
        #         [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bs)],
        #         dim=1,
        #     ),
        #     edge_attr=edge_h_to_h_latent,
        #     shape_nodes=shape_h_proc,
        #     mgroupdef=mgroupdef,
        # )

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
            shape_nodes=(shape_h_bwd, shape_x_bwd),
            size=size_bwd,
            mgroupdef=mgroupdef,
            use_reentrant=False,
        )

        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=bs)

        # residual connection (just for the predicted variables)
        return x_out + x[:, -1, :, : self.in_channels]


############ testing ########

from prettytable import PrettyTable


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_pretty(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params/1.e6}")
    return total_params


def get_my_mgroup(world_size, rank, mgroup_size):
    """determine model group"""
    mgroups = np.array([x for x in np.arange(0, world_size)])
    mgroups = np.split(mgroups, world_size / mgroup_size)

    my_mgroup = None
    imgroup = None
    for i, mgroup in enumerate(mgroups):
        if rank in mgroup:
            imgroup = i
            my_mgroup = mgroup
            mgroup_rank = np.ravel(np.asarray(mgroup == rank).nonzero())[0]
    return imgroup, my_mgroup, mgroup_rank


import hydra
from hydra import compose, initialize
from torch_geometric import seed_everything

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  #'localhost'
    os.environ["MASTER_PORT"] = "12435"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_parallel(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


# @hydra.main(version_base=None)
def test_gnn(rank, world_size):
    print(f"Run GNN DDP on rank {rank}.")
    setup(rank, world_size)

    import time

    # create model and move it to GPU with id rank
    initialize(config_path="../config", job_name="test_msg", version_base=None)
    config = compose(
        config_name="config",
        overrides=[
            # 'model.trainable_parameters.era=0',
            # 'model.trainable_parameters.hidden=0',
            # 'model.trainable_parameters.era2hidden=0',
            # 'model.trainable_parameters.hidden2era=0',
            # 'model.trainable_parameters.hidden2hidden=0',
            "model.trainable_parameters.era=8",
            "model.trainable_parameters.hidden=8",
            "model.trainable_parameters.era2hidden=8",
            "model.trainable_parameters.hidden2era=8",
            "model.trainable_parameters.hidden2hidden=8",
            # "model.trainable_parameters.era=1",
            # "model.trainable_parameters.hidden=1",
            # "model.trainable_parameters.era2hidden=1",
            # "model.trainable_parameters.hidden2era=1",
            # "model.trainable_parameters.hidden2hidden=1",
            "model.hidden.num_layers=16",  # 16',
            # "model.num_channels=2",  # 16',
            "dataloader.batch_size.training=1",
            "data.num_features=98",
            "data.num_aux_features=13",
            # "data.num_features=3",
            # "data.num_aux_features=1",
            # "training.multistep_input=1",
            "training.multistep_input=2",
            'hardware.paths.graph="/home/mlx/data/graphs/"',
            'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs1_o32_h_0_1_2.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered_20230723094641_o96_h_0_1_2_3_4.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered_20230723094050_o96_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5_6.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5_6.pt"',
        ],
    )

    mgroup_size = 2

    if mgroup_size == 1:
        lr = 0.01
    else:
        lr = 0.01 * mgroup_size

    assert world_size % mgroup_size == 0

    comms_groups_ranks = np.split(np.array([x for x in range(0, world_size)]), int(world_size / mgroup_size))
    comms_groups = [dist.new_group(x) for x in comms_groups_ranks]
    print(comms_groups)
    imgroup, my_mgroup, my_mgroup_rank = get_my_mgroup(world_size, rank, mgroup_size)
    comms_group = comms_groups[imgroup]
    print(
        f" rank's {rank} mgroup is {my_mgroup}, group number {imgroup}, with local group rank {my_mgroup_rank}, comms_group ranks {comms_groups_ranks[imgroup]}"
    )

    # if mgroup_size > 1:
    #     comms_group = dist.new_group(my_mgroup)
    # else:
    #     comms_group = 0

    mgroupdef = (comms_group, len(my_mgroup), my_mgroup_rank)

    # sys.exit(0)

    iseed = 1234  # + imgroup * 100000
    print(f"rank's {rank} seed {iseed}")
    seed_everything(iseed)

    model = GraphMSG(
        config,
    ).to(rank)

    # count_parameters_pretty(model)

    gnn = DDP(model, device_ids=[rank])  # , find_unused_parameters=True)

    params = []
    for param in gnn.parameters():
        params.append(param.view(-1))
    params1 = torch.cat(params)

    optimizer = torch.optim.SGD(gnn.parameters(), lr=lr, momentum=0.9)

    torch.autograd.set_detect_anomaly(True)

    # ### data:
    _ERA_SIZE = model._era_size
    x_input = torch.randn(
        config.dataloader.batch_size.training, config.training.multistep_input, _ERA_SIZE, config.data.num_features
    ).to(rank)
    x_input2 = torch.randn(
        config.dataloader.batch_size.training, config.training.multistep_input, _ERA_SIZE, config.data.num_features
    ).to(rank)
    y = torch.randn(config.dataloader.batch_size.training, _ERA_SIZE, 98 - 13).to(rank)
    # y = torch.randn(config.dataloader.batch_size.training, _ERA_SIZE, 3 - 1).to(rank)

    optimizer.zero_grad()

    start_time = time.time()

    y_pred = gnn(x_input, mgroupdef)
    loss = (y_pred - y).sum()

    # print(loss)
    loss.backward()

    print("{rank} --- %s seconds ---" % (time.time() - start_time))

    print(f" =====##### rank {rank} has loss1 {loss:.20f}")

    optimizer.step()

    y_pred2 = gnn(x_input2, mgroupdef)
    print(f" =====##### rank {rank} has loss2 {y_pred2.sum():.20f}")

    for name, param in gnn.named_parameters():
        if param.grad is None:
            print(name)

    params = []
    for param in gnn.parameters():
        params.append(param.view(-1))
    params2 = torch.cat(params)

    print(f"model diff: {(((params2 - params1)**2.)**(1./2.)).sum():.20f}")
    print(f"model param sum: {params2.abs().sum():.20f}")

    print(f"rank {rank} max memory alloc: {torch.cuda.max_memory_allocated(torch.device(rank))/1.e6}")
    print(f"rank {rank} max memory reserved: {torch.cuda.max_memory_reserved(torch.device(rank))/1.e6}")

    # print("grad check ...")

    # def test_gnn_grad(x):
    #     y = gnn(x, mgroupdef)
    #     return y

    # x_input = torch.randn(config.dataloader.batch_size.training, config.training.multistep_input,
    #                     _ERA_SIZE, config.data.num_features, requires_grad=True).to(rank)
    # torch.autograd.gradcheck(test_gnn_grad, x_input, eps=1.0, fast_mode=False)

    # for name, parameter in gnn.named_parameters():
    #     # if name == "module.h_trainable":
    #     if name == "module.backward_mapper.emb_edges.2.weight":
    #         print(parameter.grad)

    cleanup()


if __name__ == "__main__":
    import os

    # from torch.profiler import profile, record_function, ProfilerActivity
    # from timeit import default_timer as timer

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8 required

    seed_everything(1234)

    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    run_parallel(test_gnn, world_size)

    # aifs-train dataloader.num_workers.training=1 dataloader.num_workers.validation=1 diagnostics.logging.wandb=False model.num_channels=128 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=1 training.rollout.max=1 hardware.num_gpus_per_node=2 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4.pt
    # aifs-train dataloader.num_workers.training=8 dataloader.num_workers.validation=4 diagnostics.logging.wandb=True model.num_channels=256 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.rollout.start=1 training.rollout.epoch_increment=1 training.rollout.max=1 hardware.num_gpus_per_node=2 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4.pt
    # full rollout test: aifs-train training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 diagnostics.logging.wandb=False model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=0 training.rollout.max=12 hardware.num_gpus_per_node=4 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ data.resolution=o96 hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4_5.pt
    # full rollout test, gpu=4 per node, group size=2 : wandb offline ; aifs-train training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=0 training.rollout.max=12 hardware.num_gpus_per_node=4 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.group_size=2 data.resolution=o96 hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4_5.pt
    # plot debug full rollout test, gpu=4 per node, group size=2 : wandb offline ; aifs-train training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=128 dataloader.limit_batches.training=20 dataloader.limit_batches.validation=5 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=10 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.group_size=1 data.resolution=o96 hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4_5.pt
    # broken ... n320 full rollout test, gpu=4 per node, group size=2 : aifs-train training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 diagnostics.logging.wandb=False model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=0 training.rollout.max=12 hardware.num_gpus_per_node=4 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.group_size=2 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.training=aifs_n320_1979_2015_1h.zarr hardware.files.validation=aifs_n320_2016_2017_1h.zarr hardware.files.test=aifs_n320_2016_2017_1h.zarr hardware.files.predict=aifs_n320_2016_2017_1h.zarr data.resolution=n320 hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5.pt
    # o160 full rollout test, gpu=4 per node, group size=2 : aifs-train training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 diagnostics.logging.wandb=False model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.training=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=0 training.rollout.max=12 hardware.num_gpus_per_node=4 hardware.num_nodes=1 training.initial_seed=24 hardware.paths.graph=/home/mlx/data/graphs/ hardware.group_size=2 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.training=aifs-era5-o160-1979-2015-6h.zarr hardware.files.validation=aifs-era5-o160-2016-2017-6h.zarr hardware.files.test=aifs-era5-o160-2016-2017-6h.zarr hardware.files.predict=aifs-era5-o160-2016-2017-6h.zarr data.resolution=o160 hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5_6.pt


    # n320 test ; aifs-train dataloader.batch_size.training=1 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=0 training.rollout.max=12 hardware.group_size=2 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=n320 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs_n320_2016_2017_1h.zarr      hardware.files.training=aifs_n320_1979_2015_1h.zarr       hardware.files.test=aifs_n320_2016_2017_1h.zarr      hardware.files.predict=aifs_n320_2016_2017_1h.zarr       hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5.pt
    #             aifs-train dataloader.batch_size.training=1 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=0 training.rollout.max=12 hardware.group_size=2 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=o160 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs-era5-o160-2016-2017-6h.zarr hardware.files.training=aifs-era5-o160-1979-2015-6h.zarr  hardware.files.test=aifs-era5-o160-2016-2017-6h.zarr hardware.files.predict=aifs-era5-o160-2016-2017-6h.zarr  hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5_6.pt

    # hardware.paths.graph=/home/mlx/data/graphs/

    # latest rollout test:
    # n320, aifs-train dataloader.batch_size.training=1 hardware.group_size=4 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=0 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=11 training.rollout.epoch_increment=0 training.rollout.max=11 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=n320 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs_n320_2016_2017_1h.zarr      hardware.files.training=aifs_n320_1979_2015_1h.zarr       hardware.files.test=aifs_n320_2016_2017_1h.zarr      hardware.files.predict=aifs_n320_2016_2017_1h.zarr       hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5_6.pt
    # o160, aifs-train dataloader.batch_size.training=1 hardware.group_size=1 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=128 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=o160 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs-era5-o160-2016-2017-6h.zarr     hardware.files.training=aifs-era5-o160-1979-2015-6h.zarr   hardware.files.test=aifs-era5-o160-2016-2017-6h.zarr  hardware.files.predict=aifs-era5-o160-2016-2017-6h.zarr   hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5_6.pt
    # o96,  aifs-train dataloader.batch_size.training=1 hardware.group_size=1 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=128 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=o96  hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5_6.pt