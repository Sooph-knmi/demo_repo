from typing import List
from typing import Optional
from typing import Tuple

import einops
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj
from torch_geometric.typing import Size

from aifs.distributed.helpers import change_channels_in_shape
from aifs.distributed.helpers import get_shape_shards
from aifs.model.layers import GNNMapper
from aifs.model.layers import GNNProcessor
from aifs.model.layers import TransformerProcessor
from aifs.utils.config import DotConfig
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class GraphMSG(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        config: DotConfig,
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

        self._graph_data = graph_data

        self.in_channels = config.data.num_features - config.data.num_aux_features
        self.multi_step = config.training.multistep_input
        self.aux_in_channels = config.data.num_aux_features

        LOGGER.debug("self.in_channels + self.aux_channels == %d", self.in_channels + self.aux_in_channels)

        self.activation = config.model.activation

        assert config.model.processor.type in [
            "GNN",
            "Transformer",
        ], "Processor type not supported, supported are only GNN and Transformer"
        self.processor_type = config.model.processor.type

        # Create Graph edges
        self._create_edges()

        # Define Sizes of different tensors
        self._era_size = self._graph_data[("era", "to", "era")].ecoords_rad.shape[0]
        self._h_size = self._graph_data[("h", "to", "h")].hcoords_rad.shape[0]

        self.era_trainable_size = config.model.trainable_parameters.era
        self.h_trainable_size = config.model.trainable_parameters.hidden
        self.e2h_trainable_size = config.model.trainable_parameters.era2hidden
        self.h2e_trainable_size = config.model.trainable_parameters.hidden2era
        self.h2h_trainable_size = config.model.trainable_parameters.hidden2hidden

        # Create trainable tensors
        self._create_trainable_attr()

        # Register edge increments
        self._register_edge_inc("e2h", self._era_size, self._h_size)
        self._register_edge_inc("h2e", self._h_size, self._era_size)
        self._register_edge_inc("h2h", self._h_size, self._h_size)

        # Register lat/lon
        self._register_latlon("era")
        self._register_latlon("h")

        self.num_channels = config.model.num_channels
        self.proc_channels = config.model.processor.num_channels
        mlp_extra_layers = config.model.mlp.extra_layers

        input_dimension = (
            self.multi_step * (self.in_channels + self.aux_in_channels) + self.era_latlons.shape[1] + self.era_trainable_size
        )

        # Encoder from ERA -> H
        self.forward_mapper = GNNMapper(
            in_channels_src=input_dimension,
            in_channels_dst=self.h_latlons.shape[1] + self.h_trainable_size,
            hidden_dim=self.num_channels,
            mlp_extra_layers=mlp_extra_layers,
            heads=config.model.encoder.heads,
            mlp_hidden_ratio=config.model.encoder.mlp_hidden_ratio,
            edge_dim=self.e2h_edge_attr.shape[1] + self.e2h_trainable_size,
            activation=self.activation,
        )

        # Processor H -> H
        if self.processor_type == "GNN":
            self.h_processor = GNNProcessor(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                hidden_dim=self.proc_channels,
                hidden_layers=config.model.processor.num_layers,
                mlp_extra_layers=mlp_extra_layers,
                edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
                chunks=config.model.processor.chunks,
                heads=config.model.processor.heads,
                mlp_hidden_ratio=config.model.processor.mlp_hidden_ratio,
                activation=self.activation,
            )

        if self.processor_type == "Transformer":
            self.h_processor = TransformerProcessor(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                hidden_dim=self.proc_channels,
                hidden_layers=config.model.processor.num_layers,
                window_size=config.model.processor.window_size,
                chunks=config.model.processor.chunks,
                heads=config.model.processor.heads,
                mlp_hidden_ratio=config.model.processor.mlp_hidden_ratio,
                activation=self.activation,
            )

        # Decoder H -> ERA5
        self.backward_mapper = GNNMapper(
            in_channels_src=self.num_channels,
            # in_channels_dst=self.num_channels,
            in_channels_dst=input_dimension,
            out_channels_dst=self.in_channels,
            hidden_dim=self.num_channels,
            mlp_extra_layers=mlp_extra_layers,
            heads=config.model.decoder.heads,
            mlp_hidden_ratio=config.model.decoder.mlp_hidden_ratio,
            edge_dim=self.h2e_edge_attr.shape[1] + self.h2e_trainable_size,
            backward_mapper=True,
            activation=self.activation,
        )

    def _register_latlon(self, name: str) -> None:
        """Register lat/lon buffers.

        Parameters
        ----------
        name : str
            Name of grid to map
        """
        self.register_buffer(
            f"{name}_latlons",
            torch.cat(
                [
                    torch.as_tensor(np.sin(self._graph_data[(name, "to", name)][f"{name[:1]}coords_rad"])),
                    torch.as_tensor(np.cos(self._graph_data[(name, "to", name)][f"{name[:1]}coords_rad"])),
                ],
                dim=-1,
            ),
            persistent=True,
        )

    def _register_edge_inc(self, name: str, src_size: int, dst_size: int) -> None:
        """Register edge increments.

        Parameters
        ----------
        name : str
            Name of buffer of edge
        src_size : int
            Source size
        dst_size : int
            Target size
        """
        self.register_buffer(
            f"_{name}_edge_inc", torch.from_numpy(np.asarray([[src_size], [dst_size]], dtype=np.int64)), persistent=True
        )

    def _create_edges(self) -> None:
        """Create all edge buffers."""
        mappings = (("era", "era"), ("h", "era"), ("era", "h"), ("h", "h"))
        for src, dst in mappings:
            self.register_buffer(f"{src[:1]}2{dst[:1]}_edge_index", self._graph_data[(src, "to", dst)].edge_index, persistent=False)
            self.register_buffer(f"{src[:1]}2{dst[:1]}_edge_attr", self._graph_data[(src, "to", dst)].edge_attr, persistent=False)

    def _create_trainable_attr(self) -> None:
        """Create all trainable attributes."""
        self.era_trainable = self._create_trainable_tensor(tensor_size=self._era_size, trainable_size=self.era_trainable_size)
        self.h_trainable = self._create_trainable_tensor(tensor_size=self._h_size, trainable_size=self.h_trainable_size)
        self.e2h_trainable = self._create_trainable_tensor(graph=("era", "h"), trainable_size=self.e2h_trainable_size)
        self.h2e_trainable = self._create_trainable_tensor(graph=("h", "era"), trainable_size=self.h2e_trainable_size)
        if self.processor_type == "GNN":
            self.h2h_trainable = self._create_trainable_tensor(graph=("h", "h"), trainable_size=self.h2h_trainable_size)

    def _create_trainable_tensor(
        self, trainable_size: int, tensor_size: Optional[int] = None, graph: Optional[Tuple[str]] = None
    ) -> Optional[nn.Parameter]:
        """Create trainable tensor.

        This can use tensor_size or graph as the first dimension.

        Parameters
        ----------
        trainable_size : int
            Second tensor dimension
        tensor_size : Optional[int]
            First tensor dimension
        graph : Optional[Tuple[str, str]]
            String for graph data, e.g. ("era", "h")

        Returns
        -------
        nn.Parameter, optional
            Tensor with zeros for trainable edges
        """
        assert tensor_size is None or graph is None
        return (
            nn.Parameter(
                torch.zeros(
                    tensor_size if tensor_size is not None else self._graph_data[(graph[0], "to", graph[1])].edge_attr.shape[0],
                    trainable_size,
                )
            )
            if trainable_size > 0
            else None
        )

    def _fuse_trainable_tensors(self, edge: Tensor, trainable: Optional[Tensor]) -> Tensor:
        """Fuse edge and trainable tensors.

        Parameters
        ----------
        edge : Tensor
            Edge tensor
        trainable : Optional[Tensor]
            Tensor with trainable edges

        Returns
        -------
        Tensor
            Fused tensors for latent space
        """
        latent = [einops.repeat(edge, "e f -> (repeat e) f", repeat=self.batch_size)]
        if trainable is not None:
            latent.append(einops.repeat(trainable, "e f -> (repeat e) f", repeat=self.batch_size))
        return torch.cat(
            latent,
            dim=-1,  # feature dimension
        )

    def _expand_edges(self, edge_index: Adj, edge_inc: Tensor) -> Adj:
        """Expand edge index correct number of times while adding the proper number to
        the edge index.

        Parameters
        ----------
        edge_index : Adj
            Edge index to start
        edge_inc : Tensor
            Edge increment to use

        Returns
        -------
        Tensor
            Edge Index
        """
        edge_index = torch.cat(
            [edge_index + i * edge_inc for i in range(self.batch_size)],
            dim=1,
        )

        return edge_index

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: Tuple[Tensor],
        edge_index: Adj,
        edge_inc: Tensor,
        edge_attr: Tensor,
        shape_nodes: Tuple[List, List],
        batch_size: int,
        size: Size,
        model_comm_group: ProcessGroup,
        use_reentrant: bool = False,
    ):
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : Tuple[Tensor]
            Tuple of data to pass in
        edge_index : int
            Edge index to start
        edge_inc : int
            Edge increment to use
        edge_attr : Tensor
            Trainable edge attribute tensor
        shape_nodes: Tuple[List, List]
            Shapes of input fileds the task holds when running with multiple GPUs
        batch_size: int,
            Batch size
        size: Size
            Number of source and target nodes of bipartite graph
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        _type_
            _description_
        """
        return checkpoint(
            mapper,
            data,
            edge_index=self._expand_edges(edge_index, edge_inc),
            edge_attr=edge_attr,
            shape_nodes=shape_nodes,
            batch_size=batch_size,
            size=size,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        self.batch_size = x.shape[0]

        # add ERA positional info (lat/lon)
        x_era_latent = torch.cat(
            (einops.rearrange(x, "b m n f -> (b n) (m f)"), self._fuse_trainable_tensors(self.era_latlons, self.era_trainable)),
            dim=-1,  # feature dimension
        )

        x_h_latent = self._fuse_trainable_tensors(self.h_latlons, self.h_trainable)
        edge_e_to_h_latent = self._fuse_trainable_tensors(self.e2h_edge_attr, self.e2h_trainable)
        if self.processor_type == "GNN":
            edge_h_to_h_latent = self._fuse_trainable_tensors(self.h2h_edge_attr, self.h2h_trainable)
        edge_h_to_e_latent = self._fuse_trainable_tensors(self.h2e_edge_attr, self.h2e_trainable)

        # size for mappers and processor:
        size_fwd = (x_era_latent.shape[0], x_h_latent.shape[0])
        size_bwd = (x_h_latent.shape[0], x_era_latent.shape[0])
        size_proc = x_h_latent.shape[0]

        # shapes of node shards:
        shape_x_fwd = get_shape_shards(x_era_latent, 0, model_comm_group)
        shape_h_fwd = get_shape_shards(x_h_latent, 0, model_comm_group)
        shape_h_proc = change_channels_in_shape(shape_h_fwd, self.num_channels)
        shape_h_bwd = shape_h_proc

        x_latent = self._run_mapper(
            self.forward_mapper,
            (x_era_latent, x_h_latent),
            self.e2h_edge_index,
            self._e2h_edge_inc,
            edge_e_to_h_latent,
            shape_nodes=(shape_x_fwd, shape_h_fwd),
            batch_size=self.batch_size,
            size=size_fwd,
            model_comm_group=model_comm_group,
        )

        if self.processor_type == "GNN":
            x_latent_proc = self.h_processor(
                x_latent,
                edge_index=self._expand_edges(self.h2h_edge_index, self._h2h_edge_inc),
                edge_attr=edge_h_to_h_latent,
                shape_nodes=shape_h_proc,
                batch_size=self.batch_size,
                size=size_proc,
                model_comm_group=model_comm_group,
            )

        if self.processor_type == "Transformer":
            x_latent_proc = self.h_processor(
                x_latent,
                shape_nodes=shape_h_proc,
                batch_size=self.batch_size,
                model_comm_group=model_comm_group,
            )

        # add skip connection (H -> H)
        x_latent_proc = x_latent_proc + x_latent

        x_out = self._run_mapper(
            self.backward_mapper,
            (x_latent_proc, x_era_latent),
            self.h2e_edge_index,
            self._h2e_edge_inc,
            edge_h_to_e_latent,
            shape_nodes=(shape_h_bwd, shape_x_fwd),
            batch_size=self.batch_size,
            size=size_bwd,
            model_comm_group=model_comm_group,
        )

        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=self.batch_size)

        # residual connection (just for the predicted variables)
        return x_out + x[:, -1, :, : self.in_channels]



############ testing ########


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_my_mgroup(world_size, rank, mgroup_size):
    """Determine model group."""
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


from hydra import compose, initialize
from torch_geometric import seed_everything

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from pathlib import Path


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

    # import nvidia_dlprof_pytorch_nvtx
    # nvidia_dlprof_pytorch_nvtx.init()

    print(f"Run GNN DDP on rank {rank}.")
    setup(rank, world_size)

    import time

    iseed = 1234  # + imgroup * 100000
    print(f"rank's {rank} seed {iseed}")
    seed_everything(iseed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # required
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8

    torch.set_printoptions(precision=25)

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
            'model.processor.type="Transformer"',
            # 'model.processor.type="GNN"',
            "model.processor.num_layers=16",  # 16',
            "model.num_channels=1024",
            "model.processor.num_channels=1024",  # 16',
            "model.processor.heads=16",  # 16',
            "model.processor.chunks=2",  # 16',
            "model.processor.window_size=511",  # 16',
            # "model.processor.window_size=2048",  # 16',
            "model.processor.num_channels=1024",  # 16',
            "data.num_features=98",
            "data.num_aux_features=13",
            # "data.num_features=3",
            # "data.num_aux_features=1",
            "dataloader.batch_size.training=1",
            # "training.multistep_input=1",
            "training.multistep_input=2",
            'hardware.paths.graph="/home/mlx/data/graphs/"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs1_o32_h_0_1_2.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered_lat_lon_20231106121941_o96_h_0_1_2_3_4.pt"',

            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_20231121125206_o96_h_o48.pt"',
            'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4_5_6.pt"',


            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered_lat_lon_20231106115258_o96_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered_lat_lon_20231106121544_n320_h_0_1_2_3_4_5_6.pt"',
            #'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered_20230723094641_o96_h_0_1_2_3_4.pt"',

            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered2_20231101132919_o96_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_ordered_20230723094050_o96_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5_6.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5.pt"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5.pt"',
        ],
    )

    mgroup_size = 1

    lr = 0.1

    assert world_size % mgroup_size == 0

    comms_groups_ranks = np.split(np.array([x for x in range(0, world_size)]), int(world_size / mgroup_size))
    comms_groups = [dist.new_group(x) for x in comms_groups_ranks]
    print(comms_groups)
    imgroup, my_mgroup, my_mgroup_rank = get_my_mgroup(world_size, rank, mgroup_size)
    comms_group = comms_groups[imgroup]
    print(
        f" rank's {rank} mgroup is {my_mgroup}, group number {imgroup}, with local group rank {my_mgroup_rank}, comms_group ranks {comms_groups_ranks[imgroup]}"
    )

    print("-------->>>> ", [dist.get_world_size(group=comms_group), comms_group.size()])

    iseed = 1234  # + imgroup * 100000
    print(f"rank's {rank} seed {iseed}")
    seed_everything(iseed)

    graph_data1 = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph))

    model = GraphMSG(
        config,
        graph_data1,
    ).to(rank)

    if comms_group:
        for name, param in model.named_parameters():
            if param.requires_grad is True and "trainable" not in name:
                param.register_hook(lambda grad: grad * float(mgroup_size))

    # count_parameters_pretty(model)

    gnn = DDP(model, device_ids=[rank])  # , find_unused_parameters=True)

    # params = []
    # for param in gnn.parameters():
    #     params.append(param.view(-1))
    # params1 = torch.cat(params)

    optimizer = torch.optim.SGD(gnn.parameters(), lr=lr, momentum=0.9)

    # torch.autograd.set_detect_anomaly(True)

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

    scaler = torch.cuda.amp.GradScaler()

    optimizer.zero_grad()

    start_time = time.time()

    cast_to = torch.float16

    # with torch.autograd.profiler.emit_nvtx():
    with torch.autocast(device_type="cuda", dtype=cast_to):
        y_pred = gnn(x_input, model_comm_group=comms_group)
        loss = (y_pred - y).sum()

    # print(loss)
    if True:
        scaler.scale(loss).backward()

        for name, param in gnn.named_parameters():
            if param.grad is None:
                print(name)

        print("{rank} --- %s seconds ---" % (time.time() - start_time))

        print(f" =====##### rank {rank} has loss1 {loss:.20f}")

        scaler.step(optimizer)
        scaler.update()

    if True:
        with torch.autocast(device_type="cuda", dtype=cast_to):
            y_pred2 = gnn(x_input2, model_comm_group=comms_group)

        print(f" =====##### rank {rank} has loss2 {y_pred2.sum():.20f}")

        for name, param in gnn.named_parameters():
            if param.grad is None:
                print(name)

        # params = []
        # for param in gnn.parameters():
        #     params.append(param.view(-1))
        # params2 = torch.cat(params)

        # print(f"model diff: {(((params2 - params1)**2.)**(1./2.)).sum():.20f}")
        # print(f"model param sum: {params2.abs().sum():.20f}")

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
    #     # if name == "module.backward_mapper.emb_edges.2.weight":
    #     # if name == "module.backward_mapper.emb_edges.2.weight":
    #     if param.grad is not None:
    #         print(name, parameter.grad)
    print("done")
    cleanup()


if __name__ == "__main__":
    import os

    # import nvidia_dlprof_pytorch_nvtx
    # nvidia_dlprof_pytorch_nvtx.init()

    # from torch.profiler import profile, record_function, ProfilerActivity
    # from timeit import default_timer as timer

    seed_everything(1234)

    n_gpus = torch.cuda.device_count()
    world_size = 2 #n_gpus
    print(f"world size {world_size}")
    run_parallel(test_gnn, world_size)
    # run_parallel(test_gnn, 1)

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
    # n320, aifs-train dataloader.batch_size.training=1 hardware.group_size=4 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=0 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=n320 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs_n320_2016_2017_1h.zarr      hardware.files.training=aifs_n320_1979_2015_1h.zarr       hardware.files.test=aifs_n320_2016_2017_1h.zarr      hardware.files.predict=aifs_n320_2016_2017_1h.zarr       hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5_6.pt
    # o160, aifs-train dataloader.batch_size.training=1 hardware.group_size=1 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=128 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=o160 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs-era5-o160-2016-2017-6h.zarr     hardware.files.training=aifs-era5-o160-1979-2015-6h.zarr   hardware.files.test=aifs-era5-o160-2016-2017-6h.zarr  hardware.files.predict=aifs-era5-o160-2016-2017-6h.zarr   hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o160_h_0_1_2_3_4_5_6.pt
    # o96,  aifs-train dataloader.batch_size.training=1 hardware.group_size=1 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=1 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=o96  hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4_5.pt
    #
    # some debug      aifs-train model.hidden.num_layers=1 dataloader.batch_size.training=1 hardware.group_size=2 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=128 dataloader.limit_batches.training=3 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=o96  hardware.files.graph=graph_mappings_normed_edge_attrs_ordered_20230723094050_o96_h_0_1_2_3_4_5.pt
    # debug plots:    cd /ec/res4/scratch/nesl/aifs/o96/plots ; ddir=$(ls -lth|head -2|tail -n 1|awk '{print $NF}') ; echo $ddir ; imgcat $ddir/plots/h_trainable_epoch000.png

    # aifs-train dataloader.batch_size.training=1 hardware.group_size=4 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=3 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=1 training.rollout.epoch_increment=0 training.rollout.max=1 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=n320 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs_n320_2016_2017_6h.zarr      hardware.files.training=aifs_n320_1979_2015_6h.zarr       hardware.files.test=aifs_n320_2016_2017_6h.zarr      hardware.files.predict=aifs_n320_2016_2017_6h.zarr       hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5_6.pt
    # aifs-train dataloader.batch_size.training=1 model.mlp.extra_layers=1 hardware.group_size=4 training.multistep_input=2 dataloader.num_workers.training=1 dataloader.num_workers.validation=1 model.num_channels=512 dataloader.limit_batches.training=3 dataloader.limit_batches.validation=1 dataloader.batch_size.validation=1 dataloader.batch_size.test=1 dataloader.batch_size.predict=1 training.max_epochs=1 training.rollout.start=12 training.rollout.epoch_increment=0 training.rollout.max=12 hardware.num_gpus_per_node=4 hardware.num_nodes=1 data.resolution=n320 hardware.paths.training=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.validation=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.test=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.paths.predict=/lus/h2resw01/fws4/lb/project/ai-ml/zarrs/ hardware.files.validation=aifs_n320_2016_2017_6h.zarr      hardware.files.training=aifs_n320_1979_2015_6h.zarr       hardware.files.test=aifs_n320_2016_2017_6h.zarr      hardware.files.predict=aifs_n320_2016_2017_6h.zarr       hardware.files.graph=graph_mappings_normed_edge_attrs_2023062700_n320_h_0_1_2_3_4_5_6.pt

