from typing import List
from typing import Optional
from typing import Tuple

import einops
import numpy as np
import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj
from torch_geometric.typing import Size
from torch_geometric.utils import contains_isolated_nodes
from torch_geometric.utils import dropout_edge

from aifs.distributed.helpers import change_channels_in_shape
from aifs.distributed.helpers import get_shape_shards
from aifs.model.layers import GNNMapper
from aifs.model.layers import GNNProcessor
from aifs.utils.config import DotConfig
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)


class GraphMSG(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        config: DotConfig,
        graph_data: HeteroData = None,
        fp32_comm_ops: bool = True,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData, optional
            Graph definition, by default None
        fp32_comm_ops: Bool, optional
            Perform some communication operations (e.g. reduce) in FP32, by default True
            Set this to false when doing e.g. gradient checks (to use torch.double)
        """
        super().__init__()

        self._graph_data = graph_data

        self.in_channels = config.data.num_features - config.data.num_aux_features
        self.multi_step = config.training.multistep_input
        self.aux_in_channels = config.data.num_aux_features
        self.proc_noise_channels = config.model.processor.num_noise_channels

        LOGGER.debug("in_channels + aux_channels == %d", self.in_channels + self.aux_in_channels)
        LOGGER.debug("processor noise channels = %d", self.proc_noise_channels)

        self.activation = config.model.activation

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
        mlp_extra_layers = config.model.mlp.extra_layers

        # Dropout options
        mlp_dropout = config.model.mlp.dropout  # MLPs
        self.dropout_h2h = config.model.edge_dropout.h2h  # processor edges

        # Encoder from ERA -> H
        self.forward_mapper = GNNMapper(
            in_channels_src=self.multi_step * (self.in_channels + self.aux_in_channels)
            + self.era_latlons.shape[1]
            + self.era_trainable_size,
            in_channels_dst=self.h_latlons.shape[1] + self.h_trainable_size,
            hidden_dim=self.num_channels,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.e2h_edge_attr.shape[1] + self.e2h_trainable_size,
            activation=self.activation,
            num_chunks=config.model.encoder.num_chunks,
            fp32_comm_ops=fp32_comm_ops,
        )

        # Processor H -> H
        self.h_processor = GNNProcessor(
            hidden_dim=self.num_channels + self.proc_noise_channels,
            output_dim=self.num_channels,
            hidden_layers=config.model.processor.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
            chunks=config.model.processor.chunks,
            activation=self.activation,
            mlp_dropout=mlp_dropout,
        )

        # Decoder H -> ERA5
        self.backward_mapper = GNNMapper(
            in_channels_src=self.num_channels,
            in_channels_dst=self.num_channels,
            out_channels_dst=self.in_channels,
            hidden_dim=self.num_channels,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2e_edge_attr.shape[1] + self.h2e_trainable_size,
            backward_mapper=True,
            activation=self.activation,
            num_chunks=config.model.decoder.num_chunks,
            fp32_comm_ops=fp32_comm_ops,
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

    def _fuse_trainable_tensors(
        self, edge_or_node_tensor: torch.Tensor, batch_size: int, trainable: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse edge / node tensors, trainable tensors and (optionally) noise tensors.

        Parameters
        ----------
        edge_or_node_tensor : torch.Tensor
            Edge or node tensor
        batch_size: int
            Batch size
        trainable : Optional[torch.Tensor]
            Tensor with trainable node or edge features

        Returns
        -------
        torch.Tensor
            Fused tensors for latent space
        """
        latent = [einops.repeat(edge_or_node_tensor, "e f -> (repeat e) f", repeat=batch_size)]
        if trainable is not None:
            latent.append(einops.repeat(trainable, "e f -> (repeat e) f", repeat=batch_size))
        return torch.cat(
            latent,
            dim=-1,  # feature dimension
        )

    def _expand_edges(self, edge_index: Adj, edge_inc: torch.Tensor, batch_size: int) -> Adj:
        """Expand edge index correct number of times while adding the proper number to
        the edge index.

        Parameters
        ----------
        edge_index : Adj
            Edge index to start
        edge_inc : torch.Tensor
            Edge increment to use

        Returns
        -------
        torch.Tensor
            Edge Index
        """
        edge_index = torch.cat(
            [edge_index + i * edge_inc for i in range(batch_size)],
            dim=1,
        )

        return edge_index

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: Tuple[torch.Tensor],
        edge_index: Adj,
        edge_inc: torch.Tensor,
        edge_attr: torch.Tensor,
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
        data : Tuple[torch.Tensor]
            Tuple of data to pass in
        edge_index : int
            Edge index to start
        edge_inc : int
            Edge increment to use
        edge_attr : torch.Tensor
            Trainable edge attribute tensor
        shape_nodes: Tuple[List, List]
            Shapes of input fileds the task holds when running with multiple GPUs
        batch_size: int
            Batch size (includes the ensemble dimension, if any).
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
            edge_index=self._expand_edges(edge_index, edge_inc, batch_size),
            edge_attr=edge_attr,
            shape_nodes=shape_nodes,
            size=size,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def dropout_edge_force_undirected(
        self, edge_index: torch.Tensor, p: float = 0.5, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly drops edges from the adjacency matrix
        :obj:`edge_index` with probability :obj:`p` using samples from
        a Bernoulli distribution.

        If an edge is dropped in one direction then it will be automatically dropped in
        the other direction ie. will either drop or keep both edges

        Parameters
        ----------
        edge_index : Tensor
            Edge index to start
        p : int
            Dropout rate
        training" (bool, optional)
            If set to False then this operation does not occur during training

        Returns
        -------
        edge_index : Tensor
            New edge index with dropped-out edges
        edge_mask : Tensor
            Masked Tensor with edges to be dropped

        """

        # dropout edges using torch geometric functionality
        _, edge_mask = dropout_edge(edge_index, p=p, training=training)
        row, col = edge_index

        #  in one direction so that only have unidirectional edges
        edge_mask[row > col] = False

        # remo
        edge_index = edge_index[:, edge_mask]

        edge_index_concat = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        return edge_index_concat, edge_mask

    def forward(self, x: torch.Tensor, model_comm_group: Optional[ProcessGroup] = None, inject_noise: bool = True) -> torch.Tensor:
        """Forward operator.

        Args:
            x: torch.Tensor
                Input tensor, shape (bs, e, m, n, f)
            model_comm_group: Optional[ProcessGroup], optional
                Model communication group
            inject_noise: bool, optional
                Inject white noise in the processor, default True.
                If False then we inject an all-zeros noise tensor (do this e.g. when pre-training with a deterministic loss).
        Returns:
            Output tensor
        """
        bs, e = x.shape[0], x.shape[1]
        bse = bs * e  # merge the batch and ensemble dimensions

        if self.dropout_h2h > 0:
            isolated_nodes_h2h = True

            total_num_h2h_nodes = self._h_size

            while isolated_nodes_h2h:
                edge_index_h2h_edge, edge_h2h_mask = self.dropout_edge_force_undirected(self.h2h_edge_index, p=self.dropout_h2h)
                isolated_nodes_h2h = contains_isolated_nodes(edge_index_h2h_edge, num_nodes=total_num_h2h_nodes)

            edge_attr = self.h2h_edge_attr[edge_h2h_mask, :]

            h2h_attr = torch.cat([edge_attr, edge_attr], dim=0)

            if self.h2h_trainable is not None:
                h2h_train = self.h2h_trainable[edge_h2h_mask, :]
                h2h_trainable = torch.cat([h2h_train, h2h_train], dim=0)

        else:
            edge_index_h2h_edge = self.h2h_edge_index
            h2h_attr = self.h2h_edge_attr
            h2h_trainable = self.h2h_trainable

        # add ERA positional info (lat/lon)
        x_era_latent = torch.cat(
            (
                einops.rearrange(x, "bs e m n f -> (bs e n) (m f)"),
                self._fuse_trainable_tensors(self.era_latlons, bse, self.era_trainable),
            ),
            dim=-1,  # feature dimension
        )

        x_h_latent = self._fuse_trainable_tensors(self.h_latlons, bse, self.h_trainable)
        edge_e_to_h_latent = self._fuse_trainable_tensors(self.e2h_edge_attr, bse, self.e2h_trainable)
        edge_h_to_h_latent = self._fuse_trainable_tensors(h2h_attr, bse, h2h_trainable)
        edge_h_to_e_latent = self._fuse_trainable_tensors(self.h2e_edge_attr, bse, self.h2e_trainable)

        LOGGER.debug("x_h_latent.shape = %s", x_h_latent.shape)

        # size for mappers:
        size_fwd = (x_era_latent.shape[0], x_h_latent.shape[0])
        size_bwd = (x_h_latent.shape[0], x_era_latent.shape[0])

        # shapes of node shards:
        shape_x_fwd = get_shape_shards(x_era_latent, 0, model_comm_group)
        shape_h_fwd = get_shape_shards(x_h_latent, 0, model_comm_group)
        # the processor input and output feature dimension lengths are different (input has an extra self.proc_noise_channels)
        shape_h_proc = change_channels_in_shape(shape_h_fwd, self.num_channels + self.proc_noise_channels)
        shape_h_bwd = change_channels_in_shape(shape_h_fwd, self.num_channels)
        shape_x_bwd = change_channels_in_shape(shape_x_fwd, self.num_channels)

        LOGGER.debug("shape_h_fwd = %s, shape_h_proc = %s, shape_h_bwd = %s", shape_h_fwd, shape_h_proc, shape_h_bwd)

        x_era_latent, x_latent = self._run_mapper(
            self.forward_mapper,
            (x_era_latent, x_h_latent),
            self.e2h_edge_index,
            self._e2h_edge_inc,
            edge_e_to_h_latent,
            shape_nodes=(shape_x_fwd, shape_h_fwd),
            batch_size=bse,
            size=size_fwd,
            model_comm_group=model_comm_group,
        )

        LOGGER.debug("x_era_latent.shape = %s, x_latent.shape = %s", x_era_latent.shape, x_latent.shape)

        # generate noise tensor
        noise_shape = (*x_latent.shape[:-1], self.proc_noise_channels)
        z = torch.randn(noise_shape).type_as(x_latent) if inject_noise else torch.zeros(noise_shape).type_as(x_latent)
        z.requires_grad = False
        LOGGER.debug("z.shape = %s, z.norm: %.9e", z.shape, torch.linalg.norm(z))

        x_latent_proc = self.h_processor(
            # concat noise tensor to the latent features
            x=torch.cat([x_latent, z], dim=-1),
            edge_index=self._expand_edges(edge_index_h2h_edge, self._h2h_edge_inc, bse),
            edge_attr=edge_h_to_h_latent,
            shape_nodes=shape_h_proc,
            model_comm_group=model_comm_group,
        )

        # skip connection (H -> H)
        x_latent_proc = x_latent_proc + x_latent

        _, x_out = self._run_mapper(
            self.backward_mapper,
            (x_latent_proc, x_era_latent),
            self.h2e_edge_index,
            self._h2e_edge_inc,
            edge_h_to_e_latent,
            shape_nodes=(shape_h_bwd, shape_x_bwd),
            batch_size=bse,
            size=size_bwd,
            model_comm_group=model_comm_group,
        )

        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=bs)

        # residual connection (just for the predicted variables at the current step)
        # x.shape = (bs, e, m, n, f)
        return x_out + x[:, :, -1, :, : self.in_channels]


if __name__ == "__main__":
    from pathlib import Path
    from hydra import compose, initialize
    from torch_geometric import seed_everything

    # from torch.profiler import profile, record_function, ProfilerActivity

    from timeit import default_timer as timer

    initialize(config_path="../config", job_name="test_msg")
    cfg_ = compose(
        config_name="ens-kcrps-h4",
        overrides=[
            # "model.trainable_parameters.era=8",
            # "model.trainable_parameters.hidden=8",
            # "model.trainable_parameters.era2hidden=8",
            # "model.trainable_parameters.hidden2era=8",
            # "model.trainable_parameters.hidden2hidden=8",
            # "model.num_channels=128",
            # "dataloader.batch_size.training=1",
            # "dataloader.batch_size.validation=1",
            # "data.num_features=98",
            # "data.num_aux_features=13",
            # "training.multistep_input=2",
            # 'hardware.paths.graph="/home/mlx/data/graphs/"',
            # 'hardware.files.graph="graph_mappings_normed_edge_attrs_2023062700_o96_h_0_1_2_3_4.pt"',
        ],
    )

    seed_everything(1234)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.debug("Running on device: %s ...", device)

    gdata = torch.load(Path(cfg_.hardware.paths.graph, cfg_.hardware.files.graph))
    gnn_ = GraphMSG(cfg_, graph_data=gdata).to(device)

    _ERA_SIZE = gnn_._era_size
    x_input = torch.randn(
        cfg_.dataloader.batch_size.training,
        cfg_.training.ensemble_size,
        cfg_.training.multistep_input,
        _ERA_SIZE,
        cfg_.data.num_features,
    ).to(device)

    LOGGER.debug("Input shape: %s", x_input.shape)
    start = timer()

    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof:

    y_pred = gnn_(x_input)
    LOGGER.debug("Output shape: %s", y_pred.shape)
    LOGGER.debug("Model parameter count: %d M", count_parameters(gnn_) / 1.0e6)

    loss = y_pred.sum()
    LOGGER.debug("Running backward on a dummy loss ...")
    loss.backward()

    end = timer()
    LOGGER.debug("Ran backward. All good!")

    # LOGGER.debug(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

    LOGGER.debug("Runtime %.1f s", (end - start))

    for pname, pval in gnn_.named_parameters():
        if pval.grad is None:
            print(pname)

    if torch.cuda.is_available():
        LOGGER.debug("max memory allocated:  %5.1f MB", torch.cuda.max_memory_allocated(torch.device(0)) / (1000.0 * 1024))
        LOGGER.debug("max memory reserved:   %5.1f MB", torch.cuda.max_memory_reserved(torch.device(0)) / (1000.0 * 1024))
