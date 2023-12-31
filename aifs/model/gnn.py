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
from aifs.utils.config import DotConfig
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class GraphMSG(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        config: DotConfig,
        data_indices: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()

        self._graph_data = graph_data

        # Calculate shapes and indices
        self.num_input_channels = len(data_indices.model.input)
        self.num_output_channels = len(data_indices.model.output)
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

        assert len(self._internal_output_idx) == len(data_indices.model.output.full) - len(data_indices.model.output.diagnostic), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and the output indices excluding "
            f"diagnostic variables ({len(data_indices.model.output.full)-len(data_indices.model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx
        ), f"Model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

        self.multi_step = config.training.multistep_input
        self.activation = config.model.activation

        # Create Graph edges
        self._create_edges()

        # Define Sizes of different tensors
        self._data_grid_size = self._graph_data[("era", "to", "era")].ecoords_rad.shape[0]
        self._hidden_grid_size = self._graph_data[("h", "to", "h")].hcoords_rad.shape[0]

        self.era_trainable_size = config.model.trainable_parameters.era
        self.h_trainable_size = config.model.trainable_parameters.hidden
        self.e2h_trainable_size = config.model.trainable_parameters.era2hidden
        self.h2e_trainable_size = config.model.trainable_parameters.hidden2era
        self.h2h_trainable_size = config.model.trainable_parameters.hidden2hidden

        # Create trainable tensors
        self._create_trainable_attr()

        # Register edge increments
        self._register_edge_inc("e2h", self._data_grid_size, self._hidden_grid_size)
        self._register_edge_inc("h2e", self._hidden_grid_size, self._data_grid_size)
        self._register_edge_inc("h2h", self._hidden_grid_size, self._hidden_grid_size)

        # Register lat/lon
        self._register_latlon("era")
        self._register_latlon("h")

        self.num_channels = config.model.num_channels
        mlp_extra_layers = config.model.mlp.extra_layers

        # Encoder from ERA -> H
        self.forward_mapper = GNNMapper(
            in_channels_src=self.multi_step * self.num_input_channels + self.era_latlons.shape[1] + self.era_trainable_size,
            in_channels_dst=self.h_latlons.shape[1] + self.h_trainable_size,
            hidden_dim=self.num_channels,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.e2h_edge_attr.shape[1] + self.e2h_trainable_size,
            activation=self.activation,
            num_chunks=config.model.encoder.num_chunks,
        )

        # Processor H -> H
        self.h_processor = GNNProcessor(
            hidden_dim=self.num_channels,
            hidden_layers=config.model.processor.num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
            chunks=config.model.processor.chunks,
            activation=self.activation,
        )

        # Decoder H -> ERA5
        self.backward_mapper = GNNMapper(
            in_channels_src=self.num_channels,
            in_channels_dst=self.num_channels,
            out_channels_dst=self.num_output_channels,
            hidden_dim=self.num_channels,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2e_edge_attr.shape[1] + self.h2e_trainable_size,
            backward_mapper=True,
            activation=self.activation,
            num_chunks=config.model.decoder.num_chunks,
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
                    torch.sin(self._graph_data[(name, "to", name)][f"{name[:1]}coords_rad"]),
                    torch.cos(self._graph_data[(name, "to", name)][f"{name[:1]}coords_rad"]),
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
        self.era_trainable = self._create_trainable_tensor(tensor_size=self._data_grid_size, trainable_size=self.era_trainable_size)
        self.h_trainable = self._create_trainable_tensor(tensor_size=self._hidden_grid_size, trainable_size=self.h_trainable_size)
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
            Shapes of input fields the task holds when running with multiple GPUs
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
        shape_x_bwd = change_channels_in_shape(shape_x_fwd, self.num_channels)

        x_era_latent, x_latent = self._run_mapper(
            self.forward_mapper,
            (x_era_latent, x_h_latent),
            self.e2h_edge_index,
            self._e2h_edge_inc,
            edge_e_to_h_latent,
            shape_nodes=(shape_x_fwd, shape_h_fwd),
            size=size_fwd,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.h_processor(
            x_latent,
            edge_index=self._expand_edges(self.h2h_edge_index, self._h2h_edge_inc),
            edge_attr=edge_h_to_h_latent,
            shape_nodes=shape_h_proc,
            size=size_proc,
            model_comm_group=model_comm_group,
        )

        # add skip connection (H -> H)
        x_latent_proc = x_latent_proc + x_latent

        _, x_out = self._run_mapper(
            self.backward_mapper,
            (x_latent_proc, x_era_latent),
            self.h2e_edge_index,
            self._h2e_edge_inc,
            edge_h_to_e_latent,
            shape_nodes=(shape_h_bwd, shape_x_bwd),
            size=size_bwd,
            model_comm_group=model_comm_group,
        )

        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=self.batch_size).to(dtype=x.dtype).clone()

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x[:, -1, :, self._internal_input_idx]
        return x_out
