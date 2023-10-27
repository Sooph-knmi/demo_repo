from typing import Optional
from typing import Tuple

import einops
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.utils import contains_isolated_nodes
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import mask_select

from aifs.model.layers import MessagePassingMapper
from aifs.model.layers import NoisyMessagePassingProcessor
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
        self.noise_channels = config.model.hidden.num_noise_channels

        LOGGER.debug("in_channels + aux_channels == %d", self.in_channels + self.aux_in_channels)
        LOGGER.debug("noise_channels = %d", self.noise_channels)

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

        encoder_out_channels = config.model.num_channels
        mlp_extra_layers = config.model.mlp.extra_layers

        # Dropout options
        mlp_dropout = config.model.mlp.dropout
        self.dropout_h2e = config.model.edge_dropout.h2e
        self.dropout_h2h = config.model.edge_dropout.h2h

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
        )

        # "Noisy" processor H -> H
        self.h_processor = NoisyMessagePassingProcessor(
            hidden_dim=encoder_out_channels,
            noise_dim=self.noise_channels,
            hidden_layers=config.model.hidden.num_layers,
            edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
            chunks=2,
            mlp_extra_layers=mlp_extra_layers,
            mlp_dropout=mlp_dropout,
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

    def _fuse_trainable_tensors(self, edge: torch.Tensor, batch_size: int, trainable: Optional[torch.Tensor]) -> torch.Tensor:
        """Fuse edge and trainable tensors.

        Parameters
        ----------
        edge : torch.Tensor
            Edge tensor
        batch_size: int
            Batch size
        trainable : Optional[torch.Tensor]
            Tensor with trainable edges

        Returns
        -------
        torch.Tensor
            Fused tensors for latent space
        """
        latent = [einops.repeat(edge, "e f -> (repeat e) f", repeat=batch_size)]
        if trainable is not None:
            latent.append(einops.repeat(trainable, "e f -> (repeat e) f", repeat=batch_size))
        return torch.cat(
            latent,
            dim=-1,  # feature dimension
        )

    def _create_processor(
        self,
        mapper: nn.Module,
        data: Tuple[torch.Tensor],
        edge_index: int,
        edge_inc: int,
        edge_attr: torch.Tensor,
        batch_size: int,
        use_reentrant: bool = False,
    ):
        """Create processor from checkpoint.

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
        batch_size: int
            Batch size
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
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [edge_index + i * edge_inc for i in range(batch_size)],
                dim=1,
            ),
            edge_attr=edge_attr,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward operator.

        Args:
            x: input tensor, shape (bs, e, m, n, f)
        Returns:
            Output tensor
        """
        bs, e = x.shape[0], x.shape[1]
        bse = bs * e  # merge the batch and ensemble dimensions

        if self.dropout_h2e > 0:
            isolated_nodes_h2e = True

            total_num_h2e_nodes = self.h2e_edge_index.max() + 1

            while isolated_nodes_h2e:
                edge_index_h2e_edge, edge_h2e_mask = dropout_edge(self.h2e_edge_index, p=self.dropout_h2e)
                isolated_nodes_h2e = contains_isolated_nodes(edge_index_h2e_edge, num_nodes=total_num_h2e_nodes)

        else:
            edge_index_h2e_edge, edge_h2e_mask = dropout_edge(self.h2e_edge_index, p=0)

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
        edge_h_to_e_latent = self._fuse_trainable_tensors(
            mask_select(self.h2e_edge_attr, 0, edge_h2e_mask), bse, mask_select(self.h2e_trainable, 0, edge_h2e_mask)
        )

        x_era_latent, x_latent = self._create_processor(
            self.forward_mapper,
            (x_era_latent, x_h_latent),
            self.e2h_edge_index,
            self._e2h_edge_inc,
            edge_e_to_h_latent,
            bse,
        )

        # generate noise tensor
        z = torch.randn(*x_latent.shape[:-1], self.noise_channels).type_as(x_latent)

        x_latent_proc = self.h_processor(
            # concat noise tensor to the latent features
            x_noisy=torch.cat([x_latent, z], dim=-1),
            edge_index=torch.cat(
                [edge_index_h2h_edge + i * self._h2h_edge_inc for i in range(bse)],
                dim=1,
            ),
            edge_attr=edge_h_to_h_latent,
        )

        # skip connection (H -> H)
        x_latent_proc = x_latent_proc + x_latent

        _, x_out = self._create_processor(
            self.backward_mapper,
            (x_latent_proc, x_era_latent),
            edge_index_h2e_edge,
            self._h2e_edge_inc,
            edge_h_to_e_latent,
            bse,
        )

        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=bs)

        # residual connection (just for the predicted variables at the current step)
        # x.shape = (bs, e, m, n, f)
        return x_out + x[:, :, -1, :, : self.in_channels]


if __name__ == "__main__":

    def dropout_edge_force_undirected(
        edge_index: torch.Tensor, p: float = 0.5, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, edge_mask = dropout_edge(edge_index, p=p, training=training)
        row, col = edge_index

        edge_mask[row > col] = False

        edge_index = edge_index[:, edge_mask]

        edge_index_concat = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        return edge_index_concat, edge_mask

    edge_tensor = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    edge_index_mc, edge_mask_mc = dropout_edge_force_undirected(edge_tensor, p=0.2)

    edge_index_torch, edge_mask_torch = dropout_edge(edge_tensor, p=0.2, force_undirected=True)
