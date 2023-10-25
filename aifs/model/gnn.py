from typing import Optional
from typing import Tuple

import einops
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from aifs.model.layers import MessagePassingMapper
from aifs.model.layers import MessagePassingProcessor
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

        # Processor H -> H
        self.h_processor = MessagePassingProcessor(
            hidden_dim=encoder_out_channels + self.noise_channels,  # includes the noise channels
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward operator.

        Args:
            x: input tensor, shape (bs, e, m, n, f)
        Returns:
            Output tensor
        """
        bs, e = x.shape[0], x.shape[1]
        bse = bs * e  # merge the batch and ensemble dimensions

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
        edge_h_to_h_latent = self._fuse_trainable_tensors(self.h2h_edge_attr, bse, self.h2h_trainable)
        edge_h_to_e_latent = self._fuse_trainable_tensors(self.h2e_edge_attr, bse, self.h2e_trainable)

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
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bse)],
                dim=1,
            ),
            edge_attr=edge_h_to_h_latent,
        )

        # skip connection (H -> H), leaves out the last output channel
        # workaround needed b/c the h_processor inputs and outputs have equal channel size
        # TODO [Mihai]: if you see a cleaner way to inject the noise, i'm all ears
        x_latent_proc = x_latent_proc[..., : -self.noise_channels] + x_latent
        x_latent_proc = x_latent_proc + x_latent

        _, x_out = self._create_processor(
            self.backward_mapper,
            (x_latent_proc, x_era_latent),
            self.h2e_edge_index,
            self._h2e_edge_inc,
            edge_h_to_e_latent,
            bse,
        )

        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=bs)

        # residual connection (just for the predicted variables at the current step)
        # x.shape = (bs, e, m, n, f)
        return x_out + x[:, :, -1, :, : self.in_channels]
