import einops
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from aifs.model.layers import MessagePassingMapper
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


class GraphAE(nn.Module):
    def __init__(
        self,
        graph_data: HeteroData,
        in_channels: int,
        aux_in_channels: int,
        encoder_out_channels: int,
        activation: str,
        mlp_extra_layers: int = 0,
        mapper_num_layers: int = 1,
    ) -> None:
        super().__init__()

        LOGGER.debug("self.in_channels + self.aux_channels == %d", in_channels + aux_in_channels)

        # create mappings
        self._graph_data = graph_data
        self.in_channels = in_channels

        self.register_buffer("h2e_edge_index", self._graph_data[("h", "to", "era")].edge_index, persistent=False)
        self.register_buffer("e2h_edge_index", self._graph_data[("era", "to", "h")].edge_index, persistent=False)

        self.register_buffer("h2e_edge_attr", self._graph_data[("h", "to", "era")].edge_attr, persistent=False)
        self.register_buffer("e2h_edge_attr", self._graph_data[("era", "to", "h")].edge_attr, persistent=False)

        self._era_size = self._graph_data[("era", "to", "era")].ecoords_rad.shape[0]
        self._h_size = self._graph_data[("h", "to", "h")].hcoords_rad.shape[0]

        self.register_buffer(
            "_e2h_edge_inc", torch.from_numpy(np.asarray([[self._era_size], [self._h_size]], dtype=np.int64)), persistent=False
        )
        self.register_buffer(
            "_h2e_edge_inc", torch.from_numpy(np.asarray([[self._h_size], [self._era_size]], dtype=np.int64)), persistent=False
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

        # ERA -> H
        self.encoder = MessagePassingMapper(
            in_channels_src=in_channels + aux_in_channels + self.era_latlons.shape[1],
            in_channels_dst=self.h_latlons.shape[1],
            hidden_dim=encoder_out_channels,
            hidden_layers=mapper_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.e2h_edge_attr.shape[1],
            chunks=1,
            activation=activation,
        )

        # H -> ERA5
        self.decoder = MessagePassingMapper(
            in_channels_src=encoder_out_channels,
            in_channels_dst=encoder_out_channels,
            out_channels_dst=in_channels,
            hidden_dim=encoder_out_channels,
            hidden_layers=mapper_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2e_edge_attr.shape[1],
            backward_mapper=True,
            chunks=1,
            activation=activation,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes the weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        # reshape and add ERA positional info (lat/lon)
        x_in = torch.cat(
            [
                einops.rearrange(x, "b n f -> (b n) f"),
                einops.repeat(self.era_latlons, "n f -> (repeat n) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        # --------------
        # ENCODER
        # --------------
        x_era_latent, x_latent = checkpoint(
            self.encoder,
            (x_in, einops.repeat(self.h_latlons, "n f -> (repeat n) f", repeat=bs)),
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self.e2h_edge_index + i * self._e2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.e2h_edge_attr, "e f -> (repeat e) f", repeat=bs),
            use_reentrant=False,
        )

        # --------------
        # DECODER
        # --------------
        _, x_out = checkpoint(
            self.decoder,
            x=(x_latent, x_era_latent),
            edge_index=torch.cat(
                [self.h2e_edge_index + i * self._h2e_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h2e_edge_attr, "e f -> (repeat e) f", repeat=bs),
            use_reentrant=False,
        )

        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=bs)

        return x_out
