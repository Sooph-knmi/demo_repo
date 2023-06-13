import einops
import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData

from torch.utils.checkpoint import checkpoint

from torch.autograd.graph import save_on_cpu

from aifs.model.layers import (
    MessagePassingProcessor,
    MessagePassingMapper,
)
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


class GraphMSG(nn.Module):
    def __init__(
        self,
        graph_data: HeteroData,
        in_channels: int,
        aux_in_channels: int,
        multistep: int,
        encoder_num_layers: int,
        encoder_hidden_channels: int,
        encoder_out_channels: int,
        activation: str,
        mlp_extra_layers: int = 0,
        encoder_mapper_num_layers: int = 1,
        era_trainable_size: int = 8,
        h_trainable_size: int = 8,
        e2h_trainable_size: int = 8,
        h2e_trainable_size: int = 8,
        h2h_trainable_size: int = 0,
    ) -> None:
        super().__init__()

        LOGGER.debug("self.in_channels + self.aux_channels == %d", in_channels + aux_in_channels)

        # create mappings
        self._graph_data = graph_data
        self.in_channels = in_channels
        self.mstep = multistep

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

        self.era_trainable_size = era_trainable_size
        self.era_trainable = (
            nn.Parameter(torch.zeros(self._era_size, self.era_trainable_size)) if self.era_trainable_size > 0 else None
        )

        self.h_trainable_size = h_trainable_size
        self.h_trainable = nn.Parameter(torch.zeros(self._h_size, self.h_trainable_size)) if self.h_trainable_size > 0 else None

        self.e2h_trainable_size = e2h_trainable_size
        self.e2h_trainable = (
            nn.Parameter(torch.zeros(self._graph_data[("era", "to", "h")].edge_attr.shape[0], self.e2h_trainable_size))
            if self.e2h_trainable_size > 0
            else None
        )

        self.h2e_trainable_size = h2e_trainable_size
        self.h2e_trainable = (
            nn.Parameter(torch.zeros(self._graph_data[("h", "to", "era")].edge_attr.shape[0], self.h2e_trainable_size))
            if self.h2e_trainable_size > 0
            else None
        )

        self.h2h_trainable_size = h2h_trainable_size
        self.h2h_trainable = (
            nn.Parameter(torch.zeros(self._graph_data[("h", "to", "h")].edge_attr.shape[0], self.h2h_trainable_size))
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

        # ERA -> H
        self.forward_mapper = MessagePassingMapper(
            in_channels_src=self.mstep * (in_channels + aux_in_channels) + self.era_latlons.shape[1] + self.era_trainable_size,
            in_channels_dst=self.h_latlons.shape[1] + self.h_trainable_size,
            hidden_dim=encoder_out_channels,
            hidden_layers=encoder_mapper_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.e2h_edge_attr.shape[1] + self.e2h_trainable_size,
            chunks=1,
            activation=activation,
        )

        # H -> H
        self.h_processor = MessagePassingProcessor(
            hidden_dim=encoder_out_channels,
            hidden_layers=encoder_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2h_edge_attr.shape[1] + self.h2h_trainable_size,
            chunks=2,
            activation=activation,
        )

        # H -> ERA5
        self.backward_mapper = MessagePassingMapper(
            in_channels_src=encoder_out_channels,
            in_channels_dst=encoder_out_channels,
            out_channels_dst=in_channels,
            hidden_dim=encoder_out_channels,
            hidden_layers=encoder_mapper_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=self.h2e_edge_attr.shape[1] + self.h2e_trainable_size,
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
