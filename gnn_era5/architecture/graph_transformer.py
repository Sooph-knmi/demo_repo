from typing import Optional, Any, Dict

import einops
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from gnn_era5.architecture.layers import TransformerMapper, GATEncoder, MessagePassingEncoder
from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__)


class MixedTransformer(nn.Module):
    def __init__(
        self,
        graph_data: HeteroData,
        in_channels: int,
        aux_in_channels: int,
        encoder_num_layers: int,
        encoder_hidden_channels: int,
        encoder_out_channels: int,
        encoder_num_heads: int = 2,
        encoder_dropout: float = 0.0,
        encoder_activation: Optional[str] = "gelu",
        encoder_jk_mode: Optional[str] = "last",
        use_dynamic_context: bool = True,
        encoder_type: Optional[str] = "GAT",
    ) -> None:
        super().__init__()

        LOGGER.debug("self.in_channels + self.aux_channels == %d", in_channels + aux_in_channels)

        # create mappings
        self._graph_data = graph_data
        self.in_channels = in_channels
        self.pos_channels = 4

        self.register_buffer("e2e_edge_index", self._graph_data[("o160", "to", "o160")].edge_index, persistent=False)
        self.register_buffer("h2e_edge_index", self._graph_data[("h", "to", "o160")].edge_index, persistent=False)
        self.register_buffer("e2h_edge_index", self._graph_data[("o160", "to", "h")].edge_index, persistent=False)
        self.register_buffer("h2h_edge_index", self._graph_data[("h", "to", "h")].edge_index, persistent=False)

        self.register_buffer("e2e_edge_attr", self._graph_data[("o160", "to", "o160")].edge_attr, persistent=False)
        self.register_buffer("h2e_edge_attr", self._graph_data[("h", "to", "o160")].edge_attr, persistent=False)
        self.register_buffer("e2h_edge_attr", self._graph_data[("o160", "to", "h")].edge_attr, persistent=False)
        self.register_buffer("h2h_edge_attr", self._graph_data[("h", "to", "h")].edge_attr, persistent=False)

        self._era_size = self._graph_data[("o160", "to", "o160")].ecoords_rad.shape[0]
        self._h_size = self._graph_data[("h", "to", "h")].hcoords_rad.shape[0]

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
                    torch.as_tensor(np.sin(self._graph_data[("o160", "to", "o160")].ecoords_rad)),
                    torch.as_tensor(np.cos(self._graph_data[("o160", "to", "o160")].ecoords_rad)),
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

        # Latent graph (ERA5 -> H)
        self.forward_mapper = TransformerMapper(
            in_channels + aux_in_channels + self.pos_channels,
            out_channels=encoder_out_channels,
            context_size=self._h_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )

        # H -> H
        if encoder_type == "GAT":
            self.h_encoder = GATEncoder(
                num_layers=encoder_num_layers,
                in_channels=encoder_out_channels,
                hidden_channels=encoder_hidden_channels,
                out_channels=encoder_out_channels,
                num_heads=encoder_num_heads,
                dropout=encoder_dropout,
                activation=encoder_activation,
                jk_mode=encoder_jk_mode,
            )
        else:
            self.h_encoder = MessagePassingEncoder(
                in_channels=encoder_out_channels,
                out_channels=encoder_out_channels,
                hidden_dim=encoder_hidden_channels,
                edge_dim=3,
                proc_layers=encoder_num_layers,
            )

        # H -> ERA5
        self.backward_mapper = TransformerMapper(
            in_channels=encoder_out_channels + self.pos_channels,
            out_channels=in_channels,  # leave out the auxiliary and positional info
            context_size=self._era_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )

        # trainable positional embedding
        self.era_pos_embed = nn.Parameter(torch.zeros(self._era_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        x_in = einops.rearrange(x, "b n f -> (b n) f")

        # x_in = x_in + self.era_pos_embed[None, ...]

        # add ERA positional info (lat/lon)
        x_in = torch.cat(
            [
                x_in,
                einops.repeat(self.era_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        # encoder: processes era5 data
        x_latent = self.forward_mapper(
            x_in,
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self.e2h_edge_index + i * self._e2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            # copy edge attributes bs times
            edge_attr=einops.repeat(self.e2h_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self.h_latlons,
            batch_size=bs,
        )

        x_latent_proc = self.h_encoder(
            x=x_latent,
            edge_index=torch.cat(
                [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h2h_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

        # added skip connection (H -> H)
        x_latent_proc = x_latent_proc + x_latent

        # add positional info (lat/lon) for the hidden grid
        x_latent_proc = torch.cat(
            [
                x_latent_proc,
                einops.repeat(self.h_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        x_out = self.backward_mapper(
            x=x_latent_proc,
            edge_index=torch.cat(
                [self.h2e_edge_index + i * self._h2e_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h2e_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self.era_latlons,
            batch_size=bs,
        )

        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=bs)

        # residual connection (just for the predicted variables)
        return x_out + x[..., : self.in_channels]


# if __name__ == "__main__":
#     from gnn_era5.utils.constants import _ERA_O160_LATLON

#     tgnn = MixedTransformer(
#         in_channels=2,
#         aux_in_channels=0,
#         encoder_num_layers=2,
#         encoder_hidden_channels=32,
#         encoder_out_channels=32,
#         encoder_num_heads=2,
#     )

#     x = torch.randn(1, _ERA_O160_LATLON, 2)  # input tensor
#     LOGGER.debug(x.norm())
#     y_pred = tgnn(x)
#     LOGGER.debug(x.norm())
#     LOGGER.debug(y_pred.shape)
#     y_pred.sum().backward()
