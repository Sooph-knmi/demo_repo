import einops
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from aifs.model.layers import (
    MessagePassingProcessor,
    MessagePassingMapper,
    MessagePassingMLP,
)
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


class GraphMSG(nn.Module):
    def __init__(
        self,
        graph_data: HeteroData,
        in_channels: int,
        aux_in_channels: int,
        encoder_num_layers: int,
        encoder_hidden_channels: int,
        encoder_out_channels: int,
        activation: str,
        mlp_extra_layers: int = 0,
        encoder_mapper_num_layers: int = 1,
    ) -> None:
        super().__init__()

        LOGGER.debug("self.in_channels + self.aux_channels == %d", in_channels + aux_in_channels)

        # create mappings
        self._graph_data = graph_data
        self.in_channels = in_channels

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

        self.era_trainable_size = 8
        self.era_trainable = nn.Parameter(torch.zeros(self._era_size,self.era_trainable_size))
        self.h3_trainable_size = 8
        self.h3_trainable  = nn.Parameter(torch.zeros(self._h_size,self.h3_trainable_size))
        self.e2h_trainable_size = 8
        self.e2h_trainable  = nn.Parameter(torch.zeros(self._graph_data[("era", "to", "h")].edge_attr.shape[0],self.e2h_trainable_size))
        self.h2e_trainable_size = 8
        self.h2e_trainable  = nn.Parameter(torch.zeros(self._graph_data[("h", "to", "era")].edge_attr.shape[0],self.h2e_trainable_size))


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

        LOGGER.debug(f"----> {self.h_latlons.shape[1]}, f{self.e2h_edge_attr.shape[1]}")

        # latent nodes:
        self.node_era_embedder = MessagePassingMLP(
            in_channels=in_channels + aux_in_channels + self.era_latlons.shape[1] + self.era_trainable_size,
            latent_dim=encoder_out_channels,
            out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
        )

        self.node_h_embedder = MessagePassingMLP(
            in_channels=self.h_latlons.shape[1] + self.h3_trainable_size,
            latent_dim=encoder_out_channels,
            out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
        )  # position channels only

        self.edge_era_to_h_embedder = MessagePassingMLP(
            in_channels=self.e2h_edge_attr.shape[1] + self.e2h_trainable_size,
            latent_dim=encoder_out_channels,
            out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
        )

        self.edge_h_to_h_embedder = MessagePassingMLP(
            in_channels=self.h2h_edge_attr.shape[1],
            latent_dim=encoder_out_channels,
            out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
        )

        self.edge_h_to_era_embedder = MessagePassingMLP(
            in_channels=self.h2e_edge_attr.shape[1]+ self.h2e_trainable_size,
            latent_dim=encoder_out_channels,
            out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
        )

        # extract features:
        self.node_era_extractor = MessagePassingMLP(
            in_channels=encoder_out_channels,
            latent_dim=encoder_out_channels,
            out_channels=in_channels,
            mlp_extra_layers=mlp_extra_layers + 1,  # add decoder head
            activation=activation,
            layer_norm=False,
            final_activation=False,
        )

        # ERA -> H
        self.forward_mapper = MessagePassingMapper(
            hidden_dim=encoder_out_channels,
            hidden_layers=encoder_mapper_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            chunks=1,
            activation=activation,
        )

        # H -> H
        self.h_processor = MessagePassingProcessor(
            hidden_dim=encoder_hidden_channels,
            hidden_layers=encoder_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            chunks=2,
            activation=activation,
        )

        # H -> ERA5
        self.backward_mapper = MessagePassingMapper(
            hidden_dim=encoder_out_channels,
            hidden_layers=encoder_mapper_num_layers,
            mlp_extra_layers=mlp_extra_layers,
            chunks=1,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        x_in = einops.rearrange(x, "b n f -> (b n) f")

        # add ERA positional info (lat/lon)
        x_in = torch.cat(
            [
                x_in,
                einops.repeat(self.era_latlons, "e f -> (repeat e) f", repeat=bs),
                einops.repeat(self.era_trainable, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        x_era_latent = self.node_era_embedder(x_in)
        x_h_latent = self.node_h_embedder(
             torch.cat(
            [           
                einops.repeat(self.h_latlons, "e f -> (repeat e) f", repeat=bs),
                einops.repeat(self.h3_trainable, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
            )
        )

        edge_era_to_h_latent = self.edge_era_to_h_embedder(
             torch.cat(
            [           
                einops.repeat(self.e2h_edge_attr, "e f -> (repeat e) f", repeat=bs),
                einops.repeat(self.e2h_trainable, "e f -> (repeat e) f", repeat=bs)
            ],
            dim=-1,  # feature dimension
            )
        )  # copy edge attributes bs times
        (x_era_latent, x_latent) = self.forward_mapper(
            (x_era_latent, x_h_latent),
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self.e2h_edge_index + i * self._e2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_era_to_h_latent,
        )

        edge_h_to_h_latent = self.edge_h_to_h_embedder(einops.repeat(self.h2h_edge_attr, "e f -> (repeat e) f", repeat=bs))
        x_latent_proc = self.h_processor(  # has skipped connections
            x=x_latent,
            edge_index=torch.cat(
                [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_h_to_h_latent,
        )

        # added skip connection (H -> H)
        x_latent_proc = x_latent_proc + x_latent

        edge_h_to_e_latent = self.edge_h_to_era_embedder(
             torch.cat(
            [           
                einops.repeat(self.h2e_edge_attr, "e f -> (repeat e) f", repeat=bs),
                einops.repeat(self.h2e_trainable, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
            )
        )
        (_, x_out) = self.backward_mapper(
            x=(x_latent_proc, x_era_latent),
            edge_index=torch.cat(
                [self.h2e_edge_index + i * self._h2e_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=edge_h_to_e_latent,
        )

        x_out = self.node_era_extractor(x_out)
        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=bs)

        # residual connection (just for the predicted variables)
        return x_out + x[..., : self.in_channels]
