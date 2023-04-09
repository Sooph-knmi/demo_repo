from typing import Optional, Any, Dict

import einops
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from gnn_era5.architecture.layers import TransformerMapper, GATEncoder
from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__)
GRID_NAMES = ["o160", "o80", "o48", "o32"]


class GraphUNet(nn.Module):
    def __init__(
        self,
        multigraph_data: HeteroData,
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
    ) -> None:
        super().__init__()

        LOGGER.debug("self.in_channels + self.aux_channels == %d", in_channels + aux_in_channels)

        # create mappings
        self._graph_data = multigraph_data
        self.in_channels = in_channels
        self.pos_channels = 4

        self._o160_size = self._graph_data[("o160", "to", "o160")].coords_rad.shape[0]
        self._o80_size = self._graph_data[("o80", "to", "o80")].coords_rad.shape[0]
        self._o48_size = self._graph_data[("o48", "to", "o48")].coords_rad.shape[0]
        self._o32_size = self._graph_data[("o32", "to", "o32")].coords_rad.shape[0]

        self._register_graph_buffers()

        # mappers
        # down
        self._o160_o80_mapper = TransformerMapper(
            in_channels + aux_in_channels + self.pos_channels,
            out_channels=encoder_out_channels,
            context_size=self._o80_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )

        self._o80_o48_mapper = TransformerMapper(
            encoder_out_channels + self.pos_channels,
            out_channels=encoder_out_channels,
            context_size=self._o48_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )

        self._o48_o32_mapper = TransformerMapper(
            encoder_out_channels + self.pos_channels,
            out_channels=encoder_out_channels,
            context_size=self._o32_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )
        # up
        self._o32_o48_mapper = TransformerMapper(
            encoder_out_channels + self.pos_channels,
            out_channels=encoder_out_channels,
            context_size=self._o48_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )

        self._o48_o80_mapper = TransformerMapper(
            encoder_out_channels + self.pos_channels,
            out_channels=encoder_out_channels,
            context_size=self._o80_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )

        self._o80_o160_mapper = TransformerMapper(
            encoder_out_channels + self.pos_channels,
            out_channels=self.in_channels,
            context_size=self._o160_size,
            trainable_context_channels=1,
            dynamic_context_channels=4 if use_dynamic_context else 0,
        )

        # encoders (on hidden meshes)
        self._o80_encoder = GATEncoder(
            num_layers=encoder_num_layers,
            in_channels=encoder_out_channels,
            hidden_channels=encoder_hidden_channels,
            out_channels=encoder_out_channels,
            num_heads=encoder_num_heads,
            dropout=encoder_dropout,
            activation=encoder_activation,
            jk_mode=encoder_jk_mode,
        )

        self._o48_encoder = GATEncoder(
            num_layers=encoder_num_layers,
            in_channels=encoder_out_channels,
            hidden_channels=encoder_hidden_channels,
            out_channels=encoder_out_channels,
            num_heads=encoder_num_heads,
            dropout=encoder_dropout,
            activation=encoder_activation,
            jk_mode=encoder_jk_mode,
        )

        self._o32_encoder = GATEncoder(
            num_layers=encoder_num_layers,
            in_channels=encoder_out_channels,
            hidden_channels=encoder_hidden_channels,
            out_channels=encoder_out_channels,
            num_heads=encoder_num_heads,
            dropout=encoder_dropout,
            activation=encoder_activation,
            jk_mode=encoder_jk_mode,
        )

        # ------------------------------------------------------------
        # TODO: add learnable positional encoding for ERA mesh?
        # ------------------------------------------------------------

    def _register_graph_buffers(self) -> None:
        for s_t_map in [(s, "to", t) for s in GRID_NAMES for t in GRID_NAMES]:
            if self._graph_data[s_t_map]:
                self.register_buffer(
                    f"_{s_t_map[0]}_{s_t_map[2]}_edge_index", self._graph_data[s_t_map].edge_index, persistent=True
                )
                self.register_buffer(f"_{s_t_map[0]}_{s_t_map[2]}_edge_attr", self._graph_data[s_t_map].edge_attr, persistent=True)

        for s_t_map in [(s, "to", t) for s in GRID_NAMES for t in GRID_NAMES]:
            if self._graph_data[s_t_map]:
                self.register_buffer(
                    f"_{s_t_map[0]}_{s_t_map[2]}_edge_inc",
                    torch.from_numpy(
                        np.asarray(
                            [
                                [self._graph_data[(s_t_map[0], "to", s_t_map[0])].coords_rad.shape[0]],
                                [self._graph_data[(s_t_map[2], "to", s_t_map[2])].coords_rad.shape[0]],
                            ],
                            dtype=np.int64,
                        )
                    ),
                    persistent=True,
                )

        for s in GRID_NAMES:
            self.register_buffer(
                f"_{s}_latlons",
                torch.cat(
                    [
                        torch.as_tensor(np.sin(self._graph_data[(s, "to", s)].coords_rad)),
                        torch.as_tensor(np.cos(self._graph_data[(s, "to", s)].coords_rad)),
                    ],
                    dim=-1,
                ),
                persistent=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        x_in = einops.rearrange(x, "b n f -> (b n) f")

        # x_in = x_in + self.era_pos_embed[None, ...]

        # add ERA (o160) positional info (lat/lon)
        x_in = torch.cat(
            [
                x_in,
                einops.repeat(self._o160_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        # ----------------------------------------------------------------------------
        # "down" branch
        # ERA (O160) -> O80
        # ----------------------------------------------------------------------------
        x_80 = self._o160_o80_mapper(
            x_in,
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self._o160_o80_edge_index + i * self._o160_o80_edge_inc for i in range(bs)],
                dim=1,
            ),
            # copy edge attributes bs times
            edge_attr=einops.repeat(self._o160_o80_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self._o80_latlons,
            batch_size=bs,
        )

        x_80_proc = self._o80_encoder(
            x=x_80,
            edge_index=torch.cat(
                [self._o80_o80_edge_index + i * self._o80_o80_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self._o80_o80_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

        # added skip connection (o80 -> o80)
        x_80_proc = x_80_proc + x_80

        # add positional info (lat/lon) for the hidden grid
        x_80_proc = torch.cat(
            [
                x_80_proc,
                einops.repeat(self._o80_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        # ----------------------------------------------------------------------------
        # o80 -> o48
        # ----------------------------------------------------------------------------
        x_48 = self._o80_o48_mapper(
            x_80_proc,
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self._o80_o48_edge_index + i * self._o80_o48_edge_inc for i in range(bs)],
                dim=1,
            ),
            # copy edge attributes bs times
            edge_attr=einops.repeat(self._o80_o48_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self._o48_latlons,
            batch_size=bs,
        )

        x_48_proc = self._o48_encoder(
            x=x_48,
            edge_index=torch.cat(
                [self._o48_o48_edge_index + i * self._o48_o48_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self._o48_o48_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

        # skip connection (o48 -> o48)
        x_48_proc = x_48_proc + x_48

        # add positional info (lat/lon) for the hidden grid
        x_48_proc = torch.cat(
            [
                x_48_proc,
                einops.repeat(self._o48_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        # ----------------------------------------------------------------------------
        # o48 -> o32
        # ----------------------------------------------------------------------------
        x_32 = self._o48_o32_mapper(
            x_48_proc,
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self._o48_o32_edge_index + i * self._o48_o32_edge_inc for i in range(bs)],
                dim=1,
            ),
            # copy edge attributes bs times
            edge_attr=einops.repeat(self._o48_o32_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self._o32_latlons,
            batch_size=bs,
        )

        x_32_proc = self._o32_encoder(
            x=x_32,
            edge_index=torch.cat(
                [self._o32_o32_edge_index + i * self._o32_o32_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self._o32_o32_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

        # skip connection (o32 -> o32)
        x_32_proc = x_32_proc + x_32

        x_32_proc = torch.cat(
            [
                x_32_proc,
                einops.repeat(self._o32_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )

        # ----------------------------------------------------------------------------
        # "up" branch
        # o32 -> o48
        # ----------------------------------------------------------------------------
        x_48_up = self._o32_o48_mapper(
            x_32_proc,
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self._o32_o48_edge_index + i * self._o32_o48_edge_inc for i in range(bs)],
                dim=1,
            ),
            # copy edge attributes bs times
            edge_attr=einops.repeat(self._o32_o48_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self._o48_latlons,
            batch_size=bs,
        )

        # u-net residual
        x_48_up = x_48_up + x_48_proc[..., : -self.pos_channels]
        x_48_up = torch.cat(
            [
                x_48_up,
                einops.repeat(self._o48_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )
        # LOGGER.debug("x_48_up.shape = %s", x_48_up.shape)

        # ----------------------------------------------------------------------------
        # o48 -> o80
        # ----------------------------------------------------------------------------
        x_80_up = self._o48_o80_mapper(
            x_48_up,
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self._o48_o80_edge_index + i * self._o48_o80_edge_inc for i in range(bs)],
                dim=1,
            ),
            # copy edge attributes bs times
            edge_attr=einops.repeat(self._o48_o80_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self._o80_latlons,
            batch_size=bs,
        )

        # u-net residual
        x_80_up = x_80_up + x_80_proc[..., : -self.pos_channels]
        x_80_up = torch.cat(
            [
                x_80_up,
                einops.repeat(self._o80_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )
        # LOGGER.debug("x_80_up.shape = %s", x_80_up.shape)

        # ----------------------------------------------------------------------------
        # o80 -> o160
        # ----------------------------------------------------------------------------
        x_out = self._o80_o160_mapper(
            x_80_up,
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self._o80_o160_edge_index + i * self._o80_o160_edge_inc for i in range(bs)],
                dim=1,
            ),
            # copy edge attributes bs times
            edge_attr=einops.repeat(self._o80_o160_edge_attr, "e f -> (repeat e) f", repeat=bs),
            dynamic_context=self._o160_latlons,
            batch_size=bs,
        )

        x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=bs)

        # final residual connection (just for the predicted variables)
        return x_out + x[..., : self.in_channels]


if __name__ == "__main__":
    # you'll need to run this test on a worker node
    import os

    num_inputs = 32
    num_aux_inputs = 4

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    input_dir = "/ec/res4/hpcperm/syma/gnn/"
    graph_mappings = torch.load(os.path.join(input_dir, "graph_mappings_normed_edge_attrs_gauss_grids.pt"))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.debug("Running on device: %s ...", device)

    gnn = GraphUNet(
        multigraph_data=graph_mappings,
        in_channels=num_inputs,
        aux_in_channels=num_aux_inputs,
        encoder_num_layers=6,
        encoder_hidden_channels=128,
        encoder_out_channels=128,
        encoder_num_heads=8,
    ).to(device)

    _O160_SIZE = graph_mappings[("o160", "to", "o160")].coords_rad.shape[0]

    x_input = torch.randn(4, _O160_SIZE, num_inputs + num_aux_inputs).to(device)  # input tensor
    LOGGER.debug("Input shape: %s", x_input.shape)
    y_pred = gnn(x_input)
    LOGGER.debug("Output shape: %s", y_pred.shape)
    LOGGER.debug("Model parameter count: %d", count_parameters(gnn))
    loss = y_pred.sum()
    LOGGER.debug("Running backward on a dummy loss ...")
    loss.backward()
    LOGGER.debug("Ran backward. All good!")
