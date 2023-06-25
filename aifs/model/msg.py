import os
import einops
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from aifs.model.layers import (
    MessagePassingProcessor,
    MessagePassingMapper,
)
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


class GraphMSG(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        graph_data: HeteroData = None,
    ) -> None:
        super().__init__()

        # create mappings
        if graph_data is None:
            self.graph_data = torch.load(os.path.join(config.paths.graph, config.files.graph))
        else:
            self._graph_data = graph_data

        self.in_channels = config.data.num_features - config.data.num_aux_features
        self.multi_step = config.training.multistep_input
        self.aux_in_channels = config.data.num_aux_features

        LOGGER.debug("self.in_channels + self.aux_channels == %d", self.in_channels + self.aux_in_channels)

        self.activation = config.model.activation

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

        self.era_trainable_size = config.model.trainable_parameters.era
        self.era_trainable = (
            nn.Parameter(torch.zeros(self._era_size, self.era_trainable_size)) if self.era_trainable_size > 0 else None
        )

        self.h_trainable_size = config.model.trainable_parameters.hidden
        self.h_trainable = nn.Parameter(torch.zeros(self._h_size, self.h_trainable_size)) if self.h_trainable_size > 0 else None

        self.e2h_trainable_size = config.model.trainable_parameters.era2hidden
        self.e2h_trainable = (
            nn.Parameter(torch.zeros(self._graph_data[("era", "to", "h")].edge_attr.shape[0], self.e2h_trainable_size))
            if self.e2h_trainable_size > 0
            else None
        )

        self.h2e_trainable_size = config.model.trainable_parameters.hidden2era
        self.h2e_trainable = (
            nn.Parameter(torch.zeros(self._graph_data[("h", "to", "era")].edge_attr.shape[0], self.h2e_trainable_size))
            if self.h2e_trainable_size > 0
            else None
        )

        self.h2h_trainable_size = config.model.trainable_parameters.hidden2hidden
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
            hidden_dim=encoder_out_channels + 1,  # includes a noise channel
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
        """
        Forward operator.
        Args:
            x: input tensor, shape (bs, e, m, n, f)
        Returns:
            Output tensor
        """
        bs, e = x.shape[0], x.shape[1]
        bse = bs * e  # merge the batch and ensemble dimensions

        # add ERA positional info (lat/lon)
        x_in = [
            einops.rearrange(x, "bs e m n f -> (bs e n) (m f)"),
            einops.repeat(self.era_latlons, "n f -> (repeat n) f", repeat=bse),
        ]
        if self.era_trainable is not None:
            x_in.append(einops.repeat(self.era_trainable, "n f -> (repeat n) f", repeat=bse))

        x_era_latent = torch.cat(
            x_in,
            dim=-1,  # feature dimension
        )

        x_h_latent = [einops.repeat(self.h_latlons, "e f -> (repeat e) f", repeat=bse)]
        if self.h_trainable is not None:
            x_h_latent.append(einops.repeat(self.h_trainable, "e f -> (repeat e) f", repeat=bse))
        x_h_latent = torch.cat(
            x_h_latent,
            dim=-1,  # feature dimension
        )

        edge_era_to_h_latent = [einops.repeat(self.e2h_edge_attr, "e f -> (repeat e) f", repeat=bse)]
        if self.e2h_trainable is not None:
            edge_era_to_h_latent.append(einops.repeat(self.e2h_trainable, "e f -> (repeat e) f", repeat=bse))
        edge_era_to_h_latent = torch.cat(
            edge_era_to_h_latent,
            dim=-1,  # feature dimension
        )  # copy edge attributes bs times

        (x_era_latent, x_latent) = checkpoint(
            self.forward_mapper,
            (x_era_latent, x_h_latent),
            # expand edge index correct number of times while adding the proper number to the edge index
            edge_index=torch.cat(
                [self.e2h_edge_index + i * self._e2h_edge_inc for i in range(bse)],
                dim=1,
            ),
            edge_attr=edge_era_to_h_latent,
            use_reentrant=False,
        )

        edge_h_to_h_latent = [einops.repeat(self.h2h_edge_attr, "e f -> (repeat e) f", repeat=bse)]
        if self.h2h_trainable is not None:
            edge_h_to_h_latent.append(einops.repeat(self.h2h_trainable, "e f -> (repeat e) f", repeat=bse))
        edge_h_to_h_latent = torch.cat(
            edge_h_to_h_latent,
            dim=-1,  # feature dimension
        )

        # generate noise tensor
        z = torch.randn(*x_latent.shape[:-1], 1).type_as(x_latent)

        x_latent_proc = self.h_processor(
            # concat noise tensor to the latent features
            x=torch.cat([x_latent, z], dim=-1),
            edge_index=torch.cat(
                [self.h2h_edge_index + i * self._h2h_edge_inc for i in range(bse)],
                dim=1,
            ),
            edge_attr=edge_h_to_h_latent,
        )

        # skip connection (H -> H), leaves out the last output channel
        # workaround needed b/c the h_processor inputs and outputs have equal channel size
        # TODO [Mihai]: if you see a cleaner way to inject the noise, i'm all ears
        x_latent_proc = x_latent_proc[..., :-1] + x_latent

        edge_h_to_e_latent = [einops.repeat(self.h2e_edge_attr, "e f -> (repeat e) f", repeat=bse)]
        if self.h2e_trainable is not None:
            edge_h_to_e_latent.append(einops.repeat(self.h2e_trainable, "e f -> (repeat e) f", repeat=bse))
        edge_h_to_e_latent = torch.cat(
            edge_h_to_e_latent,
            dim=-1,  # feature dimension
        )
        _, x_out = checkpoint(
            self.backward_mapper,
            x=(x_latent_proc, x_era_latent),
            edge_index=torch.cat(
                [self.h2e_edge_index + i * self._h2e_edge_inc for i in range(bse)],
                dim=1,
            ),
            edge_attr=edge_h_to_e_latent,
            use_reentrant=False,
        )

        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=bs)

        # residual connection (just for the predicted variables at the current step)
        # x.shape = (bs, e, m, n, f)
        return x_out + x[:, :, -1, :, : self.in_channels]
