from typing import Tuple

import einops
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import HeteroData

from aifs.diagnostics.logger import get_logger
from aifs.model.layers import MessagePassingMapper
from aifs.model.layers import MessagePassingProcessor

LOGGER = get_logger(__name__)


class Critic(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        graph_data: HeteroData = None,
    ) -> None:
        # ) -> None:
        super().__init__()

        self.config = config

        # create mappings
        if graph_data is None:
            self._graph_data = torch.load(os.path.join(config.hardware.paths.graph.critic, config.hardware.files.graph.critic))
        else:
            self._graph_data = graph_data

        self.in_channels = config.data.num_features - config.data.num_aux_features
        self.aux_in_channels = config.data.num_aux_features

        self._register_graph_buffers()

        # processors
        self.h33_processor = self._get_processor(
            hidden_dim=config.model.critic.num_channels,
            edge_dim=self.h33_h33_edge_attr.shape[1],
        )
        self.h32_processor = self._get_processor(
            hidden_dim=config.model.critic.num_channels * 2,
            edge_dim=self.h32_h32_edge_attr.shape[1],
        )
        self.h31_processor = self._get_processor(
            hidden_dim=config.model.critic.num_channels * 2,
            edge_dim=self.h31_h31_edge_attr.shape[1],
        )
        self.h30_processor = self._get_processor(
            hidden_dim=config.model.critic.num_channels * 4,
            edge_dim=self.h30_h30_edge_attr.shape[1],
        )

        # mappers (no trainable features, yet)
        self.h33_mapper = self._get_mapper(
            in_channels_src=self.in_channels + self.era_latlons.shape[1],
            in_channels_dst=self.h33_latlons.shape[1],
            hidden_dim=self.config.model.critic.num_channels,
            edge_dim=self.era_h33_edge_attr.shape[1],
        )
        self.h32_mapper = self._get_mapper(
            in_channels_src=config.model.critic.num_channels,
            in_channels_dst=self.h32_latlons.shape[1],
            hidden_dim=self.config.model.critic.num_channels * 2,
            edge_dim=self.h33_h32_edge_attr.shape[1],
        )
        self.h31_mapper = self._get_mapper(
            in_channels_src=config.model.critic.num_channels * 2,
            in_channels_dst=self.h31_latlons.shape[1],
            hidden_dim=self.config.model.critic.num_channels * 2,
            edge_dim=self.h32_h31_edge_attr.shape[1],
        )
        self.h30_mapper = self._get_mapper(
            in_channels_src=config.model.critic.num_channels * 2,
            in_channels_dst=self.h30_latlons.shape[1],
            hidden_dim=self.config.model.critic.num_channels * 4,
            edge_dim=self.h31_h30_edge_attr.shape[1],
        )

        self.final_layers = nn.Sequential(
            nn.Linear(config.model.critic.num_channels * 4, config.model.critic.num_channels * 2),
            nn.SiLU(),
            nn.Linear(config.model.critic.num_channels * 2, 1, bias=False),
        )

    def _register_graph_buffers(self) -> None:
        # mappings: _src[i] -> _dst[i]
        _src = ["era", "h33", "h32", "h31", "h30", "era", "h33", "h32", "h31"]
        _dst = ["era", "h33", "h32", "h31", "h30", "h33", "h32", "h31", "h30"]

        for s, d in zip(_src, _dst):
            sd_key = (s, "to", d)
            self.register_buffer(f"{s}_{d}_edge_index", self._graph_data[sd_key].edge_index, persistent=True)
            self.register_buffer(
                f"{s}_{d}_edge_inc",
                torch.from_numpy(
                    np.asarray(
                        [
                            [self._graph_data[(s, "to", s)].scoords_rad.shape[0]],
                            [self._graph_data[(d, "to", d)].scoords_rad.shape[0]],
                        ],
                        dtype=np.int64,
                    )
                ),
                persistent=True,
            )
            self.register_buffer(f"{s}_{d}_edge_attr", self._graph_data[sd_key].edge_attr, persistent=True)

        for s in ["era", "h33", "h32", "h31", "h30"]:
            self.register_buffer(
                f"{s}_latlons",
                torch.cat(
                    [
                        torch.as_tensor(np.sin(self._graph_data[(s, "to", s)].scoords_rad)),
                        torch.as_tensor(np.cos(self._graph_data[(s, "to", s)].scoords_rad)),
                    ],
                    dim=-1,
                ),
                persistent=True,
            )

    def _get_mapper(self, in_channels_src: int, in_channels_dst: int, hidden_dim: int, edge_dim: int) -> nn.Module:
        return MessagePassingMapper(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            hidden_layers=self.config.model.critic.encoder.num_layers,
            mlp_extra_layers=self.config.model.critic.mlp.extra_layers,
            edge_dim=edge_dim,
            chunks=1,
            activation=self.config.model.critic.activation,
        )

    def _get_processor(self, hidden_dim: int, edge_dim: int) -> nn.Module:
        return MessagePassingProcessor(
            hidden_dim=hidden_dim,
            hidden_layers=self.config.model.critic.hidden.num_layers,
            edge_dim=edge_dim,
            mlp_extra_layers=self.config.model.critic.mlp.extra_layers,
            chunks=2,
            activation=self.config.model.critic.activation,
        )

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = einops.rearrange(x, "b n f -> (b n) f")
        # add ERA positional info (lat/lon)
        x = torch.cat(
            [
                x,
                einops.repeat(self.era_latlons, "e f -> (repeat e) f", repeat=bs),
            ],
            dim=-1,  # feature dimension
        )
        return x

    def _era_h33_map(self, mapper: nn.Module, x: Tuple[torch.Tensor, torch.Tensor], bs: int) -> torch.Tensor:
        return mapper(
            x=x,
            edge_index=torch.cat(
                [self.era_h33_edge_index + i * self.era_h33_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.era_h33_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def _h33_proc(self, processor: nn.Module, x: torch.Tensor, bs: int) -> torch.Tensor:
        return processor(
            x=x,
            edge_index=torch.cat(
                [self.h33_h33_edge_index + i * self.h33_h33_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h33_h33_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def _h33_h32_map(self, mapper: nn.Module, x: Tuple[torch.Tensor, torch.Tensor], bs: int) -> torch.Tensor:
        return mapper(
            x=x,
            edge_index=torch.cat(
                [self.h33_h32_edge_index + i * self.h33_h32_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h33_h32_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def _h32_proc(self, processor: nn.Module, x: torch.Tensor, bs: int) -> torch.Tensor:
        return processor(
            x=x,
            edge_index=torch.cat(
                [self.h32_h32_edge_index + i * self.h32_h32_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h32_h32_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def _h32_h31_map(self, mapper: nn.Module, x: Tuple[torch.Tensor, torch.Tensor], bs: int) -> torch.Tensor:
        return mapper(
            x=x,
            edge_index=torch.cat(
                [self.h32_h31_edge_index + i * self.h32_h31_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h32_h31_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def _h31_proc(self, processor: nn.Module, x: torch.Tensor, bs: int) -> torch.Tensor:
        return processor(
            x=x,
            edge_index=torch.cat(
                [self.h31_h31_edge_index + i * self.h31_h31_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h31_h31_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def _h31_h30_map(self, mapper: nn.Module, x: Tuple[torch.Tensor, torch.Tensor], bs: int) -> torch.Tensor:
        return mapper(
            x=x,
            edge_index=torch.cat(
                [self.h31_h30_edge_index + i * self.h31_h30_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h31_h30_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def _h30_proc(self, processor: nn.Module, x: torch.Tensor, bs: int) -> torch.Tensor:
        return processor(
            x=x,
            edge_index=torch.cat(
                [self.h30_h30_edge_index + i * self.h30_h30_edge_inc for i in range(bs)],
                dim=1,
            ),
            edge_attr=einops.repeat(self.h30_h30_edge_attr, "e f -> (repeat e) f", repeat=bs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        x = self._reshape_input(x)

        # era -> h33 mapper
        x_h33_latent = einops.repeat(self.h33_latlons, "e f -> (repeat e) f", repeat=bs)
        _, x_h33_latent = self._era_h33_map(self.h33_mapper, (x, x_h33_latent), bs)

        # h33 processor (w/ skip connections)
        x_h33_latent_proc = self._h33_proc(self.h33_processor, x_h33_latent, bs)
        x_h33_latent_proc = x_h33_latent_proc + x_h33_latent

        # h33 -> h32 mapper
        x_h32_latent = einops.repeat(self.h32_latlons, "e f -> (repeat e) f", repeat=bs)
        _, x_h32_latent = self._h33_h32_map(self.h32_mapper, (x_h33_latent_proc, x_h32_latent), bs)

        # h32 processor
        x_h32_latent_proc = self._h32_proc(self.h32_processor, x_h32_latent, bs)
        x_h32_latent_proc = x_h32_latent_proc + x_h32_latent

        # h32 -> h31 mapper
        x_h31_latent = einops.repeat(self.h31_latlons, "e f -> (repeat e) f", repeat=bs)
        _, x_h31_latent = self._h32_h31_map(self.h31_mapper, (x_h32_latent_proc, x_h31_latent), bs)

        # h31 processors
        x_h31_latent_proc = self._h31_proc(self.h31_processor, x_h31_latent, bs)
        x_h31_latent_proc = x_h31_latent_proc + x_h31_latent

        # h31 -> h30 mapper
        x_h30_latent = einops.repeat(self.h30_latlons, "e f -> (repeat e) f", repeat=bs)
        _, x_h30_latent = self._h31_h30_map(self.h30_mapper, (x_h31_latent_proc, x_h30_latent), bs)

        # h30 processors
        x_h30_latent_proc = self._h30_proc(self.h30_processor, x_h30_latent, bs)
        x_h30_latent_proc = x_h30_latent_proc + x_h30_latent

        # mix the latents
        x_out = einops.rearrange(x_h30_latent_proc, "(b n) f -> b n f", b=bs)
        x_out = x_out.mean(dim=1)  # average over all nodes, resulting shape (bs, encoder_out_channels)
        x_out = self.final_layers(x_out)  # shape = (bs, 1)
        return x_out


if __name__ == "__main__":
    import os
    from prettytable import PrettyTable
    from hydra import compose, initialize
    from torch_geometric import seed_everything
    from aifs.model.generator import Generator

    seed_everything(1234)

    initialize(config_path="../config", job_name="test_gan")
    config_ = compose(
        config_name="gan",
        overrides=[
            "dataloader.batch_size.training=1",
            "data.num_features=11",
            "data.num_aux_features=3",
            "training.multistep_input=1",
            "model.generator.num_channels=16",
            "model.generator.noise_channels=2",
            "model.critic.num_channels=16",
            'hardware.paths.graph.generator="/home/mlx/data/graphs/"',
            'hardware.files.graph.generator="graph_mappings_normed_edge_attrs1_o96_h_0_1_2_3_4.pt"',
            'hardware.paths.graph.critic="/ec/res4/hpcperm/syma/gnn/"',
            'hardware.files.graph.critic="gan_critic_graph_mappings_normed_edge_attrs_o96_h_0_1_2_3.pt"',
        ],
    )

    LOGGER.debug("config_.data.num_aux_features = %d", config_.data.num_aux_features)

    bs_ = 2
    mstep = 1
    num_inputs = 8
    num_aux_inputs = 3
    num_noise_channels = 2

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters_pretty(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for pn, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([pn, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params/1.e6:.2f}M")
        return total_params

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.debug("Running on device: %s ...", device)

    generator_graph = torch.load(os.path.join(config_.hardware.paths.graph.generator, config_.hardware.files.graph.generator))
    critic_graph = torch.load(os.path.join(config_.hardware.paths.graph.critic, config_.hardware.files.graph.critic))

    generator = Generator(config=config_, graph_data=generator_graph).to(device)
    critic = Critic(config=config_, graph_data=critic_graph).to(device)

    LOGGER.debug("Generator parameter count: %d", count_parameters(generator))
    LOGGER.debug("Critic parameter count: %d", count_parameters(critic))

    _ERA_SIZE = generator_graph[("era", "to", "era")].ecoords_rad.shape[0]
    _H_SIZE = generator_graph[("h", "to", "h")].hcoords_rad.shape[0]

    LOGGER.debug("_ERA_SIZE: %d, _H_SIZE = %d", _ERA_SIZE, _H_SIZE)

    z = torch.randn(bs_ * _H_SIZE, num_noise_channels).to(device)
    x_in = torch.randn(bs_, mstep, _ERA_SIZE, num_inputs + num_aux_inputs).to(device)
    LOGGER.debug("Input shapes : z.shape = %s, x_in.shape = %s", z.shape, x_in.shape)

    x_gen = generator(z, x_in)
    LOGGER.debug("Generator output shape: %s", x_gen.shape)
    out = critic(x_gen)
    LOGGER.debug("Critic output shape: %s", out.shape)

    assert x_gen.shape == (bs_, _ERA_SIZE, num_inputs), f"Generator output has an incorrect shape! {x_gen.shape}"
    assert out.shape == (bs_, 1), f"Critic output has an incorrect shape! {out.shape}"

    LOGGER.debug("Running backward on a dummy loss ...")
    out.sum().backward()
    LOGGER.debug("Ran backward. All good!")

    x_real = torch.randn_like(x_gen, device=device)

    epsilon = torch.rand((bs_, 1, 1), device=device, dtype=torch.float32).expand_as(x_gen)
    interp = x_real * epsilon + x_gen * (1.0 - epsilon)
    interp = interp.requires_grad_(True)
    mixed_scores: torch.Tensor = critic(interp)

    gradient = torch.autograd.grad(
        inputs=interp,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    LOGGER.debug("interp.shape = %s, mixed_scores.shape = %s", interp.shape, mixed_scores.shape)
    LOGGER.debug("Critic gradient shape: %s, norm: %.3e", gradient.shape, gradient.flatten().norm())
