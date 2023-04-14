from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__)


class TransformerMapper(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_size: int,
        trainable_context_channels: int = 1,
        dynamic_context_channels: int = 0,
        num_heads: int = 1,
        dropout: float = 0.0,
        edge_dim: int = 3,
    ) -> None:
        super().__init__()
        self.dynamic_context_channels = dynamic_context_channels
        context_channels = trainable_context_channels + self.dynamic_context_channels

        if context_channels > 0:
            self.conv = tgnn.GATConv(
                in_channels=(in_channels, context_channels),
                out_channels=out_channels,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        else:
            self.conv = tgnn.GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=edge_dim,
            )

        if trainable_context_channels > 0:
            self.trainable_context = nn.Parameter(torch.zeros((context_size, trainable_context_channels), dtype=torch.float32))
        else:
            self.trainable_context = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        dynamic_context: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        context = self.trainable_context

        if dynamic_context is not None:
            assert (
                dynamic_context.shape[1] == self.dynamic_context_channels
            ), f"Expected {dynamic_context.shape[1]} dynamic context channels and got {self.dynamic_context_channels}!"
            if context is None:
                context = dynamic_context
            else:
                context = torch.cat([context, dynamic_context], dim=1)

        if context is not None:
            if batch_size > 1:
                context = einops.repeat(context, "n f -> (repeat n) f", repeat=batch_size)
            assert edge_index[0].max() < x.size(0) and edge_index[1].max() < context.size(0), "Your edge index tensor is invalid."
            out = self.conv(x=(x, context), edge_index=edge_index, edge_attr=edge_attr, size=(x.shape[0], context.shape[0]))
        else:
            out = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return out


class GATEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        activation: Optional[str] = "gelu",
        jk_mode: Optional[str] = "last",
        edge_dim: int = 3,
    ) -> None:
        super().__init__()

        act_fn = None
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky-relu":
            act_fn = nn.LeakyReLU(negative_slope=0.2)

        self.encoder = tgnn.GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act_fn,
            norm=tgnn.LayerNorm(in_channels=hidden_channels, affine=True),
            v2=True,
            jk=jk_mode,
            heads=num_heads,
            add_self_loops=False,
            bias=False,
            edge_dim=edge_dim,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)


def gen_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    activation_func: str = "SiLU",
    layer_norm: bool = True,
    checkpoints: bool = True,
) -> nn.Module:
    if activation_func == "Gaussian":
        act_func = GaussianActivation
    else:
        try:
            act_func = getattr(nn, activation_func)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation_func)
            raise RuntimeError from ae

    mlp1 = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        act_func(),
        nn.Linear(hidden_dim, hidden_dim),
        act_func(),
        nn.Linear(hidden_dim, out_features),
    )
    if layer_norm:
        mlp1.append(nn.LayerNorm(out_features))

    return CheckpointWrapper(mlp1) if checkpoints else mlp1


class GaussianActivation(nn.Module):
    def __init__(self, alpha: int = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x**2.0 / self.alpha**2.0)


class MessagePassingNodeEmbedder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, activation: str = "SiLU", checkpoints: bool = True) -> None:
        super().__init__()
        self.node_emb = gen_mlp(
            in_features=in_channels,
            hidden_dim=latent_dim,
            out_features=latent_dim,
            activation_func=activation,
            layer_norm=True,
            checkpoints=checkpoints,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_emb(x)


class MessagePassingNodeExtractor(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, activation: str = "SiLU", checkpoints: bool = False) -> None:
        super().__init__()
        self.node_ext = gen_mlp(
            in_features=latent_dim,
            hidden_dim=latent_dim,
            out_features=out_channels,
            activation_func=activation,
            layer_norm=False,
            checkpoints=checkpoints,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_ext(x)


class MessagePassingMapper(nn.Module):
    def __init__(
        self, hidden_dim: int, edge_dim: int, hidden_layers: int, activation: str = "SiLU", checkpoints: bool = True
    ) -> None:
        super().__init__()

        self.hidden_layers = hidden_layers
        self.edge_enc = gen_mlp(
            in_features=edge_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            activation_func=activation,
            layer_norm=True,
            checkpoints=checkpoints,
        )
        self.proc = nn.ModuleList(
            [MessagePassingBlock(hidden_dim, hidden_dim, activation=activation) for _ in range(self.hidden_layers)]
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x_src, x_dst = x
        edge_attr = self.edge_enc(edge_attr)
        for i in range(self.hidden_layers):
            # here only x_dst is updated for the next layer, x_src always stays the same, is this what we want? I assume yes
            x_dst, edge_attr = self.proc[i]((x_src, x_dst), edge_index, edge_attr, size=(x_src.shape[0], x_dst.shape[0]))

        return x_dst


class MessagePassingProcessor(nn.Module):
    def __init__(
        self, hidden_dim: int, edge_dim: int, hidden_layers: int, activation: str = "SiLU", checkpoints: bool = True
    ) -> None:
        super().__init__()

        self.hidden_layers = hidden_layers
        self.edge_enc = gen_mlp(
            in_features=edge_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            activation_func=activation,
            layer_norm=True,
            checkpoints=checkpoints,
        )
        self.proc = nn.ModuleList(
            [MessagePassingBlock(hidden_dim, hidden_dim, activation=activation) for _ in range(self.hidden_layers)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_attr = self.edge_enc(edge_attr)
        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr, size=None)

        return x


class MessagePassingBlock(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "SiLU", checkpoints: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        self.node_mlp = gen_mlp(2 * in_channels, out_channels, out_channels, activation_func=activation, checkpoints=checkpoints)
        self.edge_mlp = gen_mlp(3 * in_channels, out_channels, out_channels, activation_func=activation, checkpoints=checkpoints)

    def forward(self, x, edge_index, edge_attr, size=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, edges_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if isinstance(x, torch.Tensor):
            nodes_new = torch.cat([x, out], dim=1)
            nodes_new = self.node_mlp(nodes_new) + x
        else:
            nodes_new = torch.cat([x[1], out], dim=1)
            nodes_new = self.node_mlp(nodes_new) + x[1]

        return nodes_new, edges_new

    def message(self, x_i, x_j, edge_attr) -> torch.Tensor:
        edges_new = torch.cat([x_i, x_j, edge_attr], dim=1)
        edges_new = self.edge_mlp(edges_new) + edge_attr

        return edges_new

    def aggregate(self, edges_new, edge_index, dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # out = scatter(edges_new, edge_index[0, :], dim=0, reduce = 'sum')
        out = scatter(edges_new, edge_index[1, :], dim=0, reduce="sum")

        return out, edges_new


class CheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        return checkpoint(self.module, *args, use_reentrant=False)


if __name__ == "__main__":
    bs, nlatlon, nfeat = 8, 1024, 64
    hdim, ofeat = 128, 36
    x_in = torch.randn((bs, nlatlon, nfeat), dtype=torch.float32, requires_grad=True)
    mlp_1 = gen_mlp(nfeat, hdim, hdim, layer_norm=True, checkpoints=True)
    mlp_2 = gen_mlp(hdim, hdim, hdim, layer_norm=True, checkpoints=False)
    mlp_3 = gen_mlp(hdim, hdim, ofeat, layer_norm=True, checkpoints=True)
    y = mlp_1(x_in)
    LOGGER.debug("mlp_1(x).shape = %s", y.shape)
    y = mlp_2(y)
    LOGGER.debug("mlp_2(mlp_1(x)).shape = %s", y.shape)
    y = mlp_3(y)
    LOGGER.debug("mlp_3(mlp_2(mlp_1(x))).shape = %s", y.shape)
    loss = y.sum()
    LOGGER.debug("running backward on the dummy loss ...")
    loss.backward()
    LOGGER.debug("done.")
