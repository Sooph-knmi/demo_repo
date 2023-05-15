from typing import Optional, Tuple, Union  # , List

import einops
import torch
from torch import nn
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.utils.checkpoint import checkpoint
import torch_geometric.nn as tgnn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size, PairTensor
from torch_geometric.utils import scatter

from aifs.utils.logger import get_logger

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
        x: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        dynamic_context: Optional[Tensor] = None,
        batch_size: int = 1,
    ) -> Tensor:
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

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Optional[Tensor] = None) -> Tensor:
        return self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)


def gen_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    n_extra_layers: int = 0,
    activation_func: str = "SiLU",
    final_activation: bool = True,
    layer_norm: bool = True,
    checkpoints: bool = False,
) -> nn.Module:
    try:
        act_func = getattr(nn, activation_func)
    except AttributeError as ae:
        LOGGER.error("Activation function %s not supported", activation_func)
        raise RuntimeError from ae

    mlp1 = nn.Sequential(nn.Linear(in_features, hidden_dim), act_func())
    for _ in range(n_extra_layers):
        mlp1.append(nn.Linear(hidden_dim, hidden_dim))
        mlp1.append(act_func())
    mlp1.append(nn.Linear(hidden_dim, out_features))

    if final_activation:
        mlp1.append(act_func())

    if layer_norm:
        mlp1.append(nn.LayerNorm(out_features))

    return CheckpointWrapper(mlp1) if checkpoints else mlp1


class MessagePassingMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        final_activation: bool = True,
        layer_norm: bool = True,
        checkpoints: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = gen_mlp(
            in_features=in_channels,
            hidden_dim=latent_dim,
            out_features=out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
            final_activation=final_activation,
            layer_norm=layer_norm,
            checkpoints=checkpoints,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class MessagePassingProcessor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        chunks: int = 4,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        cpu_offload: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_layers = chunks
        chunk_size = int(hidden_layers / chunks)

        assert hidden_layers % chunks == 0

        self.proc = nn.ModuleList(
            [
                MessagePassingProcessorChunk(
                    hidden_dim, hidden_layers=chunk_size, mlp_extra_layers=mlp_extra_layers, activation=activation
                )
                for _ in range(self.hidden_layers)
            ]
        )

        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        for i in range(self.hidden_layers):
            x, edge_attr = checkpoint(self.proc[i], x, edge_index, edge_attr, size=None, use_reentrant=False)

        return x


class MessagePassingMapper(MessagePassingProcessor):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def forward(self, x: PairTensor, edge_index: Adj, edge_attr: Tensor) -> PairTensor:
        x_src, x_dst = x
        for i in range(self.hidden_layers):
            (x_src, x_dst), edge_attr = checkpoint(
                self.proc[i], (x_src, x_dst), edge_index, edge_attr, size=(x_src.shape[0], x_dst.shape[0]), use_reentrant=False
            )

        return x_src, x_dst


class MessagePassingProcessorChunk(nn.Module):
    """Wraps X message passing blocks for checkpointing in Processor / Mapper"""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
    ) -> None:
        super().__init__()

        self.hidden_layers = hidden_layers

        self.proc = nn.ModuleList(
            [
                MessagePassingBlock(
                    hidden_dim,
                    hidden_dim,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                )
                for _ in range(self.hidden_layers)
            ]
        )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor, size: Size = None) -> PairTensor:
        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr, size=size)

        return x, edge_attr


class MessagePassingBlock(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.node_mlp = gen_mlp(
            2 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )
        self.edge_mlp = gen_mlp(
            3 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor, size: Size = None):
        out, edges_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if isinstance(x, Tensor):
            nodes_new = self.node_mlp(torch.cat([x, out], dim=1)) + x
        else:
            nodes_new_dst = self.node_mlp(torch.cat([x[1], out], dim=1)) + x[1]
            nodes_new_src = self.node_mlp(torch.cat([x[0], x[0]], dim=1)) + x[0]  # todo: only update this in the forward mapper...
            nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edges_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        edges_new = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=1)) + edge_attr

        return edges_new

    def aggregate(self, edges_new: Tensor, edge_index: Adj, dim_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        out = scatter(edges_new, edge_index[1], dim=0, reduce="sum")

        return out, edges_new


class CheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


if __name__ == "__main__":
    bs, nlatlon, nfeat = 1, 1024, 64
    hdim, ofeat = 128, 36
    x_in = torch.randn((bs, nlatlon, nfeat), dtype=torch.float32, requires_grad=True)
    mlp_1 = gen_mlp(nfeat, hdim, hdim, layer_norm=True)
    mlp_2 = gen_mlp(hdim, hdim, hdim, layer_norm=True)
    mlp_3 = gen_mlp(hdim, hdim, ofeat, layer_norm=True)
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
