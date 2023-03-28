from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
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


# for MSG, where should this go?
def gen_mlp(in_features, hidden_dim, out_features, layer_norm=True):
    mlp1 = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_features),
    )
    if layer_norm:
        mlp1.append(nn.LayerNorm(out_features))

    return mlp1


class MessagePassingNodeEmbedder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(MessagePassingNodeEmbedder, self).__init__()
        self.node_emb = gen_mlp(in_features=in_channels, hidden_dim=latent_dim, out_features=latent_dim, layer_norm=True)

    def forward(self, x: torch.Tensor):
        return self.node_emb(x)


class MessagePassingNodeExtractor(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(MessagePassingNodeExtractor, self).__init__()
        self.node_ext = gen_mlp(in_features=latent_dim, hidden_dim=latent_dim, out_features=out_channels, layer_norm=False)

    def forward(self, x: torch.Tensor):
        return self.node_ext(x)


class MessagePassingMapper(nn.Module):  # should we remove self loops to be able to use same graph as Encoder?
    def __init__(self, in_channels, out_channels, hidden_dim, edge_dim, hidden_layers):
        # TODO: if in_channels and out_channels are not used, can we get rid of them?
        super().__init__()

        self.hidden_layers = hidden_layers
        self.edge_enc = gen_mlp(in_features=edge_dim, hidden_dim=hidden_dim, out_features=hidden_dim, layer_norm=True)
        self.proc = nn.ModuleList([MessagePassingBlock(hidden_dim, hidden_dim) for _ in range(self.hidden_layers)])

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:  # these are probably wrong ..
        x_src, x_dst = x  # this should be Union[Tensor, PairTensor] or something from typing ... Union
        edge_attr = self.edge_enc(edge_attr)
        for i in range(self.hidden_layers):
            # here only x_dst is updated for the next layer, x_src always stays the same, is this what we want? I assume yes
            x_dst, edge_attr = self.proc[i]((x_src, x_dst), edge_index, edge_attr, size=(x_src.shape[0], x_dst.shape[0]))

        return x_dst


class MessagePassingEncoder(nn.Module):  # should we remove self loops to be able to use same graph as Encoder?
    def __init__(self, in_channels, out_channels, hidden_dim, edge_dim, hidden_layers):
        super(MessagePassingEncoder, self).__init__()

        self.hidden_layers = hidden_layers

        self.edge_enc = gen_mlp(in_features=edge_dim, hidden_dim=hidden_dim, out_features=hidden_dim, layer_norm=True)

        self.proc = nn.ModuleList([MessagePassingBlock(hidden_dim, hidden_dim) for _ in range(self.hidden_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        edge_attr = self.edge_enc(edge_attr)
        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr, size=None)

        return x


class MessagePassingBlock(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MessagePassingBlock, self).__init__(**kwargs)
        # these should also be generated via gen_mlp, todo
        self.node_mlp = gen_mlp(2 * in_channels, out_channels, out_channels)

        self.edge_mlp = gen_mlp(3 * in_channels, out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, size=None):
        out, edges_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if isinstance(x, torch.Tensor):
            nodes_new = torch.cat([x, out], dim=1)
            nodes_new = x + self.node_mlp(nodes_new)
        else:
            nodes_new = torch.cat([x[1], out], dim=1)
            nodes_new = self.node_mlp(nodes_new) + x[1]

        return nodes_new, edges_new

    def message(self, x_i, x_j, edge_attr):
        edges_new = torch.cat([x_i, x_j, edge_attr], dim=1)
        edges_new = self.edge_mlp(edges_new) + edge_attr

        return edges_new

    def aggregate(self, edges_new, edge_index, dim_size=None):
        # out = scatter(edges_new, edge_index[0, :], dim=0, reduce = 'sum')
        out = scatter(edges_new, edge_index[1, :], dim=0, reduce="sum")

        return out, edges_new


class EdgePoolEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 16,
        num_layers: int = 2,
        hidden_channels: int = 16,
        num_heads: int = 2,
        dropout: float = 0.0,
        activation: Optional[str] = "gelu",
        jk_mode: Optional[str] = "last",
    ) -> None:
        super().__init__()

        # we need to pool otherwise memory consumption will be large
        # OK to use dropout? the EdgePooling paper reports good results with dropout = 0.2
        self.pool = tgnn.EdgePooling(in_channels=in_channels, edge_score_method=tgnn.EdgePooling.compute_edge_score_softmax)

        self.encoder = GATEncoder(
            num_layers=num_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            jk_mode=jk_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:

        del edge_attr  # not used

        # x is on ERA grid
        batch_nodes = torch.cat(
            [torch.ones(x.shape[0] // batch_size, dtype=torch.int64, device=x.device) * b for b in range(batch_size)]
        )

        # pool first
        x_pool, edge_pool, _, unpool_info = self.pool(x=x, edge_index=edge_index, batch=batch_nodes)

        # transform
        x_trans = self.encoder(x_pool, edge_pool)

        # unpool
        x_res, _, _ = self.pool.unpool(x_trans, unpool_info)

        return x_res


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)

        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, x, edge_index, edge_attr, size=None):
        out, edges_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        nodes_new = torch.cat([x, out], dim=1)
        nodes_new = x + self.node_mlp(nodes_new)

        return nodes_new, edges_new

    def message(self, x_i, x_j, edge_attr):
        edges_new = torch.cat([x_i, x_j, edge_attr], dim=1)
        edges_new = self.edge_mlp(edges_new) + edge_attr

        return edges_new

    def aggregate(self, edges_new, edge_index, dim_size=None):
        del dim_size  # not used
        out = scatter(edges_new, edge_index[0, :], dim=0, reduce="sum")

        return out, edges_new
