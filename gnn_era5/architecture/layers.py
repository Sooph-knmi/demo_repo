from typing import Optional

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


class MessagePassingEncoder(nn.Module):  # should we remove self loops to be able to use same graph as Encoder?
    def __init__(self, in_channels, out_channels, hidden_dim: int = 128, edge_dim: int = 3, proc_layers: int = 3):
        super().__init__()

        self.proc_layers = proc_layers

        self.node_enc = self._gen_mlp(in_features=in_channels, hidden_dim=hidden_dim, out_features=hidden_dim, layer_norm=True)
        self.edge_enc = self._gen_mlp(in_features=edge_dim, hidden_dim=hidden_dim, out_features=hidden_dim, layer_norm=True)

        self.proc = nn.ModuleList([ProcessorLayer(hidden_dim, hidden_dim) for _ in range(self.proc_layers)])

        self.decoder = self._gen_mlp(in_features=hidden_dim, hidden_dim=hidden_dim, out_features=out_channels, layer_norm=False)

    def _gen_mlp(self, in_features, hidden_dim, out_features, layer_norm=True) -> nn.Module:
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.node_enc(x)  # do we need this or should this happen in TransformerMapper instead?
        edge_attr = self.edge_enc(edge_attr)
        for i in range(self.proc_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr)
        out1 = self.decoder(x)

        return out1


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


# if __name__ == "__main__":
#     import os
#     from gnn_era5.utils.constants import _ERA_O160_LATLON
#     from gnn_era5.utils.config import YAMLConfig

#     era_size = _ERA_O160_LATLON
#     bs, nf, npos = 4, 16, 2

#     config = YAMLConfig("/home/syma/dask/codes/gnn-era5/gnn_era5/config/atos.yaml")
#     graphs = torch.load(os.path.join(config["graph:data-basedir"], config["graph:data-file"]))
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     e2e_edge_index = graphs[("era", "to", "era")].edge_index.to(device)
#     e2e_edge_attr = graphs[("era", "to", "era")].edge_attr.to(device)
#     encoder = EdgePoolEncoder(in_channels=nf + npos, out_channels=32).to(device)

#     x_in = torch.randn(bs, era_size, nf, dtype=torch.float32, device=device)
#     x_pos = torch.randn(bs, era_size, npos, dtype=torch.float32, device=device)

#     x_in = torch.cat([x_in, x_pos], dim=-1)
#     LOGGER.debug("x_in.shape = %s", x_in.shape)

#     x_in = einops.rearrange(x_in, "b n f -> (b n) f")
#     x_out = encoder(
#         x_in,
#         # expand edge index correct number of times while adding the proper number to the edge index
#         edge_index=torch.cat(
#             [e2e_edge_index + i * e2e_edge_index.max(1, keepdim=True).values + i for i in range(bs)],
#             dim=1,
#         ),
#         # copy edge attributes bs times
#         edge_attr=einops.repeat(e2e_edge_attr, "e f -> (repeat e) f", repeat=bs),
#         batch_size=bs,
#     )
#     x_out = einops.rearrange(x_out, "(b n) f -> b n f", b=bs)
#     LOGGER.debug("x_out.shape = %s", x_out.shape)
