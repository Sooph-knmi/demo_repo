import math
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import OptTensor
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size
from torch_geometric.utils import scatter
from torch_geometric.utils import segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class AutocastLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm with output autocast to x.type.

        During mixed-precision training, this will cast the LayerNorm output back to
        (b)float16 (from float32).
        """
        t = x.dtype
        return super().forward(x).to(dtype=t)


def gen_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    n_extra_layers: int = 0,
    activation_func: str = "SiLU",
    dropout: float = 0.0,
    final_activation: bool = False,
    start_with_layernorm=False,
    layer_norm: bool = True,
    checkpoints: bool = False,
) -> nn.Module:
    """Generate a multi-layer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features
    hidden_dim : int
        Hidden dimensions
    out_features : int
        Number of output features
    n_extra_layers : int, optional
        Number of extra layers in MLP, by default 0
    activation_func : str, optional
        Activation function, by default "SiLU"
    dropout: float, optional
        Dropout rate
    final_activation : bool, optional
        Whether to apply a final activation function to last layer, by default True
    layer_norm : bool, optional
        Whether to apply layer norm after activation, by default True
    checkpoints : bool, optional
        Whether to provide checkpoints, by default False

    Returns
    -------
    nn.Module
        Returns a MLP module

    Raises
    ------
    RuntimeError
        If activation function is not supported
    """
    try:
        act_func = getattr(nn, activation_func)
    except AttributeError as ae:
        LOGGER.error("Activation function %s not supported", activation_func)
        raise RuntimeError from ae

    mlp1 = nn.Sequential()
    if start_with_layernorm:
        mlp1 = nn.Sequential(AutocastLayerNorm(in_features))

    mlp1.append(nn.Sequential(nn.Linear(in_features, hidden_dim), act_func()))
    for _ in range(n_extra_layers):
        mlp1.append(nn.Linear(hidden_dim, hidden_dim))
        mlp1.append(nn.Dropout(p=dropout))
        mlp1.append(act_func())
    mlp1.append(nn.Linear(hidden_dim, out_features))

    if final_activation:
        mlp1.append(act_func())

    if layer_norm:
        mlp1.append(AutocastLayerNorm(out_features))

    return CheckpointWrapper(mlp1) if checkpoints else mlp1


class GNNProcessor(nn.Module):
    """Message Passing Processor Graph Neural Network."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        chunks: int = 2,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        mlp_dropout: float = 0.0,
        ptype: str = "proc",  # proc, fmapper, bmapper :this is ugly ... fix this!
    ) -> None:
        """Initialize MessagePassingProcessor.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension
        hidden_layers : int
            Number of hidden layers
        edge_dim : int
            Input features of MLP
        chunks : int, optional
            Number of chunks, by default 2
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        activation : str, optional
            Activation funciton, by default "SiLU"
        dropout: float, optional
            Dropout rate for the processor MLPs.
        """
        super().__init__()

        self.hidden_layers = chunks
        chunk_size = int(hidden_layers / chunks)

        assert hidden_layers % chunks == 0
        assert hidden_dim % num_heads == 0

        # needed in mapper
        self.hidden_dim = hidden_dim
        self.mlp_extra_layers = mlp_extra_layers
        self.activation = activation
        self.mlp_dropout = mlp_dropout

        self.proc = nn.ModuleList(
            [
                GNNProcessorChunk(
                    hidden_dim,
                    edge_dim,
                    hidden_layers=chunk_size,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    mlp_dropout=mlp_dropout,
                    ptype=ptype,
                )
                for _ in range(self.hidden_layers)
            ]
        )

        self.emb_nodes_out = gen_mlp(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
            out_features=output_dim,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
            final_activation=True,
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # x.shape[-1] == hidden_dim == (num_hidden_features + num_noise_features)
        for i in range(self.hidden_layers):
            x, edge_attr = checkpoint(self.proc[i], x, edge_index, edge_attr, use_reentrant=False)

        return self.emb_nodes_out(x)


class GNNMapper(nn.Module):
    """Mapper from h -> era or era -> h."""

    def __init__(
        self,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        backward_mapper: bool = False,
        out_channels_dst: Optional[int] = None,
    ) -> None:
        """Initialize the mapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        backward_mapper : bool, optional
            Map from (true) hidden to era or (false) reverse, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.backward_mapper = backward_mapper
        self.out_channels_dst = out_channels_dst

        assert hidden_dim % num_heads == 0

        self.emb_edges = gen_mlp(
            in_features=edge_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )

        self.proc = GNNTransformerBlock(
            hidden_dim,
            4 * hidden_dim,
            hidden_dim,
            edge_dim=hidden_dim,
            heads=num_heads,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            ptype="bmapper" if backward_mapper else "fmapper",
        )

        if backward_mapper:  # h -> era
            self.node_era_extractor = gen_mlp(
                in_features=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                out_features=out_channels_dst,
                n_extra_layers=mlp_extra_layers,
                activation_func=activation,
                layer_norm=False,
                final_activation=False,
                start_with_layernorm=True,
            )
        else:  # era -> h
            self.emb_nodes_src = gen_mlp(
                in_features=in_channels_src,
                hidden_dim=self.hidden_dim,
                out_features=self.hidden_dim,
                n_extra_layers=mlp_extra_layers,
                activation_func=activation,
            )

            self.emb_nodes_dst = gen_mlp(
                in_features=in_channels_dst,
                hidden_dim=self.hidden_dim,
                out_features=self.hidden_dim,
                n_extra_layers=mlp_extra_layers,
                activation_func=activation,
            )

    def forward(self, x: PairTensor, edge_index: Adj, edge_attr: Tensor) -> PairTensor:
        edge_attr = self.emb_edges(edge_attr)

        if self.backward_mapper:
            x_src, x_dst = x
        else:
            x_src = self.emb_nodes_src(x[0])
            x_dst = self.emb_nodes_dst(x[1])

        (x_src, x_dst), edge_attr = self.proc((x_src, x_dst), edge_index, edge_attr, size=(x_src.shape[0], x_dst.shape[0]))

        if self.backward_mapper:
            x_dst = self.node_era_extractor(x_dst)

        return x_src, x_dst


class GNNProcessorChunk(nn.Module):
    """Wraps X message passing blocks for checkpointing in Processor / Mapper."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        hidden_layers: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        mlp_dropout: float = 0.0,
        ptype: str = "proc",
    ) -> None:
        """Initialize MessagePassingProcessorChunk.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension of the message passing blocks.
        hidden_layers : int
            Number of message passing blocks.
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        mlp_dropout: float, optional
            Dropout rate for the MLPs.
        """
        super().__init__()

        self.hidden_layers = hidden_layers

        self.proc = nn.ModuleList(
            [
                GNNTransformerBlock(
                    hidden_dim,
                    4 * hidden_dim,
                    hidden_dim,
                    edge_dim,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    mlp_dropout=mlp_dropout,
                    ptype=ptype,
                )
                for _ in range(self.hidden_layers)
            ]
        )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor, size: Size = None):
        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr, size=size)

        return x, edge_attr


class GNNTransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,  # hidden dim of mlp
        out_channels: int,
        edge_dim: int,
        heads: int = 8,
        concat: bool = True,
        bias: bool = True,
        activation: str = "SiLU",
        ptype: str = "proc",
        mlp_dropout: float = 0.0,
        mlp_extra_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.ptype = ptype

        self.conv = GNNTransformer(
            in_channels=in_channels,
            out_channels=int(out_channels / heads),
            heads=heads,
            concat=concat,
            bias=bias,
            edge_dim=edge_dim,
            dropout=mlp_dropout,
            **kwargs,
        )

        self.proj = nn.Linear(out_channels, out_channels)

        self.node_dst_mlp = gen_mlp(
            out_channels,
            hidden_dim,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
            start_with_layernorm=True,
            final_activation=False,
            layer_norm=False,
            dropout=mlp_dropout,
        )

        self.layer_norm1 = AutocastLayerNorm(in_channels)

        if ptype in ("fmapper", "bmapper"):
            self.layer_norm2 = AutocastLayerNorm(in_channels)

        if ptype == "fmapper":
            self.node_src_mlp = gen_mlp(
                out_channels,
                hidden_dim,
                out_channels,
                n_extra_layers=mlp_extra_layers,
                activation_func=activation,
                start_with_layernorm=True,
                final_activation=False,
                layer_norm=False,
            )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor, size: Size = None):
        del size  # not used (?)
        x_skip = x

        if isinstance(x_skip, Tensor):
            x = self.layer_norm1(x)
        else:
            x = (self.layer_norm1(x[0]), self.layer_norm2(x[1]))

        out = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)  # , size=size)
        out = self.proj(out)

        if isinstance(x_skip, Tensor):
            out = out + x_skip
            nodes_new = self.node_dst_mlp(out) + out
        else:
            out = out + x_skip[1]
            nodes_new_dst = self.node_dst_mlp(out) + out

            if self.ptype == "fmapper":
                nodes_new_src = self.node_src_mlp(x[0]) + x_skip[0]
            else:
                nodes_new_src = x[0]

            nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edge_attr


class GNNTransformer(MessagePassing):
    """The graph transformer."""

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.register_parameter("lin_beta", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ):
        """Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        del return_attention_weights  # not relevant here

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels}, heads={self.heads})"


def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    """Sparse softmax."""
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce="max")
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = (src - src_max).exp()
        out_sum = segment(out, ptr, reduce="sum") + 1.0  # 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce="sum") + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / out_sum


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


if __name__ == "__main__":
    import einops
    import numpy as np

    bs, nlatlon, nf = 2, 1024, 64
    hdim, ofeat = 128, 36
    nnoise = 8
    x_in = torch.randn((bs, nlatlon, nf), dtype=torch.float32, requires_grad=True)
    z_in = torch.randn((bs, nlatlon, nnoise), dtype=torch.float32, requires_grad=False)
    edim = 3

    noisy_processor = GNNProcessor(
        hidden_dim=nf + nnoise,
        hidden_layers=2,
        edge_dim=edim,
        output_dim=nf,
    )

    nedges = 4500
    eattr = torch.randn(nedges, edim)
    eidx = torch.randint(0, nlatlon, size=(2, nedges))

    eattr_batched = torch.cat([einops.repeat(eattr, "e f -> (repeat e) f", repeat=bs)], dim=-1)
    edge_inc = torch.from_numpy(np.asarray([[nlatlon], [nlatlon]], dtype=np.int64))
    eidx_batched = torch.cat([eidx + i * edge_inc for i in range(bs)], dim=1)

    x_in = einops.rearrange(x_in, "bs n f -> (bs n) f")
    z_in = einops.rearrange(z_in, "bs n f -> (bs n) f")

    x_out = noisy_processor(torch.cat([x_in, z_in], dim=-1), eidx_batched, eattr_batched)
    LOGGER.debug("x_out.shape = %s", x_out.shape)
