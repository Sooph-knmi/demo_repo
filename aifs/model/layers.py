import math
from typing import Optional
from typing import Tuple

import einops
import torch
import torch.nn.functional as F
from local_attention import LocalAttention
from torch import nn
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import OptTensor
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax

from aifs.distributed.helpers import change_channels_in_shape
from aifs.distributed.helpers import gather_tensor
from aifs.distributed.helpers import get_shape_shards
from aifs.distributed.helpers import shard_heads
from aifs.distributed.helpers import shard_sequence
from aifs.distributed.helpers import shard_tensor
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def gen_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    n_extra_layers: int = 0,
    activation: str = "SiLU",
    start_with_layer_norm=True,
    final_activation: bool = False,
    final_layer_norm: bool = False,
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
    activation : str, optional
        Activation function, by default "SiLU"
    start_with_layer_norm : bool, optional
        Whether to apply layer in the beginning, by default True
    final_activation : bool, optional
        Whether to apply a final activation function to last layer, by default True
    final_layer_norm : bool, optional
        Whether to apply layer norm after activation, by default False
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
        act_func = getattr(nn, activation)
    except AttributeError as ae:
        LOGGER.error("Activation function %s not supported", activation)
        raise RuntimeError from ae

    mlp1 = nn.Sequential()
    if start_with_layer_norm:
        mlp1 = nn.Sequential(nn.LayerNorm(in_features))
    mlp1.append(nn.Sequential(nn.Linear(in_features, hidden_dim), act_func()))
    for _ in range(n_extra_layers):
        mlp1.append(nn.Linear(hidden_dim, hidden_dim))
        mlp1.append(act_func())
    mlp1.append(nn.Linear(hidden_dim, out_features))

    if final_activation:
        mlp1.append(act_func())

    if final_layer_norm:
        mlp1.append(AutocastLayerNorm(out_features))

    return CheckpointWrapper(mlp1) if checkpoints else mlp1


class AutocastLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with explicit autocast back to the input type.

        This casts the output to (b)float16 (instead of float32) when we run in mixed
        precision.
        """
        return super().forward(x).type_as(x)


class GNNProcessor(nn.Module):
    """Processor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int,
        edge_dim: int,
        chunks: int = 2,
        mlp_extra_layers: int = 0,
        heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "SiLU",
        cpu_offload: bool = False,
    ) -> None:
        """Initialize GNNProcessor.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        hidden_dim : int
            Hidden dimension
        hidden_layers : int
            Number of hidden layers
        edge_dim : int
            Input features of edge MLP
        chunks : int, optional
            Number of chunks, by default 2
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "SiLU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        """
        super().__init__()

        self.hidden_layers = chunks
        chunk_size = int(hidden_layers / chunks)
        assert (
            hidden_layers % chunks == 0
        ), f"Number of processor layers ({hidden_layers}) has to be divisible by the number of processor chunks ({chunks})."

        self.proc = nn.ModuleList()
        for i in range(self.hidden_layers):
            self.proc.append(
                GNNProcessorChunk(
                    in_channels=in_channels if i == 0 else hidden_dim,
                    out_channels=out_channels if i == (self.hidden_layers - 1) else hidden_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=chunk_size,
                    mlp_extra_layers=mlp_extra_layers,
                    heads=heads,
                    mlp_hidden_ratio=mlp_hidden_ratio,
                    activation=activation,
                    edge_dim=edge_dim,
                )
            )

        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        shape_nodes: list,
        batch_size,
        size: Size,
        model_comm_group: ProcessGroup,
    ) -> Tensor:
        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        for i in range(self.hidden_layers):
            x, edge_attr = checkpoint(
                self.proc[i],
                x,
                edge_index,
                edge_attr,
                (shape_nodes, shape_nodes, shapes_edge_attr),
                batch_size,
                model_comm_group,
                use_reentrant=False,
            )

        return x


class GNNProcessorChunk(nn.Module):
    """Wraps edge embedding and X message passing blocks for checkpointing in
    Processor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int,
        mlp_extra_layers: int = 0,
        heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "SiLU",
        edge_dim: int = None,
    ) -> None:
        """Initialize GNNProcessorChunk.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimention of the message passing blocks.
        hidden_layers : int
            Number of message passing blocks.
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "SiLU"
        edge_dim: int, by default None
            Embedd edges with input dimension edge_dim
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.hidden_dim:
            self.lin_in = nn.Linear(in_channels, hidden_dim)
            self.layernorm_in = AutocastLayerNorm(in_channels)

        if self.out_channels != self.hidden_dim:
            self.lin_out = nn.Linear(hidden_dim, out_channels)
            self.layernorm_out = AutocastLayerNorm(hidden_dim)

        self.proc = nn.ModuleList(
            [
                GNNBlock(
                    hidden_dim,
                    mlp_hidden_ratio * hidden_dim,
                    hidden_dim,
                    heads=heads,
                    edge_dim=edge_dim,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                )
                for _ in range(self.hidden_layers)
            ]
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_index: Adj,
        edge_attr: Tensor,
        shapes: Tuple,
        batch_size: int,
        model_comm_group: ProcessGroup,
        size: Size = None,
    ):
        if self.in_channels != self.hidden_dim:
            x = self.layernorm_in(x)
            x = self.lin_in(x)

        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr, shapes, batch_size, model_comm_group, size=size)

        if self.out_channels != self.hidden_dim:
            x = self.layernorm_out(x)
            x = self.lin_out(x)

        return x, edge_attr


class GNNMapper(nn.Module):
    """Mapper from h -> era or era -> h."""

    def __init__(
        self,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        edge_dim: int,
        mlp_extra_layers: int = 0,
        heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "SiLU",
        cpu_offload: bool = False,
        backward_mapper: bool = False,
        out_channels_dst: Optional[int] = None,
    ) -> None:
        """Initialize GNNMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        edge_dim : int
            Input features of edge MLP
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "SiLU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        backward_mapper : bool, optional
            Map from (true) hidden to era or (false) reverse, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.backward_mapper = backward_mapper
        self.out_channels_dst = out_channels_dst

        update_src_nodes = False  # if backward_mapper else True
        self.proc = GNNBlock(
            hidden_dim,
            mlp_hidden_ratio * hidden_dim,
            hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=update_src_nodes,
            ptype="mapper",
        )

        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

        if backward_mapper:  # h -> era
            self.node_era_extractor = nn.Linear(hidden_dim, out_channels_dst)
        else:  # era -> h
            self.emb_nodes_src = nn.Linear(in_channels_src, hidden_dim)

        self.emb_nodes_dst = nn.Linear(in_channels_dst, hidden_dim)

    def forward(
        self,
        x: PairTensor,
        edge_index: Adj,
        edge_attr: Tensor,
        shape_nodes: Tuple,
        batch_size: int,
        size: Size,
        model_comm_group: ProcessGroup,
    ) -> PairTensor:
        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x_src, x_dst = x
        shapes_src, shapes_dst = shape_nodes

        if not self.backward_mapper:
            x_src = shard_tensor(x_src, 0, shapes_src, model_comm_group)
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
            x_src = self.emb_nodes_src(x_src)
            x_dst = self.emb_nodes_dst(x_dst)
            shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
            shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        else:
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
            x_dst = self.emb_nodes_dst(x_dst)
            shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst),
            edge_index,
            edge_attr,
            (shapes_src, shapes_dst, shapes_edge_attr),
            batch_size,
            model_comm_group,
            size=size,
        )

        if self.backward_mapper:
            x_dst = self.node_era_extractor(x_dst)
            x_dst = gather_tensor(x_dst, 0, change_channels_in_shape(shapes_dst, self.out_channels_dst), model_comm_group)

        return x_dst


class GNNBlock(nn.Module):
    """Message passing block with MLPs for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 16,
        bias: bool = True,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        update_src_nodes: bool = False,
        ptype: str = "processor",
        **kwargs,
    ) -> None:
        """Initialize GNNBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        edge_dim : int,
            Edge dimension
        heads : int,
            Number of heads
        bias : bool, by default True,
            Add bias or not
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        update_src_nodes: bool, by default True
            Update src if src and dst nodes are given
        ptype : str, by default "processor",
            Type of block, either processor or mapper
        """
        super().__init__(**kwargs)

        self.update_src_nodes = update_src_nodes

        self.out_channels_conv = int(out_channels / heads)
        self.heads = heads

        self.lin_key = nn.Linear(in_channels, heads * self.out_channels_conv)
        self.lin_query = nn.Linear(in_channels, heads * self.out_channels_conv)
        self.lin_value = nn.Linear(in_channels, heads * self.out_channels_conv)
        self.lin_self = nn.Linear(in_channels, heads * self.out_channels_conv, bias=bias)
        self.lin_edge = nn.Linear(edge_dim, heads * self.out_channels_conv)  # , bias=False)

        self.conv = TransformerConv(
            out_channels=self.out_channels_conv,
            heads=heads,
            bias=bias,
            edge_dim=edge_dim,
        )

        self.proj = nn.Linear(out_channels, out_channels)

        self.node_dst_mlp = gen_mlp(
            out_channels,
            hidden_dim,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation=activation,
            start_with_layer_norm=True,
        )

        self.layer_norm1 = nn.LayerNorm(in_channels)

        if ptype == "mapper":
            self.layer_norm2 = nn.LayerNorm(in_channels)

        if self.update_src_nodes:
            self.node_src_mlp = gen_mlp(
                out_channels,
                hidden_dim,
                out_channels,
                n_extra_layers=mlp_extra_layers,
                activation=activation,
                start_with_layer_norm=True,
            )

    def forward(
        self,
        x: OptPairTensor,
        edge_index: Adj,
        edge_attr: Tensor,
        shapes: Tuple,
        batch_size: int,
        model_comm_group: ProcessGroup,
        size: Size = None,
    ):
        x_skip = x

        H, C = self.heads, self.out_channels_conv

        if isinstance(x_skip, Tensor):
            x = self.layer_norm1(x)

            # combine in 1 or 2 lin layer and also sync combined
            query = self.lin_query(x)
            key = self.lin_key(x)
            value = self.lin_value(x)
            x_r = self.lin_self(x)

        else:
            x = (self.layer_norm1(x[0]), self.layer_norm2(x[1]))

            # combine in 1 or 2 lin layer and also sync combined
            query = self.lin_query(x[1])
            key = self.lin_key(x[0])
            value = self.lin_value(x[0])
            x_r = self.lin_self(x[1])

        edge_attr_emb = self.lin_edge(edge_attr)

        query, key, value, edge_attr_emb = map(
            lambda t: einops.rearrange(t, "(b n) (h c) -> b h n c", h=H, c=C, b=batch_size), (query, key, value, edge_attr_emb)
        )

        query = shard_heads(query, shapes=shapes[1], dim_split=1, dim_concatenate=2, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes[0], dim_split=1, dim_concatenate=2, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes[0], dim_split=1, dim_concatenate=2, mgroup=model_comm_group)
        edge_attr_emb = shard_heads(edge_attr_emb, shapes=shapes[2], dim_split=1, dim_concatenate=2, mgroup=model_comm_group)

        query, key, value, edge_attr_emb = map(
            lambda t: einops.rearrange(t, "b h n c -> (b n) h c"), (query, key, value, edge_attr_emb)
        )

        out = self.conv(query=query, key=key, value=value, edge_index=edge_index, edge_attr=edge_attr_emb, size=size)
        out = einops.rearrange(out, "(b n) h c -> b h n c", b=batch_size)

        out = shard_sequence(out, shapes=shapes[1], dim_split=2, dim_concatenate=1, mgroup=model_comm_group)
        out = einops.rearrange(out, "b h n c -> (b n) (h c)")

        out = self.proj(out + x_r)

        if isinstance(x_skip, Tensor):
            out = out + x_skip
            nodes_new = self.node_dst_mlp(out) + out
        else:
            out = out + x_skip[1]
            nodes_new_dst = self.node_dst_mlp(out) + out

            if self.update_src_nodes:
                nodes_new_src = self.node_src_mlp(x_skip[0]) + x_skip[0]
            else:
                nodes_new_src = x_skip[0]

            nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edge_attr


class TransformerConv(MessagePassing):
    r"""Message passing part of graph transformer operator."""

    def __init__(
        self,
        out_channels: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels
        self.dropout = dropout

    def forward(self, query: Tensor, key: Tensor, value: Tensor, edge_index: Adj, edge_attr: OptTensor = None, size: Size = None):
        dim_size = query.shape[0]
        heads = query.shape[1]

        out = self.propagate(
            heads=heads, edge_index=edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=size, dim_size=dim_size
        )

        return out

    def message(
        self,
        heads: int,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if edge_attr is not None:
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return value_j * alpha.view(-1, heads, 1)  # + edge_attr to value_j?


class TransformerProcessor(nn.Module):
    """Message Passing Processor Graph Neural Network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int,
        window_size: int,
        heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        chunks: int = 2,
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        hidden_dim : int
            Hidden dimension
        hidden_layers : int
            Number of hidden layers
        window_size: int,
            1/2 size of shifted window for attention computation
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        """
        super().__init__()

        self.hidden_layers = chunks
        chunk_size = int(hidden_layers / chunks)
        assert (
            hidden_layers % chunks == 0
        ), f"Number of processor layers ({hidden_layers}) has to be divisible by the number of processor chunks ({chunks})."

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.proc = nn.ModuleList(
            [
                TransformerProcessorChunk(
                    in_channels=in_channels if i == 0 else hidden_dim,
                    out_channels=out_channels if i == (self.hidden_layers - 1) else hidden_dim,
                    hidden_dim=hidden_dim,
                    mlp_hidden_ratio=mlp_hidden_ratio,
                    heads=heads,
                    hidden_layers=chunk_size,
                    window_size=window_size,
                    activation=activation,
                )
                for i, _ in enumerate(range(self.hidden_layers))
            ]
        )

    def forward(self, x: Tensor, shape_nodes: list, batch_size: int, model_comm_group: ProcessGroup) -> Tensor:
        shapes = (shape_nodes, shape_nodes)

        assert model_comm_group.size() == 1 or batch_size == 1, "Either one GPU per model instance, or batch_size has to be 1"

        for i in range(self.hidden_layers):
            x = checkpoint(self.proc[i], x, shapes, batch_size, model_comm_group=model_comm_group, use_reentrant=False)

        return x


class TransformerProcessorChunk(nn.Module):
    """Message Passing Processor Graph Neural Network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int,
        window_size: int,
        heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension
        hidden_layers : int
            Number of hidden layers
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.hidden_layers = hidden_layers

        self.proc = nn.ModuleList(
            [
                TransformerLayer(
                    channels=hidden_dim,
                    hidden_dim=(mlp_hidden_ratio * hidden_dim),
                    num_heads=heads,
                    activation=activation,
                    window_size=window_size,
                )
                for _ in range(self.hidden_layers)
            ]
        )

        if self.in_channels != self.hidden_dim:
            self.lin_in = nn.Linear(in_channels, hidden_dim)
            self.layernorm_in = AutocastLayerNorm(in_channels)

        if self.out_channels != self.hidden_dim:
            self.lin_out = nn.Linear(hidden_dim, out_channels)
            self.layernorm_out = AutocastLayerNorm(hidden_dim)

    def forward(self, x: Tensor, shapes: list, batch_size: int, model_comm_group: ProcessGroup) -> Tensor:
        if self.in_channels != self.hidden_dim:
            x = self.layernorm_in(x)
            x = self.lin_in(x)

        for i in range(self.hidden_layers):
            x = self.proc[i](x, shapes, batch_size, model_comm_group=model_comm_group)

        if self.out_channels != self.hidden_dim:
            x = self.layernorm_out(x)
            x = self.lin_out(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, channels, hidden_dim, num_heads, activation, window_size: int):
        """
        Attention Block ...
        """
        super().__init__()

        try:
            act_func = getattr(nn, activation)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation)
            raise RuntimeError from ae

        self.lnorm1 = nn.LayerNorm(channels)
        self.mhsa = MultiHeadSelfAttention(
            num_heads=num_heads, embed_dimension=channels, window_size=window_size, bias=False, is_causal=False, dropout=0.0
        )
        self.proj = nn.Linear(channels, channels, bias=True)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            act_func(),
            nn.Linear(hidden_dim, channels),
        )
        self.lnorm2 = nn.LayerNorm(channels)

    def forward(self, x: Tensor, shapes: list, batch_size: int, model_comm_group: ProcessGroup) -> Tensor:
        x1 = x
        x = self.lnorm1(x)
        x = self.proj(self.mhsa(x, shapes, batch_size, model_comm_group=model_comm_group)) + x1
        x = self.mlp(self.lnorm2(x)) + x

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dimension: int,
        bias: bool = False,
        is_causal: bool = False,
        window_size: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert embed_dimension % num_heads == 0

        self.lin_qkv = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # self.attn = F.scaled_dot_product_attention

        self.dropout = dropout
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        self.head_dim = embed_dimension // num_heads  # q k v

        self.is_causal = is_causal

        self.atten = LocalAttention(
            dim=self.head_dim,
            window_size=window_size,
            causal=is_causal,
            autopad=True,
            scale=None,
            exact_windowsize=False,
            use_xpos=False,
            xpos_scale_base=None,
            look_forward=1,
            dropout=self.dropout,
        )

    def forward(self, x: Tensor, shapes: list, batch_size: int, model_comm_group: ProcessGroup) -> Tensor:
        query, key, value = self.lin_qkv(x).chunk(3, -1)

        query, key, value = map(
            lambda t: einops.rearrange(t, "(b n) (h c) -> b h n c", b=batch_size, h=self.num_heads), (query, key, value)
        )

        # combine to 1 or two comms?
        query = shard_heads(query, shapes=shapes[1], dim_split=1, dim_concatenate=2, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes[0], dim_split=1, dim_concatenate=2, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes[0], dim_split=1, dim_concatenate=2, mgroup=model_comm_group)

        # out = self.atten(query, key, value, attn_mask=None, dropout_p=self.dropout, is_causal=self.is_causal)
        out = self.atten(query, key, value, mask=None, attn_bias=None)

        out = shard_sequence(out, shapes=shapes[1], dim_split=2, dim_concatenate=1, mgroup=model_comm_group)
        out = einops.rearrange(out, "b h n c -> (b n) (h c)")

        return out


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

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
