from typing import Optional
from typing import Tuple
from typing import Union

import einops
import torch
import torch_geometric.nn as tgnn
from torch import nn
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size
from torch_geometric.utils import scatter

from aifs.utils.distributed import shard_tensor1, gather_tensor1, reduce_tensor1, get_shape_shards1, change_channels_in_shape1
from aifs.diagnostics.logger import get_logger

LOGGER = get_logger(__name__)


class TransformerMapper(MessagePassing):
    """Transformer mapper layer."""

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
        """Initialize the transformer mapper layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        context_size : int
            Size of the context vector.
        trainable_context_channels : int, optional
            Number of trainable context channels, by default 1
        dynamic_context_channels : int, optional
            Number of dynamic context channels, by default 0
        num_heads : int, optional
            Number of attention heads, by default 1
        dropout : float, optional
            Dropout probability, by default 0.0
        edge_dim : int, optional
            Edge feature dimension, by default 3
        """
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
            out = self.conv(
                x=(x, context),
                edge_index=edge_index,
                edge_attr=edge_attr,
                size=(x.shape[0], context.shape[0]),
            )
        else:
            out = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return out


class GATEncoder(nn.Module):
    """Graph Attention Transformer encoder."""

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
        """Initialize the GAT encoder.

        Parameters
        ----------
        num_layers : int
            Number of layers
        in_channels : int
            Number of input channels
        hidden_channels : int
            Number of hidden channels
        out_channels : int
            Number of output channels
        num_heads : int, optional
            Number of heads in transformer, by default 4
        dropout : float, optional
            Dropout probability, by default 0.0
        activation : Optional[str], optional
            Activation function, by default "gelu"
        jk_mode : Optional[str], optional
            Jumping Knowledge mode (None, "last", "cat", "max", "lstm"), by default "last"
        edge_dim : int, optional
            Edge dimension of graph, by default 3
        """
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


class MessagePassingProcessorWraper(nn.Module):
    """Message Passing Processor Wraper for Graph Neural Network."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        edge_dim: int,
        chunks: int = 1,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        cpu_offload: bool = False,
    ) -> None:

        super().__init__()

        self.processor0 = MessagePassingProcessor(
            hidden_dim=hidden_dim,
            hidden_layers=int(hidden_layers/2),
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=edge_dim,
            chunks=chunks,
            activation=activation,
            emb_edges=True,
            cpu_offload=cpu_offload,
        )

        # Processor H -> H
        self.processor1 = MessagePassingProcessor(
            hidden_dim=hidden_dim,
            hidden_layers=int(hidden_layers/2),
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=0,
            chunks=chunks,
            activation=activation,
            emb_edges=False,
            cpu_offload=cpu_offload,
        )


    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, shape_nodes, mgroupdef) -> Tensor:

        x0, edge_attr0, edge_index0 = checkpoint(self.processor0,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            shape_nodes=shape_nodes,
            mgroupdef=mgroupdef,
            use_reentrant=False,
        )

        x1, _, _ = checkpoint(self.processor1,  # has skipped connections and checkpoints inside
            x=x0,
            edge_index=edge_index0,
            edge_attr=edge_attr0,
            shape_nodes=shape_nodes,
            mgroupdef=mgroupdef,
            use_reentrant=False,
        )

        return x1


class MessagePassingProcessor(nn.Module):
    """Message Passing Processor Graph Neural Network."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        edge_dim: int,
        chunks: int = 2,
        emb_edges: bool = True,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        cpu_offload: bool = False,
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
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        """
        super().__init__()

        self.hidden_layers = chunks
        chunk_size = int(hidden_layers / chunks)
        assert hidden_layers % chunks == 0

        # needed in mapper
        self.hidden_dim = hidden_dim
        self.mlp_extra_layers = mlp_extra_layers
        self.activation = activation
        self.emb_edges = emb_edges

        if self.emb_edges:
            self.emb_edges = gen_mlp(
                in_features=edge_dim,
                hidden_dim=hidden_dim,
                out_features=hidden_dim,
                n_extra_layers=mlp_extra_layers,
                activation_func=activation,
            )

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

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, shape_nodes, mgroupdef) -> Tensor:

        if self.emb_edges:
            shapes_edge_idx = get_shape_shards1(edge_index, 1, mgroupdef[0])
            shapes_edge_attr = get_shape_shards1(edge_attr, 0, mgroupdef[0])
            edge_index = shard_tensor1(edge_index, 1, shapes_edge_idx, mgroupdef[0])
            edge_attr = shard_tensor1(edge_attr, 0, shapes_edge_attr, mgroupdef[0])

            edge_attr = checkpoint(self.emb_edges, edge_attr, use_reentrant=False)

        # for i in range(self.hidden_layers):
            # x, edge_attr = checkpoint(
            #     self.proc[i], x, edge_index, edge_attr, (shape_nodes, shape_nodes), mgroupdef, use_reentrant=False
            # )

        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i]( x, edge_index, edge_attr, (shape_nodes, shape_nodes), mgroupdef)

        return x, edge_attr, edge_index # x_out[-1]


class MessagePassingMapper(MessagePassingProcessor):
    """Mapper from h -> era or era -> h."""

    def __init__(
        self,
        in_channels_src: int,
        in_channels_dst: int,
        backward_mapper: bool = False,
        out_channels_dst: Optional[int] = None,
        **kwargs,
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
        super().__init__(**kwargs)

        self.backward_mapper = backward_mapper
        self.out_channels_dst = out_channels_dst

        if backward_mapper:  # h -> era
            self.node_era_extractor = gen_mlp(
                in_features=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                out_features=out_channels_dst,
                n_extra_layers=self.mlp_extra_layers + 1,
                activation_func=self.activation,
                layer_norm=False,
                final_activation=False,
            )
        else:  # era -> h
            self.emb_nodes_src = gen_mlp(
                in_features=in_channels_src,
                hidden_dim=self.hidden_dim,
                out_features=self.hidden_dim,
                n_extra_layers=self.mlp_extra_layers,
                activation_func=self.activation,
            )

            self.emb_nodes_dst = gen_mlp(
                in_features=in_channels_dst,
                hidden_dim=self.hidden_dim,
                out_features=self.hidden_dim,
                n_extra_layers=self.mlp_extra_layers,
                activation_func=self.activation,
            )

    def forward(self, x: PairTensor, edge_index: Adj, edge_attr: Tensor, shape_nodes, size, mgroupdef) -> PairTensor:
        shapes_edge_idx = get_shape_shards1(edge_index, 1, mgroupdef[0])
        shapes_edge_attr = get_shape_shards1(edge_attr, 0, mgroupdef[0])
        edge_index = shard_tensor1(edge_index, 1, shapes_edge_idx, mgroupdef[0])
        edge_attr = shard_tensor1(edge_attr, 0, shapes_edge_attr, mgroupdef[0])
        edge_attr = self.emb_edges(edge_attr)

        x_src, x_dst = x
        shapes_src, shapes_dst = shape_nodes

        if not self.backward_mapper:
            x_src = shard_tensor1(x_src, 0, shapes_src, mgroupdef[0])
            x_dst = shard_tensor1(x_dst, 0, shapes_dst, mgroupdef[0])
            x_src = self.emb_nodes_src(x_src)
            x_dst = self.emb_nodes_dst(x_dst)
            shapes_src = change_channels_in_shape1(shapes_src, self.hidden_dim)
            shapes_dst = change_channels_in_shape1(shapes_dst, self.hidden_dim)

        for i in range(self.hidden_layers):
            (x_src, x_dst), edge_attr = self.proc[i](
                (x_src, x_dst), edge_index, edge_attr, (shapes_src, shapes_dst), mgroupdef, size=size
            )

        if self.backward_mapper:
            x_dst = self.node_era_extractor(x_dst)
            x_dst = gather_tensor1(x_dst, 0, change_channels_in_shape1(shapes_dst, self.out_channels_dst), mgroupdef[0])
            # x_src not gathered because not used ...

        return x_src, x_dst


class MessagePassingProcessorChunk(nn.Module):
    """Wraps X message passing blocks for checkpointing in Processor / Mapper."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
    ) -> None:
        """Initialize MessagePassingProcessorChunk.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimention of the message passing blocks.
        hidden_layers : int
            Number of message passing blocks.
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        """
        super().__init__()

        self.hidden_layers = hidden_layers

        self.proc = nn.ModuleList(
            [
                MessagePassingBlock(hidden_dim, hidden_dim, mlp_extra_layers=mlp_extra_layers, activation=activation)
                for _ in range(self.hidden_layers)
            ]
        )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, edge_attr, shapes, mgroupdef, size: Size = None):
        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr, shapes, mgroupdef, size=size)

        return x, edge_attr


class MessagePassingBlock(nn.Module):
    """Message passing block with MLPs for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        **kwargs,
    ) -> None:
        """Initialize MessagePassingBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        """
        super().__init__(**kwargs)

        self.node_mlp = gen_mlp(
            2 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )

        self.conv = NodeEdgeInteractions(
            in_channels=in_channels, out_channels=out_channels, mlp_extra_layers=mlp_extra_layers, activation=activation
        )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, edge_attr, shapes, mgroupdef, size: Size = None):
        if isinstance(x, Tensor):
            x_in = gather_tensor1(x, 0, shapes[1], mgroupdef[0])
            # out = torch.zeros_like(x_in)
            nchunks = 1
        else:
            x_src = gather_tensor1(x[0], 0, shapes[0], mgroupdef[0])
            x_dst = gather_tensor1(x[1], 0, shapes[1], mgroupdef[0])
            x_in = (x_src, x_dst)
            out = torch.zeros_like(x_dst)
            nchunks = 4

        if nchunks > 1:
            edge_index_list = torch.tensor_split(edge_index, nchunks, dim=1)
            edge_attr_list = torch.tensor_split(edge_attr, nchunks, dim=0)
            edges_out = []
            for i in range(nchunks):
                out1, edges_out1 = self.conv(x_in, edge_index_list[i], edge_attr_list[i], size=size)
                # out1, edges_out1 = checkpoint(self.conv, x_in, edge_index_list[i], edge_attr_list[i], size=size, use_reentrant=False)
                edges_out.append(edges_out1)
                out += out1
            edges_new = torch.cat(edges_out, dim=0)
        else:
            out, edges_new = self.conv(x_in, edge_index, edge_attr, size=size)            

        out = reduce_tensor1(out, mgroupdef[0])  # complete aggregation of edges
        out = shard_tensor1(out, 0, shapes[1], mgroupdef[0])

        if isinstance(x, Tensor):
            nodes_new = self.node_mlp(torch.cat([x, out], dim=1)) + x
        else:
            nodes_new_dst = self.node_mlp(torch.cat([x[1], out], dim=1)) + x[1]
            nodes_new_src = self.node_mlp(torch.cat([x[0], x[0]], dim=1)) + x[0]  # todo: only update this in the forward mapper...
            nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edges_new #edges_out


class NodeEdgeInteractions(MessagePassing):
    """Message passing block with MLPs for node and edge embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        **kwargs,
    ) -> None:
        """Initialize MessagePassingBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        """
        super().__init__(**kwargs)

        self.edge_mlp = gen_mlp(
            3 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor, size: Size = None):
        if isinstance(x, Tensor):
            dim_size = x.shape[0]
        else:
            dim_size = x[1].shape[0]

        out, edges_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, dim_size=dim_size)

        return out, edges_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, dim_size: Optional[int] = None) -> Tensor:
        edges_new = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=1)) + edge_attr

        return edges_new

    def aggregate(self, edges_new: Tensor, edge_index: Adj, dim_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        out = scatter(edges_new, edge_index[1], dim=0, dim_size=dim_size, reduce="sum")

        return out, edges_new


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
