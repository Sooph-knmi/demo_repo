import math
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import OptTensor
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import scatter
from torch_geometric.utils import segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def get_activation_func(activation_func: str) -> Callable:
    try:
        act_func = getattr(nn, activation_func)
    except AttributeError as ae:
        LOGGER.error("Activation function %s not supported", activation_func)
        raise RuntimeError from ae
    return act_func


def gen_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    n_extra_layers: int = 0,
    activation_func: str = "SiLU",
    final_activation: bool = True,
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

    act_func = get_activation_func(activation_func)

    mlp1 = nn.Sequential()
    if start_with_layernorm:
        mlp1 = nn.Sequential(nn.LayerNorm(in_features))
    mlp1.append(nn.Sequential(nn.Linear(in_features, hidden_dim), act_func()))
    for _ in range(n_extra_layers):
        mlp1.append(nn.Linear(hidden_dim, hidden_dim))
        mlp1.append(act_func())
    mlp1.append(nn.Linear(hidden_dim, out_features))

    if final_activation:
        mlp1.append(act_func())

    if layer_norm:
        mlp1.append(nn.LayerNorm(out_features))

    return CheckpointWrapper(mlp1) if checkpoints else mlp1


class MessagePassingProcessor(nn.Module):
    """Message Passing Processor Graph Neural Network."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        edge_dim: int,
        chunks: int = 2,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        ptype: str = "proc",  # proc, fmapper, bmapper :this is ugly ... fix this!
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

        self.proc = nn.ModuleList(
            [
                MessagePassingProcessorChunk(
                    hidden_dim,
                    edge_dim,
                    hidden_layers=chunk_size,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    ptype=ptype,
                )
                for _ in range(self.hidden_layers)
            ]
        )

        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        for i in range(self.hidden_layers):
            x, edge_attr = checkpoint(self.proc[i], x, edge_index, edge_attr, use_reentrant=False)

        return x


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

        if backward_mapper:  # h -> era
            self.node_era_extractor = gen_mlp(
                in_features=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                out_features=out_channels_dst,
                n_extra_layers=self.mlp_extra_layers,
                activation_func=self.activation,
                layer_norm=False,
                final_activation=False,
                start_with_layernorm=True,
            )
        else:  # era -> h
            self.emb_nodes_src = gen_mlp(
                in_features=in_channels_src,
                hidden_dim=self.hidden_dim,
                out_features=self.hidden_dim,
                n_extra_layers=self.mlp_extra_layers,
                activation_func=self.activation,
                layer_norm=False,
            )

            self.emb_nodes_dst = gen_mlp(
                in_features=in_channels_dst,
                hidden_dim=self.hidden_dim,
                out_features=self.hidden_dim,
                n_extra_layers=self.mlp_extra_layers,
                activation_func=self.activation,
                layer_norm=False,
            )

    def forward(self, x: PairTensor, edge_index: Adj, edge_attr: Tensor) -> PairTensor:
        if self.backward_mapper:
            x_src, x_dst = x
        else:
            x_src = self.emb_nodes_src(x[0])
            x_dst = self.emb_nodes_dst(x[1])

        for i in range(self.hidden_layers):
            (x_src, x_dst), edge_attr = self.proc[i]((x_src, x_dst), edge_index, edge_attr, size=(x_src.shape[0], x_dst.shape[0]))

        if self.backward_mapper:
            x_dst = self.node_era_extractor(x_dst)

        return x_src, x_dst


class MessagePassingProcessorChunk(nn.Module):
    """Wraps X message passing blocks for checkpointing in Processor / Mapper."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        hidden_layers: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        ptype: str = "proc",
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
                TransformerBlock(
                    hidden_dim,
                    4 * hidden_dim,
                    hidden_dim,
                    edge_dim,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    ptype=ptype,
                )
                for _ in range(self.hidden_layers)
            ]
        )

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor, size: Size = None):
        for i in range(self.hidden_layers):
            x, edge_attr = self.proc[i](x, edge_index, edge_attr, size=size)

        return x, edge_attr


class MessagePassingBlock(MessagePassing):
    """Message passing block with MLPs for node and edge embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
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
            hidden_dim,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )
        self.edge_mlp = gen_mlp(
            3 * in_channels,
            hidden_dim,
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


class TransformerBlock(nn.Module):
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
        mlp_extra_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.ptype = ptype

        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=int(out_channels / heads),
            heads=heads,
            concat=concat,
            bias=bias,
            edge_dim=edge_dim,
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
        )

        self.layer_norm1 = nn.LayerNorm(in_channels)

        if ptype == "fmapper" or ptype == "bmapper":
            self.layer_norm2 = nn.LayerNorm(in_channels)

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
        x_skip = x

        if isinstance(x_skip, Tensor):
            x = self.layer_norm1(x)
        else:
            x = (self.layer_norm1(x[0]), self.layer_norm2(x[1]))

        out = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = self.proj(out)

        if isinstance(x_skip, Tensor):
            out = out + x_skip
            nodes_new = self.node_dst_mlp(out) + out
        else:
            out = out + x_skip[1]
            nodes_new_dst = self.node_dst_mlp(out) + out

            if self.ptype == "fmapper":
                nodes_new_src = self.node_src_mlp(x[0]) + x_skip[0]  # todo: only update this in the forward mapper...
            else:
                nodes_new_src = x[0]

            nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edge_attr


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
            \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
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
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

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

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None, return_attention_weights=None):
        """Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        alpha = self._alpha
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

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
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
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:

        >>> src = torch.tensor([1., 1., 1., 1.])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> ptr = torch.tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """
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
