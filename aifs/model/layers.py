from typing import Optional
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size
from torch_geometric.utils import scatter

from aifs.distributed.helpers import change_channels_in_shape
from aifs.distributed.helpers import gather_tensor
from aifs.distributed.helpers import get_shape_shards
from aifs.distributed.helpers import reduce_shard_tensor
from aifs.distributed.helpers import shard_tensor
from aifs.distributed.helpers import sync_tensor
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class AutocastLayerNorm(nn.LayerNorm):
    """LayerNorm with output autocast to x.type.

    During mixed-precision training, this will cast the LayerNorm output back to
    (b)float16 (from float32).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.dtype
        return super().forward(x).to(dtype=t)


def gen_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    n_extra_layers: int = 0,
    activation: str = "SiLU",
    dropout: float = 0.0,
    final_activation: bool = False,
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
    activation : str, optional
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
        act_func = getattr(nn, activation)
    except AttributeError as ae:
        LOGGER.error("Activation function %s not supported", activation)
        raise RuntimeError from ae

    mlp1 = nn.Sequential(nn.Linear(in_features, hidden_dim), act_func())
    for _ in range(n_extra_layers + 1):
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
    """Processor."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int,
        edge_dim: int,
        chunks: int = 2,
        mlp_dropout: float = 0.0,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        cpu_offload: bool = False,
    ) -> None:
        """Initialize GNNProcessor.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension (includes the noise channels)
        output_dim: int
            Output dimensions (noise channels removed)
        hidden_layers : int
            Number of hidden layers
        edge_dim : int
            Input features of edge MLP
        chunks : int, optional
            Number of chunks, by default 2
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        mlp_dropout: float, optional
            Rate of dropout for the MLPs
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
            if i > 0:
                edge_dim = None  # only embbed edges in first chunk
            self.proc.append(
                GNNProcessorChunk(
                    hidden_dim,
                    hidden_layers=chunk_size,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    edge_dim=edge_dim,
                    mlp_dropout=mlp_dropout,
                )
            )

        # final node embedding - gets us rid of the extra noise channels
        self.out_embedding = hidden_dim != output_dim
        self.out_emb_nodes = gen_mlp(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
            out_features=output_dim,
            dropout=mlp_dropout,
            final_activation=True,
        )

        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, shape_nodes: Tuple, model_comm_group: ProcessGroup) -> Tensor:
        shapes_edge_idx = get_shape_shards(edge_index, 1, model_comm_group)
        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        for i in range(self.hidden_layers):
            x, edge_attr = checkpoint(
                self.proc[i], x, edge_index, edge_attr, (shape_nodes, shape_nodes), model_comm_group, use_reentrant=False
            )

        return self.out_emb_nodes(x) if self.out_embedding else x


class GNNProcessorChunk(nn.Module):
    """Wraps edge embedding and X message passing blocks for checkpointing in
    Processor."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        mlp_extra_layers: int = 0,
        mlp_dropout: float = 0.0,
        activation: str = "SiLU",
        edge_dim: Optional[int] = None,
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
        mlp_dropout: float, optional
            Rate of dropout for the MLPs
        activation : str, optional
            Activation function, by default "SiLU"
        edge_dim: int, by default None
            Embedd edges with input dimension edge_dim,
            if None: assume embedding is not required
        """
        super().__init__()

        self.hidden_layers = hidden_layers

        if edge_dim:
            self.emb_edges = gen_mlp(
                in_features=edge_dim,
                hidden_dim=hidden_dim,
                out_features=hidden_dim,
                n_extra_layers=mlp_extra_layers,
                dropout=mlp_dropout,
                activation=activation,
            )
        else:
            self.emb_edges = None

        self.proc = nn.ModuleList(
            [
                GNNBlock(
                    hidden_dim,
                    hidden_dim,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    mlp_dropout=mlp_dropout,
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
        model_comm_group: ProcessGroup,
        size: Size = None,
    ):
        x_out = x * 1.0  # required for pytorch >= 2.1
        if self.emb_edges:
            edge_attr = self.emb_edges(edge_attr)

        for i in range(self.hidden_layers):
            x_out, edge_attr = self.proc[i](x_out, edge_index, edge_attr, shapes, model_comm_group, size=size)

        return x_out, edge_attr


class GNNMapper(nn.Module):
    """Mapper from h -> era or era -> h."""

    def __init__(
        self,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        edge_dim: int,
        mlp_extra_layers: int = 0,
        mlp_dropout: float = 0.0,
        activation: str = "SiLU",
        num_chunks: int = 1,
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
        mlp_dropout: float, optional
            Rate of dropout for the MLPs
        activation : str, optional
            Activation function, by default "SiLU"
        num_chunks : int
            Do message passing in X chunks
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

        self.emb_edges = gen_mlp(
            in_features=edge_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            n_extra_layers=mlp_extra_layers,
            activation=activation,
            dropout=mlp_dropout,
        )

        update_src_nodes = not backward_mapper
        self.proc = GNNBlock(
            hidden_dim,
            hidden_dim,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
            mlp_dropout=mlp_dropout,
        )

        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

        if backward_mapper:  # h -> era
            self.node_era_extractor = gen_mlp(
                in_features=hidden_dim,
                hidden_dim=hidden_dim,
                out_features=out_channels_dst,
                n_extra_layers=mlp_extra_layers,
                activation=activation,
                layer_norm=False,
                final_activation=False,
                dropout=mlp_dropout,
            )
        else:  # era -> h
            self.emb_nodes_src = gen_mlp(
                in_features=in_channels_src,
                hidden_dim=hidden_dim,
                out_features=hidden_dim,
                n_extra_layers=mlp_extra_layers,
                activation=activation,
                dropout=mlp_dropout,
            )

            self.emb_nodes_dst = gen_mlp(
                in_features=in_channels_dst,
                hidden_dim=hidden_dim,
                out_features=hidden_dim,
                n_extra_layers=mlp_extra_layers,
                activation=activation,
                dropout=mlp_dropout,
            )

    def forward(
        self, x: PairTensor, edge_index: Adj, edge_attr: Tensor, shape_nodes: Tuple, size: Size, model_comm_group: ProcessGroup
    ) -> PairTensor:
        shapes_edge_idx = get_shape_shards(edge_index, 1, model_comm_group)
        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)
        edge_attr = self.emb_edges(edge_attr)

        x_src, x_dst = x
        shapes_src, shapes_dst = shape_nodes

        if not self.backward_mapper:
            x_src = shard_tensor(x_src, 0, shapes_src, model_comm_group)
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
            x_src = self.emb_nodes_src(x_src)
            x_dst = self.emb_nodes_dst(x_dst)
            shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
            shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst), edge_index, edge_attr, (shapes_src, shapes_dst), model_comm_group, size=size
        )

        if self.backward_mapper:
            x_dst = self.node_era_extractor(x_dst)
            x_dst = gather_tensor(x_dst, 0, change_channels_in_shape(shapes_dst, self.out_channels_dst), model_comm_group)

        return x_src, x_dst


class GNNBlock(nn.Module):
    """Message passing block with MLPs for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        mlp_dropout: float = 0.0,
        update_src_nodes: bool = True,
        num_chunks: int = 1,
        **kwargs,
    ) -> None:
        """Initialize GNNBlock.

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
        mlp_dropout: float, optional
            Dropout rate for the MLPs.
        update_src_nodes: bool, by default True
            Update src if src and dst nodes are given
        num_chunks : int, by default 1
            do message passing in X chunks
        """
        super().__init__(**kwargs)

        self.update_src_nodes = update_src_nodes
        self.num_chunks = num_chunks

        self.node_mlp = gen_mlp(
            2 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation=activation,
            dropout=mlp_dropout,
        )

        self.conv = NodeEdgeInteractions(
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            mlp_dropout=mlp_dropout,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_index: Adj,
        edge_attr: Tensor,
        shapes: Tuple,
        model_comm_group: ProcessGroup,
        size: Size = None,
    ):
        if isinstance(x, Tensor):
            x_in = sync_tensor(x, 0, shapes[1], model_comm_group)
        else:
            x_src = sync_tensor(x[0], 0, shapes[0], model_comm_group)
            x_dst = sync_tensor(x[1], 0, shapes[1], model_comm_group)
            x_in = (x_src, x_dst)

        if self.num_chunks > 1:
            edge_index_list = torch.tensor_split(edge_index, self.num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edge_attr, self.num_chunks, dim=0)
            edges_out = []
            for i in range(self.num_chunks):
                out1, edges_out1 = self.conv(x_in, edge_index_list[i], edge_attr_list[i], size=size)
                edges_out.append(edges_out1)
                if i == 0:
                    out = out1
                else:
                    out = out + out1
            edges_new = torch.cat(edges_out, dim=0)
        else:
            out, edges_new = self.conv(x_in, edge_index, edge_attr, size=size)

        out = reduce_shard_tensor(out, 0, shapes[1], model_comm_group)

        if isinstance(x, Tensor):
            nodes_new = self.node_mlp(torch.cat([x, out], dim=1)) + x
        else:
            nodes_new_dst = self.node_mlp(torch.cat([x[1], out], dim=1)) + x[1]

            if self.update_src_nodes:  # update only needed in forward mapper
                nodes_new_src = self.node_mlp(torch.cat([x[0], x[0]], dim=1)) + x[0]
            else:
                nodes_new_src = x[0]

            nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edges_new


class NodeEdgeInteractions(MessagePassing):
    """Message passing module for node and edge interactions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        mlp_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize NodeEdgeInteractions.

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
        mlp_dropout: float, optional
            Dropout rate for the MLPs.
        """
        super().__init__(**kwargs)

        self.edge_mlp = gen_mlp(
            3 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            dropout=mlp_dropout,
            activation=activation,
        )

    def forward(self, x: OptPairTensor, edge_index: Adj, edge_attr: Tensor, size: Size = None):
        if isinstance(x, Tensor):
            dim_size = x.shape[0]
        else:
            dim_size = x[1].shape[0]

        out, edges_new = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, dim_size=dim_size)

        return out, edges_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, dim_size: Optional[int] = None) -> Tensor:
        del dim_size  # not used
        edges_new = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=1)) + edge_attr
        return edges_new

    def aggregate(self, edges_new: Tensor, edge_index: Adj, dim_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        out = scatter(edges_new, edge_index[1], dim=0, dim_size=dim_size, reduce="sum")
        return out, edges_new


class NoiseInjector(nn.Module):
    """Inject noise into the processor using a technique similar to conditional instance
    normalization.

    See https://arxiv.org/pdf/2202.07773.pdf, equation (5).
    We may want to test this out later.
    """

    def __init__(self, num_inputs: int, num_noise_inputs: int) -> None:
        """
        Args:
            num_inputs: number of physical (latent) input channels
            num_noise_inputs: number of noise input channels
        """
        super().__init__()
        self.in_norm = nn.InstanceNorm1d(num_features=num_inputs, affine=False)
        self.alpha = nn.Linear(in_features=num_noise_inputs, out_features=num_inputs)
        self.beta = nn.Linear(in_features=num_noise_inputs, out_features=num_inputs)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        return self.alpha(z) * x + self.beta(z)


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


# if __name__ == "__main__":
#     # TODO: fix this to run independently (?) will need to set up a "dummy" comm_group
#     import numpy as np

#     bs, nlatlon, nfeat = 2, 1024, 64
#     hdim, ofeat = 128, 36
#     nnoise = 4
#     x_ = torch.randn((bs, nlatlon, nfeat), dtype=torch.float32, requires_grad=True)
#     z_ = torch.randn((bs, nlatlon, nnoise), dtype=torch.float32, requires_grad=False)
#     edim = 3

#     noisy_processor = GNNProcessor(
#         hidden_dim=nfeat + nnoise,
#         hidden_layers=2,
#         output_dim=nfeat,
#         edge_dim=edim,
#     )

#     nedges = 4500
#     eattr = torch.randn(nedges, edim)
#     eidx = torch.randint(0, nlatlon, size=(2, nedges))

#     eattr_batched = torch.cat([einops.repeat(eattr, "e f -> (repeat e) f", repeat=bs)], dim=-1)
#     edge_inc = torch.from_numpy(np.asarray([[nlatlon], [nlatlon]], dtype=np.int64))
#     eidx_batched = torch.cat([eidx + i * edge_inc for i in range(bs)], dim=1)

#     noise_injector = NoiseInjector(nfeat, nnoise)

#     x_ = einops.rearrange(x_, "bs n f -> (bs n) f")
#     z_ = einops.rearrange(z_, "bs n f -> (bs n) f")

#     x_out = noisy_processor(torch.cat([x_, z_], dim=-1), eidx_batched, eattr_batched)
#     LOGGER.debug("x_out.shape = %s", x_out.shape)
