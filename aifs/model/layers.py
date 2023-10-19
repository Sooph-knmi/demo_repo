from typing import Optional
from typing import Tuple
from typing import Union

import einops
import torch
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

    mlp1 = nn.Sequential(nn.Linear(in_features, hidden_dim), act_func())
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


class NoisyMessagePassingProcessor(nn.Module):
    """Message Passing Processor Graph Neural Network."""

    def __init__(
        self,
        hidden_dim: int,
        noise_dim: int,
        hidden_layers: int,
        edge_dim: int,
        chunks: int = 2,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        mlp_dropout: float = 0.0,
    ) -> None:
        """Initialize MessagePassingProcessor.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension
        noise_dim : int
            Noise dimension
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

        # needed in mapper
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.mlp_extra_layers = mlp_extra_layers
        self.activation = activation
        self.mlp_dropout = mlp_dropout

        self.emb_edges = gen_mlp(
            in_features=edge_dim,
            hidden_dim=hidden_dim + noise_dim,
            out_features=hidden_dim + noise_dim,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )

        self.proc = nn.ModuleList(
            [
                MessagePassingProcessorChunk(
                    hidden_dim + noise_dim,
                    hidden_layers=chunk_size,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(self.hidden_layers)
            ]
        )

        self.emb_nodes_out = gen_mlp(
            in_features=hidden_dim + noise_dim,
            hidden_dim=hidden_dim + noise_dim,
            out_features=hidden_dim,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
        )

    def forward(self, x_noisy: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        edge_attr = checkpoint(self.emb_edges, edge_attr, use_reentrant=False)

        for i in range(self.hidden_layers):
            x_noisy, edge_attr = checkpoint(self.proc[i], x_noisy, edge_index, edge_attr, use_reentrant=False)

        return self.emb_nodes_out(x_noisy)


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

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        edge_attr = checkpoint(self.emb_edges, edge_attr, use_reentrant=False)

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

    def forward(self, x: PairTensor, edge_index: Adj, edge_attr: Tensor) -> PairTensor:
        if self.backward_mapper:
            x_src, x_dst = x
        else:
            x_src = self.emb_nodes_src(x[0])
            x_dst = self.emb_nodes_dst(x[1])

        edge_attr = self.emb_edges(edge_attr)

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
        hidden_layers: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        mlp_dropout: float = 0.0,
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
                MessagePassingBlock(
                    hidden_dim,
                    hidden_dim,
                    mlp_extra_layers=mlp_extra_layers,
                    activation=activation,
                    mlp_dropout=mlp_dropout,
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
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        mlp_dropout: float = 0.0,
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
        mlp_dropout: float, optional
            Dropout rate for the MLPs.
        """
        super().__init__(**kwargs)

        self.node_mlp = gen_mlp(
            2 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
            dropout=mlp_dropout,
        )
        self.edge_mlp = gen_mlp(
            3 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation_func=activation,
            dropout=mlp_dropout,
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


if __name__ == "__main__":
    import numpy as np

    bs, nlatlon, nfeat = 2, 1024, 64
    hdim, ofeat = 128, 36
    nnoise = 4
    x_in = torch.randn((bs, nlatlon, nfeat), dtype=torch.float32, requires_grad=True)
    z_in = torch.randn((bs, nlatlon, nnoise), dtype=torch.float32, requires_grad=False)
    edim = 3

    noisy_processor = NoisyMessagePassingProcessor(
        hidden_dim=nfeat,
        noise_dim=nnoise,
        hidden_layers=2,
        edge_dim=edim,
    )

    nedges = 4500
    eattr = torch.randn(nedges, edim)
    eidx = torch.randint(0, nlatlon, size=(2, nedges))

    eattr_batched = torch.cat([einops.repeat(eattr, "e f -> (repeat e) f", repeat=bs)], dim=-1)
    edge_inc = torch.from_numpy(np.asarray([[nlatlon], [nlatlon]], dtype=np.int64))
    eidx_batched = torch.cat([eidx + i * edge_inc for i in range(bs)], dim=1)

    noise_injector = NoiseInjector(nfeat, nnoise)

    x_in = einops.rearrange(x_in, "bs n f -> (bs n) f")
    z_in = einops.rearrange(z_in, "bs n f -> (bs n) f")

    xz = noise_injector(x_in, z_in)

    x_out = noisy_processor(torch.cat([x_in, z_in], dim=-1), eidx_batched, eattr_batched)
    LOGGER.debug("x_out.shape = %s", x_out.shape)

    processor = MessagePassingProcessor(
        hidden_dim=nfeat,
        hidden_layers=2,
        edge_dim=edim,
    )

    x_out_v2 = processor(xz, eidx_batched, eattr_batched)

    # outputs must both have the same shape
    assert x_out.shape == x_out_v2.shape
