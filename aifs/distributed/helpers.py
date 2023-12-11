# some following / adapted from https://github.com/NVIDIA/modulus/blob/main/modulus/utils/sfno/distributed/helpers.py
# Apache License -> http://www.apache.org/licenses/LICENSE-2.0
# License: https://github.com/NVIDIA/modulus/blob/b18419e9460f6acd3cd3d175f5d6caf6bbc9d2da/modulus/utils/sfno/distributed/helpers.py#L1C6-L1C6
# todo: add proper license information etc.
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.utils import bipartite_subgraph
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import mask_to_index

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def shard_tensor(input_: Tensor, dim: int, shapes: Tuple, mgroup: ProcessGroup, gather_in_backward: bool = True) -> Tensor:
    """Shard tensor.

    Keeps only part of the tensor that is relevant for the current rank.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to shard
    shapes : Tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group
    gather_in_backward : Optional, bool
        perform gather in backward, default True
    Returns
    -------
    Tensor
    """

    return _ShardParallelSection.apply(input_, dim, shapes, gather_in_backward, mgroup)


def gather_tensor(input_: Tensor, dim: int, shapes: Tuple, mgroup: ProcessGroup) -> Tensor:
    """Gather tensor.

    Gathers tensor shards from ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : Tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
    """

    return _GatherParallelSection.apply(input_, dim, shapes, mgroup)


def reduce_tensor(input_: Tensor, mgroup: ProcessGroup, use_fp32: bool = True) -> Tensor:
    """Reduce tensor.

    Reduces tensor across ranks.

    Parameters
    ----------
    input_ : Tensor
        Input
    mgroup : ProcessGroup
        model communication group
    use_fp32: bool
        Do the reduction using FP32.

    Returns
    -------
    Tensor
    """

    return _ReduceParallelSection.apply(input_, mgroup, use_fp32)


def sync_tensor(input_: Tensor, dim: int, shapes: Tuple, mgroup: ProcessGroup) -> Tensor:
    """Sync tensor.

    Perform a gather in the forward pass and an allreduce followed by a split in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : Tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tensor
    """

    return _SyncParallelSection.apply(input_, dim, shapes, mgroup)


def get_k_hop_edges(nodes: Tensor, edge_index: Adj, edge_attr: Tensor, num_hops: int = 1) -> Tuple[Adj, Tensor]:
    """Return 1 hop subgraph.

    Parameters
    ----------
    nodes : Tensor
        destination nodes
    edge_index : Adj
        edge index
    edge_attr : Tensor
        edge attributes
    num_hops: int, Optional, by default 1
        number of required hops

    Returns
    -------
    Tuple[Adj, Tensor]
    """
    nodes_k, edge_index_k, inv_k, edge_mask_k = k_hop_subgraph(
        node_idx=nodes, num_hops=num_hops, edge_index=edge_index, directed=True
    )

    return edge_index_k, edge_attr[mask_to_index(edge_mask_k)]


def sort_edges_1hop(
    num_nodes: Union[int, Tuple[int, int]], edge_index: Adj, edge_attr: Tensor, mgroup: Optional[ProcessGroup] = None
) -> Tuple[Adj, Tensor, List, List]:
    """Rearanges edges into 1 hop neighbourhoods for sharding across GPUs.

    Parameters
    ----------
    num_nodes : Union[int, Tuple[int, int]]
        Number of (target) nodes in Graph
    edge_index : Adj
        edge index
    edge_attr : Tensor
        edge attributes
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    Tuple[Adj, Tensor, List, List]
        edges sorted according to k hop neigh., edge attributes of sorted edges,
        shapes of edge indices for partitioning between GPUs, shapes of edge attr for partitioning between GPUs
    """

    if mgroup:
        num_chunks = dist.get_world_size(group=mgroup)

        if isinstance(num_nodes, int):
            node_chunks = torch.arange(num_nodes, device=edge_index.device).tensor_split(num_chunks)
        else:
            nodes_src = torch.arange(num_nodes[0], device=edge_index.device)
            node_chunks = torch.arange(num_nodes[1], device=edge_index.device).tensor_split(num_chunks)

        edge_index_list = []
        edge_attr_list = []
        for node_chunk in node_chunks:
            if isinstance(num_nodes, int):
                edge_index_chunk, edge_attr_chunk = get_k_hop_edges(node_chunk, edge_index, edge_attr)
            else:
                edge_index_chunk, edge_attr_chunk = bipartite_subgraph(
                    (nodes_src, node_chunk), edge_index, edge_attr, size=(num_nodes[0], num_nodes[1])
                )
            edge_index_list.append(edge_index_chunk)
            edge_attr_list.append(edge_attr_chunk)
        edge_index_shapes = [x.shape for x in edge_index_list]
        edge_attr_shapes = [x.shape for x in edge_attr_list]

        return torch.cat(edge_index_list, dim=1), torch.cat(edge_attr_list, dim=0), edge_index_shapes, edge_attr_shapes

    return edge_index, edge_attr, [], []


def reduce_shard_tensor(input_: Tensor, dim: int, shapes: Tuple, mgroup: ProcessGroup, use_fp32: bool = True) -> Tensor:
    """Reduces and then shards tensor.

    Perform an allreduce followed by a split in the forward pass and a gather in the backward pass.

    Parameters
    ----------
    input_ : Tensor
        Input
    dim : int
        dimension along which to gather
    shapes : Tuple
        Shapes of sharded Tensors
    mgroup : ProcessGroup
        model communication group
    use_fp32: bool
        Perform the reduction in fp32.

    Returns
    -------
    Tensor
    """

    return _ReduceShardParallelSection.apply(input_, dim, shapes, mgroup, use_fp32)


def get_shape_shards(tensor: Tensor, dim: int, mgroup: Optional[ProcessGroup] = None) -> List:
    """Get shape of tensor shards."""

    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    if mgroup:
        comm_size = dist.get_world_size(group=mgroup)
        if comm_size == 1:
            shape_list = [list(tensor.shape)]

        else:
            tensor_list = torch.tensor_split(tensor, comm_size, dim=dim)
            shape_list = [list(x.shape) for x in tensor_list]
    else:
        shape_list = []

    return shape_list


def change_channels_in_shape(shape_list: List, channels: int) -> List:
    """Change the number of channels in the tensor shape definition list."""

    if shape_list:
        out = [x[:-1] + [channels] for x in shape_list]
    else:
        out = []

    return out


class _SyncParallelSection(torch.autograd.Function):
    """Sync the input from parallel section."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, mgroup_, use_fp32_=True):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        ctx.use_fp32 = use_fp32_
        if mgroup_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            # no grad scaling needed
            grad_output = _reduce(grad_output, group=ctx.comm_group, use_fp32=ctx.use_fp32)
            return (
                _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ReduceShardParallelSection(torch.autograd.Function):
    """All-reduce and shard the input from the parallel section."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, mgroup_, use_fp32_=True):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            input_ = _reduce(input_, group=mgroup_, use_fp32=use_fp32_)
            return _split(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            # no grad scaling needed
            return (
                _gather(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ShardParallelSection(torch.autograd.Function):
    """Split the input and keep only the relevant chunk for each rank."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, gather_in_backward_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        ctx.gather_in_backward = gather_in_backward_
        if mgroup_:
            return _split(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            grad_scaler = 1.0 / ctx.comm_group.size()
            return (
                grad_scaler
                * _gather(grad_output, ctx.dim, ctx.shapes, gather_in_backward=ctx.gather_in_backward, group=ctx.comm_group),
                None,
                None,
                None,
                None,
            )
        else:
            return grad_output, None, None, None, None


class _GatherParallelSection(torch.autograd.Function):
    """Gather the input from parallel section and concatenate."""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            return (
                # scale gradients
                ctx.comm_group.size() * _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ReduceParallelSection(torch.autograd.Function):
    """All-reduce the input from the parallel section."""

    @staticmethod
    def forward(ctx, input_, mgroup_, use_fp32_=True):
        ctx.comm_group = mgroup_
        if mgroup_:
            return _reduce(input_, use_fp32=use_fp32_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            # scale gradient
            return ctx.comm_group.size() * grad_output, None, None
        return grad_output, None, None


def _split(input_: Tensor, dim_: int, shapes_: Tuple, group: ProcessGroup) -> Tensor:
    """Split the tensor along dim and keep the relevant slice."""

    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot split along {dim_} for tensor with {input_.dim()} dimensions."

    input_list = torch.split(input_, [x[dim_] for x in shapes_], dim=dim_)

    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def _gather(
    input_: Tensor, dim_: int, shapes: Tuple, gather_in_backward: Optional[bool] = True, group: Optional[ProcessGroup] = None
) -> Tensor:
    """Gather tensors and concatinate along the last dimension."""

    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    input_ = input_.contiguous(memory_format=input_format)
    tensor_list = [
        torch.empty(shapes[rank], dtype=input_.dtype, layout=input_.layout, device=input_.device, memory_format=input_format)
        for rank in range(comm_size)
    ]

    tensor_list[comm_rank] = input_
    if gather_in_backward:
        dist.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)

    return output


def _reduce(input_: Tensor, use_fp32: Optional[bool] = True, group: Optional[ProcessGroup] = None) -> Tensor:
    """All-reduce the input tensor across model parallel group."""

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        input_format = get_memory_format(input_)
        input_ = input_.contiguous(memory_format=input_format)
        dist.all_reduce(input_, group=group)

    return input_


def get_memory_format(tensor: Tensor):
    """Helper routine to get the memory format of a tensor."""

    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    return torch.contiguous_format
