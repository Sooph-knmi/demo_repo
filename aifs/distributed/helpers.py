# some following / adapted from https://github.com/NVIDIA/modulus/blob/main/modulus/utils/sfno/distributed/helpers.py
# Apache License -> http://www.apache.org/licenses/LICENSE-2.0
# License: https://github.com/NVIDIA/modulus/blob/b18419e9460f6acd3cd3d175f5d6caf6bbc9d2da/modulus/utils/sfno/distributed/helpers.py#L1C6-L1C6
# todo: add proper license information etc.
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def shard_tensor(input_: Tensor, dim: int, shapes: Tuple, mgroup: ProcessGroup) -> Tensor:
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

    Returns
    -------
    Tensor
    """

    return _ShardParallelSection.apply(input_, dim, shapes, mgroup)


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
    def forward(ctx, input_, dim_, shapes_, mgroup_):
        ctx.dim = dim_
        ctx.comm_group = mgroup_
        ctx.shapes = shapes_
        if mgroup_:
            return _split(input_, dim_, shapes_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.comm_group:
            grad_scaler = 1.0 / ctx.comm_group.size()
            return (
                # scale gradients
                grad_scaler * _gather(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


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
    del shapes_  # not used

    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along dim
    input_list = split_tensor_dim(input_, dim_, comm_size)

    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def split_tensor_dim(tensor: Tensor, dim: int, num_chunks: int) -> List[Tensor]:
    """Helper routine to split a tensor along a given dimension."""

    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    tensor_list = torch.tensor_split(tensor, num_chunks, dim=dim)

    return tensor_list


def _gather(input_: Tensor, dim_: int, shapes: Tuple, group: Optional[ProcessGroup] = None) -> Tensor:
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
