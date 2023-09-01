# following / adapted from https://github.com/NVIDIA/modulus/blob/main/modulus/utils/sfno/distributed/helpers.py
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Apache License -> http://www.apache.org/licenses/LICENSE-2.0
# License:
# https://github.com/NVIDIA/modulus/blob/b18419e9460f6acd3cd3d175f5d6caf6bbc9d2da/modulus/utils/sfno/distributed/helpers.py#L1C6-L1C6
import torch
import torch.distributed as dist


def gather_tensor(input_, dim, shapes, mgroup):
    """Gather helper."""
    return _GatherParallelSection.apply(input_, dim, shapes, mgroup)


def reduce_tensor(input_, mgroup):
    """Reduce helper."""
    return _ReduceParallelSection.apply(input_, mgroup)


def sync_tensor(input_, dim, shapes, mgroup):
    """Sync helper."""
    return _SyncParallelSection.apply(input_, dim, shapes, mgroup)


class _GatherParallelSection(torch.autograd.Function):
    """Gather the input from parallel section and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_, mgroup_):
        """"""
        if mgroup_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

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
                _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _SyncParallelSection(torch.autograd.Function):
    """Sync the input from parallel section."""

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_, mgroup_):
        """"""
        if mgroup_:
            return _gather(input_, dim_, shapes_, group=mgroup_)
        return input_

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
            grad_output = _reduce(grad_output, group=ctx.comm_group)
            return (
                _split(grad_output, ctx.dim, ctx.shapes, group=ctx.comm_group),
                None,
                None,
                None,
            )
        return grad_output, None, None, None


class _ReduceParallelSection(torch.autograd.Function):
    """All-reduce the input from the parallel section."""

    @staticmethod
    def symbolic(graph, input_, mgroup_):
        """Symbolic method."""
        if mgroup_:
            return _reduce(input_, group=mgroup_)
        return input_

    @staticmethod
    def forward(ctx, input_, mgroup_):
        if mgroup_:
            return _reduce(input_, group=mgroup_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _split(input_, dim_, shapes_, group=None):  # pragma: no cover
    """Split the tensor along dim and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along dim
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # does torch.tensor_split create contiguous tensors by default?
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def split_tensor_along_dim(tensor, dim, num_chunks):
    """Helper routine to split a tensor along a given dimension."""
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"

    tensor_list = torch.tensor_split(tensor, num_chunks, dim=dim)
    return tensor_list


def _gather(input_, dim_, shapes, group=None):
    """Gather tensors and concatenate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert dim_ < input_.dim(), f"Error, cannot gather along dimension {dim_} for tensor with {input_.dim()} dimensions."

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


def _reduce(input_, use_fp32=True, group=None):
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
        dist.all_reduce(input_, group=group)

    return input_


def get_memory_format(tensor):
    """Helper routine to get the memory format."""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    return torch.contiguous_format
