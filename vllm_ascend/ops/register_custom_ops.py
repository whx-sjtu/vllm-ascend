import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter,
                              get_ep_group,
                              get_dp_group)
from vllm.forward_context import get_forward_context
from vllm.utils import direct_register_custom_op

import vllm_ascend.envs as envs_ascend


def _maybe_chunk_residual_impl(x: torch.Tensor,
                               residual: torch.Tensor) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        flashcomm_v1_enabled = forward_context.flashcomm_v1_enabled
        assert flashcomm_v1_enabled is True, (
            "Currently, this situation only occurs "
            "when flashcomm_v1 is enabled")
        pad_size = forward_context.pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor,
                                           label: bool, is_moe: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    flashcomm_v1_enabled = forward_context.flashcomm_v1_enabled
    if flashcomm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_moe:
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = forward_context.pad_size
            if pad_size > 0:
                x = x[:-pad_size, :]
        else:
            x = get_ep_group().all_gather(x, 0)
            # unpad
            cu_tokens_across_dp_cpu = dp_metadata.cu_tokens_across_dp_cpu
            result = torch.empty((cu_tokens_across_dp_cpu[-1], *x.shape[1:]),
                                        device=x.device,
                                        dtype=x.dtype)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, forward_context.padded_length, *x.shape[1:])
            for idx in range(dp_size):
                start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
                end = cu_tokens_across_dp_cpu[idx]
                num_tokens_dp = end - start
                result[start:end, :] = x[idx, :num_tokens_dp, :]
            x = result

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_moe: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    if not forward_context.flashcomm_v1_enabled:
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_moe:
        pad_size = forward_context.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        # padding
        dp_size = get_dp_group().world_size
        cu_tokens_across_dp_cpu = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
        padded_x = torch.empty((dp_size, forward_context.padded_length, *x.shape[1:]),
                                           device=x.device,
                                           dtype=x.dtype)
        for idx in range(dp_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            num_tokens_dp = end - start
            padded_x[idx, :num_tokens_dp] = x[start:end]

        return get_ep_group().reduce_scatter(padded_x.view(-1, *x.shape[1:]), 0)


def _maybe_prefetch_mlp_gate_up_proj_impl(x_dependency: torch.Tensor,
                                          prefix: str) -> None:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return

    if not forward_context.prefetch_mlp_enabled:
        return
    model_instance = forward_context.model_instance
    prefetch_stream = forward_context.prefetch_stream
    layer_idx = int(prefix.split('.')[2])

    # start point of gate_up_proj weight prefetch
    if prefix.split('.')[-2] == "self_attn":
        forward_context.prefetch_mlp_gate_up_proj = True
    if forward_context.prefetch_mlp_gate_up_proj:
        prefetch_stream.wait_stream(torch.npu.current_stream())

        with torch.npu.stream(prefetch_stream):
            mlp_gate_up_prefetch_size = envs_ascend.VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE
            torch_npu.npu_prefetch(model_instance.model.layers[layer_idx].mlp.gate_up_proj.weight, \
                                x_dependency, mlp_gate_up_prefetch_size)
    return


def _maybe_prefetch_mlp_gate_up_proj_impl_fake(x_dependency: torch.Tensor,
                                               prefix: str) -> None:
    return


def _maybe_prefetch_mlp_down_proj_impl(x_dependency: torch.Tensor) -> None:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return

    if not forward_context.prefetch_mlp_enabled:
        return
    forward_context.prefetch_mlp_down_proj = True
    model_instance = forward_context.model_instance
    prefetch_stream = forward_context.prefetch_stream
    layer_idx = forward_context.layer_idx

    # start point of down_proj weight prefetch
    prefetch_stream.wait_stream(torch.npu.current_stream())

    with torch.npu.stream(prefetch_stream):
        mlp_down_prefetch_size = envs_ascend.VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE
        torch_npu.npu_prefetch(model_instance.model.layers[layer_idx].mlp.down_proj.weight, \
                            x_dependency, mlp_down_prefetch_size)
    forward_context.layer_idx += 1
    return


def _maybe_prefetch_mlp_down_proj_impl_fake(
        x_dependency: torch.Tensor) -> None:
    return


def _maybe_wait_prefetch_done_impl(x: torch.Tensor) -> None:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return

    if not forward_context.prefetch_mlp_enabled:
        return
    if forward_context.prefetch_mlp_gate_up_proj or \
        forward_context.prefetch_mlp_down_proj:
        prefetch_stream = forward_context.prefetch_stream
        # wait until prefetch done
        torch.npu.current_stream().wait_stream(prefetch_stream)
        forward_context.prefetch_mlp_gate_up_proj = False
        forward_context.prefetch_mlp_down_proj = False
    return


def _maybe_wait_prefetch_done_impl_fake(x: torch.Tensor) -> None:
    return


direct_register_custom_op(op_name="maybe_chunk_residual",
                          op_func=_maybe_chunk_residual_impl,
                          fake_impl=lambda x, residual: residual,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_all_gather_and_maybe_unpad",
                          op_func=_maybe_all_gather_and_maybe_unpad_impl,
                          fake_impl=lambda x, label: x,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_pad_and_reduce",
                          op_func=_maybe_pad_and_reduce_impl,
                          fake_impl=lambda x: x,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_prefetch_mlp_gate_up_proj",
                          op_func=_maybe_prefetch_mlp_gate_up_proj_impl,
                          fake_impl=_maybe_prefetch_mlp_gate_up_proj_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_prefetch_mlp_down_proj",
                          op_func=_maybe_prefetch_mlp_down_proj_impl,
                          fake_impl=_maybe_prefetch_mlp_down_proj_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_wait_prefetch_done",
                          op_func=_maybe_wait_prefetch_done_impl,
                          fake_impl=_maybe_wait_prefetch_done_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")
