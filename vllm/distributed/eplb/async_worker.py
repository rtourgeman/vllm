# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The async worker that transfers experts in the background.
"""

import threading
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import init_logger

from .eplb_utils import CpuGpuEvent
from .rebalance_execute import AsyncEplbLayerResult, transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)


def start_async_worker(
    state: "EplbState",
    is_profile: bool = False,
) -> threading.Thread:
    rank = get_eplb_group().device_group.rank()
    device_index = state.cuda_device_index
    assert state.is_async

    def thread_target() -> None:
        assert device_index is not None
        torch.accelerator.set_device_index(device_index)
        cuda_stream = torch.cuda.Stream(device=device_index)
        try:
            transfer_run_periodically(
                state=state,
                cuda_stream=cuda_stream,
                is_profile=is_profile,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def run_rebalance_experts(
    model_state: "EplbModelState",
    eplb_state: "EplbState",
    physical_to_logical_map_cpu: torch.Tensor,
    cuda_stream: torch.cuda.Stream,
) -> torch.Tensor:
    assert model_state.eplb_stats is not None
    eplb_stats = model_state.eplb_stats

    # Move the global expert load window to CPU for computation.
    with torch.cuda.stream(cuda_stream):
        global_expert_load_window = eplb_stats.global_expert_load_window.cpu()
    # Compute new expert mappings for the model
    new_physical_to_logical_map = eplb_state.policy.rebalance_experts(
        global_expert_load_window,
        eplb_stats.num_replicas,
        eplb_stats.num_groups,
        eplb_stats.num_nodes,
        eplb_stats.num_gpus,
        physical_to_logical_map_cpu,
    )
    assert new_physical_to_logical_map.device == torch.device("cpu")

    return new_physical_to_logical_map


def transfer_run_periodically(
    state: "EplbState",
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
) -> None:
    while True:
        # Signal idle before blocking so the main thread can quiesce us safely.
        state.async_idle_event.set()
        state.rearrange_event.wait(stream=cuda_stream)
        state.async_idle_event.clear()
        logger.info("async worker woke up for EPLB transfer")

        # An elastic reshuffle is taking over the shared communicator; abandon
        # any signalled rearrange and return to idle without touching NCCL.
        if state.async_quiesce_event.is_set():
            for model_state in state.model_states.values():
                model_state.rebalanced = False
                model_state.pending_result = None
            continue

        # Re-fetch the EPLB group on every wakeup. Elastic scaling replaces
        # the active EPLB group, so a group captured when the worker started
        # would be stale (wrong EP size) after a scale transition.
        eplb_group = get_eplb_group().device_group

        assert state.is_async
        for model_state in state.model_states.values():
            layer_idx = 0
            # Set the async worker's CUDA stream on the communicator
            model_state.communicator.set_stream(cuda_stream)
            num_layers = model_state.model.num_moe_layers

            # Snapshot the physical_to_logical_map (synchronized with
            # rearrange_event) and copy it to CPU
            with torch.cuda.stream(cuda_stream):
                physical_to_logical_map_cpu = model_state.physical_to_logical_map.cpu()

            new_physical_to_logical_map = run_rebalance_experts(
                model_state, state, physical_to_logical_map_cpu, cuda_stream
            )
            logger.info(
                "Async worker computed new indices for model %s",
                model_state.model_name,
            )

            # Execute one EPLB layer transfer per model forward pass. Each iteration
            # of this loop will copy the new set of expert weights into
            # model_state.expert_buffer, which will be consumed by the main thread in
            # move_to_workspace
            while model_state.rebalanced and layer_idx < num_layers:
                # Stop touching the shared communicator if the main thread is
                # about to run an elastic reshuffle. Abandon the in-flight
                # rearrange; it will be recomputed after the transition.
                if state.async_quiesce_event.is_set():
                    model_state.rebalanced = False
                    model_state.pending_result = None
                    break
                transfer_metadata = transfer_layer(
                    old_layer_indices=physical_to_logical_map_cpu[layer_idx],
                    new_layer_indices=new_physical_to_logical_map[layer_idx],
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weights_buffer=model_state.expert_buffer,
                    communicator=model_state.communicator,
                    ep_group=eplb_group,
                    is_profile=is_profile,
                    cuda_stream=cuda_stream,
                    layer_idx=layer_idx,
                )

                # Wait until all writes to expert_buffer have finished before making the
                # AsyncEplbLayerResult visible to the main thread.
                cuda_stream.synchronize()

                # This event guarantees that expert_buffer will not be overwritten by
                # subsequent iterations of this loop until the main thread has consumed
                # it. Record is called by the main thread after move_from_buffer().
                consumed_event = CpuGpuEvent()

                model_state.pending_result = AsyncEplbLayerResult(
                    layer_idx=layer_idx,
                    new_physical_to_logical_map=new_physical_to_logical_map[layer_idx],
                    transfer_metadata=transfer_metadata,
                    consumed_event=consumed_event,
                )

                # Block this thread until the main thread and main stream
                # finish copying model_state.expert_buffer into
                # model_state.model.expert_weights[layer_idx]
                logger.warning(
                    "[EPLB NCCL DEBUG] async worker PARKED thread=%s layer=%d/%d "
                    "comm_id=%s comm_group_started=%s waiting for main thread",
                    threading.get_ident(),
                    layer_idx,
                    num_layers,
                    id(model_state.communicator),
                    getattr(model_state.communicator, "_group_started", None),
                )
                consumed_event.wait(stream=cuda_stream)
                logger.warning(
                    "[EPLB NCCL DEBUG] async worker RESUMED thread=%s layer=%d",
                    threading.get_ident(),
                    layer_idx,
                )
                logger.debug("Layer %d transfer complete", layer_idx)
                assert model_state.pending_result is None
                layer_idx += 1
