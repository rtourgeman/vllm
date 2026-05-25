# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import time
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: "Worker", *, skip_flashinfer_autotune: bool = False):
    total_start = time.perf_counter()
    timings: dict[str, str] = {}

    # Deep GEMM warmup
    stage_start = time.perf_counter()
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)
        timings["deep_gemm"] = f"{(time.perf_counter() - stage_start) * 1000:.2f}ms"
    else:
        timings["deep_gemm"] = "skipped"

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    stage_start = time.perf_counter()
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
        timings["flashinfer_autotune"] = "disabled"
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(
            worker.model_runner,
            autotune=not skip_flashinfer_autotune,
        )
        elapsed = (time.perf_counter() - stage_start) * 1000
        if skip_flashinfer_autotune:
            timings["flashinfer_autotune"] = f"dummy_only={elapsed:.2f}ms"
        else:
            timings["flashinfer_autotune"] = f"{elapsed:.2f}ms"
    else:
        timings["flashinfer_autotune"] = "skipped"

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    stage_start = time.perf_counter()
    do_flashinfer_attention_warmup = (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    )
    if do_flashinfer_attention_warmup:
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
        timings["flashinfer_attention"] = (
            f"{(time.perf_counter() - stage_start) * 1000:.2f}ms"
        )
    else:
        timings["flashinfer_attention"] = "skipped"

    total_ms = (time.perf_counter() - total_start) * 1000
    logger.info(
        "Kernel warmup timing: deep_gemm=%s, flashinfer_autotune=%s, "
        "flashinfer_attention=%s, total=%.2fms",
        timings["deep_gemm"],
        timings["flashinfer_autotune"],
        timings["flashinfer_attention"],
        total_ms,
    )


def flashinfer_autotune(
    runner: "GPUModelRunner", *, autotune: bool = True
) -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    if not autotune:
        # Keep the same dummy run/collective ordering without entering
        # FlashInfer's expensive autotune benchmark context.
        num_tokens = runner.scheduler_config.max_num_batched_tokens
        logger.info(
            "Starting FlashInfer dummy-only warmup run with %d tokens.",
            num_tokens,
        )
        dummy_start = time.perf_counter()
        with torch.inference_mode():
            runner._dummy_run(
                num_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        logger.info(
            "Finished FlashInfer dummy-only warmup run in %.2fms.",
            (time.perf_counter() - dummy_start) * 1000,
        )
        return

    import vllm.utils.flashinfer as fi_utils

    with torch.inference_mode(), fi_utils.autotune():
        # Certain FlashInfer kernels (e.g. nvfp4 routed moe) are
        # incompatible with autotuning. This state is used to skip
        # those kernels during the autotuning process.
        fi_utils._is_fi_autotuning = True
        try:
            # We skip EPLB here since we don't want to record dummy metrics
            # When autotuning with number of tokens m, flashinfer will autotune
            # operations for all number of tokens up to m.
            # So we only need to run with the max number of tokens.
            num_tokens = runner.scheduler_config.max_num_batched_tokens
            logger.info(
                "Starting FlashInfer autotune warmup run with %d tokens.",
                num_tokens,
            )
            dummy_start = time.perf_counter()
            runner._dummy_run(
                num_tokens,
                skip_eplb=True,
                is_profile=True,
            )
            logger.info(
                "Finished FlashInfer autotune warmup run in %.2fms.",
                (time.perf_counter() - dummy_start) * 1000,
            )
        finally:
            fi_utils._is_fi_autotuning = False
