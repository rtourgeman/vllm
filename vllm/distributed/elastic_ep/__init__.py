# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Elastic EP package; tiny opt-in memory trace via VLLM_KV_MEM_TRACE=1."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_last_used: dict[str, int] = {}


def kv_mem_trace_enabled() -> bool:
    return os.environ.get("VLLM_KV_MEM_TRACE", "0") in ("1", "true", "True")


def kv_mem_used_free_mib_local() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return (total - free) // (1 << 20), free // (1 << 20)


def kv_mem_trace_reset(role: str, rank: int) -> None:
    if kv_mem_trace_enabled():
        _last_used.pop(f"{role}:{rank}", None)


def kv_mem_trace(ckpt: str, *, rank: int, role: str = "existing", **fields) -> None:
    if not kv_mem_trace_enabled():
        return
    used, free = kv_mem_used_free_mib_local()
    key = f"{role}:{rank}"
    prev = _last_used.get(key)
    _last_used[key] = used
    parts = [f"rank={rank}", f"role={role}", f"ckpt={ckpt}",
             f"used={used}MiB", f"free={free}MiB"]
    if prev is not None:
        parts.append(f"delta={used - prev:+d}MiB")
    parts.extend(f"{k}={v}" for k, v in fields.items() if v is not None)
    logger.info("[KV_MEM_TRACE] %s", " ".join(parts))


def kv_mem_trace_summary(step: str, used: list[int], free: list[int]) -> None:
    if not kv_mem_trace_enabled() or not used:
        return
    logger.info(
        "[KV_MEM_TRACE_SUMMARY] step=%s gpus_active=%d min_free=%dMiB "
        "max_used=%dMiB per_gpu_used=[%s]",
        step, len(used), min(free), max(used),
        ",".join(f"{u}MiB" for u in used),
    )
