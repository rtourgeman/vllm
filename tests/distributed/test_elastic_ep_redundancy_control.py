# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EPLB redundancy control during elastic EP scaling.

Covers:
  - Startup cap initialization
  - /set_redundant_experts (valid, invalid, unchanged, edge cases)
  - /scale_elastic_ep with and without explicit num_redundant_experts
  - Scale-down implicit fallback to full capacity
  - Repeated multi-step operation sequences
  - Error-contract checks (HTTP 400 + closest valid suggestion)

Policy notes for readers:
  - Startup config defines the initial numeric EPLB cap.
  - Scale-up without explicit redundancy preserves the existing cap.
  - Scale-up with incompatible cap is rejected (HTTP 400).
  - Scale-down with incompatible cap falls back to full capacity.
  - Explicit num_redundant_experts during scale-down is rejected.
"""

import os
import subprocess
import time

import pytest
import requests

from ..utils import RemoteOpenAIServer, multi_gpu_test

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

INITIAL_DP_SIZE = 4
SMOKE_PROMPT = "The capital of France is"
SCALE_TIMEOUT = 300
REQUEST_TIMEOUT = 60


def _build_serve_args(
    initial_dp_size: int = INITIAL_DP_SIZE,
    num_redundant: int = 0,
    all2all_backend: str = "allgather_reducescatter",
) -> list[str]:
    args = [
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "4096",
        "--max-num-seqs", "32",
        "--enable-expert-parallel",
        "--all2all-backend", all2all_backend,
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts", str(num_redundant),
        "--data-parallel-backend", "ray",
        "--data-parallel-size", str(initial_dp_size),
        "--api-server-count", "1",
    ]
    leader = os.environ.get("LEADER_ADDRESS")
    if leader:
        args.extend(["--data-parallel-address", leader])
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale(server: RemoteOpenAIServer, new_dp: int,
           num_redundant: int | None = None) -> requests.Response:
    payload: dict = {"new_data_parallel_size": new_dp}
    if num_redundant is not None:
        payload["num_redundant_experts"] = num_redundant
    return requests.post(
        server.url_for("scale_elastic_ep"),
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=SCALE_TIMEOUT,
    )


def _set_redundant(server: RemoteOpenAIServer,
                   num_redundant: int) -> requests.Response:
    return requests.post(
        server.url_for("set_redundant_experts"),
        json={"num_redundant_experts": num_redundant},
        headers={"Content-Type": "application/json"},
        timeout=SCALE_TIMEOUT,
    )


def _smoke(server: RemoteOpenAIServer, label: str) -> None:
    resp = requests.post(
        server.url_for("v1/completions"),
        json={
            "model": MODEL_NAME,
            "prompt": SMOKE_PROMPT,
            "max_tokens": 8,
            "temperature": 0.0,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"]
    print(f"[{label}] smoke: {text!r}")
    assert text and text.strip()


def _assert_ok(resp: requests.Response, label: str) -> None:
    assert resp.status_code == 200, (
        f"[{label}] expected 200 but got {resp.status_code}: "
        f"{resp.text}"
    )


def _assert_bad_request(resp: requests.Response, label: str,
                        *fragments: str) -> None:
    """Verify HTTP 400 and that every fragment appears in the response."""
    assert resp.status_code == 400, (
        f"[{label}] expected 400 but got {resp.status_code}: "
        f"{resp.text}"
    )
    body = resp.text.lower()
    for frag in fragments:
        assert frag.lower() in body, (
            f"[{label}] expected '{frag}' in response: {resp.text}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_ray():
    subprocess.run(["ray", "stop", "--force"],
                   timeout=30, capture_output=True)
    time.sleep(5)
    yield


# ---------------------------------------------------------------------------
# Tests — /set_redundant_experts
# ---------------------------------------------------------------------------

# DeepSeek-V2-Lite has 64 routed experts, 2 expert groups.
# At DP=4 EP=4: per_gpu = (64+R)/4, total = (64+R).
# For these tests we start with num_redundant=0, so per_gpu=16, total=64.

@multi_gpu_test(num_gpus=8)
class TestSetRedundantExperts:
    """Tests for /set_redundant_experts at steady-state (no scaling)."""

    def _make_server(self):
        return RemoteOpenAIServer(
            MODEL_NAME,
            _build_serve_args(initial_dp_size=4, num_redundant=0),
            env_dict={},
            max_wait_seconds=600,
        )

    def test_set_redundant_valid(self):
        """Valid redundancy: 64+0=64 total, EP=4, set to 0 (full cap)."""
        with self._make_server() as server:
            resp = _set_redundant(server, 0)
            _assert_ok(resp, "set_redundant=0")
            _smoke(server, "after set_redundant=0")

    def test_set_redundant_invalid_not_divisible(self):
        """64+3=67 not divisible by EP=4 -> rejected with suggestion."""
        with self._make_server() as server:
            resp = _set_redundant(server, 3)
            _assert_bad_request(
                resp, "set_redundant=3",
                "closest valid", "num_redundant_experts",
            )

    def test_set_redundant_invalid_exceeds_capacity(self):
        """64+100=164 > 64 total slots -> rejected."""
        with self._make_server() as server:
            resp = _set_redundant(server, 100)
            _assert_bad_request(
                resp, "set_redundant=100",
                "exceeds total slots",
            )


# ---------------------------------------------------------------------------
# Tests — /scale_elastic_ep validation
# ---------------------------------------------------------------------------

@multi_gpu_test(num_gpus=8)
class TestScaleValidation:
    """Validation-only tests: check error contracts without full scaling."""

    def _make_server(self):
        return RemoteOpenAIServer(
            MODEL_NAME,
            _build_serve_args(initial_dp_size=4, num_redundant=0),
            env_dict={},
            max_wait_seconds=600,
        )

    def test_scale_up_invalid_redundancy(self):
        """Scale to EP=8 with num_redundant=5 (64+5=69 % 8 != 0)."""
        with self._make_server() as server:
            resp = _scale(server, 8, num_redundant=5)
            _assert_bad_request(
                resp, "scale_up_invalid",
                "closest valid", "num_redundant_experts",
            )

    def test_scale_down_explicit_redundancy_rejected(self):
        """Explicit redundancy during scale-down is always rejected."""
        with self._make_server() as server:
            resp = _scale(server, 2, num_redundant=0)
            _assert_bad_request(
                resp, "scale_down_explicit",
                "only supported during scale-up",
            )


# ---------------------------------------------------------------------------
# Tests — Full scale-up / scale-down sequences
# ---------------------------------------------------------------------------

@multi_gpu_test(num_gpus=8)
class TestScaleSequences:
    """Multi-step scaling sequences that exercise state tracking."""

    def _make_server(self, num_redundant: int = 0):
        return RemoteOpenAIServer(
            MODEL_NAME,
            _build_serve_args(initial_dp_size=4, num_redundant=num_redundant),
            env_dict={},
            max_wait_seconds=600,
        )

    def test_scale_up_preserves_cap(self):
        """Scale up DP=4->8 without explicit redundancy.

        Startup cap = 64 (num_logical=64 + num_redundant=0).
        After scale-up: new_total=128, cap 64 preserved (64%8==0).
        EPLB should use 64 active experts out of 128 slots.
        """
        with self._make_server() as server:
            resp = _scale(server, 8)
            _assert_ok(resp, "scale_up_4_to_8")
            time.sleep(10)
            _smoke(server, "after_scale_up")

    def test_scale_up_with_explicit_redundancy(self):
        """Scale up DP=4->8 with num_redundant=64 (128 active, full cap)."""
        with self._make_server() as server:
            resp = _scale(server, 8, num_redundant=64)
            _assert_ok(resp, "scale_up_with_redundant_64")
            time.sleep(10)
            _smoke(server, "after_scale_up_redundant")

    def test_scale_down_preserves_compatible_cap(self):
        """Up to 8, then down to 4. Cap=64 is compatible with both."""
        with self._make_server() as server:
            resp = _scale(server, 8)
            _assert_ok(resp, "up_4_to_8")
            time.sleep(10)

            resp = _scale(server, 4)
            _assert_ok(resp, "down_8_to_4")
            time.sleep(5)
            _smoke(server, "after_down")

    def test_scale_down_falls_back_on_incompatible_cap(self):
        """Up to 8, set redundancy to 16 (cap=80), then down to 4.

        cap=80 > new_total=64, so scale-down should fall back to
        full capacity (64) instead of rejecting.
        """
        with self._make_server() as server:
            resp = _scale(server, 8)
            _assert_ok(resp, "up_4_to_8")
            time.sleep(10)

            resp = _set_redundant(server, 16)
            _assert_ok(resp, "set_redundant_16")

            resp = _scale(server, 4)
            _assert_ok(resp, "down_8_to_4_fallback")
            time.sleep(5)
            _smoke(server, "after_fallback_down")

    def test_set_redundant_after_scale_up_expanded_capacity(self):
        """After scale-up, set_redundant should use expanded capacity.

        Startup: total=64. Scale to 8: total=128.
        set_redundant(32): 64+32=96 <= 128 and 96%8==0 -> valid.
        This specifically tests the _total_physical_slots fix.
        """
        with self._make_server() as server:
            resp = _scale(server, 8)
            _assert_ok(resp, "up_4_to_8")
            time.sleep(10)

            resp = _set_redundant(server, 32)
            _assert_ok(resp, "set_redundant_32_after_scale")
            _smoke(server, "after_set_redundant")

    def test_repeated_up_down_cycle(self):
        """Repeated up/down cycles preserve correctness.

        Catches stale engine-side state across multiple operations.
        """
        with self._make_server() as server:
            for cycle in range(1, 4):
                resp = _scale(server, 8)
                _assert_ok(resp, f"cycle_{cycle}_up")
                time.sleep(10)

                resp = _scale(server, 4)
                _assert_ok(resp, f"cycle_{cycle}_down")
                time.sleep(5)

            _smoke(server, "after_3_cycles")

    def test_up_set_redundant_down_up(self):
        """Up -> set_redundant -> down -> up: full state lifecycle.

        Up to 8 (total=128, cap=64).
        Set redundant=8 (cap=72, compatible with both 8 and 4).
        Down to 4 (total=64, cap 72>64 -> fallback to 64).
        Up to 8 again (total=128, cap=64 preserved from fallback).
        """
        with self._make_server() as server:
            resp = _scale(server, 8)
            _assert_ok(resp, "up_4_to_8")
            time.sleep(10)

            resp = _set_redundant(server, 8)
            _assert_ok(resp, "set_redundant_8")

            resp = _scale(server, 4)
            _assert_ok(resp, "down_8_to_4")
            time.sleep(5)

            resp = _scale(server, 8)
            _assert_ok(resp, "up_4_to_8_again")
            time.sleep(10)
            _smoke(server, "after_full_lifecycle")
