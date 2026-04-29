# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import time

import pytest
import requests

from ..evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from ..utils import RemoteOpenAIServer, multi_gpu_test


@pytest.fixture(autouse=True)
def cleanup_ray_between_tests():
    """Force-stop any lingering Ray processes between tests."""
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

NUM_GSM8K_QUESTIONS = 256
EXPECTED_ACCURACY = 0.58
ACCURACY_TOL = 0.08
MAX_NUM_SEQS = 32

INITIAL_DP_SIZE = 2
SCALE_UP_DP_SIZE = 4
NUM_SCALE_CYCLES = 1

SMOKE_PROMPT = "The capital of France is"


def _build_vllm_serve_args(all2all_backend: str) -> list[str]:
    vllm_serve_args = [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.8",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--enable-expert-parallel",
        "--all2all-backend",
        all2all_backend,
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        "0",
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        str(INITIAL_DP_SIZE),
        "--api-server-count",
        "1",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    return vllm_serve_args


def _send_scale_command(server: RemoteOpenAIServer, new_dp_size: int) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _run_gsm8k_eval(server: RemoteOpenAIServer, stage: str) -> float:
    assert server.port is not None
    result = evaluate_gsm8k(
        num_questions=NUM_GSM8K_QUESTIONS,
        host=f"http://{server.host}",
        port=server.port,
    )
    accuracy = result["accuracy"]
    print(
        f"[{stage}] GSM8K accuracy: {accuracy:.3f} "
        f"({result['num_questions']} questions)"
    )
    assert accuracy >= EXPECTED_ACCURACY, (
        f"[{stage}] GSM8K accuracy {accuracy:.3f} is below "
        f"expected threshold {EXPECTED_ACCURACY}"
    )
    return accuracy


def _run_smoke_completion(server: RemoteOpenAIServer, stage: str) -> None:
    response = requests.post(
        server.url_for("v1/completions"),
        json={
            "model": MODEL_NAME,
            "prompt": SMOKE_PROMPT,
            "max_tokens": 8,
            "temperature": 0.0,
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    text = result["choices"][0]["text"]
    print(f"[{stage}] Smoke completion: {text!r}")
    assert text is not None and text.strip()


def _wait_for_scale(server: RemoteOpenAIServer, stage: str, wait_seconds: int) -> None:
    time.sleep(wait_seconds)
    _run_smoke_completion(server, stage)


def _run_repeated_scaling_test(all2all_backend: str) -> None:
    vllm_serve_args = _build_vllm_serve_args(all2all_backend)

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(
            server,
            f"Initial ({INITIAL_DP_SIZE} GPUs, {all2all_backend})",
        )

        for cycle in range(1, NUM_SCALE_CYCLES + 1):
            assert _send_scale_command(server, SCALE_UP_DP_SIZE), (
                f"[Cycle {cycle}] failed to scale up to {SCALE_UP_DP_SIZE}"
            )
            _wait_for_scale(
                server,
                f"Cycle {cycle} after scale up "
                f"({SCALE_UP_DP_SIZE} GPUs, {all2all_backend})",
                wait_seconds=10,
            )

            assert _send_scale_command(server, INITIAL_DP_SIZE), (
                f"[Cycle {cycle}] failed to scale down to {INITIAL_DP_SIZE}"
            )
            _wait_for_scale(
                server,
                f"Cycle {cycle} after scale down "
                f"({INITIAL_DP_SIZE} GPUs, {all2all_backend})",
                wait_seconds=5,
            )

        time.sleep(10)
        final_accuracy = _run_gsm8k_eval(
            server,
            f"Final after {NUM_SCALE_CYCLES} cycles "
            f"({INITIAL_DP_SIZE} GPUs, {all2all_backend})",
        )

        assert final_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Final accuracy {final_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        print("\nAccuracy Summary (Repeated Scaling):")
        print(f"  Backend:    {all2all_backend}")
        print(f"  Initial:    {initial_accuracy:.3f}")
        print(f"  Final:      {final_accuracy:.3f}")
        print(f"  Diff:       {final_accuracy - initial_accuracy:+.3f}")
        print(f"  Cycles:     {NUM_SCALE_CYCLES}")
        print(f"  Tolerance:  {ACCURACY_TOL:.3f}")


#@multi_gpu_test(num_gpus=4)
#def test_elastic_ep_repeated_scaling_allgather_reducescatter():
#    _run_repeated_scaling_test("allgather_reducescatter")


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_repeated_scaling_nixl_ep():
    _run_repeated_scaling_test("nixl_ep")


