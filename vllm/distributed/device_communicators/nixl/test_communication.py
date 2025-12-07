# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for NixlCommunicationManager.
Run with: python -m pytest vllm/distributed/device_communicators/nixl/test_communication.py -v
Or directly: python vllm/distributed/device_communicators/nixl/test_communication.py
"""

import torch
import time

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:
    from nixl_cu12._api import nixl_agent, nixl_agent_config

from communication import (
    NixlCommunicationManager, NixlRequestStatus
)


def create_agent_pair():
    """Create a pair of agents (without metadata exchange - do that after registering buffers)."""
    config = nixl_agent_config(backends=["UCX"])
    agent_a = nixl_agent("test_agent_a", config)
    agent_b = nixl_agent("test_agent_b", config)
    
    comm_a = NixlCommunicationManager(agent_a)
    comm_b = NixlCommunicationManager(agent_b)
    
    return comm_a, comm_b


def register_and_sync(comm_a, comm_b, tensors_a, tensors_b):
    """Register tensors with agents and sync metadata after registration."""
    # Register memory with each agent
    reg_a = comm_a.agent.register_memory(tensors_a) if tensors_a else None
    reg_b = comm_b.agent.register_memory(tensors_b) if tensors_b else None
    
    # Sync metadata after registration
    md_a = comm_a.agent.get_agent_metadata()
    md_b = comm_b.agent.get_agent_metadata()
    comm_a.agent.add_remote_agent(md_b)
    comm_b.agent.add_remote_agent(md_a)
    
    return reg_a, reg_b


def progress_until_done(requests_a, requests_b, comm_a, comm_b, timeout=10.0):
    start = time.time()
    while time.time() - start < timeout:
        all_done = True
        for req in requests_a:
            status = comm_a.progress(req)
            if status == NixlRequestStatus.ERROR:
                raise RuntimeError(f"Request failed: {req}")
            if status != NixlRequestStatus.DONE:
                all_done = False
        for req in requests_b:
            status = comm_b.progress(req)
            if status == NixlRequestStatus.ERROR:
                raise RuntimeError(f"Request failed: {req}")
            if status != NixlRequestStatus.DONE:
                all_done = False
        if all_done:
            return
        time.sleep(0.001)
    raise TimeoutError("Transfers did not complete in time")


def test_single_transfer():
    print("test_single_transfer...")
    comm_a, comm_b = create_agent_pair()
    
    send_tensor = torch.randn(1024, device="cuda")
    recv_tensor = torch.zeros(1024, device="cuda")
    
    # Register buffers and sync metadata
    register_and_sync(comm_a, comm_b, [send_tensor], [recv_tensor])
    
    send_req = comm_a.isend(send_tensor, "test_agent_b")
    recv_req = comm_b.irecv(recv_tensor, "test_agent_a")
    
    progress_until_done([send_req], [recv_req], comm_a, comm_b)
    
    assert torch.allclose(send_tensor, recv_tensor), "Data mismatch"
    print("  PASSED")


def test_multiple_transfers_same_direction():
    print("test_multiple_transfers_same_direction (100 transfers)...")
    comm_a, comm_b = create_agent_pair()
    
    n = 100
    send_tensors = [torch.randn(512, device="cuda") for _ in range(n)]
    recv_tensors = [torch.zeros(512, device="cuda") for _ in range(n)]
    
    # Register buffers and sync metadata
    register_and_sync(comm_a, comm_b, send_tensors, recv_tensors)
    
    send_reqs = []
    recv_reqs = []
    for i in range(n):
        send_reqs.append(comm_a.isend(send_tensors[i], "test_agent_b"))
        recv_reqs.append(comm_b.irecv(recv_tensors[i], "test_agent_a"))
    
    progress_until_done(send_reqs, recv_reqs, comm_a, comm_b)
    
    for i in range(n):
        assert torch.allclose(send_tensors[i], recv_tensors[i]), f"Data mismatch at index {i}"
    print("  PASSED")


def test_bidirectional_transfers():
    print("test_bidirectional_transfers (50 each direction)...")
    comm_a, comm_b = create_agent_pair()
    
    n = 50
    a_to_b_send = [torch.randn(256, device="cuda") for _ in range(n)]
    a_to_b_recv = [torch.zeros(256, device="cuda") for _ in range(n)]
    b_to_a_send = [torch.randn(256, device="cuda") for _ in range(n)]
    b_to_a_recv = [torch.zeros(256, device="cuda") for _ in range(n)]
    
    # Register buffers and sync metadata (agent_a has send + recv, agent_b has recv + send)
    register_and_sync(comm_a, comm_b, a_to_b_send + b_to_a_recv, a_to_b_recv + b_to_a_send)
    
    reqs_a = []
    reqs_b = []
    for i in range(n):
        reqs_a.append(comm_a.isend(a_to_b_send[i], "test_agent_b"))
        reqs_b.append(comm_b.irecv(a_to_b_recv[i], "test_agent_a"))
        reqs_b.append(comm_b.isend(b_to_a_send[i], "test_agent_a"))
        reqs_a.append(comm_a.irecv(b_to_a_recv[i], "test_agent_b"))
    
    progress_until_done(reqs_a, reqs_b, comm_a, comm_b)
    
    for i in range(n):
        assert torch.allclose(a_to_b_send[i], a_to_b_recv[i]), f"A->B mismatch at {i}"
        assert torch.allclose(b_to_a_send[i], b_to_a_recv[i]), f"B->A mismatch at {i}"
    print("  PASSED")


def test_multiple_tags():
    print("test_multiple_tags...")
    comm_a, comm_b = create_agent_pair()
    
    tags = ["weights", "activations", "gradients"]
    n_per_tag = 20
    
    send_data = {tag: [torch.randn(128, device="cuda") for _ in range(n_per_tag)] for tag in tags}
    recv_data = {tag: [torch.zeros(128, device="cuda") for _ in range(n_per_tag)] for tag in tags}
    
    # Register all buffers and sync metadata
    all_send_tensors = [t for tag in tags for t in send_data[tag]]
    all_recv_tensors = [t for tag in tags for t in recv_data[tag]]
    register_and_sync(comm_a, comm_b, all_send_tensors, all_recv_tensors)
    
    send_reqs = []
    recv_reqs = []
    for tag in tags:
        for i in range(n_per_tag):
            send_reqs.append(comm_a.isend(send_data[tag][i], "test_agent_b", tag=tag))
            recv_reqs.append(comm_b.irecv(recv_data[tag][i], "test_agent_a", tag=tag))
    
    progress_until_done(send_reqs, recv_reqs, comm_a, comm_b)
    
    for tag in tags:
        for i in range(n_per_tag):
            assert torch.allclose(send_data[tag][i], recv_data[tag][i]), f"Mismatch {tag}[{i}]"
    print("  PASSED")


def test_out_of_order_recv():
    print("test_out_of_order_recv (recv submitted before send)...")
    comm_a, comm_b = create_agent_pair()
    
    send_tensor = torch.randn(1024, device="cuda")
    recv_tensor = torch.zeros(1024, device="cuda")
    
    # Register buffers and sync metadata
    register_and_sync(comm_a, comm_b, [send_tensor], [recv_tensor])
    
    recv_req = comm_b.irecv(recv_tensor, "test_agent_a")
    
    for _ in range(10):
        status = comm_b.progress(recv_req)
        assert status == NixlRequestStatus.IN_PROGRESS
        time.sleep(0.01)
    
    send_req = comm_a.isend(send_tensor, "test_agent_b")
    
    progress_until_done([send_req], [recv_req], comm_a, comm_b)
    
    assert torch.allclose(send_tensor, recv_tensor), "Data mismatch"
    print("  PASSED")


def test_large_tensor():
    print("test_large_tensor (100MB)...")
    comm_a, comm_b = create_agent_pair()
    
    size = 100 * 1024 * 1024 // 4
    send_tensor = torch.randn(size, device="cuda")
    recv_tensor = torch.zeros(size, device="cuda")
    
    # Register buffers and sync metadata
    register_and_sync(comm_a, comm_b, [send_tensor], [recv_tensor])
    
    send_req = comm_a.isend(send_tensor, "test_agent_b")
    recv_req = comm_b.irecv(recv_tensor, "test_agent_a")
    
    progress_until_done([send_req], [recv_req], comm_a, comm_b, timeout=30.0)
    
    assert torch.allclose(send_tensor, recv_tensor), "Data mismatch"
    print("  PASSED")


def test_sequence_ordering():
    print("test_sequence_ordering (interleaved sends, sequential recvs)...")
    comm_a, comm_b = create_agent_pair()
    
    n = 50
    send_tensors = [torch.full((64,), float(i), device="cuda") for i in range(n)]
    recv_tensors = [torch.zeros(64, device="cuda") for _ in range(n)]
    
    # Register buffers and sync metadata
    register_and_sync(comm_a, comm_b, send_tensors, recv_tensors)
    
    send_reqs = [comm_a.isend(send_tensors[i], "test_agent_b") for i in range(n)]
    
    start = time.time()
    while time.time() - start < 5.0:
        for req in send_reqs:
            comm_a.progress(req)
        time.sleep(0.01)
        break
    
    recv_reqs = [comm_b.irecv(recv_tensors[i], "test_agent_a") for i in range(n)]
    
    progress_until_done(send_reqs, recv_reqs, comm_a, comm_b)
    
    for i in range(n):
        expected = float(i)
        actual = recv_tensors[i][0].item()
        assert actual == expected, f"Order mismatch at {i}: expected {expected}, got {actual}"
    print("  PASSED")


def test_completion_notification_prefix_bug():
    print("test_completion_notification_prefix_bug (seq 1 vs 10 vs 100)...")
    comm_a, comm_b = create_agent_pair()
    
    send_tensors = [torch.full((32,), float(i), device="cuda") for i in range(150)]
    recv_tensors = [torch.zeros(32, device="cuda") for _ in range(150)]
    
    # Register buffers and sync metadata
    register_and_sync(comm_a, comm_b, send_tensors, recv_tensors)
    
    send_reqs = [comm_a.isend(send_tensors[i], "test_agent_b") for i in range(150)]
    recv_reqs = [comm_b.irecv(recv_tensors[i], "test_agent_a") for i in range(150)]
    
    progress_until_done(send_reqs, recv_reqs, comm_a, comm_b)
    
    for i in range(150):
        expected = float(i)
        actual = recv_tensors[i][0].item()
        assert actual == expected, f"Mismatch at {i}: expected {expected}, got {actual}"
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("NixlCommunicationManager Tests")
    print("=" * 60)
    
    test_single_transfer()
    test_multiple_transfers_same_direction()
    test_bidirectional_transfers()
    test_multiple_tags()
    test_out_of_order_recv()
    test_large_tensor()
    test_sequence_ordering()
    test_completion_notification_prefix_bug()
    
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
