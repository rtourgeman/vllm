#!/usr/bin/env bash
# Environment setup sourced inside the Slurm container on every node.

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/vllm/tools/ep_kernels/elastic_ep/eep_kernels_workspace/DeepEP/build/lib.linux-x86_64-cpython-310:/nixl/install/lib/python3/dist-packages:${PYTHONPATH:-}"
# export RAY_DEDUP_LOGS=0
unset UCX_NET_DEVICES

mkdir -p /data/cache
export HOME=/data
export XDG_CACHE_HOME=/data/cache

export HF_HOME=/rtourgeman/hf-cache
export HUGGINGFACE_HUB_CACHE=/rtourgeman/hf-cache/hub

mkdir -p /data/cache/uv /data/tmp
export UV_CACHE_DIR=/data/cache/uv
export TMPDIR=/data/tmp

mkdir -p /data/cache/flashinfer /root/.cache
rm -rf /root/.cache/flashinfer 2>/dev/null || true
ln -s /data/cache/flashinfer /root/.cache/flashinfer 2>/dev/null || true

export VLLM_NO_USAGE_STATS=1
