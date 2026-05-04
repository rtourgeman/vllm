# DeepSeek V3 Multi-Node Benchmark

Run DeepSeek V3 with vLLM on Slurm batch nodes using expert parallelism and NIXL EP.

## Files

| File | Purpose |
|------|---------|
| `config.sh` | Shared defaults (image, model, ports, timeouts) |
| `helpers.sh` | Reusable functions (IP resolution, health checks, Ray polling) |
| `env.sh` | Container environment setup (sourced inside the container) |
| `serve.sh` | vLLM serve command builder |
| `bench.sh` | vLLM benchmark runner |
| `batch.slurm` | Slurm batch job orchestrator |
| `head_node.sh` | Head node logic (primary or secondary, based on ROLE) |
| `worker_node.sh` | Worker node logic (primary or secondary, based on ROLE) |
| `submit.sh` | sbatch submission wrapper with CLI options |

## Quick Start

```bash
cd /lustre/fsw/portfolios/network/users/rtourgeman/vllm/run_large_model

# Run benchmark on 4 nodes (32 GPUs)
./submit.sh -N 4 -n 512 -c 256 -i 1024 -o 1024

# Elastic scale-up: 32 -> 64 GPUs (dual vLLM, zero idle GPUs)
./submit.sh -x 32 -y 64 -n 1024 -c 256

# Skip baseline benchmark before scale-up
./submit.sh -x 32 -y 64 -B -n 1024 -c 256
```

## Elastic Scale-Up Flow

When `-x` < `-y`, the job runs two independent vLLM instances:
- Nodes 1-4: primary Ray cluster + vLLM (full benchmark)
- Nodes 5-8: secondary Ray cluster + vLLM (small benchmark)

After both benchmarks complete, nodes 5-8 tear down their vLLM, join the primary Ray cluster, and scale.py scales to the full GPU count. No idle GPUs at any point.

## Options

```bash
./submit.sh -h
```

## Logs

Slurm output: `logs/<date>/<job_name>_<job_id>.out`

Per-run artifacts: `logs/<job_id>/`:
- `vllm_server.log` -- primary vLLM server
- `vllm_server_B.log` -- secondary vLLM server (elastic runs)
- `benchmark_initial_<N>gpu.log` -- primary baseline benchmark
- `benchmark_secondary_<N>gpu.log` -- secondary benchmark
- `benchmark_final_<N>gpu.log` -- post-scale-up benchmark
- `ray_logs/` -- Ray worker logs (for debugging crashes)

## Defaults

All defaults live in `config.sh`. Override via environment variables or CLI flags.
