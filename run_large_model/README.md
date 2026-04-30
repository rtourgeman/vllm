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
| `head_node.sh` | Head node logic (Ray head, vLLM, scale-up, benchmark) |
| `worker_node.sh` | Worker node logic (wait for signal, join Ray) |
| `submit.sh` | sbatch submission wrapper with CLI options |

## Quick Start

```bash
cd /lustre/fsw/portfolios/network/users/rtourgeman/vllm/run_large_model

# Run benchmark on 4 nodes (32 GPUs)
./submit.sh -N 4 -n 512 -c 256 -i 1024 -o 1024

# Elastic scale-up: start at 32 GPUs, scale to 64
./submit.sh -x 32 -y 64 -n 1024 -c 256

# Skip baseline benchmark before scale-up
./submit.sh -x 32 -y 64 -B -n 1024 -c 256
```

## Token Setup

Export your Hugging Face token before submitting:

```bash
export HF_TOKEN=<your-token>
```

Or use a local model path with `-m /path/to/DeepSeek-V3`.

## Options

```bash
./submit.sh -h
```

Key options:

- `-N`: number of 8-GPU nodes
- `-x`: initial DP size for elastic scale-up
- `-y`: target DP size after scale-up
- `-n`: number of benchmark prompts
- `-c`: max benchmark concurrency
- `-i`: random input token length
- `-o`: random output token length
- `-t`: Slurm time limit
- `-C`: container image
- `-m`: model path or HF name

## Logs

Slurm output goes to `logs/<date>/<job_name>_<job_id>.out`.

Per-run artifacts (vLLM server log, benchmark results) go to `logs/<job_id>/`:

- `vllm_server.log`
- `benchmark_initial_<N>gpu.log` (if elastic)
- `benchmark_final_<N>gpu.log`

## Defaults

All defaults live in `config.sh`. Override via environment variables or CLI flags.
