# DeepSeek V3 Multi-Node Benchmark

Run DeepSeek V3 with vLLM on Slurm batch nodes using expert parallelism and NIXL EP.

## Files

| File | Purpose |
|------|---------|
| `run_bench.sh` | Submit a static benchmark (no scale-up) |
| `run_bench_scaleup.sh` | Submit a scale-up benchmark |
| `slurm_static.sh` | Slurm job for static benchmark (do not run directly) |
| `slurm_scaleup.sh` | Slurm job for scale-up benchmark (do not run directly) |
| `config.sh` | Shared defaults (image, model, ports, timeouts) |
| `env.sh` | Container environment setup |
| `helpers.sh` | Reusable functions (IP resolution, health checks, Ray polling) |
| `serve.sh` | Launch vLLM server |
| `bench.sh` | Run vLLM benchmark |

## Static Benchmark (no scale-up)

```bash
# 32 GPUs (4 nodes), default settings
./run_bench.sh -N 4

# 40 GPUs (5 nodes), 24 redundant experts, 8192 prompts, concurrency 512
./run_bench.sh -N 5 -r 24 -n 8192 -c 512

# Custom time limit
./run_bench.sh -N 4 -n 1024 -c 128 -t 02:00:00
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-N` | Number of 8xH100 nodes | 4 |
| `-r` | Redundant experts | 0 |
| `-n` | Number of prompts | 8192 |
| `-c` | Max concurrency | 256 |
| `-t` | Slurm time limit | 01:00:00 |

**Flow:** Start Ray -> Start vLLM -> Warmup (1000 prompts) -> Real benchmark -> Shutdown

## Scale-Up Benchmark

```bash
# 32->40 GPUs, 24 redundant experts after scale
./run_bench_scaleup.sh -x 32 -y 40 -R 24 -n 8192 -c 512

# Custom initial redundant experts
./run_bench_scaleup.sh -x 32 -y 40 -r 0 -R 24 -n 1024 -c 256 -t 02:00:00
```

The script automatically computes the number of nodes needed (e.g. `-x 32 -y 40` allocates 5 nodes).

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-x` | Initial GPU count at startup | 32 |
| `-y` | Target GPU count after scale-up | 40 |
| `-r` | Initial redundant experts | 0 |
| `-R` | Redundant experts in scale-up request | 24 |
| `-n` | Number of prompts | 8192 |
| `-c` | Max concurrency | 256 |
| `-t` | Slurm time limit | 01:00:00 |

**Flow:** Start Ray (initial nodes) -> Start vLLM (x GPUs) -> Signal extra node(s) -> Scale to y GPUs -> Warmup (1000 prompts) -> Real benchmark -> Shutdown

## Logs

Slurm output: `logs/<date>/bench_*_<job_id>.out` or `logs/<date>/scale_*_<job_id>.out`

Per-run artifacts: `logs/<job_id>_<description>/`
- `vllm_server.log` -- vLLM server log
- `bench_warmup_*.log` -- warmup benchmark
- `bench_*gpu_*.log` -- real benchmark results

## Defaults

All defaults live in `config.sh`. Override via environment variables or CLI flags.
