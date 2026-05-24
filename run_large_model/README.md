# DeepSeek V3 Multi-Node Benchmark

Run DeepSeek V3 with vLLM on Slurm batch nodes using expert parallelism and NIXL EP.

## Files


| File                   | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `run_bench.sh`         | Submit a static benchmark (no scale-up)                        |
| `run_bench_scaleup.sh` | Submit a scale-up benchmark                                    |
| `slurm_static.sh`      | Slurm job for static benchmark (do not run directly)           |
| `slurm_scaleup.sh`     | Slurm job for scale-up benchmark (do not run directly)         |
| `config.sh`            | Shared defaults (image, model, ports, timeouts)                |
| `env.sh`               | Container environment setup                                    |
| `helpers.sh`           | Reusable functions (IP resolution, health checks, Ray polling) |
| `serve.sh`             | Launch vLLM server                                             |
| `bench.sh`             | Run a single vLLM benchmark                                    |


## Benchmark suite (same for all runs)


| Step   | Prompts | Concurrency |
| ------ | ------- | ----------- |
| warmup | 8192    | unlimited   |
| bench1 | 8192    | 1024        |
| bench2 | 4096    | 512         |
| bench3 | 2048    | 256         |
| bench4 | 1024    | 128         |
| bench5 | 512     | 64          |


---

## Run 1 — Scale-up (32 → 40 GPUs)

```bash
./run_bench_scaleup.sh -x 32 -y 40 -r 0 -R 24 -t 02:00:00
```

Flow: allocate 5 nodes → start vLLM on 32 GPUs → scale up to 40 GPUs → warmup → bench1…bench5 → shutdown

**Options:**


| Flag | Description                           | Default  |
| ---- | ------------------------------------- | -------- |
| `-x` | Initial GPU count                     | 32       |
| `-y` | Target GPU count after scale-up       | 40       |
| `-r` | Redundant experts at initial serve    | 0        |
| `-R` | Redundant experts in scale-up request | 24       |
| `-t` | Slurm time limit                      | 02:00:00 |


---

## Run 2 — Static 32 GPUs

```bash
./run_bench.sh -N 4 -r 0 -t 02:00:00
```

Flow: allocate 4 nodes → start vLLM on 32 GPUs → warmup → bench1…bench5 → shutdown

---

## Run 3 — Static 40 GPUs

```bash
./run_bench.sh -N 5 -r 24 -t 02:00:00
```

Flow: allocate 5 nodes → start vLLM on 40 GPUs (24 redundant experts) → warmup → bench1…bench5 → shutdown

**Options for `run_bench.sh`:**


| Flag | Description            | Default  |
| ---- | ---------------------- | -------- |
| `-N` | Number of 8xH100 nodes | 4        |
| `-r` | Redundant experts      | 0        |
| `-t` | Slurm time limit       | 02:00:00 |


---

## Logs

Slurm output: `logs/<date>/<job_tag>_<job_id>.out`

Per-run artifacts: `logs/<job_id>_<description>/`

- `vllm_server.log` — vLLM server log
- `bench_<tag>_warmup_np8192.log` — warmup results
- `bench_<tag>_np<N>_c<C>.log` — individual benchmark results

## Defaults

All defaults live in `config.sh`. Override via environment variables or CLI flags.