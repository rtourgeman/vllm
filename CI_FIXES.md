# CI Fixes for Elastic EP Rebase

---

### Code fixes (verified locally)

| CI Check | Fix | Run Locally |
|----------|-----|-------------|
| **distributed-2-gpus** | Mutable ref pattern in `async_llm.py` to fix circular reference / memory leak | `pytest tests/v1/shutdown/test_delete.py -v -x --timeout=120` |
| **fusion-and-compile-tests-b200** | Move `initialize_model_parallel` inside `set_current_vllm_config` context | `pytest tests/compile/passes/distributed/test_fusion_all_reduce.py -v -x --timeout=300` |
| **lora-1** | Wrap `worker.init_device()` in `set_current_vllm_config` context | `pytest tests/lora/test_worker.py::test_worker_apply_lora -v -x --timeout=120` |

### Transient failures (will pass on fresh CI run)

| CI Check | Cause | Confidence |
|----------|-------|------------|
| **distributed-tests-4-gpus** | flashinfer IPC socket collision -- verified fails on upstream too | High |
| **lora-2** | HuggingFace 504 timeout (Jan 28 outage) | High |
| **v1-e2e-plus-engine** | HuggingFace 504 timeout (99/100 tests passed) | High |
| **entrypoints-integration-responses-api** | Flaky model output, test has `@pytest.mark.flaky(reruns=5)` | Medium |

---

## Fix 1: GPU Memory Leak on Shutdown

**CI:** distributed-2-gpus | **File:** `vllm/v1/engine/async_llm.py`

The elastic EP branch used `self.logger_manager` inside the `output_handler` closure, creating a circular reference (`AsyncLLM` -> task -> `self` -> `AsyncLLM`) that prevents garbage collection on shutdown.

**Fix:** Use a mutable list ref. The closure captures `logger_ref` (local variable pointing to a list), not `self`. During scaling, `self._logger_ref[0]` is updated so the closure picks up the new logger.

```python
self._logger_ref = [self.logger_manager]
logger_ref = self._logger_ref

async def output_handler():
    if logger_ref[0]:              # no ref to self
        logger_ref[0].record(...)

# During scaling:
self._logger_ref[0] = new_logger  # closure sees the update
```

---

## Fix 2: ElasticEPScalingExecutor for All Models

**CI:** distributed-2-gpus | **File:** `vllm/v1/worker/gpu_worker.py`

`ElasticEPScalingExecutor` was created unconditionally for every worker, pulling in heavy imports for non-elastic-EP models.

**Fix:** Guard behind `enable_elastic_ep`:

```python
if vllm_config.parallel_config.enable_elastic_ep:
    from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
    self.elastic_ep_executor = ElasticEPScalingExecutor(self)
else:
    self.elastic_ep_executor = None
```

---

## Fix 3: Tests Missing vllm_config Context

**CI:** lora-1, fusion-and-compile-tests-b200 | **Files:** `tests/lora/test_worker.py`, `tests/compile/passes/distributed/test_fusion_all_reduce.py`

The elastic EP branch requires `get_current_vllm_config()` to be set when calling `initialize_model_parallel` (commit `554649f` by Tyler Smith). Some tests called it without setting the config.

**Fix:** Wrap calls in `set_current_vllm_config()` context (do not revert Tyler's change):

```python
# LoRA test
with set_current_vllm_config(vllm_config):
    worker.init_device()
    worker.load_model()

# Fusion test: move initialize_model_parallel inside existing context
with set_current_vllm_config(vllm_config):
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    ...
```
