# vLLM MoE + EPLB Flow Documentation

This document explains how **Mixture of Experts (MoE)** layers and the **Expert Parallelism Load Balancer (EPLB)** work in vLLM.

**Main focus:**
- What data flows through the MoE layer
- How EPLB hooks into the MoE layer to collect load statistics
- When and how EPLB rebalances experts across GPUs

---

## Table of Contents

1. [Key Concepts](#key-concepts) - Hidden states, requests vs tokens
2. [MoE Layer Deep Dive](#moe-layer-deep-dive) - Router, top-K, expert computation
3. [MoE ↔ EPLB Interaction Points](#moe--eplb-interaction-points) - **Where they call each other**
4. [EPLB Deep Dive](#eplb-deep-dive) - Data structures, load collection, rearrangement
5. [End-to-End Flow](#end-to-end-flow-request--token--moe--eplb) - Request → Token → MoE → EPLB
6. [Your Deployment Architecture](#your-deployment-architecture) - Ray DP, DPEngineCoreActor
7. [Reference](#reference) - Key files, glossary

---

## Key Concepts

### What are Hidden States?

Hidden states are the **intermediate numerical representations** of tokens as they flow through the transformer model.

```
User prompt: "Hello world"
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  TOKENIZER                                                       │
│    "Hello world" → [15496, 995]  (token IDs)                     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  EMBEDDING LAYER                                                 │
│    [15496, 995] → tensor of shape [2, 2560]                      │
│                                                                  │
│    Token "Hello" → [0.12, -0.45, 0.89, ..., 0.33]  (2560 floats) │
│    Token "world" → [0.56, 0.23, -0.11, ..., 0.78]  (2560 floats) │
│                                                                  │
│    This tensor IS the "hidden states"                            │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  TRANSFORMER LAYERS (repeat N times)                             │
│    hidden_states → Attention → MoE/MLP → hidden_states           │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT HEAD                                                     │
│    hidden_states → logits → next token probabilities             │
└──────────────────────────────────────────────────────────────────┘
```

**Key point:** Hidden states are learned numerical vectors (2560 floats each for DeepSeek-V3-Lite). Each dimension encodes abstract features learned during training.

### Requests vs Tokens vs Batches

| Concept | Unit | Where it exists |
|---------|------|-----------------|
| **Request** | One user prompt + generation | API server, scheduler |
| **Batch** | Multiple requests grouped together | Scheduler output → model forward |
| **Token** | Single position in sequence (one row in hidden_states) | Model forward, MoE layer |

**Critical insight:**
- The **scheduler** operates on **requests** (add, schedule, finish)
- The **MoE layer** operates on **tokens** (hidden_states tensor)
- **EPLB** collects **per-expert token counts**, NOT request counts

---

## MoE Layer Deep Dive

### What is MoE?

MoE (Mixture of Experts) layers use **sparse activation** - each token is processed by only a subset of "experts" (small FFN networks), not all of them.

**Why MoE?** Models like DeepSeek-V3 have huge capacity (671B parameters) but only activate ~37B per token. This gives:
- **Large capacity** (many experts store knowledge)
- **Low compute** (only top-K experts run per token)

```
                    ┌─────────────────┐
                    │   Input Token   │
                    │  hidden_states  │  ← Shape: [1, hidden_size]
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Router      │  ← Linear layer: hidden_size → num_experts
                    │  (Gating Net)   │     Output: [1, 72] logits (for 72 experts)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Top-K Select  │  ← softmax → pick 6 highest scores
                    └────────┬────────┘     Output: topk_ids=[12, 5, 0, 33, 71, 8]
                             │                       topk_weights=[0.2, 0.18, ...]
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Expert 12  │   │  Expert 5   │   │  Expert 0   │   ... (6 experts total)
    │   (FFN)     │   │   (FFN)     │   │   (FFN)     │   Each: 2560 → 10240 → 2560
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           │ out_12          │ out_5           │ out_0
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Weighted Sum   │  ← output = 0.2*out_12 + 0.18*out_5 + ...
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Output Token   │  ← Shape: [1, hidden_size]
                    └─────────────────┘
```

### DeepSeek-V3-Lite MoE Specs

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_logical_experts` | 72 | Original model design |
| `num_redundant_experts` | 16 | Extra copies for load balancing |
| `num_physical_experts` | 88 | Total = 72 + 16 |
| `top_k` | 6 | Each token activates 6 experts |
| `num_moe_layers` | 29 | Layers 1-29 have MoE (layer 0 is dense) |
| `hidden_size` | 2560 | Embedding dimension |
| `intermediate_size` | 10240 | FFN hidden dimension |
| `ep_size` | 4 | 4 GPUs for Expert Parallelism |
| `experts_per_gpu` | 22 | 88 / 4 = 22 physical experts per GPU |

### Complete Call Path: HTTP Request → MoE Layer

The following flow has been **verified with actual trace logs**. Each `[REQ_FLOW_*]` marker corresponds to a print statement in the code.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  1. HTTP REQUEST (API Server Process)                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Client sends: POST /v1/completions {"prompt": "Hello", "max_tokens": 10}           │
│                                                                                     │
│  [REQ_FLOW_01] HTTP /v1/completions received                                        │
│      │                                                                              │
│      └── api_server.py: create_completion()                                         │
│              │                                                                      │
│  [REQ_FLOW_02] OpenAIServingCompletion.create_completion()                          │
│              │                                                                      │
│  [REQ_FLOW_03a] InputProcessor.process_inputs()                                     │
│              │                                                                      │
│  [REQ_FLOW_03] AsyncLLM.generate()                                                  │
│              │                                                                      │
│  [REQ_FLOW_04b] DPLBAsyncMPClient sending request via ZMQ                           │
│              │                                                                      │
│  [REQ_FLOW_04] Request added to EngineCore                                          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ ZMQ socket (routes to dp_rank)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  2. ENGINE CORE (DPEngineCoreActor - Ray Actor, one per GPU)                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_05] EngineCore received ADD request via ZMQ                              │
│      │                                                                              │
│  [REQ_FLOW_06] EngineCore dispatching ADD to scheduler                              │
│      │                                                                              │
│  [REQ_FLOW_07] Scheduler.add_request() | num_tokens=6                               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ EngineCore main loop
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  3. ENGINE CORE STEP (runs continuously)                                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_10] EngineCore.step() - calling scheduler.schedule()                     │
│      │                                                                              │
│  [REQ_FLOW_08] Scheduler.schedule() | waiting=1 | running=0                         │
│      │                                                                              │
│  [REQ_FLOW_09] Scheduler batch ready | num_new_reqs=1 | total_tokens=6              │
│      │                                                                              │
│  [REQ_FLOW_11] EngineCore.step() - executing model | total_tokens=6                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ UniProcExecutor (in-process)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  4. MODEL EXECUTION (Worker + GPUModelRunner)                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_13] Worker.execute_model() | rank=0 | total_tokens=6                     │
│      │                                                                              │
│  [REQ_FLOW_14] GPUModelRunner.execute_model() | total_tokens=6                      │
│      │                                                                              │
│      ├── _prepare_inputs()                    # Build input tensors                 │
│      │                                                                              │
│  [REQ_FLOW_15] GPUModelRunner._model_forward() starting | num_tokens=8              │
│      │                                       (padded from 6 to 8 for alignment)     │
│      │                                                                              │
│  [MODEL_FLOW_01] DeepseekV2ForCausalLM.forward() | input_ids.shape=[8]              │
│      │                                                                              │
│      └── (29 transformer layers with MoE)                                           │
│                                                                                     │
│  [MODEL_FLOW_04] DeepseekV2ForCausalLM.forward() complete | hidden_states=[8, 2560] │
│      │                                                                              │
│  [REQ_FLOW_16] GPUModelRunner._model_forward() complete                             │
│      │                                                                              │
│  [REQ_FLOW_16a] GPUModelRunner.sample_tokens() starting                             │
│      │                                                                              │
│  [REQ_FLOW_17] Scheduler.update_from_output() | num_requests=1                      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ DECODE LOOP (repeats for each token)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  5. DECODE ITERATIONS (10 times for max_tokens=10)                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_10] EngineCore.step() - calling scheduler.schedule()                     │
│  [REQ_FLOW_08] Scheduler.schedule() | waiting=0 | running=1                         │
│  [REQ_FLOW_09] Scheduler batch ready | num_cached_reqs=1 | total_tokens=1           │
│  [REQ_FLOW_11] EngineCore.step() - executing model | total_tokens=1                 │
│  [REQ_FLOW_13] Worker.execute_model() | total_tokens=1                              │
│  [REQ_FLOW_14] GPUModelRunner.execute_model() | total_tokens=1                      │
│  [REQ_FLOW_15] GPUModelRunner._model_forward() starting | num_tokens=1              │
│  [REQ_FLOW_16] GPUModelRunner._model_forward() complete                             │
│  [REQ_FLOW_16a] GPUModelRunner.sample_tokens() starting                             │
│  [REQ_FLOW_17] Scheduler.update_from_output() | num_requests=1                      │
│                                                                                     │
│          ... (repeats 10 times until max_tokens reached) ...                        │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ Output streaming (parallel)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  6. OUTPUT PROCESSING (back to API Server)                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_18] AsyncLLM output_handler received | num_outputs=1                     │
│      │                                                                              │
│  [REQ_FLOW_18a] OutputProcessor.process_outputs() | finished=0   (multiple times)   │
│      │                                                                              │
│  [REQ_FLOW_18a] OutputProcessor.process_outputs() | finished=1   (final)            │
│      │                                                                              │
│  [REQ_FLOW_19] Request complete | request_id=cmpl-xxx                               │
│      │                                                                              │
│  HTTP 200 OK → Client receives response                                             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ Background (periodic)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  7. EPLB LOAD BALANCING (background, every 100 steps for logging)                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [EPLB_FLOW_01] EplbState.step() | rearrangement_step=X/3000 | window_step=Y/1000   │
│      │                                                                              │
│      └── (Every 3000 steps triggers rearrangement: EPLB_FLOW_02, 03, 04)            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Verified Flow Trace Summary

Based on actual server logs from a real request:

```
API SERVER:
  [REQ_FLOW_01] HTTP /v1/completions received
  [REQ_FLOW_02] OpenAIServingCompletion.create_completion()
  [REQ_FLOW_03a] InputProcessor.process_inputs()
  [REQ_FLOW_03] AsyncLLM.generate()
  [REQ_FLOW_04b] DPLBAsyncMPClient sending request via ZMQ
  [REQ_FLOW_04] Request added to EngineCore

ENGINE CORE (DPEngineCoreActor):
  [REQ_FLOW_05] EngineCore received ADD request via ZMQ
  [REQ_FLOW_06] EngineCore dispatching ADD to scheduler
  [REQ_FLOW_07] Scheduler.add_request()

PREFILL (first forward pass):
  [REQ_FLOW_10] EngineCore.step() - calling scheduler.schedule()
  [REQ_FLOW_08] Scheduler.schedule() | waiting=1, running=0
  [REQ_FLOW_09] Scheduler batch ready | num_new_reqs=1
  [REQ_FLOW_11] EngineCore.step() - executing model
  [REQ_FLOW_13] Worker.execute_model()
  [REQ_FLOW_14] GPUModelRunner.execute_model()
  [REQ_FLOW_15] GPUModelRunner._model_forward() starting
  [MODEL_FLOW_01] DeepseekV2ForCausalLM.forward()
  [MODEL_FLOW_04] DeepseekV2ForCausalLM.forward() complete
  [REQ_FLOW_16] GPUModelRunner._model_forward() complete
  [REQ_FLOW_16a] GPUModelRunner.sample_tokens() starting
  [REQ_FLOW_17] Scheduler.update_from_output()

DECODE (repeats for each generated token):
  [REQ_FLOW_10] → [REQ_FLOW_08] → [REQ_FLOW_09] → [REQ_FLOW_11]
  [REQ_FLOW_13] → [REQ_FLOW_14] → [REQ_FLOW_15] → [REQ_FLOW_16]
  [REQ_FLOW_16a] → [REQ_FLOW_17]
  (... repeats N times for max_tokens ...)

OUTPUT STREAMING:
  [REQ_FLOW_18] AsyncLLM output_handler received
  [REQ_FLOW_18a] OutputProcessor.process_outputs() | finished=0
  (... multiple times ...)
  [REQ_FLOW_18a] OutputProcessor.process_outputs() | finished=1
  [REQ_FLOW_19] Request complete

BACKGROUND:
  [EPLB_FLOW_01] EplbState.step() (periodic logging)
```

**Summary of the path:**
```
HTTP Request → FastAPI → AsyncLLM → ZMQ → EngineCore → Scheduler → Worker → Model → Layer → MoE
```

### What the MoE Layer Receives

The MoE layer receives a **tensor**, not individual requests:

```python
# Input to FusedMoE.forward()
hidden_states: torch.Tensor  # shape: [num_tokens, hidden_size]
```

**Prefill vs Decode - how the shape changes:**

```
PREFILL STAGE (processing the prompt):
─────────────────────────────────────────────────────────────────────────
  Your prompt: "Hello, how are you today?" (7 tokens)
  
  hidden_states shape: [7, 2560]
                        │
                        └── 7 tokens from your prompt
  
  If prompt is longer (e.g., 1024 tokens):
  hidden_states shape: [1024, 2560]


DECODE STAGE (generating tokens one by one):
─────────────────────────────────────────────────────────────────────────
  Generating the response, one token at a time
  
  hidden_states shape: [1, 2560]    # Just 1 new token per request
                        │
                        └── Only the NEW token being generated
  
  If 5 requests are running simultaneously:
  hidden_states shape: [5, 2560]    # 5 requests × 1 token each = 5 tokens
```

**Summary:**

| Stage | What's happening | hidden_states shape | Example |
|-------|------------------|---------------------|---------|
| **Prefill** | Processing your prompt | `[prompt_length, 2560]` | `[1024, 2560]` for 1024-token prompt |
| **Decode** | Generating response | `[num_requests, 2560]` | `[1, 2560]` for 1 request |

So you're correct:
- **Prefill**: `[1024, 2560]` if your prompt has 1024 tokens
- **Decode**: `[1, 2560]` for a single request (1 token generated per step)

### What if Prompt > Batch Size? (Chunked Prefill)

When the prompt is larger than `max_num_batched_tokens` (batch size), vLLM splits the prefill into multiple chunks:

```
Example: prompt = 1024 tokens, batch_size = 256
─────────────────────────────────────────────────────────────────────────
  
  Chunk 1: hidden_states = [256, 2560]   # tokens 0-255
  Chunk 2: hidden_states = [256, 2560]   # tokens 256-511
  Chunk 3: hidden_states = [256, 2560]   # tokens 512-767
  Chunk 4: hidden_states = [256, 2560]   # tokens 768-1023
  
  Total: 4 prefill steps to process the prompt


Example: prompt = 1600 tokens, batch_size = 256
─────────────────────────────────────────────────────────────────────────
  
  Chunk 1: hidden_states = [256, 2560]   # tokens 0-255
  ...
  Chunk 6: hidden_states = [256, 2560]   # tokens 1280-1535
  Chunk 7: hidden_states = [64, 2560]    # tokens 1536-1599 (remainder)
  
  Total: 7 prefill steps (6 full + 1 partial)
```

**Why chunk?**
- GPU memory is limited
- Allows mixing prefill chunks with decode tokens from other requests
- Enables better scheduling and continuous batching

**Formula:**
```
num_prefill_chunks = ceil(prompt_length / batch_size)
```

| Prompt Length | Batch Size | Chunks | Shape per Chunk |
|---------------|------------|--------|-----------------|
| 1024 | 256 | 4 | `[256, 2560]` |
| 2048 | 256 | 8 | `[256, 2560]` |
| 1600 | 256 | 7 | `[256, 2560]` (last: `[64, 2560]`) |
| 100 | 256 | 1 | `[100, 2560]` (smaller than batch) |

**The MoE layer does NOT know:**
- Which tokens belong to which request
- Request boundaries
- Request IDs

### MoE Layer Internal Flow (with EPLB)

**File:** `vllm/model_executor/layers/fused_moe/layer.py` → `FusedMoE.forward_impl()`

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    FusedMoE.forward_impl(hidden_states, router_logits)               │
│                                                                                      │
│  INPUT:                                                                              │
│    hidden_states: torch.Tensor [num_tokens, 2560]                                    │
│    (Example: 1024 tokens in batch, each is 2560-dim vector)                          │
│                                                                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  STEP 1: ROUTER (Gating Network)                                                     │
│  ─────────────────────────────────                                                   │
│  Code: router_logits = self.gate(hidden_states)                                      │
│                                                                                      │
│    self.gate: nn.Linear(2560, 72)  # hidden_size → num_logical_experts               │
│                                                                                      │
│    router_logits: [1024, 72]  # score for each expert for each token                 │
│                                                                                      │
│    Example (token 0):                                                                │
│      [2.1, -0.5, 1.8, 3.2, ..., 0.9]  # 72 scores, higher = more relevant            │
│        │     │    │    │                                                             │
│       exp0  exp1 exp2 exp3 ...                                                       │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  STEP 2: TOP-K SELECTION                                                             │
│  ────────────────────────────                                                        │
│  Code: topk_weights, topk_ids = select_experts(router_logits, top_k=6)               │
│                                                                                      │
│  >>> [MOE_FLOW_01] logged here (first call per layer)                                │
│                                                                                      │
│    topk_ids: [1024, 6]      # 6 LOGICAL expert IDs per token                         │
│    topk_weights: [1024, 6]  # normalized weights (sum to ~1 per token)               │
│                                                                                      │
│    Example (token 0):                                                                │
│      topk_ids[0]     = [33, 12, 5, 71, 8, 0]     # best 6 experts                    │
│      topk_weights[0] = [0.21, 0.19, 0.17, 0.16, 0.14, 0.13]                          │
│                                                                                      │
│    Total routing decisions this forward pass: 1024 × 6 = 6144                        │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  STEP 3: EPLB MAPPING (if enable_eplb=True)                                          │
│  ────────────────────────────────────────────────                                    │
│  Code: topk_ids = eplb_map_to_physical_and_record(                                   │
│            topk_ids,                    # [1024, 6] logical IDs                      │
│            expert_load_view,            # [88] accumulator for this layer            │
│            logical_to_physical_map,     # [72, max_replicas] mapping table           │
│            logical_replica_count,       # [72] how many copies of each logical       │
│        )                                                                             │
│                                                                                      │
│  >>> [MOE_FLOW_03] logged here (first call per layer)                                │
│                                                                                      │
│  WHAT HAPPENS INSIDE:                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │  1. Map logical → physical expert ID                                           │  │
│  │     logical_expert=33 might map to physical_expert=45 (or 78 if replicated)    │  │
│  │     Selection is pseudo-random: (token_position % replica_count)               │  │
│  │                                                                                │  │
│  │  2. Record load statistics                                                     │  │
│  │     expert_load_view.scatter_add_(index=topk_ids.flatten(),                    │  │
│  │                                   src=ones_like(topk_ids))                     │  │
│  │                                                                                │  │
│  │     This adds +1 to expert_load_view[physical_expert_id] for each routing      │  │
│  │     decision. After this forward pass:                                         │  │
│  │       expert_load_view = [143, 267, 89, 312, ...]  # 88 values                 │  │
│  │                           │     │    │    │                                    │  │
│  │                          exp0  exp1 exp2 exp3 ...                              │  │
│  │                          got   got  got  got                                   │  │
│  │                          143   267  89   312 routing decisions                 │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  topk_ids now contains PHYSICAL expert IDs (0-87 instead of 0-71)                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  STEP 4: EXPERT PARALLEL DISPATCH (all-to-all)                                       │
│  ───────────────────────────────────────────────                                     │
│  Code: dispatched = ep_dispatch(hidden_states, topk_ids)                             │
│                                                                                      │
│    Each GPU has 22 physical experts. Tokens need to go to the GPU that has           │
│    their required expert.                                                            │
│                                                                                      │
│    GPU 0: experts 0-21                                                               │
│    GPU 1: experts 22-43                                                              │
│    GPU 2: experts 44-65                                                              │
│    GPU 3: experts 66-87                                                              │
│                                                                                      │
│    If token 0 needs expert 45, it gets sent to GPU 2 (which owns 44-65)              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  STEP 5: EXPERT COMPUTATION (local FFN)                                              │
│  ────────────────────────────────────────                                            │
│  Code: expert_out = expert_ffn(dispatched_tokens)                                    │
│                                                                                      │
│    Each GPU runs its local experts on the tokens it received:                        │
│      FFN: Linear(2560→10240) → SiLU → Linear(10240→2560)                             │
│                                                                                      │
│    This is the actual "expert work" - the rest is routing overhead                   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  STEP 6: EXPERT PARALLEL COMBINE (all-to-all back)                                   │
│  ─────────────────────────────────────────────────                                   │
│  Code: output = ep_combine(expert_out, topk_weights)                                 │
│                                                                                      │
│    Gather expert outputs back to original GPU, weighted sum:                         │
│      output[token_i] = Σ (topk_weights[i,k] × expert_output[i,k])                    │
│                                                                                      │
│  >>> [MOE_FLOW_02] logged here (first call per layer)                                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  OUTPUT:                                                                             │
│    hidden_states: torch.Tensor [1024, 2560]  # same shape as input                   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### The Key Tensors in MoE + EPLB

| Tensor | Shape | Owner | Purpose |
|--------|-------|-------|---------|
| `hidden_states` | `[num_tokens, 2560]` | Model forward | Token representations |
| `router_logits` | `[num_tokens, 72]` | Router (gate) | Expert scores per token |
| `topk_ids` | `[num_tokens, 6]` | select_experts() | Selected expert IDs |
| `topk_weights` | `[num_tokens, 6]` | select_experts() | Routing weights |
| `expert_load_view` | `[88]` | EPLB (per layer) | Token counts per physical expert |
| `logical_to_physical_map` | `[72, max_replicas]` | EPLB | Logical→Physical mapping |
| `logical_replica_count` | `[72]` | EPLB | How many copies of each logical expert |

---

## MoE ↔ EPLB Interaction Points

This section shows **exactly where** MoE and EPLB call each other.

### Overview: Two Interaction Points

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        MoE ↔ EPLB INTERACTION DIAGRAM                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  INTERACTION POINT 1: MoE CALLS EPLB (during forward pass)                              │
│  ═══════════════════════════════════════════════════════                                │
│                                                                                         │
│  FusedMoE.forward_impl()                                                                │
│       │                                                                                 │
│       ├── select_experts()           # Pure MoE: pick top-K experts                     │
│       │                                                                                 │
│       └── eplb_map_to_physical_and_record()   <══ EPLB CALLED HERE                      │
│               │                                                                         │
│               ├── Maps logical → physical expert IDs (uses EPLB's mapping tables)       │
│               └── Records load statistics (writes to EPLB's expert_load_view)           │
│                                                                                         │
│  WHERE: vllm/model_executor/layers/fused_moe/layer.py:1638                              │
│  WHEN:  Every forward pass, for every MoE layer (29 times per step)                     │
│  WHAT:  MoE gives routing decisions to EPLB, EPLB returns physical expert IDs           │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  INTERACTION POINT 2: EPLB CALLS MoE (during rearrangement)                             │
│  ════════════════════════════════════════════════════════                               │
│                                                                                         │
│  EplbState.rearrange()                                                                  │
│       │                                                                                 │
│       ├── Policy computes new expert placement                                          │
│       │                                                                                 │
│       └── rearrange_expert_weights_inplace()   <══ EPLB MODIFIES MoE WEIGHTS            │
│               │                                                                         │
│               ├── Transfers expert weight tensors between GPUs                          │
│               └── Updates MoE layer's routing tables (logical_to_physical_map)          │
│                                                                                         │
│  WHERE: vllm/distributed/eplb/rebalance_execute.py                                      │
│  WHEN:  Every 3000 steps (step_interval)                                                │
│  WHAT:  EPLB moves expert weights and updates MoE's mapping tables                      │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Interaction Point 1: MoE → EPLB (Load Collection)

**When:** Every forward pass, inside `FusedMoE.forward_impl()`

**Key insight:** The EPLB call is just a **function call** in the middle of MoE's forward pass. After it returns, execution continues in the MoE layer.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FusedMoE.forward_impl()    <-- ENTIRE FUNCTION IS IN MOE LAYER             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  # Step 1-2: Pure MoE logic                                                 │
│  router_logits = self.gate(hidden_states)      # Router scores              │
│  topk_ids = select_experts(router_logits)      # Pick top-K (LOGICAL IDs)   │
│                                                                             │
│  # Step 3: Call into EPLB (just a function call, returns immediately)       │
│  if self.enable_eplb:                                                       │
│      ┌─────────────────────────────────────────────────────────────────┐    │
│      │  topk_ids = eplb_map_to_physical_and_record(...)                │    │
│      │                                                                 │    │
│      │  Inside this function (in fused_moe.py):                        │    │
│      │    1. Map logical → physical IDs                                │    │
│      │    2. expert_load_view.scatter_add_(...) # Record load          │    │
│      │    3. return topk_ids  # <-- RETURNS BACK TO MOE                │    │
│      └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  # Step 4-6: BACK IN MOE LAYER, continue with physical expert IDs           │
│  dispatched = ep_dispatch(hidden_states, topk_ids)   # Send to GPUs         │
│  expert_out = expert_ffn(dispatched)                 # Compute              │
│  output = ep_combine(expert_out, topk_weights)       # Gather back          │
│                                                                             │
│  return output   # <-- MoE forward complete                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Flow summary:**
```
MoE layer ──call──> EPLB function ──return──> MoE layer continues
   │                     │                         │
   │                     │                         ├── ep_dispatch
   │                     │                         ├── expert_ffn  
   │                     │                         └── ep_combine
   │                     │
   │                     └── Just updates EPLB's tensors, then returns
   │
   └── Owns the entire forward pass
```

**What gets passed:**

| Direction | Data | Purpose |
|-----------|------|---------|
| MoE → EPLB function | `topk_ids` (logical) | Which experts each token wants |
| MoE → EPLB function | `expert_load_view` (tensor ref) | Where to record load |
| MoE → EPLB function | `logical_to_physical_map` | Mapping table |
| EPLB function → MoE | `return topk_ids` (physical) | Mapped expert IDs |
| EPLB side effect | `expert_load_view += counts` | Load recorded in EPLB's tensor |

**Note:** The EPLB function doesn't "take over" - it's a simple function that:
1. Transforms the expert IDs (logical → physical)
2. Records load statistics as a side effect
3. Returns immediately to the MoE layer

### Interaction Point 2: EPLB → MoE (Weight Rearrangement)

**When:** Every 3000 steps, after model forward completes

```
GPUModelRunner.execute_model()
    │
    ├── _model_forward()              # Runs all layers including MoE
    │
    └── self.eplb_step()              # Called AFTER forward
            │
            └── EplbState.step()
                    │
                    ├── (every step) Save load to window, reset accumulator
                    │
                    └── (every 3000 steps) self.rearrange()
                            │
                            ├── All-reduce load across GPUs
                            ├── Policy computes new placement
                            │
                            │  # EPLB CALLS BACK INTO MoE ════════════════════
                            └── rearrange_expert_weights_inplace(model, ...)
                                    │
                                    ├── For each MoE layer in model:
                                    │       │
                                    │       ├── P2P transfer expert weights
                                    │       │   (move weight tensors between GPUs)
                                    │       │
                                    │       └── Update layer's EPLB tensors:
                                    │           - logical_to_physical_map
                                    │           - logical_replica_count
                                    │           - expert_load_view (reset)
                                    │
                                    └── All MoE layers now have new mappings
```

**What gets modified:**

| What EPLB Changes | In MoE Layer | Effect |
|-------------------|--------------|--------|
| Expert weight tensors | `layer.experts.w1`, `w2`, `w3` | Physical expert parameters moved |
| `logical_to_physical_map` | `layer.logical_to_physical_map` | New routing table |
| `logical_replica_count` | `layer.logical_replica_count` | Updated replica counts |

### Timeline: How They Interact Over Time

```
Step 1:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ MoE forward: eplb_map_to_physical_and_record() writes to EPLB       │
  └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ EPLB.step(): saves load data, resets accumulator                    │
  └─────────────────────────────────────────────────────────────────────┘

Step 2-2999:
  (same pattern repeats, load statistics accumulate in window)

Step 3000:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ MoE forward: eplb_map_to_physical_and_record() (as usual)           │
  └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ EPLB.step() detects step == 3000, triggers rearrange()              │
  │                                                                     │
  │   1. Sum load across window                                         │
  │   2. All-reduce across GPUs                                         │
  │   3. Policy computes new placement                                  │
  │   4. ══════════════════════════════════════════════════════════     │
  │      EPLB MODIFIES MoE: rearrange_expert_weights_inplace()          │
  │      - Transfers weights between GPUs                               │
  │      - Updates MoE layer mapping tables                             │
  │      ══════════════════════════════════════════════════════════     │
  └─────────────────────────────────────────────────────────────────────┘

Step 3001:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ MoE forward: uses NEW mappings from EPLB rearrangement              │
  └─────────────────────────────────────────────────────────────────────┘
```

### Code Locations Summary

| Interaction | File | Function |
|-------------|------|----------|
| MoE calls EPLB (load collection) | `fused_moe/layer.py` | `forward_impl()` |
| EPLB function called | `fused_moe/fused_moe.py` | `eplb_map_to_physical_and_record()` |
| EPLB step entry | `gpu_model_runner.py` | `eplb_step()` |
| EPLB state step | `eplb/eplb_state.py` | `step()` |
| EPLB calls MoE (rearrange) | `eplb/eplb_state.py` | `rearrange()` |
| Weight transfer | `eplb/rebalance_execute.py` | `rearrange_expert_weights_inplace()` |

---

### Logical vs Physical Experts

| Term | Definition |
|------|------------|
| **Logical Expert** | Expert ID in the model's original design (e.g., 0-71 for DeepSeek-V3-Lite) |
| **Physical Expert** | Actual instantiated copy on a GPU (can include redundant replicas) |
| **Redundant Expert** | Extra copies of popular experts for load balancing |

**Example (DeepSeek-V3-Lite):**
- 72 logical experts
- 16 redundant copies → 88 physical experts total
- 4 GPUs → 22 physical experts per GPU

---

## EPLB Deep Dive

### What is EPLB?

EPLB (Expert Parallelism Load Balancer) dynamically redistributes expert weights across GPUs to balance computational load.

**Problem it solves:** Some experts become "hot" (receive many tokens) while others are "cold" (receive few tokens). This creates load imbalance across GPUs.

```
WITHOUT EPLB (static placement):         ────▶   WITH EPLB (dynamic rebalancing):
┌─────────────────────────────────────┐          ┌─────────────────────────────────────┐
│ GPU 0: experts 0-21                 │          │ GPU 0: experts 0-17, 33(hot), 45    │
│        ████████████ (120% load)     │          │        ██████████ (100% load)       │
│                                     │          │                                     │
│ GPU 1: experts 22-43                │          │ GPU 1: experts 18-32, 33(replica)   │
│        ███████████████ (150% load)  │          │        ██████████ (100% load)       │
│        (expert 33 is HOT!)          │          │                                     │
│                                     │          │ GPU 2: experts 34-54, 33(replica)   │
│ GPU 2: experts 44-65                │          │        ██████████ (100% load)       │
│        ██████ (60% load)            │          │                                     │
│                                     │          │ GPU 3: experts 55-71, 33(replica)   │
│ GPU 3: experts 66-87                │          │        ██████████ (100% load)       │
│        ████ (40% load)              │          │                                     │
└─────────────────────────────────────┘          └─────────────────────────────────────┘
```

**Solution:** 
1. **Monitor** token counts per expert during inference
2. **Identify** hot/cold experts based on observed traffic
3. **Replicate** hot experts on multiple GPUs (redundant experts)
4. **Transfer** expert weights to new locations

### EPLB Data Structures (The Heart of EPLB)

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                          EPLB DATA STRUCTURE HIERARCHY                                │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  EplbState (one per EP rank)                                                          │
│  ├── expert_rearrangement_step: int = 0      # counts toward step_interval (3000)     │
│  ├── expert_load_window_step: int = 0        # current position in sliding window     │
│  ├── expert_load_window_size: int = 1000     # size of sliding window                 │
│  ├── expert_rearrangement_step_interval: int = 3000  # trigger rearrangement          │
│  │                                                                                    │
│  └── model_states: Dict[str, EplbModelState]                                          │
│       │                                                                               │
│       └── EplbModelState (one per model)                                              │
│            │                                                                          │
│            ├── expert_load_pass: Tensor [29, 88]                                      │
│            │   └── Accumulates load during ONE forward pass (all 29 MoE layers)       │
│            │       Zeroed after each step()                                           │
│            │                                                                          │
│            ├── expert_load_window: Tensor [1000, 29, 88]                              │
│            │   └── Sliding window of last 1000 steps' load data                       │
│            │       expert_load_window[step % 1000] = expert_load_pass                 │
│            │                                                                          │
│            ├── physical_to_logical_map: Tensor [29, 88]                               │
│            │   └── Which logical expert does each physical expert represent?          │
│            │       physical_to_logical_map[layer][phys_id] = logical_id               │
│            │                                                                          │
│            ├── logical_to_physical_map: Tensor [29, 72, max_replicas]                 │
│            │   └── Which physical experts implement each logical expert?              │
│            │       logical_to_physical_map[layer][log_id] = [phys1, phys2, ...]       │
│            │                                                                          │
│            └── logical_replica_count: Tensor [29, 72]                                 │
│                └── How many physical copies of each logical expert?                   │
│                    Most = 1, hot experts = 2-4                                        │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### How expert_load_view Connects to expert_load_window

This is the critical connection that makes EPLB data collection work:

```python
# In EplbState.register_model() - called once during model initialization
expert_load_pass = torch.zeros(num_moe_layers, num_physical_experts)  # [29, 88]
expert_load_window = torch.zeros(window_size, num_moe_layers, num_physical_experts)  # [1000, 29, 88]

# In model.set_eplb_state() - connects each MoE layer to its slice
for moe_layer_idx, moe_layer in enumerate(model.moe_layers):
    moe_layer.expert_load_view = expert_load_pass[moe_layer_idx]  # [88] - view, not copy!
    #                                             ^^^^^^^^^^^^^^^^^
    #                                             This is a VIEW into expert_load_pass
    #                                             When MoE writes to expert_load_view,
    #                                             it actually writes to expert_load_pass

# During MoE forward (called 29 times per step, once per MoE layer)
# In eplb_map_to_physical_and_record():
expert_load_view.scatter_add_(index=topk_ids.flatten(), src=ones)  # Writes to expert_load_pass!

# After model forward completes, in EplbState.step():
expert_load_window[window_step] = expert_load_pass.clone()  # Save to history
expert_load_pass.zero_()  # Reset for next step
```

**Visual timeline:**

```
Step 0:
  MoE layer 0: expert_load_view[expert_45] += 1  → writes to expert_load_pass[0, 45]
  MoE layer 1: expert_load_view[expert_12] += 1  → writes to expert_load_pass[1, 12]
  ... (29 layers)
  EplbState.step():
    expert_load_window[0] = expert_load_pass.clone()  # [29, 88] saved
    expert_load_pass.zero_()

Step 1:
  MoE layer 0: expert_load_view[expert_33] += 1  → writes to expert_load_pass[0, 33]
  ...
  EplbState.step():
    expert_load_window[1] = expert_load_pass.clone()
    expert_load_pass.zero_()

...

Step 999:
  ...
  EplbState.step():
    expert_load_window[999] = expert_load_pass.clone()
    expert_load_pass.zero_()

Step 1000:  (wraps around)
  ...
  EplbState.step():
    expert_load_window[0] = expert_load_pass.clone()  # Overwrites step 0's data
    expert_load_pass.zero_()
```

### EPLB Key Classes

#### `EplbState` (Global Controller)

**File:** `vllm/distributed/eplb/eplb_state.py`

```python
class EplbState:
    # STEP COUNTERS
    expert_rearrangement_step: int = 0     # 0 → 3000, then trigger rearrange
    expert_load_window_step: int = 0       # 0 → 999, then wrap to 0
    expert_rearrangement_step_interval: int = 3000  # from config
    expert_load_window_size: int = 1000             # from config
    
    # PER-MODEL STATE
    model_states: Dict[str, EplbModelState]
    
    # KEY METHODS
    def step(is_dummy, is_profile):
        """Called after every model forward"""
        # 1. Save current pass to window
        expert_load_window[window_step] = expert_load_pass.clone()
        expert_load_pass.zero_()
        window_step = (window_step + 1) % window_size
        
        # 2. Check if rearrangement needed
        rearrangement_step += 1
        if rearrangement_step >= step_interval:
            self.rearrange()
    
    def rearrange():
        """Trigger expert rebalancing"""
        # 1. Sum load across window: [1000,29,88] → [29,88]
        # 2. Map physical→logical: [29,88] → [29,72]
        # 3. All-reduce across EP ranks
        # 4. Call policy to compute new mapping
        # 5. Transfer weights between GPUs
```

#### `EplbModelState` (Per-Model State)

**File:** `vllm/distributed/eplb/eplb_state.py`

Holds all the tensors for one model:
- `expert_load_pass` - Current step accumulator
- `expert_load_window` - Historical load data
- `physical_to_logical_map` - Mapping tables
- `logical_to_physical_map` - Mapping tables
- `logical_replica_count` - Replication counts

#### `DefaultEplbPolicy` (Rebalancing Algorithm)

**File:** `vllm/distributed/eplb/policy/default.py`

```python
def rebalance_experts(global_expert_load, num_replicas, num_groups, num_nodes, num_gpus):
    """
    Input: global_expert_load [29, 72] - total tokens per logical expert across all ranks
    Output: new_physical_to_logical_map [88] - which logical expert each physical slot gets
    
    Algorithm:
    1. Identify hot experts (high token count)
    2. Assign replicas to hot experts (up to num_redundant_experts total)
    3. Pack experts into GPUs using hierarchical bin packing:
       - First pack into groups (if any)
       - Then pack groups into nodes
       - Then pack nodes into GPUs
    4. Return new mapping
    """
```

#### `rearrange_expert_weights_inplace` (Weight Transfer)

**File:** `vllm/distributed/eplb/rebalance_execute.py`

```python
def rearrange_expert_weights_inplace(model, old_mapping, new_mapping):
    """
    Actually move expert weight tensors between GPUs.
    
    For each MoE layer:
    1. Identify which experts need to move where
    2. Use P2P communication (send/recv) to transfer weights
    3. Update local expert weights in-place
    4. Update routing tables
    
    This is a collective operation - all EP ranks must participate!
    """
```

### How EPLB Gets Its Data (Complete Flow)

**EPLB does NOT receive requests or tokens. It receives per-expert routing decision counts.**

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           EPLB DATA COLLECTION FLOW                                   │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  DURING MODEL FORWARD (called every step)                                             │
│  ─────────────────────────────────────────                                            │
│                                                                                       │
│  For each MoE layer (29 times per step):                                              │
│    ┌───────────────────────────────────────────────────────────────────────────────┐  │
│    │  FusedMoE.forward_impl():                                                     │  │
│    │    1. Router computes scores: [num_tokens, 72]                                │  │
│    │    2. Top-K selection: topk_ids [num_tokens, 6]                               │  │
│    │                                                                               │  │
│    │    3. eplb_map_to_physical_and_record():                                      │  │
│    │       ┌─────────────────────────────────────────────────────────────────────┐ │  │
│    │       │  # topk_ids contains 6 expert IDs per token                         │ │  │
│    │       │  # Example: batch of 1024 tokens → 6144 routing decisions           │ │  │
│    │       │                                                                     │ │  │
│    │       │  topk_ids_flatten = topk_ids.flatten()  # [6144]                    │ │  │
│    │       │  # e.g., [45, 12, 33, 8, 71, 0, 33, 45, 12, ...]                    │ │  │
│    │       │                                                                     │ │  │
│    │       │  expert_load_view.scatter_add_(                                     │ │  │
│    │       │      index=topk_ids_flatten,  # which expert                        │ │  │
│    │       │      src=ones(6144)           # add 1 for each routing decision     │ │  │
│    │       │  )                                                                  │ │  │
│    │       │                                                                     │ │  │
│    │       │  # After this, expert_load_view looks like:                         │ │  │
│    │       │  # expert_load_view = [143, 267, 89, 312, ..., 198, 45]             │ │  │
│    │       │  #                     │     │         │                            │ │  │
│    │       │  #                   exp0  exp1      exp3 got 312 routings          │ │  │
│    │       └─────────────────────────────────────────────────────────────────────┘ │  │
│    └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                       │
│  After 29 MoE layers complete:                                                        │
│    expert_load_pass: [29, 88] contains routing counts for ALL layers this step        │
│                                                                                       │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  AFTER MODEL FORWARD (EplbState.step())                                               │
│  ──────────────────────────────────────                                               │
│                                                                                       │
│    ┌───────────────────────────────────────────────────────────────────────────────┐  │
│    │  # Save this step's data to history window                                    │  │
│    │  expert_load_window[window_step] = expert_load_pass.clone()                   │  │
│    │  #                  ^^^^^^^^^^^                                               │  │
│    │  #                  0-999, wraps around                                       │  │
│    │                                                                               │  │
│    │  # Reset for next step                                                        │  │
│    │  expert_load_pass.zero_()                                                     │  │
│    │                                                                               │  │
│    │  # Advance counters                                                           │  │
│    │  window_step = (window_step + 1) % 1000                                       │  │
│    │  rearrangement_step += 1                                                      │  │
│    │                                                                               │  │
│    │  # Check if time to rebalance                                                 │  │
│    │  if rearrangement_step >= 3000:  # step_interval                              │  │
│    │      self.rearrange()  # TRIGGER REBALANCING                                  │  │
│    └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### The Rearrangement Process (Every 3000 Steps)

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           REARRANGEMENT FLOW (EPLB_FLOW_02-04)                        │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  STEP 1: Aggregate Load Data                                                          │
│  ────────────────────────────                                                         │
│    # Sum across sliding window: [1000, 29, 88] → [29, 88]                             │
│    total_load_physical = expert_load_window.sum(dim=0)                                │
│                                                                                       │
│    # Map physical → logical experts: [29, 88] → [29, 72]                              │
│    # (Combine counts from replicas of same logical expert)                            │
│    total_load_logical = scatter_add(total_load_physical, physical_to_logical_map)     │
│                                                                                       │
│    Example:                                                                           │
│      Physical expert 45 (logical 33): 15000 routings                                  │
│      Physical expert 78 (logical 33): 12000 routings  ← replica of 33                 │
│      → Logical expert 33 total: 27000 routings                                        │
│                                                                                       │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  STEP 2: All-Reduce Across EP Ranks     >>> EPLB_FLOW_02 logged here <<<              │
│  ─────────────────────────────────────                                                │
│    # Each GPU only sees tokens routed TO it, need global view                         │
│    global_load = all_reduce(total_load_logical, op=SUM)  # [29, 72]                   │
│                                                                                       │
│    Now all 4 GPUs have the same global_load tensor showing                            │
│    total routings to each logical expert across ALL GPUs                              │
│                                                                                       │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  STEP 3: Policy Computes New Mapping    >>> EPLB_FLOW_03 logged here <<<              │
│  ───────────────────────────────────                                                  │
│    new_physical_to_logical = DefaultEplbPolicy.rebalance_experts(                     │
│        global_load,           # [29, 72] - what we observed                           │
│        num_replicas=88,       # total physical expert slots                           │
│        num_gpus=4,            # EP size                                               │
│    )                                                                                  │
│                                                                                       │
│    Algorithm:                                                                         │
│    1. Rank logical experts by load: [33, 12, 45, 0, 71, ...]  (hottest first)         │
│    2. Assign replicas: expert 33 gets 3 copies, expert 12 gets 2, rest get 1          │
│    3. Pack into GPUs: balance load per GPU                                            │
│                                                                                       │
│    Output: new mapping table                                                          │
│      new_physical_to_logical = [0, 1, 2, ..., 33, 33, 33, 12, 12, ...]                │
│                                            ^^^^^^^^^^^^ replicas                      │
│                                                                                       │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  STEP 4: Weight Transfer                >>> EPLB_FLOW_04 logged here <<<              │
│  ───────────────────────────                                                          │
│    rearrange_expert_weights_inplace(model, old_mapping, new_mapping)                  │
│                                                                                       │
│    For each MoE layer:                                                                │
│      For each physical expert slot:                                                   │
│        if old_mapping[slot] != new_mapping[slot]:                                     │
│          # Need to load different expert into this slot                               │
│          source_gpu = find_gpu_with_expert(new_mapping[slot])                         │
│          p2p_transfer(weights, source_gpu → current_gpu)                              │
│                                                                                       │
│    This is COLLECTIVE: all GPUs execute this together                                 │
│    (some send, some receive, some do both)                                            │
│                                                                                       │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  STEP 5: Update Routing Tables                                                        │
│  ────────────────────────────                                                         │
│    # Update all mapping tensors to reflect new placement                              │
│    physical_to_logical_map = new_physical_to_logical                                  │
│    logical_to_physical_map = compute_inverse(new_physical_to_logical)                 │
│    logical_replica_count = count_replicas(new_physical_to_logical)                    │
│                                                                                       │
│    # Reset counters                                                                   │
│    rearrangement_step = 0                                                             │
│    expert_load_window.zero_()  # Start fresh observation                              │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### When is `EplbState.step()` Called?

**Location:** `vllm/v1/worker/gpu_model_runner.py` → `GPUModelRunner.execute_model()`

```
EngineCore.step()                                    # core.py
    │
    ├── scheduler.schedule()                         # scheduler.py
    │       │
    │       └── Returns SchedulerOutput with batch info
    │
    └── model_executor.execute_model(scheduler_output)
            │
            └── Worker.execute_model()               # gpu_worker.py
                    │
                    └── GPUModelRunner.execute_model()  # gpu_model_runner.py
                            │
                            ├── _model_forward()     # Actual transformer forward
                            │       │
                            │       └── For each layer:
                            │               ├── Attention
                            │               └── FusedMoE.forward_impl()
                            │                       │
                            │                       └── eplb_map_to_physical_and_record()
                            │                               └── Writes to expert_load_view
                            │
                            ├── (sampling, bookkeeping)
                            │
                            └── self.eplb_step()     ◀◀◀ CALLED HERE [EPLB_FLOW_00]
                                    │
                                    └── self.eplb_state.step()  [EPLB_FLOW_01]
                                            │
                                            ├── Save expert_load_pass to window
                                            ├── Zero expert_load_pass
                                            ├── Increment counters
                                            │
                                            └── If step == 3000:
                                                    └── self.rearrange() [EPLB_FLOW_02-04]
```

**Code reference** (`gpu_model_runner.py:3456`):

```python
# After model forward and sampling complete
with record_function_or_nullcontext("gpu_model_runner: eplb"):
    self.eplb_step()  # ← EPLB step called HERE, after every forward
```

### EPLB Configuration

```python
EPLBConfig(
    window_size=1000,           # Sliding window size (how many steps to remember)
    step_interval=3000,         # Steps between rearrangements
    num_redundant_experts=16,   # Extra physical expert slots for replicas
    log_balancedness=False,     # Log load imbalance stats
    use_async=False,            # Async weight transfer (experimental)
    policy='default'            # Rebalancing algorithm
)
```

| Parameter | Value | What it means |
|-----------|-------|---------------|
| `window_size=1000` | Remember last 1000 steps of load data. Older data is overwritten. |
| `step_interval=3000` | Trigger rearrangement every 3000 steps (~3000 decode iterations) |
| `num_redundant_experts=16` | 72 logical + 16 redundant = 88 physical expert slots |

### EPLB Timeline Example

```
Step 0:
  └── Model forward → MoE layers write to expert_load_pass
  └── EplbState.step() → expert_load_window[0] = expert_load_pass

Step 1-999:
  └── Same pattern, filling window positions 1-999
  └── expert_load_window now has 1000 samples

Step 1000:
  └── Window wraps: expert_load_window[0] overwritten (oldest data discarded)
  └── Still collecting, rearrangement_step = 1000

Step 2000:
  └── Window wraps again: expert_load_window[0] = step 2000's data
  └── rearrangement_step = 2000

Step 2999:
  └── Last step before rearrangement
  └── rearrangement_step = 2999

Step 3000:                    ◀◀◀ REARRANGEMENT TRIGGERED ◀◀◀
  └── Model forward completes normally
  └── EplbState.step() detects step == 3000:
      │
      ├── [EPLB_FLOW_02] Sum load across window, all-reduce across GPUs
      ├── [EPLB_FLOW_03] Policy computes new expert placement
      ├── [EPLB_FLOW_04] P2P weight transfer between GPUs
      │
      └── Reset: rearrangement_step = 0, expert_load_window.zero_()

Step 3001:
  └── Fresh start, new observation period begins
  └── rearrangement_step = 1
```

---

## End-to-End Flow: Request → Token → MoE → EPLB

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. REQUEST LEVEL (API Server + Scheduler)                                  │
│     ─────────────────────────────────────                                   │
│     Request 1: "Hello, how are you?" (6 tokens)                             │
│     Request 2: "What is 2+2?"        (5 tokens)                             │
│                     │                                                       │
│                     ▼                                                       │
│     Scheduler batches them together                                         │
│                     │                                                       │
├─────────────────────┼───────────────────────────────────────────────────────┤
│                     ▼                                                       │
│  2. BATCH LEVEL (SchedulerOutput → Worker)                                  │
│     ────────────────────────────────────────                                │
│     SchedulerOutput:                                                        │
│       - total_num_scheduled_tokens: 11                                      │
│       - request metadata (for output routing)                               │
│                     │                                                       │
│                     ▼                                                       │
├─────────────────────┼───────────────────────────────────────────────────────┤
│                     ▼                                                       │
│  3. TOKEN LEVEL (GPUModelRunner → Model → MoE)                              │
│     ───────────────────────────────────────────                             │
│     hidden_states: torch.Tensor [11, 2560]                                  │
│                                                                             │
│     The model sees 11 tokens. It does NOT know:                             │
│       - Which tokens belong to which request                                │
│       - Request boundaries                                                  │
│                     │                                                       │
│                     ▼                                                       │
│     MoE Layer processes all 11 tokens:                                      │
│       - Router computes scores for each token                               │
│       - Top-K selection picks 6 experts per token                           │
│       - Total routing decisions: 11 × 6 = 66                                │
│                     │                                                       │
│                     ▼                                                       │
├─────────────────────┼───────────────────────────────────────────────────────┤
│                     ▼                                                       │
│  4. EPLB LEVEL (expert_load_window)                                         │
│     ───────────────────────────────                                         │
│     EPLB records: 66 routing decisions added to expert_load_window          │
│                                                                             │
│     EPLB sees:                                                              │
│       expert_load_window[layer][expert_id] += count                         │
│                                                                             │
│     EPLB does NOT see:                                                      │
│       - "Request 1" or "Request 2"                                          │
│       - Token boundaries                                                    │
│       - Just aggregate counts                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Summary: What Each Component Sees

| Component | Sees | Does NOT see |
|-----------|------|--------------|
| **Scheduler** | Requests, priorities, KV cache | Token content, embeddings |
| **GPUModelRunner** | Batch of tokens as tensors | Individual request boundaries |
| **MoE Layer** | Hidden states `[num_tokens, 2560]` | Which request each token belongs to |
| **Router** | Score per expert per token `[num_tokens, 72]` | Why a token needs certain experts |
| **EPLB** | Per-expert routing counts (integers) | Request IDs, individual tokens, timing |

### EPLB Critical Numbers (Your Setup)

| Quantity | Value | Formula |
|----------|-------|---------|
| Routing decisions per token | 6 | `top_k = 6` |
| Routing decisions per step (prefill 1024 tokens) | 6144 | `1024 × 6` |
| Routing decisions per step (decode, 1 token) | 6 | `1 × 6` |
| Total routing decisions per MoE layer per step | 6-6144 | depends on batch |
| Total MoE layers | 29 | layers 1-29 |
| Load entries recorded per step | 29 × 88 | `num_layers × num_physical_experts` |
| Window history depth | 1000 steps | `window_size` |
| Steps between rearrangements | 3000 | `step_interval` |

### The Key Insight

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  REQUESTS ≠ TOKENS ≠ ROUTING DECISIONS                                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  1 Request "Hello world" (2 tokens)                                            │
│      │                                                                         │
│      └─── 2 Tokens in batch                                                    │
│               │                                                                │
│               └─── 12 Routing decisions (2 tokens × 6 experts each)            │
│                        │                                                       │
│                        └─── 12 increments to expert_load_view                  │
│                                 │                                              │
│                                 └─── Distributed across 88 physical experts    │
│                                                                                │
│  EPLB only sees: "expert 33 got 3 routings, expert 12 got 2, ..."              │
│  EPLB never knows: "these came from 1 request with 2 tokens"                   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Your Deployment Architecture

**Configuration:** Ray Data Parallel (DP) with one GPU per DP rank.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           API Server Process                             │
│  ┌─────────────┐    ┌───────────────────┐    ┌─────────────────────────┐ │
│  │   FastAPI   │───>│ OpenAIServingChat │───>│  DPLBAsyncMPClient      │ │
│  │  (Uvicorn)  │    │                   │    │  (DP Load Balancer)     │ │
│  └─────────────┘    └───────────────────┘    └──────────┬──────────────┘ │
└─────────────────────────────────────────────────────────┼────────────────┘
                                                          │ ZMQ (routes by dp_rank)
          ┌────────────────┬────────────────┬─────────────┴─────────────┐
          ▼                ▼                ▼                           ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ DPEngineCoreActor│ │ DPEngineCoreActor│ │ DPEngineCoreActor│ │ DPEngineCoreActor│
│    (dp_rank=0)   │ │    (dp_rank=1)   │ │    (dp_rank=2)   │ │    (dp_rank=3)   │
│ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │
│ │ EngineCore   │ │ │ │ EngineCore   │ │ │ │ EngineCore   │ │ │ │ EngineCore   │ │
│ │ + Scheduler  │ │ │ │ + Scheduler  │ │ │ │ + Scheduler  │ │ │ │ + Scheduler  │ │
│ ├──────────────┤ │ │ ├──────────────┤ │ │ ├──────────────┤ │ │ ├──────────────┤ │
│ │UniProcExec   │ │ │ │UniProcExec   │ │ │ │UniProcExec   │ │ │ │UniProcExec   │ │
│ │ + Worker     │ │ │ │ + Worker     │ │ │ │ + Worker     │ │ │ │ + Worker     │ │
│ │ + ModelRunner│ │ │ │ + ModelRunner│ │ │ │ + ModelRunner│ │ │ │ + ModelRunner│ │
│ └──────┬───────┘ │ │ └──────┬───────┘ │ │ └──────┬───────┘ │ │ └──────┬───────┘ │
│        │         │ │        │         │ │        │         │ │        │         │
│      GPU 0       │ │      GPU 1       │ │      GPU 2       │ │      GPU 3       │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
       EP rank 0          EP rank 1           EP rank 2           EP rank 3
```

### What is `DPEngineCoreActor`?

A **Ray actor** that hosts a complete vLLM engine core for one DP rank:

- **One actor per DP rank** (owns one GPU)
- Runs in-process: `EngineCore` + `Scheduler` + `UniProcExecutor` + `Worker` + `GPUModelRunner`
- Receives requests routed by `DPAsyncMPClient` via ZMQ
- With `UniProcExecutor`: no worker subprocess, so `REQ_FLOW_12` is skipped in traces

---

## Reference

### Trace Markers (Print Statements)

The following trace markers are embedded in the codebase for debugging the request flow:

#### Request Flow Markers (`REQ_FLOW_*`)

| Marker | File | Function | Description |
|--------|------|----------|-------------|
| `REQ_FLOW_01` | `api_server.py` | `create_completion()` / `create_chat_completion()` | HTTP request received |
| `REQ_FLOW_02` | `serving_completion.py` / `serving_chat.py` | `create_completion()` / `create_chat_completion()` | Serving handler called |
| `REQ_FLOW_03` | `async_llm.py` | `generate()` | AsyncLLM.generate() entry |
| `REQ_FLOW_03a` | `input_processor.py` | `process_inputs()` | Input processing |
| `REQ_FLOW_04` | `async_llm.py` | `add_request()` | Request added to EngineCore |
| `REQ_FLOW_04a` | `core_client.py` | `AsyncMPClient.add_request_async()` | ZMQ send (single DP) |
| `REQ_FLOW_04b` | `core_client.py` | `DPLBAsyncMPClient.add_request_async()` | ZMQ send (DP load balanced) |
| `REQ_FLOW_05` | `core.py` | `_handle_client_request()` | EngineCore receives request via ZMQ |
| `REQ_FLOW_06` | `core.py` | `add_request()` | Dispatching to scheduler |
| `REQ_FLOW_07` | `scheduler.py` | `add_request()` | Scheduler adds request |
| `REQ_FLOW_08` | `scheduler.py` | `schedule()` | Scheduler.schedule() entry |
| `REQ_FLOW_09` | `scheduler.py` | `schedule()` | Batch ready |
| `REQ_FLOW_10` | `core.py` | `step()` | EngineCore.step() calling schedule |
| `REQ_FLOW_11` | `core.py` | `step()` | EngineCore.step() executing model |
| `REQ_FLOW_12` | `multiproc_executor.py` / `ray_executor.py` | `execute_model()` | Executor dispatching to workers |
| `REQ_FLOW_13` | `gpu_worker.py` | `execute_model()` | Worker.execute_model() |
| `REQ_FLOW_14` | `gpu_model_runner.py` | `execute_model()` | GPUModelRunner.execute_model() |
| `REQ_FLOW_15` | `gpu_model_runner.py` | `execute_model()` | _model_forward() starting |
| `REQ_FLOW_16` | `gpu_model_runner.py` | `execute_model()` | _model_forward() complete |
| `REQ_FLOW_16a` | `gpu_model_runner.py` | `sample_tokens()` | Sampling starting |
| `REQ_FLOW_17` | `scheduler.py` | `update_from_output()` | Scheduler updates from model output |
| `REQ_FLOW_18` | `async_llm.py` | `output_handler()` | Output handler received outputs |
| `REQ_FLOW_18a` | `output_processor.py` | `process_outputs()` | Output processing |
| `REQ_FLOW_19` | `async_llm.py` | `generate()` | Request complete |

#### Model Flow Markers (`MODEL_FLOW_*`)

| Marker | File | Function | Description |
|--------|------|----------|-------------|
| `MODEL_FLOW_01` | `deepseek_v2.py` | `DeepseekV2ForCausalLM.forward()` | Model forward entry |
| `MODEL_FLOW_04` | `deepseek_v2.py` | `DeepseekV2ForCausalLM.forward()` | Model forward complete |

**Note:** Prints inside `@support_torch_compile` decorated methods are not allowed (breaks torch.dynamo).

#### MoE Flow Markers (`MOE_FLOW_*`)

| Marker | File | Function | Description |
|--------|------|----------|-------------|
| `MOE_FLOW_01` | `layer.py` | `select_experts()` | Expert selection (first call per layer) |
| `MOE_FLOW_02` | `layer.py` | `forward_impl()` | FusedMoE forward entry (first call per layer) |
| `MOE_FLOW_03` | `layer.py` | `select_experts()` | EPLB mapping called (first call per layer) |
| `MOE_FLOW_03a/b/c` | `layer.py` | `select_experts()` | EPLB mapping details |
| `MOE_FLOW_04` | `layer.py` | `forward_impl()` | MoE layer complete (first call per layer) |

#### EPLB Flow Markers (`EPLB_FLOW_*`)

| Marker | File | Function | Description |
|--------|------|----------|-------------|
| `EPLB_FLOW_01` | `eplb_state.py` | `step()` | EplbState.step() (every 100 steps) |
| `EPLB_FLOW_01a/b/c` | `eplb_state.py` | `step()` | Load window save details (first call) |
| `EPLB_FLOW_02` | `eplb_state.py` | `rearrange()` | Rearrangement starting |
| `EPLB_FLOW_03` | `policy/default.py` | `rebalance_experts()` | Policy computing new mapping |
| `EPLB_FLOW_04` | `rebalance_execute.py` | `rearrange_expert_weights_inplace()` | Weight transfer |

#### All-to-All Flow Markers (`ALL2ALL_FLOW_*`)

| Marker | File | Function | Description |
|--------|------|----------|-------------|
| `ALL2ALL_FLOW_01` | `all2all.py` | `dispatch()` | All-to-all dispatch starting |
| `ALL2ALL_FLOW_02` | `all2all.py` | `dispatch()` | All-to-all dispatch complete |
| `ALL2ALL_FLOW_03` | `all2all.py` | `combine()` | All-to-all combine starting |
| `ALL2ALL_FLOW_04` | `all2all.py` | `combine()` | All-to-all combine complete |

### Filtering Logs

```bash
# Full request flow
grep -E "REQ_FLOW" server.log

# Model + MoE internals
grep -E "(MODEL_FLOW|MOE_FLOW)" server.log

# EPLB load balancing
grep -E "EPLB_FLOW" server.log

# All-to-all communication (EP)
grep -E "ALL2ALL_FLOW" server.log

# Everything
grep -E "(REQ_FLOW|MODEL_FLOW|MOE_FLOW|EPLB_FLOW|ALL2ALL_FLOW)" server.log
```

---

### Files and Classes

#### MoE

| File | Class/Function | Role |
|------|----------------|------|
| `model_executor/layers/fused_moe/layer.py` | `FusedMoE` | Main MoE layer |
| `model_executor/layers/fused_moe/layer.py` | `select_experts()` | Top-K routing |
| `model_executor/layers/fused_moe/layer.py` | `forward_impl()` | MoE forward pass |
| `model_executor/layers/fused_moe/fused_moe.py` | `eplb_map_to_physical_and_record()` | EPLB mapping + load recording |
| `model_executor/layers/fused_moe/all2all_utils.py` | Dispatch/Combine | EP all-to-all |

#### EPLB

| File | Class/Function | Role |
|------|----------------|------|
| `distributed/eplb/eplb_state.py` | `EplbState` | Global controller |
| `distributed/eplb/eplb_state.py` | `EplbModelState` | Per-model state |
| `distributed/eplb/policy/default.py` | `DefaultEplbPolicy` | Rebalancing algorithm |
| `distributed/eplb/rebalance_execute.py` | `rearrange_expert_weights_inplace()` | Weight transfer |

#### Request Flow

| File | Class/Function | Role |
|------|----------------|------|
| `entrypoints/openai/api_server.py` | `create_completion()` | HTTP endpoint |
| `v1/engine/async_llm.py` | `AsyncLLM` | Frontend engine |
| `v1/engine/core.py` | `EngineCoreProc` | Main loop, ZMQ |
| `v1/core/sched/scheduler.py` | `Scheduler` | Batch formation |
| `v1/worker/gpu_worker.py` | `Worker` | Per-GPU wrapper |
| `v1/worker/gpu_model_runner.py` | `GPUModelRunner` | Model forward |
