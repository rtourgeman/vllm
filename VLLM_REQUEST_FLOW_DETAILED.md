# vLLM Request Flow - Detailed Guide

## Complete Call Path: HTTP Request → MoE Layer → Response

The following flow has been **verified with actual trace logs**. Each `[REQ_FLOW_*]` marker corresponds to a print statement in the code.

```
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                     ║
║   COMPLETE vLLM REQUEST FLOW - FROM HTTP REQUEST TO RESPONSE                        ║
║                                                                                     ║
║   This diagram shows every step a "packet" takes from when the user sends           ║
║   a request until they receive the response.                                        ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: HTTP REQUEST ENTRY                                                         │
│  File: vllm/entrypoints/openai/api_server.py                                        │
│  Function: create_completion()                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  User sends:                                                                        │
│    curl -X POST http://localhost:8006/v1/completions \                              │
│         -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'                    │
│                                                                                     │
│  [REQ_FLOW_01] HTTP /v1/completions received | model=... | stream=False             │
│      │                                                                              │
│      │   FastAPI receives HTTP POST request                                         │
│      │   Request is validated and routed to handler                                 │
│      │                                                                              │
│      └──────────────────────────────────────────────────────────────────────────────│
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: REQUEST VALIDATION & PREPARATION                                           │
│  File: vllm/entrypoints/openai/serving_completion.py                                │
│  Function: OpenAIServingCompletion.create_completion()                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_02] OpenAIServingCompletion.create_completion() | model=... | n=1        │
│      │                                                                              │
│      ├── Validate model exists                                                      │
│      ├── Create SamplingParams (temperature, max_tokens, etc.)                      │
│      └── Call engine_client.generate()                                              │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: ASYNC ENGINE ENTRY                                                         │
│  File: vllm/v1/engine/async_llm.py                                                  │
│  Function: AsyncLLM.generate()                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_03] AsyncLLM.generate() called | request_id=cmpl-xxx                     │
│      │                                                                              │
│      ├── This is the MAIN ENTRY POINT to the vLLM engine                            │
│      ├── Create RequestOutputCollector (queue for receiving outputs)                │
│      ├── Start output_handler background task                                       │
│      └── Call add_request() to send to EngineCore                                   │
│                                                                                     │
│  NOTE: Input processing happens inside AsyncLLM.add_request():                      │
│                                                                                     │
│  [REQ_FLOW_03a] InputProcessor.process_inputs() | request_id=cmpl-xxx               │
│      ├── Tokenize prompt (or accept token IDs / embeds)                             │
│      ├── Validate params (sampling, LoRA, DP rank, etc.)                            │
│      ├── Create EngineCoreRequest with all request data                             │
│      └── assign_request_id(): request_id is MUTATED (e.g., cmpl-xxx-<8chars>)       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: ZMQ SEND TO ENGINE CORE                                                    │
│  File: vllm/v1/engine/core_client.py                                                │
│  Function: EngineCoreClient.add_request_async()                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_04b] DPLBAsyncMPClient sending request via ZMQ | request_id=... | dp=0   │
│      │          (or REQ_FLOW_04a for non-DPLB path)                                 │
│      │                                                                              │
│      │   ┌─────────────────────────────────────────────────────────────────────┐    │
│      │   │  IPC BOUNDARY - ZeroMQ Socket                                       │    │
│      │   │                                                                     │    │
│      │   │  Frontend process/thread ───────────────► EngineCore consumer       │    │
│      │   │  (handles HTTP)                           (may be separate process) │    │
│      │   │                                                                     │    │
│      │   │  Request is serialized and sent via ZMQ                             │    │
│      │   │  For Data Parallel (DP), routes to one of multiple EngineCores      │    │
│      │   └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                              │
│  [REQ_FLOW_04] Request added to EngineCore | request_id=cmpl-xxx-<8chars>           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ ═══════ ZMQ IPC BOUNDARY ═══════
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: ENGINE CORE RECEIVES REQUEST                                               │
│  File: vllm/v1/engine/core.py                                                       │
│  Function: EngineCore._handle_client_request()                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  NOW IN ENGINE CORE PROCESS (e.g., DPEngineCoreActor - Ray Actor)           │    │
│  │  This process runs on the GPU and does the actual model inference           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│  [REQ_FLOW_05] EngineCore received ADD request via ZMQ | request_id=cmpl-xxx        │
│      │                                                                              │
│  [REQ_FLOW_06] EngineCore dispatching ADD to scheduler | request_id=cmpl-xxx        │
│      │                                                                              │
│      └── Put request in input_queue for scheduler to process                        │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: SCHEDULER ADDS REQUEST TO QUEUE                                            │
│  File: vllm/v1/core/sched/scheduler.py                                              │
│  Function: Scheduler.add_request()                                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_07] Scheduler.add_request() | request_id=cmpl-xxx | num_tokens=6         │
│      │                                                                              │
│      │   ┌─────────────────────────────────────────────────────────────────────┐    │
│      │   │  SCHEDULER QUEUES                                                   │    │
│      │   │                                                                     │    │
│      │   │  waiting: [cmpl-xxx] ◄── New request goes here                      │    │
│      │   │  running: []                                                        │    │
│      │   │                                                                     │    │
│      │   │  Request waits until there's GPU memory and compute available       │    │
│      │   └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                              │
│      └── Request added to "waiting" queue                                           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: SCHEDULER CREATES BATCH                                                    │
│  File: vllm/v1/core/sched/scheduler.py                                              │
│  Function: Scheduler.schedule()                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_10] EngineCore.step() - calling scheduler.schedule()                     │
│      │                                                                              │
│  [REQ_FLOW_08] Scheduler.schedule() | waiting=1 | running=0                         │
│      │                                                                              │
│      │   ┌─────────────────────────────────────────────────────────────────────┐    │
│      │   │  BATCH CREATION - This is where vLLM's efficiency comes from        │    │
│      │   │                                                                     │    │
│      │   │  • Check which waiting requests can start (have KV cache space)     │    │
│      │   │  • Allocate KV cache blocks for new requests                        │    │
│      │   │  • Create SchedulerOutput with batch to execute                     │    │
│      │   │                                                                     │    │
│      │   │  NOTE: The scheduler is phase-agnostic. It outputs which            │    │
│      │   │  tokens/requests to run based on budgets and constraints.           │    │
│      │   │  "PREFILL" vs "DECODE" are labels for logging, not distinct modes.  │    │
│      │   └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                              │
│  [REQ_FLOW_09] Scheduler batch ready | num_new_reqs=1 | num_cached_reqs=0 | tok=6   │
│      │                                                                              │
│  [REQ_FLOW_11] EngineCore.step() - executing model | total_tokens=6                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: GPU WORKER EXECUTES BATCH                                                  │
│  File: vllm/v1/worker/gpu_worker.py                                                 │
│  Function: Worker.execute_model()                                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_13] Worker.execute_model() | rank=0 | local_rank=0 | total_tokens=6      │
│      │                                                                              │
│      │   ┌─────────────────────────────────────────────────────────────────────┐    │
│      │   │  GPU WORKER                                                         │    │
│      │   │                                                                     │    │
│      │   │  • Each worker handles one GPU                                      │    │
│      │   │  • For Tensor Parallelism (TP), multiple workers coordinate         │    │
│      │   │  • Calls model_runner.execute_model() to run the actual model       │    │
│      │   └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                              │
│      └── Calls GPUModelRunner.execute_model()                                       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 9: MODEL RUNNER PREPARES INPUTS                                               │
│  File: vllm/v1/worker/gpu_model_runner.py                                           │
│  Function: GPUModelRunner.execute_model()                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_14] GPUModelRunner.execute_model() | total_tokens=6                      │
│      │                                                                              │
│      ├── _update_states(): Update request state (new tokens, finished)              │
│      │                                                                              │
│      ├── _prepare_inputs(): Build input tensors                                     │
│      │       • input_ids: [15496, 11, 703, 527, 499, 30, pad, pad] → shape [8]      │
│      │       • positions: [0, 1, 2, 3, 4, 5, 6, 7]                                  │
│      │       • Prepare attention metadata (KV cache pointers, etc.)                 │
│      │                                                                              │
│      └── Ready to call model forward                                                │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                     ║
║  **STEP 10: NEURAL NETWORK FORWARD PASS**                                           ║
║  File: vllm/v1/worker/gpu_model_runner.py → model file                              ║
║                                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐    ║
║  │  vLLM labels a forward as "PREFILL" vs "DECODE" only as a log convenience:  │    ║
║  │  it uses whether SchedulerOutput contains any scheduled_new_reqs.            │    ║
║  │                                                                             │    ║
║  │  The forward processes *exactly the tokens scheduled this iteration*.       │    ║
║  │  For long prompts, this may be CHUNKED across multiple iterations.          │    ║
║  │                                                                             │    ║
║  │  Also note: tensors are often PADDED for batching / CUDA-graph stability,   │    ║
║  │  so logged shapes may be larger than the "real" token count.                │    ║
║  └─────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
│                                                                                     │
│  [REQ_FLOW_15] GPUModelRunner._model_forward() starting                             │
│               | phase=<PREFILL|DECODE>                                              │
│               | num_tokens=<num_tokens_padded> | new_reqs=<...> | cached_reqs=<...> │
│      │                                                                              │
│      └── self.model(input_ids, positions, ...)                                      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 11: MODEL FORWARD                                                             │
│  File: vllm/model_executor/models/deepseek_v2.py (or other model file)              │
│  Function: DeepseekV2ForCausalLM.forward()                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [MODEL_FLOW_01] DeepseekV2ForCausalLM.forward() | input_ids.shape=...              │
│      │                                                                              │
│      │   ┌─────────────────────────────────────────────────────────────────────┐    │
│      │   │  MODEL FORWARD (high level):                                        │    │
│      │   │                                                                     │    │
│      │   │  • Embedding: input_ids → hidden_states                             │    │
│      │   │  • Transformer layers: attention + MLP/MoE                          │    │
│      │   │  • Final norm                                                       │    │
│      │   │                                                                     │    │
│      │   │  NOTE: Hard-coded architecture numbers (layers/experts/top-k) are   │    │
│      │   │  configuration-dependent. Avoid treating them as universally true.  │    │
│      │   └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                              │
│      ├── 1. EMBEDDING: input_ids → hidden_states [num_tokens, hidden_size]          │
│      │                                                                              │
│      ├── 2. TRANSFORMER LAYERS (loop N times):                                      │
│      │       │                                                                      │
│      │       ├── Self-Attention (with KV cache)                                     │
│      │       │                                                                      │
│      │       └── MoE/MLP block ──────────────────────────► SEE STEP 12              │
│      │           (STEP 12 runs once per MoE layer = num_moe_layers times total)     │
│      │                                                                              │
│      └── 3. FINAL LAYER NORM                                                        │
│                                                                                     │
│  [MODEL_FLOW_04] DeepseekV2ForCausalLM.forward() complete | hidden_states.shape=... │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ (Inside each transformer layer with MoE)
                                              ▼
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                     ║
║  STEP 12: MoE (MIXTURE OF EXPERTS) LAYER - DETAILED                                 ║
║  File: vllm/model_executor/layers/fused_moe/layer.py                                ║
║                                                                                     ║
║  **This step runs num_moe_layers times per model forward pass.**                    ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                        MoE LAYER ARCHITECTURE                               │    │
│  │                                                                             │    │
│  │  Input: hidden_states [num_tokens, hidden_size]                             │    │
│  │                         │                                                   │    │
│  │                         ▼                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  STEP 12a: ROUTER (select_experts)                                  │    │    │
│  │  │                                                                     │    │    │
│  │  │  router_logits = gate(hidden_states) → [num_tokens, num_experts]    │    │    │
│  │  │                                                                     │    │    │
│  │  │  For each token, compute score for all experts                      │    │    │
│  │  └─────────────────────────────────────────────────────────────────────┘    │    │
│  │                         │                                                   │    │
│  │                         ▼                                                   │    │
│  │  [MOE_FLOW_01] FusedMoE.select_experts() | layer=... | num_tokens=...       │    │
│  │                         │                                                   │    │
│  │                         ▼                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  STEP 12b: TOP-K SELECTION                                          │    │    │
│  │  │                                                                     │    │    │
│  │  │  topk_weights, topk_ids = topk(scoring(router_logits), k=top_k)     │    │    │
│  │  │                                                                     │    │    │
│  │  │  Token 0: experts [23, 45, 12, ...] with weights [0.2, ...]         │    │    │
│  │  │  Token 1: experts [45, 23, 71, ...] with weights [0.3, ...]         │    │    │
│  │  │  ...                                                                │    │    │
│  │  └─────────────────────────────────────────────────────────────────────┘    │    │
│  │                         │                                                   │    │
│  │                         │  If EPLB enabled:                                 │    │
│  │                         ▼                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  STEP 12c: EPLB MAPPING (eplb_map_to_physical_and_record)           │    │    │
│  │  │                                                                     │    │    │
│  │  │  [MOE_FLOW_03] MoE calling EPLB | routing_decisions=num_tokens*top_k│    │    │
│  │  │  [MOE_FLOW_03a] topk_ids BEFORE (logical): [23, 45, 12, ...]        │    │    │
│  │  │                                                                     │    │    │
│  │  │  EPLB does TWO things:                                              │    │    │
│  │  │  1. Map logical expert ID → physical expert ID (for load balance)   │    │    │
│  │  │  2. Record load statistics (expert_load_view += 1 for each route)   │    │    │
│  │  │                                                                     │    │    │
│  │  │  [MOE_FLOW_03b] topk_ids AFTER (physical): [23, 78, 12, ...]        │    │    │
│  │  │  [MOE_FLOW_03c] expert_load_view updated: sum=...                   │    │    │
│  │  └─────────────────────────────────────────────────────────────────────┘    │    │
│  │                         │                                                   │    │
│  │                         ▼                                                   │    │
│  │  [MOE_FLOW_02] FusedMoE.forward_impl() | layer=... | use_ep=...             │    │
│  │                         │                                                   │    │
│  │                         │  If dispatch/combine path is active:              │    │
│  │                         ▼                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  STEP 12d: ALL2ALL DISPATCH (collective gather/broadcast)           │    │    │
│  │  │                                                                     │    │    │
│  │  │  [ALL2ALL_FLOW_01] dispatch() | hidden_states.shape=...             │    │    │
│  │  │                                                                     │    │    │
│  │  │  NOTE: In this codebase, dispatch() performs gather/broadcast       │    │    │
│  │  │  of tensors across ranks (implementation-dependent).                │    │    │
│  │  │                                                                     │    │    │
│  │  │  [ALL2ALL_FLOW_02] dispatch() complete | output_hidden.shape=...    │    │    │
│  │  └─────────────────────────────────────────────────────────────────────┘    │    │
│  │                         │                                                   │    │
│  │                         ▼                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  STEP 12e: EXPERT COMPUTATION (fused_experts kernel)                │    │    │
│  │  │                                                                     │    │    │
│  │  │  Each GPU computes ONLY its local experts:                          │    │    │
│  │  │                                                                     │    │    │
│  │  │  for each token:                                                    │    │    │
│  │  │    for each selected expert (if local to this GPU):                 │    │    │
│  │  │      gate = linear(hidden, gate_weights[expert])                    │    │    │
│  │  │      up   = linear(hidden, up_weights[expert])                      │    │    │
│  │  │      hidden = SiLU(gate) * up                                       │    │    │
│  │  │      output = linear(hidden, down_weights[expert])                  │    │    │
│  │  │                                                                     │    │    │
│  │  │  This is SPARSE computation - only selected experts run!            │    │    │
│  │  └─────────────────────────────────────────────────────────────────────┘    │    │
│  │                         │                                                   │    │
│  │                         ▼                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  STEP 12f: ALL2ALL COMBINE (collective reduce/scatter)              │    │    │
│  │  │                                                                     │    │    │
│  │  │  [ALL2ALL_FLOW_03] combine() | hidden_states.shape=...              │    │    │
│  │  │                                                                     │    │    │
│  │  │  NOTE: combine() is a collective communication primitive.           │    │    │
│  │  │  Per-token weighting by topk_weights happens in the MoE compute     │    │    │
│  │  │  path, not inside combine().                                        │    │    │
│  │  │                                                                     │    │    │
│  │  │  [ALL2ALL_FLOW_04] combine() complete | output.shape=...            │    │    │
│  │  └─────────────────────────────────────────────────────────────────────┘    │    │
│  │                         │                                                   │    │
│  │                         ▼                                                   │    │
│  │  Output: hidden_states [num_tokens, hidden_size] (same shape as input)      │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ (After all N layers complete)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 13: COMPUTE LOGITS & SAMPLE                                                   │
│  File: vllm/v1/worker/gpu_model_runner.py                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_16] GPUModelRunner._model_forward() complete                             │
│      │                                                                              │
│      ├── hidden_states [num_tokens, hidden_size] from model                         │
│      │                                                                              │
│      ├── compute_logits():                                                          │
│      │       logits = hidden_states[-1] @ lm_head_weights  → [1, vocab_size]        │
│      │       (Only compute logits for last token position per request)              │
│      │                                                                              │
│  [REQ_FLOW_16a] GPUModelRunner.sample_tokens() starting                             │
│      │                                                                              │
│      └── sample():                                                                  │
│              next_token = sample(logits, temperature, top_p, ...)                   │
│              → next_token_id = 358 ("I")                                            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 13a: EPLB STEP (after every model forward)                                    │
│  File: vllm/v1/worker/gpu_model_runner.py                                           │
│  Function: GPUModelRunner.eplb_step()                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [EPLB_FLOW_00] GPUModelRunner.eplb_step() called                                   │
│      │                                                                              │
│  [EPLB_FLOW_01] EplbState.step() | rearrangement_step=.../step_interval             │
│      │                                                                              │
│      ├── If is_dummy=False: Save expert_load_pass to window, reset pass             │
│      ├── Increment rearrangement_step counter                                       │
│      │                                                                              │
│      └── When rearrangement_step >= step_interval: TRIGGER REARRANGEMENT            │
│          [EPLB_FLOW_02] rearrange() starts                                          │
│          [EPLB_FLOW_03] policy computes new mapping                                 │
│          [EPLB_FLOW_04] weight transfer between GPUs                                │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 14: UPDATE SCHEDULER STATE                                                    │
│  File: vllm/v1/core/sched/scheduler.py                                              │
│  Function: Scheduler.update_from_output()                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_17] Scheduler.update_from_output() | num_requests=1                      │
│      │                                                                              │
│      ├── Extract sampled token IDs from model output                                │
│      ├── Append new token to request's output_token_ids                             │
│      ├── Check finish conditions:                                                   │
│      │       • EOS token generated?                                                 │
│      │       • max_tokens reached?                                                  │
│      │       • Stop string found?                                                   │
│      │                                                                              │
│      └── If stopped: mark finished and free request resources (KV blocks, etc.)     │
│                                                                                     │
│  NOTE: Requests move waiting → running during schedule(), when KV slots are         │
│  successfully allocated (not in update_from_output).                                │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ ═══════ ITERATION COMPLETE ═══════
                                              │
                                              │ EngineCore busy loop continues...
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  DECODE-LIKE ITERATION (example: generating token 1)                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  [REQ_FLOW_10] EngineCore.step() - calling scheduler.schedule()                     │
│  [REQ_FLOW_08] Scheduler.schedule() | waiting=0 | running=1                         │
│  [REQ_FLOW_09] Scheduler batch | num_new_reqs=0 | num_cached_reqs=1 | total_tok=1   │
│  [REQ_FLOW_11] EngineCore.step() - executing model | total_tokens=1                 │
│  [REQ_FLOW_13] Worker.execute_model() | total_tokens=1                              │
│  [REQ_FLOW_14] GPUModelRunner.execute_model() | total_tokens=1                      │
│  [REQ_FLOW_15] GPUModelRunner._model_forward() | phase=DECODE | num_tokens=1        │
│               | new_reqs=0 | cached_reqs=1                                          │
│                                                                                     │
│      │   ┌─────────────────────────────────────────────────────────────────────┐    │
│      │   │  DECODE FORWARD PASS (this is STEP 10-11):                          │    │
│      │   │                                                                     │    │
│      │   │  input_ids: [358]  (just the new token, e.g., "I")                  │    │
│      │   │  positions: [6]    (next position after prompt)                     │    │
│      │   │                                                                     │    │
│      │   │  Through N transformer layers:                                      │    │
│      │   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│      │   │  │  For each layer:                                              │  │    │
│      │   │  │    • Attention: reads from KV cache, writes new position      │  │    │
│      │   │  │    • MoE/MLP: if MoE layer → **STEP 12 runs here**            │  │    │
│      │   │  │                (router → top-k → EPLB map → dispatch →        │  │    │
│      │   │  │                 expert compute → combine)                     │  │    │
│      │   │  └───────────────────────────────────────────────────────────────┘  │    │
│      │   │                                                                     │    │
│      │   │  STEP 12 runs **num_moe_layers times** per forward pass.            │    │
│      │   │                                                                     │    │
│      │   │  Output: logits for next token                                      │    │
│      │   │  Sample: next_token = "am" (token_id=716)                           │    │
│      │   └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                              │
│  [REQ_FLOW_16] GPUModelRunner._model_forward() complete                             │
│  [REQ_FLOW_16a] GPUModelRunner.sample_tokens() starting                             │
│  [REQ_FLOW_17] Scheduler.update_from_output() | num_requests=1                      │
│                                                                                     │
│          ... (this whole iteration repeats until max_tokens / EOS / stop) ...       │
│                                                                                     │
│  Generated so far: "I am good. How about you?\n\nAssistant: I am good..."           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ Output sent back via ZMQ (streaming)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 15: OUTPUT PROCESSING (back to API Server)                                    │
│  File: vllm/v1/engine/async_llm.py                                                  │
│  Function: output_handler()                                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│      │   ┌─────────────────────────────────────────────────────────────────────┐    │
│      │   │  IPC BOUNDARY - ZeroMQ Socket (OUTPUT DIRECTION)                    │    │
│      │   │                                                                     │    │
│      │   │  EngineCore Process  ────────────────►  API Server Process          │    │
│      │   │  (generated tokens)                      (detokenize & respond)     │    │
│      │   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│  [REQ_FLOW_18] AsyncLLM output_handler received | num_outputs=1                     │
│      │                                                                              │
│  [REQ_FLOW_18a] OutputProcessor.process_outputs() | num_outputs=1 | finished=0      │
│      │                                                                              │
│      ├── Detokenize: [358, 716, ...] → "I am good..."                               │
│      ├── Put RequestOutput into queue for generate() to yield                       │
│      │                                                                              │
│      │   (... multiple iterations as tokens stream in ...)                          │
│      │                                                                              │
│  [REQ_FLOW_18a] OutputProcessor.process_outputs() | num_outputs=1 | finished=1      │
│      │                                                                              │
│  [REQ_FLOW_19] Request complete | request_id=cmpl-xxx                               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  FINAL: HTTP RESPONSE TO CLIENT                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  HTTP 200 OK                                                                        │
│  {                                                                                  │
│    "id": "cmpl-xxx",                                                                │
│    "choices": [{                                                                    │
│      "text": "I am good. How about you?\n\nAssistant: ...",                         │
│      "finish_reason": "length"                                                      │
│    }],                                                                              │
│    "usage": {"prompt_tokens": 6, "completion_tokens": 50, "total_tokens": 56}       │
│  }                                                                                  │
│                                                                                     │
│  Client receives response!                                                          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Explanation

### STEP 1: HTTP REQUEST ENTRY

**File:** `vllm/entrypoints/openai/api_server.py`  
**Function:** `create_completion()`

**What happens:**
- FastAPI receives an HTTP POST request to `/v1/completions` or `/v1/chat/completions`
- The request body contains the prompt, model name, and generation parameters (max_tokens, temperature, etc.)
- FastAPI validates the request format and routes it to the appropriate handler

**Code comment:**
```python
# STEP 1: HTTP REQUEST ENTRY
# [REQ_FLOW_01] - Entry point for OpenAI-compatible completion requests
# FastAPI receives the HTTP POST and routes to this handler
async def create_completion(request: CompletionRequest, ...):
    # Validate request and dispatch to serving layer
    ...
```

---

### STEP 2: REQUEST VALIDATION & PREPARATION

**File:** `vllm/entrypoints/openai/serving_completion.py`  
**Function:** `OpenAIServingCompletion.create_completion()`

**What happens:**
- Validates that the requested model is loaded
- Creates `SamplingParams` from the request (temperature, top_p, max_tokens, stop strings, etc.)
- Normalizes/validates the prompt format
- Calls the engine client to start generation

**Code comment:**
```python
# STEP 2: REQUEST VALIDATION & PREPARATION
# [REQ_FLOW_02] - Validate model, create sampling params
# This step validates the request but does NOT tokenize yet
async def create_completion(self, request: CompletionRequest, ...):
    # STEP 2a: Validate model exists
    # STEP 2b: Create SamplingParams from request parameters
    # STEP 2c: Call engine_client.generate()
    ...
```

---

### STEP 3: ASYNC ENGINE ENTRY

**File:** `vllm/v1/engine/async_llm.py`  
**Function:** `AsyncLLM.generate()`

**What happens:**
- Main entry point to the vLLM engine
- Creates a `RequestOutputCollector` to receive outputs
- Starts a background task (`output_handler`) to process outputs
- Calls `add_request()` which:
  - Calls `InputProcessor.process_inputs()` to tokenize and validate
  - Mutates the request_id to include a random suffix for uniqueness
  - Creates an `EngineCoreRequest` with all request data

**Code comment:**
```python
# STEP 3: ASYNC ENGINE ENTRY
# [REQ_FLOW_03] - Main entry point to vLLM engine
async def generate(self, request_id: str, ...):
    # STEP 3a: Create output collector
    # STEP 3b: Start output_handler background task
    # STEP 3c: Call add_request() which does tokenization
    ...

# Inside add_request():
# [REQ_FLOW_03a] - Input processing happens HERE (not in STEP 2)
# InputProcessor.process_inputs():
#   - STEP 3c.1: Tokenize prompt
#   - STEP 3c.2: Validate sampling params
#   - STEP 3c.3: Create EngineCoreRequest
#   - STEP 3c.4: assign_request_id() - MUTATES request_id (adds -<8chars>)
```

---

### STEP 4: ZMQ SEND TO ENGINE CORE

**File:** `vllm/v1/engine/core_client.py`  
**Function:** `EngineCoreClient.add_request_async()`

**What happens:**
- Serializes the `EngineCoreRequest` 
- Sends it via ZeroMQ to the EngineCore process
- For Data Parallel (DP) deployments, routes to one of multiple EngineCores based on load balancing

**Code comment:**
```python
# STEP 4: ZMQ SEND TO ENGINE CORE
# [REQ_FLOW_04a/04b] - Send request via ZMQ to EngineCore
# This crosses an IPC boundary - may be a separate process
async def add_request_async(self, request: EngineCoreRequest):
    # STEP 4a: Serialize request
    # STEP 4b: Send via ZMQ socket
    # STEP 4c: For DPLB: select EngineCore based on dp_rank
    ...

# [REQ_FLOW_04] logged after send completes
```

---

### STEP 5: ENGINE CORE RECEIVES REQUEST

**File:** `vllm/v1/engine/core.py`  
**Function:** `EngineCore._handle_client_request()`

**What happens:**
- EngineCore receives the ZMQ message in its busy loop
- Deserializes the request
- Identifies it as an ADD request (vs ABORT)
- Dispatches to scheduler

**Code comment:**
```python
# STEP 5: ENGINE CORE RECEIVES REQUEST
# [REQ_FLOW_05] - EngineCore receives ADD request via ZMQ
# [REQ_FLOW_06] - Dispatch to scheduler
def _handle_client_request(self, request_type, request):
    # STEP 5a: Deserialize request from ZMQ
    # STEP 5b: Check request type (ADD or ABORT)
    if request_type == ADD:
        # STEP 5c: Dispatch to scheduler.add_request()
        self.scheduler.add_request(request)
```

---

### STEP 6: SCHEDULER ADDS REQUEST TO QUEUE

**File:** `vllm/v1/core/sched/scheduler.py`  
**Function:** `Scheduler.add_request()`

**What happens:**
- Creates a `Request` object from the `EngineCoreRequest`
- Adds it to the `waiting` queue
- Request will stay in waiting until KV cache space is available

**Code comment:**
```python
# STEP 6: SCHEDULER ADDS REQUEST TO QUEUE
# [REQ_FLOW_07] - Add request to waiting queue
def add_request(self, request: EngineCoreRequest):
    # STEP 6a: Create Request object
    # STEP 6b: Add to self.waiting queue
    # Request waits here until schedule() allocates KV blocks
    self.waiting.append(request)
```

---

### STEP 7: SCHEDULER CREATES BATCH

**File:** `vllm/v1/core/sched/scheduler.py`  
**Function:** `Scheduler.schedule()`

**What happens:**
- Called by `EngineCore.step()` each iteration
- Checks which requests can run (have KV cache space, within token budget)
- Moves requests from `waiting` to `running` when KV allocated
- Creates `SchedulerOutput` describing which tokens to process

**Important:** The scheduler is **phase-agnostic** - it doesn't have explicit "prefill" or "decode" modes. It just schedules tokens based on budgets.

**Code comment:**
```python
# STEP 7: SCHEDULER CREATES BATCH
# [REQ_FLOW_10] - EngineCore.step() calls schedule()
# [REQ_FLOW_08] - Scheduler.schedule() starts
# [REQ_FLOW_09] - Batch ready
def schedule(self) -> SchedulerOutput:
    # STEP 7a: Check running requests (can they continue?)
    # STEP 7b: Try to start waiting requests (allocate KV blocks)
    #   - Requests move waiting → running HERE when KV allocated
    # STEP 7c: Build SchedulerOutput with batch info
    # NOTE: No explicit "prefill" or "decode" mode - scheduler is phase-agnostic
    ...
```

---

### STEP 8: GPU WORKER EXECUTES BATCH

**File:** `vllm/v1/worker/gpu_worker.py`  
**Function:** `Worker.execute_model()`

**What happens:**
- Worker receives the `SchedulerOutput`
- For Tensor Parallelism, coordinates with other workers
- Calls `GPUModelRunner.execute_model()`

**Code comment:**
```python
# STEP 8: GPU WORKER EXECUTES BATCH
# [REQ_FLOW_13] - Worker.execute_model()
def execute_model(self, scheduler_output: SchedulerOutput):
    # STEP 8a: Receive scheduler output
    # STEP 8b: For TP: coordinate with other workers
    # STEP 8c: Call model_runner.execute_model()
    return self.model_runner.execute_model(scheduler_output)
```

---

### STEP 9: MODEL RUNNER PREPARES INPUTS

**File:** `vllm/v1/worker/gpu_model_runner.py`  
**Function:** `GPUModelRunner.execute_model()`

**What happens:**
- Updates internal request states
- Prepares input tensors (`input_ids`, `positions`, attention metadata)
- Handles padding for CUDA graphs

**Code comment:**
```python
# STEP 9: MODEL RUNNER PREPARES INPUTS
# [REQ_FLOW_14] - GPUModelRunner.execute_model()
def execute_model(self, scheduler_output: SchedulerOutput):
    # STEP 9a: _update_states() - update request state
    # STEP 9b: _prepare_inputs() - build tensors:
    #   - input_ids: token IDs to process
    #   - positions: position indices
    #   - attention metadata (KV cache pointers, etc.)
    # STEP 9c: Tensors may be PADDED for CUDA graph stability
    ...
```

---

### STEP 10: NEURAL NETWORK FORWARD PASS

**File:** `vllm/v1/worker/gpu_model_runner.py`  
**Function:** `GPUModelRunner._model_forward()`

**What happens:**
- Calls `self.model(input_ids, positions, ...)`
- The "phase" (PREFILL/DECODE) is just a label based on whether there are new requests
- Processes exactly the tokens scheduled this iteration

**Code comment:**
```python
# STEP 10: NEURAL NETWORK FORWARD PASS
# [REQ_FLOW_15] - _model_forward() starting
def _model_forward(self, ...):
    # STEP 10a: Determine phase label (PREFILL if new_reqs > 0, else DECODE)
    #   NOTE: This is just for logging, scheduler is phase-agnostic
    # STEP 10b: Call model forward
    hidden_states = self.model(input_ids, positions, ...)
    # STEP 10c: hidden_states shape may be padded
```

---

### STEP 11: MODEL FORWARD

**File:** `vllm/model_executor/models/deepseek_v2.py` (or other model file)  
**Function:** `DeepseekV2ForCausalLM.forward()`

**What happens:**
- Embedding: `input_ids` → `hidden_states`
- Loop through N transformer layers:
  - Self-Attention (with KV cache read/write)
  - MoE/MLP block (**STEP 12 runs here for each MoE layer**)
- Final layer norm

**Code comment:**
```python
# STEP 11: MODEL FORWARD
# [MODEL_FLOW_01] - DeepseekV2ForCausalLM.forward() entry
def forward(self, input_ids, positions, ...):
    # STEP 11a: EMBEDDING
    hidden_states = self.embed_tokens(input_ids)
    
    # STEP 11b: TRANSFORMER LAYERS (loop N times)
    for layer in self.layers:
        # STEP 11b.1: Self-Attention
        hidden_states = layer.attention(hidden_states, ...)
        # STEP 11b.2: MoE/MLP - **STEP 12 runs here for MoE layers**
        hidden_states = layer.mlp(hidden_states)  # or layer.moe(...)
    
    # STEP 11c: FINAL LAYER NORM
    hidden_states = self.norm(hidden_states)
    
# [MODEL_FLOW_04] - forward() complete
```

---

### STEP 12: MoE (MIXTURE OF EXPERTS) LAYER

**File:** `vllm/model_executor/layers/fused_moe/layer.py`  
**Function:** `FusedMoE.forward_impl()`

**This step runs `num_moe_layers` times per model forward pass.**

#### STEP 12a: ROUTER

**What happens:**
- Router (gating network) computes scores for each expert
- `router_logits = gate(hidden_states)` → shape `[num_tokens, num_experts]`

**Code comment:**
```python
# STEP 12a: ROUTER
# Compute expert scores for each token
router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]
```

#### STEP 12b: TOP-K SELECTION

**What happens:**
- Select top-k experts per token
- `topk_weights, topk_ids = topk(scoring(router_logits), k=top_k)`

**Code comment:**
```python
# STEP 12b: TOP-K SELECTION
# [MOE_FLOW_01] - select_experts()
topk_weights, topk_ids = select_experts(
    router_logits, 
    top_k=self.top_k
)
# topk_ids: [num_tokens, top_k] - LOGICAL expert IDs
# topk_weights: [num_tokens, top_k] - routing weights
```

#### STEP 12c: EPLB MAPPING (if enabled)

**What happens:**
- Map logical expert IDs → physical expert IDs (for load balancing)
- Record load statistics (`expert_load_view += 1` for each routing decision)

**Code comment:**
```python
# STEP 12c: EPLB MAPPING
# [MOE_FLOW_03] - MoE calling EPLB
if self.enable_eplb:
    # [MOE_FLOW_03a] - topk_ids BEFORE (logical)
    topk_ids = eplb_map_to_physical_and_record(
        topk_ids,                    # [num_tokens, top_k] logical IDs
        expert_load_view,            # [num_physical_experts] accumulator
        logical_to_physical_map,     # [num_logical_experts, max_slots]
        logical_replica_count,       # [num_logical_experts]
    )
    # [MOE_FLOW_03b] - topk_ids AFTER (physical)
    # [MOE_FLOW_03c] - expert_load_view updated
```

#### STEP 12d: ALL2ALL DISPATCH

**What happens:**
- Collective gather/broadcast tensors across ranks
- Implementation-dependent (NaiveAll2AllManager, AgRsAll2AllManager)

**Code comment:**
```python
# STEP 12d: ALL2ALL DISPATCH
# [ALL2ALL_FLOW_01] - dispatch() starting
dispatched = self.all2all_manager.dispatch(
    hidden_states, 
    router_logits,
    ...
)
# [ALL2ALL_FLOW_02] - dispatch() complete
# NOTE: This is gather/broadcast across ranks, not point-to-point routing
```

#### STEP 12e: EXPERT COMPUTATION

**What happens:**
- Each GPU computes only its local experts
- Sparse computation: only selected experts run for each token
- Expert FFN: `Linear → SiLU → Linear`

**Code comment:**
```python
# STEP 12e: EXPERT COMPUTATION
# Each GPU computes ONLY its local experts
# This is SPARSE: only top_k experts run per token
for token in tokens:
    for expert_id in token.selected_experts:
        if expert_is_local(expert_id):
            # Expert FFN computation
            gate = linear(hidden, gate_weights[expert_id])
            up = linear(hidden, up_weights[expert_id])
            hidden = silu(gate) * up
            output = linear(hidden, down_weights[expert_id])
```

#### STEP 12f: ALL2ALL COMBINE

**What happens:**
- Collective reduce/scatter results back
- Per-token weighting by `topk_weights` happens in MoE compute path, not in `combine()`

**Code comment:**
```python
# STEP 12f: ALL2ALL COMBINE
# [ALL2ALL_FLOW_03] - combine() starting
output = self.all2all_manager.combine(
    expert_outputs,
    ...
)
# [ALL2ALL_FLOW_04] - combine() complete
# NOTE: Per-token weighting happens in MoE compute, not here
```

---

### STEP 13: COMPUTE LOGITS & SAMPLE

**File:** `vllm/v1/worker/gpu_model_runner.py`  
**Function:** `GPUModelRunner.sample_tokens()`

**What happens:**
- Compute logits from hidden states (only for last position per request)
- Sample next token using temperature, top_p, etc.

**Code comment:**
```python
# STEP 13: COMPUTE LOGITS & SAMPLE
# [REQ_FLOW_16] - _model_forward() complete
# [REQ_FLOW_16a] - sample_tokens() starting

def sample_tokens(self, ...):
    # STEP 13a: Compute logits
    # Only compute for last token position per request
    logits = hidden_states @ self.lm_head.weight.T
    
    # STEP 13b: Sample next token
    next_token = self.sampler.sample(
        logits,
        sampling_params,  # temperature, top_p, etc.
    )
```

---

### STEP 13a: EPLB STEP

**File:** `vllm/v1/worker/gpu_model_runner.py`  
**Function:** `GPUModelRunner.eplb_step()`

**What happens:**
- Called after every model forward
- Saves load statistics to sliding window (if not dummy step)
- When counter reaches `step_interval`: triggers rearrangement

**Code comment:**
```python
# STEP 13a: EPLB STEP
# [EPLB_FLOW_00] - GPUModelRunner.eplb_step() called
def eplb_step(self, is_dummy: bool = False):
    self.eplb_state.step(is_dummy)

# Inside EplbState.step():
# [EPLB_FLOW_01] - EplbState.step()
def step(self, is_dummy: bool):
    # STEP 13a.1: If not dummy, save load to window
    if not is_dummy:
        expert_load_window[window_step] = expert_load_pass.clone()
        expert_load_pass.zero_()
        window_step = (window_step + 1) % window_size
    
    # STEP 13a.2: Increment counter
    rearrangement_step += 1
    
    # STEP 13a.3: Check if rearrangement needed
    if rearrangement_step >= step_interval:
        # [EPLB_FLOW_02] - rearrange() starts
        # [EPLB_FLOW_03] - policy computes new mapping
        # [EPLB_FLOW_04] - weight transfer
        self.rearrange()
```

---

### STEP 14: UPDATE SCHEDULER STATE

**File:** `vllm/v1/core/sched/scheduler.py`  
**Function:** `Scheduler.update_from_output()`

**What happens:**
- Extract sampled token IDs from model output
- Append new tokens to request's output
- Check finish conditions (EOS, max_tokens, stop string)
- Free resources for finished requests

**Code comment:**
```python
# STEP 14: UPDATE SCHEDULER STATE
# [REQ_FLOW_17] - Scheduler.update_from_output()
def update_from_output(self, model_output):
    for request_id, output in model_output.items():
        # STEP 14a: Get sampled token
        new_token = output.sampled_token_id
        
        # STEP 14b: Append to request's output
        request.output_token_ids.append(new_token)
        
        # STEP 14c: Check finish conditions
        if self.should_stop(request, new_token):
            # STEP 14d: Mark finished, free KV blocks
            self.free_request(request)
```

---

### STEP 15: OUTPUT PROCESSING

**File:** `vllm/v1/engine/async_llm.py`  
**Function:** `output_handler()`

**What happens:**
- Receives outputs via ZMQ from EngineCore
- Detokenizes token IDs to text
- Puts `RequestOutput` into queue for `generate()` to yield
- Handles streaming outputs

**Code comment:**
```python
# STEP 15: OUTPUT PROCESSING
# [REQ_FLOW_18] - AsyncLLM output_handler received
async def output_handler(self, ...):
    while True:
        # STEP 15a: Receive outputs via ZMQ
        outputs = await self.engine_client.get_outputs()
        
        # [REQ_FLOW_18a] - OutputProcessor.process_outputs()
        for output in outputs:
            # STEP 15b: Detokenize
            text = self.tokenizer.decode(output.token_ids)
            
            # STEP 15c: Put in queue for generate() to yield
            self.output_queue.put(RequestOutput(...))
            
            # STEP 15d: Check if finished
            if output.finished:
                # [REQ_FLOW_19] - Request complete
                break
```

---

## EPLB (Expert Parallelism Load Balancer) Background Flow

EPLB runs **synchronously** within the model runner (not in parallel), collecting load statistics from MoE layers and periodically rebalancing expert distribution.

### Key Data Structures

```python
# Per-model EPLB state
expert_load_pass     # [num_moe_layers, num_physical_experts] - current step accumulator
expert_load_window   # [window_size, num_moe_layers, num_physical_experts] - history

# Mapping tables (can differ per layer!)
physical_to_logical_map   # [num_moe_layers, num_physical_experts]
logical_to_physical_map   # [num_moe_layers, num_logical_experts, max_slots]
logical_replica_count     # [num_moe_layers, num_logical_experts]
```

### EPLB Interaction Points

1. **MoE → EPLB** (during forward): `eplb_map_to_physical_and_record()` maps IDs and records load
2. **EPLB → MoE** (during rearrangement): `rearrange_expert_weights_inplace()` moves weights and updates maps

### When Rearrangement Happens

```python
# rearrangement_step is initialized to ~3/4 of step_interval at startup
# (so first rearrangement happens sooner)

# When rearrangement_step >= step_interval:
# 1. Aggregate load across window
# 2. All-reduce across EP ranks
# 3. Policy computes new placement
# 4. Transfer weights between GPUs
# 5. Update mapping tables
# 6. Reset counter (rearrangement_step = 0)
```

---

## Trace Markers Reference

| Marker | File | Description |
|--------|------|-------------|
| `REQ_FLOW_01` | `api_server.py` | HTTP request received |
| `REQ_FLOW_02` | `serving_completion.py` | Request validation |
| `REQ_FLOW_03` | `async_llm.py` | AsyncLLM.generate() entry |
| `REQ_FLOW_03a` | `input_processor.py` | Input processing (tokenization) |
| `REQ_FLOW_04a/b` | `core_client.py` | ZMQ send |
| `REQ_FLOW_05` | `core.py` | EngineCore receives |
| `REQ_FLOW_06` | `core.py` | Dispatch to scheduler |
| `REQ_FLOW_07` | `scheduler.py` | Add to waiting queue |
| `REQ_FLOW_08` | `scheduler.py` | schedule() starts |
| `REQ_FLOW_09` | `scheduler.py` | Batch ready |
| `REQ_FLOW_10` | `core.py` | step() calling schedule |
| `REQ_FLOW_11` | `core.py` | step() executing model |
| `REQ_FLOW_13` | `gpu_worker.py` | Worker.execute_model() |
| `REQ_FLOW_14` | `gpu_model_runner.py` | GPUModelRunner.execute_model() |
| `REQ_FLOW_15` | `gpu_model_runner.py` | _model_forward() starting |
| `REQ_FLOW_16` | `gpu_model_runner.py` | _model_forward() complete |
| `REQ_FLOW_16a` | `gpu_model_runner.py` | sample_tokens() starting |
| `REQ_FLOW_17` | `scheduler.py` | update_from_output() |
| `REQ_FLOW_18` | `async_llm.py` | output_handler received |
| `REQ_FLOW_18a` | `output_processor.py` | process_outputs() |
| `REQ_FLOW_19` | `async_llm.py` | Request complete |
| `MODEL_FLOW_01` | `deepseek_v2.py` | Model forward entry |
| `MODEL_FLOW_04` | `deepseek_v2.py` | Model forward complete |
| `MOE_FLOW_01` | `layer.py` | select_experts() |
| `MOE_FLOW_02` | `layer.py` | forward_impl() |
| `MOE_FLOW_03` | `layer.py` | EPLB mapping called |
| `EPLB_FLOW_00` | `gpu_model_runner.py` | eplb_step() called |
| `EPLB_FLOW_01` | `eplb_state.py` | EplbState.step() |
| `EPLB_FLOW_02` | `eplb_state.py` | rearrange() starting |
| `EPLB_FLOW_03` | `policy/default.py` | Policy computing |
| `EPLB_FLOW_04` | `rebalance_execute.py` | Weight transfer |
| `ALL2ALL_FLOW_01` | `all2all.py` | dispatch() starting |
| `ALL2ALL_FLOW_02` | `all2all.py` | dispatch() complete |
| `ALL2ALL_FLOW_03` | `all2all.py` | combine() starting |
| `ALL2ALL_FLOW_04` | `all2all.py` | combine() complete |
