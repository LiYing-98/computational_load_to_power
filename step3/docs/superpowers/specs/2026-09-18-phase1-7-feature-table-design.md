# Phase 1.7 Complete Ex-Ante Feature Table Design

## Authority and scope

This design implements the user-authorized taskbook `Codex_Phase1.7任务书_完整事前特征补齐与建模输入表定稿.docx`. The user's direct request overrides the taskbook only for the output location: all new artifacts live under `step3/` in the existing local repository.

The population is exactly the 694 ML.ENERGY V3 text-LLM inference runs from `gpqa`, `lm-arena-chat`, and `sourcegraph-fim`; the primary-stable subset remains 565 runs. Phase 1.7 ends after feature completion, quality audit, and modeling-readiness judgment. It must not train Ridge, XGBoost, neural networks, or any other prediction model, and it must not expand to Training, Diffusion, CMU, NLR, MIT, full raw timelines, Prometheus, or model weights.

## Source hierarchy

1. Phase 1.5 run table and static benchmark fields.
2. Phase 1.6 planned-workload table, pinned tokenizer revisions, bounded result metadata, and validation outputs.
3. ML.ENERGY benchmark source pinned to commit `c1d6557fbf7c7c495749b0fcf649f71774eabde9`.
4. GPQA pinned to dataset revision `633f5ee89ab8ad4522a9f850766b73f62147ffdd`, subset `gpqa_diamond`, split `train`, seed `48105`.
5. Model repositories at a versioned revision manifest. Only config, tokenizer, vocabulary, and chat-template files may be fetched. Weight formats such as `.safetensors`, `.bin`, `.pt`, `.gguf`, and `.onnx` are denied.
6. vLLM OpenAI server source matching the benchmark default image `vllm/vllm-openai:v0.11.1`, used only to verify the Chat Completions message-normalization and template semantics.

GPQA and Google Gemma access may be used through the user's authorized Hugging Face credential. Meta Llama access is treated as rejected: retain its runs and public benchmark fields, but do not use mirrors or substitute tokenizers/configs. Credentials and gated raw data remain under ignored `step3/data/` and never enter Git.

## Architecture

Phase 1.7 is a deterministic, staged pipeline with versioned manifests and independently testable transforms:

1. `acquire_static_sources.py` copies or fetches only pinned lightweight inputs and records hashes.
2. `reconstruct_planned_workload.py` reuses the Phase 1.6 implementation and cached successes, adding newly authorized GPQA and Gemma coverage without changing request order or prompt construction.
3. `build_effective_workload.py` reconstructs the exact OpenAI Chat messages or FIM prompt and computes effective token lengths without overwriting benchmark prompt lengths.
4. `build_model_features.py` maps pinned model configs into a canonical schema while retaining source fields and mapping status.
5. `build_physics_features.py` computes explicit theoretical prefill, planned decode-upper-bound, KV-cache, weight-size, and per-GPU proxies.
6. `build_master_table.py` joins all one-row-per-run blocks and derives eligibility flags.
7. `validate_phase17.py`, unit tests, and an executed notebook independently check coverage, keys, missingness, leakage, formulas, and output claims.

Each table carries source revision/hash and status/reason fields. Missing and unresolved are never converted to zero and never filled with a different model's values.

## Workload semantics

### Benchmark prompt length

The existing `planned_*` columns retain ML.ENERGY `SampleRequest.prompt_len` semantics. Phase 1.7 additionally aliases the aggregate family with an explicit `benchmark_*` prefix in the final master table. The original Phase 1.6 columns remain available for backward compatibility.

### Effective input length

For `gpqa`, the benchmark prompt string is wrapped as one OpenAI Chat `user` message. For `lm-arena-chat`, the benchmark's list is mapped to alternating `user`, `assistant`, `user`, … messages exactly as in pinned `benchmark.py`. Any pinned system prompt is prepended. Text-part payloads are normalized according to the pinned vLLM OpenAI parser, then the pinned model tokenizer applies its own chat template with `tokenize=True` and `add_generation_prompt=True`.

A Chat pair is `resolved` only when the model revision, tokenizer files, chat template, benchmark message construction, and vLLM normalization rule are all available and template application succeeds. Otherwise the pair is `unresolved` with a specific reason; benchmark length remains intact.

For `sourcegraph-fim`, effective length is the token count of the official pinned `render_fim_prompt` output with that model's pinned tokenizer. No Chat template is applied. This should equal benchmark prompt length and is validated as an invariant.

The effective table contains the requested total, mean, p50, p90, p95, min, max, standard deviation, and sum-of-squares fields, scaled by `num_request_repeats` where additive. It also records base-request count, unique-input count, vector hash, template hash, tokenizer revision, status, and reason.

## Canonical model schema

The model table is one row per `model_id` and includes:

- identity/provenance: model ID, pinned config revision, config SHA-256, config class/model type, mapping status, missing reason;
- canonical architecture: architecture family, hidden layers, hidden size, intermediate size, attention heads, KV heads, head dimension, vocabulary size, context length;
- MoE: local/total experts, active experts/top-k, MoE intermediate size, shared-expert fields when present;
- hybrid-specific source fields: state-space/Mamba layer counts and dimensions where available;
- benchmark metadata: total/activated parameters and actual benchmark weight precision, which must not be overwritten by repository dtype.

Every canonical value has a `source_field` entry in a long-form mapping table. Deterministic derivations such as `head_dim = hidden_size / num_attention_heads` are labeled `derived` and rejected when divisibility fails. Dense Transformer, MoE Transformer, and Hybrid are assigned only from config evidence; unsupported fields remain null.

## Physics-derived features

All FLOP values are proxies for decoder-only inference and exclude embeddings, normalization, activation functions, routing overhead, communication, and the LM-head unless separately stated. Let `L` be effective input length, `d` hidden size, `h_q` query heads, `h_kv` KV heads, `d_h` head dimension, `d_ff` dense or expert intermediate size, `k` active experts, and `N` layers.

Per-layer prefill projection FLOPs:

`2*L*d*(h_q*d_h) + 4*L*d*(h_kv*d_h) + 2*L*(h_q*d_h)*d`

Per-layer causal-attention matmul FLOPs:

`4*h_q*d_h*L^2`

Dense gated-MLP FLOPs use `6*L*d*d_ff`; non-gated MLP uses `4*L*d*d_ff`. MoE routed MLP uses the gated/non-gated formula multiplied by active experts `k`, plus a separately evidenced shared-expert term. Total prefill proxy multiplies the sum by `N`.

Decode upper bound uses each request's configured `max_output_tokens=O`. Projection and MLP costs are evaluated for `O` tokens; attention uses `sum_{t=0}^{O-1}(L+t)`. It is explicitly named `theoretical_decode_flops_upper_bound` and never interpreted as expected or actual work.

KV-cache bytes per sequence are `2*N*h_kv*d_h*sequence_length*bytes_per_element`. Phase 1.7 reports input-only and input-plus-output-cap upper-bound variants. Weight bytes are `total_params * precision_bits / 8`. Per-GPU weight proxy divides by the model-parallel group size `num_gpus / data_parallel` when integral and positive. Per-GPU work proxy divides aggregate theoretical work by `num_gpus` and is labeled as an idealized balance, not a measured allocation.

Physics eligibility requires resolved effective workload plus the architecture fields needed by the applicable formula. Hybrid or unsupported architectures remain in the master table but are not forced through a pure-Transformer formula.

## Master table and leakage boundary

`master_feature_table.parquet` has exactly one row per Phase 1.5 run key. Columns are grouped by prefixes and documented as:

- `X_workload`: planned/benchmark/effective lengths, requests, output cap, arrival/protocol descriptors;
- `X_model`: canonical static structure and benchmark parameter/precision fields;
- `X_hardware`: GPU model/count and static theoretical specifications included in scope;
- `X_deployment`: TP/EP/DP and `max_num_seqs`;
- `X_protocol`: task, endpoint, repeat, and benchmark controls;
- `Y`: GPU steady-state energy/power labels only;
- diagnostics/provenance: model ID, stability, result path, status/reason fields, post-run comparison fields.

`model_id` and nickname are diagnostics/split keys, not default primary-model features. Post-run fields such as actual output tokens, throughput, latency, actual batch, duration, utilization, clock, temperature, stability reason, and completion count are denied from the X allowlist.

Eligibility flags are deterministic:

- `eligible_base_exante`: required base X fields and GPU target are present; Llama can qualify;
- `eligible_full_workload`: benchmark and reliable effective workload blocks are complete;
- `eligible_physics_feature`: full workload plus all required canonical structure fields and physics proxies are present;
- `eligible_primary_stable`: Phase 1/1.5 stability rule is true;
- `restricted_feature_reason`: ordered, semicolon-separated reasons such as `llama_access_rejected`, `tokenizer_unavailable`, `chat_template_unresolved`, or `config_missing`.

## RAPL time box

The pipeline performs a bounded search of already downloaded result metadata, Phase 1/1.5/1.6 artifacts, pinned Zeus source, and any existing lightweight logs. It does not download full timelines or broad raw logs. A zone name is reported only when a direct string mapping is found. Otherwise zone 1 remains `platform-like diagnostic` and the unresolved result does not block Phase 1.7.

## Acceptance criteria

- 694 unique runs and 565 stable runs reconcile to Phase 1.5/1.6.
- Planned totals match result totals exactly only for complete runs; every exception is enumerated.
- Coverage is reported for base, full workload, and physics eligibility over all/stable populations and by task, model family, GPU, and architecture family.
- Benchmark versus effective length differences are reported overall and by task × model family.
- Canonical-field missingness and mapping reasons are reported.
- Llama row retention and restricted-feature impact are quantified.
- Automated allowlist/denylist checks prove no post-run leakage in X blocks.
- Master key uniqueness, target non-nullness, numeric ranges, duplicates, and join cardinalities pass.
- The notebook executes top-to-bottom and its rendered HTML is visually inspected.
- No credentials, gated raw data, model weights, full timelines, or Prometheus data are tracked.
- No formal prediction model is trained.

