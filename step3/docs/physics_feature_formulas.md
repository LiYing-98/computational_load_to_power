# Phase 1.7 physics-derived feature formulas

These are deterministic ex-ante proxies, not calibrated energy or power predictions. They use effective input length `L`, layer count `N`, hidden width `d`, query heads `h_q`, key/value heads `h_kv`, head width `d_h`, dense/active-expert intermediate width `d_ff`, output cap `O`, and bytes per runtime element `b`.

- Attention projections per layer: `2 L d (h_q d_h) + 4 L d (h_kv d_h) + 2 L (h_q d_h) d`.
- Causal prefill attention per layer: `4 h_q d_h L²`.
- Gated MLP per layer: `6 L d d_ff`; non-gated MLP: `4 L d d_ff`.
- MoE MLP multiplies the routed term by the configured active-expert count. A shared-expert term is included only when both its count and width are explicit.
- Decode is an upper-bound scenario at the configured output cap. Projection and MLP work are repeated `O` times; attention sums context lengths `L+1` through `L+O`.
- KV bytes per request: `2 N h_kv d_h sequence_length b`. Runtime dtype from static config determines `b`; weight quantization is not reused for KV precision.
- Weight bytes: `parameter_count × benchmark_weight_precision_bits / 8`.
- Work and KV totals are divided by all GPUs for an idealized per-GPU proxy. Weight bytes divide by `num_gpus / data_parallel`, reflecting one replicated model copy per data-parallel replica.

The formulas intentionally exclude embedding/logit FLOPs, communication, kernel efficiency, batching effects, cache paging, speculative decoding, and hardware-specific coefficients. DeepSeek MLA and Nemotron Mamba/Transformer hybrids are marked unsupported rather than forced into the GQA formula. Sliding-window attention remains an upper-bound full-context proxy where applicable.
