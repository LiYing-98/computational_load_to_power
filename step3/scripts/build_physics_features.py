#!/usr/bin/env python3
"""Compute transparent theoretical ex-ante work, KV-cache, and weight proxies."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _validate_nonnegative(**values: float | int | None) -> None:
    for name, value in values.items():
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def projection_flops(seq_len: int, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, head_dim: int) -> float:
    _validate_nonnegative(seq_len=seq_len, hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim)
    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    return float(2 * seq_len * hidden_size * q_width + 4 * seq_len * hidden_size * kv_width + 2 * seq_len * q_width * hidden_size)


def causal_attention_flops(seq_len: int, num_attention_heads: int, head_dim: int) -> float:
    _validate_nonnegative(seq_len=seq_len, num_attention_heads=num_attention_heads, head_dim=head_dim)
    return float(4 * num_attention_heads * head_dim * seq_len**2)


def mlp_flops(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    gated: bool,
    active_experts: int = 1,
    shared_experts: int = 0,
    shared_intermediate_size: int | None = None,
) -> float:
    _validate_nonnegative(seq_len=seq_len, hidden_size=hidden_size, intermediate_size=intermediate_size, active_experts=active_experts, shared_experts=shared_experts)
    factor = 6 if gated else 4
    routed = factor * seq_len * hidden_size * intermediate_size * active_experts
    shared = 0
    if shared_experts:
        if shared_intermediate_size is None:
            raise ValueError("shared_intermediate_size is required when shared_experts > 0")
        _validate_nonnegative(shared_intermediate_size=shared_intermediate_size)
        shared = factor * seq_len * hidden_size * shared_intermediate_size * shared_experts
    return float(routed + shared)


def _stack_mlp_flops(
    seq_len: int,
    hidden_size: int,
    dense_intermediate_size: int | None,
    dense_layer_count: int,
    moe_intermediate_size: int | None,
    moe_layer_count: int,
    active_experts: int | None,
    gated: bool,
    shared_experts: int = 0,
    shared_intermediate_size: int | None = None,
) -> float:
    total = 0.0
    if dense_layer_count:
        if dense_intermediate_size is None:
            raise ValueError("dense intermediate size is required")
        total += dense_layer_count * mlp_flops(seq_len, hidden_size, dense_intermediate_size, gated=gated)
    if moe_layer_count:
        if moe_intermediate_size is None or active_experts is None:
            raise ValueError("MoE intermediate size and active experts are required")
        total += moe_layer_count * mlp_flops(
            seq_len,
            hidden_size,
            moe_intermediate_size,
            gated=gated,
            active_experts=active_experts,
            shared_experts=shared_experts,
            shared_intermediate_size=shared_intermediate_size,
        )
    return float(total)


def decode_work_upper_bound(
    *,
    input_len: int,
    output_cap: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    dense_intermediate_size: int | None,
    dense_layer_count: int,
    moe_intermediate_size: int | None,
    moe_layer_count: int,
    active_experts: int | None,
    gated: bool,
    shared_experts: int = 0,
    shared_intermediate_size: int | None = None,
) -> float:
    _validate_nonnegative(input_len=input_len, output_cap=output_cap, num_layers=num_layers)
    projection_per_token = num_layers * projection_flops(1, hidden_size, num_attention_heads, num_key_value_heads, head_dim)
    mlp_per_token = _stack_mlp_flops(1, hidden_size, dense_intermediate_size, dense_layer_count, moe_intermediate_size, moe_layer_count, active_experts, gated, shared_experts, shared_intermediate_size)
    context_sum = output_cap * input_len + output_cap * (output_cap + 1) / 2
    attention = num_layers * 4 * num_attention_heads * head_dim * context_sum
    return float(output_cap * (projection_per_token + mlp_per_token) + attention)


def kv_cache_bytes(seq_len: int, num_layers: int, num_key_value_heads: int, head_dim: int, bytes_per_element: float) -> float:
    _validate_nonnegative(seq_len=seq_len, num_layers=num_layers, num_key_value_heads=num_key_value_heads, head_dim=head_dim, bytes_per_element=bytes_per_element)
    return float(2 * num_layers * num_key_value_heads * head_dim * seq_len * bytes_per_element)


def weight_bytes(parameter_count: float, precision_bits: int) -> float:
    _validate_nonnegative(parameter_count=parameter_count, precision_bits=precision_bits)
    return float(parameter_count * precision_bits / 8)


def ideal_per_gpu(value: float, num_gpus: int, *, data_parallel: int = 1, replicated_per_dp: bool = False) -> float:
    _validate_nonnegative(value=value, num_gpus=num_gpus, data_parallel=data_parallel)
    if num_gpus < 1 or data_parallel < 1 or num_gpus % data_parallel:
        raise ValueError("num_gpus must be a positive multiple of data_parallel")
    divisor = num_gpus / data_parallel if replicated_per_dp else num_gpus
    return float(value / divisor)


def _required_model_values(model: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "num_layers", "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "dense_layer_count", "moe_layer_count",
        "runtime_dtype_bits", "param_count_total", "precision_bits", "mlp_structure",
    )
    missing = [field for field in fields if pd.isna(model.get(field))]
    if int(model.get("moe_layer_count") or 0) > 0:
        for field in ("moe_intermediate_size", "active_expert_count"):
            if pd.isna(model.get(field)):
                missing.append(field)
    shared_count = model.get("shared_expert_count")
    if not pd.isna(shared_count) and int(shared_count) > 0 and pd.isna(model.get("shared_expert_intermediate_size")):
        # The shared-expert term is omitted unless its width is explicit; routed work remains valid.
        model = dict(model)
        model["shared_expert_count"] = 0
    if missing:
        raise ValueError("missing canonical fields: " + ", ".join(sorted(set(missing))))
    return model


def _vector(task: str, model_id: str) -> list[int]:
    path = ROOT / "step3/data/effective_vectors" / f"{task}--{model_id.replace('/', '--')}.json"
    if not path.exists():
        raise FileNotFoundError("effective base-length vector unavailable")
    return [int(value) for value in json.loads(path.read_text(encoding="utf-8"))]


def build_run_features(run: dict[str, Any], effective: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    base = {"run_key": run["run_key"], "task": run["task"], "model_id": run["model_id"]}
    if effective["effective_status"] != "available":
        return {**base, "physics_status": "missing", "physics_missing_reason": f"effective_workload_{effective['effective_status']}: {effective.get('effective_missing_reason')}"}
    if model.get("attention_variant") in {"mla", "hybrid"}:
        return {**base, "physics_status": "unsupported", "physics_missing_reason": f"unsupported_attention_variant: {model.get('attention_variant')}"}
    try:
        model = _required_model_values(model)
        lengths = _vector(run["task"], run["model_id"])
        repeats = int(run["num_request_repeats"])
        layers = int(model["num_layers"])
        hidden = int(model["hidden_size"])
        q_heads = int(model["num_attention_heads"])
        kv_heads = int(model["num_key_value_heads"])
        head_dim = int(model["head_dim"])
        dense_layers = int(model["dense_layer_count"])
        moe_layers = int(model["moe_layer_count"])
        active = None if pd.isna(model.get("active_expert_count")) else int(model["active_expert_count"])
        moe_intermediate = None if pd.isna(model.get("moe_intermediate_size")) else int(model["moe_intermediate_size"])
        shared = 0 if pd.isna(model.get("shared_expert_count")) else int(model["shared_expert_count"])
        shared_intermediate = None if pd.isna(model.get("shared_expert_intermediate_size")) else int(model["shared_expert_intermediate_size"])
        gated = model["mlp_structure"] == "gated"
        output_cap = int(run["max_output_tokens"])

        prefill_projection = repeats * sum(layers * projection_flops(length, hidden, q_heads, kv_heads, head_dim) for length in lengths)
        prefill_attention = repeats * sum(layers * causal_attention_flops(length, q_heads, head_dim) for length in lengths)
        prefill_mlp = repeats * sum(_stack_mlp_flops(length, hidden, int(model["intermediate_size"]), dense_layers, moe_intermediate, moe_layers, active, gated, shared, shared_intermediate) for length in lengths)
        decode_upper = repeats * sum(decode_work_upper_bound(
            input_len=length,
            output_cap=output_cap,
            num_layers=layers,
            hidden_size=hidden,
            num_attention_heads=q_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            dense_intermediate_size=int(model["intermediate_size"]),
            dense_layer_count=dense_layers,
            moe_intermediate_size=moe_intermediate,
            moe_layer_count=moe_layers,
            active_experts=active,
            gated=gated,
            shared_experts=shared,
            shared_intermediate_size=shared_intermediate,
        ) for length in lengths)
        bytes_per_element = float(model["runtime_dtype_bits"]) / 8
        kv_prefill_sum = repeats * sum(kv_cache_bytes(length, layers, kv_heads, head_dim, bytes_per_element) for length in lengths)
        kv_prefill_max = max(kv_cache_bytes(length, layers, kv_heads, head_dim, bytes_per_element) for length in lengths)
        kv_cap_sum = repeats * sum(kv_cache_bytes(length + output_cap, layers, kv_heads, head_dim, bytes_per_element) for length in lengths)
        weights = weight_bytes(float(model["param_count_total"]), int(model["precision_bits"]))
        total_prefill = prefill_projection + prefill_attention + prefill_mlp
        num_gpus = int(run["num_gpus"])
        data_parallel = int(run["data_parallel"])
        return {
            **base,
            "physics_status": "available",
            "physics_missing_reason": None,
            "prefill_projection_flop_proxy": prefill_projection,
            "prefill_attention_flop_proxy": prefill_attention,
            "prefill_mlp_flop_proxy": prefill_mlp,
            "prefill_total_flop_proxy": total_prefill,
            "decode_output_cap_tokens_per_request": output_cap,
            "decode_upper_bound_flop_proxy": decode_upper,
            "kv_cache_prefill_sum_bytes": kv_prefill_sum,
            "kv_cache_prefill_max_request_bytes": kv_prefill_max,
            "kv_cache_output_cap_sum_bytes": kv_cap_sum,
            "weight_bytes_total": weights,
            "prefill_total_flop_proxy_per_gpu_ideal": ideal_per_gpu(total_prefill, num_gpus),
            "decode_upper_bound_flop_proxy_per_gpu_ideal": ideal_per_gpu(decode_upper, num_gpus),
            "kv_cache_prefill_sum_bytes_per_gpu_ideal": ideal_per_gpu(kv_prefill_sum, num_gpus),
            "weight_bytes_per_gpu_ideal": ideal_per_gpu(weights, num_gpus, data_parallel=data_parallel, replicated_per_dp=True),
            "model_parallel_gpus_per_replica": num_gpus // data_parallel,
            "kv_bytes_per_element": bytes_per_element,
            "shared_expert_term_included": bool(shared),
        }
    except Exception as error:
        return {**base, "physics_status": "missing", "physics_missing_reason": f"{type(error).__name__}: {error}"}


def main() -> None:
    planned = pd.read_parquet(ROOT / "step3/analysis/planned_workload_features.parquet")
    effective = pd.read_parquet(ROOT / "step3/analysis/effective_workload_features.parquet").set_index("run_key")
    models = pd.read_parquet(ROOT / "step3/analysis/model_architecture_features.parquet").set_index("model_id")
    rows = [
        build_run_features(run, effective.loc[run["run_key"]].to_dict(), models.loc[run["model_id"]].to_dict())
        for run in planned.to_dict(orient="records")
    ]
    frame = pd.DataFrame(rows)
    analysis = ROOT / "step3/analysis"
    frame.to_parquet(analysis / "physics_features.parquet", index=False)
    frame.to_csv(analysis / "physics_features.csv", index=False)
    print(f"wrote {len(frame)} physics rows; available={(frame['physics_status'] == 'available').sum()}")


if __name__ == "__main__":
    main()
