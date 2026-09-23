#!/usr/bin/env python3
"""Build auditable canonical model architecture features from pinned static configs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoConfig

from step3.scripts.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PRECISION_BITS = {"bfloat16": 16, "fp8": 8, "mxfp4": 4}
DTYPE_BITS = {"bfloat16": 16, "float16": 16, "float32": 32, "float64": 64}
ARCHITECTURE_FAMILY = {
    "Dense Transformer": "dense_transformer",
    "MoE": "mixture_of_experts",
    "Mamba-Transformer Hybrid": "hybrid_mamba_transformer",
}
GATED_MODEL_TYPES = {"qwen3", "qwen3_moe", "gemma3_text", "deepseek_v3", "gpt_oss"}


def _json_value(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def canonicalize_model_config(
    model_id: str,
    raw_config: dict[str, Any] | None,
    benchmark: dict[str, Any],
    *,
    resolved_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mappings: list[dict[str, Any]] = []
    row: dict[str, Any] = {"model_id": model_id}
    raw_config = raw_config or {}
    is_nested = isinstance(raw_config.get("text_config"), dict)
    config = raw_config["text_config"] if is_nested else raw_config
    resolved = (resolved_config or {}).get("text_config", resolved_config or {}) if is_nested else (resolved_config or {})
    prefix = "config.text_config" if is_nested else "config"

    def put(field: str, value: Any, source_field: str | None, raw_value: Any, transform: str = "identity", status: str | None = None, reason: str | None = None) -> None:
        row[field] = value
        mappings.append({
            "model_id": model_id,
            "canonical_field": field,
            "source_field": source_field,
            "raw_value": _json_value(raw_value),
            "transform": transform,
            "status": status or ("available" if value is not None else "missing"),
            "reason": reason,
        })

    for field in ("architecture", "total_params_billions", "activated_params_billions", "weight_precision"):
        value = benchmark.get(field)
        canonical = "architecture_family" if field == "architecture" else field
        if field == "architecture":
            value = ARCHITECTURE_FAMILY.get(value, str(value).lower().replace(" ", "_") if value else None)
            transform = "benchmark architecture taxonomy"
        else:
            transform = "identity"
        put(canonical, value, f"benchmark.{field}", benchmark.get(field), transform)

    precision = benchmark.get("weight_precision")
    put("precision_bits", PRECISION_BITS.get(precision), "benchmark.weight_precision", precision, "precision label to bits")
    for source_field, output_field in (("total_params_billions", "param_count_total"), ("activated_params_billions", "param_count_active")):
        raw = benchmark.get(source_field)
        put(output_field, None if raw is None else float(raw) * 1e9, f"benchmark.{source_field}", raw, "billions * 1e9")

    model_type = config.get("model_type") or raw_config.get("model_type")
    put("model_type", model_type, f"{prefix}.model_type" if config.get("model_type") else "config.model_type", model_type)
    architecture_class = (raw_config.get("architectures") or [None])[0]
    put("architecture_class", architecture_class, "config.architectures[0]", architecture_class)

    direct = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "vocab_size": "vocab_size",
        "context_length": "max_position_embeddings",
        "sliding_window": "sliding_window",
        "moe_intermediate_size": "moe_intermediate_size",
        "shared_expert_count": "n_shared_experts",
        "shared_expert_intermediate_size": "shared_expert_intermediate_size",
        "qk_nope_head_dim": "qk_nope_head_dim",
        "qk_rope_head_dim": "qk_rope_head_dim",
        "v_head_dim": "v_head_dim",
        "mamba_head_dim": "mamba_head_dim",
        "mamba_num_heads": "mamba_num_heads",
    }
    for field, source in direct.items():
        value = config.get(source)
        source_name = f"{prefix}.{source}" if source in config else None
        if value is None and source in {"vocab_size", "max_position_embeddings", "head_dim"} and source in resolved:
            value = resolved[source]
            source_name = f"transformers_resolved.{prefix}.{source}"
            put(field, value, source_name, value, "Transformers 4.57.1 deterministic config default")
        else:
            put(field, value, source_name, value, reason="source field absent" if value is None else None)

    state_source = next((name for name in ("mamba_state_dim", "ssm_state_size") if raw_config.get(name) is not None), None)
    put("mamba_state_size", raw_config.get(state_source) if state_source else None, f"config.{state_source}" if state_source else None, raw_config.get(state_source) if state_source else None, reason="no Mamba state-size field" if not state_source else None)

    expert_sources = ("num_experts", "num_local_experts", "n_routed_experts")
    expert_source = next((name for name in expert_sources if config.get(name) is not None), None)
    put("expert_count", config.get(expert_source) if expert_source else None, f"{prefix}.{expert_source}" if expert_source else None, config.get(expert_source) if expert_source else None, reason="no expert-count field" if not expert_source else None)
    active_sources = ("num_experts_per_tok", "experts_per_token")
    active_source = next((name for name in active_sources if config.get(name) is not None), None)
    put("active_expert_count", config.get(active_source) if active_source else None, f"{prefix}.{active_source}" if active_source else None, config.get(active_source) if active_source else None, reason="no active-expert field" if not active_source else None)

    if row.get("moe_intermediate_size") is None and model_type == "gpt_oss" and config.get("intermediate_size") is not None:
        row["moe_intermediate_size"] = config["intermediate_size"]
        for item in mappings:
            if item["canonical_field"] == "moe_intermediate_size":
                item.update({
                    "source_field": f"{prefix}.intermediate_size",
                    "raw_value": _json_value(config["intermediate_size"]),
                    "transform": "gpt_oss expert MLP field mapping",
                    "status": "available",
                    "reason": None,
                })
                break

    explicit_head = config.get("head_dim")
    if explicit_head is not None:
        put("head_dim", explicit_head, f"{prefix}.head_dim", explicit_head)
    elif model_type in {"qwen3", "qwen3_moe"} and config.get("hidden_size") and config.get("num_attention_heads") and config["hidden_size"] % config["num_attention_heads"] == 0:
        put("head_dim", config["hidden_size"] // config["num_attention_heads"], f"{prefix}.hidden_size + {prefix}.num_attention_heads", {"hidden_size": config["hidden_size"], "num_attention_heads": config["num_attention_heads"]}, "hidden_size / num_attention_heads")
    elif model_type == "deepseek_v3" and config.get("qk_nope_head_dim") is not None and config.get("qk_rope_head_dim") is not None:
        put("head_dim", config["qk_nope_head_dim"] + config["qk_rope_head_dim"], f"{prefix}.qk_nope_head_dim + {prefix}.qk_rope_head_dim", {"qk_nope_head_dim": config["qk_nope_head_dim"], "qk_rope_head_dim": config["qk_rope_head_dim"]}, "sum MLA query-key head components")
    elif resolved.get("head_dim") is not None:
        put("head_dim", resolved["head_dim"], f"transformers_resolved.{prefix}.head_dim", resolved["head_dim"], "Transformers 4.57.1 deterministic config default")
    else:
        put("head_dim", None, None, None, reason="no explicit or architecture-safe derivation")

    family = row["architecture_family"]
    layers = row.get("num_layers")
    first_dense = config.get("first_k_dense_replace")
    if family == "dense_transformer" and layers is not None:
        dense_layers, moe_layers = layers, 0
    elif family == "mixture_of_experts" and layers is not None:
        dense_layers = int(first_dense or 0)
        moe_layers = int(layers) - dense_layers
    else:
        dense_layers = moe_layers = None
    put("dense_layer_count", dense_layers, f"{prefix}.first_k_dense_replace" if first_dense is not None else "benchmark.architecture + config.num_hidden_layers", first_dense if first_dense is not None else {"architecture": benchmark.get("architecture"), "num_hidden_layers": layers}, "architecture-specific layer split", reason="not a pure dense/MoE stack" if dense_layers is None else None)
    put("moe_layer_count", moe_layers, f"{prefix}.first_k_dense_replace" if first_dense is not None else "benchmark.architecture + config.num_hidden_layers", first_dense if first_dense is not None else {"architecture": benchmark.get("architecture"), "num_hidden_layers": layers}, "architecture-specific layer split", reason="not a pure dense/MoE stack" if moe_layers is None else None)

    pattern = raw_config.get("hybrid_override_pattern")
    put("hybrid_pattern", pattern, "config.hybrid_override_pattern" if pattern else None, pattern, reason="not a declared hybrid pattern" if pattern is None else None)
    for field, symbol in (("hybrid_mamba_layer_count", "M"), ("hybrid_attention_layer_count", "*"), ("hybrid_mlp_layer_count", "-")):
        put(field, pattern.count(symbol) if pattern else None, "config.hybrid_override_pattern" if pattern else None, pattern, f"count '{symbol}' symbols" if pattern else "identity", reason="not a declared hybrid pattern" if pattern is None else None)

    if family == "hybrid_mamba_transformer":
        attention_variant = "hybrid"
    elif model_type == "deepseek_v3":
        attention_variant = "mla"
    elif row.get("num_attention_heads") is not None and row.get("num_key_value_heads") is not None:
        attention_variant = "gqa" if row["num_key_value_heads"] < row["num_attention_heads"] else "mha"
    else:
        attention_variant = None
    put("attention_variant", attention_variant, f"{prefix}.model_type + attention head fields", {"model_type": model_type, "num_attention_heads": row.get("num_attention_heads"), "num_key_value_heads": row.get("num_key_value_heads")}, "architecture rule", reason="insufficient fields" if attention_variant is None else None)
    mlp_structure = "gated" if model_type in GATED_MODEL_TYPES else ("non_gated" if model_type == "nemotron_h" else None)
    put("mlp_structure", mlp_structure, f"{prefix}.model_type", model_type, "declarative architecture rule", reason="unsupported model_type" if mlp_structure is None else None)
    quantization = raw_config.get("quantization_config")
    put("quant_method_config", quantization.get("quant_method") if isinstance(quantization, dict) else None, "config.quantization_config.quant_method" if isinstance(quantization, dict) else None, quantization.get("quant_method") if isinstance(quantization, dict) else None, reason="config has no quantization method" if not isinstance(quantization, dict) else None)
    runtime_dtype = raw_config.get("torch_dtype") or config.get("torch_dtype")
    put("runtime_dtype_config", runtime_dtype, "config.torch_dtype" if raw_config.get("torch_dtype") else (f"{prefix}.torch_dtype" if config.get("torch_dtype") else None), runtime_dtype, reason="runtime dtype absent" if runtime_dtype is None else None)
    put("runtime_dtype_bits", DTYPE_BITS.get(runtime_dtype), "config.torch_dtype" if runtime_dtype else None, runtime_dtype, "dtype label to bits", reason="runtime dtype absent or unsupported" if DTYPE_BITS.get(runtime_dtype) is None else None)
    return row, mappings


def _resolved_config(path: Path) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return config.to_dict()


def main() -> None:
    runs = pd.read_parquet(ROOT / "step2/data/runs/llm.parquet")
    runs = runs[runs["task"].isin(["gpqa", "lm-arena-chat", "sourcegraph-fim"])]
    metadata_columns = ["model_id", "nickname", "architecture", "total_params_billions", "activated_params_billions", "weight_precision"]
    models = runs[metadata_columns].drop_duplicates("model_id").sort_values("model_id")
    revisions = json.loads((ROOT / "step3/config/model_revisions.json").read_text(encoding="utf-8"))["models"]
    rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for benchmark in models.to_dict(orient="records"):
        model_id = benchmark["model_id"]
        revision = revisions[model_id]
        path = ROOT / "step3/data/static_models" / model_id.replace("/", "--") / "config.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            try:
                resolved = _resolved_config(path.parent)
                resolved_status = "available"
            except Exception:
                resolved = None
                resolved_status = "unavailable"
            row, mappings = canonicalize_model_config(model_id, raw, benchmark, resolved_config=resolved)
            row.update({
                "config_status": "available",
                "restricted_feature_reason": None,
                "config_revision": revision["revision"],
                "config_sha256": sha256_file(path),
                "resolved_config_status": resolved_status,
            })
        else:
            row, mappings = canonicalize_model_config(model_id, None, benchmark)
            row.update({
                "config_status": "restricted",
                "restricted_feature_reason": "Meta Llama static config access rejected; benchmark public fields retained",
                "config_revision": revision["revision"],
                "config_sha256": None,
                "resolved_config_status": "restricted",
            })
        rows.append(row)
        mapping_rows.extend(mappings)
    frame = pd.DataFrame(rows)
    mapping = pd.DataFrame(mapping_rows)
    analysis = ROOT / "step3/analysis"
    reports = ROOT / "step3/reports"
    frame.to_parquet(analysis / "model_architecture_features.parquet", index=False)
    frame.to_csv(analysis / "model_architecture_features.csv", index=False)
    mapping.to_csv(analysis / "model_field_mapping.csv", index=False)
    feature_fields = [column for column in frame.columns if column not in {"model_id", "config_status", "restricted_feature_reason", "config_revision", "config_sha256", "resolved_config_status"}]
    missingness = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": int(len(frame)),
        "config_available": int((frame["config_status"] == "available").sum()),
        "config_restricted": int((frame["config_status"] == "restricted").sum()),
        "fields": {
            field: {
                "available": int(frame[field].notna().sum()),
                "missing": int(frame[field].isna().sum()),
                "coverage": float(frame[field].notna().mean()),
            }
            for field in feature_fields
        },
    }
    (reports / "model_feature_missingness.json").write_text(json.dumps(missingness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(frame)} model rows; configs={missingness['config_available']}; restricted={missingness['config_restricted']}")


if __name__ == "__main__":
    main()
