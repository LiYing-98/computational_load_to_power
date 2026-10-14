"""Build strict ex-ante information tiers from frozen local evidence only."""
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd

from step7.scripts.common import ROOT, STEP7, TARGET, TASKBOOK, assert_feature_boundary, read_json, sha256, verify_sources, write_json

STATIC_CORE = [
    "x_model_num_layers", "x_model_hidden_size", "x_model_intermediate_size",
    "x_model_head_dim", "x_model_context_length", "x_model_num_attention_heads",
    "x_model_num_key_value_heads", "x_model_dense_layer_count", "x_model_moe_layer_count",
]
WORK_STATS = ["mean", "p50", "p90", "p95", "min", "max", "std", "second_moment", "rms", "cv"]


def deployment_proxies(num_gpus, dp, max_seqs, weight_bytes, active_bytes, capacity_bytes):
    if num_gpus < 1 or dp < 1 or num_gpus % dp or capacity_bytes <= 0:
        raise ValueError("GPU count must be a positive integer multiple of DP; capacity positive")
    parallel = num_gpus / dp
    return {
        "model_parallel_gpus_per_replica": parallel,
        "max_num_seqs_per_gpu": max_seqs / num_gpus,
        "weight_bytes_per_model_parallel_gpu": weight_bytes / parallel,
        "active_weight_bytes_per_model_parallel_gpu": active_bytes / parallel,
        "weight_hbm_ratio": weight_bytes / parallel / capacity_bytes,
        "active_weight_hbm_ratio": active_bytes / parallel / capacity_bytes,
    }


def compact_physics(*, layers, hidden, heads, kv_heads, head_dim, intermediate,
                    active_experts, mlp_terms, mean_length, second_moment,
                    p95_length, max_length, kv_element_bytes):
    # Dense/GQA full-square FLOP convention inherited from frozen Step3.
    projection_one = 2 * hidden * head_dim * (2 * heads + 2 * kv_heads) * layers
    mlp_one = 2 * mlp_terms * hidden * intermediate * active_experts * layers
    kv_per_token = 2 * layers * kv_heads * head_dim * kv_element_bytes
    result = {
        "prefill_projection_flop_per_request": projection_one * mean_length,
        "prefill_attention_flop_per_request": 4 * layers * heads * head_dim * second_moment,
        "prefill_mlp_flop_per_request": mlp_one * mean_length,
        "kv_mean_request_bytes": kv_per_token * mean_length,
        "kv_p95_request_bytes": kv_per_token * p95_length,
        "kv_max_request_bytes": kv_per_token * max_length,
        "decode_one_token_flop_mean_context": projection_one + mlp_one + 4 * layers * heads * head_dim * (mean_length + 1),
    }
    result["prefill_total_flop_per_request"] = sum(result[x] for x in ["prefill_projection_flop_per_request", "prefill_attention_flop_per_request", "prefill_mlp_flop_per_request"])
    return result


def freeze_sources():
    path = STEP7 / "config/source_freeze_phase5.json"
    if path.exists():
        verify_sources()
        return
    inputs = [
        "step3/analysis/master_feature_table.parquet", "step3/analysis/feature_role_manifest.json",
        "step3/analysis/planned_workload_provenance.json", "step3/analysis/effective_workload_provenance.json",
        "step3/analysis/model_architecture_features.parquet", "step3/scripts/build_physics_features.py",
        "step3/scripts/reconstruct_planned_workload.py", "step5/analysis/phase3_modeling_table.parquet",
        "step5/config/gpu_hardware_specs.json", "step5/config/hardware_source_manifest.json",
        "step6/analysis/phase4_power_modeling_table.parquet", "step6/config/locked_candidate_phase4.json",
        "step6/reports/main_model_comparison.csv", "step6/reports/confirmatory_metrics.csv",
        "step6/reports/error_slices.csv", "step6/reports/static_subset_comparison.csv",
        "step4/reports/ablation_summary.csv", "step5/reports/feature_ablation.csv",
        "step2/data/planned_requests/lm-arena-chat.json", "step2/data/planned_requests/sourcegraph-fim.json",
        "step3/data/hf_cache/datasets--Idavidrein--gpqa/snapshots/633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_diamond.csv",
    ]
    write_json(path, {"schema_version": "1.0", "text_hash_policy": "UTF-8 LF; binary byte exact", "taskbook_sha256": sha256(TASKBOOK), "inputs": [{"path": p, "sha256": sha256(ROOT / p)} for p in inputs]})


def request_text_audit(frame):
    provenance = read_json(ROOT / "step3/analysis/planned_workload_provenance.json")
    result = subprocess.run(
        [str(ROOT / "step3/.venv/Scripts/python.exe"), "-c", "import json; from step3.scripts.reconstruct_planned_workload import build_gpqa_requests; print(json.dumps(build_gpqa_requests(), ensure_ascii=False))"],
        cwd=ROOT, capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    requests = {
        "lm-arena-chat": read_json(ROOT / "step2/data/planned_requests/lm-arena-chat.json"),
        "sourcegraph-fim": read_json(ROOT / "step2/data/planned_requests/sourcegraph-fim.json"),
    }
    if result.returncode == 0:
        requests["gpqa"] = json.loads(result.stdout)
    rows, metrics = [], {}
    for task in sorted(frame.x_protocol_task.unique()):
        req = requests.get(task)
        if req is None:
            rows.append({"task": task, "status": "unavailable", "reason": "pinned GPQA reconstruction failed", "n_requests": 0})
            continue
        payload = json.dumps(req, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        request_hash = hashlib.sha256(payload).hexdigest()
        known = provenance.get("request_hashes", {}).get(task)
        expected = set(frame.loc[frame.x_protocol_task.eq(task), "prov_planned_request_list_sha256"].dropna())
        if known:
            expected.add(known)
        if request_hash not in expected:
            rows.append({"task": task, "status": "unverified", "reason": "request list hash not in frozen evidence", "n_requests": len(req)})
            continue
        char_lengths, byte_lengths, messages = [], [], []
        for item in req:
            prompt = item.get("prompt")
            parts = prompt if isinstance(prompt, list) else [prompt] if isinstance(prompt, str) else [item["prefix"], item["suffix"]]
            char_lengths.append(sum(len(s) for s in parts))
            byte_lengths.append(sum(len(s.encode("utf-8")) for s in parts))
            messages.append(len(parts) if prompt is not None else np.nan)
        stats = {}
        for name, lengths in [("characters", char_lengths), ("utf8_bytes", byte_lengths)]:
            a = np.asarray(lengths, dtype=float)
            for label, value in {"mean": a.mean(), "p50": np.quantile(a, .5), "p90": np.quantile(a, .9), "p95": np.quantile(a, .95), "max": a.max(), "std": a.std(), "second_moment": np.mean(a*a)}.items():
                stats[f"x_text_{name}_per_request_{label}"] = float(value)
        metrics[task] = stats
        rows.append({"task": task, "status": "verified", "reason": "exact frozen request-list hash; no tokenizer or completion used", "n_requests": len(req), "request_list_sha256": request_hash})
    pd.DataFrame(rows).to_csv(STEP7 / "analysis/tokenizer_free_text_audit.csv", index=False)
    for task, values in metrics.items():
        for name, value in values.items():
            frame.loc[frame.x_protocol_task.eq(task), name] = value
    return len(metrics) == frame.x_protocol_task.nunique()


def build_table():
    freeze_sources()
    frame = pd.read_parquet(ROOT / "step6/analysis/phase4_power_modeling_table.parquet").copy()
    if frame.run_key.duplicated().any() or not frame.diag_quality_is_stable.all():
        raise ValueError("stable input/run_key grain mismatch")
    frame["x_p5_precision_bits"] = frame.x_model_benchmark_weight_precision.map({"bfloat16": 16., "fp8": 8., "mxfp4": 4.})
    frame["x_p5_weight_bytes_total"] = frame.x_model_benchmark_total_params_billions * 1e9 * frame.x_p5_precision_bits / 8
    frame["x_p5_active_weight_bytes_total"] = frame.x_model_benchmark_activated_params_billions * 1e9 * frame.x_p5_precision_bits / 8
    frame["x_p5_active_param_ratio"] = frame.x_model_benchmark_activated_params_billions / frame.x_model_benchmark_total_params_billions
    records = []
    for row in frame.itertuples():
        values = deployment_proxies(row.x_hardware_num_gpus, row.x_deployment_data_parallel, row.x_deployment_max_num_seqs, row.x_p5_weight_bytes_total, row.x_p5_active_weight_bytes_total, row.x_hardware_hbm_capacity_gb * 1e9)
        records.append({f"x_p5_{k}": v for k, v in values.items()})
    frame = pd.concat([frame, pd.DataFrame(records)], axis=1)
    # Transformer dense/MoE composition is structurally N/A for hybrid;
    # the hybrid pattern is retained in the extended sensitivity.
    hybrid = frame.diag_architecture_class.eq("hybrid_mamba_transformer")
    frame["x_p5_transformer_composition_applicable"] = (~hybrid).astype(float)
    for column in ["x_model_dense_layer_count", "x_model_moe_layer_count"]:
        frame.loc[hybrid, column] = 0.
    frame["x_p5_gqa_ratio_applicable"] = frame.x_model_attention_variant.eq("gqa").astype(float)
    frame["x_p5_gqa_ratio"] = np.where(frame.x_model_attention_variant.eq("gqa"), frame.x_model_num_attention_heads / frame.x_model_num_key_value_heads, 0.)
    for name, values in {
        "ffn_expansion_ratio": frame.x_model_intermediate_size / frame.x_model_hidden_size,
        "moe_layer_fraction": frame.x_model_moe_layer_count / frame.x_model_num_layers,
        "dense_layer_fraction": frame.x_model_dense_layer_count / frame.x_model_num_layers,
        "params_per_layer": frame.x_model_benchmark_total_params_billions / frame.x_model_num_layers,
        "active_params_per_layer": frame.x_model_benchmark_activated_params_billions / frame.x_model_num_layers,
    }.items():
        frame[f"x_p5_{name}"] = values

    physics_names = list(compact_physics(layers=1, hidden=1, heads=1, kv_heads=1, head_dim=1, intermediate=1, active_experts=1, mlp_terms=3, mean_length=1, second_moment=1, p95_length=1, max_length=1, kv_element_bytes=2))
    for name in physics_names:
        frame[f"x_p5_{name}"] = np.nan
    frame["diag_p5_physics_reason"] = "missing static/workload or unsupported MLA/hybrid"
    for index, row in frame.iterrows():
        required = ["x_model_num_layers", "x_model_hidden_size", "x_model_num_attention_heads", "x_model_num_key_value_heads", "x_model_head_dim", "x_workload_benchmark_input_tokens_mean", "x_workload_benchmark_input_tokens_second_moment", "x_workload_benchmark_input_tokens_p95", "x_workload_benchmark_input_tokens_max"]
        if row.x_model_attention_variant != "gqa" or row[required].isna().any():
            continue
        moe_layers = float(row.x_model_moe_layer_count)
        if moe_layers and (pd.isna(row.x_model_moe_intermediate_size) or pd.isna(row.x_model_active_expert_count)):
            continue
        # Avoid inventing a runtime KV dtype. GPT-OSS runtime dtype is unknown.
        if pd.isna(row.x_model_runtime_dtype_bits):
            frame.loc[index, "diag_p5_physics_reason"] = "runtime KV element dtype unknown"
            continue
        intermediate = row.x_model_moe_intermediate_size if moe_layers else row.x_model_intermediate_size
        active = row.x_model_active_expert_count if moe_layers else 1.
        values = compact_physics(layers=row.x_model_num_layers, hidden=row.x_model_hidden_size, heads=row.x_model_num_attention_heads, kv_heads=row.x_model_num_key_value_heads, head_dim=row.x_model_head_dim, intermediate=intermediate, active_experts=active, mlp_terms=3 if row.x_model_mlp_structure == "gated" else 2, mean_length=row.x_workload_benchmark_input_tokens_mean, second_moment=row.x_workload_benchmark_input_tokens_second_moment, p95_length=row.x_workload_benchmark_input_tokens_p95, max_length=row.x_workload_benchmark_input_tokens_max, kv_element_bytes=row.x_model_runtime_dtype_bits / 8)
        for name, value in values.items():
            frame.loc[index, f"x_p5_{name}"] = value
        frame.loc[index, "diag_p5_physics_reason"] = "supported GQA; theoretical full-square FLOP convention"

    text_complete = request_text_audit(frame)
    basic_cat = ["x_hardware_gpu_model", "diag_architecture_class", "x_model_benchmark_weight_precision"]
    basic_num = ["x_hardware_num_gpus", "x_model_benchmark_total_params_billions", "x_p5_precision_bits", "x_p5_weight_bytes_total", "x_hardware_rated_power_w", "x_hardware_aggregate_rated_power_w", "x_hardware_hbm_capacity_gb", "x_hardware_hbm_bandwidth_gbps", "x_hardware_precision_matched_peak_compute_tflops", "x_hardware_interconnect_bandwidth_gbps"]
    levels = {}
    def add(name, cat, numeric, support, *, sensitivity=False):
        assert_feature_boundary(cat+numeric, task_sensitivity=name == "L1-TaskSensitivity")
        eligibility = "eligible_p5_" + name.split("-")[0].lower()
        frame[eligibility] = support.astype(bool)
        levels[name] = {"categorical": list(cat), "numeric": list(numeric), "columns": list(cat)+list(numeric), "eligibility": eligibility, "sensitivity_only": sensitivity}
    all_rows = pd.Series(True, index=frame.index)
    add("L0-Basic", basic_cat, basic_num, all_rows)
    active_num = basic_num + ["x_model_benchmark_activated_params_billions", "x_p5_active_param_ratio", "x_p5_active_weight_bytes_total"]
    add("L0A-ActiveAware", basic_cat, active_num, all_rows)
    deployment_num = active_num + ["x_deployment_max_num_seqs", "x_deployment_tensor_parallel", "x_deployment_data_parallel", "x_deployment_expert_parallel"] + ["x_p5_"+x for x in records[0] if False]
    deployment_num += list(records[0])
    add("L1-Deployment", basic_cat, deployment_num, all_rows)
    static_cat = basic_cat + ["x_model_attention_variant", "x_model_mlp_structure"]
    static_num = deployment_num + STATIC_CORE + ["x_p5_"+x for x in ["ffn_expansion_ratio", "gqa_ratio", "gqa_ratio_applicable", "transformer_composition_applicable", "moe_layer_fraction", "dense_layer_fraction", "params_per_layer", "active_params_per_layer"]]
    static_support = frame[STATIC_CORE+["x_model_attention_variant", "x_model_mlp_structure"]].notna().all(axis=1)
    add("L2-StaticCore", static_cat, static_num, static_support)
    workload_num = ["x_workload_max_output_tokens"] + ["x_workload_benchmark_input_tokens_"+x for x in WORK_STATS]
    work_support = static_support & frame[workload_num].notna().all(axis=1)
    add("L3-Workload", static_cat, static_num+workload_num, work_support)
    physics_cols = ["x_p5_"+x for x in physics_names if x != "prefill_total_flop_per_request"]
    physics_support = work_support & frame[physics_cols].notna().all(axis=1)
    add("L4-CompactPhysics", static_cat, static_num+workload_num+physics_cols, physics_support)
    effective_num = ["x_workload_max_output_tokens"] + ["x_workload_effective_input_tokens_"+x for x in WORK_STATS]
    effective_support = static_support & frame[effective_num].notna().all(axis=1)
    add("L3E-Effective", static_cat, static_num+effective_num, effective_support, sensitivity=True)
    # Structural N/A set to zero only where architecture proves absence;
    # applicable-but-unknown fields retain NaN and explicit missing indicators.
    extended_cols = ["x_model_expert_count", "x_model_active_expert_count", "x_model_moe_intermediate_size", "x_model_shared_expert_count", "x_model_shared_expert_intermediate_size", "x_model_mamba_head_dim", "x_model_mamba_num_heads", "x_model_mamba_state_size", "x_model_hybrid_mamba_layer_count", "x_model_hybrid_attention_layer_count", "x_model_hybrid_mlp_layer_count", "x_model_sliding_window", "x_model_qk_nope_head_dim", "x_model_qk_rope_head_dim", "x_model_v_head_dim"]
    extended = []
    for column in extended_cols:
        new = column.replace("x_model_", "x_p5_extended_")
        applicable = hybrid if any(x in column for x in ["mamba", "hybrid"]) else frame.diag_architecture_class.eq("mixture_of_experts") if any(x in column for x in ["expert", "moe"]) else frame.x_model_attention_variant.eq("mla") if any(x in column for x in ["qk_", "v_head_dim"]) else static_support
        frame[new] = frame[column].where(applicable, 0.)
        frame[new+"_applicable"] = applicable.astype(float)
        frame[new+"_unknown"] = (applicable & frame[column].isna()).astype(float)
        extended += [new, new+"_applicable", new+"_unknown"]
    add("L2X-StaticExtended", static_cat, static_num+extended, static_support, sensitivity=True)
    old_cols = ["x_hardware_physics_compute_time_proxy_seconds", "x_hardware_physics_memory_time_proxy_seconds", "x_hardware_physics_weight_capacity_ratio", "x_hardware_physics_kv_capacity_ratio_mean", "x_hardware_physics_compute_to_memory_pressure_ratio"]
    add("L4B5-OldB5Sensitivity", static_cat, static_num+workload_num+physics_cols+old_cols, physics_support & frame[old_cols].notna().all(axis=1), sensitivity=True)
    add("L1-TaskSensitivity", basic_cat+["x_protocol_task"], deployment_num, all_rows, sensitivity=True)
    # A sensitivity must not overwrite its parent cohort eligibility column.
    levels["L1-TaskSensitivity"]["eligibility"] = "eligible_p5_l1"
    if text_complete:
        text_cols = [c for c in frame if c.startswith("x_text_")]
        add("L1T-TokenizerFreeTextProxy", basic_cat, deployment_num+text_cols, frame[text_cols].notna().all(axis=1), sensitivity=True)

    write_json(STEP7 / "config/feature_hierarchy_phase5.json", {"schema_version": "1.0", "levels": levels, "dropped_duplicate": {"x_p5_prefill_total_flop_per_request": "exact sum of three retained compute mechanisms"}, "text_branch_status": "eligible" if text_complete else "audit_only"})
    frame.to_parquet(STEP7 / "analysis/phase5_master_table.parquet", index=False)
    coverage = []
    for level, bundle in levels.items():
        support = frame.loc[frame[bundle["eligibility"]]]
        coverage.append({"feature_level": level, "n_rows": len(support), "n_models": support.diag_model_id.nunique(), "n_lineages": support.diag_parent_lineage.nunique(), "n_architectures": support.diag_architecture_class.nunique(), "gpu_composition": json.dumps(support.x_hardware_gpu_model.value_counts().to_dict()), "task_composition": json.dumps(support.x_protocol_task.value_counts().to_dict()), "gpu_count_composition": json.dumps({str(k):int(v) for k,v in support.x_hardware_num_gpus.value_counts().items()})})
    pd.DataFrame(coverage).to_csv(STEP7 / "analysis/cohort_coverage.csv", index=False)
    feature_rows = []
    for column in sorted({c for b in levels.values() for c in b["columns"]}):
        applicable = frame[column+"_applicable"] if column+"_applicable" in frame else pd.Series(True, index=frame.index)
        feature_rows.append({"feature": column, "source": "step6/analysis/phase4_power_modeling_table.parquet" if column in pd.read_parquet(ROOT / "step6/analysis/phase4_power_modeling_table.parquet").columns else "Phase5 ex-ante formula or frozen request text", "formal_role": "exante", "missing_rows": int(frame[column].isna().sum()), "unique_count": frame[column].nunique(), "structural_na_rows": int((~applicable.astype(bool)).sum()), "unknown_rows": int((applicable.astype(bool) & frame[column].isna()).sum())})
    audit = pd.DataFrame(feature_rows)
    audit.to_csv(STEP7 / "analysis/feature_semantics_audit.csv", index=False)
    audit.to_csv(STEP7 / "analysis/feature_coverage.csv", index=False)
    write_json(STEP7 / "config/feature_formulas_phase5.json", {"weight_bytes_total": "total_params_billions * 1e9 * benchmark_precision_bits / 8", "active_weight_bytes_total": "activated_params_billions * 1e9 * benchmark_precision_bits / 8", "model_parallel_gpus_per_replica": "num_gpus/data_parallel with positive integer validation", "max_num_seqs_per_gpu": "max_num_seqs/num_gpus; not a per-replica capacity claim", "gqa_ratio": "heads/kv_heads only for GQA; structural zero plus applicability flag otherwise", "physics": "See physics_feature_formulas_phase5.md", "text": "Sum characters and UTF-8 bytes of raw input fields, excluding completion, template markers and tokenizer; exact frozen request-list hash verified"})
    (STEP7 / "reports/feature_semantics_audit.md").write_text("# Phase5 特征语义审计\n\n主cohort保持565条stable GPU-side inference记录。详细static未伪造Llama信息。Hybrid的dense/MoE transformer layer composition为结构N/A：显式0加applicability指示，其原始统计位于只读Step6。MLA/hybrid不套GQA公式；GQA runtime KV dtype未知时L4不可用。Text proxy用冻结原始输入列表并排除completion；长度与UTF-8 bytes只表示输入文本，不代表模型token。主层无task标签。\n\n覆盖和N/A/unknown计数见analysis CSV；未把较小support的绝对MAE当作特征增益。\n", encoding="utf-8")
    print(pd.DataFrame(coverage)[["feature_level", "n_rows", "n_models"]].to_string(index=False))


def main():
    for name in ["analysis", "config", "reports"]:
        (STEP7 / name).mkdir(parents=True, exist_ok=True)
    build_table()


if __name__ == "__main__":
    main()
