#!/usr/bin/env python3
"""Assemble the one-row-per-run Phase 1.7 ex-ante feature table."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from step3.scripts.common import EXPECTED_RUNS, EXPECTED_STABLE_RUNS, RunKey, TEXT_LLM_TASKS


ROOT = Path(__file__).resolve().parents[2]
EXANTE_CONTRACT_PATH = Path("step3/config/exante_feature_contract.json")

TARGET_MAP = {
    "steady_state_energy_joules": "y_gpu_steady_state_energy_joules",
    "steady_state_duration_seconds": "y_gpu_steady_state_duration_seconds",
    "avg_power_watts": "y_gpu_avg_power_watts",
    "energy_per_token_joules": "y_gpu_energy_per_token_joules",
    "energy_per_request_joules": "y_gpu_energy_per_request_joules",
}

RESULT_DIAGNOSTIC_FIELDS = (
    "output_throughput_tokens_per_sec",
    "request_throughput_req_per_sec",
    "total_output_tokens",
    "completed_requests",
    "avg_output_len",
    "mean_itl_ms",
    "median_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "avg_batch_size",
)


def _assert_unique(frame: pd.DataFrame, key: str, name: str, expected: int | None = None) -> None:
    if key not in frame.columns:
        raise ValueError(f"{name} is missing key {key}")
    duplicates = int(frame[key].duplicated().sum())
    if duplicates:
        raise ValueError(f"{name} has {duplicates} duplicate {key} values")
    if expected is not None and len(frame) != expected:
        raise ValueError(f"{name} expected {expected} rows, found {len(frame)}")


def _model_family(model_id: str) -> str:
    repository = model_id.split("/", 1)[-1]
    families = (
        ("Qwen3-Coder", "Qwen3-Coder"),
        ("Qwen3", "Qwen3"),
        ("DeepSeek", "DeepSeek"),
        ("gemma-3", "Gemma-3"),
        ("Llama-3.1", "Llama-3.1"),
        ("Llama-3.3", "Llama-3.3"),
        ("Llama-4", "Llama-4"),
        ("NVIDIA-Nemotron-Nano", "Nemotron-Nano-v2"),
        ("gpt-oss", "GPT-OSS"),
    )
    for prefix, family in families:
        if repository.startswith(prefix):
            return family
    return repository


def restriction_reasons(row: Mapping[str, Any]) -> str | None:
    """Return ordered machine-readable limitations for one retained run."""

    reasons: list[str] = []
    model_id = str(row.get("diag_model_id") or "")
    tokenization_status = row.get("prov_tokenization_status")
    effective_status = row.get("prov_effective_status")
    reconciliation_status = row.get("diag_quality_input_reconciliation_status")
    config_status = row.get("prov_config_status")
    physics_status = row.get("prov_physics_status")
    physics_reason = str(row.get("prov_physics_missing_reason") or "").lower()

    if model_id.startswith("meta-llama/"):
        reasons.append("llama_access_rejected")
    if tokenization_status != "available":
        reasons.append("tokenizer_unavailable")
    if effective_status == "unresolved":
        reasons.append("chat_template_unresolved")
    elif effective_status not in (None, "available") and "llama_access_rejected" not in reasons:
        reasons.append("effective_workload_missing")
    if reconciliation_status == "mismatch_unresolved_historical_equivalence":
        reasons.append("historical_input_mismatch_unresolved")
    if config_status not in (None, "available"):
        reasons.append("config_missing")
    if physics_status == "unsupported":
        if "hybrid" in physics_reason:
            reasons.append("physics_unsupported_hybrid")
        elif "mla" in physics_reason:
            reasons.append("physics_unsupported_mla")
        else:
            reasons.append("physics_unsupported_architecture")
    elif physics_status not in (None, "available"):
        reasons.append("physics_inputs_missing")
    return ";".join(dict.fromkeys(reasons)) or None


def _base_runs(project_root: Path) -> pd.DataFrame:
    run_path = project_root / "step2/data/runs/llm.parquet"
    if not run_path.exists():
        run_path = project_root / "step1/data/runs/llm.parquet"
    runs = pd.read_parquet(run_path)
    runs = runs[runs["task"].isin(TEXT_LLM_TASKS)].copy()
    runs["run_key"] = [RunKey.from_mapping(row).canonical() for row in runs.to_dict(orient="records")]
    _assert_unique(runs, "run_key", "text-LLM run table", EXPECTED_RUNS)
    if int(runs["is_stable"].sum()) != EXPECTED_STABLE_RUNS:
        raise ValueError("stable-run count does not match the Phase 1.5/1.6 contract")

    result = pd.DataFrame({"run_key": runs["run_key"]})
    result["x_protocol_task"] = runs["task"]
    result["x_protocol_seed"] = runs["seed"]
    result["x_protocol_num_request_repeats"] = runs["num_request_repeats"]
    result["x_protocol_endpoint_type"] = runs["task"].map(
        {"gpqa": "openai-chat", "lm-arena-chat": "openai-chat", "sourcegraph-fim": "openai"}
    )
    result["x_hardware_gpu_model"] = runs["gpu_model"]
    result["x_hardware_num_gpus"] = runs["num_gpus"]
    result["x_deployment_max_num_seqs"] = runs["max_num_seqs"]
    result["x_deployment_tensor_parallel"] = runs["tensor_parallel"]
    result["x_deployment_expert_parallel"] = runs["expert_parallel"]
    result["x_deployment_data_parallel"] = runs["data_parallel"]
    result["x_model_benchmark_architecture"] = runs["architecture"]
    result["x_model_benchmark_total_params_billions"] = runs["total_params_billions"]
    result["x_model_benchmark_activated_params_billions"] = runs["activated_params_billions"]
    result["x_model_benchmark_weight_precision"] = runs["weight_precision"]
    result["diag_model_id"] = runs["model_id"]
    result["diag_model_nickname"] = runs["nickname"]
    result["diag_model_family"] = runs["model_id"].map(_model_family)
    result["diag_quality_is_stable"] = runs["is_stable"].astype(bool)
    result["diag_quality_unstable_reason"] = runs["unstable_reason"]
    result["prov_results_path"] = runs["results_path"]
    result["prov_prometheus_path_not_downloaded"] = runs["prometheus_path"]
    for source, target in TARGET_MAP.items():
        result[target] = runs[source]
    for field in RESULT_DIAGNOSTIC_FIELDS:
        result[f"diag_result_{field}"] = runs[field]
    return result


def _planned_block(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_unique(frame, "run_key", "planned workload", EXPECTED_RUNS)
    result = frame[["run_key"]].copy()
    result["x_workload_max_output_tokens"] = frame["max_output_tokens"]
    for field in frame.columns:
        if field.startswith("planned_"):
            result[f"x_workload_benchmark_{field.removeprefix('planned_')}"] = frame[field]
    result["prov_tokenization_status"] = frame["tokenization_status"]
    result["prov_tokenization_missing_reason"] = frame["missing_reason"]
    for field in (
        "benchmark_commit", "dataset_revision", "request_list_sha256", "prompt_template_sha256",
        "tokenizer_revision", "tokenizer_revision_requested", "prompt_length_semantics",
        "base_input_lengths_sha256",
    ):
        result[f"prov_planned_{field}"] = frame[field]
    return result


def _effective_block(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_unique(frame, "run_key", "effective workload", EXPECTED_RUNS)
    result = frame[["run_key"]].copy()
    for field in frame.columns:
        if field.startswith("effective_") and field not in {"effective_status", "effective_missing_reason"}:
            result[f"x_workload_effective_{field.removeprefix('effective_')}"] = frame[field]
    result["prov_effective_status"] = frame["effective_status"]
    result["prov_effective_missing_reason"] = frame["effective_missing_reason"]
    for field in (
        "chat_content_format", "chat_template_sha256", "system_prompt_sha256",
        "chat_template_kwargs_json", "base_effective_lengths_sha256",
    ):
        result[f"prov_effective_{field}"] = frame[field]
    return result


def _model_block(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_unique(frame, "model_id", "canonical model table", 27)
    result = pd.DataFrame({"diag_model_id": frame["model_id"]})
    excluded = {
        "model_id", "total_params_billions", "activated_params_billions", "weight_precision",
        "config_status", "restricted_feature_reason", "config_revision", "config_sha256",
        "resolved_config_status",
    }
    for field in frame.columns:
        if field not in excluded:
            result[f"x_model_{field}"] = frame[field]
    result["prov_config_status"] = frame["config_status"]
    result["prov_config_restricted_reason"] = frame["restricted_feature_reason"]
    result["prov_config_revision"] = frame["config_revision"]
    result["prov_config_sha256"] = frame["config_sha256"]
    result["prov_config_resolved_status"] = frame["resolved_config_status"]
    return result


def _physics_block(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_unique(frame, "run_key", "physics table", EXPECTED_RUNS)
    result = frame[["run_key"]].copy()
    excluded = {"run_key", "task", "model_id", "physics_status", "physics_missing_reason"}
    for field in frame.columns:
        if field not in excluded:
            prefix = "x_deployment_" if field == "model_parallel_gpus_per_replica" else "x_physics_"
            result[f"{prefix}{field}"] = frame[field]
    result["prov_physics_status"] = frame["physics_status"]
    result["prov_physics_missing_reason"] = frame["physics_missing_reason"]
    return result


def _reconciliation_block(report: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in report["detail"]:
        rows.append(
            {
                "run_key": item["run_key"],
                "diag_quality_input_reconciliation_status": item["reconciliation_status"],
                "diag_quality_input_reconciliation_reason": item["reconciliation_reason"],
                "diag_result_input_token_exact_match": item["exact_match"],
                "diag_result_total_input_tokens": item["result_total_input_tokens"],
                "diag_result_input_token_absolute_error": item["absolute_error"],
                "diag_result_input_token_relative_error": item["relative_error"],
            }
        )
    frame = pd.DataFrame(rows)
    _assert_unique(frame, "run_key", "planned/result reconciliation", EXPECTED_RUNS)
    return frame


def _derive_eligibility(master: pd.DataFrame) -> pd.DataFrame:
    # Consolidate the many joined source blocks before adding derived columns.
    master = master.copy()
    base_required = (
        "x_protocol_task", "x_protocol_num_request_repeats", "x_hardware_gpu_model",
        "x_hardware_num_gpus", "x_deployment_max_num_seqs", "x_deployment_tensor_parallel",
        "x_deployment_expert_parallel", "x_deployment_data_parallel",
        "x_model_benchmark_total_params_billions", "x_model_benchmark_activated_params_billions",
        "x_model_benchmark_weight_precision", "x_workload_max_output_tokens",
        *TARGET_MAP.values(),
    )
    master["eligible_base_exante"] = master[list(base_required)].notna().all(axis=1)
    benchmark_columns = [column for column in master if column.startswith("x_workload_benchmark_")]
    effective_columns = [column for column in master if column.startswith("x_workload_effective_")]
    physics_columns = [column for column in master if column.startswith("x_physics_")]
    def finite_nonnegative(columns: list[str]) -> pd.Series:
        valid = pd.Series(True, index=master.index)
        for column in columns:
            values = pd.to_numeric(master[column], errors="coerce")
            valid &= values.notna() & values.map(math.isfinite) & values.ge(0)
        return valid

    workload_complete = finite_nonnegative(benchmark_columns + effective_columns)
    physics_complete = finite_nonnegative(physics_columns)
    master["eligible_full_workload"] = (
        master["eligible_base_exante"]
        & master["prov_tokenization_status"].eq("available")
        & master["prov_effective_status"].eq("available")
        & master["diag_quality_input_reconciliation_status"].eq("exact")
        & workload_complete
    )
    master["eligible_physics_feature"] = (
        master["eligible_full_workload"]
        & master["prov_physics_status"].eq("available")
        & physics_complete
    )
    master["eligible_primary_stable"] = master["diag_quality_is_stable"].astype(bool)
    master["restricted_feature_reason"] = [restriction_reasons(row) for row in master.to_dict(orient="records")]
    return master


def load_exante_contract(project_root: Path = ROOT) -> dict[str, str]:
    """Load the exact, reviewed column-to-role contract for model inputs."""

    payload = json.loads((Path(project_root) / EXANTE_CONTRACT_PATH).read_text(encoding="utf-8"))
    contract = payload.get("columns", {})
    if not isinstance(contract, dict) or not contract:
        raise ValueError("ex-ante feature contract must contain a non-empty columns mapping")
    invalid = {column: role for column, role in contract.items() if role not in {
        "X_workload", "X_model", "X_hardware", "X_deployment", "X_protocol"
    }}
    if invalid:
        raise ValueError(f"ex-ante feature contract contains invalid roles: {invalid}")
    return {str(column): str(role) for column, role in contract.items()}


def make_role_manifest(master: pd.DataFrame, project_root: Path = ROOT) -> list[dict[str, Any]]:
    """Describe every column's modeling role and leakage eligibility."""

    contract = load_exante_contract(project_root)
    prefix_roles = (
        ("y_gpu_", "Y_gpu", False),
        ("diag_result_", "diagnostic_result", False),
        ("diag_quality_", "diagnostic_quality", False),
        ("diag_model_", "diagnostic_split", False),
        ("prov_", "provenance", False),
    )
    manifest: list[dict[str, Any]] = []
    for column in master.columns:
        if column.startswith("x_"):
            if column not in contract:
                raise ValueError(f"unknown X column is not in reviewed ex-ante contract: {column}")
            role, allowed = contract[column], True
        elif column == "run_key":
            role, allowed = "key", False
        elif column.startswith("eligible_") or column == "restricted_feature_reason":
            role, allowed = "eligibility", False
        else:
            matches = [(role, allowed) for prefix, role, allowed in prefix_roles if column.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"column has no unambiguous role: {column}")
            role, allowed = matches[0]
        manifest.append(
            {
                "column": column,
                "role": role,
                "exante_allowed": allowed,
                "dtype": str(master[column].dtype),
                "non_null_count": int(master[column].notna().sum()),
                "description": column.replace("_", " "),
            }
        )
    return manifest


def build_master_table(project_root: Path = ROOT) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    project_root = Path(project_root).resolve()
    master = _base_runs(project_root)
    planned = pd.read_parquet(project_root / "step3/analysis/planned_workload_features.parquet")
    effective = pd.read_parquet(project_root / "step3/analysis/effective_workload_features.parquet")
    models = pd.read_parquet(project_root / "step3/analysis/model_architecture_features.parquet")
    physics = pd.read_parquet(project_root / "step3/analysis/physics_features.parquet")
    reconciliation = json.loads(
        (project_root / "step3/reports/planned_workload_reconciliation.json").read_text(encoding="utf-8")
    )

    for block in (
        _planned_block(planned),
        _effective_block(effective),
        _physics_block(physics),
        _reconciliation_block(reconciliation),
    ):
        master = master.merge(block, on="run_key", how="left", validate="one_to_one")
    master = master.merge(_model_block(models), on="diag_model_id", how="left", validate="many_to_one")
    _assert_unique(master, "run_key", "master feature table", EXPECTED_RUNS)
    master = _derive_eligibility(master)
    return master, make_role_manifest(master, project_root)


def _coverage_records(master: pd.DataFrame, dimension: str, column: str | None) -> list[dict[str, Any]]:
    populations = (("all", master), ("stable", master[master["diag_quality_is_stable"]]))
    records: list[dict[str, Any]] = []
    for population, subset in populations:
        groups = [("all", subset)] if column is None else subset.groupby(column, dropna=False, sort=True)
        for value, group in groups:
            row: dict[str, Any] = {
                "population": population,
                "dimension": dimension,
                "value": str(value),
                "runs": int(len(group)),
            }
            for flag in ("eligible_base_exante", "eligible_full_workload", "eligible_physics_feature"):
                count = int(group[flag].sum())
                row[f"{flag}_runs"] = count
                row[f"{flag}_rate"] = count / len(group) if len(group) else None
            records.append(row)
    return records


def build_coverage(master: pd.DataFrame) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for dimension, column in (
        ("overall", None),
        ("task", "x_protocol_task"),
        ("model_family", "diag_model_family"),
        ("gpu", "x_hardware_gpu_model"),
        ("architecture_family", "x_model_architecture_family"),
    ):
        records.extend(_coverage_records(master, dimension, column))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "ML.ENERGY V3 GPU-side text-LLM inference only",
        "run_count": int(len(master)),
        "stable_run_count": int(master["diag_quality_is_stable"].sum()),
        "records": records,
    }


def _dictionary_markdown(manifest: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 1.7 master feature dictionary",
        "",
        "The table has one row per ML.ENERGY V3 text-LLM inference run. Only rows marked `exante_allowed=true` may enter X. GPU-side targets, result diagnostics, quality outcomes, identifiers, and provenance are excluded from X.",
        "",
        "| Column | Role | X allowed | Dtype | Non-null |",
        "|---|---|---:|---|---:|",
    ]
    for item in manifest:
        lines.append(
            f"| `{item['column']}` | `{item['role']}` | {str(item['exante_allowed']).lower()} | `{item['dtype']}` | {item['non_null_count']} |"
        )
    lines.extend(
        [
            "",
            "## Eligibility",
            "",
            "- `eligible_base_exante`: core execution-time-known task/model/GPU/deployment/output-cap fields and all five GPU-side target fields are present.",
            "- `eligible_full_workload`: base eligibility plus resolved benchmark and effective input-workload features.",
            "- `eligible_physics_feature`: full-workload eligibility plus supported canonical architecture and finite physics proxies.",
            "- `eligible_primary_stable`: the prior Phase 1/1.5 post-run stability rule is true; this is a selection flag, never an X feature.",
            "- `restricted_feature_reason`: ordered machine-readable limitations; missing values are never imputed as zero.",
            "",
            "`x_workload_benchmark_*` retains the benchmark/client `SampleRequest.prompt_len` semantics. `x_workload_effective_*` includes resolved server-side chat-template/FIM formatting. Output-cap and decode fields are upper bounds, not actual generation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    master, manifest = build_master_table(ROOT)
    analysis = ROOT / "step3/analysis"
    reports = ROOT / "step3/reports"
    docs = ROOT / "step3/docs"
    master.to_parquet(analysis / "master_feature_table.parquet", index=False)
    master.to_csv(analysis / "master_feature_table.csv", index=False)
    (analysis / "feature_role_manifest.json").write_text(
        json.dumps({"columns": manifest}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports / "eligibility_coverage.json").write_text(
        json.dumps(build_coverage(master), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (docs / "master_feature_dictionary.md").write_text(_dictionary_markdown(manifest), encoding="utf-8")
    print(
        f"wrote {len(master)} master rows; stable={int(master['eligible_primary_stable'].sum())}; "
        f"full={int(master['eligible_full_workload'].sum())}; physics={int(master['eligible_physics_feature'].sum())}"
    )


if __name__ == "__main__":
    main()
