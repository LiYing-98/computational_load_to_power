#!/usr/bin/env python3
"""Independent end-to-end validation for ML.ENERGY Phase 3.0."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STEP5 = ROOT / "step5"
EXPECTED_TASKBOOK_SHA256 = "DDCCAF572D291B6D454351C18FDEAA6961C9570A5B5B37591CF00F6F805C4052"
DEFAULT_TASKBOOK = Path(
    r"D:\my_work\cpecc\files\日常事务\科技项目\2025年\算电协同\课题一\阶段性研究进展\plan\first_step"
    r"\Codex_Phase3.0任务书_高负载GPU功率能耗增强预测与硬件物理建模.docx"
)

REQUIRED_ARTIFACTS = (
    "step5/config/gpu_hardware_specs.json",
    "step5/config/hardware_source_manifest.json",
    "step5/config/feature_bundles_phase3.json",
    "step5/analysis/phase3_modeling_table.parquet",
    "step5/analysis/oof_predictions.parquet",
    "step5/analysis/oof_predictions.csv",
    "step5/analysis/generalization_predictions.parquet",
    "step5/analysis/hardware_label_sensitivity_predictions.parquet",
    "step5/analysis/target_strategy_predictions.parquet",
    "step5/reports/model_comparison.csv",
    "step5/reports/feature_ablation.csv",
    "step5/reports/generalization_ladder.csv",
    "step5/reports/hardware_label_sensitivity.csv",
    "step5/reports/hardware_paired_comparisons.csv",
    "step5/reports/hardware_paired_bootstrap_draws.csv",
    "step5/reports/target_strategy_comparison.csv",
    "step5/reports/feature_importance.csv",
    "step5/reports/partial_dependence.csv",
    "step5/reports/error_slices.csv",
    "step5/notebooks/phase3_enhanced_modeling.ipynb",
    "step5/notebooks/phase3_enhanced_modeling.html",
    "step5/reports/phase3_enhanced_modeling_report.md",
    "step5/README.md",
)

REQUIRED_FIGURES = (
    "actual_vs_predicted_random_best.png",
    "actual_vs_predicted_model_holdout_best.png",
    "actual_vs_predicted_family_holdout_best.png",
    "residuals_model_holdout_best.png",
    "actual_vs_predicted_energy_best.png",
    "residuals_energy_best.png",
    "feature_importance_best.png",
    "partial_dependence_best.png",
    "physics_coverage_by_gpu_family.png",
    "load_regime_distribution.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"name": name, "status": "pass", "details": function()}
    except Exception as error:
        return {"name": name, "status": "fail", "error": f"{type(error).__name__}: {error}"}


def _metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residual = pred - true
    absolute = np.abs(residual)
    denominator = np.abs(true) + np.abs(pred)
    nonzero_true = np.abs(true) > 1e-12
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((true - true.mean()) ** 2)),
        "mdape_percent": float(np.median(absolute[nonzero_true] / np.abs(true[nonzero_true])) * 100),
        "smape_percent": float(np.mean(np.divide(2 * absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)) * 100),
    }


def _resolved_bundles(root: Path) -> dict[str, dict[str, Any]]:
    phase3 = _json(root / "step5/config/feature_bundles_phase3.json")
    bundles = dict(_json(root / phase3["inherits"])["bundles"])
    pending = dict(phase3["bundles"])
    while pending:
        progressed = False
        for name, spec in list(pending.items()):
            base_name = spec.get("base_bundle")
            if base_name not in bundles:
                continue
            base = bundles[base_name]
            drop = set(spec.get("drop", []))
            categorical = [value for value in [*base.get("categorical", []), *spec.get("categorical_add", [])] if value not in drop]
            numeric = [value for value in [*base.get("numeric", []), *spec.get("numeric_add", [])] if value not in drop]
            bundles[name] = {**spec, "categorical": list(dict.fromkeys(categorical)), "numeric": list(dict.fromkeys(numeric))}
            bundles[name]["columns"] = [*bundles[name]["categorical"], *bundles[name]["numeric"]]
            del pending[name]
            progressed = True
        _require(progressed, f"unresolved bundles: {sorted(pending)}")
    return bundles


def _source_check(root: Path) -> dict[str, Any]:
    freeze = _json(root / "step5/config/source_freeze_phase3.json")
    for item in freeze["frozen_inputs"]:
        path = root / item["path"]
        _require(path.exists(), f"missing frozen input: {item['path']}")
        _require(_sha256(path) == item["sha256"], f"frozen input changed: {item['path']}")
    taskbook = Path(os.environ.get("MLENERGY_PHASE3_TASKBOOK", str(DEFAULT_TASKBOOK)))
    _require(taskbook.exists(), f"actual Phase 3 taskbook not found: {taskbook}")
    actual = _sha256(taskbook).upper()
    _require(actual == EXPECTED_TASKBOOK_SHA256, "actual taskbook hash differs")
    _require(freeze["taskbook_sha256"].upper() == actual, "stored taskbook hash differs from actual DOCX")
    return {"verified_frozen_inputs": len(freeze["frozen_inputs"]), "actual_taskbook_sha256": actual}


def _cohort_and_formula_check(root: Path) -> dict[str, Any]:
    frame = pd.read_parquet(root / "step5/analysis/phase3_modeling_table.parquet")
    expected = {"rows": 694, "eligible_primary_stable": 565, "eligible_static_architecture": 448, "eligible_phase20_full": 381, "eligible_phase20_physics": 305}
    _require(len(frame) == expected["rows"] and frame["run_key"].nunique() == len(frame), "modeling-table grain differs")
    for column, count in expected.items():
        if column != "rows":
            _require(int(frame[column].sum()) == count, f"{column} count differs")
    profiles = _json(root / "step5/config/gpu_hardware_specs.json")["profiles"]
    precision_field = {"bfloat16": "peak_compute_bf16_dense_tflops", "fp8": "peak_compute_fp8_dense_tflops", "mxfp4": "peak_compute_fp4_dense_tflops"}
    physics = frame.loc[frame["eligible_phase20_physics"]]
    for _, row in physics.iterrows():
        profile = profiles[str(row["x_hardware_gpu_model"])]
        peak = float(profile[precision_field[str(row["x_model_benchmark_weight_precision"]).lower()]])
        model_parallel = float(row["x_deployment_model_parallel_gpus_per_replica"])
        aggregate_power = float(profile["rated_power_w"]) * float(row["x_hardware_num_gpus"])
        compute_time = float(row["x_physics_prefill_total_flop_per_request"]) / (peak * 1e12 * model_parallel)
        kv_mean = float(row["x_physics_kv_cache_prefill_mean_request_bytes"]) / model_parallel
        memory_time = (float(row["x_physics_weight_bytes_per_gpu_ideal"]) + kv_mean) / (float(profile["hbm_bandwidth_gbps"]) * 1e9)
        capacity = float(profile["hbm_capacity_gb"]) * 1e9
        rebuilt = {
            "x_hardware_aggregate_rated_power_w": aggregate_power,
            "x_hardware_physics_compute_time_proxy_seconds": compute_time,
            "x_hardware_physics_memory_time_proxy_seconds": memory_time,
            "x_hardware_physics_weight_capacity_ratio": float(row["x_physics_weight_bytes_per_gpu_ideal"]) / capacity,
            "x_hardware_physics_kv_capacity_ratio_mean": kv_mean / capacity,
            "x_hardware_physics_compute_to_memory_pressure_ratio": compute_time / memory_time,
            "y_gpu_power_fraction_of_rated": float(row["y_gpu_avg_power_watts"]) / aggregate_power,
        }
        for column, value in rebuilt.items():
            _require(np.isclose(float(row[column]), value, rtol=1e-12, atol=1e-12), f"formula mismatch: {column}")
    return {**expected, "independently_rebuilt_physics_rows": len(physics)}


def _feature_boundary_check(root: Path) -> dict[str, Any]:
    bundles = _resolved_bundles(root)
    roles = {item["column"]: item for item in _json(root / "step3/analysis/feature_role_manifest.json")["columns"]}
    phase2_bundles = _json(root / "step4/config/feature_bundles.json")["bundles"]
    frozen_phase2_columns = {
        column for bundle in phase2_bundles.values() for column in bundle["columns"]
    }
    new_allowed = {column for bundle in bundles.values() for column in bundle["columns"] if column.startswith("x_hardware_")}
    forbidden_fragments = ("actual_batch", "throughput", "latency", "duration", "utilization", "temperature", "clock", "measured_", "output_tokens_actual")
    violations: list[str] = []
    for bundle_name, bundle in bundles.items():
        for column in bundle["columns"]:
            manifest_row = roles.get(column)
            if manifest_row is not None and not manifest_row["exante_allowed"]:
                violations.append(f"{bundle_name}:{column}:{manifest_row['role']}")
            if manifest_row is None and column not in frozen_phase2_columns and column not in new_allowed:
                violations.append(f"{bundle_name}:{column}:unmanifested")
            if not column.startswith("x_") or any(fragment in column.lower() for fragment in forbidden_fragments):
                violations.append(f"{bundle_name}:{column}:forbidden-name")
    _require(not violations, f"feature-role violations: {violations}")
    profiles = _json(root / "step5/config/gpu_hardware_specs.json")["profiles"]
    _require({profile["hardware_source_status"] for profile in profiles.values()} == {"ambiguous"}, "hardware certainty overstated")
    return {"bundles": len(bundles), "role_manifest_violations": 0, "hardware_mapping": "ambiguous"}


def _split_check(root: Path) -> dict[str, Any]:
    model = pd.read_csv(root / "step4/splits/model_group_folds.csv").query("cohort == 'physics'")
    config = pd.read_csv(root / "step4/splits/config_group_folds.csv")
    physics_keys = set(
        pd.read_parquet(root / "step5/analysis/phase3_modeling_table.parquet")
        .loc[lambda frame: frame["eligible_phase20_physics"], "run_key"]
    )
    config = config.loc[config["run_key"].isin(physics_keys)].copy()
    family = pd.read_csv(root / "step5/splits/family_group_folds.csv").query("cohort == 'physics'")
    random = pd.read_csv(root / "step5/splits/random_split_manifest.csv").query("cohort == 'physics'")
    _require((model.groupby("diag_model_id")["outer_fold"].nunique() == 1).all(), "model-group leakage")
    _require((config.groupby("config_group_id")["outer_fold"].nunique() == 1).all(), "config-group leakage")
    _require((family.groupby("diag_model_family")["outer_fold"].nunique() == 1).all(), "family-group leakage")
    _require(random["repeat_index"].nunique() == 10, "random repeat count differs")
    sizes = config.groupby("config_group_id").size()
    _require(len(config) == 305 and len(sizes) == 303 and int((sizes == 1).sum()) == 301, "config-group support changed")
    return {"model_folds": int(model["outer_fold"].nunique()), "config_groups": len(sizes), "singleton_config_groups": int((sizes == 1).sum()), "family_folds": int(family["outer_fold"].nunique()), "random_repeats": 10}


def _prediction_check(root: Path) -> dict[str, Any]:
    predictions = pd.read_parquet(root / "step5/analysis/oof_predictions.parquet")
    _require(len(predictions) == 36600 and predictions["experiment_id"].nunique() == 120, "core matrix coverage differs")
    identity = ["target", "bundle", "model_name", "split_scheme", "target_transform", "target_strategy", "run_key"]
    _require(not predictions.duplicated(identity).any(), "duplicate prediction identity")
    _require((predictions.groupby("experiment_id").size() == 305).all(), "experiment support differs")
    _require("checkpoint_fingerprint" in predictions and predictions["checkpoint_fingerprint"].str.fullmatch(r"[0-9a-f]{64}").all(), "checkpoint fingerprints missing")
    comparison = pd.read_csv(root / "step5/reports/model_comparison.csv")
    _require(len(comparison) == 120, "comparison coverage differs")
    for _, reported in comparison.iterrows():
        rows = predictions.loc[predictions["experiment_id"].eq(reported["experiment_id"])]
        for metric, value in _metrics(rows["y_true"], rows["y_pred"]).items():
            _require(np.isclose(value, float(reported[metric]), rtol=0, atol=1e-9), f"metric mismatch: {reported['experiment_id']} {metric}")
    with (root / "step5/analysis/oof_predictions.csv").open("r", encoding="utf-8") as handle:
        csv_rows = sum(1 for _ in handle) - 1
    _require(csv_rows == len(predictions), "CSV/parquet prediction row count differs")
    return {"rows": len(predictions), "experiments": predictions["experiment_id"].nunique(), "all_metrics_rebuilt": len(comparison)}


def _paired_and_strategy_check(root: Path) -> dict[str, Any]:
    paired = pd.read_csv(root / "step5/reports/hardware_paired_comparisons.csv")
    draws = pd.read_csv(root / "step5/reports/hardware_paired_bootstrap_draws.csv")
    _require(len(paired) == 4 and len(draws) == 4000, "paired-comparison coverage differs")
    for _, row in paired.iterrows():
        subset = draws.loc[draws["reference_variant"].eq(row["reference_variant"]) & draws["candidate_variant"].eq(row["candidate_variant"]), "equal_model_mae_delta_candidate_minus_reference"].to_numpy()
        _require(len(subset) == int(row["bootstrap_replicates"]), "paired draw support differs")
        _require(np.isclose(np.percentile(subset, 2.5), row["equal_model_mae_delta_ci_low"]), "paired CI low mismatch")
        _require(np.isclose(np.percentile(subset, 97.5), row["equal_model_mae_delta_ci_high"]), "paired CI high mismatch")
    strategies = pd.read_parquet(root / "step5/analysis/target_strategy_predictions.parquet")
    fraction = strategies.loc[strategies["target_strategy"].eq("power_fraction")].merge(
        pd.read_parquet(root / "step5/analysis/phase3_modeling_table.parquet")[["run_key", "x_hardware_aggregate_rated_power_w"]], on="run_key", validate="one_to_one"
    )
    rebuilt = fraction["strategy_scale_prediction"] * fraction["x_hardware_aggregate_rated_power_w"]
    _require(np.allclose(rebuilt, fraction["y_pred"], rtol=1e-12, atol=1e-9), "power-fraction reconstruction mismatch")
    summary = pd.read_csv(root / "step5/reports/target_strategy_comparison.csv")
    log_rows = summary.loc[summary["target"].eq("y_gpu_energy_per_token_joules") & summary["target_transform"].eq("log1p")]
    _require(int(log_rows.iloc[0]["negative_prediction_count"]) == 0, "negative log1p prediction observed")
    return {"paired_comparisons": len(paired), "bootstrap_draws": len(draws), "power_fraction_rows_rebuilt": len(fraction), "log1p_semantics": "no negative predictions observed; positivity not guaranteed"}


def _generalization_and_report_check(root: Path) -> dict[str, Any]:
    summary = pd.read_csv(root / "step5/reports/generalization_ladder.csv")
    _require(set(summary["split_scheme"]) == {"random", "config_group", "model_group", "family_group"}, "generalization schemes differ")
    _require((summary.query("split_scheme == 'random'").groupby("model_variant")["repeat_index"].nunique() == 10).all(), "random repeat coverage differs")
    _require(set(summary.loc[summary["model_variant"].eq("B4-Hardware__XGBoost"), "analysis_role"]) == {"outcome_informed_exploratory_followup"}, "exploratory role missing")
    report = (root / "step5/reports/phase3_enhanced_modeling_report.md").read_text(encoding="utf-8")
    for token in ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "GPU 侧", "high-load steady-state", "探索性候选", "获胜者选择偏差", "301 个单例"):
        _require(token in report, f"report missing: {token}")
    hardware = pd.read_csv(root / "step5/reports/hardware_label_sensitivity.csv")
    b4 = hardware.loc[hardware["bundle"].eq("B4-Hardware")].iloc[0]
    _require(f"{b4['mae']:.1f} W" in report, "report B4 MAE does not reconcile")
    notebook = _json(root / "step5/notebooks/phase3_enhanced_modeling.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    _require(code_cells and all(cell.get("execution_count") is not None for cell in code_cells), "notebook not fully executed")
    _require(not any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])), "notebook contains errors")
    code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    _require(".fit(" not in code and "make_estimator" not in code, "notebook contains fitting")
    return {"generalization_rows": len(summary), "report_b4_mae_reconciled": float(b4["mae"]), "notebook_code_cells": len(code_cells)}


def _interpretation_check(root: Path) -> dict[str, Any]:
    importance = pd.read_csv(root / "step5/reports/feature_importance.csv")
    pdp = pd.read_csv(root / "step5/reports/partial_dependence.csv")
    _require(importance["model_variant"].nunique() >= 2, "fewer than two interpreted variants")
    _require(set(importance["fit_scope"]) == {"outer_train_only"}, "importance fit scope differs")
    _require(set(importance["method"]) == {"outer_test_permutation"}, "importance method differs")
    _require(set(importance["outer_fold"]) == {0, 1, 2, 3, 4}, "importance folds differ")
    _require(4 <= pdp["feature"].nunique() <= 6, "PDP feature count differs")
    _require(set(pdp["fit_scope"]) == {"outer_train_only"}, "PDP fit scope differs")
    _require(set(pdp["outer_fold"]) == {0, 1, 2, 3, 4}, "PDP folds differ")
    return {"importance_rows": len(importance), "interpreted_variants": importance["model_variant"].nunique(), "pdp_features": pdp["feature"].nunique()}


def _artifact_check(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_ARTIFACTS if not (root / path).exists()]
    _require(not missing, f"missing artifacts: {missing}")
    small = [name for name in REQUIRED_FIGURES if not (root / "step5/reports/figures" / name).exists() or (root / "step5/reports/figures" / name).stat().st_size <= 20_000]
    _require(not small, f"missing/trivial figures: {small}")
    return {"required_artifacts": len(REQUIRED_ARTIFACTS), "required_figures": len(REQUIRED_FIGURES)}


def _credential_check(root: Path) -> dict[str, Any]:
    # Repository credential-pattern scan; this is not a general-purpose secret scanner.
    patterns = {
        "huggingface": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "openai": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        "github_classic": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
        "github_fine_grained": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "slack": re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    excluded = {".venv", ".packages", "analysis_checkpoints", "checkpoints", "__pycache__"}
    hits: list[str] = []
    scanned = 0
    for path in (root / "step5").rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".json", ".csv", ".txt", ".ipynb", ".html"}:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in patterns.items():
            if pattern.search(text):
                hits.append(f"{path.relative_to(root)}:{label}")
    _require(not hits, f"credential-like values found: {hits}")
    return {"text_files_scanned": scanned, "credential_patterns": sorted(patterns), "hits": 0}


def run_validations(project_root: Path = ROOT, *, write: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    checks = [
        _check("frozen source and actual taskbook hashes", lambda: _source_check(root)),
        _check("cohort counts and independent B4/B5 formulas", lambda: _cohort_and_formula_check(root)),
        _check("feature-role manifest and hardware uncertainty", lambda: _feature_boundary_check(root)),
        _check("split isolation and config-group support", lambda: _split_check(root)),
        _check("core OOF coverage and all metric rebuilds", lambda: _prediction_check(root)),
        _check("paired intervals and target reconstruction", lambda: _paired_and_strategy_check(root)),
        _check("generalization, report values, and notebook", lambda: _generalization_and_report_check(root)),
        _check("fold-local interpretation artifacts", lambda: _interpretation_check(root)),
        _check("required artifacts and figures", lambda: _artifact_check(root)),
        _check("repository credential-pattern scan", lambda: _credential_check(root)),
    ]
    predictions = pd.read_parquet(root / "step5/analysis/oof_predictions.parquet")
    hardware = pd.read_csv(root / "step5/reports/hardware_label_sensitivity.csv")
    b4 = hardware.loc[hardware["bundle"].eq("B4-Hardware")].iloc[0]
    artifact_hashes = {path: _sha256(root / path) for path in REQUIRED_ARTIFACTS if (root / path).exists()}
    for name in REQUIRED_FIGURES:
        path = root / "step5/reports/figures" / name
        if path.exists():
            artifact_hashes[str(path.relative_to(root)).replace("\\", "/")] = _sha256(path)
    receipt = {
        "schema_version": "2.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "scope": "ML.ENERGY V3 GPU-side text-LLM inference, high-load steady-state Phase 3.0",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "summary": {
            "core_prediction_rows": len(predictions),
            "core_experiments": int(predictions["experiment_id"].nunique()),
            "exploratory_power_variant": "B4-Hardware__XGBoost",
            "exploratory_power_mae_watts": float(b4["mae"]),
            "selection_status": "outcome-informed exploratory; no winner-adjusted interval claimed",
        },
        "checks": checks,
        "artifact_hashes": artifact_hashes,
    }
    if write:
        (root / "step5/reports/verification_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt


def main() -> None:
    receipt = run_validations(ROOT, write=True)
    print(json.dumps({"status": receipt["status"], "checks": len(receipt["checks"]), **receipt["summary"]}, indent=2))
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
