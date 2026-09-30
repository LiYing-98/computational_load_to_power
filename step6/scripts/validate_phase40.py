#!/usr/bin/env python3
"""Independently validate the Phase 4.0 cohort, isolation, metrics, and artifacts.

This module intentionally does not import Phase 4 modeling or analysis helpers.
"""

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
DEFAULT_TASKBOOK = Path(
    r"D:\my_work\cpecc\files\日常事务\科技项目\2025年\算电协同\课题一\阶段性研究进展\plan\first_step\Codex_Phase4_Taskbook.docx"
)

REQUIRED_ARTIFACTS = [
    "step6/README.md",
    "step6/config/source_freeze_phase4.json",
    "step6/config/lineage_taxonomy.json",
    "step6/config/feature_bundles_phase4.json",
    "step6/config/confirmatory_model_holdout.json",
    "step6/config/locked_candidate_phase4.json",
    "step6/analysis/phase4_power_modeling_table.parquet",
    "step6/analysis/development_oof_predictions.parquet",
    "step6/analysis/confirmatory_predictions.parquet",
    "step6/analysis/full_cohort_oof_predictions.parquet",
    "step6/analysis/confirmatory_evaluation_receipt.json",
    "step6/splits/full_cohort_lineage_holdout_folds.csv",
    "step6/splits/full_cohort_within_task_lineage_folds.csv",
    "step6/splits/full_cohort_architecture_holdout_folds.csv",
    "step6/reports/main_model_comparison.csv",
    "step6/reports/hardware_ablation.csv",
    "step6/reports/hardware_bootstrap_draws.csv",
    "step6/reports/loss_strategy_comparison.csv",
    "step6/reports/generalization_by_lineage.csv",
    "step6/reports/within_task_lineage_holdout.csv",
    "step6/reports/architecture_holdout.csv",
    "step6/reports/error_slices.csv",
    "step6/reports/confirmatory_metrics.csv",
    "step6/reports/prediction_validity.csv",
    "step6/reports/phase4_power_model_report.md",
    "step6/models/phase4_candidate_model.joblib",
    "step6/models/phase4_candidate_metadata.json",
    "step6/notebooks/phase4_power_modeling.ipynb",
    "step6/notebooks/phase4_power_modeling.html",
]

REQUIRED_FIGURES = [
    "target_distribution.png",
    "task_parent_lineage_heatmap.png",
    "architecture_parent_lineage_heatmap.png",
    "development_actual_vs_predicted.png",
    "confirmatory_actual_vs_predicted.png",
    "residual_vs_measured_power.png",
    "residual_by_gpu_count.png",
    "residual_by_gpu_model.png",
    "residual_by_parent_lineage.png",
    "top_power_quantile_bias.png",
    "hardware_paired_mae_delta.png",
    "lineage_holdout_mae_bias.png",
]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    text_suffixes = {".csv", ".html", ".ipynb", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
    digest = hashlib.sha256()
    if path.suffix.lower() in text_suffixes:
        canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(canonical.encode("utf-8"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        details = function()
        return {"name": name, "status": "pass", "details": details}
    except Exception as exc:  # validator must report all failed contracts
        return {"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_gpu_avg_power_watts"].to_numpy(dtype=float)
    pred = frame["prediction"].to_numpy(dtype=float)
    residual = pred - y
    abs_error = np.abs(residual)
    denominator = np.abs(y) + np.abs(pred)
    total = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan"),
        "mdape": float(np.median(abs_error / np.abs(y)) * 100.0),
        "smape": float(np.mean(np.divide(2.0 * abs_error, denominator, out=np.zeros_like(abs_error), where=denominator != 0)) * 100.0),
        "signed_bias": float(residual.mean()),
    }


def _source_check(root: Path) -> dict[str, Any]:
    freeze = _json(root / "step6/config/source_freeze_phase4.json")
    for item in freeze["frozen_inputs"]:
        path = root / item["path"]
        _require(path.exists(), f"missing frozen input: {item['path']}")
        _require(_sha256(path) == item["sha256"], f"frozen input changed: {item['path']}")
    taskbook = Path(os.environ.get("MLENERGY_PHASE4_TASKBOOK", str(DEFAULT_TASKBOOK)))
    _require(taskbook.exists(), f"taskbook not found: {taskbook}")
    _require(_sha256(taskbook) == freeze["taskbook_sha256"], "taskbook SHA-256 differs")
    return {"frozen_inputs": len(freeze["frozen_inputs"]), "taskbook_sha256": freeze["taskbook_sha256"]}


def _cohort_and_feature_check(root: Path) -> dict[str, Any]:
    frame = pd.read_parquet(root / "step6/analysis/phase4_power_modeling_table.parquet")
    _require(len(frame) == 565 and frame["run_key"].nunique() == 565, "cohort grain differs")
    _require(frame["diag_model_id"].nunique() == 27, "model count differs")
    _require(int(frame["diag_parent_lineage"].eq("Llama").sum()) == 117, "Llama count differs")
    _require(set(frame["diag_parent_lineage"]) == {"Qwen", "Llama", "DeepSeek", "Gemma", "Nemotron", "GPT-OSS"}, "lineages differ")
    _require(set(frame["diag_architecture_class"]) == {"dense_transformer", "mixture_of_experts", "hybrid_mamba_transformer"}, "architectures differ")
    rebuilt_power = frame["x_hardware_rated_power_w"] * frame["x_hardware_num_gpus"]
    _require(np.allclose(rebuilt_power, frame["x_hardware_aggregate_rated_power_w"], rtol=0, atol=1e-12), "aggregate rated-power formula differs")

    phase2 = _json(root / "step4/config/feature_bundles.json")["bundles"]
    raw = _json(root / "step6/config/feature_bundles_phase4.json")
    resolved: dict[str, dict[str, list[str]]] = {}
    pending = dict(raw["bundles"])
    while pending:
        progressed = False
        for name, bundle in list(pending.items()):
            if "inherit_columns_from" in bundle:
                source = phase2[bundle["inherit_columns_from"].split(":", 1)[1]]
                categorical, numeric = list(source["categorical"]), list(source["numeric"])
            elif "base_bundle" in bundle:
                if bundle["base_bundle"] not in resolved:
                    continue
                categorical = list(resolved[bundle["base_bundle"]]["categorical"])
                numeric = list(resolved[bundle["base_bundle"]]["numeric"])
            else:
                categorical, numeric = [], []
            categorical.extend(bundle.get("categorical", []))
            categorical.extend(bundle.get("categorical_add", []))
            numeric.extend(bundle.get("numeric", []))
            numeric.extend(bundle.get("numeric_add", []))
            drop = set(bundle.get("drop", []))
            resolved[name] = {
                "categorical": list(dict.fromkeys(x for x in categorical if x not in drop)),
                "numeric": list(dict.fromkeys(x for x in numeric if x not in drop)),
            }
            del pending[name]
            progressed = True
        _require(progressed, f"unresolved feature bundles: {sorted(pending)}")

    forbidden_identity = {"diag_model_id", "diag_parent_lineage", "diag_subfamily"}
    forbidden_fragments = ("throughput", "latency", "duration", "utilization", "actual_batch", "result_", "measured_")
    for name in ("P0-Core", "P1-AggregatePower", "P2-HardwareStatic"):
        columns = resolved[name]["categorical"] + resolved[name]["numeric"]
        _require(frame[columns].notna().all().all(), f"{name} does not cover all 565 rows")
        _require(not forbidden_identity.intersection(columns), f"identity leakage in {name}")
        _require(all(column.startswith("x_") or column == "diag_architecture_class" for column in columns), f"non-ex-ante namespace in {name}")
        _require(not any(fragment in column.lower() for column in columns for fragment in forbidden_fragments), f"post-run feature in {name}")
    _require("x_hardware_gpu_model" not in resolved["P2-HardwareStatic-NoLabel"]["categorical"], "NoLabel retained GPU label")
    _require(frame["y_gpu_avg_power_watts"].notna().all(), "target missing")
    return {"rows": len(frame), "models": frame["diag_model_id"].nunique(), "llama_rows": 117, "main_bundles_complete": 3}


def _split_and_holdout_check(root: Path) -> dict[str, Any]:
    table = pd.read_parquet(root / "step6/analysis/phase4_power_modeling_table.parquet")
    folds = pd.read_csv(root / "step6/splits/development_model_folds.csv")
    holdout = _json(root / "step6/config/confirmatory_model_holdout.json")
    held = set(holdout["held_out_model_ids"])
    _require(len(held) == 6, "confirmatory model count differs")
    _require("y_gpu_avg_power_watts" not in json.dumps(holdout), "holdout definition contains y")
    _require(len(folds) == 449 and folds["run_key"].nunique() == 449, "development split support differs")
    _require(folds["fold_id"].nunique() == 5, "development fold count differs")
    _require((folds.groupby("diag_model_id")["fold_id"].nunique() == 1).all(), "model-group split leakage")
    _require(not held.intersection(folds["diag_model_id"]), "confirmatory model entered development folds")
    _require(set(table.loc[table["diag_is_confirmatory_model"], "diag_model_id"]) == held, "holdout flag differs")
    return {"development_rows": len(folds), "development_models": folds["diag_model_id"].nunique(), "held_out_models": len(held), "folds": 5}


def _development_prediction_check(root: Path) -> dict[str, Any]:
    predictions = pd.read_parquet(root / "step6/analysis/development_oof_predictions.parquet")
    comparison = pd.read_csv(root / "step6/reports/main_model_comparison.csv")
    _require(len(predictions) == 24 * 449, "development prediction matrix differs")
    _require(len(comparison) == 24, "comparison cell count differs")
    _require(predictions.groupby(["algorithm", "feature_bundle"])["run_key"].nunique().eq(449).all(), "incomplete OOF cell")
    _require(not predictions.duplicated(["algorithm", "feature_bundle", "run_key"]).any(), "duplicate OOF prediction")
    for _, row in comparison.iterrows():
        subset = predictions.loc[predictions["algorithm"].eq(row["algorithm"]) & predictions["feature_bundle"].eq(row["feature_bundle"])]
        for metric, value in _metrics(subset).items():
            _require(np.isclose(value, float(row[metric]), rtol=0, atol=1e-9), f"metric mismatch: {row['algorithm']} {row['feature_bundle']} {metric}")
    lock = _json(root / "step6/config/locked_candidate_phase4.json")
    _require(lock["status"] == "final_locked_before_confirmatory_evaluation", "candidate is not pre-confirmatory locked")
    _require(lock["selected"]["strategy_id"] == "RandomForest_P2-HardwareStatic_L1", "locked strategy differs")
    return {"prediction_rows": len(predictions), "cells": len(comparison), "all_cell_metrics_rebuilt": len(comparison), "locked_strategy": lock["selected"]["strategy_id"]}


def _ablation_and_loss_check(root: Path) -> dict[str, Any]:
    ablation = pd.read_csv(root / "step6/reports/hardware_ablation.csv")
    draws = pd.read_csv(root / "step6/reports/hardware_bootstrap_draws.csv")
    _require(len(ablation) == 18 and len(draws) == 18_000, "ablation/bootstrap support differs")
    _require((ablation["n_paired_rows"] == 449).all() and (ablation["n_models"] == 21).all(), "paired support differs")
    for _, row in ablation.iterrows():
        subset = draws.loc[draws["algorithm"].eq(row["algorithm"]) & draws["comparison"].eq(row["comparison"])]
        values = subset["row_weighted_delta_mae"].to_numpy(dtype=float)
        equal_model_values = subset["equal_model_delta_mae"].to_numpy(dtype=float)
        _require(len(values) == 1000, "bootstrap replicate count differs")
        _require(np.isclose(np.percentile(values, 2.5), row["row_weighted_ci_low"], atol=1e-9), "bootstrap low differs")
        _require(np.isclose(np.percentile(values, 97.5), row["row_weighted_ci_high"], atol=1e-9), "bootstrap high differs")
        _require(np.isclose(np.percentile(equal_model_values, 2.5), row["equal_model_ci_low"], atol=1e-9), "equal-model bootstrap low differs")
        _require(np.isclose(np.percentile(equal_model_values, 97.5), row["equal_model_ci_high"], atol=1e-9), "equal-model bootstrap high differs")

        development = pd.read_parquet(root / "step6/analysis/development_oof_predictions.parquet")
        before = development.loc[
            development["algorithm"].eq(row["algorithm"])
            & development["feature_bundle"].eq(row["before_bundle"]),
            ["run_key", "diag_model_id", "fold_id", "y_gpu_avg_power_watts", "prediction"],
        ].rename(columns={"prediction": "before_prediction"})
        after = development.loc[
            development["algorithm"].eq(row["algorithm"])
            & development["feature_bundle"].eq(row["after_bundle"]),
            ["run_key", "diag_model_id", "fold_id", "y_gpu_avg_power_watts", "prediction"],
        ].rename(columns={"prediction": "after_prediction"})
        paired = before.merge(
            after,
            on=["run_key", "diag_model_id", "fold_id", "y_gpu_avg_power_watts"],
            validate="one_to_one",
        )
        delta = (
            (paired["after_prediction"] - paired["y_gpu_avg_power_watts"]).abs()
            - (paired["before_prediction"] - paired["y_gpu_avg_power_watts"]).abs()
        )
        equal_model = delta.groupby(paired["diag_model_id"]).mean().mean()
        _require(len(paired) == int(row["n_paired_rows"]), "paired run/fold support differs")
        _require(np.isclose(delta.mean(), row["row_weighted_delta_mae"], atol=1e-9), "paired row delta differs")
        _require(np.isclose(equal_model, row["equal_model_delta_mae"], atol=1e-9), "paired equal-model delta differs")
    losses = pd.read_csv(root / "step6/reports/loss_strategy_comparison.csv")
    _require(len(losses) == 7 and set(losses["algorithm"]) == {"RandomForest", "XGBoost"}, "loss scope differs")
    _require(set(losses["evaluation_weighting"]) == {"unweighted"}, "weighted evaluation found")
    _require(losses.sort_values("mae").iloc[0]["strategy_id"] == "RandomForest_P2-HardwareStatic_L1", "loss winner differs")
    return {"paired_comparisons": len(ablation), "bootstrap_draws": len(draws), "loss_strategies": len(losses)}


def _confirmatory_check(root: Path) -> dict[str, Any]:
    path = root / "step6/analysis/confirmatory_predictions.parquet"
    confirmatory = pd.read_parquet(path)
    receipt = _json(root / "step6/analysis/confirmatory_evaluation_receipt.json")
    _require(receipt["evaluated_once"] is True, "confirmatory is not marked one-shot")
    _require(_sha256(root / "step6/config/locked_candidate_phase4.json") == receipt["locked_candidate_sha256"], "locked candidate hash differs")
    _require(_sha256(root / "step6/config/confirmatory_model_holdout.json") == receipt["holdout_definition_sha256"], "holdout definition hash differs")
    _require(_sha256(root / "step6/models/phase4_candidate_model.joblib") == receipt["candidate_model_sha256"], "candidate model hash differs")
    _require(_sha256(path) == receipt["confirmatory_predictions_sha256"], "confirmatory prediction hash differs")
    _require(len(confirmatory) == 116 and confirmatory["run_key"].nunique() == 116, "confirmatory support differs")
    _require(set(confirmatory["diag_model_id"]) == set(receipt["held_out_model_ids"]), "confirmatory model IDs differ")
    reported = pd.read_csv(root / "step6/reports/confirmatory_metrics.csv")
    overall = reported.loc[reported["slice_type"].eq("overall")].iloc[0]
    for metric, value in _metrics(confirmatory).items():
        _require(np.isclose(value, float(overall[metric]), rtol=0, atol=1e-9), f"confirmatory metric differs: {metric}")
    full_path = root / "step6/analysis/full_cohort_oof_predictions.parquet"
    full = pd.read_parquet(full_path)
    _require(_sha256(full_path) == receipt["full_cohort_oof_sha256"], "full OOF hash differs")
    _require(len(full) == 565 and full["run_key"].nunique() == 565, "full OOF support differs")
    return {"rows": len(confirmatory), "models": confirmatory["diag_model_id"].nunique(), **{key: round(value, 9) for key, value in _metrics(confirmatory).items()}, "full_oof_rows": len(full)}


def _generalization_check(root: Path) -> dict[str, Any]:
    lineage = pd.read_csv(root / "step6/reports/generalization_by_lineage.csv")
    within = pd.read_csv(root / "step6/reports/within_task_lineage_holdout.csv")
    architecture = pd.read_csv(root / "step6/reports/architecture_holdout.csv")
    _require(set(lineage["diag_parent_lineage"]) == {"Qwen", "Llama", "DeepSeek", "Gemma", "Nemotron", "GPT-OSS"}, "lineage stress coverage differs")
    _require(within.groupby("x_protocol_task")["diag_parent_lineage"].nunique().ge(3).all(), "within-task eligibility differs")
    _require(set(architecture["diag_architecture_class"]) == {"dense_transformer", "mixture_of_experts", "hybrid_mamba_transformer"}, "architecture stress coverage differs")
    _require("task_lineage_confounded" in within, "task-lineage confounding flag missing")
    predictions = pd.read_parquet(root / "step6/analysis/generalization_predictions.parquet")
    contracts = [
        (
            "parent_lineage_holdout",
            root / "step6/splits/full_cohort_lineage_holdout_folds.csv",
            lineage,
            [("held_out_group", "diag_parent_lineage")],
        ),
        (
            "within_task_lineage_holdout",
            root / "step6/splits/full_cohort_within_task_lineage_folds.csv",
            within,
            [("held_constant_task", "x_protocol_task"), ("held_out_group", "diag_parent_lineage")],
        ),
        (
            "architecture_class_holdout_stress",
            root / "step6/splits/full_cohort_architecture_holdout_folds.csv",
            architecture,
            [("held_out_group", "diag_architecture_class")],
        ),
    ]
    rebuilt_cells = 0
    for scheme, manifest_path, summary, keys in contracts:
        manifest = pd.read_csv(manifest_path)
        observed = predictions.loc[predictions["evaluation_scheme"].eq(scheme)]
        _require(len(observed) == len(manifest), f"{scheme} manifest row count differs")
        _require(set(observed["run_key"]) == set(manifest["run_key"]), f"{scheme} manifest run support differs")
        _require(manifest["post_lock_diagnostic"].all(), f"{scheme} is not marked post-lock")
        group_columns = [prediction_name for prediction_name, _ in keys]
        for group_values, group in observed.groupby(group_columns, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            mask = pd.Series(True, index=summary.index)
            for value, (_, summary_name) in zip(group_values, keys):
                mask &= summary[summary_name].eq(value)
            _require(mask.sum() == 1, f"{scheme} summary cell missing")
            reported = summary.loc[mask].iloc[0]
            for metric, value in _metrics(group).items():
                _require(np.isclose(value, float(reported[metric]), atol=1e-9), f"{scheme} metric differs: {group_values} {metric}")
            rebuilt_cells += 1
    return {"lineages": len(lineage), "within_task_cells": len(within), "architectures": len(architecture), "confounded_cells": int(within["task_lineage_confounded"].sum()), "prediction_cells_rebuilt": rebuilt_cells}


def _delivery_check(root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]
    _require(not missing, f"missing artifacts: {missing}")
    small = [name for name in REQUIRED_FIGURES if not (root / "step6/reports/figures" / name).exists() or (root / "step6/reports/figures" / name).stat().st_size <= 5000]
    _require(not small, f"missing/trivial figures: {small}")
    report = (root / "step6/reports/phase4_power_model_report.md").read_text(encoding="utf-8")
    for token in ("565", "117", "P0 → P1", "P1 → P2", "P2-NoLabel", "Confirmatory", "unseen model", "unseen parent lineage", "within-task unseen lineage", "unseen architecture class", "推荐候选", "适用边界", "主要剩余问题"):
        _require(token in report, f"report missing: {token}")
    notebook = _json(root / "step6/notebooks/phase4_power_modeling.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    _require(code_cells and all(cell.get("execution_count") is not None for cell in code_cells), "notebook is not fully executed")
    _require(not any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])), "notebook contains an error")
    code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    _require(".fit(" not in code and "joblib.load" not in code, "notebook is not read-only evidence")
    return {"required_artifacts": len(REQUIRED_ARTIFACTS), "figures": len(REQUIRED_FIGURES), "notebook_code_cells": len(code_cells)}


def _credential_check(root: Path) -> dict[str, Any]:
    patterns = {
        "huggingface": re.compile(r"\bhf" + r"_[A-Za-z0-9]{20,}\b"),
        "openai": re.compile(r"\bsk" + r"-[A-Za-z0-9_-]{20,}\b"),
        "github_classic": re.compile(r"\bgh" + r"[pousr]_[A-Za-z0-9]{30,}\b"),
        "github_fine_grained": re.compile(r"\bgithub" + r"_pat_[A-Za-z0-9_]{20,}\b"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    excluded = {".venv", ".packages", "checkpoints", "__pycache__"}
    hits: list[str] = []
    scanned = 0
    for path in (root / "step6").rglob("*"):
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
    return {"text_files_scanned": scanned, "patterns": sorted(patterns), "hits": 0}


def run_validations(project_root: Path = ROOT, *, write: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / "step6/reports/verification_receipt.json"
    prior_generated_utc: str | None = None
    if output.exists():
        try:
            prior_generated_utc = _json(output).get("generated_utc")
        except (json.JSONDecodeError, OSError):
            prior_generated_utc = None
    checks = [
        _check("frozen sources and actual taskbook", lambda: _source_check(root)),
        _check("565-run cohort and pure-ex-ante feature boundary", lambda: _cohort_and_feature_check(root)),
        _check("development split and confirmatory isolation", lambda: _split_and_holdout_check(root)),
        _check("development OOF coverage and independent metric rebuilds", lambda: _development_prediction_check(root)),
        _check("paired hardware ablation and loss scope", lambda: _ablation_and_loss_check(root)),
        _check("one-shot confirmatory hashes and metrics", lambda: _confirmatory_check(root)),
        _check("lineage, within-task, and architecture stress coverage", lambda: _generalization_check(root)),
        _check("required report, notebook, model, tables, and figures", lambda: _delivery_check(root)),
        _check("Phase 4 credential-pattern scan", lambda: _credential_check(root)),
    ]
    artifact_hashes = {
        name: _sha256(root / name)
        for name in REQUIRED_ARTIFACTS
        if (root / name).exists()
    }
    for name in REQUIRED_FIGURES:
        path = root / "step6/reports/figures" / name
        if path.exists():
            artifact_hashes[str(path.relative_to(root)).replace("\\", "/")] = _sha256(path)
    receipt = {
        "schema_version": "1.0",
        "generated_utc": prior_generated_utc or datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "scope": "ML.ENERGY V3 GPU-side stable high-load text-LLM inference Phase 4.0",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "summary": {
            "candidate": "RandomForest_P2-HardwareStatic_L1",
            "stable_runs": 565,
            "development_oof_mae_watts": 272.40183938292745,
            "confirmatory_rows": 116,
            "confirmatory_mae_watts": 295.2639480032137,
            "confirmatory_r2": 0.9209481025880099,
        },
        "checks": checks,
        "artifact_hashes": artifact_hashes,
    }
    if write:
        output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    receipt = run_validations(ROOT, write=True)
    print(json.dumps({"status": receipt["status"], "checks": len(receipt["checks"]), **receipt["summary"]}, indent=2, ensure_ascii=False))
    if receipt["status"] != "pass":
        for check in receipt["checks"]:
            if check["status"] == "fail":
                print(f"FAIL: {check['name']}: {check['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
