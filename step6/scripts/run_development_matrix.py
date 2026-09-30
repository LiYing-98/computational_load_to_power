#!/usr/bin/env python3
"""Run the Phase 4 six-algorithm development model-holdout matrix."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pandas as pd

from step6.scripts.common import (
    ROOT,
    STEP6,
    load_feature_bundles,
    load_json,
    sha256_file,
    stable_hash,
    write_json,
)
from step6.scripts.checkpoints import canonicalize_experiment_inputs, checkpoint_fingerprint
from step6.scripts.modeling import fit_predict_outer_fold, metadata_json, regression_metrics


TARGET = "y_gpu_avg_power_watts"
ALGORITHMS = [
    "DummyMean",
    "Ridge",
    "HistGradientBoosting",
    "RandomForest",
    "CatBoost",
    "XGBoost",
]
BUNDLES = ["P0-Core", "P1-AggregatePower", "P2-HardwareStatic", "P2-HardwareStatic-NoLabel"]


def _checkpoint_path(algorithm: str, bundle: str, fold_id: int, fingerprint: str) -> Path:
    key = stable_hash(f"development|{algorithm}|{bundle}|{fold_id}|{fingerprint}")[:16]
    return STEP6 / "models/checkpoints" / f"{key}_{algorithm}_{bundle}_{fold_id}.parquet"


def run_cell(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    algorithm: str,
    bundle_name: str,
    bundle: dict,
    candidates: list[dict],
) -> pd.DataFrame:
    frame, folds = canonicalize_experiment_inputs(frame, folds)
    fingerprint = checkpoint_fingerprint(
        frame,
        folds,
        spec={"algorithm": algorithm, "bundle_name": bundle_name},
        bundle=bundle,
        candidates=candidates,
        target=TARGET,
        inner_group_column="diag_model_id",
        implementation_paths=[
            Path(__file__),
            STEP6 / "scripts/modeling.py",
            STEP6 / "scripts/checkpoints.py",
        ],
    )
    chunks: list[pd.DataFrame] = []
    for fold_id in sorted(folds["fold_id"].unique()):
        test_keys = set(folds.loc[folds["fold_id"].eq(fold_id), "run_key"])
        held_out = frame.loc[frame["run_key"].isin(test_keys)].copy()
        train = frame.loc[~frame["run_key"].isin(test_keys)].copy()
        checkpoint = _checkpoint_path(algorithm, bundle_name, int(fold_id), fingerprint)
        if checkpoint.exists():
            cached = pd.read_parquet(checkpoint)
            if (
                set(cached["run_key"]) == test_keys
                and "checkpoint_fingerprint" in cached
                and set(cached["checkpoint_fingerprint"]) == {fingerprint}
            ):
                chunks.append(cached.drop(columns="checkpoint_fingerprint"))
                continue
        prediction, metadata, _ = fit_predict_outer_fold(
            train,
            held_out,
            numeric=bundle["numeric"],
            categorical=bundle["categorical"],
            target=TARGET,
            algorithm=algorithm,
            candidates=candidates,
            inner_group_column="diag_model_id",
        )
        result = held_out[
            [
                "run_key",
                "diag_model_id",
                "diag_parent_lineage",
                "diag_subfamily",
                "diag_architecture_class",
                "x_protocol_task",
                "x_hardware_gpu_model",
                "x_hardware_num_gpus",
                TARGET,
            ]
        ].copy()
        result["prediction"] = prediction
        result["residual"] = prediction - result[TARGET].to_numpy(dtype=float)
        result["algorithm"] = algorithm
        result["feature_bundle"] = bundle_name
        result["fold_id"] = int(fold_id)
        result["selection_metadata_json"] = metadata_json(metadata)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        result.assign(checkpoint_fingerprint=fingerprint).to_parquet(checkpoint, index=False)
        chunks.append(result)
    return pd.concat(chunks, ignore_index=True).sort_values("run_key").reset_index(drop=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (algorithm, bundle), group in predictions.groupby(["algorithm", "feature_bundle"], sort=False):
        rows.append(
            {
                "algorithm": algorithm,
                "feature_bundle": bundle,
                "n_rows": len(group),
                "n_models": int(group["diag_model_id"].nunique()),
                **regression_metrics(group[TARGET], group["prediction"]),
                "backend_modules": "|".join(
                    sorted(
                        {
                            json.loads(value)["backend_module"]
                            for value in group["selection_metadata_json"].unique()
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse", "algorithm", "feature_bundle"]).reset_index(drop=True)


def main() -> None:
    table_path = STEP6 / "analysis/phase4_power_modeling_table.parquet"
    fold_path = STEP6 / "splits/development_model_folds.csv"
    frame = pd.read_parquet(table_path)
    development = frame.loc[frame["eligible_phase4_development"].astype(bool)].copy()
    folds = pd.read_csv(fold_path)
    bundles = load_feature_bundles()["bundles"]
    search = load_json("step6/config/model_search_spaces_phase4.json")["algorithms"]
    results: list[pd.DataFrame] = []
    for algorithm in ALGORITHMS:
        for bundle_name in BUNDLES:
            print(f"running {algorithm} × {bundle_name}", flush=True)
            results.append(
                run_cell(
                    development,
                    folds,
                    algorithm=algorithm,
                    bundle_name=bundle_name,
                    bundle=bundles[bundle_name],
                    candidates=search[algorithm]["candidates"],
                )
            )
    predictions = pd.concat(results, ignore_index=True)
    analysis_dir = STEP6 / "analysis"
    report_dir = STEP6 / "reports"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(analysis_dir / "development_oof_predictions.parquet", index=False)
    summary = summarize(predictions)
    summary.to_csv(report_dir / "main_model_comparison.csv", index=False)
    write_json(
        analysis_dir / "training_provenance.json",
        {
            "schema_version": "1.0",
            "target": TARGET,
            "selection": "group-aware inner CV by diag_model_id within every outer-train fold",
            "outer_test_used_for_selection": False,
            "table_sha256": sha256_file(table_path),
            "folds_sha256": sha256_file(fold_path),
            "feature_config_sha256": sha256_file(STEP6 / "config/feature_bundles_phase4.json"),
            "search_config_sha256": sha256_file(STEP6 / "config/model_search_spaces_phase4.json"),
            "python": platform.python_version(),
            "rows": len(development),
            "models": int(development["diag_model_id"].nunique()),
            "experiments": len(summary),
        },
    )
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
