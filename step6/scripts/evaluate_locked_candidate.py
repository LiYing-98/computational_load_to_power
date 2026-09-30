#!/usr/bin/env python3
"""Fit the locked candidate, evaluate the confirmatory set once, and build diagnostic OOF."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from step6.scripts.common import STEP6, load_feature_bundles, load_json, sha256_file, write_json
from step6.scripts.modeling import (
    _fit,
    fit_predict_outer_fold,
    make_estimator,
    metadata_json,
    regression_metrics,
    select_candidate,
)


TARGET = "y_gpu_avg_power_watts"
RECEIPT = STEP6 / "analysis/confirmatory_evaluation_receipt.json"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _confirmatory_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    slices = [("overall", "all", predictions)]
    for column in ["diag_parent_lineage", "x_hardware_gpu_model", "x_hardware_num_gpus"]:
        for value, group in predictions.groupby(column, dropna=False):
            slices.append((column, str(value), group))
    for slice_type, slice_value, group in slices:
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "n_rows": len(group),
                "n_models": int(group["diag_model_id"].nunique()),
                **regression_metrics(group[TARGET], group["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def _existing_is_valid(lock_path: Path, prediction_path: Path) -> bool:
    if not RECEIPT.exists():
        return False
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    required = {
        prediction_path: receipt["confirmatory_predictions_sha256"],
        STEP6 / "analysis/full_cohort_oof_predictions.parquet": receipt["full_cohort_oof_sha256"],
        STEP6 / "models/phase4_candidate_model.joblib": receipt["candidate_model_sha256"],
    }
    if sha256_file(lock_path) != receipt["locked_candidate_sha256"]:
        raise RuntimeError("locked candidate changed after confirmatory evaluation")
    for path, expected_hash in required.items():
        if not path.exists() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"frozen confirmatory artifact missing or changed: {path}")
    return True


def _full_cohort_oof(
    frame: pd.DataFrame,
    *,
    selected: dict,
    bundle: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["diag_model_id", "run_key"]).reset_index(drop=True)
    splitter = GroupKFold(n_splits=5)
    prediction_chunks: list[pd.DataFrame] = []
    fold_chunks: list[pd.DataFrame] = []
    for fold_id, (train_index, test_index) in enumerate(
        splitter.split(ordered, groups=ordered["diag_model_id"])
    ):
        train = ordered.iloc[train_index]
        held_out = ordered.iloc[test_index]
        prediction, metadata, _ = fit_predict_outer_fold(
            train,
            held_out,
            numeric=bundle["numeric"],
            categorical=bundle["categorical"],
            target=TARGET,
            algorithm=selected["algorithm"],
            candidates=selected["hyperparameter_candidates"],
            inner_group_column="diag_model_id",
            objective=selected["objective"],
            use_high_power_weights=selected["use_high_power_weights"],
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
        result["fold_id"] = fold_id
        result["selection_metadata_json"] = metadata_json(metadata)
        result["evaluation_scheme"] = "full_cohort_model_group_oof_diagnostic"
        prediction_chunks.append(result)
        manifest = held_out[["run_key", "diag_model_id"]].copy()
        manifest["fold_id"] = fold_id
        fold_chunks.append(manifest)
    predictions = pd.concat(prediction_chunks, ignore_index=True).sort_values("run_key").reset_index(drop=True)
    folds = pd.concat(fold_chunks, ignore_index=True).sort_values("run_key").reset_index(drop=True)
    return predictions, folds


def main() -> None:
    lock_path = STEP6 / "config/locked_candidate_phase4.json"
    prediction_path = STEP6 / "analysis/confirmatory_predictions.parquet"
    if _existing_is_valid(lock_path, prediction_path):
        print("confirmatory evaluation already frozen and verified; no files rewritten")
        return
    if prediction_path.exists() or RECEIPT.exists():
        raise RuntimeError("partial confirmatory evaluation exists; refusing to overwrite")

    lock = load_json("step6/config/locked_candidate_phase4.json")
    if lock.get("status") != "final_locked_before_confirmatory_evaluation":
        raise RuntimeError("candidate must be final-locked before confirmatory evaluation")
    selected = lock["selected"]
    table_path = STEP6 / "analysis/phase4_power_modeling_table.parquet"
    frame = pd.read_parquet(table_path)
    development = frame.loc[frame["eligible_phase4_development"].astype(bool)].copy()
    confirmatory = frame.loc[frame["diag_is_confirmatory_model"].astype(bool)].copy()
    if set(development["diag_model_id"]) & set(confirmatory["diag_model_id"]):
        raise RuntimeError("confirmatory model IDs leaked into development data")
    bundle = load_feature_bundles()["bundles"][selected["feature_bundle"]]
    selected_index, inner_scores = select_candidate(
        development,
        numeric=bundle["numeric"],
        categorical=bundle["categorical"],
        target=TARGET,
        algorithm=selected["algorithm"],
        candidates=selected["hyperparameter_candidates"],
        inner_group_column="diag_model_id",
        objective=selected["objective"],
        use_high_power_weights=selected["use_high_power_weights"],
    )
    final_candidate = selected["hyperparameter_candidates"][selected_index]
    estimator = make_estimator(
        selected["algorithm"],
        final_candidate,
        bundle["numeric"],
        bundle["categorical"],
        objective=selected["objective"],
    )
    _fit(
        estimator,
        development,
        bundle["columns"],
        TARGET,
        use_high_power_weights=selected["use_high_power_weights"],
    )
    prediction = np.asarray(estimator.predict(confirmatory[bundle["columns"]]), dtype=float)
    confirmatory_output = confirmatory[
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
    confirmatory_output["prediction"] = prediction
    confirmatory_output["residual"] = prediction - confirmatory_output[TARGET].to_numpy(dtype=float)
    confirmatory_output["evaluation_scheme"] = "locked_confirmatory_model_holdout"
    _atomic_parquet(confirmatory_output, prediction_path)

    full_oof, full_folds = _full_cohort_oof(frame, selected=selected, bundle=bundle)
    full_path = STEP6 / "analysis/full_cohort_oof_predictions.parquet"
    _atomic_parquet(full_oof, full_path)
    (STEP6 / "splits").mkdir(parents=True, exist_ok=True)
    full_folds.to_csv(STEP6 / "splits/full_cohort_model_folds.csv", index=False)

    model_path = STEP6 / "models/phase4_candidate_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_path)
    model_metadata = {
        "schema_version": "1.0",
        "algorithm": selected["algorithm"],
        "feature_bundle": selected["feature_bundle"],
        "feature_columns": bundle["columns"],
        "target": TARGET,
        "objective": selected["objective"],
        "use_high_power_weights": selected["use_high_power_weights"],
        "selected_hyperparameter_index": selected_index,
        "selected_hyperparameters": final_candidate,
        "inner_group_cv_mae": inner_scores,
        "training_rows": len(development),
        "training_models": int(development["diag_model_id"].nunique()),
        "locked_candidate_sha256": sha256_file(lock_path),
        "table_sha256": sha256_file(table_path),
    }
    write_json(STEP6 / "models/phase4_candidate_metadata.json", model_metadata)

    metric_path = STEP6 / "reports/confirmatory_metrics.csv"
    _confirmatory_metric_rows(confirmatory_output).to_csv(metric_path, index=False)
    development_predictions = pd.read_parquet(
        STEP6 / "analysis/development_oof_predictions.parquet"
    )
    development_selected = development_predictions.loc[
        development_predictions["algorithm"].eq(selected["algorithm"])
        & development_predictions["feature_bundle"].eq(selected["feature_bundle"])
    ]
    validity_rows: list[dict] = []
    for scheme, values in [
        ("development_model_group_oof", development_selected),
        ("locked_confirmatory_model_holdout", confirmatory_output),
        ("full_cohort_model_group_oof_diagnostic", full_oof),
    ]:
        validity_rows.append(
            {
                "evaluation_scheme": scheme,
                "n_rows": len(values),
                "n_unique_runs": int(values["run_key"].nunique()),
                "n_models": int(values["diag_model_id"].nunique()),
                "prediction_finite": bool(np.isfinite(values["prediction"]).all()),
                "prediction_positive": bool((values["prediction"] > 0).all()),
                "minimum_prediction_w": float(values["prediction"].min()),
                "maximum_prediction_w": float(values["prediction"].max()),
            }
        )
    pd.DataFrame(validity_rows).to_csv(STEP6 / "reports/prediction_validity.csv", index=False)

    receipt = {
        "schema_version": "1.0",
        "evaluated_once": True,
        "policy": "Idempotent re-runs verify hashes and never refit or rewrite confirmatory artifacts.",
        "locked_candidate_sha256": sha256_file(lock_path),
        "holdout_definition_sha256": sha256_file(STEP6 / "config/confirmatory_model_holdout.json"),
        "confirmatory_predictions_sha256": sha256_file(prediction_path),
        "full_cohort_oof_sha256": sha256_file(full_path),
        "candidate_model_sha256": sha256_file(model_path),
        "prediction_rows": len(confirmatory_output),
        "held_out_model_ids": sorted(confirmatory_output["diag_model_id"].unique()),
        "development_rows": len(development),
    }
    write_json(RECEIPT, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
