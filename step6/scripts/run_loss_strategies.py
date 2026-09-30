#!/usr/bin/env python3
"""Compare bounded loss strategies for the two development candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from step6.scripts.common import STEP6, load_feature_bundles, load_json, sha256_file, stable_hash, write_json
from step6.scripts.checkpoints import canonicalize_experiment_inputs, checkpoint_fingerprint
from step6.scripts.modeling import fit_predict_outer_fold, metadata_json, regression_metrics


TARGET = "y_gpu_avg_power_watts"


def strategy_plan() -> list[dict]:
    shortlist_artifact = load_json("step6/config/development_candidate_shortlist.json")
    plan: list[dict] = []
    for candidate in shortlist_artifact["candidate_shortlist"]:
        algorithm = candidate["algorithm"]
        bundle = candidate["feature_bundle"]
        strategies = [
            ("L1", "absolute_error", False),
            ("L2", "squared_error", False),
            ("L1_weighted", "absolute_error", True),
        ]
        if algorithm == "XGBoost":
            strategies.insert(2, ("PseudoHuber", "pseudo_huber", False))
        for suffix, objective, weighted in strategies:
            plan.append(
                {
                    "strategy_id": f"{algorithm}_{bundle}_{suffix}",
                    "algorithm": algorithm,
                    "feature_bundle": bundle,
                    "objective": objective,
                    "use_high_power_weights": weighted,
                }
            )
    return plan


def _checkpoint_path(strategy_id: str, fold_id: int, fingerprint: str) -> Path:
    key = stable_hash(f"loss|{strategy_id}|{fold_id}|{fingerprint}")[:16]
    return STEP6 / "models/checkpoints" / f"{key}_loss_{fold_id}.parquet"


def run_strategy(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    spec: dict,
    bundle: dict,
    candidates: list[dict],
) -> pd.DataFrame:
    frame, folds = canonicalize_experiment_inputs(frame, folds)
    fingerprint = checkpoint_fingerprint(
        frame,
        folds,
        spec=spec,
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
        checkpoint = _checkpoint_path(spec["strategy_id"], int(fold_id), fingerprint)
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
            algorithm=spec["algorithm"],
            candidates=candidates,
            inner_group_column="diag_model_id",
            objective=spec["objective"],
            use_high_power_weights=spec["use_high_power_weights"],
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
        result["fold_id"] = int(fold_id)
        for key, value in spec.items():
            result[key] = value
        result["selection_metadata_json"] = metadata_json(metadata)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        result.assign(checkpoint_fingerprint=fingerprint).to_parquet(checkpoint, index=False)
        chunks.append(result)
    return pd.concat(chunks, ignore_index=True).sort_values("run_key").reset_index(drop=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    q75 = float(predictions[TARGET].quantile(0.75))
    q90 = float(predictions[TARGET].quantile(0.90))
    rows: list[dict] = []
    for strategy_id, group in predictions.groupby("strategy_id", sort=False):
        top25 = group.loc[group[TARGET] >= q75]
        top10 = group.loc[group[TARGET] >= q90]
        lower75 = group.loc[group[TARGET] < q75]
        rows.append(
            {
                "strategy_id": strategy_id,
                "algorithm": group["algorithm"].iloc[0],
                "feature_bundle": group["feature_bundle"].iloc[0],
                "objective": group["objective"].iloc[0],
                "high_power_weighting": bool(group["use_high_power_weights"].iloc[0]),
                "evaluation_weighting": "unweighted",
                "n_rows": len(group),
                **regression_metrics(group[TARGET], group["prediction"]),
                "top25_mae": regression_metrics(top25[TARGET], top25["prediction"])["mae"],
                "top25_signed_bias": regression_metrics(top25[TARGET], top25["prediction"])["signed_bias"],
                "top10_mae": regression_metrics(top10[TARGET], top10["prediction"])["mae"],
                "top10_signed_bias": regression_metrics(top10[TARGET], top10["prediction"])["signed_bias"],
                "lower75_mae": regression_metrics(lower75[TARGET], lower75["prediction"])["mae"],
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse", "strategy_id"]).reset_index(drop=True)


def choose_final(summary: pd.DataFrame) -> pd.Series:
    best = summary.iloc[0]
    qualified = summary.loc[
        (summary["mae"] <= 1.05 * best["mae"])
        & (summary["top25_mae"] <= 0.95 * best["top25_mae"])
        & (summary["top25_signed_bias"].abs() <= 0.80 * abs(float(best["top25_signed_bias"])))
    ]
    if qualified.empty:
        return best
    return qualified.sort_values(["top25_mae", "mae", "strategy_id"]).iloc[0]


def main() -> None:
    table = pd.read_parquet(STEP6 / "analysis/phase4_power_modeling_table.parquet")
    development = table.loc[table["eligible_phase4_development"].astype(bool)].copy()
    folds = pd.read_csv(STEP6 / "splits/development_model_folds.csv")
    bundles = load_feature_bundles()["bundles"]
    search = load_json("step6/config/model_search_spaces_phase4.json")["algorithms"]
    results: list[pd.DataFrame] = []
    for spec in strategy_plan():
        print(f"running {spec['strategy_id']}", flush=True)
        results.append(
            run_strategy(
                development,
                folds,
                spec,
                bundles[spec["feature_bundle"]],
                search[spec["algorithm"]]["candidates"],
            )
        )
    predictions = pd.concat(results, ignore_index=True)
    summary = summarize(predictions)
    selected = choose_final(summary)
    prediction_path = STEP6 / "analysis/loss_strategy_predictions.parquet"
    report_path = STEP6 / "reports/loss_strategy_comparison.csv"
    predictions.to_parquet(prediction_path, index=False)
    summary.to_csv(report_path, index=False)
    selected_spec = next(item for item in strategy_plan() if item["strategy_id"] == selected["strategy_id"])
    selected_payload = {
        **selected_spec,
        "development_metrics": {
            key: float(selected[key])
            for key in ["mae", "rmse", "r2", "mdape", "smape", "signed_bias", "top25_mae", "top25_signed_bias", "top10_mae", "top10_signed_bias", "lower75_mae"]
        },
        "hyperparameter_candidates": search[selected_spec["algorithm"]]["candidates"],
    }
    write_json(
        STEP6 / "config/locked_candidate_phase4.json",
        {
            "schema_version": "1.0",
            "status": "final_locked_before_confirmatory_evaluation",
            "decision_rule": "lowest development OOF MAE unless a strategy within 5% improves top-quartile MAE by >=5% and absolute top-quartile bias by >=20%",
            "development_predictions_sha256": sha256_file(prediction_path),
            "loss_comparison_sha256": sha256_file(report_path),
            "candidate_shortlist_source_sha256": sha256_file(
                STEP6 / "config/development_candidate_shortlist.json"
            ),
            "selected": selected_payload,
        },
    )
    print(summary.to_string(index=False))
    print(json.dumps({"selected": selected_payload}, indent=2))


if __name__ == "__main__":
    main()
