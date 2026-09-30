#!/usr/bin/env python3
"""Paired hardware ablation and pre-loss candidate shortlisting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from step6.scripts.common import SEED, STEP6, sha256_file, write_json
from step6.scripts.modeling import regression_metrics


TARGET = "y_gpu_avg_power_watts"
COMPARISONS = [
    ("P0-Core", "P1-AggregatePower", "P0_to_P1"),
    ("P1-AggregatePower", "P2-HardwareStatic", "P1_to_P2"),
    ("P2-HardwareStatic", "P2-HardwareStatic-NoLabel", "P2_to_P2_NoLabel"),
]


def _paired(predictions: pd.DataFrame, algorithm: str, before: str, after: str) -> pd.DataFrame:
    left = predictions.loc[
        predictions["algorithm"].eq(algorithm) & predictions["feature_bundle"].eq(before),
        ["run_key", "diag_model_id", TARGET, "prediction"],
    ].rename(columns={"prediction": "prediction_before"})
    right = predictions.loc[
        predictions["algorithm"].eq(algorithm) & predictions["feature_bundle"].eq(after),
        ["run_key", "prediction"],
    ].rename(columns={"prediction": "prediction_after"})
    merged = left.merge(right, on="run_key", how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"unpaired support for {algorithm} {before} -> {after}")
    merged["absolute_error_before"] = (merged["prediction_before"] - merged[TARGET]).abs()
    merged["absolute_error_after"] = (merged["prediction_after"] - merged[TARGET]).abs()
    return merged


def _bootstrap(paired: pd.DataFrame, *, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    model_ids = sorted(paired["diag_model_id"].unique())
    grouped = paired.groupby("diag_model_id", sort=True).agg(
        n_rows=("run_key", "size"),
        before_sum=("absolute_error_before", "sum"),
        after_sum=("absolute_error_after", "sum"),
        before_mean=("absolute_error_before", "mean"),
        after_mean=("absolute_error_after", "mean"),
    ).loc[model_ids]
    sampled_indices = rng.integers(0, len(model_ids), size=(draws, len(model_ids)))
    n_rows = grouped["n_rows"].to_numpy(dtype=float)[sampled_indices]
    before_sum = grouped["before_sum"].to_numpy(dtype=float)[sampled_indices]
    after_sum = grouped["after_sum"].to_numpy(dtype=float)[sampled_indices]
    denominator = n_rows.sum(axis=1)
    row_delta = after_sum.sum(axis=1) / denominator - before_sum.sum(axis=1) / denominator
    model_delta = (
        grouped["after_mean"].to_numpy(dtype=float) - grouped["before_mean"].to_numpy(dtype=float)
    )[sampled_indices].mean(axis=1)
    return pd.DataFrame(
        {
            "draw": np.arange(draws),
            "row_weighted_delta_mae": row_delta,
            "equal_model_delta_mae": model_delta,
        }
    )


def hardware_ablation(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    all_draws: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED)
    for algorithm in sorted(predictions["algorithm"].unique()):
        for before, after, label in COMPARISONS:
            paired = _paired(predictions, algorithm, before, after)
            per_model = paired.groupby("diag_model_id").agg(
                before=("absolute_error_before", "mean"), after=("absolute_error_after", "mean")
            )
            draws = _bootstrap(paired, draws=1000, rng=rng)
            draws.insert(0, "comparison", label)
            draws.insert(0, "algorithm", algorithm)
            all_draws.append(draws)
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "comparison": label,
                    "before_bundle": before,
                    "after_bundle": after,
                    "n_paired_rows": len(paired),
                    "n_models": int(paired["diag_model_id"].nunique()),
                    "row_weighted_delta_mae": float(
                        paired["absolute_error_after"].mean() - paired["absolute_error_before"].mean()
                    ),
                    "equal_model_delta_mae": float((per_model["after"] - per_model["before"]).mean()),
                    "row_weighted_ci_low": float(draws["row_weighted_delta_mae"].quantile(0.025)),
                    "row_weighted_ci_high": float(draws["row_weighted_delta_mae"].quantile(0.975)),
                    "equal_model_ci_low": float(draws["equal_model_delta_mae"].quantile(0.025)),
                    "equal_model_ci_high": float(draws["equal_model_delta_mae"].quantile(0.975)),
                }
            )
    return pd.DataFrame(summary_rows), pd.concat(all_draws, ignore_index=True)


def candidate_shortlist(predictions: pd.DataFrame) -> list[dict]:
    eligible = predictions.loc[
        predictions["feature_bundle"].isin(["P0-Core", "P1-AggregatePower", "P2-HardwareStatic"])
        & ~predictions["algorithm"].eq("DummyMean")
    ].copy()
    threshold = float(eligible[TARGET].quantile(0.75))
    rows: list[dict] = []
    for (algorithm, bundle), group in eligible.groupby(["algorithm", "feature_bundle"]):
        overall = regression_metrics(group[TARGET], group["prediction"])
        top = group.loc[group[TARGET] >= threshold]
        ordinary = group.loc[group[TARGET] < threshold]
        rows.append(
            {
                "algorithm": algorithm,
                "feature_bundle": bundle,
                "development_mae": overall["mae"],
                "development_rmse": overall["rmse"],
                "development_signed_bias": overall["signed_bias"],
                "top_quartile_mae": regression_metrics(top[TARGET], top["prediction"])["mae"],
                "top_quartile_signed_bias": regression_metrics(top[TARGET], top["prediction"])["signed_bias"],
                "lower_75pct_mae": regression_metrics(ordinary[TARGET], ordinary["prediction"])["mae"],
            }
        )
    ranked = sorted(rows, key=lambda row: (row["development_mae"], row["development_rmse"], row["algorithm"], row["feature_bundle"]))
    return ranked[:2]


def main() -> None:
    prediction_path = STEP6 / "analysis/development_oof_predictions.parquet"
    predictions = pd.read_parquet(prediction_path)
    ablation, draws = hardware_ablation(predictions)
    report_dir = STEP6 / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(report_dir / "hardware_ablation.csv", index=False)
    draws.to_csv(report_dir / "hardware_bootstrap_draws.csv", index=False)
    shortlist = candidate_shortlist(predictions)
    write_json(
        STEP6 / "config/development_candidate_shortlist.json",
        {
            "schema_version": "1.0",
            "selection_source_sha256": sha256_file(prediction_path),
            "candidate_shortlist": shortlist,
        },
    )
    write_json(
        STEP6 / "config/locked_candidate_phase4.json",
        {
            "schema_version": "1.0",
            "status": "development_shortlist_before_loss_comparison",
            "selection_source_sha256": sha256_file(prediction_path),
            "selection_primary_metric": "development model-holdout MAE",
            "guardrails": ["RMSE", "top-quartile MAE", "top-quartile signed bias", "lower-75% MAE"],
            "candidate_shortlist": shortlist,
        },
    )
    print(json.dumps({"ablation_rows": len(ablation), "bootstrap_rows": len(draws), "shortlist": shortlist}, indent=2))


if __name__ == "__main__":
    main()
