#!/usr/bin/env python3
"""Post-lock Phase 4 generalization tests, error slices, and figures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from step6.scripts.common import SEED, STEP6, load_feature_bundles, load_json
from step6.scripts.modeling import _fit, make_estimator, regression_metrics


TARGET = "y_gpu_avg_power_watts"
FIGURE_DIR = STEP6 / "reports/figures"


def _fit_fixed(train: pd.DataFrame, test: pd.DataFrame, bundle: dict, selected: dict) -> np.ndarray:
    estimator = make_estimator(
        selected["algorithm"],
        selected["hyperparameter_candidates"][selected["selected_hyperparameter_index"]],
        bundle["numeric"],
        bundle["categorical"],
        objective=selected["objective"],
    )
    _fit(
        estimator,
        train,
        bundle["columns"],
        TARGET,
        use_high_power_weights=selected["use_high_power_weights"],
    )
    return np.asarray(estimator.predict(test[bundle["columns"]]), dtype=float)


def _prediction_frame(test: pd.DataFrame, prediction: np.ndarray, **labels: object) -> pd.DataFrame:
    output = test[
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
    output["prediction"] = prediction
    output["residual"] = prediction - output[TARGET].to_numpy(dtype=float)
    for key, value in labels.items():
        output[key] = value
    return output


def _composition(values: pd.Series) -> str:
    return json.dumps(values.value_counts(dropna=False).sort_index().to_dict(), sort_keys=True)


def _summary(group: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n_runs": len(group),
        "n_models": int(group["diag_model_id"].nunique()),
        **regression_metrics(group[TARGET], group["prediction"]),
    }


def parent_lineage_holdout(
    frame: pd.DataFrame, bundle: dict, selected: dict, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for fold_id in sorted(manifest["fold_id"].unique()):
        assignment = manifest.loc[manifest["fold_id"].eq(fold_id)]
        lineage_values = assignment["held_out_group"].unique()
        if len(lineage_values) != 1:
            raise ValueError(f"lineage fold {fold_id} has {len(lineage_values)} held-out groups")
        lineage = lineage_values[0]
        test_keys = set(assignment["run_key"])
        test = frame.loc[frame["run_key"].isin(test_keys)]
        train = frame.loc[~frame["diag_parent_lineage"].eq(lineage)]
        if set(test["run_key"]) != test_keys or not test["diag_parent_lineage"].eq(lineage).all():
            raise ValueError(f"lineage manifest mismatch for {lineage}")
        result = _prediction_frame(
            test,
            _fit_fixed(train, test, bundle, selected),
            evaluation_scheme="parent_lineage_holdout",
            held_out_group=lineage,
        )
        predictions.append(result)
        summaries.append(
            {
                "diag_parent_lineage": lineage,
                **_summary(result),
                "gpu_composition_json": _composition(result["x_hardware_gpu_model"]),
                "gpu_count_composition_json": _composition(result["x_hardware_num_gpus"]),
                "task_composition_json": _composition(result["x_protocol_task"]),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def within_task_lineage_holdout(
    frame: pd.DataFrame, bundle: dict, selected: dict, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    summaries: list[dict] = []
    lineage_task_counts = frame.groupby("diag_parent_lineage")["x_protocol_task"].nunique()
    for fold_id in sorted(manifest["fold_id"].unique()):
        assignment = manifest.loc[manifest["fold_id"].eq(fold_id)]
        tasks = assignment["x_protocol_task"].unique()
        lineages = assignment["held_out_group"].unique()
        if len(tasks) != 1 or len(lineages) != 1:
            raise ValueError(f"within-task fold {fold_id} does not isolate one task-lineage cell")
        task, lineage = tasks[0], lineages[0]
        task_frame = frame.loc[frame["x_protocol_task"].eq(task)]
        test_keys = set(assignment["run_key"])
        test = task_frame.loc[task_frame["run_key"].isin(test_keys)]
        train = task_frame.loc[~task_frame["diag_parent_lineage"].eq(lineage)]
        if set(test["run_key"]) != test_keys or not test["diag_parent_lineage"].eq(lineage).all():
            raise ValueError(f"within-task manifest mismatch for {task}/{lineage}")
        result = _prediction_frame(
            test,
            _fit_fixed(train, test, bundle, selected),
            evaluation_scheme="within_task_lineage_holdout",
            held_out_group=lineage,
            held_constant_task=task,
        )
        predictions.append(result)
        summaries.append(
            {
                "x_protocol_task": task,
                "diag_parent_lineage": lineage,
                "task_lineage_confounded": bool(lineage_task_counts.loc[lineage] == 1),
                **_summary(result),
                "gpu_composition_json": _composition(result["x_hardware_gpu_model"]),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def architecture_holdout(
    frame: pd.DataFrame, bundle: dict, selected: dict, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for fold_id in sorted(manifest["fold_id"].unique()):
        assignment = manifest.loc[manifest["fold_id"].eq(fold_id)]
        architecture_values = assignment["held_out_group"].unique()
        if len(architecture_values) != 1:
            raise ValueError(f"architecture fold {fold_id} has {len(architecture_values)} held-out groups")
        architecture = architecture_values[0]
        test_keys = set(assignment["run_key"])
        test = frame.loc[frame["run_key"].isin(test_keys)]
        train = frame.loc[~frame["diag_architecture_class"].eq(architecture)]
        if set(test["run_key"]) != test_keys or not test["diag_architecture_class"].eq(architecture).all():
            raise ValueError(f"architecture manifest mismatch for {architecture}")
        result = _prediction_frame(
            test,
            _fit_fixed(train, test, bundle, selected),
            evaluation_scheme="architecture_class_holdout_stress",
            held_out_group=architecture,
        )
        predictions.append(result)
        summaries.append(
            {
                "diag_architecture_class": architecture,
                **_summary(result),
                "parent_lineage_composition_json": _composition(result["diag_parent_lineage"]),
                "gpu_composition_json": _composition(result["x_hardware_gpu_model"]),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def _calibration(group: pd.DataFrame) -> tuple[float, float]:
    actual = group[TARGET].to_numpy(dtype=float)
    predicted = group["prediction"].to_numpy(dtype=float)
    if len(group) < 2 or np.ptp(actual) == 0:
        return math.nan, math.nan
    slope, intercept = np.polyfit(actual, predicted, deg=1)
    return float(slope), float(intercept)


def error_slices(full_oof: pd.DataFrame) -> pd.DataFrame:
    working = full_oof.copy()
    working["measured_power_quartile"] = pd.qcut(
        working[TARGET], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    q75 = float(working[TARGET].quantile(0.75))
    q90 = float(working[TARGET].quantile(0.90))
    definitions: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", working)]
    for slice_type, column in [
        ("gpu_model", "x_hardware_gpu_model"),
        ("gpu_count", "x_hardware_num_gpus"),
        ("parent_lineage", "diag_parent_lineage"),
        ("architecture_class", "diag_architecture_class"),
        ("task", "x_protocol_task"),
        ("measured_power_quartile", "measured_power_quartile"),
    ]:
        for value, group in working.groupby(column, observed=True, dropna=False):
            definitions.append((slice_type, str(value), group))
    definitions.extend(
        [
            ("top_25_percent", f">={q75:.6f}W", working.loc[working[TARGET] >= q75]),
            ("top_10_percent", f">={q90:.6f}W", working.loc[working[TARGET] >= q90]),
        ]
    )
    rows: list[dict] = []
    for slice_type, slice_value, group in definitions:
        slope, intercept = _calibration(group)
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                **_summary(group),
                "residual_median": float(group["residual"].median()),
                "calibration_slope_predicted_vs_measured": slope,
                "calibration_intercept_w": intercept,
            }
        )
    return pd.DataFrame(rows)


def static_subset_experiment(frame: pd.DataFrame, bundles: dict, selected: dict) -> pd.DataFrame:
    names = ["S0-B1-Static", "S1-B1-Static-AggregatePower", "S2-B1-Static-HardwareStatic"]
    complete_columns = bundles["S2-B1-Static-HardwareStatic"]["columns"]
    support = frame.loc[frame[complete_columns].notna().all(axis=1)].copy()
    splitter = GroupKFold(n_splits=min(5, support["diag_model_id"].nunique()))
    splits = list(splitter.split(support, groups=support["diag_model_id"]))
    rows: list[dict] = []
    for name in names:
        chunks: list[pd.DataFrame] = []
        for fold_id, (train_index, test_index) in enumerate(splits):
            train = support.iloc[train_index]
            test = support.iloc[test_index]
            result = _prediction_frame(
                test,
                _fit_fixed(train, test, bundles[name], selected),
                fold_id=fold_id,
            )
            chunks.append(result)
        predictions = pd.concat(chunks, ignore_index=True)
        rows.append(
            {
                "feature_bundle": name,
                "n_rows": len(predictions),
                "n_models": int(predictions["diag_model_id"].nunique()),
                **regression_metrics(predictions[TARGET], predictions["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _scatter(actual: pd.Series, predicted: pd.Series, title: str, name: str) -> None:
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    pad = 0.04 * (high - low)
    limits = (low - pad, high + pad)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(actual, predicted, s=24, alpha=0.7, color="#176B87", edgecolor="none")
    ax.plot(limits, limits, linestyle="--", color="#B42318", linewidth=1.5, label="y = x")
    ax.set(xlabel="Measured aggregate GPU power (W)", ylabel="Predicted power (W)", title=title, xlim=limits, ylim=limits)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    ax.grid(color="#D0D5DD", linewidth=0.5, alpha=0.7)
    _save(fig, name)


def _heatmap(matrix: pd.DataFrame, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(matrix.columns)), max(3.8, 0.75 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = int(matrix.iloc[row, column])
            ax.text(column, row, str(value), ha="center", va="center", fontsize=8, color="black")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Stable runs")
    _save(fig, name)


def build_figures(
    frame: pd.DataFrame,
    development: pd.DataFrame,
    confirmatory: pd.DataFrame,
    full_oof: pd.DataFrame,
    lineage_summary: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(frame[TARGET], bins=24, color="#176B87", edgecolor="white")
    ax.set(title="All 565 stable runs: aggregate GPU power", xlabel="Measured power (W)", ylabel="Runs")
    _save(fig, "target_distribution.png")

    _heatmap(
        frame.pivot_table(index="x_protocol_task", columns="diag_parent_lineage", values="run_key", aggfunc="count", fill_value=0),
        "Task × parent-lineage coverage",
        "task_parent_lineage_heatmap.png",
    )
    _heatmap(
        frame.pivot_table(index="diag_architecture_class", columns="diag_parent_lineage", values="run_key", aggfunc="count", fill_value=0),
        "Architecture class × parent lineage",
        "architecture_parent_lineage_heatmap.png",
    )
    _scatter(development[TARGET], development["prediction"], "Development model-holdout OOF", "development_actual_vs_predicted.png")
    _scatter(confirmatory[TARGET], confirmatory["prediction"], "Locked confirmatory model holdout", "confirmatory_actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(full_oof[TARGET], full_oof["residual"], s=20, alpha=0.65, color="#176B87", edgecolor="none")
    ax.axhline(0, linestyle="--", color="#B42318")
    ax.set(title="Residual vs measured power", xlabel="Measured power (W)", ylabel="Prediction − measured (W)")
    _save(fig, "residual_vs_measured_power.png")

    for column, name, title in [
        ("x_hardware_num_gpus", "residual_by_gpu_count.png", "Residual by GPU count"),
        ("x_hardware_gpu_model", "residual_by_gpu_model.png", "Residual by GPU model"),
        ("diag_parent_lineage", "residual_by_parent_lineage.png", "Residual by parent lineage"),
    ]:
        categories = sorted(full_oof[column].unique(), key=str)
        fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(categories)), 4.8))
        ax.boxplot([full_oof.loc[full_oof[column].eq(value), "residual"] for value in categories], tick_labels=[str(value) for value in categories], showfliers=False)
        ax.axhline(0, linestyle="--", color="#B42318")
        ax.set(title=title, ylabel="Prediction − measured (W)")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, name)

    quantile = pd.qcut(full_oof[TARGET], 10, labels=False, duplicates="drop")
    bias = full_oof.assign(power_decile=quantile + 1).groupby("power_decile")["residual"].agg(["mean", "median"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(bias.index, bias["mean"], marker="o", label="Mean residual")
    ax.plot(bias.index, bias["median"], marker="s", label="Median residual")
    ax.axhline(0, linestyle="--", color="#B42318")
    ax.set(title="Bias across measured-power deciles", xlabel="Measured-power decile", ylabel="Prediction − measured (W)")
    ax.legend(frameon=False)
    _save(fig, "top_power_quantile_bias.png")

    pivot = ablation.pivot(index="algorithm", columns="comparison", values="row_weighted_delta_mae")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.24
    for index, column in enumerate(pivot.columns):
        ax.bar(x + (index - 1) * width, pivot[column], width=width, label=column)
    ax.axhline(0, color="#101828", linewidth=0.8)
    ax.set_xticks(x, labels=pivot.index, rotation=30, ha="right")
    ax.set(title="Paired hardware feature delta MAE", ylabel="After − before MAE (W)")
    ax.legend(frameon=False)
    _save(fig, "hardware_paired_mae_delta.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(lineage_summary))
    width = 0.38
    ax.bar(x - width / 2, lineage_summary["mae"], width, label="MAE")
    ax.bar(x + width / 2, lineage_summary["signed_bias"], width, label="Signed bias")
    ax.axhline(0, color="#101828", linewidth=0.8)
    ax.set_xticks(x, labels=lineage_summary["diag_parent_lineage"], rotation=30, ha="right")
    ax.set(title="Parent-lineage holdout error", ylabel="Watts")
    ax.legend(frameon=False)
    _save(fig, "lineage_holdout_mae_bias.png")


def main() -> None:
    frame = pd.read_parquet(STEP6 / "analysis/phase4_power_modeling_table.parquet")
    lock = load_json("step6/config/locked_candidate_phase4.json")
    metadata = load_json("step6/models/phase4_candidate_metadata.json")
    selected = {**lock["selected"], "selected_hyperparameter_index": metadata["selected_hyperparameter_index"]}
    bundles = load_feature_bundles()["bundles"]
    bundle = bundles[selected["feature_bundle"]]

    lineage_manifest = pd.read_csv(STEP6 / "splits/full_cohort_lineage_holdout_folds.csv")
    within_manifest = pd.read_csv(STEP6 / "splits/full_cohort_within_task_lineage_folds.csv")
    architecture_manifest = pd.read_csv(STEP6 / "splits/full_cohort_architecture_holdout_folds.csv")
    lineage_predictions, lineage_summary = parent_lineage_holdout(
        frame, bundle, selected, lineage_manifest
    )
    within_predictions, within_summary = within_task_lineage_holdout(
        frame, bundle, selected, within_manifest
    )
    architecture_predictions, architecture_summary = architecture_holdout(
        frame, bundle, selected, architecture_manifest
    )
    generalization = pd.concat(
        [lineage_predictions, within_predictions, architecture_predictions], ignore_index=True
    )
    generalization.to_parquet(STEP6 / "analysis/generalization_predictions.parquet", index=False)

    full_oof = pd.read_parquet(STEP6 / "analysis/full_cohort_oof_predictions.parquet")
    slices = error_slices(full_oof)
    static_summary = static_subset_experiment(frame, bundles, selected)
    report_dir = STEP6 / "reports"
    lineage_summary.to_csv(report_dir / "generalization_by_lineage.csv", index=False)
    within_summary.to_csv(report_dir / "within_task_lineage_holdout.csv", index=False)
    architecture_summary.to_csv(report_dir / "architecture_holdout.csv", index=False)
    slices.to_csv(report_dir / "error_slices.csv", index=False)
    static_summary.to_csv(report_dir / "static_subset_comparison.csv", index=False)

    loss_predictions = pd.read_parquet(STEP6 / "analysis/loss_strategy_predictions.parquet")
    development = loss_predictions.loc[
        loss_predictions["strategy_id"].eq(lock["selected"]["strategy_id"])
    ]
    confirmatory = pd.read_parquet(STEP6 / "analysis/confirmatory_predictions.parquet")
    ablation = pd.read_csv(report_dir / "hardware_ablation.csv")
    build_figures(frame, development, confirmatory, full_oof, lineage_summary, ablation)
    print(
        json.dumps(
            {
                "lineage_holdouts": len(lineage_summary),
                "within_task_lineage_holdouts": len(within_summary),
                "architecture_holdouts": len(architecture_summary),
                "error_slices": len(slices),
                "static_support": int(static_summary["n_rows"].iloc[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
