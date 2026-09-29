#!/usr/bin/env python3
"""Compact, non-blocking Phase 3 physics coverage diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from step5.scripts.common import ROOT, STEP5, load_feature_bundles


CONTEXT_FEATURES = (
    "x_model_benchmark_total_params_billions",
    "x_model_benchmark_activated_params_billions",
    "x_workload_effective_input_tokens_mean",
    "x_workload_effective_input_tokens_p95",
)


def distribution_summary(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        raise ValueError("distribution summary requires at least one finite value")
    numeric = numeric[np.isfinite(numeric)]
    quantiles = numeric.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0))
    p5 = float(quantiles.loc[0.05])
    p95 = float(quantiles.loc[0.95])
    positive = numeric[numeric > 0]
    return {
        "n": int(len(numeric)),
        "n_unique": int(numeric.nunique()),
        "min": float(numeric.min()),
        "p5": p5,
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": p95,
        "max": float(numeric.max()),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean != 0 else math.nan,
        "p95_p5_ratio": float(p95 / p5) if p5 > 0 else math.nan,
        "log10_range": (
            float(np.log10(positive.max()) - np.log10(positive.min()))
            if len(positive) >= 2
            else 0.0
        ),
    }


def feature_groups(project_root: Path = ROOT) -> tuple[list[str], list[str]]:
    bundles = load_feature_bundles(project_root)["bundles"]
    b3 = [column for column in bundles["B3-Physics"]["numeric"] if column not in bundles["B2b-Effective"]["numeric"]]
    b5 = [column for column in bundles["B5-HardwarePhysics"]["numeric"] if column not in bundles["B3-Physics"]["numeric"]]
    return b3, b5


def build_summary(frame: pd.DataFrame, b3: list[str], b5: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, columns in (("B3", b3), ("B5", b5)):
        for column in columns:
            rows.append({"feature": column, "feature_group": group, **distribution_summary(frame[column])})
    return pd.DataFrame(rows)


def build_pca(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = frame.loc[:, columns].astype(float)
    if matrix.isna().any().any():
        raise ValueError("PCA feature matrix must be complete")
    # Positive physics quantities span orders of magnitude. log1p preserves zeros
    # and is followed by z-scoring so PCA diagnoses geometry rather than units.
    transformed = np.log1p(matrix.clip(lower=0.0))
    standardized = StandardScaler().fit_transform(transformed)
    pca = PCA(svd_solver="full").fit(standardized)
    explained = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": np.cumsum(explained),
            "transform": "log1p_then_standardize",
            "n_rows": len(frame),
            "n_features": len(columns),
        }
    )


def _safe_association(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return math.nan, math.nan, len(pair)
    result = pearsonr(pair["x"], pair["y"]) if method == "pearson" else spearmanr(pair["x"], pair["y"])
    return float(result.statistic), float(result.pvalue), len(pair)


def build_residual_associations(
    frame: pd.DataFrame,
    phase2_predictions: pd.DataFrame,
    b3: list[str],
    b5: list[str],
) -> pd.DataFrame:
    base_filter = (
        phase2_predictions["target"].eq("y_gpu_avg_power_watts")
        & phase2_predictions["cohort"].eq("physics")
        & phase2_predictions["model_name"].eq("HistGradientBoosting")
        & phase2_predictions["split_scheme"].eq("model_group")
        & phase2_predictions["target_transform"].eq("raw")
    )
    rows: list[dict[str, Any]] = []
    for bundle, path_name, features in (
        ("B2b-Effective", "B2b residual -> B3", b3),
        ("B3-Physics", "B3 residual -> B5", b5),
    ):
        prediction = phase2_predictions.loc[base_filter & phase2_predictions["bundle"].eq(bundle)].copy()
        joined = prediction[["run_key", "y_true", "y_pred"]].merge(
            frame[["run_key", *features]], on="run_key", how="inner", validate="one_to_one"
        )
        if len(joined) != 305:
            raise ValueError(f"{path_name} requires 305 aligned predictions; found {len(joined)}")
        residual = joined["y_true"] - joined["y_pred"]
        for feature in features:
            pearson, pearson_p, n = _safe_association(joined[feature], residual, "pearson")
            spearman, spearman_p, n_rank = _safe_association(joined[feature], residual, "spearman")
            if n != n_rank:
                raise ValueError("Pearson and Spearman support differs")
            rows.append(
                {
                    "diagnostic_path": path_name,
                    "source_bundle": bundle,
                    "feature": feature,
                    "n": n,
                    "pearson_r": pearson,
                    "pearson_p_value": pearson_p,
                    "spearman_rho": spearman,
                    "spearman_p_value": spearman_p,
                }
            )
    return pd.DataFrame(rows)


def plot_coverage(frame: pd.DataFrame, output_path: Path) -> None:
    families = sorted(frame["diag_model_family"].astype(str).unique())
    palette_values = plt.get_cmap("tab10").colors
    palette = {family: palette_values[index % len(palette_values)] for index, family in enumerate(families)}
    gpus = sorted(frame["x_hardware_gpu_model"].unique())
    fig, axes = plt.subplots(1, len(gpus), figsize=(12.5, 5), sharex=True, sharey=True)
    if len(gpus) == 1:
        axes = [axes]
    for axis, gpu in zip(axes, gpus):
        subset = frame.loc[frame["x_hardware_gpu_model"].eq(gpu)]
        for family in families:
            part = subset.loc[subset["diag_model_family"].astype(str).eq(family)]
            if part.empty:
                continue
            axis.scatter(
                part["x_hardware_physics_compute_time_proxy_seconds"],
                part["x_hardware_physics_memory_time_proxy_seconds"],
                s=28,
                alpha=0.72,
                color=palette[family],
                edgecolor="white",
                linewidth=0.35,
                label=family,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{gpu} reference profile (n={len(subset)})")
        axis.set_xlabel("Compute-time proxy (s, log scale)")
        axis.grid(color="#D1D5DB", linewidth=0.6, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Memory-time proxy (s, log scale)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Physics coverage by GPU label and model family", y=1.12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = pd.read_parquet(STEP5 / "analysis/phase3_modeling_table.parquet")
    physics = frame.loc[frame["eligible_phase20_physics"]].copy()
    if len(physics) != 305:
        raise ValueError(f"expected 305 Physics-cohort rows, found {len(physics)}")
    b3, b5 = feature_groups(ROOT)
    analysis_dir = STEP5 / "analysis"
    summary = build_summary(physics, b3, b5)
    summary.to_csv(analysis_dir / "physics_coverage_summary.csv", index=False)

    correlation_columns = list(dict.fromkeys([*CONTEXT_FEATURES, *b3, *b5]))
    physics[correlation_columns].corr(method="pearson").to_csv(
        analysis_dir / "physics_correlations_pearson.csv"
    )
    physics[correlation_columns].corr(method="spearman").to_csv(
        analysis_dir / "physics_correlations_spearman.csv"
    )
    build_pca(physics, [*b3, *b5]).to_csv(analysis_dir / "physics_pca_summary.csv", index=False)

    predictions = pd.read_parquet(ROOT / "step4/analysis/oof_predictions.parquet")
    associations = build_residual_associations(physics, predictions, b3, b5)
    associations.to_csv(analysis_dir / "physics_residual_associations.csv", index=False)
    plot_coverage(
        physics,
        STEP5 / "reports/figures/physics_coverage_by_gpu_family.png",
    )
    print(
        f"physics rows={len(physics)}, B3 features={len(b3)}, "
        f"B5 increment={len(b5)}, PCA components={len(b3) + len(b5)}"
    )


if __name__ == "__main__":
    main()
