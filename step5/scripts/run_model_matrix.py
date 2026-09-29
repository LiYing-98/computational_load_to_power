#!/usr/bin/env python3
"""Execute and summarize the checkpointed Phase 3 core model matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from step4.scripts.analyze_results import METRIC_NAMES, regression_metrics
from step5.scripts.common import ROOT, SEED, STEP5, load_feature_bundles, load_json, sha256_file, verify_source_freeze
from step5.scripts.modeling import fit_predict_outer_fold, metadata_json, package_version


ALGORITHMS = (
    "DummyMean",
    "Ridge",
    "ElasticNet",
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting",
    "XGBoost",
    "CatBoost",
    "SplineGAM",
    "SmallMLP",
)
BUNDLES = (
    "B0-Core",
    "B1-Static",
    "B2b-Effective",
    "B3-Physics",
    "B4-Hardware",
    "B5-HardwarePhysics",
)
TARGETS = (
    "y_gpu_avg_power_watts",
    "y_gpu_energy_per_token_joules",
)
DIAGNOSTIC_COLUMNS = (
    "diag_model_id",
    "diag_model_family",
    "x_hardware_gpu_model",
    "x_hardware_num_gpus",
    "x_protocol_task",
    "x_model_benchmark_weight_precision",
    "x_model_benchmark_total_params_billions",
    "x_deployment_max_num_seqs",
)
ABLATION_PAIRS = (
    ("B0-Core", "B1-Static"),
    ("B1-Static", "B2b-Effective"),
    ("B2b-Effective", "B3-Physics"),
    ("B3-Physics", "B4-Hardware"),
    ("B4-Hardware", "B5-HardwarePhysics"),
)


def experiment_plan() -> list[dict[str, str]]:
    return [
        {
            "target": target,
            "bundle": bundle,
            "model_name": algorithm,
            "split_scheme": "model_group",
            "target_transform": "raw",
            "target_strategy": "direct",
        }
        for target in TARGETS
        for bundle in BUNDLES
        for algorithm in ALGORITHMS
    ]


def experiment_id(spec: dict[str, str]) -> str:
    return "__".join(
        spec[key]
        for key in (
            "target",
            "bundle",
            "model_name",
            "split_scheme",
            "target_transform",
            "target_strategy",
        )
    )


def _experiment_fingerprint(spec: dict[str, str]) -> str:
    """Fingerprint every input that can change fitted predictions."""
    paths = (
        STEP5 / "analysis/phase3_modeling_table.parquet",
        STEP5 / "config/feature_bundles_phase3.json",
        STEP5 / "config/model_search_spaces.json",
        ROOT / "step4/config/feature_bundles.json",
        ROOT / "step4/splits/model_group_folds.csv",
        STEP5 / "scripts/modeling.py",
        STEP5 / "scripts/run_model_matrix.py",
    )
    payload = {
        "spec": spec,
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
        "backends": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost")
        },
        "python": platform.python_version(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _checkpoint_path(spec: dict[str, str], fingerprint: str | None = None) -> Path:
    identity = experiment_id(spec)
    fingerprint = fingerprint or _experiment_fingerprint(spec)
    digest = hashlib.sha256(f"{identity}|{fingerprint}".encode("utf-8")).hexdigest()[:16]
    return STEP5 / "models/checkpoints" / f"{digest}_{spec['model_name']}_{spec['bundle']}.parquet"


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_complete_checkpoint(path: Path, identity: str, fingerprint: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if len(frame) != 305 or frame["run_key"].nunique() != 305:
        return None
    if set(frame["experiment_id"]) != {identity}:
        return None
    if "checkpoint_fingerprint" not in frame or set(frame["checkpoint_fingerprint"]) != {fingerprint}:
        return None
    return frame


def run_experiment(
    physics: pd.DataFrame,
    spec: dict[str, str],
    bundles: dict[str, Any],
    search_spaces: dict[str, Any],
) -> pd.DataFrame:
    identity = experiment_id(spec)
    fingerprint = _experiment_fingerprint(spec)
    checkpoint = _checkpoint_path(spec, fingerprint)
    existing = _load_complete_checkpoint(checkpoint, identity, fingerprint)
    if existing is not None:
        print(f"SKIP {identity}", flush=True)
        return existing

    bundle = bundles[spec["bundle"]]
    candidates = search_spaces["algorithms"][spec["model_name"]]["candidates"]
    outputs: list[pd.DataFrame] = []
    started = time.perf_counter()
    for fold in sorted(physics["outer_fold"].unique()):
        train = physics.loc[physics["outer_fold"].ne(fold)].copy()
        held_out = physics.loc[physics["outer_fold"].eq(fold)].copy()
        overlap = set(train["diag_model_id"].astype(str)) & set(held_out["diag_model_id"].astype(str))
        if overlap:
            raise RuntimeError(f"outer model fold {fold} leaks model IDs: {sorted(overlap)}")
        print(
            f"FIT {spec['target']} {spec['bundle']} {spec['model_name']} fold={fold} "
            f"train={len(train)} test={len(held_out)}",
            flush=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            prediction, fit_metadata, strategy_prediction = fit_predict_outer_fold(
                train,
                held_out,
                numeric=bundle["numeric"],
                categorical=bundle["categorical"],
                target=spec["target"],
                algorithm=spec["model_name"],
                candidates=candidates,
                inner_group_column="diag_model_id",
                target_transform=spec["target_transform"],
                target_strategy=spec["target_strategy"],
            )
        result = held_out[["run_key", *DIAGNOSTIC_COLUMNS]].copy()
        result["experiment_id"] = identity
        for key, value in spec.items():
            result[key] = value
        result["cohort"] = "physics"
        result["outer_fold"] = int(fold)
        result["y_true"] = held_out[spec["target"]].to_numpy(dtype=float)
        result["y_pred"] = prediction
        result["strategy_scale_prediction"] = strategy_prediction
        result["selected_candidate_index"] = int(fit_metadata["selected_candidate_index"])
        result["selected_candidate_json"] = metadata_json(fit_metadata["selected_candidate"])
        result["inner_validation_mae_json"] = json.dumps(fit_metadata["inner_validation_mae"])
        result["backend_module"] = fit_metadata["backend_module"]
        result["checkpoint_fingerprint"] = fingerprint
        outputs.append(result)
    complete = pd.concat(outputs, ignore_index=True)
    if len(complete) != 305 or complete["run_key"].nunique() != 305:
        raise RuntimeError(f"{identity} emitted incomplete OOF support")
    complete["elapsed_seconds"] = time.perf_counter() - started
    _write_parquet_atomic(complete, checkpoint)
    print(f"DONE {identity} elapsed={complete['elapsed_seconds'].iloc[0]:.1f}s", flush=True)
    return complete


def _stable_seed(text: str) -> int:
    return (SEED + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)) % (2**32 - 1)


def _cluster_bootstrap(group: pd.DataFrame, n_boot: int = 1000) -> dict[str, Any]:
    cluster_values = sorted(group["diag_model_id"].astype(str).unique())
    indices = {
        cluster: np.flatnonzero(group["diag_model_id"].astype(str).to_numpy() == cluster)
        for cluster in cluster_values
    }
    rng = np.random.default_rng(_stable_seed(str(group["experiment_id"].iloc[0])))
    draws = {metric: np.empty(n_boot, dtype=float) for metric in METRIC_NAMES}
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    for index in range(n_boot):
        sampled = rng.choice(cluster_values, size=len(cluster_values), replace=True)
        row_index = np.concatenate([indices[value] for value in sampled])
        metrics = regression_metrics(y_true[row_index], y_pred[row_index])
        for metric in METRIC_NAMES:
            draws[metric][index] = metrics[metric]
    result: dict[str, Any] = {"bootstrap_replicates": n_boot, "n_model_clusters": len(cluster_values)}
    for metric, values in draws.items():
        finite = values[np.isfinite(values)]
        result[f"{metric}_ci_low"] = float(np.percentile(finite, 2.5)) if len(finite) else math.nan
        result[f"{metric}_ci_high"] = float(np.percentile(finite, 97.5)) if len(finite) else math.nan
    return result


def build_model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity_columns = (
        "experiment_id",
        "target",
        "cohort",
        "bundle",
        "model_name",
        "split_scheme",
        "target_transform",
        "target_strategy",
    )
    for _, group in predictions.groupby("experiment_id", sort=True):
        identity = {column: group.iloc[0][column] for column in identity_columns}
        backend_values = sorted(group["backend_module"].unique())
        rows.append(
            {
                **identity,
                "backend_module": "|".join(backend_values),
                "n_rows": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
                **_cluster_bootstrap(group),
            }
        )
    return pd.DataFrame(rows)


def _paired_model_delta_interval(values: pd.Series, seed_text: str, n_boot: int = 1000) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(_stable_seed(seed_text))
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draws[index] = rng.choice(array, size=len(array), replace=True).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def build_feature_ablation(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, model_name), group in predictions.groupby(["target", "model_name"], sort=True):
        for reference, candidate in ABLATION_PAIRS:
            left = group.loc[group["bundle"].eq(reference)]
            right = group.loc[group["bundle"].eq(candidate)]
            joined = left.merge(
                right[["run_key", "outer_fold", "y_true", "y_pred"]],
                on=["run_key", "outer_fold"],
                validate="one_to_one",
                suffixes=("_reference", "_candidate"),
            )
            if len(joined) != 305 or not np.allclose(joined["y_true_reference"], joined["y_true_candidate"]):
                raise ValueError(f"paired support mismatch for {target} {model_name} {reference}->{candidate}")
            joined["absolute_error_reference"] = np.abs(joined["y_true_reference"] - joined["y_pred_reference"])
            joined["absolute_error_candidate"] = np.abs(joined["y_true_candidate"] - joined["y_pred_candidate"])
            model_delta = joined.groupby("diag_model_id").apply(
                lambda part: part["absolute_error_candidate"].mean() - part["absolute_error_reference"].mean(),
                include_groups=False,
            )
            seed_text = f"{target}|{model_name}|{reference}|{candidate}"
            ci_low, ci_high = _paired_model_delta_interval(model_delta, seed_text)
            reference_metrics = regression_metrics(joined["y_true_reference"], joined["y_pred_reference"])
            candidate_metrics = regression_metrics(joined["y_true_candidate"], joined["y_pred_candidate"])
            rows.append(
                {
                    "target": target,
                    "model_name": model_name,
                    "reference_bundle": reference,
                    "candidate_bundle": candidate,
                    "n_rows": len(joined),
                    "n_model_clusters": joined["diag_model_id"].nunique(),
                    "mae_reference": reference_metrics["mae"],
                    "mae_candidate": candidate_metrics["mae"],
                    "mae_delta_candidate_minus_reference": candidate_metrics["mae"] - reference_metrics["mae"],
                    "equal_model_mae_delta": float(model_delta.mean()),
                    "equal_model_mae_delta_ci_low": ci_low,
                    "equal_model_mae_delta_ci_high": ci_high,
                    "rmse_reference": reference_metrics["rmse"],
                    "rmse_candidate": candidate_metrics["rmse"],
                    "r2_reference": reference_metrics["r2"],
                    "r2_candidate": candidate_metrics["r2"],
                    "bootstrap_replicates": 1000,
                }
            )
    return pd.DataFrame(rows)


def build_prediction_validity(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in predictions.groupby("experiment_id", sort=True):
        values = group["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "experiment_id": group["experiment_id"].iloc[0],
                "target": group["target"].iloc[0],
                "bundle": group["bundle"].iloc[0],
                "model_name": group["model_name"].iloc[0],
                "n_predictions": len(group),
                "nonfinite_prediction_count": int((~np.isfinite(values)).sum()),
                "negative_prediction_count": int((values < 0).sum()),
                "negative_prediction_fraction": float((values < 0).mean()),
                "zero_prediction_count": int((values == 0).sum()),
                "minimum_prediction": float(np.nanmin(values)),
                "maximum_prediction": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def _environment() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "catboost", "shap")
    return {"python": platform.python_version(), **{name: package_version(name) for name in packages}}


def finalize(predictions: pd.DataFrame) -> None:
    predictions = predictions.sort_values(["experiment_id", "outer_fold", "run_key"]).reset_index(drop=True)
    if predictions["experiment_id"].nunique() != 120 or len(predictions) != 36600:
        raise ValueError("finalization requires all 120 complete experiments")
    analysis_dir = STEP5 / "analysis"
    report_dir = STEP5 / "reports"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(analysis_dir / "oof_predictions.parquet", index=False)
    predictions.to_csv(analysis_dir / "oof_predictions.csv", index=False)
    build_model_comparison(predictions).to_csv(report_dir / "model_comparison.csv", index=False)
    build_feature_ablation(predictions).to_csv(report_dir / "feature_ablation.csv", index=False)
    build_prediction_validity(predictions).to_csv(report_dir / "prediction_validity.csv", index=False)
    provenance = {
        "schema_version": "1.0",
        "scope": "Phase 3 Physics cohort, model-holdout OOF core matrix",
        "selection_protocol": "outer-train-only group-aware inner CV",
        "completed_experiments": int(predictions["experiment_id"].nunique()),
        "prediction_rows": len(predictions),
        "source_freeze_sha256": sha256_file(STEP5 / "config/source_freeze_phase3.json"),
        "feature_bundles_sha256": sha256_file(STEP5 / "config/feature_bundles_phase3.json"),
        "model_search_spaces_sha256": sha256_file(STEP5 / "config/model_search_spaces.json"),
        "model_folds_sha256": sha256_file(ROOT / "step4/splits/model_group_folds.csv"),
        "model_manifest_sha256": sha256_file(STEP5 / "models/model_manifest.json"),
        "modeling_table_sha256": sha256_file(STEP5 / "analysis/phase3_modeling_table.parquet"),
        "environment": _environment(),
    }
    (analysis_dir / "training_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def _filtered_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    plan = experiment_plan()
    if args.target:
        plan = [item for item in plan if item["target"] in args.target]
    if args.algorithm:
        plan = [item for item in plan if item["model_name"] in args.algorithm]
    if args.bundle:
        plan = [item for item in plan if item["bundle"] in args.bundle]
    if args.max_experiments is not None:
        plan = plan[: args.max_experiments]
    return plan


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append")
    parser.add_argument("--algorithm", action="append")
    parser.add_argument("--bundle", action="append")
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    failures = verify_source_freeze(ROOT)
    if failures:
        raise RuntimeError("; ".join(failures))
    frame = pd.read_parquet(STEP5 / "analysis/phase3_modeling_table.parquet")
    physics = frame.loc[frame["eligible_phase20_physics"]].copy()
    folds = pd.read_csv(ROOT / "step4/splits/model_group_folds.csv")
    folds = folds.loc[folds["cohort"].eq("physics"), ["run_key", "outer_fold"]]
    physics = physics.merge(folds, on="run_key", how="inner", validate="one_to_one")
    if len(physics) != 305:
        raise ValueError("core matrix requires exactly 305 Physics-cohort rows")
    bundles = load_feature_bundles(ROOT)["bundles"]
    search_spaces = load_json("step5/config/model_search_spaces.json")
    selected_plan = _filtered_plan(args)
    for index, spec in enumerate(selected_plan, start=1):
        print(f"EXPERIMENT {index}/{len(selected_plan)} {experiment_id(spec)}", flush=True)
        run_experiment(physics, spec, bundles, search_spaces)

    all_predictions: list[pd.DataFrame] = []
    missing: list[str] = []
    for spec in experiment_plan():
        identity = experiment_id(spec)
        fingerprint = _experiment_fingerprint(spec)
        checkpoint = _load_complete_checkpoint(_checkpoint_path(spec, fingerprint), identity, fingerprint)
        if checkpoint is None:
            missing.append(identity)
        else:
            all_predictions.append(checkpoint)
    print(f"CHECKPOINTS complete={len(all_predictions)} missing={len(missing)}", flush=True)
    if not args.no_finalize and not missing:
        finalize(pd.concat(all_predictions, ignore_index=True))
        print("FINALIZED core matrix", flush=True)
    elif not args.no_finalize and missing:
        print("NOT FINALIZED; rerun to complete missing experiments", flush=True)


if __name__ == "__main__":
    main()
