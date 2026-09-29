#!/usr/bin/env python3
"""Run Phase 3 generalization, target-strategy, interpretation, and figure analyses."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from step4.scripts.analyze_results import regression_metrics
from step5.scripts.common import ROOT, SEED, STEP5, load_feature_bundles, load_json, sha256_file, verify_source_freeze
from step5.scripts.modeling import (
    inverse_target,
    make_estimator,
    metadata_json,
    package_version,
    select_candidate,
    transform_target,
)


POWER = "y_gpu_avg_power_watts"
ENERGY = "y_gpu_energy_per_token_joules"
DIAGNOSTIC_COLUMNS = (
    "diag_model_id",
    "diag_model_family",
    "x_hardware_gpu_model",
    "x_hardware_num_gpus",
    "x_protocol_task",
    "diag_load_avg_batch_to_capacity_ratio",
)
VARIANTS = (
    {
        "model_variant": "B0-Core__HistGradientBoosting", "bundle": "B0-Core",
        "model_name": "HistGradientBoosting", "analysis_role": "historical_reference",
    },
    {
        "model_variant": "B4-Hardware__XGBoost", "bundle": "B4-Hardware",
        "model_name": "XGBoost", "analysis_role": "outcome_informed_exploratory_followup",
    },
)
FIGURE_DIR = STEP5 / "reports/figures"
CHECKPOINT_DIR = STEP5 / "models/analysis_checkpoints"
GPU_COLORS = {"H100": "#2A9D8F", "B200": "#E76F51"}


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _checkpoint(name: str) -> Path:
    paths = (
        STEP5 / "analysis/phase3_modeling_table.parquet",
        STEP5 / "analysis/oof_predictions.parquet",
        STEP5 / "config/feature_bundles_phase3.json",
        STEP5 / "config/model_search_spaces.json",
        ROOT / "step4/splits/model_group_folds.csv",
        STEP5 / "scripts/modeling.py",
        STEP5 / "scripts/analyze_phase3.py",
    )
    payload = {
        "name": name,
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
        "backends": {
            package: package_version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost")
        },
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    digest = hashlib.sha256(f"{name}|{fingerprint}".encode("utf-8")).hexdigest()[:16]
    return CHECKPOINT_DIR / f"{digest}_{name.replace(':', '_')}.parquet"


def _strategy_target(frame: pd.DataFrame, target: str, strategy: str) -> np.ndarray:
    values = frame[target].to_numpy(dtype=float)
    if strategy == "direct":
        return values
    if strategy == "power_fraction":
        denominator = frame["x_hardware_aggregate_rated_power_w"].to_numpy(dtype=float)
        if (denominator <= 0).any() or not np.isfinite(denominator).all():
            raise ValueError("power_fraction requires finite positive rated-power denominators")
        return values / denominator
    raise ValueError(strategy)


def _restore_prediction(values: np.ndarray, frame: pd.DataFrame, transform: str, strategy: str) -> np.ndarray:
    prediction = inverse_target(values, transform)
    if strategy == "power_fraction":
        prediction = prediction * frame["x_hardware_aggregate_rated_power_w"].to_numpy(dtype=float)
    return np.asarray(prediction, dtype=float)


def _fit_estimator(
    train: pd.DataFrame,
    held_out: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    target: str,
    model_name: str,
    candidates: list[dict[str, Any]],
    target_transform: str = "raw",
    target_strategy: str = "direct",
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        selected, scores = select_candidate(
            train,
            numeric=bundle["numeric"],
            categorical=bundle["categorical"],
            target=target,
            algorithm=model_name,
            candidates=candidates,
            inner_group_column="diag_model_id",
            target_transform=target_transform,
            target_strategy=target_strategy,
        )
        candidate = dict(candidates[selected])
        estimator = make_estimator(
            model_name,
            candidate,
            train,
            bundle["numeric"],
            bundle["categorical"],
        )
        columns = [*bundle["numeric"], *bundle["categorical"]]
        fit_target = transform_target(_strategy_target(train, target, target_strategy), target_transform)
        estimator.fit(train[columns], fit_target)
        latent = estimator.predict(held_out[columns])
        prediction = _restore_prediction(latent, held_out, target_transform, target_strategy)
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    return estimator, prediction, {
        "selected_candidate_index": int(selected),
        "selected_candidate": candidate,
        "inner_validation_mae": scores,
        "backend_module": model.__class__.__module__,
    }


def _prediction_frame(
    source: pd.DataFrame,
    prediction: np.ndarray,
    *,
    model_variant: str,
    bundle: str,
    model_name: str,
    split_scheme: str,
    repeat_index: float,
    outer_fold: int,
    split_role: str,
    metadata: dict[str, Any],
    target: str = POWER,
    target_transform: str = "raw",
    target_strategy: str = "direct",
    strategy_scale_prediction: np.ndarray | None = None,
) -> pd.DataFrame:
    result = source[["run_key", *DIAGNOSTIC_COLUMNS]].copy()
    result["model_variant"] = model_variant
    role = next(
        (variant["analysis_role"] for variant in VARIANTS if variant["model_variant"] == model_variant),
        "outcome_informed_exploratory_followup",
    )
    result["analysis_role"] = role
    result["bundle"] = bundle
    result["model_name"] = model_name
    result["target"] = target
    result["split_scheme"] = split_scheme
    result["repeat_index"] = repeat_index
    result["outer_fold"] = int(outer_fold)
    result["split_role"] = split_role
    result["target_transform"] = target_transform
    result["target_strategy"] = target_strategy
    result["y_true"] = source[target].to_numpy(dtype=float)
    result["y_pred"] = np.asarray(prediction, dtype=float)
    result["selected_candidate_index"] = metadata["selected_candidate_index"]
    result["selected_candidate_json"] = metadata_json(metadata["selected_candidate"])
    result["backend_module"] = metadata["backend_module"]
    if strategy_scale_prediction is not None:
        result["strategy_scale_prediction"] = np.asarray(strategy_scale_prediction, dtype=float)
    return result


def _fit_random_cell(
    physics: pd.DataFrame,
    manifest: pd.DataFrame,
    variant: dict[str, str],
    repeat_index: int,
    bundles: dict[str, Any],
    search: dict[str, Any],
) -> pd.DataFrame:
    name = f"generalization:random:{repeat_index}:{variant['model_variant']}"
    path = _checkpoint(name)
    if path.exists():
        return pd.read_parquet(path)
    roles = manifest.loc[manifest["repeat_index"].eq(repeat_index), ["run_key", "split_role"]]
    data = physics.merge(roles, on="run_key", validate="one_to_one")
    train = data.loc[data["split_role"].eq("train")].copy()
    test = data.loc[data["split_role"].eq("test")].copy()
    estimator, prediction, metadata = _fit_estimator(
        train,
        test,
        bundle=bundles[variant["bundle"]],
        target=POWER,
        model_name=variant["model_name"],
        candidates=search["algorithms"][variant["model_name"]]["candidates"],
    )
    outputs = [
        _prediction_frame(
            test,
            prediction,
            model_variant=variant["model_variant"],
            bundle=variant["bundle"],
            model_name=variant["model_name"],
            split_scheme="random",
            repeat_index=repeat_index,
            outer_fold=0,
            split_role="test",
            metadata=metadata,
        )
    ]
    if repeat_index == 0:
        columns = [*bundles[variant["bundle"]]["numeric"], *bundles[variant["bundle"]]["categorical"]]
        train_prediction = estimator.predict(train[columns])
        train_prediction = _restore_prediction(train_prediction, train, "raw", "direct")
        outputs.append(
            _prediction_frame(
                train,
                train_prediction,
                model_variant=variant["model_variant"],
                bundle=variant["bundle"],
                model_name=variant["model_name"],
                split_scheme="random",
                repeat_index=repeat_index,
                outer_fold=0,
                split_role="train",
                metadata=metadata,
            )
        )
    result = pd.concat(outputs, ignore_index=True)
    _write_parquet_atomic(result, path)
    return result


def merge_fold_manifest(
    physics: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Attach folds while retaining the exact column used to audit isolation."""
    required = ["run_key", "outer_fold", group_column]
    missing = [column for column in required if column not in fold_manifest.columns]
    if missing:
        raise ValueError(f"fold manifest missing required columns: {missing}")
    manifest = fold_manifest[required].copy()
    if group_column in physics.columns:
        manifest = manifest.drop(columns=group_column)
    return physics.merge(manifest, on="run_key", validate="one_to_one")


def _fit_grouped_cell(
    physics: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    variant: dict[str, str],
    split_scheme: str,
    group_column: str,
    bundles: dict[str, Any],
    search: dict[str, Any],
) -> pd.DataFrame:
    name = f"generalization:{split_scheme}:{variant['model_variant']}"
    path = _checkpoint(name)
    if path.exists():
        return pd.read_parquet(path)
    data = merge_fold_manifest(physics, fold_manifest, group_column)
    outputs: list[pd.DataFrame] = []
    for fold in sorted(data["outer_fold"].unique()):
        train = data.loc[data["outer_fold"].ne(fold)].copy()
        test = data.loc[data["outer_fold"].eq(fold)].copy()
        if set(train[group_column].astype(str)) & set(test[group_column].astype(str)):
            raise RuntimeError(f"{split_scheme} fold {fold} leaks {group_column}")
        _, prediction, metadata = _fit_estimator(
            train,
            test,
            bundle=bundles[variant["bundle"]],
            target=POWER,
            model_name=variant["model_name"],
            candidates=search["algorithms"][variant["model_name"]]["candidates"],
        )
        outputs.append(
            _prediction_frame(
                test,
                prediction,
                model_variant=variant["model_variant"],
                bundle=variant["bundle"],
                model_name=variant["model_name"],
                split_scheme=split_scheme,
                repeat_index=math.nan,
                outer_fold=int(fold),
                split_role="test",
                metadata=metadata,
            )
        )
        print(f"GENERALIZATION {variant['model_variant']} {split_scheme} fold={fold}", flush=True)
    result = pd.concat(outputs, ignore_index=True)
    if len(result) != 305 or result["run_key"].nunique() != 305:
        raise RuntimeError(f"incomplete {split_scheme} OOF support for {variant['model_variant']}")
    _write_parquet_atomic(result, path)
    return result


def build_generalization(
    physics: pd.DataFrame,
    bundles: dict[str, Any],
    search: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_manifest = pd.read_csv(STEP5 / "splits/random_split_manifest.csv")
    random_manifest = random_manifest.loc[random_manifest["cohort"].eq("physics")]
    manifests = {
        "config_group": (pd.read_csv(ROOT / "step4/splits/config_group_folds.csv"), "config_group_id"),
        "model_group": (pd.read_csv(ROOT / "step4/splits/model_group_folds.csv"), "diag_model_id"),
        "family_group": (pd.read_csv(STEP5 / "splits/family_group_folds.csv"), "diag_model_family"),
    }
    predictions: list[pd.DataFrame] = []
    for variant in VARIANTS:
        for repeat_index in range(10):
            print(f"GENERALIZATION {variant['model_variant']} random repeat={repeat_index}", flush=True)
            predictions.append(
                _fit_random_cell(physics, random_manifest, variant, repeat_index, bundles, search)
            )
        for split_scheme, (manifest, group_column) in manifests.items():
            if "cohort" in manifest.columns:
                manifest = manifest.loc[manifest["cohort"].eq("physics")]
            predictions.append(
                _fit_grouped_cell(
                    physics,
                    manifest,
                    variant,
                    split_scheme,
                    group_column,
                    bundles,
                    search,
                )
            )
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions = all_predictions.sort_values(
        ["model_variant", "split_scheme", "repeat_index", "split_role", "outer_fold", "run_key"],
        na_position="last",
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    test = all_predictions.loc[all_predictions["split_role"].eq("test")]
    for (variant, scheme, repeat), group in test.groupby(
        ["model_variant", "split_scheme", "repeat_index"], dropna=False, sort=True
    ):
        rows.append(
            {
                "model_variant": variant,
                "analysis_role": group["analysis_role"].iloc[0],
                "bundle": group["bundle"].iloc[0],
                "model_name": group["model_name"].iloc[0],
                "target": POWER,
                "split_scheme": scheme,
                "repeat_index": repeat,
                "n_test": len(group),
                "n_models_test": group["diag_model_id"].nunique(),
                "n_families_test": group["diag_model_family"].nunique(),
                **regression_metrics(group["y_true"], group["y_pred"]),
                "negative_prediction_count": int((group["y_pred"] < 0).sum()),
                "backend_module": "|".join(sorted(group["backend_module"].unique())),
            }
        )
    return all_predictions, pd.DataFrame(rows)


def _model_folds(physics: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "step4/splits/model_group_folds.csv")
    manifest = manifest.loc[manifest["cohort"].eq("physics"), ["run_key", "outer_fold"]]
    return physics.merge(manifest, on="run_key", validate="one_to_one")


def build_target_strategies(
    physics: pd.DataFrame,
    bundles: dict[str, Any],
    search: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _model_folds(physics)
    specs = (
        (POWER, "raw", "direct"),
        (POWER, "raw", "power_fraction"),
        (ENERGY, "raw", "direct"),
        (ENERGY, "log1p", "direct"),
    )
    outputs: list[pd.DataFrame] = []
    for target, transform, strategy in specs:
        bundle_name = "B4-Hardware" if target == POWER else "B1-Static"
        name = f"strategy:{bundle_name}:{target}:{transform}:{strategy}"
        path = _checkpoint(name)
        if path.exists():
            outputs.append(pd.read_parquet(path))
            continue
        folds: list[pd.DataFrame] = []
        for fold in sorted(data["outer_fold"].unique()):
            train = data.loc[data["outer_fold"].ne(fold)].copy()
            test = data.loc[data["outer_fold"].eq(fold)].copy()
            estimator, prediction, metadata = _fit_estimator(
                train,
                test,
                bundle=bundles[bundle_name],
                target=target,
                model_name="XGBoost",
                candidates=search["algorithms"]["XGBoost"]["candidates"],
                target_transform=transform,
                target_strategy=strategy,
            )
            columns = [*bundles[bundle_name]["numeric"], *bundles[bundle_name]["categorical"]]
            latent_prediction = estimator.predict(test[columns])
            strategy_scale_prediction = inverse_target(latent_prediction, transform)
            folds.append(
                _prediction_frame(
                    test,
                    prediction,
                    model_variant=bundle_name + "__XGBoost",
                    bundle=bundle_name,
                    model_name="XGBoost",
                    split_scheme="model_group",
                    repeat_index=math.nan,
                    outer_fold=int(fold),
                    split_role="test",
                    metadata=metadata,
                    target=target,
                    target_transform=transform,
                    target_strategy=strategy,
                    strategy_scale_prediction=strategy_scale_prediction,
                )
            )
            print(f"STRATEGY {target} {transform} {strategy} fold={fold}", flush=True)
        result = pd.concat(folds, ignore_index=True)
        _write_parquet_atomic(result, path)
        outputs.append(result)
    predictions = pd.concat(outputs, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (target, transform, strategy), group in predictions.groupby(
        ["target", "target_transform", "target_strategy"], sort=True
    ):
        values = group["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "bundle": group["bundle"].iloc[0],
                "model_name": "XGBoost",
                "split_scheme": "model_group",
                "target_transform": transform,
                "target_strategy": strategy,
                "n_rows": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
                "nonfinite_prediction_count": int((~np.isfinite(values)).sum()),
                "negative_prediction_count": int((values < 0).sum()),
                "minimum_prediction": float(np.nanmin(values)),
                "maximum_prediction": float(np.nanmax(values)),
                "backend_module": "|".join(sorted(group["backend_module"].unique())),
            }
        )
    return predictions, pd.DataFrame(rows)


def build_hardware_sensitivity(
    physics: pd.DataFrame,
    core: pd.DataFrame,
    bundles: dict[str, Any],
    search: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare B3/B4/B5 and B5 without the categorical GPU identity label."""
    bundle_names = (
        "B3-Physics",
        "B4-Hardware",
        "B5-HardwarePhysics",
        "B5-HardwarePhysics-NoLabel",
    )
    outputs: list[pd.DataFrame] = []
    for bundle_name in bundle_names:
        if bundle_name in {"B3-Physics", "B4-Hardware", "B5-HardwarePhysics"}:
            existing = core.loc[
                core["target"].eq(POWER)
                & core["bundle"].eq(bundle_name)
                & core["model_name"].eq("XGBoost")
            ].copy()
            existing["model_variant"] = bundle_name + "__XGBoost"
            existing["split_role"] = "test"
            outputs.append(existing)
            continue
        name = f"hardware:{bundle_name}:XGBoost"
        path = _checkpoint(name)
        if path.exists():
            outputs.append(pd.read_parquet(path))
            continue
        data = _model_folds(physics)
        folds: list[pd.DataFrame] = []
        for fold in sorted(data["outer_fold"].unique()):
            train = data.loc[data["outer_fold"].ne(fold)].copy()
            test = data.loc[data["outer_fold"].eq(fold)].copy()
            _, prediction, metadata = _fit_estimator(
                train,
                test,
                bundle=bundles[bundle_name],
                target=POWER,
                model_name="XGBoost",
                candidates=search["algorithms"]["XGBoost"]["candidates"],
            )
            folds.append(
                _prediction_frame(
                    test,
                    prediction,
                    model_variant=bundle_name + "__XGBoost",
                    bundle=bundle_name,
                    model_name="XGBoost",
                    split_scheme="model_group",
                    repeat_index=math.nan,
                    outer_fold=int(fold),
                    split_role="test",
                    metadata=metadata,
                )
            )
            print(f"HARDWARE {bundle_name} fold={fold}", flush=True)
        result = pd.concat(folds, ignore_index=True)
        _write_parquet_atomic(result, path)
        outputs.append(result)
    predictions = pd.concat(outputs, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for bundle_name, group in predictions.groupby("bundle", sort=False):
        group = group.copy()
        group["experiment_id"] = "hardware_sensitivity__" + bundle_name
        values = group["y_pred"].to_numpy(dtype=float)
        from step5.scripts.run_model_matrix import _cluster_bootstrap

        rows.append(
            {
                "target": POWER,
                "bundle": bundle_name,
                "model_name": "XGBoost",
                "analysis_role": "outcome_informed_exploratory_followup",
                "split_scheme": "model_group",
                "retains_gpu_label": "x_hardware_gpu_model" in bundles[bundle_name]["categorical"],
                "n_rows": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
                **_cluster_bootstrap(group),
                "negative_prediction_count": int((values < 0).sum()),
                "minimum_prediction": float(np.nanmin(values)),
                "backend_module": "|".join(sorted(group["backend_module"].unique())),
            }
        )
    return predictions, pd.DataFrame(rows)


def build_hardware_paired_comparisons(
    core: pd.DataFrame,
    hardware_predictions: pd.DataFrame,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Emit auditable, model-cluster paired deltas for named exploratory contrasts."""
    variants = {
        "B0-Core__HistGradientBoosting": core.loc[
            core["target"].eq(POWER)
            & core["bundle"].eq("B0-Core")
            & core["model_name"].eq("HistGradientBoosting")
        ],
        "B3-Physics__XGBoost": hardware_predictions.loc[hardware_predictions["bundle"].eq("B3-Physics")],
        "B4-Hardware__XGBoost": hardware_predictions.loc[hardware_predictions["bundle"].eq("B4-Hardware")],
        "B5-HardwarePhysics__XGBoost": hardware_predictions.loc[
            hardware_predictions["bundle"].eq("B5-HardwarePhysics")
        ],
        "B5-HardwarePhysics-NoLabel__XGBoost": hardware_predictions.loc[
            hardware_predictions["bundle"].eq("B5-HardwarePhysics-NoLabel")
        ],
    }
    pairs = (
        ("B0-Core__HistGradientBoosting", "B4-Hardware__XGBoost"),
        ("B3-Physics__XGBoost", "B4-Hardware__XGBoost"),
        ("B4-Hardware__XGBoost", "B5-HardwarePhysics__XGBoost"),
        ("B5-HardwarePhysics__XGBoost", "B5-HardwarePhysics-NoLabel__XGBoost"),
    )
    rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    for reference_name, candidate_name in pairs:
        reference = variants[reference_name]
        candidate = variants[candidate_name]
        joined = reference[["run_key", "outer_fold", "diag_model_id", "y_true", "y_pred"]].merge(
            candidate[["run_key", "outer_fold", "y_true", "y_pred"]],
            on=["run_key", "outer_fold"],
            validate="one_to_one",
            suffixes=("_reference", "_candidate"),
        )
        if len(joined) != 305 or not np.allclose(joined["y_true_reference"], joined["y_true_candidate"]):
            raise RuntimeError(f"paired support mismatch: {reference_name} -> {candidate_name}")
        joined["absolute_error_reference"] = np.abs(joined["y_true_reference"] - joined["y_pred_reference"])
        joined["absolute_error_candidate"] = np.abs(joined["y_true_candidate"] - joined["y_pred_candidate"])
        model_deltas = joined.groupby("diag_model_id").apply(
            lambda part: part["absolute_error_candidate"].mean() - part["absolute_error_reference"].mean(),
            include_groups=False,
        )
        seed_text = f"hardware-paired|{reference_name}|{candidate_name}"
        rng = np.random.default_rng(SEED + int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16))
        values = model_deltas.to_numpy(dtype=float)
        draws = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
        for draw_index, draw in enumerate(draws):
            draw_rows.append(
                {
                    "reference_variant": reference_name,
                    "candidate_variant": candidate_name,
                    "draw_index": draw_index,
                    "equal_model_mae_delta_candidate_minus_reference": float(draw),
                }
            )
        reference_mae = float(joined["absolute_error_reference"].mean())
        candidate_mae = float(joined["absolute_error_candidate"].mean())
        rows.append(
            {
                "reference_variant": reference_name,
                "candidate_variant": candidate_name,
                "analysis_role": "outcome_informed_exploratory_comparison",
                "n_rows": len(joined),
                "n_model_clusters": int(joined["diag_model_id"].nunique()),
                "mae_reference": reference_mae,
                "mae_candidate": candidate_mae,
                "row_weighted_mae_delta_candidate_minus_reference": candidate_mae - reference_mae,
                "equal_model_mae_delta_candidate_minus_reference": float(values.mean()),
                "equal_model_mae_delta_ci_low": float(np.percentile(draws, 2.5)),
                "equal_model_mae_delta_ci_high": float(np.percentile(draws, 97.5)),
                "bootstrap_replicates": n_boot,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


def build_interpretation(
    physics: pd.DataFrame,
    bundles: dict[str, Any],
    search: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _model_folds(physics)
    variants = (
        {"model_variant": "B4-Hardware__XGBoost", "bundle": "B4-Hardware", "model_name": "XGBoost"},
        {"model_variant": "B3-Physics__XGBoost", "bundle": "B3-Physics", "model_name": "XGBoost"},
    )
    fitted: list[tuple[int, Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    importance_rows: list[dict[str, Any]] = []
    for variant in variants:
        bundle = bundles[variant["bundle"]]
        columns = [*bundle["numeric"], *bundle["categorical"]]
        for fold in sorted(data["outer_fold"].unique()):
            train = data.loc[data["outer_fold"].ne(fold)].copy()
            test = data.loc[data["outer_fold"].eq(fold)].copy()
            estimator, prediction, metadata = _fit_estimator(
                train,
                test,
                bundle=bundle,
                target=POWER,
                model_name=variant["model_name"],
                candidates=search["algorithms"][variant["model_name"]]["candidates"],
            )
            baseline = float(np.mean(np.abs(test[POWER].to_numpy(dtype=float) - prediction)))
            for feature_index, feature in enumerate(columns):
                increases: list[float] = []
                for repeat in range(10):
                    seed = SEED + fold * 10007 + feature_index * 101 + repeat
                    rng = np.random.default_rng(seed)
                    permuted = test[columns].copy()
                    permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                    candidate = estimator.predict(permuted)
                    candidate = _restore_prediction(candidate, test, "raw", "direct")
                    mae = float(np.mean(np.abs(test[POWER].to_numpy(dtype=float) - candidate)))
                    increases.append(mae - baseline)
                importance_rows.append(
                    {
                        "model_variant": variant["model_variant"],
                        "bundle": variant["bundle"],
                        "model_name": variant["model_name"],
                        "feature": feature,
                        "feature_type": "numeric" if feature in bundle["numeric"] else "categorical",
                        "outer_fold": int(fold),
                        "method": "outer_test_permutation",
                        "fit_scope": "outer_train_only",
                        "n_train": len(train),
                        "n_test": len(test),
                        "baseline_mae_watts": baseline,
                        "importance_mae_increase_mean": float(np.mean(increases)),
                        "importance_mae_increase_std": float(np.std(increases, ddof=1)),
                        "permutation_repeats": 10,
                        "selected_candidate_json": metadata_json(metadata["selected_candidate"]),
                        "backend_module": metadata["backend_module"],
                    }
                )
            if variant["model_variant"] == "B4-Hardware__XGBoost":
                fitted.append((int(fold), estimator, train, test, metadata))
            print(f"INTERPRETATION {variant['model_variant']} fold={fold}", flush=True)
    importance = pd.DataFrame(importance_rows)
    ranking = (
        importance.loc[
            importance["model_variant"].eq("B4-Hardware__XGBoost")
            & importance["feature_type"].eq("numeric")
        ]
        .groupby("feature")["importance_mae_increase_mean"]
        .mean()
        .sort_values(ascending=False)
    )
    pdp_features = list(ranking.head(5).index)
    if len(pdp_features) < 4:
        raise RuntimeError("fewer than four numeric variables available for PDP")
    pdp_rows: list[dict[str, Any]] = []
    best_bundle = bundles["B4-Hardware"]
    columns = [*best_bundle["numeric"], *best_bundle["categorical"]]
    for fold, estimator, train, test, metadata in fitted:
        for feature in pdp_features:
            values = train[feature].dropna().to_numpy(dtype=float)
            grid = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 9)))
            for grid_value in grid:
                background = test[columns].copy()
                background[feature] = float(grid_value)
                prediction = estimator.predict(background)
                prediction = _restore_prediction(prediction, test, "raw", "direct")
                pdp_rows.append(
                    {
                        "feature": feature,
                        "outer_fold": fold,
                        "grid_value": float(grid_value),
                        "mean_prediction_watts": float(np.mean(prediction)),
                        "prediction_std_watts": float(np.std(prediction, ddof=1)),
                        "n_background": len(test),
                        "fit_scope": "outer_train_only",
                        "grid_scope": "outer_train_quantiles_5_to_95",
                        "backend_module": metadata["backend_module"],
                    }
                )
    return importance, pd.DataFrame(pdp_rows)


def build_error_slices(
    core: pd.DataFrame,
    hardware_predictions: pd.DataFrame,
    physics: pd.DataFrame,
) -> pd.DataFrame:
    power = hardware_predictions.loc[
        hardware_predictions["bundle"].eq("B4-Hardware")
        & hardware_predictions["model_name"].eq("XGBoost")
    ].copy()
    energy = core.loc[
        core["bundle"].eq("B1-Static")
        & core["model_name"].eq("XGBoost")
        & core["target"].eq(ENERGY)
    ].copy()
    chosen = pd.concat([power, energy], ignore_index=True)
    load = physics[["run_key", "diag_load_avg_batch_to_capacity_ratio"]].copy()
    chosen = chosen.drop(columns=["diag_load_avg_batch_to_capacity_ratio"], errors="ignore").merge(
        load, on="run_key", validate="many_to_one"
    )
    chosen["load_regime"] = pd.cut(
        chosen["diag_load_avg_batch_to_capacity_ratio"],
        bins=[-np.inf, 0.95, 0.99, np.inf],
        labels=["below_0.95", "0.95_to_0.99", "at_or_above_0.99"],
        right=False,
    ).astype(str)
    dimensions = {
        "gpu_model": "x_hardware_gpu_model",
        "model_family": "diag_model_family",
        "protocol_task": "x_protocol_task",
        "num_gpus": "x_hardware_num_gpus",
        "load_regime": "load_regime",
    }
    rows: list[dict[str, Any]] = []
    for target, target_rows in chosen.groupby("target"):
        for slice_type, column in dimensions.items():
            for value, group in target_rows.groupby(column, dropna=False):
                rows.append(
                    {
                        "target": target,
                        "bundle": group["bundle"].iloc[0],
                        "model_name": "XGBoost",
                        "split_scheme": "model_group",
                        "slice_type": slice_type,
                        "slice_value": str(value),
                        "n_rows": len(group),
                        "n_models": group["diag_model_id"].nunique(),
                        **regression_metrics(group["y_true"], group["y_pred"]),
                    }
                )
    return pd.DataFrame(rows)


def _metric_caption(frame: pd.DataFrame) -> str:
    metrics = regression_metrics(frame["y_true"], frame["y_pred"])
    return (
        f"n={len(frame)} | MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f} | "
        f"R²={metrics['r2']:.3f} | MdAPE={metrics['mdape_percent']:.1f}%"
    )


def _scatter_panel(ax: Any, frame: pd.DataFrame, title: str, units: str) -> None:
    for gpu, group in frame.groupby("x_hardware_gpu_model"):
        ax.scatter(
            group["y_true"],
            group["y_pred"],
            s=28,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=str(gpu),
            color=GPU_COLORS.get(str(gpu), "#4C78A8"),
        )
    values = np.concatenate([frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()])
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    padding = max((high - low) * 0.04, 1e-9)
    low, high = low - padding, high + padding
    ax.plot([low, high], [low, high], linestyle="--", color="#333333", linewidth=1.2, label="y = x")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Measured ({units})")
    ax.set_ylabel(f"Predicted ({units})")
    ax.set_title(f"{title}\n{_metric_caption(frame)}", fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)


def _save_figure(fig: Any, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fold_local_pdp_curves(rows: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return sorted, isolated fold curves; never join incompatible fold grids."""
    return {
        int(fold): fold_rows.sort_values("grid_value").reset_index(drop=True)
        for fold, fold_rows in rows.groupby("outer_fold", sort=True)
    }


def build_figures(
    generalization: pd.DataFrame,
    core: pd.DataFrame,
    importance: pd.DataFrame,
    pdp: pd.DataFrame,
) -> None:
    candidate = "B4-Hardware__XGBoost"
    canonical = generalization.loc[
        generalization["model_variant"].eq(candidate)
        & generalization["split_scheme"].eq("random")
        & generalization["repeat_index"].eq(0)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    for ax, role, label in zip(axes, ("train", "test"), ("Random train (in-sample)", "Random test (held-out)")):
        _scatter_panel(ax, canonical.loc[canonical["split_role"].eq(role)], label, "W")
    fig.suptitle("Phase 3 exploratory power candidate — canonical random split", fontsize=13)
    _save_figure(fig, "actual_vs_predicted_random_best.png")

    for scheme, name, label in (
        ("model_group", "actual_vs_predicted_model_holdout_best.png", "Model-holdout OOF"),
        ("family_group", "actual_vs_predicted_family_holdout_best.png", "Family-holdout OOF"),
    ):
        frame = generalization.loc[
            generalization["model_variant"].eq(candidate)
            & generalization["split_scheme"].eq(scheme)
            & generalization["split_role"].eq("test")
        ]
        fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        _scatter_panel(ax, frame, f"Exploratory power candidate — {label}", "W")
        _save_figure(fig, name)

    model = generalization.loc[
        generalization["model_variant"].eq(candidate)
        & generalization["split_scheme"].eq("model_group")
        & generalization["split_role"].eq("test")
    ].copy()
    model["residual"] = model["y_pred"] - model["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, x_column, x_label in zip(axes, ("y_true", "y_pred"), ("Measured power (W)", "Predicted power (W)")):
        for gpu, group in model.groupby("x_hardware_gpu_model"):
            ax.scatter(group[x_column], group["residual"], s=28, alpha=0.72, label=str(gpu), color=GPU_COLORS.get(str(gpu)))
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Residual: predicted − measured (W)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"Model-holdout residuals | {_metric_caption(model)}", fontsize=12)
    _save_figure(fig, "residuals_model_holdout_best.png")

    energy = core.loc[
        core["target"].eq(ENERGY)
        & core["bundle"].eq("B1-Static")
        & core["model_name"].eq("XGBoost")
    ].copy()
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    _scatter_panel(ax, energy, "Exploratory energy/token candidate — model-holdout OOF", "J/token")
    _save_figure(fig, "actual_vs_predicted_energy_best.png")

    energy["residual"] = energy["y_pred"] - energy["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, x_column, x_label in zip(
        axes,
        ("y_true", "y_pred"),
        ("Measured energy (J/token)", "Predicted energy (J/token)"),
    ):
        for gpu, group in energy.groupby("x_hardware_gpu_model"):
            ax.scatter(
                group[x_column], group["residual"], s=28, alpha=0.72,
                label=str(gpu), color=GPU_COLORS.get(str(gpu)),
            )
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Residual: predicted − measured (J/token)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"Energy/token model-holdout residuals | {_metric_caption(energy)}", fontsize=12)
    _save_figure(fig, "residuals_energy_best.png")

    ranking = (
        importance.loc[importance["model_variant"].eq("B4-Hardware__XGBoost")]
        .groupby("feature")["importance_mae_increase_mean"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=True)
        .tail(15)
    )
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.barh(ranking.index, ranking["mean"], xerr=ranking["std"].fillna(0), color="#457B9D", alpha=0.85)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Held-out MAE increase after permutation (W)")
    ax.set_title("Fold-local permutation importance — B4 + XGBoost")
    ax.grid(axis="x", alpha=0.2)
    _save_figure(fig, "feature_importance_best.png")

    features = list(dict.fromkeys(pdp["feature"]))
    fig, axes = plt.subplots(math.ceil(len(features) / 2), 2, figsize=(11.0, 3.6 * math.ceil(len(features) / 2)), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, feature in zip(axes, features):
        rows = pdp.loc[pdp["feature"].eq(feature)]
        curves = fold_local_pdp_curves(rows)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(curves)))
        for (fold, fold_rows), color in zip(curves.items(), colors):
            ax.plot(
                fold_rows["grid_value"],
                fold_rows["mean_prediction_watts"],
                color=color,
                alpha=0.82,
                linewidth=1.6,
                marker="o",
                markersize=2.5,
                label=f"fold {fold}",
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean predicted power (W)")
        ax.grid(alpha=0.2)
    if len(features):
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.suptitle("Fold-local partial dependence — B4 + XGBoost", fontsize=13)
    _save_figure(fig, "partial_dependence_best.png")


def main(argv: Iterable[str] | None = None) -> None:
    del argv
    failures = verify_source_freeze(ROOT)
    if failures:
        raise RuntimeError("; ".join(failures))
    frame = pd.read_parquet(STEP5 / "analysis/phase3_modeling_table.parquet")
    physics = frame.loc[frame["eligible_phase20_physics"]].copy()
    if len(physics) != 305:
        raise ValueError("Phase 3 analysis requires exactly 305 physics-complete rows")
    bundles = load_feature_bundles(ROOT)["bundles"]
    search = load_json("step5/config/model_search_spaces.json")
    core = pd.read_parquet(STEP5 / "analysis/oof_predictions.parquet")
    hardware_predictions, hardware_summary = build_hardware_sensitivity(physics, core, bundles, search)
    paired_summary, paired_draws = build_hardware_paired_comparisons(core, hardware_predictions)

    generalization_predictions, generalization_summary = build_generalization(physics, bundles, search)
    strategy_predictions, strategy_summary = build_target_strategies(physics, bundles, search)
    importance, pdp = build_interpretation(physics, bundles, search)
    slices = build_error_slices(core, hardware_predictions, physics)

    analysis_dir = STEP5 / "analysis"
    report_dir = STEP5 / "reports"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    generalization_predictions.to_parquet(analysis_dir / "generalization_predictions.parquet", index=False)
    strategy_predictions.to_parquet(analysis_dir / "target_strategy_predictions.parquet", index=False)
    hardware_predictions.to_parquet(analysis_dir / "hardware_label_sensitivity_predictions.parquet", index=False)
    generalization_summary.to_csv(report_dir / "generalization_ladder.csv", index=False)
    strategy_summary.to_csv(report_dir / "target_strategy_comparison.csv", index=False)
    hardware_summary.to_csv(report_dir / "hardware_label_sensitivity.csv", index=False)
    paired_summary.to_csv(report_dir / "hardware_paired_comparisons.csv", index=False)
    paired_draws.to_csv(report_dir / "hardware_paired_bootstrap_draws.csv", index=False)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)
    pdp.to_csv(report_dir / "partial_dependence.csv", index=False)
    slices.to_csv(report_dir / "error_slices.csv", index=False)
    build_figures(generalization_predictions, core, importance, pdp)
    summary = {
        "exploratory_power_variant": "B4-Hardware__XGBoost",
        "selection_status": "outcome-informed exploratory follow-up; not an unbiased winner estimate",
        "generalization_rows": len(generalization_summary),
        "strategy_rows": len(strategy_summary),
        "hardware_sensitivity_rows": len(hardware_summary),
        "hardware_paired_comparisons": len(paired_summary),
        "importance_rows": len(importance),
        "pdp_features": int(pdp["feature"].nunique()),
        "shap_status": "not_available; fold-local permutation importance and PDP used",
    }
    (analysis_dir / "phase3_analysis_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
