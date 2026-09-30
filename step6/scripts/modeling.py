#!/usr/bin/env python3
"""Leakage-safe, group-aware modeling primitives for Phase 4.0."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from step6.scripts.common import SEED


LOCAL_PACKAGES = Path(__file__).resolve().parents[2] / "step5" / ".packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))


def _preprocessor(
    numeric: Sequence[str], categorical: Sequence[str], *, scale_numeric: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(numeric)),
            ("categorical", categorical_pipeline, list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(
    algorithm: str,
    candidate: dict[str, Any],
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    objective: str = "absolute_error",
) -> Pipeline:
    params = dict(candidate)
    scale = algorithm == "Ridge"
    if algorithm == "DummyMean":
        model: Any = DummyRegressor(strategy="mean")
    elif algorithm == "Ridge":
        model = Ridge(alpha=float(params["alpha"]))
    elif algorithm == "HistGradientBoosting":
        loss = {"absolute_error": "absolute_error", "squared_error": "squared_error"}.get(objective)
        if loss is None:
            raise ValueError(f"HistGradientBoosting does not support objective {objective}")
        model = HistGradientBoostingRegressor(**params, loss=loss, random_state=SEED)
    elif algorithm == "RandomForest":
        criterion = {"absolute_error": "absolute_error", "squared_error": "squared_error"}.get(objective)
        if criterion is None:
            raise ValueError(f"RandomForest does not support objective {objective}")
        # Single-threaded tree aggregation avoids platform-dependent last-bit
        # drift that would otherwise change frozen prediction/lock hashes.
        model = RandomForestRegressor(**params, criterion=criterion, random_state=SEED, n_jobs=1)
    elif algorithm == "XGBoost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            **params,
            objective={
                "absolute_error": "reg:absoluteerror",
                "squared_error": "reg:squarederror",
                "pseudo_huber": "reg:pseudohubererror",
            }[objective],
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
    elif algorithm == "CatBoost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            **params,
            loss_function={"absolute_error": "MAE", "squared_error": "RMSE", "pseudo_huber": "Huber:delta=1.0"}[objective],
            random_seed=SEED,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale_numeric=scale)),
            ("model", model),
        ]
    )


def high_power_weights(target_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(target_values, dtype=float)
    median = float(np.median(values))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("high-power weighting requires a finite positive training median")
    return np.clip(np.sqrt(values / median), 0.5, 2.0)


def _fit(
    estimator: Pipeline,
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str,
    *,
    use_high_power_weights: bool,
) -> None:
    y = frame[target].to_numpy(dtype=float)
    model = estimator.named_steps["model"]
    if model.get_params().get("objective") == "reg:pseudohubererror":
        # XGBoost defaults delta/huber_slope to 1.0. On watt-scale targets that
        # makes almost every residual lie in the saturated-gradient regime.
        # Scale delta from this fit partition only, preserving fold isolation.
        model.set_params(huber_slope=float(np.median(y)))
    fit_kwargs = {"model__sample_weight": high_power_weights(y)} if use_high_power_weights else {}
    estimator.fit(frame[list(columns)], y, **fit_kwargs)


def select_candidate(
    train: pd.DataFrame,
    *,
    numeric: Sequence[str],
    categorical: Sequence[str],
    target: str,
    algorithm: str,
    candidates: Sequence[dict[str, Any]],
    inner_group_column: str,
    objective: str = "absolute_error",
    use_high_power_weights: bool = False,
) -> tuple[int, list[float]]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len(candidates) == 1:
        return 0, [math.nan]
    groups = train[inner_group_column].astype(str)
    n_splits = min(3, int(groups.nunique()))
    if n_splits < 2:
        return 0, [math.nan for _ in candidates]
    splitter = GroupKFold(n_splits=n_splits)
    columns = [*numeric, *categorical]
    scores: list[float] = []
    for candidate in candidates:
        absolute_error = 0.0
        validation_rows = 0
        for fit_index, valid_index in splitter.split(train, groups=groups):
            fit_frame = train.iloc[fit_index]
            valid_frame = train.iloc[valid_index]
            estimator = make_estimator(
                algorithm, candidate, numeric, categorical, objective=objective
            )
            _fit(
                estimator,
                fit_frame,
                columns,
                target,
                use_high_power_weights=use_high_power_weights,
            )
            prediction = estimator.predict(valid_frame[columns])
            absolute_error += float(
                np.abs(valid_frame[target].to_numpy(dtype=float) - prediction).sum()
            )
            validation_rows += len(valid_frame)
        scores.append(absolute_error / validation_rows)
    selected = min(range(len(candidates)), key=lambda index: (scores[index], index))
    return selected, scores


def fit_predict_outer_fold(
    train: pd.DataFrame,
    held_out: pd.DataFrame,
    *,
    numeric: Sequence[str],
    categorical: Sequence[str],
    target: str,
    algorithm: str,
    candidates: Sequence[dict[str, Any]],
    inner_group_column: str,
    objective: str = "absolute_error",
    use_high_power_weights: bool = False,
) -> tuple[np.ndarray, dict[str, Any], Pipeline]:
    selected, scores = select_candidate(
        train,
        numeric=numeric,
        categorical=categorical,
        target=target,
        algorithm=algorithm,
        candidates=candidates,
        inner_group_column=inner_group_column,
        objective=objective,
        use_high_power_weights=use_high_power_weights,
    )
    candidate = dict(candidates[selected])
    estimator = make_estimator(
        algorithm, candidate, numeric, categorical, objective=objective
    )
    columns = [*numeric, *categorical]
    _fit(
        estimator,
        train,
        columns,
        target,
        use_high_power_weights=use_high_power_weights,
    )
    prediction = np.asarray(estimator.predict(held_out[columns]), dtype=float)
    metadata = {
        "selected_candidate_index": selected,
        "selected_candidate": candidate,
        "inner_validation_mae": scores,
        "inner_group_column": inner_group_column,
        "inner_train_groups": int(train[inner_group_column].nunique()),
        "inner_train_rows": len(train),
        "outer_test_rows": len(held_out),
        "backend_module": estimator.named_steps["model"].__class__.__module__,
        "objective": objective,
        "use_high_power_weights": use_high_power_weights,
        "outer_huber_slope": (
            float(estimator.named_steps["model"].get_params().get("huber_slope"))
            if objective == "pseudo_huber"
            else None
        ),
    }
    return prediction, metadata, estimator


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    residual = p - y
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else math.nan
    mdape = float(np.median(np.abs(residual) / np.maximum(np.abs(y), 1e-12)) * 100.0)
    smape = float(np.mean(2.0 * np.abs(residual) / np.maximum(np.abs(y) + np.abs(p), 1e-12)) * 100.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mdape": mdape,
        "smape": smape,
        "signed_bias": float(np.mean(residual)),
    }


def metadata_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
