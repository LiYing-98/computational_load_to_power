#!/usr/bin/env python3
"""Leakage-safe nested-CV primitives for Phase 3 table models."""

from __future__ import annotations

import json
import math
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from step5.scripts.common import SEED


LOCAL_PACKAGES = Path(__file__).resolve().parents[1] / ".packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        try:
            module = __import__(name)
            return str(module.__version__)
        except (ImportError, AttributeError):
            return None


def transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if transform == "raw":
        return array
    if transform == "log1p":
        if (array < 0).any():
            raise ValueError("log1p target transform requires nonnegative values")
        return np.log1p(array)
    raise ValueError(f"unknown target transform: {transform}")


def inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if transform == "raw":
        return array
    if transform == "log1p":
        return np.expm1(array)
    raise ValueError(f"unknown target transform: {transform}")


def _preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    scale_numeric: bool,
    spline: dict[str, Any] | None = None,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    if spline is not None:
        numeric_steps.append(
            (
                "spline",
                SplineTransformer(
                    n_knots=int(spline["n_knots"]),
                    degree=int(spline["degree"]),
                    include_bias=False,
                ),
            )
        )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", Pipeline(numeric_steps), list(numeric)), ("categorical", categorical_pipeline, list(categorical))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(
    algorithm: str,
    candidate: dict[str, Any],
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> Any:
    """Construct a deterministic estimator without fitting any supplied frame."""
    del frame  # dtypes/levels are intentionally learned only by fit on outer-train.
    params = dict(candidate)
    if algorithm == "DummyMean":
        return DummyRegressor(strategy="mean")
    if algorithm == "Ridge":
        model = Ridge(alpha=float(params["alpha"]))
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=True)), ("model", model)])
    if algorithm == "ElasticNet":
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=20000,
            random_state=SEED,
        )
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=True)), ("model", model)])
    if algorithm == "RandomForest":
        model = RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=False)), ("model", model)])
    if algorithm == "ExtraTrees":
        model = ExtraTreesRegressor(**params, random_state=SEED, n_jobs=-1)
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=False)), ("model", model)])
    if algorithm == "HistGradientBoosting":
        model = HistGradientBoostingRegressor(**params, random_state=SEED)
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=False)), ("model", model)])
    if algorithm == "XGBoost":
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                **params,
                objective="reg:absoluteerror",
                random_state=SEED,
                n_jobs=1,
                verbosity=0,
            )
        except ImportError:
            model = GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 3)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 1.0)),
                loss="absolute_error",
                random_state=SEED,
            )
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=False)), ("model", model)])
    if algorithm == "CatBoost":
        try:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(**params, loss_function="MAE", random_seed=SEED, verbose=False, thread_count=1)
        except ImportError:
            model = RandomForestRegressor(
                n_estimators=max(200, int(params.get("iterations", 200))),
                max_depth=int(params.get("depth", 5)),
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            )
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=False)), ("model", model)])
    if algorithm == "SplineGAM":
        spline = {"n_knots": params.pop("n_knots"), "degree": params.pop("degree")}
        model = Ridge(alpha=float(params["alpha"]))
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=True, spline=spline)), ("model", model)])
    if algorithm == "SmallMLP":
        if isinstance(params.get("hidden_layer_sizes"), list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        model = MLPRegressor(
            **params,
            activation="relu",
            solver="lbfgs",
            max_iter=3000,
            random_state=SEED,
        )
        return Pipeline([("preprocess", _preprocessor(numeric, categorical, scale_numeric=True)), ("model", model)])
    raise ValueError(f"unknown algorithm: {algorithm}")


def _strategy_target(frame: pd.DataFrame, target: str, strategy: str) -> np.ndarray:
    values = frame[target].to_numpy(dtype=float)
    if strategy == "direct":
        return values
    if strategy == "power_fraction":
        denominator = frame["x_hardware_aggregate_rated_power_w"].to_numpy(dtype=float)
        if (denominator <= 0).any() or not np.isfinite(denominator).all():
            raise ValueError("power_fraction requires finite positive aggregate rated power")
        return values / denominator
    raise ValueError(f"unknown target strategy: {strategy}")


def _restore_strategy(
    strategy_prediction: np.ndarray,
    frame: pd.DataFrame,
    strategy: str,
) -> np.ndarray:
    if strategy == "direct":
        return np.asarray(strategy_prediction, dtype=float)
    if strategy == "power_fraction":
        return np.asarray(strategy_prediction, dtype=float) * frame[
            "x_hardware_aggregate_rated_power_w"
        ].to_numpy(dtype=float)
    raise ValueError(f"unknown target strategy: {strategy}")


def select_candidate(
    train: pd.DataFrame,
    *,
    numeric: Sequence[str],
    categorical: Sequence[str],
    target: str,
    algorithm: str,
    candidates: Sequence[dict[str, Any]],
    inner_group_column: str,
    target_transform: str,
    target_strategy: str,
) -> tuple[int, list[float]]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if algorithm == "DummyMean" or len(candidates) == 1:
        return 0, [math.nan]
    groups = train[inner_group_column].astype(str)
    n_splits = min(3, groups.nunique())
    if n_splits < 2:
        return 0, [math.nan for _ in candidates]
    splitter = GroupKFold(n_splits=n_splits)
    columns = [*numeric, *categorical]
    original_target = train[target].to_numpy(dtype=float)
    scores: list[float] = []
    for candidate in candidates:
        weighted_error = 0.0
        validation_rows = 0
        for fit_index, valid_index in splitter.split(train, groups=groups):
            fit_frame = train.iloc[fit_index]
            valid_frame = train.iloc[valid_index]
            estimator = make_estimator(algorithm, candidate, fit_frame, numeric, categorical)
            fit_target = transform_target(_strategy_target(fit_frame, target, target_strategy), target_transform)
            estimator.fit(fit_frame[columns], fit_target)
            strategy_prediction = inverse_target(estimator.predict(valid_frame[columns]), target_transform)
            prediction = _restore_strategy(strategy_prediction, valid_frame, target_strategy)
            weighted_error += mean_absolute_error(original_target[valid_index], prediction) * len(valid_frame)
            validation_rows += len(valid_frame)
        scores.append(float(weighted_error / validation_rows))
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
    target_transform: str = "raw",
    target_strategy: str = "direct",
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    selected, scores = select_candidate(
        train,
        numeric=numeric,
        categorical=categorical,
        target=target,
        algorithm=algorithm,
        candidates=candidates,
        inner_group_column=inner_group_column,
        target_transform=target_transform,
        target_strategy=target_strategy,
    )
    candidate = dict(candidates[selected])
    estimator = make_estimator(algorithm, candidate, train, numeric, categorical)
    columns = [*numeric, *categorical]
    fit_target = transform_target(_strategy_target(train, target, target_strategy), target_transform)
    estimator.fit(train[columns], fit_target)
    strategy_prediction = inverse_target(estimator.predict(held_out[columns]), target_transform)
    prediction = _restore_strategy(strategy_prediction, held_out, target_strategy)
    backend = estimator.named_steps["model"].__class__.__module__ if isinstance(estimator, Pipeline) else estimator.__class__.__module__
    metadata_out = {
        "selected_candidate_index": selected,
        "selected_candidate": candidate,
        "inner_validation_mae": scores,
        "backend_module": backend,
        "inner_train_rows": len(train),
        "outer_test_rows": len(held_out),
    }
    return np.asarray(prediction, dtype=float), metadata_out, np.asarray(strategy_prediction, dtype=float)


def metadata_json(metadata_value: dict[str, Any]) -> str:
    return json.dumps(metadata_value, sort_keys=True, separators=(",", ":"))
