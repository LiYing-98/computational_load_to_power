#!/usr/bin/env python3
"""Portable, input-complete checkpoint identities for Phase 4 experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from step6.scripts.common import sha256_file


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def dataframe_digest(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected_columns = list(dict.fromkeys(columns))
    ordered = frame[selected_columns].sort_values(selected_columns[0]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"columns": selected_columns, "dtypes": [str(ordered[column].dtype) for column in selected_columns]},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def canonicalize_experiment_inputs(
    frame: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the same stable row order for both hashing and estimator fitting."""
    return (
        frame.sort_values("run_key", kind="mergesort").reset_index(drop=True),
        folds.sort_values(["fold_id", "run_key"], kind="mergesort").reset_index(drop=True),
    )


def checkpoint_fingerprint(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    spec: dict[str, Any],
    bundle: dict[str, Any],
    candidates: list[dict[str, Any]],
    target: str,
    inner_group_column: str,
    implementation_paths: Iterable[Path],
) -> str:
    """Fingerprint every material input that can change fitted predictions."""
    feature_columns = list(bundle["columns"])
    payload = {
        "schema_version": "1.0",
        "spec": spec,
        "bundle": {
            "categorical": list(bundle["categorical"]),
            "numeric": list(bundle["numeric"]),
            "columns": feature_columns,
        },
        "candidates": candidates,
        "target": target,
        "inner_group_column": inner_group_column,
        "dataframe_sha256": dataframe_digest(
            frame, ["run_key", inner_group_column, *feature_columns, target]
        ),
        "folds_sha256": dataframe_digest(folds, ["run_key", "fold_id"]),
        "implementation_sha256": {
            str(Path(path).name): sha256_file(Path(path)) for path in implementation_paths
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": _distribution_version("numpy"),
            "scikit-learn": _distribution_version("scikit-learn"),
            "catboost": _distribution_version("catboost"),
            "xgboost": _distribution_version("xgboost"),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
