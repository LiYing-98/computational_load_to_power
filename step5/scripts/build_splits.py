#!/usr/bin/env python3
"""Create deterministic Phase 3 family and repeated-random split manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from step5.scripts.common import ROOT, SEED, STEP5, sha256_file
from step5.scripts.modeling import package_version


COHORTS = {
    "base": "eligible_primary_stable",
    "static": "eligible_static_architecture",
    "full": "eligible_phase20_full",
    "physics": "eligible_phase20_physics",
}


def family_folds(frame: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for cohort, flag in COHORTS.items():
        subset = frame.loc[frame[flag], ["run_key", "diag_model_id", "diag_model_family"]].copy()
        groups = subset["diag_model_family"].astype(str)
        n_splits = min(5, groups.nunique())
        splitter = GroupKFold(n_splits=n_splits)
        subset["outer_fold"] = -1
        for fold, (_, held_index) in enumerate(splitter.split(subset, groups=groups)):
            subset.iloc[held_index, subset.columns.get_loc("outer_fold")] = fold
        subset.insert(0, "cohort", cohort)
        if (subset.groupby("diag_model_family")["outer_fold"].nunique() != 1).any():
            raise RuntimeError(f"family leakage in {cohort}")
        outputs.append(subset)
    return pd.concat(outputs, ignore_index=True)


def repeated_random_splits(frame: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for cohort in ("base", "physics"):
        flag = COHORTS[cohort]
        keys = frame.loc[frame[flag], "run_key"].sort_values().reset_index(drop=True)
        for repeat_index in range(10):
            seed = SEED + repeat_index * 101
            train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=seed, shuffle=True)
            roles = pd.DataFrame({"run_key": keys})
            test_set = set(test_keys)
            roles["split_role"] = roles["run_key"].map(lambda key: "test" if key in test_set else "train")
            roles.insert(0, "seed", seed)
            roles.insert(0, "repeat_index", repeat_index)
            roles.insert(0, "cohort", cohort)
            outputs.append(roles)
    return pd.concat(outputs, ignore_index=True)


def model_manifest() -> dict:
    xgboost_version = package_version("xgboost")
    catboost_version = package_version("catboost")
    shap_version = package_version("shap")
    return {
        "schema_version": "1.0",
        "seed": SEED,
        "backends": {
            "XGBoost": {"available": xgboost_version is not None, "version": xgboost_version, "fallback": "sklearn.GradientBoostingRegressor"},
            "CatBoost": {"available": catboost_version is not None, "version": catboost_version, "fallback": "sklearn.RandomForestRegressor_with_one_hot"},
            "SHAP": {"available": shap_version is not None, "version": shap_version, "fallback": "permutation_importance_and_partial_dependence"},
        },
        "selection": "outer-train-only group-aware inner CV",
    }


def main() -> None:
    frame = pd.read_parquet(STEP5 / "analysis/phase3_modeling_table.parquet")
    split_dir = STEP5 / "splits"
    model_dir = STEP5 / "models"
    split_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    family_path = split_dir / "family_group_folds.csv"
    random_path = split_dir / "random_split_manifest.csv"
    family_folds(frame).to_csv(family_path, index=False)
    repeated_random_splits(frame).to_csv(random_path, index=False)
    model_path = model_dir / "model_manifest.json"
    model_path.write_text(json.dumps(model_manifest(), indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "seed": SEED,
        "family_group_folds": {"path": "step5/splits/family_group_folds.csv", "sha256": sha256_file(family_path)},
        "random_split_manifest": {"path": "step5/splits/random_split_manifest.csv", "sha256": sha256_file(random_path), "repeats": 10},
        "frozen_model_folds": "step4/splits/model_group_folds.csv",
        "frozen_config_folds": "step4/splits/config_group_folds.csv",
        "model_manifest": {"path": "step5/models/model_manifest.json", "sha256": sha256_file(model_path)},
    }
    (split_dir / "split_manifest_phase3.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"family_rows": len(family_folds(frame)), "random_rows": len(repeated_random_splits(frame)), **model_manifest()["backends"]}, indent=2))


if __name__ == "__main__":
    main()
