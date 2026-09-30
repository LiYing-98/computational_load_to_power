#!/usr/bin/env python3
"""Build deterministic Phase 4 development and generalization folds."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import GroupKFold

from step6.scripts.common import ROOT, STEP6, sha256_file, write_json


IDENTITY = [
    "run_key",
    "diag_model_id",
    "diag_parent_lineage",
    "diag_subfamily",
    "diag_architecture_class",
    "x_protocol_task",
]


def build_model_folds(development: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    ordered = development.sort_values(["diag_model_id", "run_key"]).reset_index(drop=True)
    folds = pd.Series(index=ordered.index, dtype="int64")
    splitter = GroupKFold(n_splits=n_splits)
    for fold_id, (_, test_index) in enumerate(
        splitter.split(ordered, groups=ordered["diag_model_id"])
    ):
        folds.iloc[test_index] = fold_id
    result = ordered[IDENTITY].copy()
    result["fold_id"] = folds.astype(int)
    result["split_scheme"] = "development_model_holdout"
    return result.sort_values("run_key").reset_index(drop=True)


def build_leave_group_out(development: pd.DataFrame, group_column: str, scheme: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for fold_id, group_value in enumerate(sorted(development[group_column].dropna().unique())):
        test = development.loc[development[group_column].eq(group_value), IDENTITY].copy()
        test["fold_id"] = fold_id
        test["held_out_group"] = group_value
        test["split_scheme"] = scheme
        chunks.append(test)
    return pd.concat(chunks, ignore_index=True).sort_values(["fold_id", "run_key"]).reset_index(drop=True)


def build_within_task_lineage(development: pd.DataFrame) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    fold_id = 0
    counts = development.groupby("x_protocol_task")["diag_parent_lineage"].nunique()
    for task in sorted(counts.loc[counts >= 3].index):
        task_frame = development.loc[development["x_protocol_task"].eq(task)]
        for lineage in sorted(task_frame["diag_parent_lineage"].unique()):
            test = task_frame.loc[task_frame["diag_parent_lineage"].eq(lineage), IDENTITY].copy()
            test["fold_id"] = fold_id
            test["held_out_group"] = lineage
            test["split_scheme"] = "within_task_lineage_holdout"
            chunks.append(test)
            fold_id += 1
    if not chunks:
        return pd.DataFrame(columns=[*IDENTITY, "fold_id", "held_out_group", "split_scheme"])
    return pd.concat(chunks, ignore_index=True).sort_values(["fold_id", "run_key"]).reset_index(drop=True)


def main() -> None:
    table_path = STEP6 / "analysis/phase4_power_modeling_table.parquet"
    frame = pd.read_parquet(table_path)
    development = frame.loc[frame["eligible_phase4_development"].astype(bool)].copy()
    outputs = {
        "development_model_folds.csv": build_model_folds(development),
        "lineage_holdout_folds.csv": build_leave_group_out(
            development, "diag_parent_lineage", "development_parent_lineage_holdout"
        ),
        "within_task_lineage_folds.csv": build_within_task_lineage(development),
        "architecture_holdout_folds.csv": build_leave_group_out(
            development, "diag_architecture_class", "development_architecture_holdout"
        ),
        "full_cohort_lineage_holdout_folds.csv": build_leave_group_out(
            frame, "diag_parent_lineage", "parent_lineage_holdout"
        ).assign(post_lock_diagnostic=True),
        "full_cohort_within_task_lineage_folds.csv": build_within_task_lineage(frame).assign(
            post_lock_diagnostic=True
        ),
        "full_cohort_architecture_holdout_folds.csv": build_leave_group_out(
            frame, "diag_architecture_class", "architecture_class_holdout_stress"
        ).assign(post_lock_diagnostic=True),
    }
    split_dir = STEP6 / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for filename, manifest in outputs.items():
        manifest.to_csv(split_dir / filename, index=False)
    write_json(
        split_dir / "split_manifest_phase4.json",
        {
            "schema_version": "1.0",
            "source_table": str(table_path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256_file(table_path),
            "confirmatory_rows_excluded": int(frame["diag_is_confirmatory_model"].sum()),
            "development_rows": len(development),
            "development_models": int(development["diag_model_id"].nunique()),
            "manifests": {
                filename: {"rows": len(value), "folds": int(value["fold_id"].nunique())}
                for filename, value in outputs.items()
            },
        },
    )
    print(
        json.dumps(
            {
                "development_rows": len(development),
                "development_models": int(development["diag_model_id"].nunique()),
                "fold_counts": {name: int(value["fold_id"].nunique()) for name, value in outputs.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
