#!/usr/bin/env python3
"""Build the Phase 4 all-stable power table and lock the confirmatory IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from step6.scripts.common import (
    ROOT,
    SEED,
    STEP6,
    load_feature_bundles,
    load_json,
    stable_hash,
    verify_source_freeze,
    write_json,
)


def _select_confirmatory_models(frame: pd.DataFrame) -> list[dict[str, str]]:
    candidates = frame[["diag_parent_lineage", "diag_model_id"]].drop_duplicates()
    selected: list[dict[str, str]] = []
    for lineage, group in candidates.groupby("diag_parent_lineage", sort=True):
        model_ids = sorted(group["diag_model_id"].astype(str).unique())
        if len(model_ids) < 2:
            continue
        chosen = min(model_ids, key=lambda model_id: stable_hash(f"{lineage}|{model_id}"))
        selected.append({"diag_parent_lineage": str(lineage), "diag_model_id": chosen})
    return selected


def build_phase4_table(project_root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    failures = verify_source_freeze(project_root)
    if failures:
        raise RuntimeError("; ".join(failures))
    root = Path(project_root)
    source = pd.read_parquet(root / "step5/analysis/phase3_modeling_table.parquet")
    frame = source.loc[source["eligible_primary_stable"].astype(bool)].copy()
    taxonomy = load_json("step6/config/lineage_taxonomy.json", root)
    frame["diag_subfamily"] = frame["diag_model_family"].astype(str)
    frame["diag_parent_lineage"] = frame["diag_subfamily"].map(
        taxonomy["subfamily_to_parent_lineage"]
    )
    frame["diag_architecture_class"] = frame["x_model_architecture_family"].map(
        taxonomy["architecture_mapping"]
    )
    # H100 has no native FP4 peak in the frozen reference profile. The benchmark's
    # MXFP4 weights therefore cannot be matched to a hardware FP4 rate; use the
    # documented BF16 dense reference as a conservative execution-compute proxy.
    missing_matched = frame["x_hardware_precision_matched_peak_compute_tflops"].isna()
    frame["diag_precision_compute_proxy_status"] = "precision_matched"
    frame.loc[missing_matched, "x_hardware_precision_matched_peak_compute_tflops"] = frame.loc[
        missing_matched, "x_hardware_peak_compute_bf16_dense_tflops"
    ]
    frame.loc[missing_matched, "diag_precision_compute_proxy_status"] = (
        "bf16_fallback_for_unsupported_weight_precision"
    )
    if frame[["diag_parent_lineage", "diag_architecture_class"]].isna().any().any():
        missing = frame.loc[
            frame[["diag_parent_lineage", "diag_architecture_class"]].isna().any(axis=1),
            ["diag_model_id", "diag_subfamily", "x_model_architecture_family"],
        ].drop_duplicates()
        raise ValueError(f"unmapped Phase 4 taxonomy rows: {missing.to_dict('records')}")
    if len(frame) != 565 or frame["run_key"].nunique() != 565:
        raise ValueError("Phase 4 main cohort must contain 565 unique stable runs")
    if frame["diag_model_id"].nunique() != 27:
        raise ValueError("Phase 4 main cohort must contain 27 model IDs")

    selected = _select_confirmatory_models(frame)
    payload = {
        "schema_version": "1.0",
        "created_before_phase4_training": True,
        "selection_rule": "For each parent lineage with at least two model IDs, choose the smallest SHA256(seed|lineage|model_id); target values are not read.",
        "seed": SEED,
        "stratification": "diag_parent_lineage",
        "held_out_models": selected,
        "held_out_model_ids": [item["diag_model_id"] for item in selected],
    }
    holdout_path = root / "step6/config/confirmatory_model_holdout.json"
    if holdout_path.exists():
        existing = json.loads(holdout_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("locked confirmatory model holdout differs from deterministic rule")
    else:
        write_json(holdout_path, payload)
    held_ids = set(payload["held_out_model_ids"])
    frame["diag_is_confirmatory_model"] = frame["diag_model_id"].isin(held_ids)
    frame["eligible_phase4_main"] = True
    frame["eligible_phase4_development"] = ~frame["diag_is_confirmatory_model"]

    bundles = load_feature_bundles(root)["bundles"]
    required = sorted({column for bundle in bundles.values() for column in bundle["columns"]})
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Phase 4 table is missing feature columns: {missing_columns}")
    return frame.sort_values("run_key").reset_index(drop=True), payload


def write_outputs(frame: pd.DataFrame, output_root: Path = STEP6) -> None:
    analysis = Path(output_root) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(analysis / "phase4_power_modeling_table.parquet", index=False)

    by_lineage = (
        frame.groupby(["diag_parent_lineage", "diag_subfamily", "diag_architecture_class"], dropna=False)
        .agg(n_runs=("run_key", "size"), n_models=("diag_model_id", "nunique"))
        .reset_index()
    )
    by_lineage.to_csv(analysis / "coverage_by_lineage.csv", index=False)
    by_task = (
        frame.groupby(
            ["x_protocol_task", "diag_parent_lineage", "diag_subfamily", "diag_model_id"],
            dropna=False,
        )
        .agg(n_runs=("run_key", "size"))
        .reset_index()
    )
    by_task.to_csv(analysis / "coverage_by_task_lineage.csv", index=False)


def main() -> None:
    frame, holdout = build_phase4_table(ROOT)
    write_outputs(frame, STEP6)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "models": int(frame["diag_model_id"].nunique()),
                "llama_rows": int(frame["diag_parent_lineage"].eq("Llama").sum()),
                "confirmatory_models": holdout["held_out_model_ids"],
                "confirmatory_rows": int(frame["diag_is_confirmatory_model"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
