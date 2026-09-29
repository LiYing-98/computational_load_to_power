#!/usr/bin/env python3
"""Enrich the frozen Phase 2 table with auditable hardware physics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step5.scripts.common import ROOT, STEP5, load_feature_bundles, load_json, verify_source_freeze


PROFILE_COLUMN_MAP = {
    "peak_compute_bf16_dense_tflops": "x_hardware_peak_compute_bf16_dense_tflops",
    "peak_compute_fp8_dense_tflops": "x_hardware_peak_compute_fp8_dense_tflops",
    "peak_compute_fp4_dense_tflops": "x_hardware_peak_compute_fp4_dense_tflops",
    "hbm_bandwidth_gbps": "x_hardware_hbm_bandwidth_gbps",
    "hbm_capacity_gb": "x_hardware_hbm_capacity_gb",
    "rated_power_w": "x_hardware_rated_power_w",
    "interconnect_bandwidth_gbps": "x_hardware_interconnect_bandwidth_gbps",
}


def _positive(value: Any) -> float:
    number = float(value) if value is not None and not pd.isna(value) else math.nan
    return number if np.isfinite(number) and number > 0 else math.nan


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    top = float(numerator) if numerator is not None and not pd.isna(numerator) else math.nan
    bottom = _positive(denominator)
    return float(top / bottom) if np.isfinite(top) and np.isfinite(bottom) else math.nan


def derive_hardware_features(
    row: pd.Series,
    profiles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return static and normalized hardware proxies for one run."""
    if profiles is None:
        profiles = load_json("step5/config/gpu_hardware_specs.json")["profiles"]
    label = str(row["x_hardware_gpu_model"])
    if label not in profiles:
        raise ValueError(f"unmapped GPU label: {label}")
    profile = profiles[label]
    derived: dict[str, Any] = {
        "diag_gpu_profile_id": profile["gpu_profile_id"],
        "diag_hardware_source_status": profile["hardware_source_status"],
    }
    for source, target in PROFILE_COLUMN_MAP.items():
        derived[target] = profile.get(source)

    precision = str(row["x_model_benchmark_weight_precision"]).lower()
    peak_field = {
        "bfloat16": "peak_compute_bf16_dense_tflops",
        "fp8": "peak_compute_fp8_dense_tflops",
        "mxfp4": "peak_compute_fp4_dense_tflops",
    }.get(precision)
    matched_peak = _positive(profile.get(peak_field)) if peak_field else math.nan
    derived["x_hardware_precision_matched_peak_compute_tflops"] = matched_peak

    num_gpus = _positive(row.get("x_hardware_num_gpus"))
    model_parallel_gpus = _positive(row.get("x_deployment_model_parallel_gpus_per_replica"))
    if not np.isfinite(model_parallel_gpus):
        model_parallel_gpus = _positive(row.get("x_deployment_tensor_parallel"))
    aggregate_power = _positive(profile.get("rated_power_w")) * num_gpus
    derived["x_hardware_aggregate_rated_power_w"] = aggregate_power

    theoretical_flops = _positive(row.get("x_physics_prefill_total_flop_per_request"))
    aggregate_compute_flops_per_second = matched_peak * 1e12 * model_parallel_gpus
    compute_time = _safe_ratio(theoretical_flops, aggregate_compute_flops_per_second)
    derived["x_hardware_physics_compute_time_proxy_seconds"] = compute_time

    weight_per_gpu = float(row.get("x_physics_weight_bytes_per_gpu_ideal", math.nan))
    kv_mean_per_gpu = _safe_ratio(
        row.get("x_physics_kv_cache_prefill_mean_request_bytes"), model_parallel_gpus
    )
    kv_max_per_gpu = _safe_ratio(
        row.get("x_physics_kv_cache_prefill_max_request_bytes"), model_parallel_gpus
    )
    bandwidth_bytes_per_second = _positive(profile.get("hbm_bandwidth_gbps")) * 1e9
    selected_bytes = (
        weight_per_gpu + kv_mean_per_gpu
        if np.isfinite(weight_per_gpu) and np.isfinite(kv_mean_per_gpu)
        else math.nan
    )
    memory_time = _safe_ratio(selected_bytes, bandwidth_bytes_per_second)
    derived["x_hardware_physics_memory_time_proxy_seconds"] = memory_time

    capacity_bytes = _positive(profile.get("hbm_capacity_gb")) * 1e9
    derived["x_hardware_physics_weight_capacity_ratio"] = _safe_ratio(weight_per_gpu, capacity_bytes)
    derived["x_hardware_physics_kv_capacity_ratio_mean"] = _safe_ratio(kv_mean_per_gpu, capacity_bytes)
    derived["x_hardware_physics_kv_capacity_ratio_max"] = _safe_ratio(kv_max_per_gpu, capacity_bytes)
    derived["x_hardware_physics_compute_to_memory_pressure_ratio"] = _safe_ratio(
        compute_time, memory_time
    )
    derived["y_gpu_power_fraction_of_rated"] = _safe_ratio(
        row.get("y_gpu_avg_power_watts"), aggregate_power
    )
    return derived


def build_phase3_table(project_root: Path = ROOT) -> pd.DataFrame:
    failures = verify_source_freeze(project_root)
    if failures:
        raise RuntimeError("; ".join(failures))
    root = Path(project_root)
    source = pd.read_parquet(root / "step4/analysis/modeling_table.parquet")
    profiles = load_json("step5/config/gpu_hardware_specs.json", root)["profiles"]
    derived = pd.DataFrame(
        [derive_hardware_features(row, profiles) for _, row in source.iterrows()],
        index=source.index,
    )
    enhanced = pd.concat([source.copy(), derived], axis=1)
    enhanced["diag_load_avg_batch_to_capacity_ratio"] = np.divide(
        enhanced["diag_result_avg_batch_size"].astype(float),
        enhanced["x_deployment_max_num_seqs"].astype(float),
        out=np.full(len(enhanced), np.nan),
        where=enhanced["x_deployment_max_num_seqs"].to_numpy(dtype=float) > 0,
    )
    enhanced["diag_load_request_throughput_req_per_sec"] = enhanced[
        "diag_result_request_throughput_req_per_sec"
    ]
    if len(enhanced) != 694 or enhanced["run_key"].nunique() != 694:
        raise ValueError("Phase 3 table must preserve 694 unique runs")
    bundles = load_feature_bundles(root)["bundles"]
    missing = sorted(
        {
            column
            for name, bundle in bundles.items()
            if name.startswith(("B4-", "B5-"))
            for column in bundle["columns"]
            if column not in enhanced
        }
    )
    if missing:
        raise ValueError(f"Phase 3 table is missing bundle columns: {missing}")
    return enhanced


def write_load_regime_outputs(frame: pd.DataFrame, output_root: Path = STEP5) -> None:
    stable = frame.loc[frame["eligible_primary_stable"]].copy()
    ratio = stable["diag_load_avg_batch_to_capacity_ratio"].dropna()
    throughput = stable["diag_load_request_throughput_req_per_sec"].dropna()
    stats = ratio.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

    figure_dir = Path(output_root) / "reports/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(ratio, bins=np.linspace(0.84, 1.005, 18), color="#3572A5", edgecolor="white")
    axes[0].axvline(float(ratio.median()), color="#222222", linestyle="--", linewidth=1.5)
    axes[0].set(title="Average batch occupancy", xlabel="avg batch size / max_num_seqs", ylabel="Stable runs")
    axes[1].hist(throughput, bins=18, color="#D97706", edgecolor="white")
    axes[1].set(title="Observed request throughput", xlabel="requests / second (diagnostic only)", ylabel="Stable runs")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    fig.suptitle("ML.ENERGY V3 stable LLM inference load regime")
    fig.tight_layout()
    fig.savefig(figure_dir / "load_regime_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Load regime characterization

## Finding

The 565 stable GPU-side text-LLM inference runs are best described as a
**high-load, near-saturation steady-state serving regime**. Median observed
average-batch occupancy is {stats.loc[0.5]:.4f} of configured `max_num_seqs`;
the 5th–95th percentile interval is {stats.loc[0.05]:.4f}–{stats.loc[0.95]:.4f}.

## Evidence

- Stable runs: {len(stable)}; all have an observed average batch-size diagnostic.
- Occupancy p25/p50/p75: {stats.loc[0.25]:.4f} / {stats.loc[0.5]:.4f} / {stats.loc[0.75]:.4f}.
- Minimum/maximum occupancy: {ratio.min():.4f} / {ratio.max():.4f}.
- Observed request throughput spans {throughput.min():.4f}–{throughput.max():.4f} requests/s.

## Boundary

Average batch size and request throughput are post-run, **diagnostic-only**
fields. They are not included in B0–B5 and are never formal model inputs. The
evidence supports prediction for high-load steady-state operation; it does not
identify an arbitrary request-arrival response curve `P(lambda)`.
"""
    report_path = Path(output_root) / "reports/load_regime_characterization.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    table = build_phase3_table(ROOT)
    analysis_dir = STEP5 / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(analysis_dir / "phase3_modeling_table.parquet", index=False)
    table.to_csv(analysis_dir / "phase3_modeling_table.csv", index=False)
    write_load_regime_outputs(table, STEP5)
    receipt = {
        "rows": len(table),
        "unique_runs": int(table["run_key"].nunique()),
        "stable_runs": int(table["eligible_primary_stable"].sum()),
        "physics_runs": int(table["eligible_phase20_physics"].sum()),
    }
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
