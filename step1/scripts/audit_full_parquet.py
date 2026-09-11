#!/usr/bin/env python3
"""Audit the gated ML.ENERGY V3 LLM parquet without reading raw timelines."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


TEXT_LLM_TASKS = ("gpqa", "lm-arena-chat", "sourcegraph-fim")

PHASE1_KEY_FIELDS = (
    "task",
    "model_id",
    "gpu_model",
    "num_gpus",
    "max_num_seqs",
    "tensor_parallel",
    "expert_parallel",
    "data_parallel",
)

FULL_KEY_FIELDS = PHASE1_KEY_FIELDS + ("seed", "num_request_repeats")

FIELD_CLASSIFICATIONS: dict[str, tuple[str, bool, str]] = {
    "domain": ("diagnostic", False, "Scope filter; constant within this audit."),
    "task": ("pre_known", True, "Benchmark workload is selected before launch."),
    "model_id": (
        "identifier_risk",
        False,
        "Known before launch but can let a model memorize model identity.",
    ),
    "nickname": ("diagnostic", False, "Display label duplicating model identity."),
    "architecture": ("pre_known", True, "Model architecture is known before launch."),
    "total_params_billions": ("pre_known", True, "Model metadata known before launch."),
    "activated_params_billions": (
        "pre_known",
        True,
        "Activated parameter count is model metadata.",
    ),
    "weight_precision": ("pre_known", True, "Deployment precision is configured before launch."),
    "gpu_model": ("pre_known", True, "GPU family is selected before launch."),
    "num_gpus": ("pre_known", True, "GPU count is selected before launch."),
    "max_num_seqs": (
        "pre_known",
        True,
        "Configured maximum concurrent sequences, not observed average batch.",
    ),
    "seed": ("diagnostic", False, "Preconfigured reproducibility control; not a physical driver."),
    "num_request_repeats": (
        "pre_known",
        True,
        "Planned repetition count determines total request volume before launch.",
    ),
    "tensor_parallel": ("pre_known", True, "Parallel deployment choice."),
    "expert_parallel": ("pre_known", True, "Parallel deployment choice."),
    "data_parallel": ("pre_known", True, "Parallel deployment choice."),
    "steady_state_energy_joules": (
        "target",
        False,
        "Measured aggregate GPU energy during steady state.",
    ),
    "steady_state_duration_seconds": (
        "post_run_leakage",
        False,
        "Realized measurement-window duration is known only after execution.",
    ),
    "energy_per_token_joules": ("target", False, "Measured/derived GPU energy intensity target."),
    "energy_per_request_joules": (
        "target",
        False,
        "Estimated target derived from energy per token and actual output length.",
    ),
    "output_throughput_tokens_per_sec": (
        "post_run_leakage",
        False,
        "Realized throughput is observed after execution.",
    ),
    "request_throughput_req_per_sec": (
        "post_run_leakage",
        False,
        "Realized request throughput is observed after execution.",
    ),
    "avg_power_watts": ("target", False, "Aggregate GPU steady-state power target."),
    "total_output_tokens": (
        "post_run_leakage",
        False,
        "Realized full-benchmark output volume.",
    ),
    "completed_requests": (
        "post_run_leakage",
        False,
        "Realized completed request count, distinct from planned requests.",
    ),
    "avg_output_len": (
        "post_run_leakage",
        False,
        "Actual generated output length is unavailable before execution.",
    ),
    "mean_itl_ms": ("post_run_leakage", False, "Realized inter-token latency."),
    "median_itl_ms": ("post_run_leakage", False, "Realized inter-token latency."),
    "p50_itl_ms": ("post_run_leakage", False, "Realized inter-token latency."),
    "p90_itl_ms": ("post_run_leakage", False, "Realized inter-token latency."),
    "p95_itl_ms": ("post_run_leakage", False, "Realized inter-token latency."),
    "p99_itl_ms": ("post_run_leakage", False, "Realized inter-token latency."),
    "avg_batch_size": (
        "post_run_leakage",
        False,
        "Prometheus-observed average concurrent sequences during steady state.",
    ),
    "is_stable": ("diagnostic", False, "Post-run quality filter."),
    "unstable_reason": ("diagnostic", False, "Post-run quality explanation."),
    "results_path": ("diagnostic", False, "Provenance path for bounded result inspection."),
    "prometheus_path": (
        "diagnostic",
        False,
        "Provenance path only; timeline is intentionally not downloaded.",
    ),
}

NUMERIC_FIELDS = (
    "total_params_billions",
    "activated_params_billions",
    "num_gpus",
    "max_num_seqs",
    "seed",
    "num_request_repeats",
    "tensor_parallel",
    "expert_parallel",
    "data_parallel",
    "steady_state_energy_joules",
    "steady_state_duration_seconds",
    "energy_per_token_joules",
    "energy_per_request_joules",
    "output_throughput_tokens_per_sec",
    "request_throughput_req_per_sec",
    "avg_power_watts",
    "total_output_tokens",
    "completed_requests",
    "avg_output_len",
    "mean_itl_ms",
    "median_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "avg_batch_size",
)


def classify_field(name: str) -> dict[str, Any]:
    category, allowed, rationale = FIELD_CLASSIFICATIONS.get(
        name,
        ("unclassified", False, "No reviewed execution-time classification exists."),
    )
    return {
        "field": name,
        "category": category,
        "allow_as_x_pre": allowed,
        "rationale": rationale,
    }


def candidate_key_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    available = set(columns)
    return {
        "phase1_visible_key": [field for field in PHASE1_KEY_FIELDS if field in available],
        "full_repeat_aware_key": [field for field in FULL_KEY_FIELDS if field in available],
    }


def load_llm_parquet(path: Path, text_only: bool = True) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if text_only:
        return [row for row in rows if row.get("task") in TEXT_LLM_TASKS]
    return rows


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(observed), abs(expected), 1e-12)


def check_energy_power_identity(
    rows: Sequence[dict[str, Any]], relative_tolerance: float = 1e-9
) -> dict[str, Any]:
    errors: list[float] = []
    failed_rows: list[int] = []
    for index, row in enumerate(rows):
        energy = row.get("steady_state_energy_joules")
        duration = row.get("steady_state_duration_seconds")
        power = row.get("avg_power_watts")
        if energy is None or duration in (None, 0) or power is None:
            continue
        error = _relative_error(float(power), float(energy) / float(duration))
        errors.append(error)
        if error > relative_tolerance:
            failed_rows.append(index)
    return {
        "formula": "avg_power_watts = steady_state_energy_joules / steady_state_duration_seconds",
        "checked": len(errors),
        "failed": len(failed_rows),
        "failed_rows": failed_rows[:50],
        "relative_tolerance": relative_tolerance,
        "maximum_relative_error": max(errors, default=None),
        "median_relative_error": statistics.median(errors) if errors else None,
    }


def _product_identity(
    rows: Sequence[dict[str, Any]],
    observed_field: str,
    left_field: str,
    right_field: str,
    relative_tolerance: float = 1e-9,
) -> dict[str, Any]:
    errors: list[float] = []
    failed_rows: list[int] = []
    for index, row in enumerate(rows):
        values = (row.get(observed_field), row.get(left_field), row.get(right_field))
        if any(value is None for value in values):
            continue
        observed, left, right = (float(value) for value in values)
        error = _relative_error(observed, left * right)
        errors.append(error)
        if error > relative_tolerance:
            failed_rows.append(index)
    return {
        "formula": f"{observed_field} = {left_field} * {right_field}",
        "checked": len(errors),
        "failed": len(failed_rows),
        "failed_rows": failed_rows[:50],
        "maximum_relative_error": max(errors, default=None),
    }


def _ratio_identity(
    rows: Sequence[dict[str, Any]],
    observed_field: str,
    numerator_field: str,
    denominator_field: str,
    relative_tolerance: float = 1e-9,
) -> dict[str, Any]:
    errors: list[float] = []
    failed_rows: list[int] = []
    for index, row in enumerate(rows):
        values = (
            row.get(observed_field),
            row.get(numerator_field),
            row.get(denominator_field),
        )
        if any(value is None for value in values) or float(values[2]) == 0:
            continue
        observed, numerator, denominator = (float(value) for value in values)
        error = _relative_error(observed, numerator / denominator)
        errors.append(error)
        if error > relative_tolerance:
            failed_rows.append(index)
    return {
        "formula": f"{observed_field} = {numerator_field} / {denominator_field}",
        "checked": len(errors),
        "failed": len(failed_rows),
        "failed_rows": failed_rows[:50],
        "maximum_relative_error": max(errors, default=None),
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _numeric_profile(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = {
            "count": len(values),
            "minimum": min(values, default=None),
            "q1": _percentile(values, 0.25),
            "median": statistics.median(values) if values else None,
            "q3": _percentile(values, 0.75),
            "maximum": max(values, default=None),
            "non_positive": sum(value <= 0 for value in values),
        }
    return result


def _key_profile(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    counts = Counter(tuple(row.get(field) for field in fields) for row in rows)
    duplicates = {key: count for key, count in counts.items() if count > 1}
    examples = {
        " | ".join(str(value) for value in key): count
        for key, count in list(sorted(duplicates.items(), key=lambda item: str(item[0])))[:25]
    }
    return {
        "fields": list(fields),
        "unique_keys": len(counts),
        "duplicate_groups": len(duplicates),
        "rows_in_duplicate_groups": sum(duplicates.values()),
        "multiplicity": {str(k): v for k, v in sorted(Counter(counts.values()).items())},
        "examples": examples,
    }


def _reason_category(reason: Any) -> str:
    if not reason:
        return "none"
    return str(reason).split(":", 1)[0]


def profile_full_runs(
    rows: Sequence[dict[str, Any]], field_types: dict[str, str] | None = None
) -> dict[str, Any]:
    columns = sorted({field for row in rows for field in row})
    total = len(rows)
    stable = [row for row in rows if row.get("is_stable") is True]
    unstable = [row for row in rows if row.get("is_stable") is False]
    exact_counts = Counter(json.dumps(row, sort_keys=True, default=str) for row in rows)

    nulls = {
        field: {
            "count": sum(row.get(field) is None for row in rows),
            "rate": (sum(row.get(field) is None for row in rows) / total) if total else 0.0,
        }
        for field in columns
    }
    blank_strings = {
        field: {
            "count": sum(
                isinstance(row.get(field), str) and not row.get(field).strip()
                for row in rows
            ),
            "rate": (
                sum(
                    isinstance(row.get(field), str) and not row.get(field).strip()
                    for row in rows
                )
                / total
                if total
                else 0.0
            ),
        }
        for field in columns
        if any(isinstance(row.get(field), str) for row in rows)
    }
    categorical_fields = (
        "task",
        "model_id",
        "architecture",
        "weight_precision",
        "gpu_model",
        "num_gpus",
        "max_num_seqs",
        "seed",
        "num_request_repeats",
        "tensor_parallel",
        "expert_parallel",
        "data_parallel",
    )
    coverage = {
        field: {
            str(key): count
            for key, count in sorted(
                Counter(row.get(field) for row in rows).items(), key=lambda item: str(item[0])
            )
        }
        for field in categorical_fields
        if field in columns
    }
    stability_by_task = {
        task: {
            "stable": sum(row.get("is_stable") is True for row in rows if row.get("task") == task),
            "unstable": sum(row.get("is_stable") is False for row in rows if row.get("task") == task),
        }
        for task in TEXT_LLM_TASKS
    }
    stability_by_gpu = {
        gpu: {
            "stable": sum(row.get("is_stable") is True for row in rows if row.get("gpu_model") == gpu),
            "unstable": sum(row.get("is_stable") is False for row in rows if row.get("gpu_model") == gpu),
        }
        for gpu in sorted({row.get("gpu_model") for row in rows})
    }
    models = sorted({row.get("model_id") for row in rows})
    task_gpu_model_counts = Counter(
        (row.get("task"), row.get("gpu_model"), row.get("model_id")) for row in rows
    )
    cell_sizes = sorted(task_gpu_model_counts.values())
    cross_coverage = {
        "models": len(models),
        "observed_task_gpu_model_cells": len(task_gpu_model_counts),
        "naive_full_cartesian_cells": len(models)
        * len(TEXT_LLM_TASKS)
        * len({row.get("gpu_model") for row in rows}),
        "stable_cells": len(
            {
                (row.get("task"), row.get("gpu_model"), row.get("model_id"))
                for row in stable
            }
        ),
        "rows_per_observed_cell": {
            "minimum": min(cell_sizes) if cell_sizes else None,
            "median": statistics.median(cell_sizes) if cell_sizes else None,
            "maximum": max(cell_sizes) if cell_sizes else None,
        },
    }

    itl_failed = []
    for index, row in enumerate(rows):
        values = [
            row.get("median_itl_ms"),
            row.get("p90_itl_ms"),
            row.get("p95_itl_ms"),
            row.get("p99_itl_ms"),
        ]
        if all(value is not None for value in values) and list(map(float, values)) != sorted(
            map(float, values)
        ):
            itl_failed.append(index)

    return {
        "grain": "one compiled benchmark run/configuration row",
        "scope": {
            "tasks": list(TEXT_LLM_TASKS),
            "text_llm_only": True,
            "raw_timeline_used": False,
        },
        "row_count": total,
        "column_count": len(columns),
        "columns": columns,
        "field_types": field_types or {},
        "field_classification": {field: classify_field(field) for field in columns},
        "nulls": nulls,
        "blank_strings": blank_strings,
        "exact_duplicates": {
            "duplicate_groups": sum(count > 1 for count in exact_counts.values()),
            "rows_in_duplicate_groups": sum(count for count in exact_counts.values() if count > 1),
        },
        "keys": {
            "phase1_visible": _key_profile(rows, PHASE1_KEY_FIELDS),
            "repeat_aware": _key_profile(rows, FULL_KEY_FIELDS),
            "repeat_aware_stable": _key_profile(stable, FULL_KEY_FIELDS),
        },
        "coverage": coverage,
        "cross_coverage": cross_coverage,
        "stability": {
            "stable": len(stable),
            "unstable": len(unstable),
            "stable_rate": len(stable) / total if total else None,
            "by_task": stability_by_task,
            "by_gpu": stability_by_gpu,
            "unstable_reason_categories": dict(
                sorted(Counter(_reason_category(row.get("unstable_reason")) for row in unstable).items())
            ),
            "unstable_reason_exact": dict(
                sorted(Counter(str(row.get("unstable_reason")) for row in unstable).items())
            ),
        },
        "numeric_profile": _numeric_profile(rows),
        "consistency": {
            "energy_power_duration": check_energy_power_identity(rows),
            "power_from_energy_intensity_and_throughput": _product_identity(
                rows,
                "avg_power_watts",
                "energy_per_token_joules",
                "output_throughput_tokens_per_sec",
            ),
            "request_energy_from_actual_output": _product_identity(
                rows,
                "energy_per_request_joules",
                "energy_per_token_joules",
                "avg_output_len",
            ),
            "actual_average_output_length": _ratio_identity(
                rows, "avg_output_len", "total_output_tokens", "completed_requests"
            ),
            "itl_percentile_order": {
                "checked": sum(
                    all(
                        row.get(field) is not None
                        for field in ("median_itl_ms", "p90_itl_ms", "p95_itl_ms", "p99_itl_ms")
                    )
                    for row in rows
                ),
                "failed": len(itl_failed),
                "failed_rows": itl_failed[:50],
            },
        },
        "path_checks": {
            "results_json_paths": sum(
                isinstance(row.get("results_path"), str)
                and row["results_path"].endswith("/results.json")
                for row in rows
            ),
            "prometheus_json_paths": sum(
                isinstance(row.get("prometheus_path"), str)
                and row["prometheus_path"].endswith("/prometheus.json")
                for row in rows
            ),
        },
    }


def _load_compact_rows(compact_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TEXT_LLM_TASKS:
        payload = json.loads((compact_dir / f"{task}.json").read_text(encoding="utf-8"))
        if payload.get("task") != task:
            raise ValueError(f"Unexpected task declaration in {task}.json")
        for configuration in payload.get("configurations", []):
            row = dict(configuration)
            row["task"] = task
            rows.append(row)
    return rows


def _canonical_projection(row: dict[str, Any], fields: Sequence[str]) -> str:
    return json.dumps(
        {field: row.get(field) for field in fields},
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


def compare_with_phase1(
    full_rows: Sequence[dict[str, Any]], compact_dir: Path
) -> dict[str, Any]:
    text_rows = [row for row in full_rows if row.get("task") in TEXT_LLM_TASKS]
    stable_rows = [row for row in text_rows if row.get("is_stable") is True]
    compact_rows = _load_compact_rows(compact_dir)
    public_fields = sorted({field for row in compact_rows for field in row})

    stable_counter = Counter(_canonical_projection(row, public_fields) for row in stable_rows)
    compact_counter = Counter(_canonical_projection(row, public_fields) for row in compact_rows)
    only_full = stable_counter - compact_counter
    only_compact = compact_counter - stable_counter
    full_columns = sorted({field for row in text_rows for field in row})

    return {
        "full_text_llm_rows": len(text_rows),
        "full_stable_text_llm_rows": len(stable_rows),
        "full_unstable_text_llm_rows": len(text_rows) - len(stable_rows),
        "phase1_compact_rows": len(compact_rows),
        "public_field_count": len(public_fields),
        "public_fields": public_fields,
        "full_field_count": len(full_columns),
        "full_only_fields": sorted(set(full_columns) - set(public_fields)),
        "compact_only_fields": sorted(set(public_fields) - set(full_columns)),
        "multiset_difference": sum(only_full.values()) + sum(only_compact.values()),
        "stable_rows_only_in_full_projection": sum(only_full.values()),
        "rows_only_in_phase1_compact": sum(only_compact.values()),
        "candidate_key_duplicates": {
            "full_all_text": _key_profile(text_rows, PHASE1_KEY_FIELDS),
            "full_stable_text": _key_profile(stable_rows, PHASE1_KEY_FIELDS),
            "phase1_compact": _key_profile(compact_rows, PHASE1_KEY_FIELDS),
            "full_repeat_aware_all_text": _key_profile(text_rows, FULL_KEY_FIELDS),
            "full_repeat_aware_stable_text": _key_profile(stable_rows, FULL_KEY_FIELDS),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path("step1/data/runs/llm.parquet"))
    parser.add_argument("--compact-dir", type=Path, default=Path("data/compact/llm"))
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("step1/analysis/full_parquet_audit.json"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("step1/analysis/leaderboard_comparison.json"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet)
    all_rows = table.to_pylist()
    text_rows = [row for row in all_rows if row.get("task") in TEXT_LLM_TASKS]
    field_types = {field.name: str(field.type) for field in table.schema}
    audit = profile_full_runs(text_rows, field_types=field_types)
    audit["source_population"] = {
        "parquet_all_llm_and_mllm_rows": len(all_rows),
        "excluded_mllm_rows": len(all_rows) - len(text_rows),
        "excluded_task_counts": dict(
            sorted(
                Counter(
                    row.get("task")
                    for row in all_rows
                    if row.get("task") not in TEXT_LLM_TASKS
                ).items()
            )
        ),
    }
    comparison = compare_with_phase1(all_rows, args.compact_dir)
    _write_json(args.audit_output, audit)
    _write_json(args.comparison_output, comparison)
    print(
        f"Audited {len(text_rows)} text-LLM rows from {len(all_rows)} parquet rows; "
        f"stable={audit['stability']['stable']}, unstable={audit['stability']['unstable']}"
    )
    print(
        f"Phase 1 stable projection difference={comparison['multiset_difference']}; "
        f"wrote {args.audit_output} and {args.comparison_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
