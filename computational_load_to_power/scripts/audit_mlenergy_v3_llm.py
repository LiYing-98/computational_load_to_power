#!/usr/bin/env python3
"""Audit the public compact ML.ENERGY V3 text-LLM inference snapshot.

This script intentionally reads only the three task-level JSON summaries used by
the official V3 leaderboard. It never calls raw timeline or per-request APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TASKS = ("gpqa", "lm-arena-chat", "sourcegraph-fim")

CANDIDATE_KEY_FIELDS = (
    "task",
    "model_id",
    "gpu_model",
    "num_gpus",
    "max_num_seqs",
    "tensor_parallel",
    "expert_parallel",
    "data_parallel",
)

CATEGORICAL_FIELDS = (
    "task",
    "model_id",
    "architecture",
    "weight_precision",
    "gpu_model",
    "num_gpus",
    "max_num_seqs",
    "tensor_parallel",
    "expert_parallel",
    "data_parallel",
)

NUMERIC_FIELDS = (
    "total_params_billions",
    "activated_params_billions",
    "avg_batch_size",
    "avg_output_len",
    "avg_power_watts",
    "energy_per_request_joules",
    "energy_per_token_joules",
    "median_itl_ms",
    "output_throughput_tokens_per_sec",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
)


def load_compact_records(data_dir: Path) -> list[dict[str, Any]]:
    """Load exactly the three text-only LLM inference task summaries."""
    records: list[dict[str, Any]] = []
    for expected_task in EXPECTED_TASKS:
        path = data_dir / f"{expected_task}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task") != expected_task:
            raise ValueError(
                f"{path} declares task={payload.get('task')!r}, "
                f"expected {expected_task!r}"
            )
        configurations = payload.get("configurations")
        if not isinstance(configurations, list):
            raise ValueError(f"{path} has no configurations list")
        for configuration in configurations:
            if not isinstance(configuration, dict):
                raise ValueError(f"{path} contains a non-object configuration")
            row = dict(configuration)
            row["task"] = expected_task
            records.append(row)
    return records


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _jsonable_key(key: tuple[Any, ...]) -> str:
    return " | ".join(str(value) for value in key)


def profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return structural, completeness, uniqueness, and coverage statistics."""
    columns = sorted({column for row in records for column in row})
    total = len(records)

    nulls: dict[str, dict[str, float | int]] = {}
    for column in columns:
        count = sum(row.get(column) is None for row in records)
        nulls[column] = {"count": count, "rate": _rate(count, total)}

    exact_counts = Counter(
        json.dumps(row, sort_keys=True, ensure_ascii=False) for row in records
    )
    candidate_counts = Counter(
        tuple(row.get(field) for field in CANDIDATE_KEY_FIELDS) for row in records
    )
    duplicate_candidate_groups = {
        _jsonable_key(key): count
        for key, count in candidate_counts.items()
        if count > 1
    }

    coverage = {
        field: {
            str(key): value
            for key, value in sorted(
                Counter(row.get(field) for row in records).items(),
                key=lambda item: str(item[0]),
            )
        }
        for field in CATEGORICAL_FIELDS
    }

    return {
        "row_count": total,
        "column_count": len(columns),
        "columns": columns,
        "nulls": nulls,
        "exact_duplicates": {
            "duplicate_groups": sum(count > 1 for count in exact_counts.values()),
            "rows_in_duplicate_groups": sum(
                count for count in exact_counts.values() if count > 1
            ),
        },
        "candidate_key": {
            "fields": list(CANDIDATE_KEY_FIELDS),
            "unique_keys": len(candidate_counts),
            "duplicate_groups": len(duplicate_candidate_groups),
            "rows_in_duplicate_groups": sum(duplicate_candidate_groups.values()),
            "multiplicity": dict(
                sorted(Counter(candidate_counts.values()).items())
            ),
            "duplicate_group_examples": dict(
                list(sorted(duplicate_candidate_groups.items()))[:20]
            ),
        },
        "coverage": coverage,
    }


def _relative_error(observed: float, expected: float) -> float:
    denominator = max(abs(observed), abs(expected), 1e-12)
    return abs(observed - expected) / denominator


def _identity_check(
    records: list[dict[str, Any]],
    observed_field: str,
    expected_fields: tuple[str, str],
    relative_tolerance: float,
) -> dict[str, Any]:
    checked = 0
    failed_rows: list[int] = []
    errors: list[float] = []
    left, right = expected_fields
    for index, row in enumerate(records):
        values = (row.get(observed_field), row.get(left), row.get(right))
        if any(value is None for value in values):
            continue
        observed, first, second = (float(value) for value in values)
        error = _relative_error(observed, first * second)
        checked += 1
        errors.append(error)
        if error > relative_tolerance:
            failed_rows.append(index)
    return {
        "formula": f"{observed_field} = {left} * {right}",
        "checked": checked,
        "failed": len(failed_rows),
        "failed_rows": failed_rows[:50],
        "max_relative_error": max(errors, default=None),
        "median_relative_error": statistics.median(errors) if errors else None,
        "relative_tolerance": relative_tolerance,
    }


def check_consistency(
    records: list[dict[str, Any]], relative_tolerance: float = 1e-9
) -> dict[str, Any]:
    """Check algebraic and ordered relationships exposed by compact fields."""
    power_identity = _identity_check(
        records,
        "avg_power_watts",
        ("energy_per_token_joules", "output_throughput_tokens_per_sec"),
        relative_tolerance,
    )
    request_energy_identity = _identity_check(
        records,
        "energy_per_request_joules",
        ("energy_per_token_joules", "avg_output_len"),
        relative_tolerance,
    )

    itl_failed: list[int] = []
    itl_checked = 0
    for index, row in enumerate(records):
        values = [
            row.get("median_itl_ms"),
            row.get("p90_itl_ms"),
            row.get("p95_itl_ms"),
            row.get("p99_itl_ms"),
        ]
        if any(value is None for value in values):
            continue
        itl_checked += 1
        numeric = [float(value) for value in values]
        if numeric != sorted(numeric):
            itl_failed.append(index)

    batch_ratios: list[float] = []
    batch_failed_rows: list[int] = []
    for index, row in enumerate(records):
        average = row.get("avg_batch_size")
        maximum = row.get("max_num_seqs")
        if average is None or maximum in (None, 0):
            continue
        ratio = float(average) / float(maximum)
        batch_ratios.append(ratio)
        if ratio < 0.85:
            batch_failed_rows.append(index)

    return {
        "power_identity": power_identity,
        "request_energy_identity": request_energy_identity,
        "itl_percentile_order": {
            "formula": "median_itl_ms <= p90_itl_ms <= p95_itl_ms <= p99_itl_ms",
            "checked": itl_checked,
            "failed": len(itl_failed),
            "failed_rows": itl_failed[:50],
        },
        "batch_utilization": {
            "formula": "avg_batch_size / max_num_seqs",
            "checked": len(batch_ratios),
            "below_official_stability_threshold": len(batch_failed_rows),
            "failed_rows": batch_failed_rows[:50],
            "minimum": min(batch_ratios, default=None),
            "median": statistics.median(batch_ratios) if batch_ratios else None,
            "maximum": max(batch_ratios, default=None),
        },
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


def numeric_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in records if row.get(field) is not None]
        q1 = _percentile(values, 0.25)
        q3 = _percentile(values, 0.75)
        iqr = (q3 - q1) if q1 is not None and q3 is not None else None
        lower = q1 - 1.5 * iqr if iqr is not None else None
        upper = q3 + 1.5 * iqr if iqr is not None else None
        result[field] = {
            "count": len(values),
            "minimum": min(values, default=None),
            "q1": q1,
            "median": statistics.median(values) if values else None,
            "q3": q3,
            "maximum": max(values, default=None),
            "non_positive": sum(value <= 0 for value in values),
            "iqr_outlier_count": sum(
                value < lower or value > upper for value in values
            )
            if lower is not None and upper is not None
            else 0,
        }
    return result


def _cross_tab(
    records: list[dict[str, Any]], row_field: str, column_field: str
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        result[str(row.get(row_field))][str(row.get(column_field))] += 1
    return {
        row_key: dict(sorted(columns.items()))
        for row_key, columns in sorted(result.items())
    }


def coverage_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    model_counts = Counter(row["model_id"] for row in records)
    task_models: dict[str, set[str]] = defaultdict(set)
    model_tasks: dict[str, set[str]] = defaultdict(set)
    model_gpus: dict[str, set[str]] = defaultdict(set)
    for row in records:
        task_models[row["task"]].add(row["model_id"])
        model_tasks[row["model_id"]].add(row["task"])
        model_gpus[row["model_id"]].add(row["gpu_model"])

    comparable_without_gpu: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in records:
        key = (
            row["task"],
            row["model_id"],
            row["num_gpus"],
            row["max_num_seqs"],
            row["tensor_parallel"],
            row["expert_parallel"],
            row["data_parallel"],
        )
        comparable_without_gpu[key].add(str(row["gpu_model"]))

    parallel_relation = Counter()
    for row in records:
        num_gpus = int(row["num_gpus"])
        tp = int(row["tensor_parallel"])
        dp = int(row["data_parallel"])
        ep = int(row["expert_parallel"])
        parallel_relation["tp_times_dp_equals_num_gpus"] += tp * dp == num_gpus
        parallel_relation["tp_divides_num_gpus"] += num_gpus % tp == 0
        parallel_relation["ep_divides_num_gpus"] += num_gpus % ep == 0

    return {
        "unique_models": len(model_counts),
        "models_per_task": {
            task: len(models) for task, models in sorted(task_models.items())
        },
        "tasks_per_model": {
            model: len(tasks) for model, tasks in sorted(model_tasks.items())
        },
        "gpus_per_model": {
            model: sorted(gpus) for model, gpus in sorted(model_gpus.items())
        },
        "configs_per_model": {
            "minimum": min(model_counts.values(), default=0),
            "median": statistics.median(model_counts.values()) if model_counts else 0,
            "maximum": max(model_counts.values(), default=0),
            "counts": dict(sorted(model_counts.items())),
        },
        "cross_tabs": {
            "task_by_gpu": _cross_tab(records, "task", "gpu_model"),
            "task_by_precision": _cross_tab(records, "task", "weight_precision"),
            "architecture_by_precision": _cross_tab(
                records, "architecture", "weight_precision"
            ),
            "num_gpus_by_gpu": _cross_tab(records, "num_gpus", "gpu_model"),
        },
        "gpu_comparable_configuration_keys": {
            "paired_h100_b200_keys": sum(
                {"H100", "B200"}.issubset(gpus)
                for gpus in comparable_without_gpu.values()
            ),
            "all_configuration_keys_ignoring_gpu": len(comparable_without_gpu),
        },
        "parallelism_sanity": dict(parallel_relation),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_audit(data_dir: Path) -> dict[str, Any]:
    records = load_compact_records(data_dir)
    source_files = {
        f"{task}.json": {
            "bytes": (data_dir / f"{task}.json").stat().st_size,
            "sha256": sha256(data_dir / f"{task}.json"),
        }
        for task in EXPECTED_TASKS
    }
    return {
        "scope": {
            "domain": "text-only LLM inference",
            "measurement_boundary": "aggregate GPU-side steady-state metrics",
            "raw_timeline_downloaded": False,
            "source_snapshot_last_updated": "2026-02-16",
            "source_files": source_files,
        },
        "profile": profile_records(records),
        "numeric_profile": numeric_profile(records),
        "coverage_analysis": coverage_analysis(records),
        "consistency": check_consistency(records),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/compact/llm")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/audit_results.json")
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    audit = build_audit(args.data_dir)
    write_json(args.output, audit)
    print(
        f"Audited {audit['profile']['row_count']} records; "
        f"wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
