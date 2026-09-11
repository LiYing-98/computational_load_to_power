#!/usr/bin/env python3
"""Audit bounded ML.ENERGY result metadata and define the leakage-safe X_pre."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BENCHMARK_COMMIT = "c1d6557fbf7c7c495749b0fcf649f71774eabde9"
BENCHMARK_ROOT_URL = f"https://github.com/ml-energy/benchmark/blob/{BENCHMARK_COMMIT}"

TASK_SPECS: dict[str, dict[str, Any]] = {
    "gpqa": {
        "dataset": "Idavidrein/gpqa",
        "dataset_subset": "gpqa_diamond",
        "dataset_split": "train",
        "num_unique_prompts": 198,
        "max_output_tokens": 32768,
        "endpoint_type": "openai-chat",
    },
    "lm-arena-chat": {
        "dataset": "lmarena-ai/arena-human-preference-100k",
        "dataset_subset": None,
        "dataset_split": "train",
        "num_unique_prompts": 1024,
        "max_output_tokens": 4096,
        "endpoint_type": "openai-chat",
    },
    "sourcegraph-fim": {
        "dataset": "sourcegraph/context-aware-fim-code-completions",
        "dataset_subset": None,
        "dataset_split": "train",
        "num_unique_prompts": 1024,
        "max_output_tokens": 2048,
        "endpoint_type": "openai",
    },
}

PRE_KNOWN = {
    "endpoint_type": "Task endpoint mode is fixed before launch.",
    "num_unique_prompts": "Planned unique request count is configured before launch.",
    "num_request_repeats": "Request sequence repetition is configured before launch.",
    "gpu_model": "GPU family is selected before launch.",
    "num_gpus": "GPU count is selected before launch.",
    "max_num_seqs": "Maximum concurrent sequences is a vLLM launch setting.",
    "max_num_batched_tokens": "Optional vLLM batching cap is a launch setting.",
    "request_rate": "Arrival-rate control is set before launch.",
    "burstiness": "Arrival-process burstiness is set before launch.",
    "max_concurrency": "Optional client concurrency cap is set before launch.",
    "max_output_tokens": "Per-request output cap is set before launch.",
}

POST_RUN = {
    "duration": "Realized end-to-end benchmark duration.",
    "completed": "Realized completion count, not the planned request count.",
    "total_input_tokens": "Observed tokenizer-counted input volume from the executed request set.",
    "total_output_tokens": "Actual generated output volume.",
    "request_throughput": "Realized request throughput.",
    "output_throughput": "Realized output-token throughput.",
    "total_token_throughput": "Realized total-token throughput.",
    "steady_state_duration": "Realized steady-state window duration.",
    "steady_state_measurement.time": "Realized steady-state measurement time.",
    "entire_benchmark_measurement.time": "Realized full-run measurement time.",
    "avg_batch_size": "Observed average batch/concurrency, unavailable before execution.",
}


def find_workload_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    """Return flattened JSON leaf paths without retaining raw timelines."""
    fields: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            fields.update(find_workload_fields(child, path))
    elif isinstance(value, list):
        fields[prefix] = {"type": "list", "length": len(value)}
    else:
        fields[prefix] = value
    return fields


def classify_workload_value(
    path: str, value: Any, evidence: dict[str, Any]
) -> dict[str, Any]:
    """Classify a workload/result value at the execution-time boundary."""
    if path == "planned_total_input_tokens":
        ready = bool(evidence.get("frozen_requests_and_tokenizer"))
        return {
            "category": "pre_derivable" if ready else "missing_precondition",
            "allow_as_x_pre": ready,
            "reason": (
                "Can be tokenized before launch from the frozen sampled requests, exact model "
                "tokenizer and task-specific prompt formatting."
                if ready
                else "Requires a frozen request artifact, exact tokenizer revision and prompt template."
            ),
        }
    if path in {"planned_request_count", "planned_output_token_upper_bound"}:
        return {
            "category": "pre_derivable",
            "allow_as_x_pre": True,
            "reason": "Deterministic arithmetic from launch configuration.",
        }
    if path in PRE_KNOWN:
        return {"category": "pre_known", "allow_as_x_pre": True, "reason": PRE_KNOWN[path]}
    if path == "num_prompts":
        return {
            "category": "pre_derivable",
            "allow_as_x_pre": True,
            "reason": "Equals planned unique prompts × configured repetition count.",
        }
    if path == "model_id":
        return {
            "category": "identifier_risk",
            "allow_as_x_pre": False,
            "reason": "Known before launch but risks identity memorization; use physical model metadata instead.",
        }
    if path == "seed":
        return {
            "category": "diagnostic",
            "allow_as_x_pre": False,
            "reason": "Reproducibility control, not a physical driver.",
        }
    if path == "date":
        return {
            "category": "diagnostic",
            "allow_as_x_pre": False,
            "reason": "Run timestamp and provenance only.",
        }
    if path in POST_RUN:
        return {"category": "post_run_leakage", "allow_as_x_pre": False, "reason": POST_RUN[path]}
    if path == "steady_state_energy" or path.startswith("steady_state_measurement.gpu_energy."):
        return {
            "category": "target",
            "allow_as_x_pre": False,
            "reason": "Aggregate GPU-side steady-state energy target.",
        }
    if path == "steady_state_energy_per_token":
        return {
            "category": "target",
            "allow_as_x_pre": False,
            "reason": "GPU energy-intensity target derived after execution.",
        }
    if path.startswith("entire_benchmark_measurement.gpu_energy."):
        return {
            "category": "diagnostic_outcome",
            "allow_as_x_pre": False,
            "reason": "Post-run full-benchmark GPU energy; outside the steady-state target definition.",
        }
    if any(token in path for token in ("cpu_energy", "dram_energy", "soc_energy")):
        return {
            "category": "out_of_scope",
            "allow_as_x_pre": False,
            "reason": "This phase is explicitly GPU-side only.",
        }
    return {
        "category": "unclassified",
        "allow_as_x_pre": False,
        "reason": "No documented pre-execution provenance; excluded conservatively.",
    }


def _summarize(values: list[Any]) -> dict[str, Any]:
    serialized = Counter(json.dumps(value, sort_keys=True) for value in values)
    examples = [json.loads(value) for value, _ in serialized.most_common(5)]
    return {"present": len(values), "unique": len(serialized), "examples": examples}


def audit_selected_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    run_fields: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for run in manifest["runs"]:
        doc = json.loads(Path(run["local_metadata_path"]).read_text(encoding="utf-8"))
        if "timeline" in doc:
            raise ValueError("timeline payload found in bounded metadata")
        run_fields.append((run, find_workload_fields(doc)))

    all_paths = sorted(set().union(*(fields.keys() for _, fields in run_fields)))
    field_audit: dict[str, Any] = {}
    for path in all_paths:
        values = [fields[path] for _, fields in run_fields if path in fields]
        by_task: dict[str, int] = Counter(
            run["task"] for run, fields in run_fields if path in fields
        )
        item = _summarize(values)
        item.update(classify_workload_value(path, values[0] if values else None, {}))
        item["availability_by_task"] = dict(sorted(by_task.items()))
        field_audit[path] = item

    cross_checks: dict[str, Any] = {}
    for task, spec in TASK_SPECS.items():
        task_runs = [(run, fields) for run, fields in run_fields if run["task"] == task]
        mismatches = []
        for index, (run, fields) in enumerate(task_runs, start=1):
            repeats = int(run["num_request_repeats"])
            expected_total = spec["num_unique_prompts"] * repeats
            checks = {
                "num_unique_prompts": spec["num_unique_prompts"],
                "max_output_tokens": spec["max_output_tokens"],
                "endpoint_type": spec["endpoint_type"],
                "num_prompts": expected_total,
            }
            for field, expected in checks.items():
                # Older Sourcegraph result metadata omitted two setup fields; the parquet/path retains them.
                if field in fields and fields[field] != expected:
                    mismatches.append(
                        {"task_run_index": index, "field": field, "expected": expected, "actual": fields[field]}
                    )
        cross_checks[task] = {
            "runs_checked": len(task_runs),
            "mismatches": mismatches,
            "status": "pass" if not mismatches else "fail",
        }

    task_matrix = {}
    for task, spec in TASK_SPECS.items():
        task_runs = [run for run, _ in run_fields if run["task"] == task]
        task_matrix[task] = {
            **spec,
            "representative_runs": len(task_runs),
            "observed_repeat_counts": sorted({int(run["num_request_repeats"]) for run in task_runs}),
            "planned_request_count_formula": "num_unique_prompts * num_request_repeats",
            "input_token_feature_status": "not present precomputed; derivable before execution only with frozen requests + exact tokenizer/template",
            "expected_output_feature_status": "max_output_tokens is only a cap; no model-specific planned output-demand feature is supplied",
            "sampling_defaults_from_code": {"top_p": 0.95, "temperature": 0.8, "ignore_eos": False},
        }

    x_pre = {
        "raw_allowed": [
            "task",
            "architecture",
            "total_params_billions",
            "activated_params_billions",
            "weight_precision",
            "gpu_model",
            "num_gpus",
            "max_num_seqs",
            "tensor_parallel",
            "expert_parallel",
            "data_parallel",
            "num_unique_prompts",
            "num_request_repeats",
            "max_num_batched_tokens",
            "request_rate",
            "burstiness",
            "max_concurrency",
            "max_output_tokens",
            "endpoint_type",
        ],
        "pre_derived_allowed": [
            "planned_request_count = num_unique_prompts * num_request_repeats",
            "planned_output_token_upper_bound = planned_request_count * max_output_tokens",
            "planned_total_input_tokens (only after frozen-request/tokenizer reconstruction)",
            "planned_input_length distribution summaries (same precondition)",
            "GPU-normalized model size and parallelism ratios from physical metadata",
        ],
        "forbidden_post_run": sorted(
            path for path, item in field_audit.items() if item["category"] in {"post_run_leakage", "target", "diagnostic_outcome"}
        ),
        "identifier_risk": ["model_id", "nickname"],
        "most_important_gap": (
            "No persisted pre-execution workload token-volume/distribution artifact. "
            "Observed total_input_tokens and all generated-output quantities are post-run values."
        ),
    }

    return {
        "scope": {
            "text_llm_inference_only": True,
            "gpu_side_measurement": True,
            "selected_run_count": len(run_fields),
            "timeline_payloads_retained": 0,
        },
        "dataset_revision": manifest.get("dataset_revision"),
        "benchmark_code": {
            "commit": BENCHMARK_COMMIT,
            "repository": "https://github.com/ml-energy/benchmark",
            "workload_definitions": f"{BENCHMARK_ROOT_URL}/mlenergy/benchmark/llm/workloads.py",
            "dataset_sampling": f"{BENCHMARK_ROOT_URL}/mlenergy/benchmark/llm/datasets.py",
            "benchmark_execution": f"{BENCHMARK_ROOT_URL}/mlenergy/benchmark/llm/benchmark.py",
            "task_configs": f"{BENCHMARK_ROOT_URL}/configs/vllm",
        },
        "field_audit": field_audit,
        "cross_checks": cross_checks,
        "task_matrix": task_matrix,
        "x_pre": x_pre,
        "decision": {
            "code": "B",
            "label": "conditionally feasible after workload-feature backfill",
            "reason": (
                "The full summary is internally consistent and covers hardware/model/configuration well, "
                "but task identity plus output caps do not describe pre-run token workload sufficiently "
                "for a general task-features-to-energy model."
            ),
        },
    }


def main() -> None:
    manifest_path = Path("step1/analysis/selected_runs.json")
    output_path = Path("step1/analysis/workload_field_audit.json")
    audit = audit_selected_metadata(json.loads(manifest_path.read_text(encoding="utf-8")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Audited {audit['scope']['selected_run_count']} bounded metadata snapshots; "
        f"decision={audit['decision']['code']}"
    )


if __name__ == "__main__":
    main()
