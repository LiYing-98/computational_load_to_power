#!/usr/bin/env python3
"""Build a reviewed Data-app snapshot from Phase 1.5 audit JSON outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("step1")


def read(name: str):
    return json.loads((ROOT / "analysis" / name).read_text(encoding="utf-8"))


def source(label: str, definitions: list[dict], filters: list[str]) -> dict:
    return {
        "label": label,
        "tables": [],
        "filters": filters,
        "metricDefinitions": definitions,
    }


def main() -> None:
    full = read("full_parquet_audit.json")
    comparison = read("leaderboard_comparison.json")
    selected = read("selected_runs.json")
    workload = read("workload_field_audit.json")

    stability_rows = []
    for task, values in full["stability"]["by_task"].items():
        total = values["stable"] + values["unstable"]
        stability_rows.append(
            {
                "task": task,
                "stable": values["stable"],
                "unstable": values["unstable"],
                "total": total,
                "stableRate": values["stable"] / total,
            }
        )

    snapshot = {
        "surface": "report",
        "title": "现有 ML.ENERGY 数据可靠，但正式建模前还缺事前工作量特征",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "reviewed",
        "buildStatus": "complete",
        "filters": [],
        "queries": {
            "headline": {
                "rows": [
                    {
                        "textLlmRows": full["row_count"],
                        "stableRows": full["stability"]["stable"],
                        "unstableRows": full["stability"]["unstable"],
                        "models": full["cross_coverage"]["models"],
                        "selectedRuns": selected["selected_count"],
                        "decision": workload["decision"]["code"],
                        "phase1Difference": comparison["multiset_difference"],
                    }
                ],
                "source": source(
                    "Reviewed Phase 1.5 headline checks",
                    [
                        {
                            "label": "Text LLM run rows",
                            "definition": "Rows after limiting runs/llm.parquet to GPQA, LM Arena Chat and Sourcegraph FIM.",
                            "componentIds": ["report-executive-summary", "report-scope", "report-metric-rows"],
                        },
                        {
                            "label": "Stable rows",
                            "definition": "Text LLM rows with is_stable=true.",
                            "componentIds": ["report-executive-summary", "report-stability", "report-metric-stable"],
                        },
                    ],
                    ["Text LLM inference tasks only", "GPU-side measurements"],
                ),
            },
            "stability_by_task": {
                "rows": stability_rows,
                "source": source(
                    "Full gated parquet stability profile",
                    [
                        {
                            "label": "Stable rate",
                            "definition": "Stable rows divided by all text LLM rows within task.",
                            "formula": "stable / (stable + unstable)",
                            "componentIds": ["report-stability"],
                        }
                    ],
                    ["Tasks: gpqa, lm-arena-chat, sourcegraph-fim"],
                ),
            },
            "unstable_reasons": {
                "rows": [
                    {"reason": reason, "runs": count}
                    for reason, count in full["stability"]["unstable_reason_categories"].items()
                ],
                "source": source(
                    "Post-run stability diagnostics",
                    [
                        {
                            "label": "Unstable runs",
                            "definition": "Rows with is_stable=false grouped by normalized unstable reason prefix.",
                            "componentIds": ["report-stability"],
                        }
                    ],
                    ["Unstable text LLM rows only"],
                ),
            },
            "selected_runs": {
                "rows": [
                    {
                        key: run[key]
                        for key in (
                            "task",
                            "gpu_model",
                            "architecture",
                            "model_scale",
                            "num_gpus",
                            "max_num_seqs",
                            "tensor_parallel",
                            "expert_parallel",
                            "data_parallel",
                            "weight_precision",
                        )
                    }
                    for run in selected["runs"]
                ],
                "source": source(
                    "Deterministic representative run selection",
                    [
                        {
                            "label": "Representative runs",
                            "definition": "Stable text LLM runs selected coverage-first across task, GPU, architecture, scale, card count, batch, precision and parallelism facets.",
                            "componentIds": ["report-selected-runs"],
                        }
                    ],
                    ["6–12 files", "results metadata prefix only", "no timeline or Prometheus retained"],
                ),
            },
            "workload_tasks": {
                "rows": [
                    {
                        "task": task,
                        "dataset": spec["dataset"],
                        "numUniquePrompts": spec["num_unique_prompts"],
                        "maxOutputTokens": spec["max_output_tokens"],
                        "endpointType": spec["endpoint_type"],
                        "inputFeatureStatus": spec["input_token_feature_status"],
                    }
                    for task, spec in workload["task_matrix"].items()
                ],
                "source": source(
                    "Official benchmark task definitions and bounded metadata",
                    [
                        {
                            "label": "Unique prompts",
                            "definition": "Configured unique request count before request-sequence repetition.",
                            "componentIds": ["report-workload"],
                        },
                        {
                            "label": "Maximum output tokens",
                            "definition": "Configured per-request generation cap; not actual or expected generated length.",
                            "componentIds": ["report-workload", "report-gap"],
                        },
                    ],
                    ["Benchmark commit c1d6557fbf7c7c495749b0fcf649f71774eabde9"],
                ),
            },
            "consistency_checks": {
                "rows": [
                    {
                        "check": name,
                        "checked": result["checked"],
                        "failed": result["failed"],
                        "maxRelativeError": result.get("maximum_relative_error"),
                    }
                    for name, result in full["consistency"].items()
                ],
                "source": source(
                    "Independent arithmetic checks on all text LLM rows",
                    [
                        {
                            "label": "Consistency failures",
                            "definition": "Rows failing an independently recomputed published identity.",
                            "componentIds": ["report-quality"],
                        }
                    ],
                    ["All 694 text LLM rows"],
                ),
            },
        },
    }
    output = ROOT / "analysis" / "report_snapshot.json"
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote reviewed snapshot to {output}")


if __name__ == "__main__":
    main()
