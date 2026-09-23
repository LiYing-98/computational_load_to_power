#!/usr/bin/env python3
"""Complete the Phase 1.7 benchmark-side planned-workload features."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from step3.scripts.common import RunKey, sha256_bytes, sha256_file, stable_error_message

ROOT = Path(__file__).resolve().parents[2]
SEED = 48105
GPQA_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
BENCHMARK_COMMIT = "c1d6557fbf7c7c495749b0fcf649f71774eabde9"
TASK_SPECS = {
    "gpqa": {"requests": 198, "max_output_tokens": 32768, "dataset_revision": GPQA_REVISION},
    "lm-arena-chat": {"requests": 1024, "max_output_tokens": 4096, "dataset_revision": "72e85b3ddc9c81bf7b659d6b03d4126dfd8fb34a"},
    "sourcegraph-fim": {"requests": 1024, "max_output_tokens": 2048, "dataset_revision": "59178e2aaa628a54a99588fb61ac8d04dd03193a"},
}
ADDITIVE_FIELDS = {
    "planned_request_count",
    "planned_total_input_tokens",
    "planned_sum_input_tokens_squared",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_gpqa_prompt(item: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    choices = [
        item["Incorrect Answer 1"].strip(),
        item["Incorrect Answer 2"].strip(),
        item["Incorrect Answer 3"].strip(),
        item["Correct Answer"].strip(),
    ]
    answer = choices[-1]
    rng.shuffle(choices)
    prompt = f"What is the correct answer to the following question: {item['Question']}\n\nChoices:"
    for letter, choice in zip("ABCD", choices, strict=True):
        prompt += f"\n({letter}) {choice}"
    return prompt, f"({choices.index(answer)}) {answer}"


def aggregate_planned_lengths(lengths: list[int], repeat_count: int) -> dict[str, int | float]:
    if not lengths or repeat_count < 1 or any(int(value) <= 0 for value in lengths):
        raise ValueError("positive prompt lengths and repeat_count are required")
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "planned_request_count": int(len(values) * repeat_count),
        "planned_unique_prompt_count": int(len(set(lengths))),
        "planned_total_input_tokens": int(values.sum() * repeat_count),
        "planned_input_tokens_mean": float(values.mean()),
        "planned_input_tokens_p50": float(np.quantile(values, 0.50, method="linear")),
        "planned_input_tokens_p90": float(np.quantile(values, 0.90, method="linear")),
        "planned_input_tokens_p95": float(np.quantile(values, 0.95, method="linear")),
        "planned_input_tokens_min": int(values.min()),
        "planned_input_tokens_max": int(values.max()),
        "planned_input_tokens_std": float(values.std(ddof=0)),
        "planned_sum_input_tokens_squared": int(np.square(values).sum() * repeat_count),
    }


def merge_with_prior(prior: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    if prior["run_key"].duplicated().any() or updates["run_key"].duplicated().any():
        raise ValueError("run_key must be unique before merge")
    result = prior.set_index("run_key", drop=False).copy()
    update_index = updates.set_index("run_key", drop=False)
    for run_key in result.index:
        if result.at[run_key, "tokenization_status"] == "available":
            continue
        if run_key not in update_index.index:
            continue
        for column, value in update_index.loc[run_key].items():
            if column not in result.columns:
                result[column] = None
            result.at[run_key, column] = value
    return result.reset_index(drop=True)


def reconcile_token_totals(
    planned_total: int | float | None,
    result_total: int | float | None,
    completed: int,
    planned_count: int,
) -> dict[str, Any]:
    if planned_total is None or result_total is None or pd.isna(planned_total) or pd.isna(result_total):
        return {"comparison_scope": "missing_value", "exact_match": None, "absolute_error": None, "relative_error": None}
    error = float(result_total) - float(planned_total)
    complete = int(completed) == int(planned_count)
    return {
        "comparison_scope": "all_requests_completed" if complete else "incomplete_run_diagnostic",
        "exact_match": bool(error == 0) if complete else None,
        "absolute_error": error,
        "relative_error": None if float(planned_total) == 0 else error / float(planned_total),
    }


def explain_reconciliation_record(record: dict[str, Any]) -> dict[str, str | None]:
    """Classify result reconciliation without changing the planned feature value."""

    scope = record.get("comparison_scope")
    exact = record.get("exact_match")
    if scope == "all_requests_completed" and exact is True:
        return {"reconciliation_status": "exact", "reconciliation_reason": None}
    if scope == "all_requests_completed" and exact is False:
        task = record.get("task", "unknown")
        gpu = record.get("gpu_model", "unknown")
        repeats = record.get("num_request_repeats", "unknown")
        return {
            "reconciliation_status": "mismatch_unresolved_historical_equivalence",
            "reconciliation_reason": (
                f"completed {task} result on {gpu} with repeat={repeats} differs from the pinned reconstruction; "
                "pinned code repeats an identical request vector, while the historical result does not persist "
                "the exact request vector, tokenizer revision, or benchmark commit needed to resolve collection-version equivalence"
            ),
        }
    if scope == "incomplete_run_diagnostic":
        return {
            "reconciliation_status": "not_comparable_incomplete_run",
            "reconciliation_reason": "result did not complete the full planned request count",
        }
    return {
        "reconciliation_status": "not_comparable_missing_value",
        "reconciliation_reason": "planned or result input-token total is unavailable",
    }


def run_key_for(row: dict[str, Any]) -> str:
    return RunKey.from_mapping(row).canonical()


def gpqa_csv_path() -> Path:
    path = ROOT / "step3/data/hf_cache/datasets--Idavidrein--gpqa/snapshots" / GPQA_REVISION / "gpqa_diamond.csv"
    if not path.exists():
        raise FileNotFoundError("pinned GPQA diamond CSV was not acquired")
    return path


def build_gpqa_requests() -> list[dict[str, str]]:
    stream = load_dataset(
        "csv",
        data_files=str(gpqa_csv_path()),
        split="train",
        streaming=True,
    ).shuffle(seed=SEED)
    rng = random.Random(SEED)
    requests: list[dict[str, str]] = []
    for item in stream:
        if len(requests) >= TASK_SPECS["gpqa"]["requests"]:
            break
        prompt, completion = build_gpqa_prompt(item, rng)
        requests.append({"prompt": prompt, "completion": completion})
    if len(requests) != TASK_SPECS["gpqa"]["requests"]:
        raise ValueError(f"expected 198 GPQA requests, built {len(requests)}")
    return requests


def load_lmarena_requests() -> list[dict[str, Any]]:
    path = ROOT / "step2/data/planned_requests/lm-arena-chat.json"
    requests = json.loads(path.read_text(encoding="utf-8"))
    if len(requests) != TASK_SPECS["lm-arena-chat"]["requests"]:
        raise ValueError("Phase 1.6 LMArena request count changed")
    return requests


def count_unique_planned_inputs(task: str, requests: list[dict[str, Any]]) -> int:
    if task in {"gpqa", "lm-arena-chat"}:
        return len({sha256_bytes(canonical_bytes(request["prompt"])) for request in requests})
    raise ValueError(f"unsupported task for input identity: {task}")


def local_model_path(model_id: str) -> Path:
    path = ROOT / "step3/data/static_models" / model_id.replace("/", "--")
    if not path.exists():
        raise FileNotFoundError(f"static model directory unavailable: {model_id}")
    return path


def tokenize_planned(task: str, requests: list[dict[str, Any]], tokenizer: Any) -> list[int]:
    if task == "gpqa":
        return [len(tokenizer(request["prompt"]).input_ids) for request in requests]
    if task == "lm-arena-chat":
        return [
            sum(len(tokenizer(content).input_ids) for content in request["prompt"])
            for request in requests
        ]
    raise ValueError(f"incremental reconstruction is not needed for {task}")


def build_updates(prior: pd.DataFrame, revisions: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = prior[prior["tokenization_status"] != "available"].copy()
    requests_by_task = {"gpqa": build_gpqa_requests(), "lm-arena-chat": load_lmarena_requests()}
    request_hashes = {task: sha256_bytes(canonical_bytes(requests)) for task, requests in requests_by_task.items()}
    unique_input_counts = {
        task: count_unique_planned_inputs(task, requests)
        for task, requests in requests_by_task.items()
    }
    vector_root = ROOT / "step3/data/planned_vectors"
    vector_root.mkdir(parents=True, exist_ok=True)
    pair_records: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    for (task, model_id), group in missing.groupby(["task", "model_id"], sort=True):
        model = revisions["models"][model_id]
        base = {"task": task, "model_id": model_id, "revision": model["revision"]}
        if model["access"] == "restricted":
            reason = "restricted_model_static_files: Meta Llama access rejected; no bypass or mirror used"
            for row in group.to_dict(orient="records"):
                update = dict(row)
                update.update({
                    "run_key": run_key_for(row),
                    "tokenization_status": "missing",
                    "tokenizer_revision": None,
                    "tokenizer_revision_requested": model["revision"],
                    "missing_reason": reason,
                })
                updates.append(update)
            pair_records.append({**base, "status": "restricted", "reason": reason})
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_model_path(model_id),
                local_files_only=True,
                trust_remote_code=False,
            )
            lengths = tokenize_planned(task, requests_by_task[task], tokenizer)
            vector_path = vector_root / f"{task}--{model_id.replace('/', '--')}.json"
            vector_path.write_bytes(canonical_bytes(lengths))
            for row in group.to_dict(orient="records"):
                repeats = int(row["num_request_repeats"])
                aggregate = aggregate_planned_lengths(lengths, repeats)
                aggregate["planned_unique_prompt_count"] = unique_input_counts[task]
                update = dict(row)
                update.update(aggregate)
                update.update({
                    "run_key": run_key_for(row),
                    "tokenization_status": "available",
                    "tokenizer_revision": model["revision"],
                    "tokenizer_revision_requested": model["revision"],
                    "missing_reason": None,
                    "request_list_sha256": request_hashes[task],
                    "base_input_lengths_sha256": sha256_bytes(canonical_bytes(lengths)),
                    "planned_output_token_upper_bound": int(aggregate["planned_request_count"] * TASK_SPECS[task]["max_output_tokens"]),
                })
                updates.append(update)
            pair_records.append({
                **base,
                "status": "available",
                "tokenizer_class": tokenizer.__class__.__name__,
                "request_count": len(lengths),
                "vector_sha256": sha256_file(vector_path),
            })
        except Exception as error:
            pair_records.append({**base, "status": "unavailable", "reason": stable_error_message(error)})
    return pd.DataFrame(updates), {
        "request_hashes": request_hashes,
        "unique_input_counts": unique_input_counts,
        "pairs": pair_records,
    }


def build_reconciliation(planned: pd.DataFrame) -> dict[str, Any]:
    manifest = json.loads((ROOT / "step2/analysis/result_metadata_manifest.json").read_text(encoding="utf-8"))
    metadata_by_key = {
        record["run_key"]: json.loads((ROOT / record["local_path"]).read_text(encoding="utf-8"))
        for record in manifest["records"]
    }
    detail: list[dict[str, Any]] = []
    for row in planned.to_dict(orient="records"):
        run_key = row["run_key"]
        metadata = metadata_by_key.get(run_key)
        result_total = metadata.get("total_input_tokens") if metadata else None
        completed = int(metadata.get("completed", 0)) if metadata else 0
        planned_count = int(row["planned_request_count"])
        planned_total = row.get("planned_total_input_tokens")
        comparison = reconcile_token_totals(planned_total, result_total, completed, planned_count)
        record = {
            "run_key": run_key,
            "task": row["task"],
            "model_id": row["model_id"],
            "gpu_model": row["gpu_model"],
            "num_request_repeats": int(row["num_request_repeats"]),
            "tokenization_status": row["tokenization_status"],
            "completed": completed,
            "planned_request_count": planned_count,
            "planned_total_input_tokens": None if pd.isna(planned_total) else int(planned_total),
            "result_total_input_tokens": result_total,
            **comparison,
        }
        record.update(explain_reconciliation_record(record))
        detail.append(record)
    frame = pd.DataFrame(detail)
    complete = frame[frame["comparison_scope"] == "all_requests_completed"]
    exact = complete["exact_match"].dropna()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_run_reconciliation": {
            "runs": int(len(complete)),
            "exact_matches": int(exact.sum()) if len(exact) else 0,
            "mismatches": int((~exact.astype(bool)).sum()) if len(exact) else 0,
            "exact_match_rate": float(exact.mean()) if len(exact) else None,
        },
        "incomplete_run_diagnostics": int((frame["comparison_scope"] == "incomplete_run_diagnostic").sum()),
        "missing_value_runs": int((frame["comparison_scope"] == "missing_value").sum()),
        "mismatch_explanation_counts": {
            str(key): int(value)
            for key, value in frame["reconciliation_status"].value_counts(dropna=False).items()
        },
        "pinned_repeat_semantics_evidence": {
            "benchmark_commit": BENCHMARK_COMMIT,
            "path": "mlenergy/benchmark/llm/workloads.py",
            "sha256": "735827578d5274e6dec58bbb89889e52b4b9a3bfcba2ffd483dc9b0ae282346e",
            "semantics": "when num_request_repeats > 1, pinned WorkloadConfig repeats the already sampled request list by list multiplication",
            "historical_limit": "result headers do not persist the exact request vector, tokenizer revision, or benchmark commit",
        },
        "detail": detail,
    }


def main() -> None:
    prior = pd.read_parquet(ROOT / "step2/analysis/planned_workload_features.parquet")
    prior.insert(0, "run_key", [run_key_for(row) for row in prior.to_dict(orient="records")])
    revisions = json.loads((ROOT / "step3/config/model_revisions.json").read_text(encoding="utf-8"))
    updates, provenance_detail = build_updates(prior, revisions)
    completed = merge_with_prior(prior, updates)
    completed = completed.sort_values("run_key").reset_index(drop=True)
    analysis = ROOT / "step3/analysis"
    reports = ROOT / "step3/reports"
    analysis.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    completed.to_parquet(analysis / "planned_workload_features.parquet", index=False)
    completed.to_csv(analysis / "planned_workload_features.csv", index=False)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_commit": BENCHMARK_COMMIT,
        "gpqa_dataset_revision": GPQA_REVISION,
        "phase16_table_sha256": sha256_file(ROOT / "step2/analysis/planned_workload_features.parquet"),
        "method": "byte/value-preserving reuse of Phase 1.6 available rows plus local pinned GPQA and Gemma reconstruction",
        "rows": int(len(completed)),
        "available_rows": int((completed["tokenization_status"] == "available").sum()),
        "restricted_rows": int(completed["missing_reason"].fillna("").str.contains("Llama access rejected").sum()),
        **provenance_detail,
    }
    (analysis / "planned_workload_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    reconciliation = build_reconciliation(completed)
    (reports / "planned_workload_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(completed)} rows; available={provenance['available_rows']}; "
        f"complete exact={reconciliation['complete_run_reconciliation']['exact_matches']}/"
        f"{reconciliation['complete_run_reconciliation']['runs']}"
    )


if __name__ == "__main__":
    main()
