#!/usr/bin/env python3
"""Select a small, deterministic coverage set from ML.ENERGY text-LLM runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step1.scripts.audit_full_parquet import FULL_KEY_FIELDS, TEXT_LLM_TASKS


FORBIDDEN_PATH_TERMS = (
    "prometheus",
    "timeline",
    "training",
    "diffusion",
    "image-chat",
    "video-chat",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
)


def allowed_metadata_path(path: str) -> bool:
    """Return true only for in-scope LLM run result JSON paths."""
    normalized = str(path).replace("\\", "/").lower()
    parts = PurePosixPath(normalized).parts
    return (
        normalized.startswith("llm/")
        and parts[-1:] == ("results.json",)
        and ".." not in parts
        and not any(term in normalized for term in FORBIDDEN_PATH_TERMS)
    )


def _scale(total_params_billions: Any) -> str:
    value = float(total_params_billions)
    if value <= 20:
        return "small"
    if value <= 100:
        return "medium"
    return "large"


def _batch_class(max_num_seqs: Any) -> str:
    value = int(max_num_seqs)
    if value <= 32:
        return "low"
    if value <= 256:
        return "medium"
    return "high"


def _facets(row: dict[str, Any]) -> set[str]:
    card_class = "single" if int(row["num_gpus"]) == 1 else "multi"
    return {
        f"task:{row['task']}",
        f"gpu:{row['gpu_model']}",
        f"task_gpu:{row['task']}|{row['gpu_model']}",
        f"architecture:{row['architecture']}",
        f"scale:{_scale(row['total_params_billions'])}",
        f"cards:{card_class}",
        f"batch:{_batch_class(row['max_num_seqs'])}",
        f"precision:{row.get('weight_precision', 'unknown')}",
        f"parallel:tp{row['tensor_parallel']}|ep{row['expert_parallel']}|dp{row['data_parallel']}",
        f"repeats:{'single' if int(row['num_request_repeats']) == 1 else 'multiple'}",
    }


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(str(row.get(field, "")) for field in FULL_KEY_FIELDS) + (
        str(row.get("results_path", "")),
    )


def select_representative_runs(
    rows: Iterable[dict[str, Any]], limit: int = 12, minimum: int = 9
) -> list[dict[str, Any]]:
    """Greedily cover task/hardware/model/config facets, then fill to minimum."""
    if not 1 <= limit <= 12:
        raise ValueError("limit must be between 1 and 12")
    minimum = min(minimum, limit)
    candidates = [
        dict(row)
        for row in rows
        if row.get("task") in TEXT_LLM_TASKS
        and bool(row.get("is_stable"))
        and allowed_metadata_path(str(row.get("results_path", "")))
    ]
    candidates.sort(key=_sort_key)
    if not candidates:
        return []

    all_facets = set().union(*(_facets(row) for row in candidates))
    covered: set[str] = set()
    selected: list[dict[str, Any]] = []
    remaining = candidates[:]

    while remaining and len(selected) < limit:
        scored = []
        for row in remaining:
            gained = _facets(row) - covered
            # Task×GPU coverage is the most important bounded-sample constraint.
            weighted_gain = len(gained) + 3 * sum(
                facet.startswith("task_gpu:") for facet in gained
            )
            scored.append((weighted_gain, len(gained), _sort_key(row), row, gained))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        _, _, _, chosen, gained = scored[0]
        if not gained and len(selected) >= minimum:
            break
        remaining.remove(chosen)
        covered.update(_facets(chosen))
        enriched = {
            key: chosen.get(key)
            for key in (
                "task",
                "model_id",
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
                "seed",
                "num_request_repeats",
                "is_stable",
                "results_path",
            )
        }
        enriched["model_scale"] = _scale(chosen["total_params_billions"])
        enriched["card_count_class"] = (
            "single" if int(chosen["num_gpus"]) == 1 else "multi"
        )
        enriched["batch_class"] = _batch_class(chosen["max_num_seqs"])
        enriched["selection_reasons"] = sorted(gained)
        selected.append(enriched)

        if covered == all_facets and len(selected) >= minimum:
            break
    return selected


def build_manifest(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("task") in TEXT_LLM_TASKS
        and bool(row.get("is_stable"))
        and allowed_metadata_path(str(row.get("results_path", "")))
    ]
    available_facets = sorted(set().union(*(_facets(row) for row in eligible)))
    covered_facets = sorted(
        set().union(*(_facets(row) for row in selected)) if selected else set()
    )
    return {
        "selection_policy": "stable text-LLM runs; greedy task×GPU-weighted facet coverage; deterministic key tie-break",
        "download_policy": "approved results.json paths only; retain only prefix before embedded timeline; no prometheus/model weights",
        "eligible_rows": len(eligible),
        "selected_count": len(selected),
        "available_facets": available_facets,
        "covered_facets": covered_facets,
        "uncovered_available_facets": sorted(set(available_facets) - set(covered_facets)),
        "runs": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="step1/data/runs/llm.parquet")
    parser.add_argument("--output", default="step1/analysis/selected_runs.json")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    rows = pq.read_table(args.parquet).to_pylist()
    selected = select_representative_runs(rows, limit=args.limit)
    manifest = build_manifest(rows, selected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Selected {len(selected)} stable text-LLM runs; "
        f"uncovered available facets={len(manifest['uncovered_available_facets'])}"
    )


if __name__ == "__main__":
    main()
