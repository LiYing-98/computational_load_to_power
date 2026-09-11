#!/usr/bin/env python3
"""Run final analytical, scope, security and artifact checks for Phase 1.5."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("step1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def git_visible_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def main() -> None:
    checks: list[dict[str, Any]] = []
    full = read_json(ROOT / "analysis" / "full_parquet_audit.json")
    comparison = read_json(ROOT / "analysis" / "leaderboard_comparison.json")
    selected = read_json(ROOT / "analysis" / "selected_runs.json")
    workload = read_json(ROOT / "analysis" / "workload_field_audit.json")

    record(checks, "text_llm_rows", full["row_count"] == 694, full["row_count"])
    record(
        checks,
        "stable_unstable_partition",
        full["stability"]["stable"] == 565
        and full["stability"]["unstable"] == 129,
        full["stability"],
    )
    failed_identities = {
        name: result["failed"] for name, result in full["consistency"].items()
    }
    record(
        checks,
        "all_consistency_identities",
        all(value == 0 for value in failed_identities.values()),
        failed_identities,
    )
    record(
        checks,
        "phase1_reconciliation",
        comparison["multiset_difference"] == 0,
        comparison,
    )
    record(
        checks,
        "repeat_aware_key_unique",
        full["keys"]["repeat_aware"]["duplicate_groups"] == 0,
        full["keys"]["repeat_aware"],
    )

    metadata_files = sorted((ROOT / "data" / "selected_runs_raw").glob("run_*/metadata.json"))
    metadata_docs = [read_json(path) for path in metadata_files]
    bounded_ok = (
        6 <= len(metadata_files) <= 12
        and len(metadata_files) == selected["selected_count"]
        and all(path.stat().st_size < 100_000 for path in metadata_files)
        and all("timeline" not in doc for doc in metadata_docs)
    )
    record(
        checks,
        "bounded_metadata_only",
        bounded_ok,
        {
            "files": len(metadata_files),
            "bytes": sum(path.stat().st_size for path in metadata_files),
            "timeline_keys": sum("timeline" in doc for doc in metadata_docs),
        },
    )
    retained_names = [path.name.lower() for path in (ROOT / "data").rglob("*") if path.is_file()]
    forbidden_retained = [
        name
        for name in retained_names
        if name in {"prometheus.json", "results.json"}
        or name.endswith((".safetensors", ".pt", ".pth", ".bin"))
    ]
    record(checks, "no_forbidden_retained_payload", not forbidden_retained, forbidden_retained)
    record(
        checks,
        "workload_source_cross_checks",
        all(item["status"] == "pass" for item in workload["cross_checks"].values()),
        workload["cross_checks"],
    )
    record(
        checks,
        "decision_is_B",
        workload["decision"]["code"] == "B",
        workload["decision"],
    )

    notebook = read_json(ROOT / "notebooks" / "phase1_5_complete_audit.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    notebook_errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    record(
        checks,
        "notebook_executed_without_errors",
        bool(code_cells)
        and all(cell.get("execution_count") is not None for cell in code_cells)
        and not notebook_errors,
        {"code_cells": len(code_cells), "errors": len(notebook_errors)},
    )

    build = read_json(ROOT / "report_app" / "dist" / "data-app-build.json")
    report_html = ROOT / "report_app" / "dist" / build["html"]["path"]
    report_snapshot = ROOT / "report_app" / "dist" / build["snapshot"]["path"]
    report_source = read_json(ROOT / "report_app" / "src" / "data.json")
    report_ok = (
        report_html.exists()
        and report_snapshot.exists()
        and sha256(report_html) == build["html"]["sha256"]
        and sha256(report_snapshot) == build["snapshot"]["sha256"]
        and report_source.get("buildStatus") == "complete"
    )
    record(
        checks,
        "report_build_integrity",
        report_ok,
        {
            "html_bytes": report_html.stat().st_size if report_html.exists() else None,
            "snapshot_bytes": report_snapshot.stat().st_size if report_snapshot.exists() else None,
            "buildStatus": report_source.get("buildStatus"),
        },
    )

    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "step1/data/runs/llm.parquet",
            "step1/data/selected_runs_raw/run_01/metadata.json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    ignored_paths = ignored.stdout.splitlines()
    record(checks, "gated_files_gitignored", len(ignored_paths) == 2, ignored_paths)

    token_pattern = re.compile("h" + "f_[A-Za-z0-9]{25,}")
    secret_matches: list[str] = []
    for path in git_visible_files():
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if token_pattern.search(text):
            secret_matches.append(path.as_posix())
    record(checks, "no_huggingface_token_shaped_secret", not secret_matches, secret_matches)

    no_model_code = True
    model_terms = re.compile(r"\b(xgboost|randomforest|ridge\s*\(|neuralnetwork|model\.fit\s*\()", re.I)
    model_hits = []
    training_scope = [
        path
        for path in (ROOT / "scripts").glob("*.py")
        if path.name != Path(__file__).name
    ] + [ROOT / "notebooks" / "phase1_5_complete_audit.ipynb"]
    for path in training_scope:
        text = path.read_text(encoding="utf-8")
        if model_terms.search(text):
            no_model_code = False
            model_hits.append(path.as_posix())
    record(checks, "no_formal_model_training", no_model_code, model_hits)

    receipt = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(item["passed"] for item in checks),
        "checks": checks,
        "visual_verification": {
            "status": "not_run",
            "reason": "Local browser automation inventory failed twice; localhost HTTP and build integrity were verified instead.",
        },
    }
    output = ROOT / "reports" / "verification_receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall_pass": receipt["overall_pass"], "checks": len(checks)}, ensure_ascii=False))
    if not receipt["overall_pass"]:
        for item in checks:
            if not item["passed"]:
                print(f"FAILED: {item['name']}: {item['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
