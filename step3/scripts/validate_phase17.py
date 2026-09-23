#!/usr/bin/env python3
"""Validate Phase 1.7 scope, data quality, leakage boundaries, and deliverables."""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from step3.scripts.common import EXPECTED_RUNS, EXPECTED_STABLE_RUNS, TEXT_LLM_TASKS, WEIGHT_SUFFIXES


ROOT = Path(__file__).resolve().parents[2]
EXANTE_CONTRACT_PATH = Path("step3/config/exante_feature_contract.json")
SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
POST_RUN_X_MARKERS = (
    "total_output_tokens",
    "completed_requests",
    "avg_output_len",
    "output_throughput",
    "request_throughput",
    "_itl_",
    "avg_batch_size",
    "is_stable",
    "unstable_reason",
    "steady_state_duration",
    "steady_state_energy",
    "energy_per_token",
    "energy_per_request",
    "avg_power",
    "latency",
    "duration",
    "actual_batch",
    "utilization",
    "clock",
    "temperature",
    "ttft",
)
X_ROLES = {"X_workload", "X_model", "X_hardware", "X_deployment", "X_protocol"}
TARGETS = (
    "y_gpu_steady_state_energy_joules",
    "y_gpu_steady_state_duration_seconds",
    "y_gpu_avg_power_watts",
    "y_gpu_energy_per_token_joules",
    "y_gpu_energy_per_request_joules",
)


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() == target or _contains_key(child, target) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


def _load_exante_contract(project_root: Path = ROOT) -> dict[str, str]:
    payload = json.loads((Path(project_root) / EXANTE_CONTRACT_PATH).read_text(encoding="utf-8"))
    return {str(column): str(role) for column, role in payload["columns"].items()}


def validate_role_manifest(
    manifest: Iterable[Mapping[str, Any]], project_root: Path = ROOT
) -> list[str]:
    failures: list[str] = []
    contract = _load_exante_contract(project_root)
    seen: set[str] = set()
    for item in manifest:
        column = str(item.get("column", ""))
        role = item.get("role")
        allowed = item.get("exante_allowed") is True
        if not column:
            failures.append("manifest entry has no column")
            continue
        if column in seen:
            failures.append(f"{column}: duplicate role-manifest entry")
        seen.add(column)
        if allowed and role not in X_ROLES:
            failures.append(f"{column}: non-X role is marked ex-ante allowed")
        if role in X_ROLES and not allowed:
            failures.append(f"{column}: X role is not marked ex-ante allowed")
        if column.startswith("x_") and column not in contract:
            failures.append(f"{column}: unknown X column")
        elif column in contract and role != contract[column]:
            failures.append(f"{column}: role {role!r} differs from contract {contract[column]!r}")
        if allowed and any(marker in column.lower() for marker in POST_RUN_X_MARKERS):
            failures.append(f"{column}: post-run field is marked as X")
    return failures


def _finite_nonnegative(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series(True, index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result &= values.notna() & values.map(math.isfinite) & values.ge(0)
    return result


def _recompute_eligibility(master: pd.DataFrame) -> dict[str, pd.Series]:
    base_required = (
        "x_protocol_task", "x_protocol_num_request_repeats", "x_hardware_gpu_model",
        "x_hardware_num_gpus", "x_deployment_max_num_seqs", "x_deployment_tensor_parallel",
        "x_deployment_expert_parallel", "x_deployment_data_parallel",
        "x_model_benchmark_total_params_billions", "x_model_benchmark_activated_params_billions",
        "x_model_benchmark_weight_precision", "x_workload_max_output_tokens", *TARGETS,
    )
    base = master[list(base_required)].notna().all(axis=1)
    workload_columns = [
        column for column in master
        if column.startswith(("x_workload_benchmark_", "x_workload_effective_"))
    ]
    physics_columns = [column for column in master if column.startswith("x_physics_")]
    full = (
        base
        & master["prov_tokenization_status"].eq("available")
        & master["prov_effective_status"].eq("available")
        & master["diag_quality_input_reconciliation_status"].eq("exact")
        & _finite_nonnegative(master, workload_columns)
    )
    physics = (
        full
        & master["prov_physics_status"].eq("available")
        & _finite_nonnegative(master, physics_columns)
    )
    return {
        "eligible_base_exante": base,
        "eligible_full_workload": full,
        "eligible_physics_feature": physics,
        "eligible_primary_stable": master["diag_quality_is_stable"].fillna(False).astype(bool),
    }


def validate_master_table(master: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    if len(master) != EXPECTED_RUNS:
        failures.append(f"expected {EXPECTED_RUNS} master rows, found {len(master)}")
    duplicates = int(master["run_key"].duplicated().sum()) if "run_key" in master else len(master)
    if duplicates:
        failures.append(f"run_key contains {duplicates} duplicates")
    stable = int(master["diag_quality_is_stable"].sum()) if "diag_quality_is_stable" in master else -1
    if stable != EXPECTED_STABLE_RUNS:
        failures.append(f"expected {EXPECTED_STABLE_RUNS} stable rows, found {stable}")
    tasks = set(master["x_protocol_task"].dropna()) if "x_protocol_task" in master else set()
    if tasks != set(TEXT_LLM_TASKS):
        failures.append(f"unexpected task set: {sorted(map(str, tasks))}")
    for target in TARGETS:
        if target not in master:
            failures.append(f"missing target {target}")
            continue
        missing = int(master[target].isna().sum())
        if missing:
            failures.append(f"{target} contains {missing} missing values")
        negative = int((master[target].dropna() < 0).sum())
        if negative:
            failures.append(f"{target} contains {negative} negative values")
        nonfinite = int(sum(not math.isfinite(float(value)) for value in master[target].dropna()))
        if nonfinite:
            failures.append(f"{target} contains {nonfinite} non-finite values")
    for column in (name for name in master if name.startswith("x_") and pd.api.types.is_numeric_dtype(master[name])):
        values = pd.to_numeric(master[column], errors="coerce").dropna()
        negative = int(values.lt(0).sum())
        nonfinite = int((~values.map(math.isfinite)).sum())
        if negative:
            failures.append(f"{column} contains {negative} negative values")
        if nonfinite:
            failures.append(f"{column} contains {nonfinite} non-finite values")
    expected_flags = {
        "eligible_base_exante": 694,
        "eligible_full_workload": 464,
        "eligible_physics_feature": 383,
        "eligible_primary_stable": 565,
    }
    for flag, expected in expected_flags.items():
        actual = int(master[flag].sum()) if flag in master else -1
        if actual != expected:
            failures.append(f"{flag} expected {expected}, found {actual}")
    required_for_recompute = {
        "prov_tokenization_status", "prov_effective_status", "prov_physics_status",
        "diag_quality_input_reconciliation_status", "diag_quality_is_stable", *TARGETS,
    }
    if required_for_recompute.issubset(master.columns):
        recomputed = _recompute_eligibility(master)
        labels = {
            "eligible_base_exante": "base eligibility differs",
            "eligible_full_workload": "full-workload eligibility differs",
            "eligible_physics_feature": "physics eligibility differs",
            "eligible_primary_stable": "stable eligibility differs",
        }
        for flag, expected in recomputed.items():
            if flag not in master:
                continue
            actual = master[flag].fillna(False).astype(bool)
            differences = int((actual != expected).sum())
            if differences:
                failures.append(f"{labels[flag]} on {differences} rows")
        from step3.scripts.build_master_table import restriction_reasons

        expected_reasons = pd.Series(
            [restriction_reasons(row) for row in master.to_dict(orient="records")],
            index=master.index,
        ).fillna("")
        actual_reasons = master.get("restricted_feature_reason", pd.Series("", index=master.index)).fillna("")
        reason_differences = int((actual_reasons != expected_reasons).sum())
        if reason_differences:
            failures.append(f"restricted_feature_reason differs on {reason_differences} rows")
    if {"eligible_full_workload", "eligible_base_exante"}.issubset(master):
        invalid = int((master["eligible_full_workload"] & ~master["eligible_base_exante"]).sum())
        if invalid:
            failures.append(f"{invalid} full-workload rows are not base eligible")
    if {"eligible_physics_feature", "eligible_full_workload"}.issubset(master):
        invalid = int((master["eligible_physics_feature"] & ~master["eligible_full_workload"]).sum())
        if invalid:
            failures.append(f"{invalid} physics rows are not full-workload eligible")
    llama = master[master["diag_model_id"].astype(str).str.startswith("meta-llama/")] if "diag_model_id" in master else master.iloc[0:0]
    if len(llama) != 132:
        failures.append(f"expected 132 retained Llama rows, found {len(llama)}")
    return failures


def validate_reconciliation(report: Mapping[str, Any]) -> list[str]:
    """Validate the persisted Task 3 reconciliation schema and exact counts."""

    failures: list[str] = []
    summary = report.get("complete_run_reconciliation", {})
    expected = {"runs": 562, "exact_matches": 523, "mismatches": 39}
    for field, value in expected.items():
        if summary.get(field) != value:
            failures.append(f"complete_run_reconciliation.{field} expected {value}, found {summary.get(field)}")
    detail = report.get("detail", [])
    if len(detail) != 694:
        failures.append(f"reconciliation detail expected 694 retained rows, found {len(detail)}")
    compared = [item for item in detail if item.get("comparison_scope") == "all_requests_completed"]
    if len(compared) != 562:
        failures.append(f"reconciliation detail expected 562 compared rows, found {len(compared)}")
    detail_mismatches = sum(item.get("exact_match") is False for item in compared)
    if detail_mismatches != 39:
        failures.append(f"reconciliation detail expected 39 mismatches, found {detail_mismatches}")
    exact = [item for item in compared if item.get("exact_match") is True]
    bad_exact = sum(item.get("reconciliation_status") != "exact" for item in exact)
    if bad_exact:
        failures.append(f"{bad_exact} exact reconciliation rows lack exact status")
    mismatches = [item for item in compared if item.get("exact_match") is False]
    bad_mismatch = sum(
        item.get("reconciliation_status") != "mismatch_unresolved_historical_equivalence"
        or not str(item.get("reconciliation_reason") or "").strip()
        for item in mismatches
    )
    if bad_mismatch:
        failures.append(f"{bad_mismatch} mismatch rows lack unresolved-equivalence status/reason")
    explanation_counts = report.get("mismatch_explanation_counts", {})
    expected_explanations = {
        "exact": 523,
        "not_comparable_missing_value": 132,
        "mismatch_unresolved_historical_equivalence": 39,
    }
    if explanation_counts != expected_explanations:
        failures.append(
            f"mismatch_explanation_counts expected {expected_explanations}, found {explanation_counts}"
        )
    evidence = report.get("pinned_repeat_semantics_evidence", {})
    if not all(evidence.get(field) for field in ("benchmark_commit", "path", "sha256", "semantics", "historical_limit")):
        failures.append("pinned_repeat_semantics_evidence is incomplete")
    return failures


def _candidate_files(root: Path) -> list[Path]:
    """Return tracked plus untracked/non-ignored files scoped to root."""

    root = root.resolve()
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True, capture_output=True,
        ).stdout
        paths = [root / item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item]
        return sorted(path for path in paths if path.is_file())
    except subprocess.CalledProcessError:
        excluded = {".git", ".venv", "__pycache__"}
        return sorted(
            path for path in root.rglob("*")
            if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts)
        )


def _python_has_training_call(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.split(".", 1)[0] in {"sklearn", "xgboost", "lightgbm"} for name in names):
                return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "train"}:
                return True
            if isinstance(node.func, ast.Name) and node.func.id in {"fit", "train"}:
                return True
    return False


def scan_forbidden_artifacts(root: Path) -> list[dict[str, str]]:
    """Scan every tracked or untracked/non-ignored file in the requested scope."""

    root = Path(root).resolve()
    findings: list[dict[str, str]] = []
    text_suffixes = {".py", ".md", ".json", ".csv", ".ipynb", ".txt"}
    for path in _candidate_files(root):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            findings.append({"category": "model_weight", "path": relative})
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append({"category": "credential", "path": relative})
        if path.suffix.lower() == ".json":
            try:
                if _contains_key(json.loads(text), "timeline"):
                    findings.append({"category": "raw_timeline", "path": relative})
            except json.JSONDecodeError:
                pass
        source = text
        if path.suffix.lower() == ".ipynb":
            try:
                notebook = json.loads(text)
                source = "\n".join(
                    "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
                    for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
                )
            except json.JSONDecodeError:
                source = ""
        if path.suffix.lower() in {".py", ".ipynb"} and _python_has_training_call(source):
            findings.append({"category": "formal_model_training", "path": relative})
    return sorted(findings, key=lambda item: (item["category"], item["path"]))


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "evidence": evidence}


def run_validations(project_root: Path = ROOT) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    analysis = project_root / "step3/analysis"
    reports = project_root / "step3/reports"
    master = pd.read_parquet(analysis / "master_feature_table.parquet")
    roles = json.loads((analysis / "feature_role_manifest.json").read_text(encoding="utf-8"))["columns"]
    models = pd.read_parquet(analysis / "model_architecture_features.parquet")
    coverage = json.loads((reports / "eligibility_coverage.json").read_text(encoding="utf-8"))
    reconciliation = json.loads((reports / "planned_workload_reconciliation.json").read_text(encoding="utf-8"))
    rapl = json.loads((analysis / "rapl_zone_identity_evidence.json").read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []
    master_failures = validate_master_table(master)
    checks.append(_check("master_key_target_range_eligibility", not master_failures, master_failures or "694 unique / 565 stable / targets finite"))
    role_failures = validate_role_manifest(roles, project_root)
    contract_columns = set(_load_exante_contract(project_root))
    role_columns = {item["column"] for item in roles}
    manifest_complete = role_columns == set(master.columns)
    manifest_x_columns = {item["column"] for item in roles if item.get("exante_allowed") is True}
    contract_complete = manifest_x_columns == contract_columns
    checks.append(_check("x_allowlist_and_post_run_denylist", not role_failures and manifest_complete and contract_complete, role_failures or {"manifest_columns": len(role_columns), "master_columns": len(master.columns), "contract_x_columns": len(contract_columns), "manifest_x_columns": len(manifest_x_columns)}))
    checks.append(_check("canonical_model_population", len(models) == 27 and models["model_id"].nunique() == 27, {"rows": len(models), "unique": int(models["model_id"].nunique())}))
    config_status = models["config_status"].value_counts().to_dict()
    checks.append(_check("canonical_missingness_is_explicit", config_status == {"available": 20, "restricted": 7} and models.loc[models["config_status"] != "available", "restricted_feature_reason"].notna().all(), config_status))
    from step3.scripts.build_master_table import build_coverage

    overall_all = next(item for item in coverage["records"] if item["population"] == "all" and item["dimension"] == "overall")
    rebuilt_coverage = build_coverage(master)
    coverage_records_match = sorted(coverage["records"], key=lambda item: json.dumps(item, sort_keys=True)) == sorted(
        rebuilt_coverage["records"], key=lambda item: json.dumps(item, sort_keys=True)
    )
    coverage_ok = (
        coverage["run_count"] == 694
        and coverage["stable_run_count"] == 565
        and overall_all["eligible_base_exante_runs"] == 694
        and overall_all["eligible_full_workload_runs"] == 464
        and overall_all["eligible_physics_feature_runs"] == 383
        and coverage_records_match
    )
    checks.append(_check("coverage_reconciliation", coverage_ok, overall_all))
    complete = reconciliation["complete_run_reconciliation"]
    reconciliation_failures = validate_reconciliation(reconciliation)
    checks.append(_check("planned_result_reconciliation_enumerated", not reconciliation_failures, reconciliation_failures or complete))
    checks.append(_check("rapl_direct_evidence_boundary", rapl["zone_1_identity"]["status"] == "unresolved" and rapl["zone_1_identity"]["numeric_inference_used"] is False and not rapl["all_direct_mapping_hits"], rapl["zone_1_identity"]))

    required = (
        "step3/analysis/master_feature_table.parquet",
        "step3/analysis/master_feature_table.csv",
        "step3/analysis/feature_role_manifest.json",
        "step3/docs/master_feature_dictionary.md",
        "step3/docs/physics_feature_formulas.md",
        "step3/reports/eligibility_coverage.json",
        "step3/reports/rapl_zone_identity_check.md",
        "step3/notebooks/phase1_7_feature_completion_audit.ipynb",
        "step3/reports/phase1_7_feature_completion_report.md",
    )
    missing = [path for path in required if not (project_root / path).exists()]
    checks.append(_check("required_deliverables", not missing, missing or list(required)))
    forbidden = scan_forbidden_artifacts(project_root / "step3")
    checks.append(_check("no_secrets_weights_timelines_or_training", not forbidden, forbidden or "no findings"))
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "ML.ENERGY V3 GPU-side LLM inference Phase 1.7; no formal model training",
        "checks": checks,
    }


def main() -> None:
    receipt = run_validations(ROOT)
    path = ROOT / "step3/reports/verification_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": len(receipt["checks"])}, ensure_ascii=False))
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
