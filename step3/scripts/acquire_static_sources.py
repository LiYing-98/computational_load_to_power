#!/usr/bin/env python3
"""Acquire only pinned, bounded static sources needed by Phase 1.7."""

from __future__ import annotations

import argparse
import getpass
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, snapshot_download

from step3.scripts.common import (
    STATIC_MODEL_BASENAMES,
    allowed_static_model_file,
    sha256_file,
    stable_error_message,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "step3/config/model_revisions.json"
HASHES_PATH = ROOT / "step3/config/source_hashes.json"
DATA_ROOT = ROOT / "step3/data"
ANALYSIS_ROOT = ROOT / "step3/analysis"
PHASE16_HF = ROOT / "step2/data/huggingface"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SECRET_PATTERN = re.compile(r"hf_[A-Za-z0-9]+")


def validate_revision_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    models = manifest.get("models", {})
    if len(models) != 27:
        errors.append(f"expected 27 unique models, found {len(models)}")
    for model_id, record in sorted(models.items()):
        revision = record.get("revision", "")
        if not HEX40.fullmatch(revision):
            errors.append(f"{model_id}: revision is not a 40-character commit SHA")
        if record.get("access") not in {"public", "authorized", "restricted"}:
            errors.append(f"{model_id}: invalid access classification")
        if record.get("static_only") is not True:
            errors.append(f"{model_id}: static_only must be true")
    for key, field in (("gpqa_dataset", "revision"), ("benchmark", "commit")):
        if not HEX40.fullmatch(manifest.get(key, {}).get(field, "")):
            errors.append(f"{key}.{field}: invalid pinned commit")
    if manifest.get("vllm", {}).get("version") != "v0.11.1":
        errors.append("vllm.version: expected v0.11.1")
    if not HEX40.fullmatch(manifest.get("vllm", {}).get("commit", "")):
        errors.append("vllm.commit: invalid pinned commit")
    return errors


def audit_static_tree(root: Path) -> list[str]:
    failures: list[str] = []
    if not root.exists():
        return failures
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if not allowed_static_model_file(relative):
            failures.append(f"{relative}: forbidden static-model file")
    return failures


def build_source_receipt(
    root: Path, *, source_type: str, source_id: str, revision: str
) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {
        "source_type": source_type,
        "source_id": source_id,
        "revision": revision,
        "file_count": len(files),
        "files": files,
    }


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def _phase16_snapshot(model_id: str, revision: str) -> Path:
    return PHASE16_HF / f"models--{_model_slug(model_id)}" / "snapshots" / revision


def _copy_allowed(source: Path, destination: Path) -> int:
    copied = 0
    if not source.exists():
        return copied
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        if allowed_static_model_file(relative.as_posix()):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    return copied


def _download_model_static(model_id: str, revision: str, token: str | None) -> Path:
    destination = DATA_ROOT / "static_models" / _model_slug(model_id)
    destination.mkdir(parents=True, exist_ok=True)
    _copy_allowed(_phase16_snapshot(model_id, revision), destination)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=token,
                cache_dir=DATA_ROOT / "hf_cache",
                allow_patterns=sorted(STATIC_MODEL_BASENAMES),
            )
        )
        _copy_allowed(snapshot, destination)
    except OSError:
        # Windows without Developer Mode can reject Hub-cache symlink creation.
        # Fetch the same finite allowlist directly at the pinned commit instead.
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        for basename in sorted(STATIC_MODEL_BASENAMES):
            response = requests.get(
                f"https://huggingface.co/{model_id}/resolve/{revision}/{basename}",
                headers=headers,
                timeout=60,
            )
            if response.status_code == 404:
                continue
            response.raise_for_status()
            (destination / basename).write_bytes(response.content)
    failures = audit_static_tree(destination)
    if failures:
        raise RuntimeError("; ".join(failures))
    return destination


def _download_gpqa(manifest: dict[str, Any], token: str) -> Path:
    spec = manifest["gpqa_dataset"]
    return Path(
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=spec["revision"],
            token=token,
            cache_dir=DATA_ROOT / "hf_cache",
            allow_patterns=["*.csv", "*.json", "*.jsonl", "*.parquet", "README.md", ".gitattributes"],
        )
    )


def _fetch_pinned_vllm(expected: dict[str, Any]) -> Path:
    destination = DATA_ROOT / "vllm-v0.11.1"
    for relative, expected_hash in expected["files"].items():
        url = f"https://raw.githubusercontent.com/vllm-project/vllm/{expected['commit']}/{relative}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"vLLM source hash mismatch for {relative}")
    return destination


def _verify_benchmark_sources(expected: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / "step2/data/official_source"
    for relative, digest in expected["files"].items():
        path = root / relative
        if not path.exists() or sha256_file(path) != digest:
            raise RuntimeError(f"benchmark source mismatch: {relative}")
    return build_source_receipt(
        root,
        source_type="benchmark_source",
        source_id="ml-energy/benchmark",
        revision=expected["commit"],
    )


def _refresh_revisions(manifest: dict[str, Any], token: str | None) -> None:
    api = HfApi(token=token)
    for model_id, record in sorted(manifest["models"].items()):
        if record["access"] == "restricted":
            continue
        record["revision"] = api.model_info(model_id, revision="main").sha
    CONFIG_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def acquire(*, refresh_revisions: bool = False) -> dict[str, Any]:
    manifest = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    errors = validate_revision_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    hashes = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
    token = getpass.getpass("Hugging Face token (hidden; not persisted): ").strip() or None
    if token and not token.startswith("hf_"):
        raise ValueError("unexpected Hugging Face credential format")
    if refresh_revisions:
        _refresh_revisions(manifest, token)
        raise RuntimeError("revisions refreshed; review and pin source hashes before acquisition")

    records: list[dict[str, Any]] = []
    for model_id, model in sorted(manifest["models"].items()):
        if model["access"] == "restricted":
            records.append({
                "source_type": "model_static",
                "source_id": model_id,
                "revision": model["revision"],
                "status": "restricted",
                "reason": "Meta Llama access rejected; no bypass or mirror used",
                "file_count": 0,
                "files": [],
            })
            continue
        try:
            root = _download_model_static(model_id, model["revision"], token)
            record = build_source_receipt(
                root,
                source_type="model_static",
                source_id=model_id,
                revision=model["revision"],
            )
            record["status"] = "available"
            records.append(record)
        except Exception as error:  # retain bounded, sanitized failure evidence
            records.append({
                "source_type": "model_static",
                "source_id": model_id,
                "revision": model["revision"],
                "status": "unavailable",
                "reason": SECRET_PATTERN.sub("[REDACTED]", stable_error_message(error)),
                "file_count": 0,
                "files": [],
            })

    if not token:
        raise RuntimeError("authorized GPQA access requires a hidden Hugging Face token")
    gpqa_root = _download_gpqa(manifest, token)
    gpqa_record = build_source_receipt(
        gpqa_root,
        source_type="dataset_snapshot",
        source_id=manifest["gpqa_dataset"]["repo_id"],
        revision=manifest["gpqa_dataset"]["revision"],
    )
    gpqa_record["status"] = "available"
    records.append(gpqa_record)

    records.append(_verify_benchmark_sources(hashes["benchmark"]))
    vllm_root = _fetch_pinned_vllm(hashes["vllm"])
    records.append(build_source_receipt(
        vllm_root,
        source_type="vllm_source",
        source_id="vllm-project/vllm",
        revision=hashes["vllm"]["commit"],
    ))

    receipt = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "static_model_files_only": True,
            "model_weights_downloaded": False,
            "refresh_revisions": False,
            "credentials_persisted": False,
        },
        "sources": records,
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    output = ANALYSIS_ROOT / "source_manifest.json"
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-revisions", action="store_true")
    args = parser.parse_args()
    receipt = acquire(refresh_revisions=args.refresh_revisions)
    available = sum(source.get("status", "available") == "available" for source in receipt["sources"])
    print(f"Wrote source manifest with {available} available source records.")


if __name__ == "__main__":
    main()
