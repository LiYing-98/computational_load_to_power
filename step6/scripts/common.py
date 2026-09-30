#!/usr/bin/env python3
"""Shared contracts for ML.ENERGY Phase 4.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STEP6 = ROOT / "step6"
SEED = 20260929
TEXT_HASH_SUFFIXES = {
    ".csv",
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256_file(path: Path) -> str:
    """Hash text canonically (LF) and binary artifacts byte-for-byte.

    Git may materialize the same text blob with LF or CRLF depending on the
    checkout.  Normalizing text newlines keeps frozen contracts portable while
    preserving exact-byte hashes for parquet, joblib, images, and DOCX files.
    """
    path = Path(path)
    digest = hashlib.sha256()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(canonical.encode("utf-8"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative_path: str, project_root: Path = ROOT) -> dict[str, Any]:
    return json.loads((Path(project_root) / relative_path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_source_freeze(project_root: Path = ROOT) -> list[str]:
    root = Path(project_root)
    freeze = load_json("step6/config/source_freeze_phase4.json", root)
    failures: list[str] = []
    for item in freeze["frozen_inputs"]:
        path = root / item["path"]
        if not path.exists():
            failures.append(f"missing frozen input: {item['path']}")
            continue
        actual = sha256_file(path)
        if actual != item["sha256"]:
            failures.append(
                f"{item['path']} SHA256 changed: expected {item['sha256']}, found {actual}"
            )
    return failures


def stable_hash(value: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def load_feature_bundles(project_root: Path = ROOT) -> dict[str, Any]:
    root = Path(project_root)
    config = load_json("step6/config/feature_bundles_phase4.json", root)
    phase2 = load_json("step4/config/feature_bundles.json", root)["bundles"]
    resolved: dict[str, dict[str, Any]] = {}
    pending = dict(config["bundles"])
    while pending:
        progressed = False
        for name, raw in list(pending.items()):
            if "inherit_columns_from" in raw:
                source_name = raw["inherit_columns_from"].split(":", 1)[1]
                source = phase2[source_name]
                base = {
                    "categorical": list(source["categorical"]),
                    "numeric": list(source["numeric"]),
                }
            elif "base_bundle" in raw:
                base_name = raw["base_bundle"]
                if base_name not in resolved:
                    continue
                base = resolved[base_name]
            else:
                base = {"categorical": [], "numeric": []}
            drop = set(raw.get("drop", []))
            categorical = [
                c
                for c in [*base.get("categorical", []), *raw.get("categorical", []), *raw.get("categorical_add", [])]
                if c not in drop
            ]
            numeric = [
                c
                for c in [*base.get("numeric", []), *raw.get("numeric", []), *raw.get("numeric_add", [])]
                if c not in drop
            ]
            resolved[name] = {
                **raw,
                "categorical": list(dict.fromkeys(categorical)),
                "numeric": list(dict.fromkeys(numeric)),
                "columns": list(dict.fromkeys([*categorical, *numeric])),
            }
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(f"unresolved Phase 4 bundle bases: {sorted(pending)}")
    return {**config, "bundles": resolved}
