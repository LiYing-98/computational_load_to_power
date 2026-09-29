#!/usr/bin/env python3
"""Shared contracts for ML.ENERGY Phase 3.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STEP5 = ROOT / "step5"
SEED = 20260928


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative_path: str, project_root: Path = ROOT) -> dict[str, Any]:
    return json.loads((Path(project_root) / relative_path).read_text(encoding="utf-8"))


def verify_source_freeze(project_root: Path = ROOT) -> list[str]:
    root = Path(project_root)
    freeze = load_json("step5/config/source_freeze_phase3.json", root)
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


def load_feature_bundles(project_root: Path = ROOT) -> dict[str, Any]:
    root = Path(project_root)
    config = load_json("step5/config/feature_bundles_phase3.json", root)
    inherited = load_json(config["inherits"], root)
    bundles = dict(inherited["bundles"])
    overlap = set(bundles) & set(config["bundles"])
    if overlap:
        raise ValueError(f"Phase 3 bundles may not override frozen Phase 2 bundles: {sorted(overlap)}")
    pending = dict(config["bundles"])
    while pending:
        progressed = False
        for name, raw in list(pending.items()):
            base_name = raw.get("base_bundle")
            if base_name not in bundles:
                continue
            base = bundles[base_name]
            drop = set(raw.get("drop", []))
            categorical = [
                column
                for column in [*base.get("categorical", []), *raw.get("categorical_add", [])]
                if column not in drop
            ]
            numeric = [
                column
                for column in [*base.get("numeric", []), *raw.get("numeric_add", [])]
                if column not in drop
            ]
            bundles[name] = {
                **raw,
                "categorical": list(dict.fromkeys(categorical)),
                "numeric": list(dict.fromkeys(numeric)),
                "columns": list(dict.fromkeys([*categorical, *numeric])),
            }
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(f"unresolved Phase 3 bundle bases: {sorted(pending)}")
    return {**config, "bundles": bundles}
