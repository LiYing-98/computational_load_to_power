#!/usr/bin/env python3
"""Shared contracts for the Phase 1.7 feature-completion pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


TEXT_LLM_TASKS = ("gpqa", "lm-arena-chat", "sourcegraph-fim")
EXPECTED_RUNS = 694
EXPECTED_STABLE_RUNS = 565

STATIC_MODEL_BASENAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")


def validate_population_counts(runs: int, stable_runs: int) -> list[str]:
    failures: list[str] = []
    if runs != EXPECTED_RUNS:
        failures.append(f"expected {EXPECTED_RUNS} runs, found {runs}")
    if stable_runs != EXPECTED_STABLE_RUNS:
        failures.append(
            f"expected {EXPECTED_STABLE_RUNS} stable runs, found {stable_runs}"
        )
    return failures


def stable_error_message(error: Exception) -> str:
    message = re.sub(r"\s*\(Request ID:[^)]+\)", "", str(error))
    return f"{type(error).__name__}: {message.strip()}"


def allowed_static_model_file(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith(WEIGHT_SUFFIXES):
        return False
    return Path(normalized).name in STATIC_MODEL_BASENAMES


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def resolve_run_table_path() -> Path:
    candidates = (
        Path("step2/data/runs/llm.parquet"),
        Path("step1/data/runs/llm.parquet"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("run table not found in step2/data or step1/data")


@dataclass(frozen=True)
class RunKey:
    task: str
    model_id: str
    gpu_model: str
    num_gpus: int
    max_num_seqs: int
    tensor_parallel: int
    expert_parallel: int
    data_parallel: int
    seed: int
    num_request_repeats: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunKey":
        return cls(**{field.name: value[field.name] for field in fields(cls)})

    def canonical(self) -> str:
        return json.dumps(
            {field.name: getattr(self, field.name) for field in fields(self)},
            ensure_ascii=False,
            separators=(",", ":"),
        )

