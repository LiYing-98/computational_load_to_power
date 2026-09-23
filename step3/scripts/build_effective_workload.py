#!/usr/bin/env python3
"""Reconstruct execution-time-known effective input lengths at the vLLM boundary."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from step3.scripts.common import sha256_bytes, sha256_file, stable_error_message
from step3.scripts.reconstruct_planned_workload import (
    canonical_bytes,
    load_lmarena_requests,
    build_gpqa_requests,
    local_model_path,
)

ROOT = Path(__file__).resolve().parents[2]
VLLM_COMMIT = "439368496db48d8f992ba8c606a0c0b1eebbfa69"


def build_chat_messages(
    task: str,
    request: dict[str, Any],
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    if task not in {"gpqa", "lm-arena-chat"}:
        raise ValueError(f"task does not use Chat Completions: {task}")
    prompt = request["prompt"]
    prompts = [prompt] if isinstance(prompt, str) else prompt
    messages: list[dict[str, Any]] = []
    for index, text in enumerate(prompts):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    return isinstance(node, jinja2.nodes.Name) and node.ctx == "load" and node.name == varname


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return _is_var_access(node.node, varname) and isinstance(node.arg, jinja2.nodes.Const) and node.arg.value == key
    return isinstance(node, jinja2.nodes.Getattr) and _is_var_access(node.node, varname) and node.attr == key


def _is_var_or_elems_access(node: jinja2.nodes.Node, varname: str, key: str | None = None) -> bool:
    if isinstance(node, (jinja2.nodes.Filter, jinja2.nodes.Test)):
        return _is_var_or_elems_access(node.node, varname, key)
    if isinstance(node, jinja2.nodes.Getitem) and isinstance(node.arg, jinja2.nodes.Slice):
        return _is_var_or_elems_access(node.node, varname, key)
    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _iter_assigned_vars(root: jinja2.nodes.Node, varname: str):
    yield root, varname
    related = deque([varname])
    while related:
        source = related.popleft()
        for assignment in root.find_all(jinja2.nodes.Assign):
            if _is_var_or_elems_access(assignment.node, source):
                if not isinstance(assignment.target, jinja2.nodes.Name):
                    continue
                yield assignment, assignment.target.name
                if assignment.target.name != source:
                    related.append(assignment.target.name)


@lru_cache(maxsize=64)
def detect_chat_content_format(chat_template: str) -> str:
    """Vendored vLLM v0.11.1 AST rule: loop over message content means OpenAI parts."""
    root = jinja2.Environment().parse(chat_template)
    message_vars: list[str] = []
    for loop in root.find_all(jinja2.nodes.For):
        if not isinstance(loop.target, jinja2.nodes.Name):
            continue
        if any(_is_var_or_elems_access(loop.iter, name) for _, name in _iter_assigned_vars(root, "messages")):
            message_vars.append(loop.target.name)
    for loop in root.find_all(jinja2.nodes.For):
        if any(_is_var_or_elems_access(loop.iter, name, "content") for name in message_vars):
            return "openai"
    return "string"


def normalize_chat_messages(messages: list[dict[str, Any]], content_format: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = content
        rendered: Any = parts if content_format == "openai" else "\n".join(part["text"] for part in parts)
        normalized.append({"role": message["role"], "content": rendered})
    return normalized


def chat_effective_length(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    template = tokenizer.get_chat_template() if hasattr(tokenizer, "get_chat_template") else getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("chat template unavailable; effective length left unresolved")
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    return len(token_ids)


def fim_prompt(model_id: str, prefix: str, suffix: str) -> str:
    if model_id.lower().startswith("qwen/qwen3-coder"):
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    raise NotImplementedError(f"unsupported official FIM renderer: {model_id}")


def effective_length_for_request(
    task: str,
    model_id: str,
    request: dict[str, Any],
    tokenizer: Any,
    *,
    system_prompt: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    if task == "sourcegraph-fim":
        return len(tokenizer(fim_prompt(model_id, request["prefix"], request["suffix"])).input_ids)
    client_messages = build_chat_messages(task, request, system_prompt)
    template = tokenizer.get_chat_template() if hasattr(tokenizer, "get_chat_template") else getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("chat template unavailable; effective length left unresolved")
    content_format = detect_chat_content_format(template)
    conversation = normalize_chat_messages(client_messages, content_format)
    return chat_effective_length(tokenizer, conversation, chat_template_kwargs)


def aggregate_effective_lengths(lengths: list[int], repeat_count: int) -> dict[str, int | float]:
    if not lengths or repeat_count < 1 or any(int(value) <= 0 for value in lengths):
        raise ValueError("positive effective lengths and repeat_count are required")
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "effective_request_count": int(len(values) * repeat_count),
        "effective_total_input_tokens": int(values.sum() * repeat_count),
        "effective_input_tokens_mean": float(values.mean()),
        "effective_input_tokens_p50": float(np.quantile(values, 0.50, method="linear")),
        "effective_input_tokens_p90": float(np.quantile(values, 0.90, method="linear")),
        "effective_input_tokens_p95": float(np.quantile(values, 0.95, method="linear")),
        "effective_input_tokens_min": int(values.min()),
        "effective_input_tokens_max": int(values.max()),
        "effective_input_tokens_std": float(values.std(ddof=0)),
        "effective_sum_input_tokens_squared": int(np.square(values).sum() * repeat_count),
    }


def request_overrides(task: str, model_id: str) -> tuple[str | None, dict[str, Any]]:
    if model_id.startswith("nvidia/NVIDIA-Nemotron-Nano"):
        return ("/think" if task == "gpqa" else "/no_think"), {}
    if model_id in {"Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "Qwen/Qwen3-8B"}:
        return None, {"enable_thinking": task == "gpqa"}
    if model_id == "deepseek-ai/DeepSeek-V3.1":
        return None, {"thinking": task == "gpqa"}
    return None, {}


def override_source_pattern(task: str, model_id: str) -> str | None:
    base = f"configs/vllm/{task}/{model_id}/{{gpu_model}}"
    if model_id.startswith("nvidia/NVIDIA-Nemotron-Nano"):
        return f"{base}/system_prompt.txt"
    if model_id in {"Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-V3.1", "openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
        return f"{base}/extra_body.json"
    return None


def load_requests() -> dict[str, list[dict[str, Any]]]:
    sourcegraph = json.loads((ROOT / "step2/data/planned_requests/sourcegraph-fim.json").read_text(encoding="utf-8"))
    return {
        "gpqa": build_gpqa_requests(),
        "lm-arena-chat": load_lmarena_requests(),
        "sourcegraph-fim": sourcegraph,
    }


def build_pair_features(planned: pd.DataFrame) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    requests = load_requests()
    pair_features: dict[tuple[str, str], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    vector_root = ROOT / "step3/data/effective_vectors"
    vector_root.mkdir(parents=True, exist_ok=True)
    for task, model_id in planned[["task", "model_id"]].drop_duplicates().sort_values(["task", "model_id"]).itertuples(index=False, name=None):
        pair = (str(task), str(model_id))
        planned_pair = planned[(planned["task"] == task) & (planned["model_id"] == model_id)].iloc[0]
        base = {"task": task, "model_id": model_id}
        if planned_pair["tokenization_status"] != "available":
            reason = str(planned_pair["missing_reason"])
            pair_features[pair] = {"effective_status": "missing", "effective_missing_reason": reason}
            records.append({**base, "status": "missing", "reason": reason})
            continue
        if str(model_id).startswith("openai/gpt-oss"):
            reason = "unresolved_vllm_harmony_path: v0.11.1 bypasses the Hugging Face chat template and the container-resolved openai-harmony version is not recorded"
            pair_features[pair] = {"effective_status": "unresolved", "effective_missing_reason": reason}
            records.append({
                **base,
                "status": "unresolved",
                "reason": reason,
                "benchmark_config_source": override_source_pattern(task, model_id),
            })
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path(model_id), local_files_only=True, trust_remote_code=False)
            system_prompt, template_kwargs = request_overrides(task, model_id)
            if task == "sourcegraph-fim":
                content_format = "not_applicable_fim"
                template_hash = None
            else:
                template = tokenizer.get_chat_template()
                if not template:
                    raise ValueError("chat template unavailable; effective length left unresolved")
                content_format = detect_chat_content_format(template)
                template_hash = sha256_bytes(template.encode("utf-8"))
            lengths = [
                effective_length_for_request(
                    task,
                    model_id,
                    request,
                    tokenizer,
                    system_prompt=system_prompt,
                    chat_template_kwargs=template_kwargs,
                )
                for request in requests[task]
            ]
            vector_path = vector_root / f"{task}--{model_id.replace('/', '--')}.json"
            vector_path.write_bytes(canonical_bytes(lengths))
            if task == "sourcegraph-fim":
                planned_total = int(planned_pair["planned_total_input_tokens"] / int(planned_pair["num_request_repeats"]))
                if sum(lengths) != planned_total:
                    raise ValueError("FIM effective lengths differ from official benchmark prompt lengths")
            feature = {
                "effective_status": "available",
                "effective_missing_reason": None,
                "chat_content_format": content_format,
                "chat_template_sha256": template_hash,
                "system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")) if system_prompt else None,
                "chat_template_kwargs_json": json.dumps(template_kwargs, sort_keys=True),
                "base_effective_lengths_sha256": sha256_file(vector_path),
                "base_lengths": lengths,
            }
            pair_features[pair] = feature
            records.append({
                **base,
                "status": "available",
                "request_count": len(lengths),
                "content_format": content_format,
                "chat_template_sha256": template_hash,
                "system_prompt": system_prompt,
                "chat_template_kwargs": template_kwargs,
                "benchmark_config_source": override_source_pattern(task, model_id),
                "vector_sha256": feature["base_effective_lengths_sha256"],
            })
        except Exception as error:
            reason = stable_error_message(error)
            pair_features[pair] = {"effective_status": "unresolved", "effective_missing_reason": reason}
            records.append({**base, "status": "unresolved", "reason": reason})
    return pair_features, records


def difference_report(frame: pd.DataFrame) -> dict[str, Any]:
    available = frame[frame["effective_status"] == "available"].copy()
    available["total_delta"] = available["effective_total_input_tokens"] - available["planned_total_input_tokens"]
    pair = available.drop_duplicates(["task", "model_id"])
    pair_records = []
    for row in pair.to_dict(orient="records"):
        pair_records.append({
            "task": row["task"],
            "model_id": row["model_id"],
            "benchmark_mean": row["planned_input_tokens_mean"],
            "effective_mean": row["effective_input_tokens_mean"],
            "mean_delta": row["effective_input_tokens_mean"] - row["planned_input_tokens_mean"],
            "mean_ratio": row["effective_input_tokens_mean"] / row["planned_input_tokens_mean"],
        })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_runs": int(len(available)),
        "zero_delta_runs": int((available["total_delta"] == 0).sum()),
        "positive_delta_runs": int((available["total_delta"] > 0).sum()),
        "total_delta_min": int(available["total_delta"].min()) if len(available) else None,
        "total_delta_median": float(available["total_delta"].median()) if len(available) else None,
        "total_delta_max": int(available["total_delta"].max()) if len(available) else None,
        "pair_differences": pair_records,
    }


def main() -> None:
    planned = pd.read_parquet(ROOT / "step3/analysis/planned_workload_features.parquet")
    pairs, pair_records = build_pair_features(planned)
    rows: list[dict[str, Any]] = []
    metric_names = list(aggregate_effective_lengths([1], 1))
    for row in planned.to_dict(orient="records"):
        pair = pairs[(row["task"], row["model_id"])]
        output = {"run_key": row["run_key"], "task": row["task"], "model_id": row["model_id"]}
        output.update({key: value for key, value in pair.items() if key != "base_lengths"})
        if pair["effective_status"] == "available":
            output.update(aggregate_effective_lengths(pair["base_lengths"], int(row["num_request_repeats"])))
        else:
            output.update({name: None for name in metric_names})
        rows.append(output)
    frame = pd.DataFrame(rows)
    analysis = ROOT / "step3/analysis"
    reports = ROOT / "step3/reports"
    frame.to_parquet(analysis / "effective_workload_features.parquet", index=False)
    frame.to_csv(analysis / "effective_workload_features.csv", index=False)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_commit": "c1d6557fbf7c7c495749b0fcf649f71774eabde9",
        "vllm_commit": VLLM_COMMIT,
        "vllm_defaults": {"add_generation_prompt": True, "continue_final_message": False, "add_special_tokens": False},
        "benchmark_message_builder": "typed text parts; alternating user/assistant; optional system message prepended",
        "content_format_detection": "vendored vLLM v0.11.1 Jinja AST rule",
        "rows": len(frame),
        "available_rows": int((frame["effective_status"] == "available").sum()),
        "pair_records": pair_records,
    }
    (analysis / "effective_workload_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = difference_report(planned.merge(frame, on=["run_key", "task", "model_id"], validate="one_to_one"))
    (reports / "benchmark_vs_effective_input.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(frame)} effective rows; available={provenance['available_rows']}")


if __name__ == "__main__":
    main()
