# Phase 1.7 Complete Ex-Ante Feature Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete legally accessible planned/effective workload and static architecture features, derive transparent physics proxies, and deliver a validated one-row-per-run master feature table without training a prediction model.

**Architecture:** Build focused deterministic stages under `step3/scripts/`, each producing a versioned table and provenance record consumed by the next stage. Reuse Phase 1.6 artifacts, keep gated data and caches ignored, and enforce the ex-ante leakage boundary in a final validator and executed notebook.

**Tech Stack:** Python 3.11, pandas 3.0.1, pyarrow 25.0.1, NumPy 2.3.4, datasets 4.4.1, transformers 4.57.1, tokenizers 0.22.1, huggingface-hub 0.36.2, nbformat/nbclient/nbconvert, matplotlib, unittest.

**Spec:** `step3/docs/superpowers/specs/2026-09-18-phase1-7-feature-table-design.md`

## Global Constraints

- Scope is exactly 694 ML.ENERGY V3 text-LLM inference runs and 565 stable runs across `gpqa`, `lm-arena-chat`, and `sourcegraph-fim`.
- Reuse Phase 1.6 outputs; do not redo completed Phase 1/1.5/1.6 audits.
- GPQA revision is `633f5ee89ab8ad4522a9f850766b73f62147ffdd`; benchmark commit is `c1d6557fbf7c7c495749b0fcf649f71774eabde9`; seed is `48105`.
- Use authorized GPQA/Gemma access. Do not bypass Meta Llama gating or use mirrors/substitute tokenizers.
- Download static config/tokenizer/chat-template files only. Reject all model-weight extensions and do not fetch full raw timelines or Prometheus.
- Preserve benchmark prompt length and effective input length as separate semantics.
- Use only execution-time-known X fields. Never put actual outputs, throughput, latency, actual batch, duration, telemetry, or stability outcomes into X.
- Finish after tables, audit, report, notebook, tests, and verification receipt. Do not train a formal model.

---

### Task 1: Reproducible Phase 1.7 foundation

**Files:**
- Create: `step3/README.md`
- Create: `step3/__init__.py`
- Create: `step3/scripts/__init__.py`
- Create: `step3/scripts/common.py`
- Create: `step3/requirements-phase17.txt`
- Create: `step3/tests/__init__.py`
- Create: `step3/tests/test_phase17.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `step1/data/runs/llm.parquet` or `step2/data/runs/llm.parquet`; Phase 1.6 analysis artifacts.
- Produces: `resolve_run_table_path()`, `stable_error_message()`, `sha256_*()`, `RunKey`, schema constants, and one local `step3/.venv` execution path.

- [x] Add failing tests that require exact 694/565 input counts, stable run-key construction, deterministic error sanitization, and rejection of weight-file suffixes.
- [x] Run `step3/.venv/Scripts/python.exe -m unittest step3.tests.test_phase17 -v` and confirm the new imports fail.
- [x] Implement the minimal common utilities and pinned requirements; add `step3/data/`, `step3/.venv/`, `step3/.secrets/`, and rendered notebook HTML to `.gitignore`.
- [x] Create the local virtual environment, install the pinned requirements, and run the tests to green.
- [x] Commit the independently working foundation.

### Task 2: Bounded static-source acquisition and revision manifests

**Files:**
- Create: `step3/scripts/acquire_static_sources.py`
- Create: `step3/config/model_revisions.json`
- Create: `step3/config/source_hashes.json`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: Phase 1.6 tokenizer revision manifest and authorized token entered through hidden `getpass`.
- Produces: ignored `step3/data/static_models/<model>/`, ignored pinned GPQA dataset cache, ignored pinned vLLM source, and `step3/analysis/source_manifest.json`.

- [x] Add tests for exact revision coverage of all 27 unique models, allowlisted static filenames, denied weight extensions, hash receipts, and no credential persistence.
- [x] Run the acquisition tests and confirm they fail because the acquisition module is absent.
- [x] Implement reuse of Phase 1.6 cached static files and bounded Hub downloads at pinned revisions. Add explicit `--refresh-revisions`; the default must never query current HEAD.
- [x] Fetch and hash the minimum vLLM v0.11.1 source needed to establish OpenAI message normalization. Refuse unpinned or mismatched source.
- [x] Execute acquisition with hidden token, verify no weight file is present, then run tests to green and commit.

### Task 3: Complete planned workload for GPQA and Gemma

**Files:**
- Create: `step3/scripts/reconstruct_planned_workload.py`
- Create: `step3/analysis/planned_workload_features.parquet`
- Create: `step3/analysis/planned_workload_features.csv`
- Create: `step3/analysis/planned_workload_provenance.json`
- Create: `step3/reports/planned_workload_reconciliation.json`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: pinned task datasets, request builders, Phase 1.6 successful rows/vector hashes, tokenizer revisions, and bounded result metadata.
- Produces: the Phase 1.7 benchmark-length workload block at one-row-per-run grain plus model/task base-length vectors in ignored cache.

- [x] Add tests proving GPQA choice shuffling and prompt construction match the pinned benchmark, repetition scales additive metrics only, cached Phase 1.6 successes remain byte/value consistent, and complete-run reconciliation is exact.
- [x] Run the tests and confirm the new reconstruction interfaces fail.
- [x] Implement incremental reconstruction: copy verified Phase 1.6 successes, build GPQA once per tokenizer, rebuild Gemma, and preserve explicit missing reasons for rejected Llama.
- [x] Execute reconstruction with the already authorized pinned local sources and write coverage/reconciliation outputs.
- [x] Verify every complete reconstructed run against result `total_input_tokens`, enumerate any mismatch, run tests to green, and commit.

### Task 4: Effective workload reconstruction

**Files:**
- Create: `step3/scripts/build_effective_workload.py`
- Create: `step3/analysis/effective_workload_features.parquet`
- Create: `step3/analysis/effective_workload_features.csv`
- Create: `step3/analysis/effective_workload_provenance.json`
- Create: `step3/reports/benchmark_vs_effective_input.json`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: base request lists, pinned tokenizer/chat templates, pinned benchmark message builder, and pinned vLLM normalization evidence.
- Produces: `build_chat_messages(task, request, system_prompt)`, `effective_lengths(...)`, requested effective aggregates, status/reason fields, and task/model difference statistics.

- [x] Add exact message-shape tests for GPQA and multi-turn LMArena, a Chat-template test with `add_generation_prompt=True`, an unresolved-template test, a FIM no-Chat-template test, and additive-repeat aggregation tests.
- [x] Run the targeted tests and confirm failure before implementation.
- [x] Implement normalized Chat messages and template application from pinned static files; mark any unsupported historical behavior unresolved rather than guessing.
- [x] Implement FIM effective lengths using the official renderer and assert equality with benchmark lengths for resolved Sourcegraph rows.
- [x] Execute, write difference reports, run tests to green, and commit.

### Task 5: Canonical model architecture features

**Files:**
- Create: `step3/config/canonical_model_field_map.json`
- Create: `step3/scripts/build_model_features.py`
- Create: `step3/analysis/model_architecture_features.parquet`
- Create: `step3/analysis/model_architecture_features.csv`
- Create: `step3/analysis/model_field_mapping.csv`
- Create: `step3/reports/model_feature_missingness.json`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: pinned model `config.json` files and Phase 1.5 benchmark model metadata.
- Produces: one row per model canonical schema plus long-form `source_field -> canonical_field`, raw value, transform, status, and reason records.

- [x] Add fixture tests for Dense GQA, MoE, and hybrid configs; verify head-dimension derivation, expert/top-k mapping, null preservation, and benchmark precision precedence.
- [x] Run tests and confirm mapping functions are missing.
- [x] Implement declarative architecture-specific mappings and deterministic derivations without model-name guessing.
- [x] Execute across all 27 models, retain Llama/public-field rows with restricted reasons, and write missingness output.
- [x] Manually inspect representative Qwen dense/MoE, DeepSeek MoE, Nemotron hybrid, Gemma, gpt-oss, and Llama records; run tests to green and commit.

### Task 6: Physics-derived ex-ante features

**Files:**
- Create: `step3/scripts/build_physics_features.py`
- Create: `step3/analysis/physics_features.parquet`
- Create: `step3/analysis/physics_features.csv`
- Create: `step3/docs/physics_feature_formulas.md`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: effective base-length vectors, canonical model features, request counts/output caps, GPU/deployment settings, and precision.
- Produces: named theoretical prefill/decode/KV/weight/per-GPU proxies and `physics_status`/`physics_missing_reason`.

- [x] Add hand-calculated toy tests for GQA projection FLOPs, causal-attention FLOPs, dense gated/non-gated MLP, MoE active-expert MLP, decode summation, KV bytes, precision bytes, and per-GPU normalization.
- [x] Run tests and confirm the formula functions fail to import.
- [x] Implement pure formula functions with finite/nonnegative guards and explicit unsupported-hybrid handling.
- [x] Aggregate per-request features to run grain and write the formula/assumption document.
- [x] Recompute toy cases independently, run full tests, and commit.

### Task 7: Master feature table and eligibility flags

**Files:**
- Create: `step3/scripts/build_master_table.py`
- Create: `step3/analysis/master_feature_table.parquet`
- Create: `step3/analysis/master_feature_table.csv`
- Create: `step3/analysis/feature_role_manifest.json`
- Create: `step3/docs/master_feature_dictionary.md`
- Create: `step3/reports/eligibility_coverage.json`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: Phase 1.5 targets/base fields, planned/effective/model/physics tables, and stability status.
- Produces: exactly 694 run rows, role-tagged feature manifest, five eligibility/restriction fields, and coverage cuts.

- [x] Add tests for one-to-one joins, exact run-key uniqueness, 694/565 counts, target presence, Llama retention, ordered restriction reasons, and eligibility truth tables.
- [x] Run tests and confirm the master builder is missing.
- [x] Implement validated one-to-one joins, explicit benchmark aliases, Y/diagnostic separation, eligibility rules, and role manifest generation.
- [x] Generate overall/stable and task/model-family/GPU/architecture-family coverage tables.
- [x] Run tests and commit the master table stage.

### Task 8: Time-boxed RAPL identity check

**Files:**
- Create: `step3/scripts/audit_rapl_identity.py`
- Create: `step3/reports/rapl_zone_identity_check.md`
- Create: `step3/analysis/rapl_zone_identity_evidence.json`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: existing bounded metadata, existing logs/source receipts, and pinned Zeus source only.
- Produces: direct-evidence hits with file/hash/location or an explicit unresolved neutral result.

- [x] Add tests that prohibit inferring names from watts and require direct string evidence for a resolved zone identity.
- [x] Search only the bounded sources and record inspected paths/hashes.
- [x] Write the neutral result if no direct mapping exists; do not block downstream outputs.
- [x] Run tests and commit.

### Task 9: Leakage, data-quality, notebook, and final report

**Files:**
- Create: `step3/scripts/validate_phase17.py`
- Create: `step3/scripts/build_notebook.py`
- Create: `step3/notebooks/phase1_7_feature_completion_audit.ipynb`
- Create: `step3/reports/phase1_7_feature_completion_report.md`
- Create: `step3/reports/verification_receipt.json`
- Modify: `step3/README.md`
- Modify: `step3/tests/test_phase17.py`

**Interfaces:**
- Consumes: every final table, provenance file, field dictionary, and role manifest.
- Produces: deterministic validation checks, executed notebook, ignored HTML preview, quantitative final report, and pass/fail receipt.

- [x] Add validator tests for X allowlist/denylist, no formal-training markers, exact coverage reconciliation, canonical missingness, key/target/range/duplicate checks, no secrets, no timeline keys, no weight files, and required deliverables.
- [x] Run validator tests and confirm failure before implementation.
- [x] Implement validation and generate the notebook with `tl;dr`, context/methods, data-quality profile, workload coverage, benchmark-vs-effective distributions, canonical missingness, physics eligibility, Llama impact, and takeaways.
- [x] Execute the notebook top-to-bottom, render HTML, and visually inspect all charts/tables at reading size.
- [x] Write the final report with exact coverage, limitations, leakage result, and Phase 2 readiness judgment; explicitly state that no model was trained.
- [x] Run the full unit suite, validator, notebook execution, secret scan, `git diff --check`, and Git status review.
- [x] Commit the complete Phase 1.7 tree and stop before formal training.
