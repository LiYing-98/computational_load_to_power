# ML.ENERGY Phase 1.5 Complete Summary and Workload Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the gated ML.ENERGY V3 `runs/llm.parquet` and a bounded set of representative run metadata to determine whether a leakage-safe, physics-informed `X_pre` can be constructed before execution.

**Architecture:** Keep gated source files local and ignored, then produce deterministic Python audit outputs under `step1/analysis/`. Separate full-parquet profiling from selected-run workload inspection so no Prometheus/power timeline is fetched. Use a reproducible notebook and a local technical report app to expose evidence, limitations and the final A/B/C decision without training a model.

**Tech Stack:** Python 3.12, pandas/pyarrow, Python standard library, unittest, Jupyter/nbclient, Matplotlib, Data report app runtime.

**Spec:** `D:\my_work\cpecc\files\日常事务\科技项目\2025年\算电协同\课题一\阶段性研究进展\plan\first_step\Codex_Phase1.5任务书_MLENERGY完整汇总与工作量字段审计.docx`

## Global Constraints

- Only ML.ENERGY Benchmark V3 text LLM inference is in scope.
- Download only `runs/llm.parquet` plus approximately 6–12 selected run result/config metadata files.
- Never download the full raw timeline or bulk Prometheus/GPU power timeline.
- Never write the Hugging Face token to source, notebook, report, shell history artifact or Git.
- Classify every variable as pre-execution known, pre-execution derivable, target, post-execution leakage or diagnostic/provenance.
- Do not train Ridge, Random Forest, XGBoost, neural networks or any formal predictive model.
- Do not expand to Training, Diffusion, MLLM, CPU, Node Residual, NLR or CMU/LBNL data.
- Stop after the evidence-backed A/B/C decision.

---

### Task 1: Authenticate and acquire only the gated summary

**Files:**
- Modify: `.gitignore`
- Create: `step1/README.md`
- Create locally, ignored: `step1/data/runs/llm.parquet`
- Create: `step1/analysis/source_manifest.json`

**Interfaces:**
- Consumes: user-provided Hugging Face read token through process stdin only.
- Produces: authenticated `llm.parquet`, byte size, SHA-256, source URL and retrieval timestamp.

- [x] **Step 1: Add gated/raw paths to `.gitignore`**

```gitignore
step1/data/runs/
step1/data/selected_runs_raw/
```

- [x] **Step 2: Verify authorization before download**

Run a short-lived Python process that reads the token from stdin, calls `https://huggingface.co/api/whoami-v2`, prints only non-secret identity/access status, and exits nonzero on 401/403.

- [x] **Step 3: Download only the parquet summary**

In the same authenticated process, stream only:

```text
https://huggingface.co/datasets/ml-energy/benchmark-v3/resolve/main/runs/llm.parquet
```

Reject redirects or filenames indicating Prometheus/timeline bulk data, then calculate bytes and SHA-256.

- [x] **Step 4: Verify the file is a readable parquet**

Run:

```powershell
python -c "import pandas as pd; print(pd.read_parquet('step1/data/runs/llm.parquet').shape)"
```

Expected: a non-empty two-dimensional LLM run table.

### Task 2: Profile the full parquet and reconcile Phase 1

**Files:**
- Create: `step1/tests/test_phase15_audit.py`
- Create: `step1/scripts/audit_full_parquet.py`
- Create: `step1/analysis/full_parquet_audit.json`
- Create: `step1/analysis/leaderboard_comparison.json`
- Create: `step1/docs/full_field_dictionary.md`

**Interfaces:**
- Produces: `load_llm_parquet(path)`, `classify_field(name)`, `candidate_key_columns(columns)`, `profile_full_runs(df)`, `compare_with_phase1(df, compact_dir)`.

- [x] **Step 1: Write failing tests for classification and reconciliation rules**

Tests must assert that `avg_output_len` is leakage, `max_num_seqs` is pre-execution, energy/power fields are targets, identity checks flag a deliberately corrupted row, and stable-only Phase 1 matching excludes unstable rows.

- [x] **Step 2: Run tests and confirm the expected import/function failures**

```powershell
python -m unittest step1.tests.test_phase15_audit -v
```

- [x] **Step 3: Implement the minimum full-parquet audit**

Profile row/column counts, dtypes, nulls, exact duplicates, candidate-key multiplicity, category coverage, stable/unstable counts and reasons, numeric ranges and the independent `energy / duration = power` check. Reconcile schema, stable counts and candidate-key duplicates against the Phase 1 compact JSON.

- [x] **Step 4: Run tests and generate reviewed JSON outputs**

```powershell
python -m unittest step1.tests.test_phase15_audit -v
python step1/scripts/audit_full_parquet.py
```

### Task 3: Select bounded representative runs and fetch metadata only

**Files:**
- Modify: `step1/tests/test_phase15_audit.py`
- Create: `step1/scripts/select_representative_runs.py`
- Create: `step1/analysis/selected_runs.json`
- Create locally, ignored: `step1/data/selected_runs_raw/**/results.json`

**Interfaces:**
- Produces: `select_representative_runs(df, limit=12)` and a manifest containing task, GPU, architecture, model scale, card count, batch/parallel settings, results path, selection reasons and hashes.

- [x] **Step 1: Write failing selection tests**

Assert a maximum of 12 rows, deterministic output, coverage of all three tasks and both GPUs where the source supports them, Dense and MoE, one-card and multi-card configurations, and rejection of `prometheus_path` as a download target.

- [x] **Step 2: Implement deterministic coverage-first selection**

Score candidates against uncovered task/GPU/architecture/scale/card-count facets, break ties by stable status then candidate key, and document any impossible facet.

- [x] **Step 3: Read only the pre-timeline metadata prefix of selected `results_path` files**

Use token through hidden process input; fetch no `prometheus_path` or model weight. Because `results.json` embeds timeline data, stop reading at the `timeline` key, retain only the configuration/summary prefix, save a source manifest, and reject paths outside allowed result metadata.

- [x] **Step 4: Verify bounded acquisition**

Check file count is 6–12, each file is JSON, every hash is recorded, and zero filenames/paths contain `prometheus`, `timeline`, model weight extensions or Diffusion/Training domains.

### Task 4: Audit workload fields and construct the `X_pre` boundary

**Files:**
- Modify: `step1/tests/test_phase15_audit.py`
- Create: `step1/scripts/audit_workload_metadata.py`
- Create: `step1/analysis/workload_field_audit.json`
- Create: `step1/docs/workload_field_matrix.md`
- Create: `step1/docs/x_pre_feature_catalog.md`

**Interfaces:**
- Produces: `find_workload_fields(result_json)`, `classify_workload_value(path, value, evidence)`, per-task availability and one of `pre_known`, `pre_derivable`, `target`, `post_run_leakage`, `diagnostic`.

- [x] **Step 1: Write failing tests using small synthetic metadata fixtures**

Tests must keep `max_tokens`/planned request count as pre-known, actual output tokens/throughput/batch/duration as leakage, and mark prompt length pre-derivable only when the fixed request set and tokenizer are known before launch.

- [x] **Step 2: Implement recursive field discovery with explicit evidence**

Record JSON path, example type/value summary, task coverage, whether it is configured before launch, whether it can be derived before launch, future modeling permission and the exact reason.

- [x] **Step 3: Reconcile result metadata with public benchmark workload/config source**

Use fixed official Git commits and task definitions to determine input dataset selection, request count, output budget, sampling settings, runtime configuration and whether tokenization is model-dependent but executable before the benchmark.

- [x] **Step 4: Generate the workload matrix and `X_pre` catalog**

Separate raw pre-known variables, pre-derived physics variables, targets, forbidden post-run variables and diagnostics. Do not calculate feature/target correlations.

### Task 5: Build the reproducible notebook and technical report

**Files:**
- Create: `step1/notebooks/phase1_5_complete_audit.ipynb`
- Create: `step1/report_app/**`
- Create: `step1/reports/phase1_5_audit_summary.md`
- Create: `step1/reports/verification_receipt.json`

**Interfaces:**
- Consumes: reviewed JSON outputs only; never embeds the token or gated raw records.
- Produces: executed notebook, local report `step1/report_app/dist/index.html`, concise Markdown handoff and A/B/C decision.

- [x] **Step 1: Create notebook sections and execute top-to-bottom**

Use `tl;dr`, Context & Methods, Data, Results and Takeaways. Include bounded coverage/stability charts and exact lookup tables; save all outputs and verify zero error cells.

- [x] **Step 2: Build one technical report app from reviewed outputs**

Lead with the A/B/C decision, then show parquet-versus-leaderboard differences, stability/duplicate evidence, selected-run coverage, per-task workload matrix, `X_pre` catalog, limitations and the single most important missing variable if applicable.

- [x] **Step 3: Validate claims and render**

Recompute headline counts from source outputs, run unit tests, build the report, inspect desktop/full-page rendering, and record any unverified interaction checks.

- [x] **Step 4: Run secret and scope checks**

Search tracked files for `hf_`, token-shaped strings, Prometheus/timeline payloads, model weight extensions and out-of-scope domain data. Confirm gated raw files are ignored.

- [x] **Step 5: Commit without pushing**

```powershell
git add .
git diff --cached --check
git commit -m "audit ML.ENERGY V3 full LLM summary and workload metadata"
```

Do not configure a remote or push. Stop after the A/B/C decision.
