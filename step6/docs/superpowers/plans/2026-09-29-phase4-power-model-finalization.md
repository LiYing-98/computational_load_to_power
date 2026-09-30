# Phase 4.0 GPU Power Model Finalization Implementation Plan

> **Execution mode:** inline execution with an on-disk ledger, test-first implementation, evidence-based validation, and one whole-branch review at the end.

**Spec:** `D:/my_work/cpecc/files/日常事务/科技项目/2025年/算电协同/课题一/阶段性研究进展/plan/first_step/Codex_Phase4_Taskbook.docx`

**Goal:** Build and independently validate one pure-ex-ante predictor for aggregate steady-state GPU power on all 565 stable ML.ENERGY V3 text-LLM inference runs, while separating model, parent-lineage, and architecture generalization claims.

**Architecture:** Reuse the frozen Phase 3 table and hardware provenance without modifying step3–step5. Build a Phase 4 table with normalized lineage diagnostics and four explicit feature bundles, lock a y-independent confirmatory model holdout before development, run bounded group-aware model selection on the remaining models, lock one candidate, evaluate the confirmatory holdout once, then produce full-cohort OOF diagnostics, generalization stress tests, figures, report, notebook, and an independent verification receipt.

**Runtime:** Python 3.12 from `step4/.venv`, pandas, pyarrow, scikit-learn, xgboost, catboost, matplotlib, seaborn, joblib, nbformat/nbconvert.

## Global constraints

- Scope is only GPU-side, stable, text-LLM inference and target `y_gpu_avg_power_watts`.
- Formal X must be execution-time-known. Model identity, parent lineage, subfamily, task outcome, actual tokens, duration, throughput, latency, utilization, and all post-run telemetry are forbidden.
- Llama must be included in P0/P1/P2 without invented static or tokenizer values.
- Confirmatory model IDs are selected by stable hash stratified by parent lineage, never by y; their errors stay unread until the candidate is locked.
- Step3–step5 are immutable inputs. No new raw timeline download and no new hardware-spec search unless an existing numeric error is demonstrated.
- No energy/token, CPU/DRAM/node, Training, Diffusion, transient curves, or arrival-rate modeling.
- All stochastic choices use fixed seeds; every fold and prediction row is persisted.

## Task 1: Freeze inputs, taxonomy, bundles, and confirmatory holdout

**Files:**
- Create: `step6/config/source_freeze_phase4.json`
- Create: `step6/config/lineage_taxonomy.json`
- Create: `step6/config/feature_bundles_phase4.json`
- Create: `step6/config/model_search_spaces_phase4.json`
- Create: `step6/config/confirmatory_model_holdout.json`
- Create: `step6/scripts/common.py`
- Create: `step6/scripts/build_modeling_table.py`
- Test: `step6/tests/test_phase40.py`

1. Write failing contract tests for the 565-row stable cohort, 27 model IDs, 117 Llama rows, three diagnostic layers, bundle membership, forbidden-input exclusion, frozen input hashes, and deterministic y-independent holdout selection.
2. Run the focused tests and confirm failure because Phase 4 contracts do not exist.
3. Implement the minimum taxonomy/config/common/table-builder code and generate the locked holdout before any training.
4. Re-run focused tests; expect pass.
5. Commit configs, table builder, generated modeling table/coverage tables, tests, and source provenance.

## Task 2: Deterministic development and stress-test splits

**Files:**
- Create: `step6/scripts/build_splits.py`
- Create: `step6/splits/development_model_folds.csv`
- Create: `step6/splits/lineage_holdout_folds.csv`
- Create: `step6/splits/within_task_lineage_folds.csv`
- Create: `step6/splits/architecture_holdout_folds.csv`
- Modify: `step6/tests/test_phase40.py`

1. Write failing tests for no confirmatory rows in development folds, model-group isolation, lineage isolation, within-task eligibility (at least three lineages), architecture support rules, and deterministic regeneration.
2. Verify RED.
3. Implement split construction and coverage matrices.
4. Verify focused GREEN and full Phase 4 suite GREEN.
5. Commit split code and manifests.

## Task 3: Group-aware modeling engine and development matrix

**Files:**
- Create: `step6/scripts/modeling.py`
- Create: `step6/scripts/run_development_matrix.py`
- Create: `step6/analysis/development_oof_predictions.parquet`
- Create: `step6/reports/main_model_comparison.csv`
- Modify: `step6/tests/test_phase40.py`

1. Write failing tests for six algorithms, bounded search spaces, group-aware inner CV on outer-train only, unseen-category handling, deterministic predictions, complete OOF coverage, and metric formulas.
2. Verify RED.
3. Implement the minimal engine and execute 6 algorithms × P0/P1/P2 plus P2-NoLabel sensitivity on development data.
4. Verify GREEN; persist environment and training provenance.
5. Commit code and compact formal outputs; keep checkpoints ignored.

## Task 4: Hardware ablation and candidate locking

**Files:**
- Create: `step6/scripts/analyze_development.py`
- Create: `step6/reports/hardware_ablation.csv`
- Create: `step6/reports/hardware_bootstrap_draws.csv`
- Create: `step6/config/locked_candidate_phase4.json`
- Modify: `step6/tests/test_phase40.py`

1. Write failing tests for paired identical-support comparisons, row-weighted and equal-model delta MAE, 1000 model-cluster bootstrap draws, and deterministic candidate selection from development evidence only.
2. Verify RED.
3. Implement ablation and lock the best candidate using development MAE with RMSE/high-power bias guardrails documented in metadata.
4. Verify GREEN.
5. Commit evidence and lock file.

## Task 5: Directed high-power loss experiment and final candidate lock

**Files:**
- Create: `step6/scripts/run_loss_strategies.py`
- Create: `step6/analysis/loss_strategy_predictions.parquet`
- Create: `step6/reports/loss_strategy_comparison.csv`
- Modify: `step6/config/locked_candidate_phase4.json`
- Modify: `step6/tests/test_phase40.py`

1. Write failing tests for L1/L2/pseudo-Huber-if-supported/mild-weighting candidates, train-only median weighting, unweighted evaluation, and high-power slice metrics.
2. Verify RED.
3. Run only the top one or two development candidates, update the lock once from development evidence, and record the decision rule.
4. Verify GREEN.
5. Commit the final pre-confirmatory candidate lock and loss comparison.

## Task 6: One-shot confirmatory evaluation and full-cohort OOF

**Files:**
- Create: `step6/scripts/evaluate_locked_candidate.py`
- Create: `step6/analysis/confirmatory_predictions.parquet`
- Create: `step6/analysis/full_cohort_oof_predictions.parquet`
- Create: `step6/models/phase4_candidate_model.joblib`
- Create: `step6/models/phase4_candidate_metadata.json`
- Create: `step6/reports/prediction_validity.csv`
- Modify: `step6/tests/test_phase40.py`

1. Write failing tests that the lock predates predictions, confirmatory IDs were absent from development decisions, all held-out rows are predicted once, the final model uses all development rows, and no post-confirmatory retuning field exists.
2. Verify RED.
3. Fit on all development rows, evaluate the confirmatory set once, freeze predictions/metrics/model, then separately generate full-565 model-group OOF predictions for diagnostic slices only.
4. Verify GREEN and cryptographically freeze confirmatory artifacts.
5. Commit the one-shot evidence and final model.

## Task 7: Generalization, error slices, and figures

**Files:**
- Create: `step6/scripts/analyze_phase4.py`
- Create: `step6/reports/generalization_by_lineage.csv`
- Create: `step6/reports/within_task_lineage_holdout.csv`
- Create: `step6/reports/architecture_holdout.csv`
- Create: `step6/reports/error_slices.csv`
- Create: `step6/reports/figures/*.png`
- Modify: `step6/tests/test_phase40.py`

1. Write failing tests for per-lineage composition/metrics/bias, task-lineage confounding labels, supported architecture stress tests, GPU/count/power-quantile slices, calibration diagnostics, and all required figures with y=x/equal axes where applicable.
2. Verify RED.
3. Run the locked candidate across lineage, within-task-lineage, and supported architecture holdouts; generate all required diagnostic tables and figures.
4. Verify GREEN.
5. Commit analysis outputs and figures.

## Task 8: Report, executable evidence notebook, and independent validator

**Files:**
- Create: `step6/reports/phase4_power_model_report.md`
- Create: `step6/notebooks/phase4_power_modeling.ipynb`
- Create: `step6/notebooks/phase4_power_modeling.html`
- Create: `step6/scripts/build_notebook.py`
- Create: `step6/scripts/validate_phase40.py`
- Create: `step6/reports/verification_receipt.json`
- Create: `step6/README.md`
- Modify: `step6/tests/test_phase40.py`

1. Write failing tests for the required report answers, artifact set, read-only executed notebook, independent recalculation of cohort/boundary/split/prediction/metric contracts, and secret scan.
2. Verify RED.
3. Build the scientific report and notebook from persisted artifacts; implement and run the validator.
4. Run `python -m step6.scripts.validate_phase40`, `python -m unittest discover -s step6/tests -v`, and the full repository test suite. Expect all pass.
5. Commit the final documentation and verification evidence.

## Review focus

- Any route by which confirmatory labels or errors could influence bundle, algorithm, hyperparameter, objective, weighting, or threshold choices.
- Leakage through diagnostic identity fields, task outcome, or post-run telemetry.
- Incorrect pooling or overclaiming of parent-lineage/architecture holdouts with sparse or confounded support.
- Misleading hardware generalization claims from only H100/B200 categorical support.
- Paired-ablation support/fold mismatches and bootstrap clustering mistakes.
- Failure to distinguish development OOF, one-shot confirmatory performance, and full-cohort diagnostic OOF.

## Final verification commands

Run from repository root:

```powershell
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.validate_phase40
& 'step4\.venv\Scripts\python.exe' -m unittest discover -s step6/tests -v
& 'step4\.venv\Scripts\python.exe' -m unittest discover -s step5/tests -v
git status --short
```
