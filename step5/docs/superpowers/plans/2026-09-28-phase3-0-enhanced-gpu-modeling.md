# Phase 3.0 enhanced GPU modeling implementation plan

> **For Codex:** Use `superpowers:executing-plans` and follow this plan task by
> task. Tests are written and observed failing before implementation. Record
> rulings and task completions in the plan workspace ledger.

**Goal:** Deliver a reproducible Phase 3.0 study of stronger high-load LLM
inference GPU power/energy prediction with auditable hardware-physics features.

**Architecture:** Extend, but never mutate, the frozen Phase 2 artifacts. A
Phase 3 source contract validates hashes, an enrichment layer joins hardware
profiles and derives B4/B5, a unified nested-CV runner emits long-form predictions,
and analysis/report builders consume only those frozen predictions.

**Tech stack:** Python 3.12, pandas/pyarrow, NumPy/SciPy, scikit-learn,
Matplotlib, optional XGBoost/CatBoost/SHAP, nbformat/nbconvert, unittest.

**Spec:** `step5/docs/superpowers/specs/2026-09-28-phase3-0-enhanced-gpu-modeling-design.md`

## Global constraints

- Write all new files under `step5/`; do not alter `step3/` or `step4/`.
- Formal X contains execution-time-known information only.
- Scope is GPU-side text LLM inference at high-load steady state.
- Post-run values are diagnostic-only and never appear in a feature bundle.
- All algorithm/bundle ablations use identical Physics-cohort rows and folds.
- Do not push, publish, or extend to Training, Diffusion, CPU/DRAM, node residual,
  arbitrary request-rate modeling, or new datasets.

## Task 1: Freeze inputs and hardware provenance

**Files:**
- Create: `step5/config/source_freeze_phase3.json`
- Create: `step5/config/gpu_hardware_specs.json`
- Create: `step5/config/hardware_source_manifest.json`
- Create: `step5/config/feature_bundles_phase3.json`
- Create: `step5/config/model_search_spaces.json`
- Create: `step5/requirements-phase30.txt`
- Test: `step5/tests/test_phase30.py`

1. Add failing tests that verify all Phase 2 input hashes, B0-B3 identity, required
   B4/B5 fields, allowed source-status values, forbidden post-run features, and
   bounded deterministic search spaces.
2. Run the focused tests and confirm failures are due to missing Phase 3 modules/config.
3. Research official NVIDIA/benchmark evidence and add provenance-rich H100/B200
   profiles without claiming a more specific SKU than the evidence supports.
4. Implement the Phase 3 source/config loader and contract validation.
5. Run the focused tests, then the complete existing suite; commit.

## Task 2: Build the Phase 3 table and characterize load regime

**Files:**
- Create: `step5/scripts/common.py`
- Create: `step5/scripts/build_modeling_table.py`
- Create: `step5/analysis/phase3_modeling_table.parquet`
- Create: `step5/analysis/phase3_modeling_table.csv`
- Create: `step5/reports/load_regime_characterization.md`
- Create: `step5/reports/figures/load_regime_distribution.png`
- Modify: `step5/tests/test_phase30.py`

1. Add failing tests for a one-to-one 694-run join, preserved Phase 2 columns and
   targets, B4/B5 formulas and units, profile ambiguity columns excluded from X,
   cohort counts, and diagnostic-only load fields.
2. Confirm RED, then implement deterministic enrichment and load diagnostics.
3. Verify hand-calculated H100/B200 rows and zero/invalid denominator behavior.
4. Run tests, hash generated outputs, and commit.

## Task 3: Physics coverage and identifiability

**Files:**
- Create: `step5/scripts/analyze_physics_coverage.py`
- Create: `step5/analysis/physics_coverage_summary.csv`
- Create: `step5/analysis/physics_correlations_pearson.csv`
- Create: `step5/analysis/physics_correlations_spearman.csv`
- Create: `step5/analysis/physics_pca_summary.csv`
- Create: `step5/analysis/physics_residual_associations.csv`
- Create: `step5/reports/figures/physics_coverage_by_gpu_family.png`
- Modify: `step5/tests/test_phase30.py`

1. Add failing tests for required quantiles/CV/range, finite-safe ratios, symmetric
   correlations, deterministic standardized PCA, and residual/support alignment.
2. Confirm RED, implement the compact diagnostic, and generate tables/figure.
3. Reconcile at least two summary values independently from the modeling table.
4. Run tests and commit.

## Task 4: Unified nested-CV model runner

**Files:**
- Create: `step5/scripts/modeling.py`
- Create: `step5/scripts/build_splits.py`
- Create: `step5/splits/family_group_folds.csv`
- Create: `step5/splits/random_split_manifest.csv`
- Create: `step5/splits/split_manifest_phase3.json`
- Create: `step5/models/model_manifest.json`
- Modify: `step5/tests/test_phase30.py`

1. Add failing tests for deterministic model construction, unseen categories,
   nested group-aware selection, no outer-test access, model/config/family group
   isolation, canonical/repeated random splits, raw/log1p inverse units, and
   power-fraction reconstruction.
2. Confirm RED, implement DummyMean, Ridge, ElasticNet, RandomForest, ExtraTrees,
   HGB, XGBoost/CatBoost when available, spline/GAM substitute, and small MLP.
3. Keep candidate counts bounded and record dependency substitutions verbatim.
4. Run tests and commit.

## Task 5: Execute the algorithm-by-feature matrix

**Files:**
- Create: `step5/scripts/run_model_matrix.py`
- Create: `step5/analysis/oof_predictions.parquet`
- Create: `step5/analysis/oof_predictions.csv`
- Create: `step5/analysis/training_provenance.json`
- Create: `step5/reports/model_comparison.csv`
- Create: `step5/reports/feature_ablation.csv`
- Create: `step5/reports/prediction_validity.csv`
- Modify: `step5/tests/test_phase30.py`

1. Add failing artifact-contract tests for the complete required algorithm/bundle
   matrix, one prediction per run/experiment, identical paired support, nested
   tuning metadata, original-unit metrics, model-cluster intervals, and physical
   validity counts.
2. Confirm RED, execute the frozen matrix, write checkpoints atomically, and resume
   by experiment ID so interrupted work is not repeated.
3. Analyze predictions into comparison/ablation/validity tables without selecting
   on outer-test data. The final matrix contains B0/B1/B2b/B3/B4/B5 for all ten
   learners and both targets (120 matched experiments).
4. Run tests and commit.

## Task 6: Generalization, sensitivities, interpretation, and figures

**Files:**
- Create: `step5/scripts/analyze_phase3.py`
- Create: `step5/reports/generalization_ladder.csv`
- Create: `step5/reports/target_strategy_comparison.csv`
- Create: `step5/reports/error_slices.csv`
- Create: `step5/reports/feature_importance.csv`
- Create: `step5/reports/partial_dependence.csv`
- Create figures under `step5/reports/figures/`
- Modify: `step5/tests/test_phase30.py`

1. Add failing tests for Random/Config/Model/Family coverage, ten repeated random
   seeds, B0 and an explicitly exploratory candidate, power-fraction and energy raw/log1p results,
   fold-local permutation importance, 4-6 PDP variables, and required plots.
2. Confirm RED, run the generalization ladder and sensitivities, then compute
   error slices and explainability without fitting on held-out outcomes.
3. Generate equal-axis measured-vs-predicted plots with y=x, residual plots,
   GPU encoding, and captions containing n/MAE/RMSE/R²/MdAPE.
4. Visually inspect every figure for labels, scales, clipping, and truthful scope;
   run tests and commit.

## Task 7: Reproducible notebook and scientific report

**Files:**
- Create: `step5/scripts/build_notebook.py`
- Create: `step5/notebooks/phase3_enhanced_modeling.ipynb`
- Create: `step5/notebooks/phase3_enhanced_modeling.html`
- Create: `step5/reports/phase3_enhanced_modeling_report.md`
- Create: `step5/README.md`
- Modify: `step5/tests/test_phase30.py`

1. Add failing tests for notebook section order, no hidden model training in the
   notebook, source/provenance references, and report coverage of RQ1-RQ5,
   uncertainty, limits, and the high-load-only claim.
2. Confirm RED, build the notebook as an inspectable reader of frozen outputs,
   execute it top-to-bottom, render HTML, and inspect saved figures/tables.
3. Write the answer-first technical report from verified tables, explicitly
   separating observation, interpretation, and limitation.
4. Run tests and commit.

## Task 8: Final validation and receipt

**Files:**
- Create: `step5/scripts/validate_phase30.py`
- Create: `step5/reports/verification_receipt.json`
- Modify: `step5/tests/test_phase30.py`

1. Add a failing end-to-end test that independently rebuilds source hashes,
   cohort/bundle coverage, split isolation, prediction uniqueness, metric samples,
   required artifact schemas, notebook execution state, and report/figure presence.
2. Confirm RED, implement the validator, generate a hash-rich receipt, and run it.
3. Run Phase 3 tests plus the complete repository test suite and inspect git diff.
4. Request one fresh whole-branch code/science review; fix Critical/Important
   findings in one RED→GREEN pass, ledger minors/rulings, re-run the full suite,
   commit, and stop Phase 3.0.

## Review focus

- Any post-run or target-derived field leaking into formal X.
- SKU certainty overstated relative to benchmark evidence.
- Outer-test information influencing tuning or best-model selection.
- Feature ablations using different rows/folds.
- Family/config group leakage and random-split mislabeling.
- Proxy units or power-fraction reconstruction errors.
- Negative prediction handling hidden by clipping.
- Claims extending beyond two GPU labels or high-load steady state.
