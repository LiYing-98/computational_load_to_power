# Phase 5 hierarchical GPU power modeling implementation plan

> Execution: native inline implementation with durable progress, focused tests, and one independent whole-branch review at completion.

Completion audited2026-10-14: all five tasks below fulfilled (checkboxes retain original plan text as historical requirements); see reports/progress.md final record, final_code_review.md and expanded9/9 verification receipt.20 Step7 tests pass. L2X correction is separately frozen and selective, not an unrecorded plan change.

**Goal:** Quantify additional ex-ante information and total/per-GPU target effects with independent per-algorithm nested model CV and paired matched-support comparisons.

**Architecture:** Read frozen Step3/5/6 sources; derive and audit Step7 features; define strict support manifests; freeze finite candidates and folds before training; checkpoint each algorithm/level/target/fold; derive comparisons from persisted OOF; fit labeled development candidates before post-hoc stresses; build reports, notebook, figures, and independent validator.

**Stack:** Python 3.12, pandas, numpy, pyarrow, sklearn, CatBoost, XGBoost, matplotlib, nbformat/nbclient/nbconvert.

**Spec:** D:/my_work/cpecc/files/日常事务/科技项目/2025年/算电协同/课题一/阶段性研究进展/plan/first_step/Codex_Phase5_Taskbook.docx

## Global constraints

- Do not modify Step1–6 outputs; only stable GPU-side text LLM inference; do not download new raw timelines.
- 565 unique stable runs expected, verified from sources; higher supports computed from required semantic fields.
- Identity/lineage/subfamily/targets/post-run/task labels are excluded from primary X.
- Every algorithm has its own inner GroupKFold; G-perGPU selection scores reconstructed total watts.
- Preprocessors and any target-dependent choice use train only.
- Matched-support comparisons share exact run keys and outer fold assignments.
- Existing Phase4 Confirmatory is not a new blind holdout; best-of-level OOF has winner selection bias.

## Review focus

- Unsupported MLA/hybrid formulas or structural N/A silently imputed as unknown.
- G-perGPU inner selection using per-GPU rather than total MAE.
- Support shrinkage falsely claimed as feature gain or folds differing within pairs.
- Cache fingerprints omitting groups, order, candidate/bundle/code/version changes.
- Post-hoc stress performance feeding candidate selection or false confirmatory claims.

## Tasks and interfaces

### 1. Prior evidence, sources, semantic master table (7.0–7.3)

- [ ] Write and run failing tests for frozen-source integrity, 565/27 grain, ex-ante boundaries, hand-checked deployment/physics formulas, N/A flags and strict eligibility.
- [ ] `scripts/build_features.py` produces master parquet, feature hierarchy/formulas, semantic coverage tables, request-text audit and coverage manifest.
- [ ] `reports/prior_evidence_summary.md` is written before model implementation/training with exact historical source links.
- [ ] Verify tests and source hashes, commit only step7.

### 2. Freeze candidates/folds and algorithm-isolated nested engine (7.4)

- [ ] Write failing tests for deterministic candidate counts/folds, group isolation, total-space G scoring, unseen categories and robust cache invalidation.
- [ ] `scripts/modeling.py` and `scripts/freeze_protocol.py` produce sampled candidates, five-fold support assignments and protocol hash; freeze before any outer results.
- [ ] Use the same uniformly reduced 8-candidate rule per nonbaseline algorithm across every level/target, declared before training. This is a bounded exploration with higher search uncertainty than the taskbook's default 24; no adaptive expansion after results.
- [ ] Verify focused tests and commit.

### 3. Main/matched-support/sensitivity matrix (7.5–7.8)

- [ ] `scripts/run_matrix.py` produces OOF predictions for L0/L0A/L1/L2/L3/L4, both targets, all eight algorithms, matched lower-on-higher support, and eligible sensitivity counterparts.
- [ ] Each checkpoint fingerprints canonical rows/groups, resolved features, complete folds, sampled candidates, objectives, code, versions, and protocol.
- [ ] Failures remain explicit; no partial-cell metric reported as complete. Resume completed folds only after fingerprint verification.
- [ ] Verify no group leakage, exact support and reconstructed total predictions; commit compact outputs.

### 4. Paired statistics, candidate fits and post-hoc stresses (7.6–7.9)

- [ ] `scripts/analyze_results.py` computes total/per-GPU metrics, adjacent increments, target comparisons and >=1000 model-cluster draws with row-weighted/equal-model CIs.
- [ ] `scripts/fit_candidates.py` locks L0/L1/L2 development choices, fits by 3-fold group CV, saves feature manifest/model/hash with non-external-validation label.
- [ ] `scripts/run_stress.py` consumes locked candidates and exact persisted lineage/within-task/architecture manifests; no return to selection.
- [ ] Verify independently reconciled pairs, target scales, slices and stress support.

### 5. Report/notebook/figures and independent verification (7.10)

- [ ] Build six required figure families from CSV sources and inspect figures.
- [ ] Produce the eight RQ answers with matched-support evidence, budget/coverage/selection caveats and candidate information requirements.
- [ ] Execute a read-only evidence notebook and export HTML.
- [ ] Independent `scripts/validate_phase50.py` checks source/split/candidate/OOF/formula/metrics/pairs/forbidden inputs, secrets and required artifacts; save receipt.
- [ ] Run focused/full Step7 tests and relevant readonly prior checks, then one whole-branch review; repair consequential findings without retuning from stresses.

## Decisions made before training

User supplied a complete scientific specification and authorized implementation; carry routine engineering choices forward without further approval stages. Work in the requested normal checkout on `codex/mlenergy-phase5-0`; do not create a separate delivery path. Uniform 8-candidate search (versus default 24) keeps the expanded matrix tractable; disclose search limitations. Tokenizer-free branch is audit-only if frozen request lists cannot cover all three tasks reliably.
