# Phase5 independent whole-branch review and repair record

Review range:42d1c0a→130ee9570e85d5e833a1efbf2a6bf7e383d2c5e0. One fresh-context read-only reviewer, no implementation delegation or second review. Actual taskbook, implementation plan, source evidence, scripts, persisted results and scientific claims were inspected.

## Initial verdict

Changes requested: no Critical findings, three Important findings; no Minor findings reported.

1. `scripts/build_features.py:211–214`:276 rows/12 Qwen models explicitly disable sliding windows, but L2X marked them applicable/unknown. Raw config byte hashes exactly match the already-frozen Step3 architecture table. Median imputation could substitute windows from another architecture. Fix via a transparent, frozen, selective L2X amendment, identical support/folds/candidates/objectives, no main/candidate feedback.
2. `scripts/run_stress.py:64`:stress cache fingerprint omitted shared modeling/common/protocol dependencies. No current numerical corruption established; cache reuse contract was incomplete.
3. `scripts/validate_phase50.py:145–157`:independent metric/fold reconciliation accepted deliberately stale OOF fingerprints in an in-memory fault injection. Self-consistent outputs are not proof of current preprocessing provenance.

Strengths independently confirmed:224 completed cells; restored-total G selection; train-local preprocessing; exact matched-support/folds; paired model-cluster bootstrap; explicit winner/post-hoc limitations; Step1–6 unchanged. Original final validator passed8/8, reconciling224 OOF,6640 slices,272 comparisons/544000 draws, but missed the above semantic/provenance defects.

## Repair scope and tests

- Issue1:rule/evidence/table/code frozen in `config/l2x_semantic_amendment.json` before corrected training. Original master/protocol and superseded16OOF/original matrix preserved. Only16 L2X cells rerun;208 original cells retained; independent validator checks all unrelated table columns and main result rows unchanged. Regression `test_explicitly_disabled_qwen_windows_are_structural_not_unknown` RED→GREEN; dependent sensitivity/statistical outputs must finish before completion.
- Issue2:`stress_dependencies()` now binds modeling/common/protocol/source freeze. Regression `test_stress_cache_key_tracks_shared_engine_and_protocol` RED→GREEN; final stress provenance independently reconstructed; `test_final_validator_rejects_stale_stress_fingerprints` confirmed old validator accepted bad fingerprint (RED) and new validator rejects it (GREEN).
- Issue3:independent validator reconstructs original or amended expected fingerprints without importing the training hash helper. `test_final_validator_rejects_stale_oof_fingerprints` reproduced missing rejection (RED), then rejects stale evidence (GREEN). Original and amended stored fingerprints spot-checked against the independent implementation.
- Author visual/execution check also found notebook cwd portability issue. Saved first cell failed from `step7/notebooks` (RED), ancestor-based root resolution and regenerated/executed notebook passed (GREEN). Summary now records actual observed results, not generic methods alone.

The above is a single repair pass, not a second reviewer verdict. Numerical pipeline/final suite/receipt and final progress record provide completion evidence; no new tuning or independent-validation claim.

Final repair acceptance2026-10-14 07:12 CST: all16 L2X cells and dependent outputs completed,208 originals preserved; full20/20 tests PASS; expanded independent9/9 final validation PASS, including original/amended OOF fingerprints, stress shared-code fingerprints and unchanged candidate/main evidence. All3 Important findings resolved with no outstanding Critical/Important/Minor findings. Negative predictions in two L2X T-total linear sensitivities are disclosed and diagnosed in negative_prediction_audit.md, not clipped or deployed. HTML top visually rechecked; deep-fragment headless captures unavailable after bounded retry, so whole-page visual coverage is not claimed; all6 standalone PNGs retain previous verified appearance.

## Review limits / declined to judge

Reviewer did not independently retrain all models or run artifact-rebuilding full tests because review was read-only; author runs the full Step7 suite and independent validation. Reviewer did not independently rejudge figure appearance because author inspected all6 PNGs and rendered HTML. These limits are accepted; analytical source mappings/scientific interpretation were reviewed. No other in-scope behavior declined. No findings deferred as Minor.
