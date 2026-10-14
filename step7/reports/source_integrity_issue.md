# Phase5 implementation integrity issue and selective amendment

Independent review of commit130ee95 found three Important implementation defects; no Step1–6 source corruption and no Critical finding was established.

1. L2X sliding-window semantics:276 runs/12 Qwen model IDs explicitly disable sliding windows in existing config files. Each raw byte SHA256 equals `config_sha256` in the already-frozen Step3 architecture table. Phase5 incorrectly classified these as applicable-but-unknown; they must be structural zero with applicability0 and unknown0. This affects L2X only, not primary L0–L4.
2. Stress checkpoints omitted shared modeling/common/protocol hashes. Their reuse contract must include all dependencies.
3. Final OOF validation reconciled values but did not reject stale engine fingerprints. Add independent original/amended provenance verification with fault-injection regression.

Repair decision: preserve original source/master/protocol, splits, candidates and prior numerical evidence. Freeze an explicit evidence-based L2X amendment before refitting its16 cells on the same448-row support and same candidates/folds/objectives. Use an amended effective table only for L2X; the original master remains a pre-amendment frozen snapshot. Retain superseded L2X outputs and original matrix under `analysis/superseded_l2x/`. Regenerate dependent sensitivity statistics/slices and semantic documentation. Do not use corrected L2X or stress results to alter development choices. Their original lock refers to the archived original matrix; verify that all main rows/choices remain identical numerically.

No full primary retraining or historical-source edits are authorized/needed for this correction. The amendment is a semantic bug repair after review, NOT a result-based search/feature expansion; no new blind-validation claim is made. Exact evidence hashes, changed fields and effective table hash will be frozen in `config/l2x_semantic_amendment.json`.
