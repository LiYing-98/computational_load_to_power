# ML.ENERGY Phase 3.0

This directory contains the completed hardware-aware modeling study for ML.ENERGY Benchmark V3 GPU-side, high-load steady-state text-LLM inference.

## Review first

- Scientific report: `reports/phase3_enhanced_modeling_report.md`
- Executed evidence notebook: `notebooks/phase3_enhanced_modeling.ipynb`
- Exploratory hardware sensitivity: `reports/hardware_label_sensitivity.csv`
- Auditable paired hardware comparisons: `reports/hardware_paired_comparisons.csv`
- Core 120-experiment matrix: `reports/model_comparison.csv`
- Generalization ladder: `reports/generalization_ladder.csv`
- Row-level OOF predictions: `analysis/oof_predictions.parquet`
- Hardware/source provenance: `config/hardware_source_manifest.json`

## Scope

- Targets: aggregate steady-state GPU average power (W) and GPU energy per generated token (J/token).
- Formal X: execution-time-known B0–B5 features only.
- Primary scientific split: deterministic model-group holdout.
- Stress ladder: ten repeated random 80/20 splits, config, model, and family holdout.
- Excluded: post-run telemetry as X, arbitrary arrival-rate modeling, Training, Diffusion, CPU/DRAM/node targets, full raw timelines, and deployment claims.

## Reproduce

Run from the repository root in PowerShell. Phase 3 uses the frozen Phase 2 environment plus isolated optional wheels under ignored `step5/.packages`.

```powershell
& 'step4\.venv\Scripts\python.exe' -m pip install --target step5/.packages -r step5/requirements-phase30.txt
& 'step4\.venv\Scripts\python.exe' -m step5.scripts.build_modeling_table
& 'step4\.venv\Scripts\python.exe' -m step5.scripts.analyze_physics_coverage
& 'step4\.venv\Scripts\python.exe' -m step5.scripts.build_splits
& 'step4\.venv\Scripts\python.exe' -m step5.scripts.run_model_matrix
& 'step4\.venv\Scripts\python.exe' -m step5.scripts.analyze_phase3
& 'step4\.venv\Scripts\python.exe' -m step5.scripts.build_notebook
& 'step4\.venv\Scripts\python.exe' -m step5.scripts.validate_phase30
& 'step4\.venv\Scripts\python.exe' -m unittest discover -s step5/tests -v
```

The independent validator hashes the actual Phase 3 taskbook DOCX. On a different
machine, set `MLENERGY_PHASE3_TASKBOOK` to that file's absolute path before running
the validator; the expected SHA-256 remains frozen in `source_freeze_phase3.json`.

The model and analysis runners checkpoint completed cells under ignored directories. Checkpoint identity includes the source table, feature definitions, search spaces, folds, modeling/runner code, Python version, and estimator backends; a change to any of these inputs forces refitting.

## Main result

B4-Hardware + XGBoost is retained as an **outcome-informed exploratory candidate**, with model-holdout MAE 204.4 W and R² 0.960 on the 305-run physics cohort. It was followed up after inspecting outer-test results, so this is not an unbiased winner estimate or winner-adjusted interval. Family-holdout MAE is 736.7 W, and the data contain only two GPU labels; the result does not support general cross-family or cross-accelerator deployment.
