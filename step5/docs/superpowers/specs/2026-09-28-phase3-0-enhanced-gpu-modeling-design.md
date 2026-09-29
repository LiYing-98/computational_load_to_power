# Phase 3.0 enhanced GPU modeling design

## Authority and scope

This design records the user-approved Phase 3.0 taskbook at SHA-256
`DDCCAF572D291B6D454351C18FDEAA6961C9570A5B5B37591CF00F6F805C4052`.
The taskbook is the scientific specification; this document is an implementation
interpretation. The user's direct request authorizes execution in `step5/`.

The study remains limited to ML.ENERGY Benchmark V3 GPU-side text-LLM inference.
It predicts high-load/near-saturation steady-state aggregate GPU power and GPU
energy/token using only execution-time-known inputs. Post-run utilization,
actual batch size, throughput, latency, output tokens, duration, and targets may
be used only for diagnostics. `step3/` and `step4/` are frozen inputs.

## Design decisions

1. Reuse the Phase 2 modeling table, feature bundle definitions, and frozen
   model-group/config-group folds after hash and contract verification.
2. Resolve `H100` and `B200` to the best evidence-supported benchmark hardware
   profiles. Every field carries a source URL, access date, interpretation, and
   `exact`, `benchmark-supported`, or `ambiguous` status. Ambiguity is disclosed
   and does not halt modeling.
3. Preserve B0-B3 exactly. Add B4 continuous hardware fields and B5 normalized
   workload/hardware pressure proxies. The primary B4/B5 variants retain the GPU
   label; a hardware-only sensitivity removes it.
4. Build one leakage-safe model runner. Hyperparameter selection happens only
   within each outer-training partition using group-aware inner validation.
   Deterministic bounded candidate lists replace unconstrained AutoML.
5. Run the required algorithm-by-feature matrix on the identical Physics cohort
   and frozen model folds. Third-party XGBoost/CatBoost are preferred; if they
   cannot be installed reproducibly, record the reason and use the documented
   nearest available boosting alternative without stopping.
6. Select the best primary power model by model-holdout MAE only. Use it, plus
   B0/HGB historical controls, for Random, Config, Model, and Family evaluation.
7. Treat theoretical time and capacity quantities as lower-bound/pressure proxies,
   never as observed execution time or utilization. The power-fraction strategy
   predicts measured power divided by aggregate rated power and reconstructs W.
8. Produce static PNG figures because the taskbook explicitly requires notebook
   figures and files under `reports/figures/`; no separate report web app is part
   of the requested output.

## Statistical protocol

- Primary selection metric: MAE in original units; also report RMSE, R², MdAPE,
  and sMAPE.
- Primary uncertainty: model-cluster bootstrap for model-holdout results.
- Feature comparisons: identical `run_key` and fold support, with paired
  model-cluster deltas.
- Random split: canonical 80/20 seed plus ten deterministic repeated seeds.
- Energy/token: raw and log1p strategies evaluated in J/token; clipping, if shown,
  is sensitivity-only.
- Physical validity: count and expose negative/non-finite predictions before any
  optional sensitivity transformation.

## Deliverables

The taskbook's named configuration, modeling, diagnostics, prediction, result,
figure, notebook, and report artifacts are mandatory. Tests verify source hashes,
feature leakage boundaries, identical ablation support, split isolation, nested
selection behavior, metric units, artifact schemas, notebook execution, and the
final receipt.

