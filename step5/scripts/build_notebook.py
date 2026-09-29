#!/usr/bin/env python3
"""Build and execute the read-only Phase 3.0 evidence notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from step5.scripts.common import ROOT


def notebook_blueprint() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (Phase 3.0)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            """# ML.ENERGY Phase 3.0 — hardware-aware high-load GPU inference modeling

## tl;dr

- In an outcome-informed exploratory comparison on the 305-run physics-complete cohort, **B4-Hardware + XGBoost** has model-holdout MAE **204.4 W**, RMSE **370.6 W**, R² **0.960**, and MdAPE **8.6%**.
- Relative to the historical B0-Core + HGB control, its observed model-holdout MAE difference is **−48.9 W**. This is not a winner-adjusted confirmatory estimate because the candidate was followed up after inspecting outer-test results.
- B5 normalized pressure proxies do **not** improve the exploratory XGBoost candidate: MAE rises to **241.7 W**. B4 shows learner-dependent signal, but the two-GPU support cannot establish broad hardware scaling laws.
- Difficulty matters more than headline random-split accuracy: B4 + XGBoost averages **97.2 W** MAE over ten random splits, reaches **94.7 W** for config holdout and **204.4 W** for model holdout, then degrades to **736.7 W** for family holdout.
- Energy/token remains harder: B1 + XGBoost gives **0.620 J/token** MAE and R² **0.677**; log1p removes the single negative raw prediction but slightly worsens MAE to **0.650 J/token**.

This notebook is a read-only companion to frozen result artifacts. It performs no model fitting.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Scope and provenance

Only ML.ENERGY Benchmark V3 **GPU-side text-LLM inference** is in scope. Inputs are known before execution; post-run batch/load, throughput, latency, duration, utilization, and targets are diagnostic-only. The population represents **high-load steady-state / near-saturation** runs, not arbitrary request-rate dynamics.

Pinned inputs and training provenance are recorded in `step5/config/source_freeze_phase3.json` and `step5/analysis/training_provenance.json`. Hardware-profile evidence is recorded in `step5/config/hardware_source_manifest.json`; H100/B200 SKU mappings remain explicitly `ambiguous`.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
assert (ROOT / "step5").exists(), "Execute from the repository root"
STEP5 = ROOT / "step5"
REPORTS = STEP5 / "reports"
FIGURES = REPORTS / "figures"

comparison = pd.read_csv(REPORTS / "model_comparison.csv")
ablation = pd.read_csv(REPORTS / "feature_ablation.csv")
generalization = pd.read_csv(REPORTS / "generalization_ladder.csv")
hardware = pd.read_csv(REPORTS / "hardware_label_sensitivity.csv")
hardware_pairs = pd.read_csv(REPORTS / "hardware_paired_comparisons.csv")
strategies = pd.read_csv(REPORTS / "target_strategy_comparison.csv")
slices = pd.read_csv(REPORTS / "error_slices.csv")
importance = pd.read_csv(REPORTS / "feature_importance.csv")
coverage = pd.read_csv(STEP5 / "analysis/physics_coverage_summary.csv")
pca = pd.read_csv(STEP5 / "analysis/physics_pca_summary.csv")
source_freeze = json.loads((STEP5 / "config/source_freeze_phase3.json").read_text(encoding="utf-8"))
provenance = json.loads((STEP5 / "analysis/training_provenance.json").read_text(encoding="utf-8"))

pd.Series({
    "pinned inputs": len(source_freeze["frozen_inputs"]),
    "core experiments": provenance["completed_experiments"],
    "core OOF rows": provenance["prediction_rows"],
    "physics cohort rows": int(hardware["n_rows"].iloc[0]),
    "hardware sensitivity rows": len(hardware),
}).to_frame("value")
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Physics coverage

The B3/B5 physics quantities are not constant, but their geometry is low-dimensional. On log1p-transformed, standardized values, PCA components 1–3 explain 82.1% and components 1–5 explain 95.8% of variance. Hardware-static fields have only two support values because the benchmark contains only H100 and B200 labels. This limits identifiability even when a hardware feature improves prediction.
"""
        ),
        nbf.v4.new_code_cell(
            """display(pca.head(8).round(4))
display(
    coverage.sort_values("log10_range")
    [["feature", "feature_group", "n_unique", "p5", "median", "p95", "cv", "log10_range"]]
    .head(10).round(4)
)
display(Image(filename=str(FIGURES / "physics_coverage_by_gpu_family.png"), width=1000))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Algorithm and feature matrix

The core matrix contains 120 matched experiments: two targets × six feature bundles × ten algorithms, all on the same 305 runs and frozen five model-group folds. Hyperparameters were selected only inside each outer-training partition. B4 is included for every learner, so B3→B4→B5 can be inspected without learner-specific omission. Ranking these outer-test results remains exploratory; they are not used for an unbiased winner estimate.
"""
        ),
        nbf.v4.new_code_cell(
            """power = comparison.loc[comparison["target"].eq("y_gpu_avg_power_watts")]
energy = comparison.loc[comparison["target"].eq("y_gpu_energy_per_token_joules")]
display(power.sort_values("mae")[["bundle", "model_name", "mae", "rmse", "r2", "mdape_percent", "mae_ci_low", "mae_ci_high"]].head(12).round(3))
display(energy.sort_values("mae")[["bundle", "model_name", "mae", "rmse", "r2", "mdape_percent", "mae_ci_low", "mae_ci_high"]].head(8).round(3))

b3_increment = ablation.loc[
    ablation["target"].eq("y_gpu_avg_power_watts")
    & ablation["reference_bundle"].eq("B2b-Effective")
]
display(b3_increment[["model_name", "mae_reference", "mae_candidate", "mae_delta_candidate_minus_reference", "equal_model_mae_delta_ci_low", "equal_model_mae_delta_ci_high"]].sort_values("mae_delta_candidate_minus_reference").round(3))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Generalization ladder

Random/config results describe interpolation or known-model configuration transfer. Model holdout is the primary scientific estimate for unseen models. Family holdout is a stress test and exposes the major unresolved failure mode.
"""
        ),
        nbf.v4.new_code_cell(
            """ladder = (
    generalization.groupby(["model_variant", "split_scheme"], as_index=False)
    .agg(runs=("mae", "size"), mae_mean=("mae", "mean"), mae_std=("mae", "std"), r2_mean=("r2", "mean"))
)
display(ladder.round(3))
for name in (
    "actual_vs_predicted_random_best.png",
    "actual_vs_predicted_model_holdout_best.png",
    "actual_vs_predicted_family_holdout_best.png",
):
    display(Image(filename=str(FIGURES / name), width=1000))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Hardware and target strategies

B4 continuous hardware fields show a suggestive observed improvement for the outcome-informed XGBoost follow-up, but its B3→B4 paired interval crosses zero. B5 workload/hardware pressure ratios degrade that candidate. Removing the categorical GPU label from B5 changes MAE only modestly and with an interval spanning zero; with only two hardware classes, continuous fields cannot be interpreted as validated cross-GPU scaling laws. The rated-power-fraction target also underperforms direct B4 power prediction.
"""
        ),
        nbf.v4.new_code_cell(
            """display(hardware.round(3))
display(hardware_pairs.round(3))
display(strategies.round(3))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Error analysis and interpretation

Fold-local permutation importance and PDP use models fitted only on each outer-training fold. SHAP was unavailable in the frozen environment. Aggregate rated power, GPU count, and model parameter count dominate B4 importance, but wide fold-to-fold error bars and separated PDP curves show instability. Load-regime bins use post-run information for diagnosis only and never enter X.
"""
        ),
        nbf.v4.new_code_cell(
            """best_importance = (
    importance.loc[importance["model_variant"].eq("B4-Hardware__XGBoost")]
    .groupby("feature", as_index=False)["importance_mae_increase_mean"].mean()
    .sort_values("importance_mae_increase_mean", ascending=False)
)
display(best_importance.head(15).round(3))
display(slices.loc[slices["slice_type"].isin(["gpu_model", "model_family", "load_regime"])].round(3))
for name in (
    "residuals_model_holdout_best.png",
    "actual_vs_predicted_energy_best.png",
    "residuals_energy_best.png",
    "feature_importance_best.png",
    "partial_dependence_best.png",
):
    display(Image(filename=str(FIGURES / name), width=1000))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## RQ1–RQ5 conclusions

1. **RQ1 — algorithm ceiling:** The matrix can identify the lowest observed outer-test error, but it cannot provide an unbiased algorithm ceiling because the same outer-test results were inspected across many candidates. B4 + XGBoost is retained only as an outcome-informed exploratory candidate (204.4 W MAE, R² 0.960).
2. **RQ2 — why B3 did not help reliably:** XGBoost alone gains 16.8 W from B2b→B3, with a paired interval spanning zero; most other learners worsen. Combined with PCA concentration and correlated proxies, this points to representation/redundancy and limited support, not simply an HGB limitation.
3. **RQ3 — hardware physics:** B4 shows a 33.5 W row-weighted improvement over B3 for exploratory XGBoost, but the equal-model paired interval includes zero. This is suggestive, not stable evidence of incremental value. B5 worsens B4 for that candidate; two GPU points do not identify portable hardware effects.
4. **RQ4 — degradation with difficulty:** The exploratory B4 + XGBoost candidate moves from 97.2 W random mean to 94.7 W config, 204.4 W model, and 736.7 W family MAE. Config has 303 groups for 305 rows (301 singleton groups), so it is nearly run-level and not a strong distinct generalization tier.
5. **RQ5 — applicability:** conclusions support ex-ante prediction of aggregate GPU-side power and energy/token for this high-load steady-state benchmark population. They do not support arbitrary request-rate P(λ), transient power, unseen accelerator families, CPU/DRAM/node residuals, training, or diffusion.

Stop condition: this notebook closes Phase 3.0. Further algorithms, datasets, and dynamic-load modeling require a separately reviewed next phase.
"""
        ),
    ]
    return notebook


def build_and_execute(project_root: Path = ROOT) -> Path:
    project_root = Path(project_root).resolve()
    notebook_path = project_root / "step5/notebooks/phase3_enhanced_modeling.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = notebook_blueprint()
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(notebook, notebook_path)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(notebook)
    notebook_path.with_suffix(".html").write_text(html, encoding="utf-8")
    return notebook_path


def main() -> None:
    path = build_and_execute(ROOT)
    print(f"executed notebook: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
