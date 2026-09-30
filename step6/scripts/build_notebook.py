#!/usr/bin/env python3
"""Build and execute the read-only Phase 4.0 evidence notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from step6.scripts.common import ROOT


def notebook_blueprint() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (Phase 4.0)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            """# ML.ENERGY Phase 4.0 — 565-run GPU steady-state power candidate

## tl;dr

- The locked candidate is **RandomForest + P2-HardwareStatic + L1**.
- Development model-group OOF: **272.4 W MAE**, **474.3 W RMSE**, **R² 0.941** on 449 rows / 21 models.
- One-shot confirmatory holdout: **295.3 W MAE**, **424.6 W RMSE**, **R² 0.921** on 116 rows / 6 models.
- All 565 stable runs, including 117 Llama runs, enter P0/P1/P2. The remaining failure modes are lineage extrapolation and high-power / 8-GPU underprediction.

This is a read-only evidence companion. It loads persisted artifacts and performs no fitting or model selection.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

Scope is ML.ENERGY V3 **GPU-side text-LLM inference** under high-load / near-saturation steady-state operation. The only formal target is aggregate GPU average power in watts. Formal inputs are execution-time-known. Model identity, lineage, task outcome, actual tokens, throughput, latency, duration, utilization, and other post-run telemetry are excluded from X.

Six model IDs were selected by stable hash within parent lineage before Phase 4 training and held out from every development decision. The notebook distinguishes development model-group OOF, the one-shot confirmatory evaluation, and full-cohort diagnostic OOF.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
assert (ROOT / "step6").exists(), "Execute from the repository root"
STEP6 = ROOT / "step6"
REPORTS = STEP6 / "reports"
FIGURES = REPORTS / "figures"

table = pd.read_parquet(STEP6 / "analysis/phase4_power_modeling_table.parquet")
main = pd.read_csv(REPORTS / "main_model_comparison.csv")
ablation = pd.read_csv(REPORTS / "hardware_ablation.csv")
losses = pd.read_csv(REPORTS / "loss_strategy_comparison.csv")
confirmatory = pd.read_csv(REPORTS / "confirmatory_metrics.csv")
lineage = pd.read_csv(REPORTS / "generalization_by_lineage.csv")
within_task = pd.read_csv(REPORTS / "within_task_lineage_holdout.csv")
architecture = pd.read_csv(REPORTS / "architecture_holdout.csv")
slices = pd.read_csv(REPORTS / "error_slices.csv")
static = pd.read_csv(REPORTS / "static_subset_comparison.csv")
lock = json.loads((STEP6 / "config/locked_candidate_phase4.json").read_text(encoding="utf-8"))
holdout = json.loads((STEP6 / "config/confirmatory_model_holdout.json").read_text(encoding="utf-8"))

pd.Series({
    "stable runs": len(table),
    "model IDs": table["diag_model_id"].nunique(),
    "Llama runs": table["diag_parent_lineage"].eq("Llama").sum(),
    "development runs": table["eligible_phase4_development"].sum(),
    "confirmatory runs": table["diag_is_confirmatory_model"].sum(),
}).to_frame("count")
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Data

P0-Core, P1-AggregatePower, and P2-HardwareStatic all cover the full 565-run cohort. Lineage and subfamily fields are diagnostic only; architecture class is the structural category permitted in X. Hardware fields reuse the frozen Phase 3 H100/B200 reference proxies and do not establish cross-generation physical scaling.
"""
        ),
        nbf.v4.new_code_cell(
            """coverage = pd.concat([
    table.groupby("diag_parent_lineage").agg(runs=("run_key", "size"), models=("diag_model_id", "nunique")).assign(level="parent_lineage"),
    table.groupby("diag_architecture_class").agg(runs=("run_key", "size"), models=("diag_model_id", "nunique")).assign(level="architecture_class"),
])
display(coverage.reset_index().rename(columns={"index": "group"}))
display(table.groupby("x_hardware_gpu_model").size().rename("runs").to_frame())
display(table.groupby("x_hardware_num_gpus").size().rename("runs").to_frame())
display(table.groupby("x_protocol_task").size().rename("runs").to_frame())
display(Image(filename=str(FIGURES / "target_distribution.png"), width=950))
display(Image(filename=str(FIGURES / "task_parent_lineage_heatmap.png"), width=950))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Results — development selection and hardware ablation

The main matrix uses the same 449 development rows and frozen model-group folds for six algorithms and four bundles. Candidate selection uses MAE first, with RMSE and high-power bias guardrails. P2-NoLabel remains a sensitivity analysis.
"""
        ),
        nbf.v4.new_code_cell(
            """display(
    main.loc[main["feature_bundle"].isin(["P0-Core", "P1-AggregatePower", "P2-HardwareStatic"])]
    .sort_values("mae")
    [["algorithm", "feature_bundle", "mae", "rmse", "r2", "mdape", "smape", "signed_bias"]]
    .round(3)
)
display(
    ablation.loc[ablation["algorithm"].isin(["RandomForest", "CatBoost", "Ridge", "XGBoost"])]
    [["algorithm", "comparison", "row_weighted_delta_mae", "row_weighted_ci_low", "row_weighted_ci_high", "equal_model_delta_mae"]]
    .round(3)
)
display(static.round(3))
display(Image(filename=str(FIGURES / "hardware_paired_mae_delta.png"), width=950))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Results — loss strategy and confirmatory holdout

L2, pseudo-Huber, and mild high-power weighting do not improve the overall/high-power tradeoff. The final L1 candidate was locked before confirmatory predictions were generated. Confirmatory rows are never fed back into tuning.
"""
        ),
        nbf.v4.new_code_cell(
            """display(
    losses[["strategy_id", "mae", "rmse", "r2", "top25_mae", "top25_signed_bias", "top10_mae", "top10_signed_bias", "lower75_mae"]]
    .sort_values("mae").round(3)
)
display(confirmatory.round(3))
display(pd.Series({"locked strategy": lock["selected"]["strategy_id"], "held-out models": len(holdout["held_out_model_ids"])}).to_frame("value"))
for name in ("development_actual_vs_predicted.png", "confirmatory_actual_vs_predicted.png"):
    display(Image(filename=str(FIGURES / name), width=850))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Results — generalization boundaries and high-power bias

Unseen model, unseen parent lineage, within-task unseen lineage, and unseen architecture class are different questions. These lineage and architecture stress tests use separately persisted full-565 manifests after candidate/confirmatory freezing. They include the original confirmatory models but never feed back into candidate selection or the one-shot confirmatory result. Task/lineage/architecture composition is not balanced enough for causal attribution.
"""
        ),
        nbf.v4.new_code_cell(
            """display(lineage[["diag_parent_lineage", "n_runs", "n_models", "mae", "rmse", "r2", "signed_bias"]].round(3))
display(within_task[["x_protocol_task", "diag_parent_lineage", "task_lineage_confounded", "n_runs", "n_models", "mae", "signed_bias"]].round(3))
display(architecture[["diag_architecture_class", "n_runs", "n_models", "mae", "rmse", "r2", "signed_bias"]].round(3))
display(
    slices.loc[slices["slice_type"].isin(["gpu_model", "gpu_count", "top_25_percent", "top_10_percent", "calibration"])]
    .round(3)
)
for name in ("residual_vs_measured_power.png", "top_power_quantile_bias.png", "lineage_holdout_mae_bias.png"):
    display(Image(filename=str(FIGURES / name), width=950))
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Move the main GPU-power study from the 305-run physics cohort to all 565 stable runs; retain smaller cohorts only for mechanism diagnostics.
2. Keep RandomForest + P2 + L1 as the Phase 4 candidate baseline. Its one-shot confirmatory MAE is 295.3 W, but per-lineage outcomes vary sharply.
3. Aggregate rated power is useful for some learners and explains most of the static-subset gain; the final RandomForest also benefits from the broader P2 representation.
4. Reintroducing Llama and other lineages improves coverage, not lineage robustness. DeepSeek, Llama, Qwen, 8-GPU, and top-power cases remain the priority.
5. Do not yet treat this as a general cross-accelerator or node-power model. First collect difficult lineage/high-power combinations and define extrapolation acceptance gates.
"""
        ),
    ]
    return notebook


def build_notebook(project_root: Path = ROOT) -> tuple[Path, Path]:
    root = Path(project_root).resolve()
    output_dir = root / "step6/notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "phase4_power_modeling.ipynb"
    html_path = output_dir / "phase4_power_modeling.html"

    notebook = notebook_blueprint()
    executed = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbf.write(executed, notebook_path)

    exporter = HTMLExporter()
    exporter.template_name = "lab"
    body, _ = exporter.from_notebook_node(executed)
    html_path.write_text(body, encoding="utf-8")
    return notebook_path, html_path


def main() -> None:
    notebook_path, html_path = build_notebook(ROOT)
    print(f"wrote {notebook_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
