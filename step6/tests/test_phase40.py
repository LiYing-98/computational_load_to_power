from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from step6.scripts.common import load_feature_bundles, sha256_file, verify_source_freeze


ROOT = Path(__file__).resolve().parents[2]
STEP6 = ROOT / "step6"


class TestTask1Contracts(unittest.TestCase):
    def test_build_command_creates_the_frozen_565_run_contract(self) -> None:
        """Catches a missing/broken builder or a cohort that silently drops stable runs."""
        command = [sys.executable, "-m", "step6.scripts.build_modeling_table"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

        table = pd.read_parquet(STEP6 / "analysis/phase4_power_modeling_table.parquet")
        self.assertEqual(len(table), 565)
        self.assertEqual(table["run_key"].nunique(), 565)
        self.assertEqual(table["diag_model_id"].nunique(), 27)
        self.assertEqual(int(table["diag_parent_lineage"].eq("Llama").sum()), 117)
        self.assertEqual(
            set(table["diag_architecture_class"].unique()),
            {"dense_transformer", "mixture_of_experts", "hybrid_mamba_transformer"},
        )

        holdout = json.loads(
            (STEP6 / "config/confirmatory_model_holdout.json").read_text(encoding="utf-8")
        )
        held_models = set(holdout["held_out_model_ids"])
        self.assertTrue(held_models)
        self.assertTrue(held_models < set(table["diag_model_id"]))
        self.assertFalse("y_gpu_avg_power_watts" in json.dumps(holdout))

    def test_main_bundles_cover_all_runs_without_identity_or_post_run_leakage(self) -> None:
        """Catches incomplete hardware proxies and forbidden diagnostic/post-run X columns."""
        table = pd.read_parquet(STEP6 / "analysis/phase4_power_modeling_table.parquet")
        bundles = load_feature_bundles()["bundles"]
        forbidden = {
            "diag_model_id",
            "diag_parent_lineage",
            "diag_subfamily",
        }
        for name in ["P0-Core", "P1-AggregatePower", "P2-HardwareStatic"]:
            columns = bundles[name]["columns"]
            self.assertEqual(int(table[columns].notna().all(axis=1).sum()), 565, name)
            self.assertFalse(forbidden.intersection(columns), name)
            self.assertFalse(
                any(column.startswith(("y_", "diag_result_", "diag_load_", "prov_")) for column in columns),
                name,
            )
        self.assertEqual(verify_source_freeze(), [])


class TestTask2Splits(unittest.TestCase):
    def test_split_command_is_deterministic_and_keeps_confirmatory_models_out(self) -> None:
        """Catches leaked confirmatory rows and group assignments that drift or overlap."""
        command = [sys.executable, "-m", "step6.scripts.build_splits"]
        first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        paths = [
            STEP6 / "splits/development_model_folds.csv",
            STEP6 / "splits/lineage_holdout_folds.csv",
            STEP6 / "splits/within_task_lineage_folds.csv",
            STEP6 / "splits/architecture_holdout_folds.csv",
            STEP6 / "splits/full_cohort_lineage_holdout_folds.csv",
            STEP6 / "splits/full_cohort_within_task_lineage_folds.csv",
            STEP6 / "splits/full_cohort_architecture_holdout_folds.csv",
        ]
        snapshots = [path.read_bytes() for path in paths]
        second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(snapshots, [path.read_bytes() for path in paths])

        full_lineage = pd.read_csv(STEP6 / "splits/full_cohort_lineage_holdout_folds.csv")
        full_within = pd.read_csv(STEP6 / "splits/full_cohort_within_task_lineage_folds.csv")
        full_architecture = pd.read_csv(STEP6 / "splits/full_cohort_architecture_holdout_folds.csv")
        self.assertEqual(len(full_lineage), 565)
        self.assertEqual(len(full_within), 515)
        self.assertEqual(len(full_architecture), 565)
        self.assertTrue(full_lineage["post_lock_diagnostic"].all())
        self.assertTrue(full_within["post_lock_diagnostic"].all())
        self.assertTrue(full_architecture["post_lock_diagnostic"].all())

        table = pd.read_parquet(STEP6 / "analysis/phase4_power_modeling_table.parquet")
        dev = table.loc[table["eligible_phase4_development"]]
        model_folds = pd.read_csv(paths[0])
        self.assertEqual(set(model_folds["run_key"]), set(dev["run_key"]))
        self.assertEqual(model_folds["run_key"].nunique(), len(dev))
        self.assertEqual(model_folds.groupby("diag_model_id")["fold_id"].nunique().max(), 1)
        self.assertFalse(model_folds["diag_model_id"].isin(table.loc[table["diag_is_confirmatory_model"], "diag_model_id"]).any())

        lineage = pd.read_csv(paths[1])
        self.assertEqual(set(lineage["run_key"]), set(dev["run_key"]))
        self.assertEqual(lineage.groupby("diag_parent_lineage")["fold_id"].nunique().max(), 1)

        within = pd.read_csv(paths[2])
        eligible_tasks = set(
            dev.groupby("x_protocol_task")["diag_parent_lineage"].nunique().loc[lambda s: s >= 3].index
        )
        self.assertEqual(set(within["x_protocol_task"]), eligible_tasks)
        architecture = pd.read_csv(paths[3])
        self.assertEqual(set(architecture["diag_architecture_class"]), set(dev["diag_architecture_class"]))


class TestTask3DevelopmentMatrix(unittest.TestCase):
    def test_checkpoint_identity_changes_with_material_experiment_inputs(self) -> None:
        """Catches stale cache reuse after data, folds, features, or candidates change."""
        from step6.scripts.checkpoints import (
            canonicalize_experiment_inputs,
            checkpoint_fingerprint,
        )

        frame = pd.DataFrame(
            {
                "run_key": ["a", "b", "c", "d"],
                "num": [1.0, 2.0, 3.0, 4.0],
                "extra": [4.0, 3.0, 2.0, 1.0],
                "cat": ["x", "x", "y", "y"],
                "diag_model_id": ["m1", "m1", "m2", "m2"],
                "y": [10.0, 20.0, 30.0, 40.0],
            }
        )
        folds = pd.DataFrame({"run_key": frame["run_key"], "fold_id": [0, 0, 1, 1]})
        bundle = {"numeric": ["num"], "categorical": ["cat"], "columns": ["cat", "num"]}
        arguments = {
            "spec": {"algorithm": "Ridge", "bundle_name": "synthetic"},
            "bundle": bundle,
            "candidates": [{"alpha": 1.0}],
            "target": "y",
            "inner_group_column": "diag_model_id",
            "implementation_paths": [STEP6 / "scripts/modeling.py"],
        }
        baseline = checkpoint_fingerprint(frame, folds, **arguments)
        changed_target = frame.copy()
        changed_target.loc[0, "y"] += 1.0
        changed_folds = folds.copy()
        changed_folds.loc[0, "fold_id"] = 1
        changed_groups = frame.copy()
        changed_groups["diag_model_id"] = ["m1", "m2", "m1", "m2"]
        changed_bundle = {"numeric": ["num", "extra"], "categorical": ["cat"], "columns": ["cat", "num", "extra"]}
        self.assertNotEqual(baseline, checkpoint_fingerprint(changed_target, folds, **arguments))
        self.assertNotEqual(baseline, checkpoint_fingerprint(frame, changed_folds, **arguments))
        self.assertNotEqual(baseline, checkpoint_fingerprint(changed_groups, folds, **arguments))
        self.assertEqual(baseline, checkpoint_fingerprint(frame.iloc[::-1], folds, **arguments))
        canonical_frame, canonical_folds = canonicalize_experiment_inputs(
            frame.iloc[::-1], folds.iloc[::-1]
        )
        self.assertEqual(canonical_frame["run_key"].tolist(), ["a", "b", "c", "d"])
        self.assertEqual(canonical_folds["run_key"].tolist(), ["a", "b", "c", "d"])
        self.assertNotEqual(
            baseline,
            checkpoint_fingerprint(frame, folds, **{**arguments, "bundle": changed_bundle}),
        )
        self.assertNotEqual(
            baseline,
            checkpoint_fingerprint(frame, folds, **{**arguments, "candidates": [{"alpha": 2.0}]}),
        )

    def test_model_factory_handles_unseen_categories_and_selects_on_train_groups(self) -> None:
        """Catches preprocessing leakage and estimators that fail on an unseen GPU category."""
        self.assertIsNotNone(importlib.util.find_spec("step6.scripts.modeling"))
        from step6.scripts.modeling import fit_predict_outer_fold

        train = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b", "c", "c"],
                "cat": ["H100", "H100", "H100", "H100", "B200", "B200"],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0],
            }
        )
        held_out = pd.DataFrame({"group": ["d"], "cat": ["future_gpu"], "num": [7.0], "y": [40.0]})
        prediction, metadata, _ = fit_predict_outer_fold(
            train,
            held_out,
            numeric=["num"],
            categorical=["cat"],
            target="y",
            algorithm="Ridge",
            candidates=[{"alpha": 1.0}, {"alpha": 100.0}],
            inner_group_column="group",
        )
        self.assertEqual(len(prediction), 1)
        self.assertTrue(float(prediction[0]) > 0)
        self.assertEqual(metadata["outer_test_rows"], 1)
        self.assertEqual(metadata["inner_train_rows"], 6)

    def test_development_runner_produces_complete_six_by_four_oof_matrix(self) -> None:
        """Catches missing algorithms/bundles, duplicate OOF rows, and metric drift."""
        command = [sys.executable, "-m", "step6.scripts.run_development_matrix"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        predictions = pd.read_parquet(STEP6 / "analysis/development_oof_predictions.parquet")
        comparison = pd.read_csv(STEP6 / "reports/main_model_comparison.csv")
        self.assertEqual(
            set(predictions["algorithm"]),
            {"DummyMean", "Ridge", "HistGradientBoosting", "RandomForest", "CatBoost", "XGBoost"},
        )
        self.assertEqual(
            set(predictions["feature_bundle"]),
            {"P0-Core", "P1-AggregatePower", "P2-HardwareStatic", "P2-HardwareStatic-NoLabel"},
        )
        support = predictions.groupby(["algorithm", "feature_bundle"])["run_key"].agg(["size", "nunique"])
        self.assertTrue((support["size"] == 449).all())
        self.assertTrue((support["nunique"] == 449).all())
        self.assertEqual(len(comparison), 24)
        self.assertTrue(comparison[["mae", "rmse", "mdape", "smape"]].notna().all().all())


class TestTask4HardwareAblation(unittest.TestCase):
    def test_paired_ablation_bootstraps_models_and_locks_only_development_shortlist(self) -> None:
        """Catches unpaired feature comparisons, row bootstrap, and premature confirmatory use."""
        command = [sys.executable, "-m", "step6.scripts.analyze_development"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        ablation = pd.read_csv(STEP6 / "reports/hardware_ablation.csv")
        draws = pd.read_csv(STEP6 / "reports/hardware_bootstrap_draws.csv")
        self.assertEqual(len(ablation), 18)
        self.assertTrue((ablation["n_paired_rows"] == 449).all())
        self.assertTrue((ablation["n_models"] == 21).all())
        self.assertEqual(len(draws), 18_000)
        self.assertTrue((draws.groupby(["algorithm", "comparison"])["draw"].nunique() == 1000).all())
        lock = json.loads((STEP6 / "config/locked_candidate_phase4.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "development_shortlist_before_loss_comparison")
        self.assertEqual(len(lock["candidate_shortlist"]), 2)
        self.assertNotIn("confirmatory", json.dumps(lock).lower().replace("confirmatory_not_used", ""))


class TestTask5LossStrategies(unittest.TestCase):
    def test_high_power_weights_use_only_the_supplied_training_targets(self) -> None:
        """Catches test-fold influence on the weight scale or missing clipping."""
        from step6.scripts import modeling

        self.assertTrue(hasattr(modeling, "high_power_weights"))
        high_power_weights = modeling.high_power_weights

        weights = high_power_weights([25.0, 100.0, 400.0, 10_000.0])
        # Median is 250; the first and last values hit the specified 0.5/2.0 clips.
        self.assertAlmostEqual(float(weights[0]), 0.5)
        self.assertAlmostEqual(float(weights[-1]), 2.0)
        self.assertAlmostEqual(float(weights[1]), (100.0 / 250.0) ** 0.5)

    def test_pseudo_huber_scales_delta_from_training_targets(self) -> None:
        """Catches XGBoost's 1-W default Huber slope exploding on kilowatt-scale labels."""
        from step6.scripts.modeling import fit_predict_outer_fold

        train = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b", "c", "c"],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            }
        )
        held_out = pd.DataFrame({"group": ["d"], "num": [7.0], "y": [700.0]})
        prediction, metadata, _ = fit_predict_outer_fold(
            train,
            held_out,
            numeric=["num"],
            categorical=[],
            target="y",
            algorithm="XGBoost",
            candidates=[{"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05}],
            inner_group_column="group",
            objective="pseudo_huber",
        )
        self.assertGreater(float(prediction[0]), 0.0)
        self.assertLess(float(prediction[0]), 6000.0)
        self.assertAlmostEqual(metadata["outer_huber_slope"], 350.0)

    def test_loss_runner_compares_only_shortlisted_candidates_and_finalizes_lock(self) -> None:
        """Catches scope expansion, weighted evaluation, and an unlocked final choice."""
        command = [sys.executable, "-m", "step6.scripts.run_loss_strategies"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        comparison = pd.read_csv(STEP6 / "reports/loss_strategy_comparison.csv")
        predictions = pd.read_parquet(STEP6 / "analysis/loss_strategy_predictions.parquet")
        self.assertEqual(len(comparison), 7)
        self.assertEqual(set(comparison["algorithm"]), {"RandomForest", "XGBoost"})
        self.assertEqual(set(comparison["evaluation_weighting"]), {"unweighted"})
        self.assertTrue((predictions.groupby("strategy_id")["run_key"].nunique() == 449).all())
        lock = json.loads((STEP6 / "config/locked_candidate_phase4.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "final_locked_before_confirmatory_evaluation")
        self.assertIn(lock["selected"]["strategy_id"], set(comparison["strategy_id"]))
        self.assertNotIn("held_out_model_ids", lock)
        self.assertNotIn("confirmatory_metrics", lock)


class TestTask6ConfirmatoryEvaluation(unittest.TestCase):
    def test_confirmatory_evaluation_is_one_shot_idempotent_and_complete(self) -> None:
        """Catches pre-lock evaluation, held-out leakage, rewrites, and incomplete diagnostic OOF."""
        command = [sys.executable, "-m", "step6.scripts.evaluate_locked_candidate"]
        first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        confirmatory_path = STEP6 / "analysis/confirmatory_predictions.parquet"
        before = confirmatory_path.read_bytes()
        second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(before, confirmatory_path.read_bytes())

        table = pd.read_parquet(STEP6 / "analysis/phase4_power_modeling_table.parquet")
        confirmatory = pd.read_parquet(confirmatory_path)
        full = pd.read_parquet(STEP6 / "analysis/full_cohort_oof_predictions.parquet")
        expected = table.loc[table["diag_is_confirmatory_model"]]
        self.assertEqual(len(confirmatory), len(expected))
        self.assertEqual(set(confirmatory["run_key"]), set(expected["run_key"]))
        self.assertFalse(set(confirmatory["run_key"]) & set(table.loc[table["eligible_phase4_development"], "run_key"]))
        self.assertEqual(len(full), 565)
        self.assertEqual(full["run_key"].nunique(), 565)
        receipt = json.loads((STEP6 / "analysis/confirmatory_evaluation_receipt.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["evaluated_once"])
        self.assertEqual(receipt["prediction_rows"], len(confirmatory))
        self.assertTrue((STEP6 / "models/phase4_candidate_model.joblib").exists())


class TestTask7GeneralizationAndFigures(unittest.TestCase):
    def test_locked_candidate_generalization_outputs_cover_distinct_levels(self) -> None:
        """Catches pooled family claims, missing task controls, and unsupported figure omissions."""
        command = [sys.executable, "-m", "step6.scripts.analyze_phase4"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        lineage = pd.read_csv(STEP6 / "reports/generalization_by_lineage.csv")
        within = pd.read_csv(STEP6 / "reports/within_task_lineage_holdout.csv")
        architecture = pd.read_csv(STEP6 / "reports/architecture_holdout.csv")
        slices = pd.read_csv(STEP6 / "reports/error_slices.csv")
        static = pd.read_csv(STEP6 / "reports/static_subset_comparison.csv")
        self.assertEqual(set(lineage["diag_parent_lineage"]), {"Qwen", "Llama", "DeepSeek", "Gemma", "Nemotron", "GPT-OSS"})
        self.assertEqual(set(architecture["diag_architecture_class"]), {"dense_transformer", "mixture_of_experts", "hybrid_mamba_transformer"})
        self.assertTrue((within.groupby("x_protocol_task")["diag_parent_lineage"].nunique() >= 3).all())
        self.assertTrue({"gpu_model", "gpu_count", "parent_lineage", "measured_power_quartile", "top_25_percent", "top_10_percent"}.issubset(set(slices["slice_type"])))
        self.assertEqual(set(static["feature_bundle"]), {"S0-B1-Static", "S1-B1-Static-AggregatePower", "S2-B1-Static-HardwareStatic"})
        self.assertEqual(static["n_rows"].nunique(), 1)

        required_figures = {
            "target_distribution.png",
            "task_parent_lineage_heatmap.png",
            "architecture_parent_lineage_heatmap.png",
            "development_actual_vs_predicted.png",
            "confirmatory_actual_vs_predicted.png",
            "residual_vs_measured_power.png",
            "residual_by_gpu_count.png",
            "residual_by_gpu_model.png",
            "residual_by_parent_lineage.png",
            "top_power_quantile_bias.png",
            "hardware_paired_mae_delta.png",
            "lineage_holdout_mae_bias.png",
        }
        figure_dir = STEP6 / "reports/figures"
        self.assertTrue(all((figure_dir / name).stat().st_size > 5000 for name in required_figures))

        generalization_predictions = pd.read_parquet(STEP6 / "analysis/generalization_predictions.parquet")
        manifests = {
            "parent_lineage_holdout": pd.read_csv(STEP6 / "splits/full_cohort_lineage_holdout_folds.csv"),
            "within_task_lineage_holdout": pd.read_csv(STEP6 / "splits/full_cohort_within_task_lineage_folds.csv"),
            "architecture_class_holdout_stress": pd.read_csv(STEP6 / "splits/full_cohort_architecture_holdout_folds.csv"),
        }
        for scheme, manifest in manifests.items():
            observed = generalization_predictions.loc[
                generalization_predictions["evaluation_scheme"].eq(scheme), "run_key"
            ]
            self.assertEqual(len(observed), len(manifest), scheme)
            self.assertEqual(set(observed), set(manifest["run_key"]), scheme)


class TestTask8DeliveryAndIndependentValidation(unittest.TestCase):
    def test_text_hashes_are_portable_across_lf_and_crlf_checkouts(self) -> None:
        """Catches Windows worktree hashes that fail on an LF checkout of the same Git blob."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lf = base / "lf.json"
            crlf = base / "crlf.json"
            lf.write_bytes(b'{\n  "answer": 42\n}\n')
            crlf.write_bytes(b'{\r\n  "answer": 42\r\n}\r\n')
            self.assertEqual(sha256_file(lf), sha256_file(crlf))

        confirmatory = json.loads(
            (STEP6 / "analysis/confirmatory_evaluation_receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            sha256_file(STEP6 / "config/locked_candidate_phase4.json"),
            confirmatory["locked_candidate_sha256"],
        )
        self.assertEqual(
            sha256_file(STEP6 / "config/confirmatory_model_holdout.json"),
            confirmatory["holdout_definition_sha256"],
        )

    def test_report_notebook_and_independent_receipt_are_complete(self) -> None:
        """Catches an unexecuted evidence notebook or a report that omits required decisions."""
        build = subprocess.run(
            [sys.executable, "-m", "step6.scripts.build_notebook"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        validate = subprocess.run(
            [sys.executable, "-m", "step6.scripts.validate_phase40"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(validate.returncode, 0, validate.stderr)
        receipt_path = STEP6 / "reports/verification_receipt.json"
        receipt_before = receipt_path.read_bytes()
        validate_again = subprocess.run(
            [sys.executable, "-m", "step6.scripts.validate_phase40"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(validate_again.returncode, 0, validate_again.stderr)
        self.assertEqual(receipt_before, receipt_path.read_bytes())

        report = (STEP6 / "reports/phase4_power_model_report.md").read_text(encoding="utf-8")
        for token in (
            "565",
            "117",
            "P0 → P1",
            "P1 → P2",
            "P2-NoLabel",
            "Confirmatory",
            "unseen model",
            "unseen parent lineage",
            "within-task unseen lineage",
            "unseen architecture class",
            "推荐候选",
            "适用边界",
            "主要剩余问题",
            "GPU 侧",
            "high-load / near-saturation steady-state",
        ):
            self.assertIn(token, report)

        notebook = json.loads(
            (STEP6 / "notebooks/phase4_power_modeling.ipynb").read_text(encoding="utf-8")
        )
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("execution_count") is not None for cell in code_cells))
        self.assertFalse(
            any(
                output.get("output_type") == "error"
                for cell in code_cells
                for output in cell.get("outputs", [])
            )
        )
        code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        self.assertNotIn(".fit(", code)
        self.assertNotIn("joblib.load", code)

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "pass")
        self.assertTrue(receipt["checks"])
        self.assertTrue(all(check["status"] == "pass" for check in receipt["checks"]))
        self.assertTrue((STEP6 / "notebooks/phase4_power_modeling.html").stat().st_size > 100_000)


if __name__ == "__main__":
    unittest.main()
