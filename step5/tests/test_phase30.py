from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STEP5 = ROOT / "step5"


class TestPhase30Contracts(unittest.TestCase):
    def _json(self, relative: str) -> dict:
        path = STEP5 / relative
        self.assertTrue(path.exists(), f"missing Phase 3 contract: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_source_freeze_verifies_all_pinned_inputs(self) -> None:
        module_path = STEP5 / "scripts/common.py"
        self.assertTrue(module_path.exists(), "Phase 3 source verifier is missing")
        common = importlib.import_module("step5.scripts.common")
        self.assertEqual([], common.verify_source_freeze())
        freeze = self._json("config/source_freeze_phase3.json")
        self.assertEqual(
            "DDCCAF572D291B6D454351C18FDEAA6961C9570A5B5B37591CF00F6F805C4052",
            freeze["taskbook_sha256"],
        )
        self.assertGreaterEqual(len(freeze["frozen_inputs"]), 7)

    def test_phase2_b0_b3_are_preserved_exactly(self) -> None:
        common = importlib.import_module("step5.scripts.common")
        phase2 = json.loads(
            (ROOT / "step4/config/feature_bundles.json").read_text(encoding="utf-8")
        )["bundles"]
        phase3 = common.load_feature_bundles()["bundles"]
        for name in ("B0-Core", "B0-Protocol", "B1-Static", "B2a-Benchmark", "B2b-Effective", "B3-Physics"):
            self.assertEqual(phase2[name], phase3[name], f"{name} drifted from Phase 2")

    def test_b4_b5_contain_required_hardware_physics_without_leakage(self) -> None:
        common = importlib.import_module("step5.scripts.common")
        bundles = common.load_feature_bundles()["bundles"]
        required = {
            "x_hardware_peak_compute_bf16_dense_tflops",
            "x_hardware_peak_compute_fp8_dense_tflops",
            "x_hardware_hbm_bandwidth_gbps",
            "x_hardware_hbm_capacity_gb",
            "x_hardware_rated_power_w",
            "x_hardware_interconnect_bandwidth_gbps",
            "x_hardware_aggregate_rated_power_w",
            "x_hardware_physics_compute_time_proxy_seconds",
            "x_hardware_physics_memory_time_proxy_seconds",
            "x_hardware_physics_weight_capacity_ratio",
            "x_hardware_physics_kv_capacity_ratio_mean",
            "x_hardware_physics_kv_capacity_ratio_max",
            "x_hardware_physics_compute_to_memory_pressure_ratio",
        }
        self.assertTrue(required.issubset(set(bundles["B5-HardwarePhysics"]["columns"])))
        self.assertNotIn("x_hardware_gpu_model", bundles["B5-HardwarePhysics-NoLabel"]["columns"])
        forbidden = (
            "actual", "throughput", "latency", "utilization", "temperature",
            "clock", "duration", "quality", "repeat", "seed",
        )
        for name, bundle in bundles.items():
            for column in bundle["columns"]:
                self.assertTrue(column.startswith("x_"), f"{name}: non-X field {column}")
                self.assertFalse(column.startswith(("y_", "diag_", "prov_")), f"{name}: forbidden {column}")
                self.assertFalse(any(token in column.lower() for token in forbidden), f"{name}: forbidden {column}")

    def test_hardware_profiles_are_positive_and_uncertainty_is_explicit(self) -> None:
        specs = self._json("config/gpu_hardware_specs.json")
        self.assertEqual({"H100", "B200"}, set(specs["profiles"]))
        numeric = (
            "peak_compute_bf16_dense_tflops", "peak_compute_fp8_dense_tflops",
            "hbm_bandwidth_gbps", "hbm_capacity_gb", "rated_power_w",
            "interconnect_bandwidth_gbps",
        )
        for label, profile in specs["profiles"].items():
            self.assertIn(profile["hardware_source_status"], {"exact", "benchmark-supported", "ambiguous"})
            self.assertEqual("dense_without_structured_sparsity", profile["peak_compute_convention"])
            for field in numeric:
                self.assertGreater(profile[field], 0, f"{label}.{field}")
        manifest = self._json("config/hardware_source_manifest.json")
        self.assertEqual({"H100", "B200"}, set(manifest["gpu_label_resolution"]))
        for resolution in manifest["gpu_label_resolution"].values():
            self.assertEqual("ambiguous", resolution["hardware_source_status"])
            self.assertGreaterEqual(len(resolution["evidence_urls"]), 2)
        self.assertTrue(all(source["url"].startswith("https://") for source in manifest["sources"]))

    def test_search_spaces_are_bounded_and_cover_required_algorithms(self) -> None:
        config = self._json("config/model_search_spaces.json")
        required = {
            "DummyMean", "Ridge", "ElasticNet", "RandomForest", "ExtraTrees",
            "HistGradientBoosting", "XGBoost", "CatBoost", "SplineGAM", "SmallMLP",
        }
        self.assertEqual(required, set(config["algorithms"]))
        for name, spec in config["algorithms"].items():
            candidates = spec["candidates"]
            self.assertGreaterEqual(len(candidates), 1, name)
            self.assertLessEqual(len(candidates), 40, name)
        self.assertEqual("group_aware_inner_cv", config["selection_protocol"])
        self.assertEqual("mae", config["selection_metric"])


class TestPhase30Table(unittest.TestCase):
    def test_hardware_proxy_formulas_match_hand_calculation(self) -> None:
        module_path = STEP5 / "scripts/build_modeling_table.py"
        self.assertTrue(module_path.exists(), "Phase 3 table builder is missing")
        module = importlib.import_module("step5.scripts.build_modeling_table")
        row = pd.Series(
            {
                "x_hardware_gpu_model": "H100",
                "x_hardware_num_gpus": 2,
                "x_model_benchmark_weight_precision": "bfloat16",
                "x_deployment_model_parallel_gpus_per_replica": 2,
                "x_physics_prefill_total_flop_per_request": 1.979e15,
                "x_physics_weight_bytes_per_gpu_ideal": 40e9,
                "x_physics_kv_cache_prefill_mean_request_bytes": 16e9,
                "x_physics_kv_cache_prefill_max_request_bytes": 32e9,
                "y_gpu_avg_power_watts": 700.0,
            }
        )
        derived = module.derive_hardware_features(row)
        self.assertAlmostEqual(1.0, derived["x_hardware_physics_compute_time_proxy_seconds"])
        self.assertAlmostEqual(48e9 / 3.35e12, derived["x_hardware_physics_memory_time_proxy_seconds"])
        self.assertAlmostEqual(0.5, derived["x_hardware_physics_weight_capacity_ratio"])
        self.assertAlmostEqual(0.1, derived["x_hardware_physics_kv_capacity_ratio_mean"])
        self.assertAlmostEqual(0.2, derived["x_hardware_physics_kv_capacity_ratio_max"])
        self.assertAlmostEqual(1400.0, derived["x_hardware_aggregate_rated_power_w"])
        self.assertAlmostEqual(0.5, derived["y_gpu_power_fraction_of_rated"])

    def test_mxfp4_uses_b200_fp4_peak_and_invalid_denominators_stay_missing(self) -> None:
        self.assertTrue((STEP5 / "scripts/build_modeling_table.py").exists(), "Phase 3 table builder is missing")
        module = importlib.import_module("step5.scripts.build_modeling_table")
        row = pd.Series(
            {
                "x_hardware_gpu_model": "B200",
                "x_hardware_num_gpus": 1,
                "x_model_benchmark_weight_precision": "mxfp4",
                "x_deployment_model_parallel_gpus_per_replica": 1,
                "x_physics_prefill_total_flop_per_request": 9e15,
                "x_physics_weight_bytes_per_gpu_ideal": 0.0,
                "x_physics_kv_cache_prefill_mean_request_bytes": 0.0,
                "x_physics_kv_cache_prefill_max_request_bytes": 0.0,
                "y_gpu_avg_power_watts": 500.0,
            }
        )
        derived = module.derive_hardware_features(row)
        self.assertEqual(9000.0, derived["x_hardware_precision_matched_peak_compute_tflops"])
        self.assertAlmostEqual(1.0, derived["x_hardware_physics_compute_time_proxy_seconds"])
        self.assertTrue(np.isnan(derived["x_hardware_physics_compute_to_memory_pressure_ratio"]))

    def test_generated_table_preserves_population_and_frozen_columns(self) -> None:
        output = STEP5 / "analysis/phase3_modeling_table.parquet"
        self.assertTrue(output.exists(), "Phase 3 modeling table has not been generated")
        original = pd.read_parquet(ROOT / "step4/analysis/modeling_table.parquet")
        enhanced = pd.read_parquet(output)
        self.assertEqual((694, 694), (len(enhanced), enhanced["run_key"].nunique()))
        pd.testing.assert_frame_equal(
            original.reset_index(drop=True),
            enhanced.loc[:, original.columns].reset_index(drop=True),
            check_dtype=True,
        )
        counts = {
            "eligible_primary_stable": 565,
            "eligible_static_architecture": 448,
            "eligible_phase20_full": 381,
            "eligible_phase20_physics": 305,
        }
        for flag, expected in counts.items():
            self.assertEqual(expected, int(enhanced[flag].sum()))
        physics = enhanced.loc[enhanced["eligible_phase20_physics"]]
        common = importlib.import_module("step5.scripts.common")
        b5 = common.load_feature_bundles()["bundles"]["B5-HardwarePhysics"]["columns"]
        self.assertFalse(physics[b5].isna().any().any())
        self.assertNotIn("diag_hardware_source_status", b5)

    def test_load_regime_outputs_are_diagnostic_only_and_support_near_saturation(self) -> None:
        report = STEP5 / "reports/load_regime_characterization.md"
        figure = STEP5 / "reports/figures/load_regime_distribution.png"
        self.assertTrue(report.exists())
        self.assertTrue(figure.exists())
        text = report.read_text(encoding="utf-8")
        self.assertIn("diagnostic-only", text)
        self.assertIn("near-saturation", text)
        frame = pd.read_parquet(STEP5 / "analysis/phase3_modeling_table.parquet")
        stable = frame.loc[frame["eligible_primary_stable"]]
        ratio = stable["diag_load_avg_batch_to_capacity_ratio"]
        self.assertGreater(float(ratio.median()), 0.99)
        common = importlib.import_module("step5.scripts.common")
        all_x = {
            column
            for bundle in common.load_feature_bundles()["bundles"].values()
            for column in bundle["columns"]
        }
        self.assertNotIn("diag_load_avg_batch_to_capacity_ratio", all_x)


class TestPhysicsCoverage(unittest.TestCase):
    def test_distribution_summary_has_hand_checked_quantiles_and_cv(self) -> None:
        module_path = STEP5 / "scripts/analyze_physics_coverage.py"
        self.assertTrue(module_path.exists(), "physics coverage analyzer is missing")
        module = importlib.import_module("step5.scripts.analyze_physics_coverage")
        result = module.distribution_summary(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(1.0, result["min"])
        self.assertEqual(3.0, result["median"])
        self.assertEqual(5.0, result["max"])
        self.assertAlmostEqual(np.std([1, 2, 3, 4, 5], ddof=0) / 3.0, result["cv"])
        self.assertAlmostEqual(np.quantile([1, 2, 3, 4, 5], 0.95), result["p95"])

    def test_generated_coverage_outputs_have_complete_contract(self) -> None:
        summary_path = STEP5 / "analysis/physics_coverage_summary.csv"
        pearson_path = STEP5 / "analysis/physics_correlations_pearson.csv"
        spearman_path = STEP5 / "analysis/physics_correlations_spearman.csv"
        pca_path = STEP5 / "analysis/physics_pca_summary.csv"
        residual_path = STEP5 / "analysis/physics_residual_associations.csv"
        figure_path = STEP5 / "reports/figures/physics_coverage_by_gpu_family.png"
        for path in (summary_path, pearson_path, spearman_path, pca_path, residual_path, figure_path):
            self.assertTrue(path.exists(), f"missing physics diagnostic: {path}")

        summary = pd.read_csv(summary_path)
        required = {"feature", "feature_group", "n", "n_unique", "min", "p5", "p25", "median", "p75", "p95", "max", "cv", "p95_p5_ratio", "log10_range"}
        self.assertTrue(required.issubset(summary.columns))
        self.assertEqual({"B3", "B5"}, set(summary["feature_group"]))
        self.assertTrue((summary["n"] == 305).all())

        pearson = pd.read_csv(pearson_path, index_col=0)
        spearman = pd.read_csv(spearman_path, index_col=0)
        pd.testing.assert_frame_equal(pearson, pearson.T)
        pd.testing.assert_frame_equal(spearman, spearman.T)
        np.testing.assert_allclose(np.diag(pearson), 1.0)
        np.testing.assert_allclose(np.diag(spearman), 1.0)

        pca = pd.read_csv(pca_path)
        self.assertEqual(list(range(1, len(pca) + 1)), pca["component"].tolist())
        self.assertTrue(pca["cumulative_explained_variance_ratio"].is_monotonic_increasing)
        self.assertLessEqual(float(pca["cumulative_explained_variance_ratio"].max()), 1.0 + 1e-12)

        residual = pd.read_csv(residual_path)
        self.assertEqual({"B2b residual -> B3", "B3 residual -> B5"}, set(residual["diagnostic_path"]))
        self.assertTrue((residual["n"] == 305).all())


class TestModelingFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modeling_path = STEP5 / "scripts/modeling.py"
        cls.splits_path = STEP5 / "scripts/build_splits.py"

    def test_model_factory_is_deterministic_and_covers_algorithms(self) -> None:
        self.assertTrue(self.modeling_path.exists(), "modeling framework is missing")
        modeling = importlib.import_module("step5.scripts.modeling")
        config = json.loads((STEP5 / "config/model_search_spaces.json").read_text(encoding="utf-8"))
        frame = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
        for algorithm, spec in config["algorithms"].items():
            first = modeling.make_estimator(
                algorithm, spec["candidates"][0], frame, ["num"], ["cat"]
            )
            second = modeling.make_estimator(
                algorithm, spec["candidates"][0], frame, ["num"], ["cat"]
            )
            first_model = first.named_steps["model"] if hasattr(first, "named_steps") else first
            second_model = second.named_steps["model"] if hasattr(second, "named_steps") else second
            self.assertEqual(first_model.__class__, second_model.__class__, algorithm)
            first_params = {
                key: value
                for key, value in first_model.get_params(deep=False).items()
                if isinstance(value, (str, int, float, bool, type(None), tuple))
            }
            second_params = {
                key: value
                for key, value in second_model.get_params(deep=False).items()
                if isinstance(value, (str, int, float, bool, type(None), tuple))
            }
            self.assertEqual(first_params, second_params, algorithm)

    def test_outer_prediction_handles_unseen_category_without_using_test_in_selection(self) -> None:
        self.assertTrue(self.modeling_path.exists(), "modeling framework is missing")
        modeling = importlib.import_module("step5.scripts.modeling")
        train = pd.DataFrame(
            {
                "num": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "cat": ["a", "a", "b", "b", "a", "b"],
                "group": ["g1", "g1", "g2", "g2", "g3", "g3"],
                "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x_hardware_aggregate_rated_power_w": [10.0] * 6,
            }
        )
        held_out = pd.DataFrame(
            {
                "num": [6.0], "cat": ["unseen"], "group": ["never-in-inner"],
                "target": [999999.0], "x_hardware_aggregate_rated_power_w": [10.0],
            }
        )
        candidates = [{"alpha": 0.1}, {"alpha": 10.0}]
        prediction, metadata, _ = modeling.fit_predict_outer_fold(
            train,
            held_out,
            numeric=["num"],
            categorical=["cat"],
            target="target",
            algorithm="Ridge",
            candidates=candidates,
            inner_group_column="group",
            target_transform="raw",
            target_strategy="direct",
        )
        held_out_changed = held_out.assign(target=-999999.0)
        second, second_metadata, _ = modeling.fit_predict_outer_fold(
            train,
            held_out_changed,
            numeric=["num"],
            categorical=["cat"],
            target="target",
            algorithm="Ridge",
            candidates=candidates,
            inner_group_column="group",
            target_transform="raw",
            target_strategy="direct",
        )
        np.testing.assert_allclose(prediction, second)
        self.assertEqual(metadata["selected_candidate"], second_metadata["selected_candidate"])
        self.assertTrue(np.isfinite(prediction).all())

    def test_target_transforms_and_power_fraction_return_original_units(self) -> None:
        self.assertTrue(self.modeling_path.exists(), "modeling framework is missing")
        modeling = importlib.import_module("step5.scripts.modeling")
        values = np.array([0.0, 1.0, 9.0])
        np.testing.assert_allclose(values, modeling.inverse_target(modeling.transform_target(values, "log1p"), "log1p"))
        train = pd.DataFrame(
            {
                "num": [0.0, 1.0, 2.0, 3.0], "cat": ["a"] * 4,
                "group": ["g1", "g1", "g2", "g2"], "target": [100.0, 200.0, 300.0, 400.0],
                "x_hardware_aggregate_rated_power_w": [1000.0] * 4,
            }
        )
        test = pd.DataFrame(
            {
                "num": [4.0], "cat": ["a"], "group": ["g3"], "target": [500.0],
                "x_hardware_aggregate_rated_power_w": [2000.0],
            }
        )
        prediction, _, fraction = modeling.fit_predict_outer_fold(
            train, test, numeric=["num"], categorical=["cat"], target="target",
            algorithm="DummyMean", candidates=[{}], inner_group_column="group",
            target_transform="raw", target_strategy="power_fraction",
        )
        self.assertAlmostEqual(500.0, prediction[0])
        self.assertAlmostEqual(0.25, fraction[0])

    def test_split_manifests_are_deterministic_and_group_isolated(self) -> None:
        for path in (
            STEP5 / "splits/family_group_folds.csv",
            STEP5 / "splits/random_split_manifest.csv",
            STEP5 / "splits/split_manifest_phase3.json",
            STEP5 / "models/model_manifest.json",
        ):
            self.assertTrue(path.exists(), f"missing framework artifact: {path}")
        family = pd.read_csv(STEP5 / "splits/family_group_folds.csv")
        for cohort, rows in family.groupby("cohort"):
            self.assertTrue((rows.groupby("diag_model_family")["outer_fold"].nunique() == 1).all(), cohort)
        random = pd.read_csv(STEP5 / "splits/random_split_manifest.csv")
        self.assertEqual(10, random["repeat_index"].nunique())
        self.assertEqual({"train", "test"}, set(random["split_role"]))
        for _, rows in random.groupby(["cohort", "repeat_index"]):
            self.assertEqual(len(rows), rows["run_key"].nunique())
            fraction = (rows["split_role"] == "test").mean()
            self.assertGreater(fraction, 0.18)
            self.assertLess(fraction, 0.22)


class TestCoreMatrix(unittest.TestCase):
    def test_experiment_plan_is_complete_and_balanced(self) -> None:
        module_path = STEP5 / "scripts/run_model_matrix.py"
        self.assertTrue(module_path.exists(), "core matrix runner is missing")
        module = importlib.import_module("step5.scripts.run_model_matrix")
        plan = pd.DataFrame(module.experiment_plan())
        self.assertEqual(120, len(plan))
        self.assertEqual(120, len(plan.drop_duplicates()))
        self.assertEqual(
            {"B0-Core", "B1-Static", "B2b-Effective", "B3-Physics", "B4-Hardware", "B5-HardwarePhysics"},
            set(plan["bundle"]),
        )
        self.assertEqual(
            {"DummyMean", "Ridge", "ElasticNet", "RandomForest", "ExtraTrees", "HistGradientBoosting", "XGBoost", "CatBoost", "SplineGAM", "SmallMLP"},
            set(plan["model_name"]),
        )
        self.assertEqual({"y_gpu_avg_power_watts", "y_gpu_energy_per_token_joules"}, set(plan["target"]))
        self.assertTrue((plan.groupby(["target", "model_name"])["bundle"].nunique() == 6).all())

    def test_matrix_artifacts_have_identical_support_metrics_and_provenance(self) -> None:
        paths = {
            "predictions": STEP5 / "analysis/oof_predictions.parquet",
            "comparison": STEP5 / "reports/model_comparison.csv",
            "ablation": STEP5 / "reports/feature_ablation.csv",
            "validity": STEP5 / "reports/prediction_validity.csv",
            "provenance": STEP5 / "analysis/training_provenance.json",
        }
        for name, path in paths.items():
            self.assertTrue(path.exists(), f"missing {name}: {path}")
        predictions = pd.read_parquet(paths["predictions"])
        self.assertEqual(36600, len(predictions))
        identity = ["target", "bundle", "model_name", "split_scheme", "target_transform", "target_strategy", "run_key"]
        self.assertEqual(len(predictions), len(predictions.drop_duplicates(identity)))
        self.assertTrue((predictions.groupby("experiment_id").size() == 305).all())
        self.assertEqual({0, 1, 2, 3, 4}, set(predictions["outer_fold"]))
        self.assertTrue(np.isfinite(predictions[["y_true", "y_pred"]]).all().all())
        for _, group in predictions.groupby(["target", "model_name"]):
            supports = group.groupby("bundle").apply(lambda rows: frozenset(zip(rows["run_key"], rows["outer_fold"])), include_groups=False)
            self.assertEqual(1, supports.nunique())

        comparison = pd.read_csv(paths["comparison"])
        self.assertEqual(120, len(comparison))
        self.assertTrue({"mae", "rmse", "r2", "mdape_percent", "smape_percent", "mae_ci_low", "mae_ci_high", "n_model_clusters"}.issubset(comparison.columns))
        sample = comparison.iloc[0]
        sample_rows = predictions.loc[predictions["experiment_id"].eq(sample["experiment_id"])]
        from step4.scripts.analyze_results import regression_metrics
        metrics = regression_metrics(sample_rows["y_true"], sample_rows["y_pred"])
        self.assertAlmostEqual(metrics["mae"], sample["mae"])
        self.assertAlmostEqual(metrics["rmse"], sample["rmse"])

        ablation = pd.read_csv(paths["ablation"])
        self.assertEqual(100, len(ablation))
        self.assertTrue((ablation["n_rows"] == 305).all())
        self.assertTrue((ablation["n_model_clusters"] == 14).all())
        pairs = set(zip(ablation["reference_bundle"], ablation["candidate_bundle"]))
        self.assertIn(("B3-Physics", "B4-Hardware"), pairs)
        self.assertIn(("B4-Hardware", "B5-HardwarePhysics"), pairs)
        validity = pd.read_csv(paths["validity"])
        self.assertEqual(120, len(validity))
        self.assertEqual(36600, int(validity["n_predictions"].sum()))
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        self.assertEqual(120, provenance["completed_experiments"])
        self.assertEqual("outer-train-only group-aware inner CV", provenance["selection_protocol"])
        self.assertIn("source_freeze_sha256", provenance)
        self.assertIn("feature_bundles_sha256", provenance)
        self.assertIn("model_folds_sha256", provenance)
        self.assertTrue((STEP5 / "analysis/oof_predictions.csv").exists())
        self.assertIn("checkpoint_fingerprint", predictions.columns)

    def test_checkpoint_identity_includes_all_reproducibility_inputs(self) -> None:
        module = importlib.import_module("step5.scripts.run_model_matrix")
        spec = module.experiment_plan()[0]
        self.assertNotEqual(
            module._checkpoint_path(spec, "fingerprint-a"),
            module._checkpoint_path(spec, "fingerprint-b"),
        )


class TestGeneralizationAndInterpretation(unittest.TestCase):
    def test_pdp_plot_curves_never_connect_different_outer_folds(self) -> None:
        module = importlib.import_module("step5.scripts.analyze_phase3")
        rows = pd.DataFrame(
            {
                "outer_fold": [0, 0, 1, 1],
                "grid_value": [1.0, 2.0, 1.5, 3.0],
                "mean_prediction_watts": [10.0, 20.0, 15.0, 30.0],
            }
        )
        curves = module.fold_local_pdp_curves(rows)
        self.assertEqual({0, 1}, set(curves))
        self.assertEqual([1.0, 2.0], curves[0]["grid_value"].tolist())
        self.assertEqual([1.5, 3.0], curves[1]["grid_value"].tolist())

    def test_grouped_fold_merge_retains_the_isolation_key(self) -> None:
        module = importlib.import_module("step5.scripts.analyze_phase3")
        physics = pd.DataFrame({"run_key": ["a", "b"], "value": [1, 2]})
        manifest = pd.DataFrame(
            {"run_key": ["a", "b"], "outer_fold": [0, 1], "config_group_id": ["g1", "g2"]}
        )
        merged = module.merge_fold_manifest(physics, manifest, "config_group_id")
        self.assertEqual(["g1", "g2"], merged["config_group_id"].tolist())
        self.assertEqual([0, 1], merged["outer_fold"].tolist())

    def test_generalization_ladder_has_required_schemes_repeats_and_support(self) -> None:
        summary_path = STEP5 / "reports/generalization_ladder.csv"
        predictions_path = STEP5 / "analysis/generalization_predictions.parquet"
        self.assertTrue(summary_path.exists(), "generalization ladder is missing")
        self.assertTrue(predictions_path.exists(), "generalization predictions are missing")
        summary = pd.read_csv(summary_path)
        self.assertEqual({"random", "config_group", "model_group", "family_group"}, set(summary["split_scheme"]))
        self.assertEqual(
            {"B0-Core__HistGradientBoosting", "B4-Hardware__XGBoost"},
            set(summary["model_variant"]),
        )
        random = summary.loc[summary["split_scheme"].eq("random")]
        self.assertTrue((random.groupby("model_variant")["repeat_index"].nunique() == 10).all())
        grouped = summary.loc[~summary["split_scheme"].eq("random")]
        self.assertTrue((grouped.groupby(["model_variant", "split_scheme"]).size() == 1).all())
        self.assertTrue((summary["n_test"] > 0).all())
        self.assertTrue({"mae", "rmse", "r2", "mdape_percent", "smape_percent"}.issubset(summary.columns))

        predictions = pd.read_parquet(predictions_path)
        self.assertTrue({"run_key", "y_true", "y_pred", "split_role", "outer_fold"}.issubset(predictions.columns))
        self.assertTrue(np.isfinite(predictions[["y_true", "y_pred"]]).all().all())
        canonical = predictions.loc[
            predictions["split_scheme"].eq("random")
            & predictions["repeat_index"].eq(0)
            & predictions["model_variant"].eq("B4-Hardware__XGBoost")
        ]
        self.assertEqual({"train", "test"}, set(canonical["split_role"]))
        for scheme in ("config_group", "model_group", "family_group"):
            rows = predictions.loc[
                predictions["split_scheme"].eq(scheme)
                & predictions["model_variant"].eq("B4-Hardware__XGBoost")
            ]
            self.assertEqual(305, len(rows))
            self.assertEqual(305, rows["run_key"].nunique())

    def test_target_strategies_positive_domain_and_explainability_contracts(self) -> None:
        strategy_path = STEP5 / "reports/target_strategy_comparison.csv"
        hardware_path = STEP5 / "reports/hardware_label_sensitivity.csv"
        importance_path = STEP5 / "reports/feature_importance.csv"
        pdp_path = STEP5 / "reports/partial_dependence.csv"
        slices_path = STEP5 / "reports/error_slices.csv"
        for path in (strategy_path, hardware_path, importance_path, pdp_path, slices_path):
            self.assertTrue(path.exists(), f"missing analysis artifact: {path}")

        strategy = pd.read_csv(strategy_path)
        pairs = set(zip(strategy["target"], strategy["target_transform"], strategy["target_strategy"]))
        self.assertTrue(
            {
                ("y_gpu_avg_power_watts", "raw", "direct"),
                ("y_gpu_avg_power_watts", "raw", "power_fraction"),
                ("y_gpu_energy_per_token_joules", "raw", "direct"),
                ("y_gpu_energy_per_token_joules", "log1p", "direct"),
            }.issubset(pairs)
        )
        self.assertTrue((strategy["n_rows"] == 305).all())
        log_energy = strategy.loc[
            strategy["target"].eq("y_gpu_energy_per_token_joules")
            & strategy["target_transform"].eq("log1p")
        ]
        self.assertTrue((log_energy["negative_prediction_count"] == 0).all())

        hardware = pd.read_csv(hardware_path)
        self.assertEqual(
            {"B3-Physics", "B4-Hardware", "B5-HardwarePhysics", "B5-HardwarePhysics-NoLabel"},
            set(hardware["bundle"]),
        )
        self.assertTrue((hardware["n_rows"] == 305).all())
        no_label = hardware.loc[hardware["bundle"].eq("B5-HardwarePhysics-NoLabel")]
        self.assertEqual([False], no_label["retains_gpu_label"].tolist())
        paired_path = STEP5 / "reports/hardware_paired_comparisons.csv"
        self.assertTrue(paired_path.exists())
        paired = pd.read_csv(paired_path)
        expected_pairs = {
            ("B0-Core__HistGradientBoosting", "B4-Hardware__XGBoost"),
            ("B3-Physics__XGBoost", "B4-Hardware__XGBoost"),
            ("B4-Hardware__XGBoost", "B5-HardwarePhysics__XGBoost"),
            ("B5-HardwarePhysics__XGBoost", "B5-HardwarePhysics-NoLabel__XGBoost"),
        }
        self.assertTrue(expected_pairs.issubset(set(zip(paired["reference_variant"], paired["candidate_variant"]))))
        self.assertTrue((paired["n_rows"] == 305).all())
        self.assertTrue((paired["n_model_clusters"] == 14).all())

        importance = pd.read_csv(importance_path)
        self.assertGreaterEqual(importance["model_variant"].nunique(), 2)
        self.assertEqual({0, 1, 2, 3, 4}, set(importance["outer_fold"]))
        self.assertEqual({"outer_test_permutation"}, set(importance["method"]))
        self.assertEqual({"outer_train_only"}, set(importance["fit_scope"]))
        self.assertTrue((importance["n_test"] > 0).all())
        self.assertTrue(np.isfinite(importance["importance_mae_increase_mean"]).all())

        pdp = pd.read_csv(pdp_path)
        self.assertGreaterEqual(pdp["feature"].nunique(), 4)
        self.assertLessEqual(pdp["feature"].nunique(), 6)
        self.assertEqual({0, 1, 2, 3, 4}, set(pdp["outer_fold"]))
        self.assertEqual({"outer_train_only"}, set(pdp["fit_scope"]))
        # Some high-importance execution-known variables (for example expert
        # parallel degree) are genuinely low-cardinality; their PDP grid must
        # preserve real support instead of fabricating duplicate points.
        self.assertTrue((pdp.groupby(["feature", "outer_fold"]).size() >= 4).all())
        slices = pd.read_csv(slices_path)
        self.assertTrue({"gpu_model", "model_family", "protocol_task", "num_gpus", "load_regime"}.issubset(set(slices["slice_type"])))

    def test_required_phase3_figures_exist_and_are_nontrivial(self) -> None:
        names = (
            "actual_vs_predicted_random_best.png",
            "actual_vs_predicted_model_holdout_best.png",
            "actual_vs_predicted_family_holdout_best.png",
            "residuals_model_holdout_best.png",
            "actual_vs_predicted_energy_best.png",
            "residuals_energy_best.png",
            "feature_importance_best.png",
            "partial_dependence_best.png",
        )
        for name in names:
            path = STEP5 / "reports/figures" / name
            self.assertTrue(path.exists(), f"missing figure: {name}")
            self.assertGreater(path.stat().st_size, 20_000, f"figure too small: {name}")


class TestReportAndNotebook(unittest.TestCase):
    def test_notebook_blueprint_is_read_only_and_sections_are_ordered(self) -> None:
        module_path = STEP5 / "scripts/build_notebook.py"
        self.assertTrue(module_path.exists(), "notebook builder is missing")
        module = importlib.import_module("step5.scripts.build_notebook")
        notebook = module.notebook_blueprint()
        text = "\n".join(str(cell.get("source", "")) for cell in notebook.cells)
        code = "\n".join(
            str(cell.get("source", "")) for cell in notebook.cells if cell.get("cell_type") == "code"
        )
        headings = (
            "## Scope and provenance",
            "## Physics coverage",
            "## Algorithm and feature matrix",
            "## Generalization ladder",
            "## Hardware and target strategies",
            "## Error analysis and interpretation",
            "## RQ1–RQ5 conclusions",
        )
        positions = [text.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn(".fit(", code)
        self.assertNotIn("make_estimator", code)
        self.assertNotIn("XGBRegressor", code)
        self.assertIn("source_freeze_phase3.json", text)
        self.assertIn("training_provenance.json", text)

    def test_executed_notebook_report_and_readme_cover_scientific_contract(self) -> None:
        notebook_path = STEP5 / "notebooks/phase3_enhanced_modeling.ipynb"
        html_path = STEP5 / "notebooks/phase3_enhanced_modeling.html"
        report_path = STEP5 / "reports/phase3_enhanced_modeling_report.md"
        readme_path = STEP5 / "README.md"
        for path in (notebook_path, html_path, report_path, readme_path):
            self.assertTrue(path.exists(), f"missing deliverable: {path}")
        self.assertGreater(html_path.stat().st_size, 100_000)
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("execution_count") is not None for cell in code_cells))
        self.assertFalse(
            any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
        )
        report = report_path.read_text(encoding="utf-8")
        for token in (
            "RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "GPU 侧", "high-load steady-state",
            "不确定性", "限制", "下一阶段建议", "B4-Hardware", "Family holdout",
            "探索性候选", "获胜者选择偏差", "301 个单例",
        ):
            self.assertIn(token, report)
        self.assertNotIn("CPU/DRAM 建模结果", report)


class TestFinalValidation(unittest.TestCase):
    def test_final_validator_rebuilds_contracts_and_writes_passing_receipt(self) -> None:
        module_path = STEP5 / "scripts/validate_phase30.py"
        self.assertTrue(module_path.exists(), "final validator is missing")
        module = importlib.import_module("step5.scripts.validate_phase30")
        receipt = module.run_validations(ROOT, write=False)
        self.assertEqual("pass", receipt["status"])
        self.assertGreaterEqual(len(receipt["checks"]), 10)
        self.assertTrue(all(check["status"] == "pass" for check in receipt["checks"]))
        self.assertEqual(36600, receipt["summary"]["core_prediction_rows"])
        self.assertEqual("B4-Hardware__XGBoost", receipt["summary"]["exploratory_power_variant"])
        self.assertIn("artifact_hashes", receipt)

    def test_validator_is_independent_and_narrowly_names_secret_scan(self) -> None:
        source = (STEP5 / "scripts/validate_phase30.py").read_text(encoding="utf-8")
        self.assertNotIn("from step4.scripts.analyze_results import regression_metrics", source)
        self.assertNotIn("load_feature_bundles", source)
        self.assertNotIn("verify_source_freeze", source)
        self.assertIn("credential-pattern scan", source)


if __name__ == "__main__":
    unittest.main()
