from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STEP7 = ROOT / "step7"


class FeatureContracts(unittest.TestCase):
    def test_explicitly_disabled_qwen_windows_are_structural_not_unknown(self):
        """Catch median-imputing a disabled Qwen window from another architecture."""
        from step7.scripts.amend_l2x import corrected_table
        table=pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet")
        corrected,evidence=corrected_table(table)
        mask=table.diag_parent_lineage.eq("Qwen")
        self.assertEqual((int(mask.sum()),len(evidence)),(276,12))
        for c in ["x_p5_extended_sliding_window","x_p5_extended_sliding_window_applicable","x_p5_extended_sliding_window_unknown"]:
            self.assertTrue(corrected.loc[mask,c].eq(0).all())
        pd.testing.assert_frame_equal(corrected.loc[~mask],table.loc[~mask])

    def test_supplemental_dictionary_defines_every_frozen_feature_without_generic_units(self):
        """Each raw/derived feature needs an explicit source, unit, formula and applicability."""
        from step7.scripts.audit_feature_dictionary import definition
        hierarchy=json.loads((STEP7/"config/feature_hierarchy_phase5.json").read_text())["levels"]
        columns={c for bundle in hierarchy.values() for c in bundle["columns"]}
        for column in columns:
            entry=definition(column)
            for field in ["source","unit","formula","applicability","exante_reason","limitation"]:
                self.assertTrue(entry[field],(column,field))
        self.assertEqual(definition("x_workload_benchmark_input_tokens_second_moment")["unit"],"tokens^2/request")
        self.assertEqual(definition("x_p5_weight_hbm_ratio")["unit"],"dimensionless")
        with self.assertRaises(ValueError):
            definition("y_gpu_avg_power_watts")

    def test_master_preserves_targets_and_exante_layer_support(self):
        """Catch unstable-row inclusion, target mutation, or leaked formal features."""
        result = subprocess.run(
            [sys.executable, "-m", "step7.scripts.build_features"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        d = pd.read_parquet(STEP7 / "analysis/phase5_master_table.parquet")
        prior = pd.read_parquet(ROOT / "step6/analysis/phase4_power_modeling_table.parquet")
        self.assertEqual((len(d), d.run_key.nunique(), d.diag_model_id.nunique()), (565, 565, 27))
        np.testing.assert_array_equal(d.y_gpu_avg_power_watts, prior.y_gpu_avg_power_watts)
        layers = json.loads((STEP7 / "config/feature_hierarchy_phase5.json").read_text())
        for name in ["L0-Basic", "L0A-ActiveAware", "L1-Deployment"]:
            self.assertEqual(int(d[layers["levels"][name]["eligibility"]].sum()), 565)
        for name, bundle in layers["levels"].items():
            features = bundle["categorical"] + bundle["numeric"]
            self.assertFalse({"diag_model_id", "diag_parent_lineage", "diag_subfamily"}.intersection(features))
            self.assertFalse(any(x.startswith(("y_", "diag_result_", "diag_load_")) for x in features))
            if name != "L1-TaskSensitivity":
                self.assertNotIn("x_protocol_task", features)

    def test_deployment_and_physics_formulas_are_hand_checkable(self):
        """Catch using TP or total GPUs where the audited model-parallel group is required."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.build_features"))
        from step7.scripts.build_features import deployment_proxies, compact_physics
        values = deployment_proxies(8, 2, 128, 160e9, 16e9, 80e9)
        self.assertEqual(values["model_parallel_gpus_per_replica"], 4)
        self.assertEqual(values["max_num_seqs_per_gpu"], 16)
        self.assertEqual(values["weight_hbm_ratio"], 0.5)
        self.assertEqual(values["active_weight_hbm_ratio"], 0.05)
        with self.assertRaises(ValueError):
            deployment_proxies(8, 3, 128, 160e9, 16e9, 80e9)
        feature = compact_physics(
            layers=2, hidden=8, heads=2, kv_heads=1, head_dim=4,
            intermediate=16, active_experts=1, mlp_terms=3,
            mean_length=10, second_moment=104, p95_length=12,
            max_length=14, kv_element_bytes=2,
        )
        # Q/K/V/output matrix dimensions: 64+32+32+64 = 192 per layer.
        self.assertEqual(feature["prefill_projection_flop_per_request"], 7680)
        self.assertEqual(feature["kv_mean_request_bytes"], 320)
        self.assertEqual(feature["kv_p95_request_bytes"], 384)


class ProtocolContracts(unittest.TestCase):
    def test_elasticnet_converges_with_low_regularization_and_correlated_proxies(self):
        """Catch accepting a truncated coordinate-descent solution for collinear basic fields."""
        from sklearn.exceptions import ConvergenceWarning
        from step7.scripts.modeling import estimator
        d = pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet").sort_values("run_key")
        b = json.loads((STEP7/"config/feature_hierarchy_phase5.json").read_text())["levels"]["L0-Basic"]
        c = {"alpha":0.00033245979322709415,"l1_ratio":0.95}
        model = estimator("ElasticNet",c,b)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(d[b["numeric"]+b["categorical"]],d.y_gpu_avg_power_watts)
        self.assertTrue(np.isfinite(model.predict(d[b["numeric"]+b["categorical"]])).all())

    def test_protocol_freezes_shared_folds_and_equal_candidate_budgets(self):
        """Catch overlapping model folds or level-dependent tuning budgets."""
        result = subprocess.run([sys.executable, "-m", "step7.scripts.freeze_protocol"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads((STEP7 / "config/experiment_plan.json").read_text())
        sampled = json.loads((STEP7 / "config/sampled_hyperparameter_candidates.json").read_text())
        self.assertEqual(len(sampled["algorithms"]), 8)
        for algorithm, candidates in sampled["algorithms"].items():
            self.assertEqual(len(candidates), 1 if algorithm == "DummyMean" else 8)
        for cohort in plan["cohorts"]:
            folds = pd.read_csv(STEP7 / f"splits/outer_model_folds_{cohort}.csv")
            self.assertEqual(folds.outer_fold.nunique(), 5)
            self.assertTrue(folds.groupby("diag_model_id").outer_fold.nunique().eq(1).all())
        strategies = {x["target_strategy"] for x in plan["experiments"]}
        self.assertEqual(strategies, {"T-total", "G-perGPU"})

    def test_per_gpu_candidate_selection_scores_restored_total_watts(self):
        """Catch selecting G-perGPU hyperparameters using the wrong error scale."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.modeling"))
        from step7.scripts.modeling import selection_mae, target_values, metrics
        d = pd.DataFrame({"y_gpu_avg_power_watts": [100., 800.], "x_hardware_num_gpus": [1, 8]})
        np.testing.assert_array_equal(target_values(d, "G-perGPU"), [100., 100.])
        # Both 10W/GPU errors => total errors 10W,80W => MAE45W.
        self.assertEqual(selection_mae(d, np.array([110., 110.]), "G-perGPU"), 45.)
        measured = metrics([100., 800.], [110., 880.], [1, 8])
        self.assertEqual(measured["mae"], 45.)
        self.assertEqual(measured["per_gpu_mae"], 10.)

    def test_nested_cv_handles_unseen_categories_and_preserves_group_isolation(self):
        """Catch global preprocessing or model identity entering a fitted predictor."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.modeling"))
        from step7.scripts.modeling import fit_outer
        frame = pd.DataFrame({"run_key":list('abcdefghi'), "diag_model_id":["a"]*3+["b"]*3+["c"]*3, "x_num":np.arange(9)+1., "x_cat":["seen"]*9, "x_hardware_num_gpus":[1]*9, "y_gpu_avg_power_watts":np.arange(9)*10+100.})
        test = pd.DataFrame({"run_key":["held"], "diag_model_id":["d"], "x_num":[10.], "x_cat":["unseen"], "x_hardware_num_gpus":[8], "y_gpu_avg_power_watts":[1200.]})
        bundle = {"numeric":["x_num","x_hardware_num_gpus"], "categorical":["x_cat"]}
        pred, info, model = fit_outer(frame, test, "Ridge", bundle, [{"alpha":1.}, {"alpha":10.}], "G-perGPU")
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred).all())
        self.assertNotIn("d", info["train_models"])
        self.assertEqual(info["selection_metric"], "row_weighted_reconstructed_total_mae")
        self.assertEqual(len(info["inner_scores"]), 2)


class MatrixContracts(unittest.TestCase):
    def test_checkpoint_reuse_is_exact_and_changes_invalidate_it(self):
        """Catch stale cached predictions after candidate, source, grouping, or code changes."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.run_matrix"))
        from step7.scripts.run_matrix import checkpoint_key, valid_checkpoint
        key = checkpoint_key({"protocol":"a", "group_digest":"b", "candidates":[1]})
        self.assertNotEqual(key,checkpoint_key({"protocol":"a","group_digest":"c","candidates":[1]}))
        self.assertNotEqual(key,checkpoint_key({"protocol":"a","group_digest":"b","candidates":[2]}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fold.parquet"
            pd.DataFrame({"run_key":["a","b"],"checkpoint_fingerprint":[key,key]}).to_parquet(path,index=False)
            self.assertTrue(valid_checkpoint(path,key,{"a","b"}))
            self.assertFalse(valid_checkpoint(path,key,{"a"}))
            self.assertFalse(valid_checkpoint(path,"changed",{"a","b"}))


class PairedAnalysisContracts(unittest.TestCase):
    def test_model_cluster_bootstrap_retains_pairing_and_distinct_weighting(self):
        """Catch row bootstrap or confusing equal-model with row-weighted deltas."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.analyze_results"))
        from step7.scripts.analyze_results import paired_effect
        before = pd.DataFrame({"run_key":["a","b","c"],"diag_model_id":["m1","m1","m2"],"outer_fold":[0,0,1],"y_gpu_avg_power_watts":[100.,100.,100.],"prediction":[120.,120.,120.]})
        after = before.copy()
        after["prediction"]=[110.,110.,130.]
        summary,draws = paired_effect(before,after,n_bootstrap=1000)
        self.assertAlmostEqual(summary["row_weighted_delta_mae"],-10/3)
        self.assertEqual(summary["equal_model_delta_mae"],0.)
        self.assertEqual(len(draws),1000)
        self.assertEqual(summary["n_models"],2)
        bad = after.iloc[:-1]
        with self.assertRaises(ValueError):
            paired_effect(before,bad,n_bootstrap=1000)


class CandidateContracts(unittest.TestCase):
    def test_candidate_choice_ignores_stress_and_wrong_support_rows(self):
        """Catch using a post-hoc or smaller-support result to choose a deployed tier."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.fit_candidates"))
        from step7.scripts.fit_candidates import choose_level
        matrix = pd.DataFrame([
            {"feature_level":"L0-Basic","analysis_role":"main","cohort":"full","mae":100.,"rmse":150.,"algorithm":"Ridge","target_strategy":"T-total"},
            {"feature_level":"L0-Basic","analysis_role":"main","cohort":"full","mae":110.,"rmse":140.,"algorithm":"RandomForest","target_strategy":"G-perGPU"},
            {"feature_level":"L0-Basic","analysis_role":"post_hoc","cohort":"full","mae":1.,"rmse":1.,"algorithm":"XGBoost","target_strategy":"T-total"},
        ])
        choice=choose_level(matrix,"L0-Basic")
        self.assertEqual(choice["algorithm"],"Ridge")
        self.assertEqual(choice["target_strategy"],"T-total")


class StressContracts(unittest.TestCase):
    def test_stress_cache_key_tracks_shared_engine_and_protocol(self):
        """A preprocessing/target engine revision must invalidate every stress checkpoint."""
        from step7.scripts.run_stress import stress_dependencies
        deps=stress_dependencies()
        self.assertTrue({"modeling.py","common.py","protocol","source_freeze"}.issubset(deps))
        from step7.scripts.run_matrix import checkpoint_key
        old=checkpoint_key({"dependencies":deps})
        changed={**deps,"modeling.py":"changed-engine"}
        self.assertNotEqual(old,checkpoint_key({"dependencies":changed}))

    def test_stress_manifests_keep_the_actual_task_lineage_support(self):
        """Catch labels/rows not matching post-hoc training and held-out conditions."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.run_stress"))
        from step7.scripts.run_stress import make_stress_manifests
        d=pd.DataFrame({"run_key":list("abcdefg"),"diag_model_id":list("abcdefg"),"diag_parent_lineage":["A","B","C","A","B","C","A"],"diag_architecture_class":["dense","moe","hybrid","dense","moe","hybrid","dense"],"x_protocol_task":["t"]*6+["single"]})
        manifests=make_stress_manifests(d)
        within=manifests["within_task_lineage"]
        self.assertEqual(len(within),6)
        self.assertEqual(set(within.x_protocol_task),{"t"})
        for name,manifest in manifests.items():
            self.assertFalse(manifest.run_key.duplicated().any())
            self.assertTrue(manifest.post_hoc_only.all())


class DeliveryContracts(unittest.TestCase):
    def test_saved_notebook_loads_evidence_from_its_own_directory(self):
        """Catch ROOT=Path.cwd() assuming notebooks are launched from repository root."""
        notebook=json.loads((STEP7/"notebooks/phase5_hierarchical_modeling.ipynb").read_text(encoding="utf-8"))
        cell=next(c for c in notebook["cells"] if c["cell_type"]=="code")
        source="".join(cell["source"])+"\nprint(str(ROOT.resolve()))"
        result=subprocess.run([sys.executable,"-c",source],cwd=STEP7/"notebooks",capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(Path(result.stdout.strip()),ROOT)

    def test_cluster_interval_plot_does_not_require_point_inside_percentile_ci(self):
        """Percentile cluster intervals can exclude the original point; preserve endpoints."""
        import matplotlib.pyplot as plt
        from step7.scripts.build_delivery import draw_cluster_intervals
        fig,ax=plt.subplots()
        draw_cluster_intervals(ax,np.array([0.]),np.array([0.]),np.array([1.]),np.array([2.]),color="blue",marker="o",label="fixture")
        np.testing.assert_array_equal(ax.collections[0].get_segments()[0],[[1.,0.],[2.,0.]])
        plt.close(fig)

    def test_delivery_refuses_to_present_a_partial_matrix_as_final(self):
        """Catch issuing final research conclusions from incomplete candidate cells."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.build_delivery"))
        from step7.scripts.build_delivery import require_complete_matrix
        with self.assertRaises(ValueError):
            require_complete_matrix({"status":"in_progress","completed_experiments":1,"expected_experiments":2})
        require_complete_matrix({"status":"complete","completed_experiments":2,"expected_experiments":2})


class IndependentValidatorContracts(unittest.TestCase):
    def test_final_validator_rejects_stale_stress_fingerprints(self):
        """A self-consistent stress table cannot bypass shared-engine provenance checks."""
        from unittest.mock import patch
        from step7.scripts.validate_phase50 import candidate_and_stress
        reader=pd.read_parquet
        def altered(path,*args,**kwargs):
            rows=reader(path,*args,**kwargs)
            if Path(path).name=="post_hoc_stress_predictions.parquet":
                rows=rows.copy()
                rows["checkpoint_fingerprint"]="deliberately_stale_stress_fixture"
            return rows
        with patch("step7.scripts.validate_phase50.pd.read_parquet",side_effect=altered):
            with self.assertRaisesRegex(AssertionError,"fingerprint"):
                candidate_and_stress()

    def test_final_validator_rejects_stale_oof_fingerprints(self):
        """Self-consistent metrics must not certify an obsolete preprocessing engine."""
        from unittest.mock import patch
        from step7.scripts.validate_phase50 import oof_predictions
        reader=pd.read_parquet
        def altered(path,*args,**kwargs):
            rows=reader(path,*args,**kwargs)
            if Path(path).name.startswith("oof_predictions_"):
                rows=rows.copy()
                rows["checkpoint_fingerprint"]="deliberately_stale_test_fixture"
            return rows
        with patch("step7.scripts.validate_phase50.pd.read_parquet",side_effect=altered):
            with self.assertRaisesRegex(AssertionError,"fingerprint"):
                oof_predictions()

    def test_slice_reconciliation_checks_total_and_gpu_units_and_support(self):
        """Reject mislabeled slice membership or swapped W / W-per-GPU statistics."""
        from step7.scripts.validate_phase50 import reconcile_slice
        rows=pd.DataFrame({"diag_model_id":["a","b"],"y_gpu_avg_power_watts":[100.,800.],"prediction":[110.,880.],"x_hardware_num_gpus":[1,8]})
        from step7.scripts.validate_phase50 import independent_metrics
        summary={"n_rows":2,"n_models":2,"residual_median":45.,**independent_metrics(rows.y_gpu_avg_power_watts,rows.prediction,rows.x_hardware_num_gpus)}
        reconcile_slice(rows,summary)
        with self.assertRaises(AssertionError):
            reconcile_slice(rows,{**summary,"per_gpu_mae":45.})
        with self.assertRaises(AssertionError):
            reconcile_slice(rows,{**summary,"n_rows":1})

    def test_independent_metrics_reconcile_total_and_average_gpu_scale(self):
        """Catch total/per-GPU unit errors independently of the training metric helper."""
        self.assertIsNotNone(importlib.util.find_spec("step7.scripts.validate_phase50"))
        from step7.scripts.validate_phase50 import independent_metrics
        values=independent_metrics(np.array([100.,800.]),np.array([110.,880.]),np.array([1.,8.]))
        self.assertEqual(values["mae"],45.)
        self.assertEqual(values["per_gpu_mae"],10.)
        self.assertEqual(values["per_gpu_rmse"],10.)
        self.assertEqual(values["signed_bias"],45.)


if __name__ == "__main__":
    unittest.main()
