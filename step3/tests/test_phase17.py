from __future__ import annotations

import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class FoundationTests(unittest.TestCase):
    def test_population_contract_rejects_count_drift(self) -> None:
        from step3.scripts.common import validate_population_counts

        self.assertEqual(validate_population_counts(694, 565), [])
        self.assertEqual(
            validate_population_counts(693, 564),
            ["expected 694 runs, found 693", "expected 565 stable runs, found 564"],
        )

    def test_run_key_is_order_stable_and_configuration_sensitive(self) -> None:
        from step3.scripts.common import RunKey

        base = {
            "task": "gpqa",
            "model_id": "Qwen/Qwen3-14B",
            "gpu_model": "B200",
            "num_gpus": 1,
            "max_num_seqs": 128,
            "tensor_parallel": 1,
            "expert_parallel": 1,
            "data_parallel": 1,
            "seed": 48105,
            "num_request_repeats": 1,
        }

        self.assertEqual(
            RunKey.from_mapping(base).canonical(),
            RunKey.from_mapping(dict(reversed(list(base.items())))).canonical(),
        )
        self.assertNotEqual(
            RunKey.from_mapping(base).canonical(),
            RunKey.from_mapping({**base, "num_request_repeats": 2}).canonical(),
        )

    def test_transient_request_ids_are_removed_from_persisted_errors(self) -> None:
        from step3.scripts.common import stable_error_message

        error = OSError("403 Client Error. (Request ID: Root=abc;uuid)\nAccess denied")

        self.assertEqual(stable_error_message(error), "OSError: 403 Client Error.\nAccess denied")

    def test_static_download_allowlist_rejects_model_weights(self) -> None:
        from step3.scripts.common import allowed_static_model_file

        for allowed in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            with self.subTest(allowed=allowed):
                self.assertTrue(allowed_static_model_file(allowed))

        for denied in (
            "model.safetensors",
            "pytorch_model.bin",
            "weights.pt",
            "model.gguf",
            "model.onnx",
            "nested/adapter_model.safetensors",
        ):
            with self.subTest(denied=denied):
                self.assertFalse(allowed_static_model_file(denied))


class AcquisitionTests(unittest.TestCase):
    def test_revision_manifest_covers_exactly_27_models(self) -> None:
        from step3.scripts.acquire_static_sources import validate_revision_manifest

        manifest = json.loads(
            Path("step3/config/model_revisions.json").read_text(encoding="utf-8")
        )
        errors = validate_revision_manifest(manifest)

        self.assertEqual(errors, [])
        self.assertEqual(len(manifest["models"]), 27)
        self.assertEqual(manifest["gpqa_dataset"]["revision"], "633f5ee89ab8ad4522a9f850766b73f62147ffdd")
        self.assertEqual(manifest["benchmark"]["commit"], "c1d6557fbf7c7c495749b0fcf649f71774eabde9")

    def test_static_tree_rejects_any_weight_file(self) -> None:
        from step3.scripts.acquire_static_sources import audit_static_tree

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(audit_static_tree(root), [])
            (root / "model-00001-of-00002.safetensors").write_bytes(b"forbidden")
            self.assertEqual(
                audit_static_tree(root),
                ["model-00001-of-00002.safetensors: forbidden static-model file"],
            )

    def test_source_receipt_hashes_files_and_never_persists_credentials(self) -> None:
        from step3.scripts.acquire_static_sources import build_source_receipt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"hidden_size": 4096}', encoding="utf-8")
            receipt = build_source_receipt(
                root,
                source_type="model_static",
                source_id="example/model",
                revision="a" * 40,
            )
            serialized = json.dumps(receipt, sort_keys=True)

            self.assertEqual(receipt["file_count"], 1)
            self.assertEqual(len(receipt["files"][0]["sha256"]), 64)
            self.assertNotIn("token", serialized.lower())
            self.assertNotIn("authorization", serialized.lower())


class PlannedWorkloadTests(unittest.TestCase):
    def test_gpqa_prompt_matches_pinned_choice_shuffle(self) -> None:
        from step3.scripts.reconstruct_planned_workload import build_gpqa_prompt

        item = {
            "Question": "Which option?",
            "Incorrect Answer 1": "wrong one ",
            "Incorrect Answer 2": " wrong two",
            "Incorrect Answer 3": "wrong three",
            "Correct Answer": "right",
        }
        rng = random.Random(48105)
        prompt, completion = build_gpqa_prompt(item, rng)

        self.assertEqual(
            prompt,
            "What is the correct answer to the following question: Which option?\n\n"
            "Choices:\n(A) wrong one\n(B) wrong three\n(C) right\n(D) wrong two",
        )
        self.assertEqual(completion, "(2) right")

    def test_repetition_scales_only_additive_planned_metrics(self) -> None:
        from step3.scripts.reconstruct_planned_workload import aggregate_planned_lengths

        one = aggregate_planned_lengths([2, 4, 8], repeat_count=1)
        three = aggregate_planned_lengths([2, 4, 8], repeat_count=3)

        for field in (
            "planned_request_count",
            "planned_total_input_tokens",
            "planned_sum_input_tokens_squared",
        ):
            self.assertEqual(three[field], 3 * one[field])
        for field in (
            "planned_unique_prompt_count",
            "planned_input_tokens_mean",
            "planned_input_tokens_p50",
            "planned_input_tokens_p90",
            "planned_input_tokens_p95",
            "planned_input_tokens_min",
            "planned_input_tokens_max",
            "planned_input_tokens_std",
        ):
            self.assertEqual(three[field], one[field])

    def test_prior_available_rows_are_preserved_exactly(self) -> None:
        from step3.scripts.reconstruct_planned_workload import merge_with_prior

        prior = pd.DataFrame(
            [
                {"run_key": "keep", "tokenization_status": "available", "planned_total_input_tokens": 10},
                {"run_key": "fill", "tokenization_status": "missing", "planned_total_input_tokens": None},
            ]
        )
        updates = pd.DataFrame(
            [
                {"run_key": "keep", "tokenization_status": "available", "planned_total_input_tokens": 999},
                {"run_key": "fill", "tokenization_status": "available", "planned_total_input_tokens": 20},
            ]
        )
        merged = merge_with_prior(prior, updates).set_index("run_key")

        self.assertEqual(merged.loc["keep", "planned_total_input_tokens"], 10)
        self.assertEqual(merged.loc["fill", "planned_total_input_tokens"], 20)

    def test_reconciliation_requires_exact_match_only_for_complete_runs(self) -> None:
        from step3.scripts.reconstruct_planned_workload import reconcile_token_totals

        complete = reconcile_token_totals(100, 100, completed=10, planned_count=10)
        incomplete = reconcile_token_totals(100, 80, completed=8, planned_count=10)

        self.assertEqual(complete["comparison_scope"], "all_requests_completed")
        self.assertTrue(complete["exact_match"])
        self.assertEqual(incomplete["comparison_scope"], "incomplete_run_diagnostic")
        self.assertIsNone(incomplete["exact_match"])


class EffectiveWorkloadTests(unittest.TestCase):
    def test_gpqa_and_multiturn_chat_message_shapes_match_client(self) -> None:
        from step3.scripts.build_effective_workload import build_chat_messages

        gpqa = build_chat_messages("gpqa", {"prompt": "question"})
        arena = build_chat_messages(
            "lm-arena-chat",
            {"prompt": ["first user", "assistant reply", "second user"]},
            system_prompt="system rule",
        )

        self.assertEqual(gpqa, [{"role": "user", "content": [{"type": "text", "text": "question"}]}])
        self.assertEqual([message["role"] for message in arena], ["system", "user", "assistant", "user"])
        self.assertEqual(arena[0]["content"], "system rule")
        self.assertEqual(arena[-1]["content"][0]["text"], "second user")

    def test_chat_template_uses_vllm_defaults(self) -> None:
        from step3.scripts.build_effective_workload import chat_effective_length

        class FakeTokenizer:
            chat_template = "{{ messages }}"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return [1, 2, 3, 4]

        tokenizer = FakeTokenizer()
        length = chat_effective_length(
            tokenizer,
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )

        self.assertEqual(length, 4)
        self.assertEqual(
            tokenizer.kwargs,
            {"tokenize": True, "add_generation_prompt": True},
        )

    def test_missing_chat_template_is_unresolved_not_guessed(self) -> None:
        from step3.scripts.build_effective_workload import chat_effective_length

        class NoTemplateTokenizer:
            chat_template = None

        with self.assertRaisesRegex(ValueError, "chat template unavailable"):
            chat_effective_length(NoTemplateTokenizer(), [{"role": "user", "content": "x"}])

    def test_fim_effective_length_uses_official_rendered_tokens_without_chat(self) -> None:
        from step3.scripts.build_effective_workload import effective_length_for_request

        class FakeTokenizer:
            def __call__(self, text):
                self.text = text
                return type("Tokenized", (), {"input_ids": [1, 2, 3]})()

        tokenizer = FakeTokenizer()
        length = effective_length_for_request(
            "sourcegraph-fim",
            "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            {"prefix": "pre", "suffix": "post"},
            tokenizer,
        )

        self.assertEqual(length, 3)
        self.assertEqual(tokenizer.text, "<|fim_prefix|>pre<|fim_suffix|>post<|fim_middle|>")

    def test_effective_repetition_scales_additive_metrics_only(self) -> None:
        from step3.scripts.build_effective_workload import aggregate_effective_lengths

        one = aggregate_effective_lengths([3, 5], 1)
        two = aggregate_effective_lengths([3, 5], 2)
        self.assertEqual(two["effective_total_input_tokens"], 2 * one["effective_total_input_tokens"])
        self.assertEqual(two["effective_sum_input_tokens_squared"], 2 * one["effective_sum_input_tokens_squared"])
        self.assertEqual(two["effective_input_tokens_mean"], one["effective_input_tokens_mean"])


class CanonicalModelFeatureTests(unittest.TestCase):
    def test_dense_gqa_maps_and_derives_head_dimension(self) -> None:
        from step3.scripts.build_model_features import canonicalize_model_config

        row, mapping = canonicalize_model_config(
            "org/dense",
            {
                "model_type": "qwen3",
                "num_hidden_layers": 40,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "vocab_size": 100,
                "max_position_embeddings": 40960,
            },
            {"architecture": "Dense Transformer", "weight_precision": "bfloat16", "total_params_billions": 14.0, "activated_params_billions": 14.0},
        )

        self.assertEqual(row["head_dim"], 128)
        self.assertEqual(row["architecture_family"], "dense_transformer")
        self.assertEqual(row["precision_bits"], 16)
        head_record = next(item for item in mapping if item["canonical_field"] == "head_dim")
        self.assertEqual(head_record["transform"], "hidden_size / num_attention_heads")

    def test_moe_and_hybrid_fields_are_explicit(self) -> None:
        from step3.scripts.build_model_features import canonicalize_model_config

        moe, _ = canonicalize_model_config(
            "org/moe",
            {"model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048, "intermediate_size": 6144, "moe_intermediate_size": 768, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8},
            {"architecture": "MoE", "weight_precision": "fp8", "total_params_billions": 30.0, "activated_params_billions": 3.0},
        )
        hybrid, _ = canonicalize_model_config(
            "org/hybrid",
            {"model_type": "nemotron_h", "num_hidden_layers": 5, "hidden_size": 10, "intermediate_size": 20, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 5, "hybrid_override_pattern": "M-*--"},
            {"architecture": "Mamba-Transformer Hybrid", "weight_precision": "bfloat16", "total_params_billions": 1.0, "activated_params_billions": 1.0},
        )

        self.assertEqual((moe["expert_count"], moe["active_expert_count"], moe["moe_intermediate_size"]), (128, 8, 768))
        self.assertEqual((hybrid["hybrid_mamba_layer_count"], hybrid["hybrid_attention_layer_count"], hybrid["hybrid_mlp_layer_count"]), (1, 1, 3))

    def test_missing_values_stay_null_and_benchmark_precision_wins(self) -> None:
        from step3.scripts.build_model_features import canonicalize_model_config

        row, mapping = canonicalize_model_config(
            "org/sparse",
            {"model_type": "unknown", "torch_dtype": "float32"},
            {"architecture": "Unknown", "weight_precision": "mxfp4", "total_params_billions": 2.0, "activated_params_billions": 1.0},
        )

        self.assertIsNone(row["num_layers"])
        self.assertIsNone(row["head_dim"])
        self.assertEqual(row["precision_bits"], 4)
        precision = next(item for item in mapping if item["canonical_field"] == "weight_precision")
        self.assertEqual(precision["source_field"], "benchmark.weight_precision")


class PhysicsFeatureTests(unittest.TestCase):
    def test_gqa_projection_and_causal_attention_flops(self) -> None:
        from step3.scripts.build_physics_features import causal_attention_flops, projection_flops

        self.assertEqual(projection_flops(10, 8, 4, 2, 2), 3840)
        self.assertEqual(causal_attention_flops(10, 4, 2), 3200)

    def test_dense_gated_non_gated_and_moe_mlp_flops(self) -> None:
        from step3.scripts.build_physics_features import mlp_flops

        self.assertEqual(mlp_flops(10, 8, 16, gated=True), 7680)
        self.assertEqual(mlp_flops(10, 8, 16, gated=False), 5120)
        self.assertEqual(mlp_flops(10, 8, 4, gated=True, active_experts=3), 5760)

    def test_decode_upper_bound_sums_growing_attention_context(self) -> None:
        from step3.scripts.build_physics_features import decode_work_upper_bound

        result = decode_work_upper_bound(
            input_len=3,
            output_cap=2,
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            dense_intermediate_size=16,
            dense_layer_count=1,
            moe_intermediate_size=None,
            moe_layer_count=0,
            active_experts=None,
            gated=True,
        )
        expected = 2 * (projection_flops := 384) + 6 * 2 * 8 * 16 + 4 * 4 * 2 * ((3 + 1) + (3 + 2))
        self.assertEqual(result, expected)

    def test_kv_weight_bytes_and_per_gpu_normalization(self) -> None:
        from step3.scripts.build_physics_features import ideal_per_gpu, kv_cache_bytes, weight_bytes

        self.assertEqual(kv_cache_bytes(10, 3, 2, 4, 2), 960)
        self.assertEqual(weight_bytes(1000, 8), 1000)
        self.assertEqual(ideal_per_gpu(1000, 4), 250)
        self.assertEqual(ideal_per_gpu(1000, 4, data_parallel=2, replicated_per_dp=True), 500)

    def test_invalid_or_unsupported_inputs_are_rejected(self) -> None:
        from step3.scripts.build_physics_features import projection_flops

        with self.assertRaises(ValueError):
            projection_flops(-1, 8, 4, 2, 2)


class MasterFeatureTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from step3.scripts.build_master_table import build_master_table

        cls.master, cls.manifest = build_master_table(Path("."))

    def test_join_preserves_exact_run_population_and_gpu_targets(self) -> None:
        master = self.master
        targets = {
            "y_gpu_steady_state_energy_joules",
            "y_gpu_steady_state_duration_seconds",
            "y_gpu_avg_power_watts",
            "y_gpu_energy_per_token_joules",
            "y_gpu_energy_per_request_joules",
        }

        self.assertEqual(len(master), 694)
        self.assertEqual(master["run_key"].nunique(), 694)
        self.assertEqual(int(master["diag_quality_is_stable"].sum()), 565)
        self.assertTrue(targets.issubset(master.columns))
        self.assertFalse(master[list(targets)].isna().any().any())

    def test_llama_is_retained_in_base_but_not_full_workload(self) -> None:
        master = self.master
        llama = master[master["diag_model_id"].str.startswith("meta-llama/")]

        self.assertEqual(len(llama), 132)
        self.assertTrue(llama["eligible_base_exante"].all())
        self.assertFalse(llama["eligible_full_workload"].any())
        self.assertEqual(int(master["eligible_base_exante"].sum()), 694)
        self.assertEqual(int(master["eligible_full_workload"].sum()), 464)
        self.assertEqual(int(master["eligible_physics_feature"].sum()), 383)
        self.assertEqual(int(master["eligible_primary_stable"].sum()), 565)

    def test_restriction_reasons_are_deterministic_and_ordered(self) -> None:
        from step3.scripts.build_master_table import restriction_reasons

        reason = restriction_reasons(
            {
                "diag_model_id": "meta-llama/example",
                "prov_tokenization_status": "missing",
                "prov_effective_status": "unresolved",
                "prov_config_status": "restricted",
                "prov_physics_status": "unsupported",
                "prov_effective_missing_reason": "historical template unavailable",
                "prov_physics_missing_reason": "unsupported hybrid architecture",
            }
        )

        self.assertEqual(
            reason,
            "llama_access_rejected;tokenizer_unavailable;"
            "chat_template_unresolved;config_missing;physics_unsupported_hybrid",
        )

    def test_role_manifest_keeps_result_fields_outside_x(self) -> None:
        roles = {item["column"]: item for item in self.manifest}

        self.assertEqual(roles["x_workload_benchmark_total_input_tokens"]["role"], "X_workload")
        self.assertTrue(roles["x_workload_benchmark_total_input_tokens"]["exante_allowed"])
        self.assertEqual(roles["diag_result_total_output_tokens"]["role"], "diagnostic_result")
        self.assertFalse(roles["diag_result_total_output_tokens"]["exante_allowed"])
        self.assertEqual(roles["y_gpu_avg_power_watts"]["role"], "Y_gpu")
        self.assertFalse(roles["y_gpu_avg_power_watts"]["exante_allowed"])
        self.assertEqual(roles["diag_quality_is_stable"]["role"], "diagnostic_quality")
        self.assertFalse(roles["diag_quality_is_stable"]["exante_allowed"])


class RaplIdentityTests(unittest.TestCase):
    def test_numeric_watts_never_resolve_a_zone_name(self) -> None:
        from step3.scripts.audit_rapl_identity import classify_zone_identity

        result = classify_zone_identity(
            zone_index=1,
            direct_evidence=[],
            observed_power_w={"median": 5932.08, "min": 2665.80, "max": 8615.44},
        )

        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["resolved_zone_name"])
        self.assertEqual(result["neutral_label"], "platform-like diagnostic")
        self.assertEqual(result["numeric_inference_used"], False)

    def test_resolved_identity_requires_direct_index_to_name_string(self) -> None:
        from step3.scripts.audit_rapl_identity import classify_zone_identity

        result = classify_zone_identity(
            zone_index=1,
            direct_evidence=[
                {
                    "source_path": "bounded/log.txt",
                    "line_number": 8,
                    "zone_index": 1,
                    "zone_name": "psys",
                    "evidence_type": "direct_index_name_mapping",
                }
            ],
            observed_power_w={"median": 5932.08},
        )

        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["resolved_zone_name"], "psys")
        self.assertEqual(result["evidence"][0]["line_number"], 8)


class FinalValidationTests(unittest.TestCase):
    def test_role_validator_rejects_post_run_field_in_x(self) -> None:
        from step3.scripts.validate_phase17 import validate_role_manifest

        bad = [
            {
                "column": "x_workload_total_output_tokens",
                "role": "X_workload",
                "exante_allowed": True,
            }
        ]

        failures = validate_role_manifest(bad)
        self.assertIn("x_workload_total_output_tokens: post-run field is marked as X", failures)
        self.assertIn("x_workload_total_output_tokens: unknown X column", failures)

    def test_role_validator_rejects_unknown_and_runtime_x_names(self) -> None:
        from step3.scripts.validate_phase17 import validate_role_manifest

        forbidden = (
            "x_workload_latency",
            "x_workload_duration_seconds",
            "x_workload_actual_batch",
            "x_workload_gpu_utilization",
            "x_workload_clock",
            "x_workload_temperature",
            "x_workload_ttft",
            "x_workload_unreviewed_but_innocent_name",
        )
        manifest = [
            {"column": column, "role": "X_workload", "exante_allowed": True}
            for column in forbidden
        ]

        failures = validate_role_manifest(manifest)

        for column in forbidden:
            self.assertIn(f"{column}: unknown X column", failures)

    def test_real_master_passes_key_target_range_and_coverage_checks(self) -> None:
        from step3.scripts.validate_phase17 import validate_master_table

        master = pd.read_parquet("step3/analysis/master_feature_table.parquet")
        self.assertEqual(validate_master_table(master), [])

        broken = pd.concat([master, master.iloc[[0]]], ignore_index=True)
        broken.loc[0, "y_gpu_avg_power_watts"] = -1
        failures = validate_master_table(broken)
        self.assertIn("expected 694 master rows, found 695", failures)
        self.assertIn("run_key contains 1 duplicates", failures)
        self.assertIn("y_gpu_avg_power_watts contains 1 negative values", failures)

    def test_master_validator_recomputes_values_and_row_level_eligibility(self) -> None:
        from step3.scripts.validate_phase17 import validate_master_table

        master = pd.read_parquet("step3/analysis/master_feature_table.parquet")
        broken = master.copy()
        physics_index = broken.index[broken["eligible_physics_feature"]][0]
        llama_index = broken.index[broken["diag_model_id"].str.startswith("meta-llama/")][0]
        full_index = broken.index[broken["eligible_full_workload"] & ~broken.index.to_series().eq(physics_index)][0]
        broken.loc[physics_index, "x_workload_benchmark_total_input_tokens"] = -1
        broken.loc[physics_index, "x_physics_prefill_total_flop_proxy"] = float("inf")
        broken.loc[physics_index, "x_workload_effective_input_tokens_p95"] = None
        broken.loc[full_index, "eligible_full_workload"] = False
        broken.loc[llama_index, "eligible_full_workload"] = True

        failures = validate_master_table(broken)

        self.assertTrue(any("x_workload_benchmark_total_input_tokens contains 1 negative" in item for item in failures))
        self.assertTrue(any("x_physics_prefill_total_flop_proxy contains 1 non-finite" in item for item in failures))
        self.assertTrue(any("full-workload eligibility differs" in item for item in failures))
        self.assertTrue(any("physics eligibility differs" in item for item in failures))

    def test_repository_scan_detects_secrets_weights_timelines_and_training(self) -> None:
        from step3.scripts.validate_phase17 import scan_forbidden_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "secret.txt").write_text("hf_" + "A" * 32, encoding="utf-8")
            (root / "weights.safetensors").write_bytes(b"x")
            (root / "raw.json").write_text('{"timeline": [1]}', encoding="utf-8")
            (root / "train.py").write_text("estimator.fit(x, y)", encoding="utf-8")

            findings = scan_forbidden_artifacts(root)

        self.assertEqual(
            {item["category"] for item in findings},
            {"credential", "model_weight", "raw_timeline", "formal_model_training"},
        )

    def test_repository_scan_includes_tracked_data_under_parent_named_data(self) -> None:
        from step3.scripts.validate_phase17 import scan_forbidden_artifacts

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "data" / "repository"
            tracked = repository / "step3" / "data" / "committed.json"
            tracked.parent.mkdir(parents=True)
            tracked.write_text(json.dumps({"credential": "hf_" + "B" * 32}), encoding="utf-8")
            subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(repository), "add", "step3/data/committed.json"], check=True)

            findings = scan_forbidden_artifacts(repository / "step3")

        self.assertIn(
            {"category": "credential", "path": "data/committed.json"},
            findings,
        )

    def test_reconciliation_validator_accepts_the_persisted_schema(self) -> None:
        from step3.scripts.validate_phase17 import validate_reconciliation

        report = json.loads(
            Path("step3/reports/planned_workload_reconciliation.json").read_text(encoding="utf-8")
        )

        self.assertEqual(validate_reconciliation(report), [])

    def test_mismatch_explanation_marks_historical_equivalence_unresolved(self) -> None:
        from step3.scripts.reconstruct_planned_workload import explain_reconciliation_record

        exact = explain_reconciliation_record(
            {"comparison_scope": "all_requests_completed", "exact_match": True}
        )
        mismatch = explain_reconciliation_record(
            {
                "comparison_scope": "all_requests_completed",
                "exact_match": False,
                "task": "gpqa",
                "gpu_model": "H100",
                "num_request_repeats": 2,
            }
        )

        self.assertEqual(exact["reconciliation_status"], "exact")
        self.assertEqual(
            mismatch["reconciliation_status"],
            "mismatch_unresolved_historical_equivalence",
        )
        self.assertIn("pinned code repeats an identical request vector", mismatch["reconciliation_reason"])
        self.assertIn("historical result does not persist", mismatch["reconciliation_reason"])


class NotebookBuildTests(unittest.TestCase):
    def test_blueprint_has_required_sections_and_no_training_cell(self) -> None:
        from step3.scripts.build_notebook import notebook_blueprint

        cells = notebook_blueprint()
        text = "\n".join(str(cell.get("source", "")) for cell in cells)

        for heading in (
            "tl;dr",
            "Context and methods",
            "Data-quality profile",
            "Workload coverage",
            "Benchmark vs effective input",
            "Canonical model missingness",
            "Physics eligibility",
            "Llama access impact",
            "Takeaways",
        ):
            self.assertIn(heading, text)
        self.assertNotIn(".fit(", text)
        self.assertNotIn("sklearn", text.lower())


if __name__ == "__main__":
    unittest.main()
