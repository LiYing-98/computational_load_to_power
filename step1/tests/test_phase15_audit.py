import json
import tempfile
import unittest
from pathlib import Path

from step1.scripts.audit_full_parquet import (
    check_energy_power_identity,
    classify_field,
    compare_with_phase1,
)
from step1.scripts.select_representative_runs import (
    allowed_metadata_path,
    select_representative_runs,
)
from step1.scripts.audit_workload_metadata import (
    classify_workload_value,
    find_workload_fields,
)


class FieldClassificationTests(unittest.TestCase):
    def test_execution_time_boundary_is_explicit(self):
        self.assertEqual(classify_field("max_num_seqs")["category"], "pre_known")
        self.assertEqual(classify_field("num_request_repeats")["category"], "pre_known")
        self.assertEqual(
            classify_field("avg_output_len")["category"], "post_run_leakage"
        )
        self.assertEqual(
            classify_field("unknown_field")["category"], "unclassified"
        )

    def test_energy_and_power_are_targets_not_features(self):
        for field in (
            "steady_state_energy_joules",
            "energy_per_token_joules",
            "energy_per_request_joules",
            "avg_power_watts",
        ):
            with self.subTest(field=field):
                classification = classify_field(field)
                self.assertEqual(classification["category"], "target")
                self.assertFalse(classification["allow_as_x_pre"])


class ConsistencyTests(unittest.TestCase):
    def test_energy_power_identity_flags_only_corrupted_row(self):
        rows = [
            {
                "steady_state_energy_joules": 100.0,
                "steady_state_duration_seconds": 20.0,
                "avg_power_watts": 5.0,
            },
            {
                "steady_state_energy_joules": 100.0,
                "steady_state_duration_seconds": 20.0,
                "avg_power_watts": 4.0,
            },
        ]

        result = check_energy_power_identity(rows, relative_tolerance=1e-12)

        self.assertEqual(result["checked"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["failed_rows"], [1])


class Phase1ComparisonTests(unittest.TestCase):
    def test_comparison_projects_only_stable_text_llm_rows(self):
        visible = {
            "task": "gpqa",
            "model_id": "org/model",
            "gpu_model": "H100",
            "num_gpus": 1,
            "max_num_seqs": 8,
            "avg_power_watts": 100.0,
        }
        full_rows = [
            {**visible, "is_stable": True, "seed": 1},
            {**visible, "is_stable": False, "seed": 2},
            {
                **visible,
                "task": "image-chat",
                "is_stable": True,
                "seed": 3,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            compact_dir = Path(tmp)
            (compact_dir / "gpqa.json").write_text(
                json.dumps(
                    {
                        "task": "gpqa",
                        "configurations": [
                            {key: value for key, value in visible.items() if key != "task"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for task in ("lm-arena-chat", "sourcegraph-fim"):
                (compact_dir / f"{task}.json").write_text(
                    json.dumps({"task": task, "configurations": []}),
                    encoding="utf-8",
                )

            result = compare_with_phase1(full_rows, compact_dir)

        self.assertEqual(result["full_text_llm_rows"], 2)
        self.assertEqual(result["full_stable_text_llm_rows"], 1)
        self.assertEqual(result["phase1_compact_rows"], 1)
        self.assertEqual(result["multiset_difference"], 0)


class RepresentativeSelectionTests(unittest.TestCase):
    def test_selection_is_bounded_deterministic_and_coverage_first(self):
        rows = []
        tasks = ("gpqa", "lm-arena-chat", "sourcegraph-fim")
        for index, task in enumerate(tasks):
            for gpu in ("H100", "B200"):
                for architecture, params in (("Dense Transformer", 8.0), ("MoE", 235.0)):
                    num_gpus = 1 if (index + int(params)) % 2 == 0 else 4
                    rows.append(
                        {
                            "task": task,
                            "model_id": f"org/{architecture}-{params}-{task}-{gpu}",
                            "architecture": architecture,
                            "total_params_billions": params,
                            "gpu_model": gpu,
                            "num_gpus": num_gpus,
                            "max_num_seqs": 8 if gpu == "H100" else 512,
                            "tensor_parallel": num_gpus,
                            "expert_parallel": 1,
                            "data_parallel": 1,
                            "seed": 48105,
                            "num_request_repeats": 1,
                            "is_stable": True,
                            "results_path": f"llm/{task}/{gpu}/{params}/results.json",
                            "prometheus_path": f"llm/{task}/{gpu}/{params}/prometheus.json",
                        }
                    )

        first = select_representative_runs(rows, limit=12)
        second = select_representative_runs(list(reversed(rows)), limit=12)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 12)
        self.assertEqual({row["task"] for row in first}, set(tasks))
        self.assertEqual({row["gpu_model"] for row in first}, {"H100", "B200"})
        self.assertTrue({"Dense Transformer", "MoE"}.issubset({row["architecture"] for row in first}))
        self.assertTrue({"single", "multi"}.issubset({row["card_count_class"] for row in first}))

    def test_only_results_json_is_an_allowed_download_target(self):
        self.assertTrue(allowed_metadata_path("llm/a/run/results.json"))
        self.assertFalse(allowed_metadata_path("llm/a/run/prometheus.json"))
        self.assertFalse(allowed_metadata_path("llm/a/run/timeline.json"))
        self.assertFalse(allowed_metadata_path("training/a/run/results.json"))
        self.assertFalse(allowed_metadata_path("llm/a/model.safetensors"))


class WorkloadBoundaryTests(unittest.TestCase):
    def test_recursive_discovery_and_leakage_boundary(self):
        metadata = {
            "max_output_tokens": 4096,
            "num_unique_prompts": 1024,
            "total_output_tokens": 900000,
            "output_throughput": 3000.0,
            "steady_state_measurement": {"time": 200.0, "gpu_energy": {"0": 1.0}},
        }

        fields = find_workload_fields(metadata)

        self.assertIn("steady_state_measurement.gpu_energy.0", fields)
        self.assertEqual(
            classify_workload_value("max_output_tokens", 4096, {})["category"],
            "pre_known",
        )
        for path in (
            "total_output_tokens",
            "output_throughput",
            "steady_state_measurement.time",
        ):
            with self.subTest(path=path):
                result = classify_workload_value(path, fields.get(path), {})
                self.assertEqual(result["category"], "post_run_leakage")
                self.assertFalse(result["allow_as_x_pre"])

    def test_prompt_tokens_are_only_pre_derivable_with_frozen_inputs(self):
        unavailable = classify_workload_value(
            "planned_total_input_tokens", None, {"frozen_requests_and_tokenizer": False}
        )
        available = classify_workload_value(
            "planned_total_input_tokens", 1234, {"frozen_requests_and_tokenizer": True}
        )

        self.assertEqual(unavailable["category"], "missing_precondition")
        self.assertFalse(unavailable["allow_as_x_pre"])
        self.assertEqual(available["category"], "pre_derivable")
        self.assertTrue(available["allow_as_x_pre"])


if __name__ == "__main__":
    unittest.main()
