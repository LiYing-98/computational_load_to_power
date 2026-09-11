import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_mlenergy_v3_llm import (
    EXPECTED_TASKS,
    check_consistency,
    coverage_analysis,
    load_compact_records,
    profile_records,
)


def valid_row(**overrides):
    row = {
        "activated_params_billions": 8.0,
        "architecture": "Dense Transformer",
        "avg_batch_size": 7.0,
        "avg_output_len": 4.0,
        "avg_power_watts": 6.0,
        "data_parallel": 1,
        "energy_per_request_joules": 8.0,
        "energy_per_token_joules": 2.0,
        "expert_parallel": 1,
        "gpu_model": "H100",
        "max_num_seqs": 8,
        "median_itl_ms": 1.0,
        "model_id": "org/model",
        "nickname": "Model",
        "num_gpus": 1,
        "output_throughput_tokens_per_sec": 3.0,
        "p90_itl_ms": 2.0,
        "p95_itl_ms": 3.0,
        "p99_itl_ms": 4.0,
        "tensor_parallel": 1,
        "total_params_billions": 8.0,
        "weight_precision": "bfloat16",
    }
    row.update(overrides)
    return row


class LoadCompactRecordsTests(unittest.TestCase):
    def test_loads_exactly_the_three_text_llm_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task in EXPECTED_TASKS:
                payload = {
                    "task": task,
                    "task_display_name": task,
                    "configurations": [valid_row(model_id=f"org/{task}")],
                }
                (root / f"{task}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            records = load_compact_records(root)

        self.assertEqual(len(records), 3)
        self.assertEqual({row["task"] for row in records}, set(EXPECTED_TASKS))


class ConsistencyTests(unittest.TestCase):
    def test_flags_only_the_row_that_breaks_power_identity(self):
        rows = [
            {"task": "gpqa", **valid_row()},
            {"task": "gpqa", **valid_row(model_id="org/bad", avg_power_watts=5.0)},
        ]

        result = check_consistency(rows, relative_tolerance=1e-12)

        self.assertEqual(result["power_identity"]["checked"], 2)
        self.assertEqual(result["power_identity"]["failed"], 1)
        self.assertEqual(result["power_identity"]["failed_rows"], [1])

    def test_flags_non_monotonic_latency_percentiles(self):
        rows = [
            {
                "task": "lm-arena-chat",
                **valid_row(p90_itl_ms=3.0, p95_itl_ms=2.0, p99_itl_ms=4.0),
            }
        ]

        result = check_consistency(rows)

        self.assertEqual(result["itl_percentile_order"]["failed"], 1)


class ProfileTests(unittest.TestCase):
    def test_counts_candidate_key_replicates_and_nulls(self):
        rows = [
            {"task": "gpqa", **valid_row()},
            {
                "task": "gpqa",
                **valid_row(avg_batch_size=None, avg_power_watts=6.5),
            },
        ]

        result = profile_records(rows)

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["candidate_key"]["duplicate_groups"], 1)
        self.assertEqual(result["candidate_key"]["rows_in_duplicate_groups"], 2)
        self.assertEqual(result["nulls"]["avg_batch_size"]["count"], 1)
        self.assertEqual(result["nulls"]["avg_batch_size"]["rate"], 0.5)

    def test_gpu_pair_count_requires_distinct_gpu_models(self):
        rows = [
            {"task": "gpqa", **valid_row()},
            {"task": "gpqa", **valid_row(avg_power_watts=6.5)},
            {
                "task": "gpqa",
                **valid_row(gpu_model="B200", avg_power_watts=7.0),
            },
        ]

        result = coverage_analysis(rows)

        self.assertEqual(
            result["gpu_comparable_configuration_keys"]["paired_h100_b200_keys"],
            1,
        )


class ReportSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.audit = json.loads(
            (repo_root / "analysis" / "audit_results.json").read_text(
                encoding="utf-8"
            )
        )
        artifact = json.loads(
            (repo_root / "reports" / "artifact.json").read_text(encoding="utf-8")
        )
        cls.datasets = artifact["snapshot"]["datasets"]

    def test_headline_matches_machine_readable_audit(self):
        headline = self.datasets["headline"][0]
        profile = self.audit["profile"]
        coverage = self.audit["coverage_analysis"]

        self.assertEqual(headline["records"], profile["row_count"])
        self.assertEqual(headline["public_fields"], profile["column_count"])
        self.assertEqual(headline["models"], coverage["unique_models"])
        self.assertEqual(
            headline["paired_keys"],
            coverage["gpu_comparable_configuration_keys"][
                "paired_h100_b200_keys"
            ],
        )

    def test_task_gpu_chart_rows_match_cross_tab(self):
        chart_counts = {
            (row["task"], row["gpu"]): row["count"]
            for row in self.datasets["task_gpu"]
        }
        audit_counts = {
            (task, gpu): count
            for task, gpu_counts in self.audit["coverage_analysis"]["cross_tabs"][
                "task_by_gpu"
            ].items()
            for gpu, count in gpu_counts.items()
        }

        self.assertEqual(chart_counts, audit_counts)

    def test_gpu_count_chart_matches_profile(self):
        chart_counts = {
            row["num_gpus"]: row["count"]
            for row in self.datasets["num_gpus"]
        }

        self.assertEqual(chart_counts, self.audit["profile"]["coverage"]["num_gpus"])


if __name__ == "__main__":
    unittest.main()
