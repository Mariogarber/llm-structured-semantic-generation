from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import read_jsonl
from llm_structured_semantic_generation.structure import yaml_to_blocks


def load_dpo_candidates_module():
    module_name = "test_build_kubernetes_dpo_candidates"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "build_kubernetes_dpo_candidates.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not load DPO candidate generation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class KubernetesDPOCandidateGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_dpo_candidates_module()

    def test_candidate_specs_cycle_temperatures_and_default_seed_sequence(self) -> None:
        specs = self.module.build_candidate_specs(
            num_candidates=4,
            temperatures=(0.1, 0.2),
            top_p=0.9,
            base_seed=11,
        )

        self.assertEqual([spec.candidate_id for spec in specs], ["c00", "c01", "c02", "c03"])
        self.assertEqual([spec.temperature for spec in specs], [0.1, 0.2, 0.1, 0.2])
        self.assertEqual([spec.seed for spec in specs], [11, 12, 13, 14])

    def test_candidate_tasks_are_unique_per_prompt_and_candidate(self) -> None:
        row = {
            "sample_id": "q1",
            "prompt_variant": "question",
            "prompt": "Create a ConfigMap.",
            "target": "<blocks>\n</blocks>",
            "target_yaml_normalized": "apiVersion: v1\nkind: ConfigMap\n",
        }
        specs = self.module.build_candidate_specs(
            num_candidates=2,
            temperatures=(0.2,),
            top_p=0.9,
            base_seed=7,
        )

        tasks = self.module.build_candidate_tasks([row], specs)

        self.assertEqual([task[2] for task in tasks], ["q1::question::c00", "q1::question::c01"])

    def test_select_sft_rows_applies_offset_before_limit(self) -> None:
        rows = [{"sample_id": f"q{index}", "prompt_variant": "question"} for index in range(10)]

        selected = self.module.select_sft_rows(rows, sample_offset=2, max_samples=3)

        self.assertEqual([row["sample_id"] for row in selected], ["q2", "q3", "q4"])

    def test_preference_score_matches_documented_proxy_formula(self) -> None:
        evaluation = {
            "yaml_parse_ok": True,
            "block_parse_ok": True,
            "prompt_requirement_f1": 0.8,
            "kubernetes_domain_validity_score": 0.6,
            "required_field_complete_resource_rate": 1.0,
            "level_exact_match_rate": 0.9,
            "kubernetes_domain_gate_pass": True,
            "reference_document_count": 1,
            "prediction_document_count": 1,
            "kind_sequence_match_rate": 1.0,
            "line_count_reference": 10,
            "line_count_prediction": 11,
        }

        score = self.module.compute_preference_score(evaluation)

        expected = 0.8 + 0.75 * 0.6 + 0.5 * 1.0 + 0.25 * 0.9 + 0.25 * 1.0
        self.assertAlmostEqual(score["preference_score"], expected)
        self.assertFalse(score["hard_invalid"])

    def test_preference_score_zeroes_hard_invalid_candidates(self) -> None:
        score = self.module.compute_preference_score(
            {
                "yaml_parse_ok": False,
                "block_parse_ok": False,
                "prompt_requirement_f1": 1.0,
            }
        )

        self.assertEqual(score["preference_score"], 0.0)
        self.assertTrue(score["hard_invalid"])

    def test_candidate_metrics_can_be_reconciled_from_primary_artifact(self) -> None:
        import tempfile

        yaml_text = (
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "metadata:\n"
            "  name: app\n"
        )
        candidate_row = {
            "candidate_uid": "q1::question::c00",
            "unit_id": "q1::question",
            "candidate_id": "c00",
            "candidate_index": 0,
            "sample_id": "q1",
            "prompt_variant": "question",
            "split": "train",
            "prompt": "Create a ConfigMap named app.",
            "reference_yaml": yaml_text,
            "checkpoint": "results/sft/checkpoint-step-1",
            "generation_ok": True,
            "generation_error_type": None,
            "generation_error": None,
            "generation_config": {
                "candidate_id": "c00",
                "candidate_index": 0,
                "temperature": 0.2,
                "seed": 42,
                "top_p": 0.9,
            },
            "input_token_count": 10,
            "generated_token_count": 20,
            "model_output_text": "<blocks>",
            "structured_output_parse_success": True,
            "predicted_blocks": [block.to_dict() for block in yaml_to_blocks(yaml_text)],
            "reconstructed_yaml": yaml_text,
            "parser_errors": [],
            "reconstruction_errors": [],
            "generated_at": "2026-05-24T00:00:00Z",
        }

        with tempfile.TemporaryDirectory() as tmp:
            run = self.module.ResumableRun.initialize(
                run_dir=Path(tmp) / "dpo-candidates" / "run-001",
                config={"run_id": "run-001", "resume_signature": {"stage": "test"}},
                total_units=1,
                unit_id_field="candidate_uid",
                primary_artifact_name="candidates",
                artifact_paths={
                    "candidates": self.module.CANDIDATES_ARTIFACT,
                    "candidate_metrics": self.module.CANDIDATE_METRICS_ARTIFACT,
                },
            )
            run.record_batch([candidate_row])

            resumed = self.module.ResumableRun.initialize(
                run_dir=run.run_dir,
                config={"run_id": "run-001", "resume_signature": {"stage": "test"}},
                total_units=1,
                unit_id_field="candidate_uid",
                primary_artifact_name="candidates",
                artifact_paths={
                    "candidates": self.module.CANDIDATES_ARTIFACT,
                    "candidate_metrics": self.module.CANDIDATE_METRICS_ARTIFACT,
                },
            )
            self.module.reconcile_candidate_metrics(resumed)

            metric_rows = read_jsonl(resumed.artifact_paths["candidate_metrics"])
            self.assertEqual(len(metric_rows), 1)
            self.assertEqual(metric_rows[0]["candidate_uid"], "q1::question::c00")
            self.assertGreater(metric_rows[0]["preference_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
