from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


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


if __name__ == "__main__":
    unittest.main()
