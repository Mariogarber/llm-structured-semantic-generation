from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import importlib.util
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json, write_jsonl
from llm_structured_semantic_generation.dpo_preference_annotation import (
    AnnotationPaths,
    PreferenceAnnotationError,
    append_agent_suggestion,
    append_human_preference,
    build_decision_packet,
    export_final_preferences,
    initialize_annotation_run,
    load_dataset_index,
    load_review_units,
    get_labeling_guide,
    write_annotation_state,
)
from llm_structured_semantic_generation.prompt_requirement_audit import (
    append_prompt_requirement_gold_annotation,
    compute_prompt_requirement_audit_report,
    seed_prompt_requirement_gold_file,
    select_prompt_requirement_audit_cases,
)


def load_agent_alpha_module():
    module_name = "test_generate_dpo_agent_alpha_pairs"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "generate_dpo_agent_alpha_pairs.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not load alpha pair generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DPOPreferenceAnnotationTest(unittest.TestCase):
    def test_review_units_merge_candidates_metrics_and_export_final_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, requirements_path, run_dir = self._write_fixture(root)
            output_dir = root / "annotations"
            paths = initialize_annotation_run(
                output_dir=output_dir,
                run_id="manual-test",
                candidate_run_dirs=[run_dir],
                dataset_path=dataset_path,
                prompt_requirements_path=requirements_path,
                split="train",
                batch_size=10,
            )

            units = load_review_units(
                candidate_run_dirs=[run_dir],
                dataset_path=dataset_path,
                prompt_requirements_path=requirements_path,
                split="train",
            )

            self.assertEqual(len(units), 1)
            unit = units[0]
            self.assertEqual(unit["unit_id"], "q1::question")
            self.assertEqual(len(unit["candidates"]), 3)
            self.assertEqual(unit["candidates"][0]["candidate_id"], "c00")
            self.assertEqual(unit["candidates"][0]["metrics"]["prompt_requirement_f1"], 1.0)

            chosen = unit["candidates"][0]["candidate_key"]
            rejected = unit["candidates"][1]["candidate_key"]
            second_rejected = unit["candidates"][2]["candidate_key"]
            append_human_preference(
                paths=paths,
                unit=unit,
                payload={
                    "decision": "preference",
                    "chosen_candidate_key": chosen,
                    "rejected_candidate_key": rejected,
                    "confidence": "high",
                    "rationale": "The chosen candidate preserves name and YAML parseability.",
                    "metric_flags": ["prompt_metric_ok"],
                },
            )
            append_human_preference(
                paths=paths,
                unit=unit,
                payload={
                    "decision": "preference",
                    "chosen_candidate_key": chosen,
                    "rejected_candidate_key": second_rejected,
                    "confidence": "medium",
                    "rationale": "The second pair is another informative rejected candidate.",
                    "metric_flags": ["hard_negative"],
                    "pair_type": "intermediate_hard_negative",
                    "score_margin": 0.5,
                },
            )
            final_rows = export_final_preferences(paths=paths, units_by_id={unit["unit_id"]: unit})
            state = write_annotation_state(paths=paths, units=units, final_rows=final_rows)

            self.assertEqual(len(final_rows), 2)
            self.assertEqual(final_rows[0]["chosen_candidate_key"], chosen)
            self.assertEqual(final_rows[0]["rejected_candidate_key"], rejected)
            self.assertEqual(final_rows[0]["chosen"], "<blocks>chosen</blocks>")
            self.assertEqual(final_rows[1]["pair_type"], "intermediate_hard_negative")
            self.assertEqual(state["approved_pair_count"], 2)
            self.assertEqual(len(read_jsonl(paths.final_preferences)), 2)

    def test_preference_schema_rejects_same_chosen_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, requirements_path, run_dir = self._write_fixture(root)
            paths = AnnotationPaths(root / "annotations")
            unit = load_review_units(
                candidate_run_dirs=[run_dir],
                dataset_path=dataset_path,
                prompt_requirements_path=requirements_path,
                split="train",
            )[0]
            candidate_key = unit["candidates"][0]["candidate_key"]

            with self.assertRaises(PreferenceAnnotationError):
                append_human_preference(
                    paths=paths,
                    unit=unit,
                    payload={
                        "decision": "preference",
                        "chosen_candidate_key": candidate_key,
                        "rejected_candidate_key": candidate_key,
                        "confidence": "medium",
                    },
                )

    def test_pending_agent_suggestion_does_not_export_until_approved_by_human(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, requirements_path, run_dir = self._write_fixture(root)
            paths = AnnotationPaths(root / "annotations")
            unit = load_review_units(
                candidate_run_dirs=[run_dir],
                dataset_path=dataset_path,
                prompt_requirements_path=requirements_path,
                split="train",
            )[0]
            chosen = unit["candidates"][0]["candidate_key"]
            rejected = unit["candidates"][1]["candidate_key"]

            append_agent_suggestion(
                paths=paths,
                unit=unit,
                payload={
                    "decision": "preference",
                    "chosen_candidate_key": chosen,
                    "rejected_candidate_key": rejected,
                    "confidence": "medium",
                    "rationale": "Looks better.",
                },
            )
            final_rows = export_final_preferences(paths=paths, units_by_id={unit["unit_id"]: unit})

            self.assertEqual(final_rows, [])

    def test_decision_packet_contains_same_context_needed_by_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, requirements_path, run_dir = self._write_fixture(root)
            unit = load_review_units(
                candidate_run_dirs=[run_dir],
                dataset_path=dataset_path,
                prompt_requirements_path=requirements_path,
                split="train",
            )[0]

            packet = build_decision_packet(unit)

            self.assertEqual(packet["unit_id"], "q1::question")
            self.assertEqual(len(packet["candidates"]), 3)
            self.assertIn("instructions", packet)
            self.assertEqual(packet["labeling_guide"]["version"], "dpo_kubernetes_labeling_guide_v1")
            self.assertIn("gate_pass", " ".join(packet["labeling_guide"]["decision_rules"]))
            self.assertIn("reconstructed_yaml", packet["candidates"][0])

    def test_labeling_guide_defines_shared_human_and_agent_rules(self) -> None:
        guide = get_labeling_guide()

        self.assertEqual(guide["version"], "dpo_kubernetes_labeling_guide_v1")
        self.assertIn("recommended_metric_flags", guide)
        self.assertIn("prompt_metric_unreliable", guide["recommended_metric_flags"])
        self.assertTrue(any("score_margin >= 0.25" in item for item in guide["pair_types"]))

    def test_alpha_pair_generator_prefers_gate_without_prompt_f1_regression(self) -> None:
        module = load_agent_alpha_module()
        unit = {
            "unit_id": "q1::question",
            "candidates": [
                self._candidate("gate", score=2.0, prompt_f1=0.8, gate=True),
                self._candidate("nogate", score=1.6, prompt_f1=0.82, gate=False),
                self._candidate("weak", score=0.9, prompt_f1=0.4, gate=False),
            ],
        }

        pairs = module.select_alpha_pairs(
            unit,
            max_pairs_per_prompt=4,
            min_strong_margin=0.25,
            min_intermediate_margin=0.15,
            min_low_margin=0.05,
            prompt_f1_tolerance=0.05,
        )

        pair_types = {pair["pair_type"] for pair in pairs}
        self.assertIn("gate_practice", pair_types)
        self.assertLessEqual(len(pairs), 4)
        self.assertTrue(all(pair["review_status"] == "pending" for pair in pairs))

    def test_alpha_pair_generator_rejects_missing_metric_candidates_as_chosen(self) -> None:
        module = load_agent_alpha_module()
        unit = {
            "unit_id": "q1::question",
            "candidates": [
                self._candidate("good", score=1.0, prompt_f1=1.0, gate=False),
                {
                    "candidate_key": "missing",
                    "preference_score": 5.0,
                    "hard_invalid": False,
                    "generation_ok": True,
                    "metrics": {},
                },
            ],
        }

        pairs = module.select_alpha_pairs(
            unit,
            max_pairs_per_prompt=4,
            min_strong_margin=0.25,
            min_intermediate_margin=0.15,
            min_low_margin=0.05,
            prompt_f1_tolerance=0.05,
        )

        self.assertTrue(all(pair["chosen_candidate_key"] == "good" for pair in pairs))

    def test_prompt_requirement_gold_set_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, requirements_path, _ = self._write_fixture(root)
            dataset_index = load_dataset_index(dataset_path)
            rows = read_jsonl(requirements_path)
            audit_cases = select_prompt_requirement_audit_cases(
                prompt_requirement_rows=rows,
                dataset_index=dataset_index,
                split="train",
                sample_size=1,
                seed=1,
            )
            seed_prompt_requirement_gold_file(output_dir=root / "audit", audit_cases=audit_cases)
            append_prompt_requirement_gold_annotation(
                output_dir=root / "audit",
                payload={
                    **audit_cases[0],
                    "gold_requirements": [
                        {"category": "kind", "value": "ConfigMap", "canonical": "kind=ConfigMap"},
                        {"category": "metadata.name", "value": "app", "canonical": "metadata.name=app"},
                    ],
                },
            )

            report = compute_prompt_requirement_audit_report(read_jsonl(root / "audit" / "prompt_requirement_gold.jsonl"))

            self.assertEqual(report["reviewed_case_count"], 1)
            self.assertEqual(report["overall"]["gold_count"], 2)
            self.assertEqual(report["overall"]["matched_count"], 2)
            self.assertEqual(report["overall"]["f1"], 1.0)

    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        dataset_path = root / "dataset_train_ready.jsonl"
        requirements_path = root / "prompt_requirements.jsonl"
        run_dir = root / "candidate_run"
        yaml_text = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: app\n"
        write_jsonl(
            dataset_path,
            [
                {
                    "sample_id": "q1",
                    "prompt_variant": "question",
                    "prompt_text": "Create a Kubernetes ConfigMap named app.",
                    "target_yaml_normalized": yaml_text,
                    "split": "train",
                }
            ],
        )
        write_jsonl(
            requirements_path,
            [
                {
                    "sample_id": "q1",
                    "prompt_variant": "question",
                    "split": "train",
                    "prompt_text": "Create a Kubernetes ConfigMap named app.",
                    "prompt_requirement_supported": True,
                    "prompt_requirements": [
                        {"category": "kind", "value": "ConfigMap", "key": None, "canonical": "kind=ConfigMap"},
                        {"category": "metadata.name", "value": "app", "key": None, "canonical": "metadata.name=app"},
                    ],
                    "reference_prompt_requirement_evaluation": {
                        "prompt_requirement_f1": 1.0,
                        "prompt_requirement_exact_match": True,
                    },
                }
            ],
        )
        run_dir.mkdir()
        write_json(run_dir / "config.json", {"run_id": "candidate-run-a"})
        base_candidate = {
            "unit_id": "q1::question",
            "sample_id": "q1",
            "prompt_variant": "question",
            "split": "train",
            "prompt": "Structural prompt",
            "reference_yaml": yaml_text,
            "generation_ok": True,
            "structured_output_parse_success": True,
            "predicted_blocks": [],
            "parser_errors": [],
            "reconstruction_errors": [],
            "generation_config": {"temperature": 0.5},
        }
        write_jsonl(
            run_dir / "candidates.jsonl",
            [
                {
                    **base_candidate,
                    "candidate_uid": "q1::question::c00",
                    "candidate_id": "c00",
                    "candidate_index": 0,
                    "model_output_text": "<blocks>chosen</blocks>",
                    "reconstructed_yaml": yaml_text,
                },
                {
                    **base_candidate,
                    "candidate_uid": "q1::question::c01",
                    "candidate_id": "c01",
                    "candidate_index": 1,
                    "model_output_text": "<blocks>rejected</blocks>",
                    "reconstructed_yaml": "apiVersion: v1\nkind: ConfigMap\n",
                },
                {
                    **base_candidate,
                    "candidate_uid": "q1::question::c02",
                    "candidate_id": "c02",
                    "candidate_index": 2,
                    "model_output_text": "<blocks>weak</blocks>",
                    "reconstructed_yaml": "kind: ConfigMap\n",
                },
            ],
        )
        write_jsonl(
            run_dir / "candidate_metrics.jsonl",
            [
                {
                    "candidate_uid": "q1::question::c00",
                    "unit_id": "q1::question",
                    "sample_id": "q1",
                    "prompt_variant": "question",
                    "split": "train",
                    "candidate_id": "c00",
                    "preference_score": 2.0,
                    "hard_invalid": False,
                    "evaluation": {
                        "yaml_parse_ok": True,
                        "block_parse_ok": True,
                        "prompt_requirement_f1": 1.0,
                    },
                },
                {
                    "candidate_uid": "q1::question::c01",
                    "unit_id": "q1::question",
                    "sample_id": "q1",
                    "prompt_variant": "question",
                    "split": "train",
                    "candidate_id": "c01",
                    "preference_score": 1.0,
                    "hard_invalid": False,
                    "evaluation": {
                        "yaml_parse_ok": True,
                        "block_parse_ok": True,
                        "prompt_requirement_f1": 0.5,
                    },
                },
                {
                    "candidate_uid": "q1::question::c02",
                    "unit_id": "q1::question",
                    "sample_id": "q1",
                    "prompt_variant": "question",
                    "split": "train",
                    "candidate_id": "c02",
                    "preference_score": 0.4,
                    "hard_invalid": False,
                    "evaluation": {
                        "yaml_parse_ok": True,
                        "block_parse_ok": True,
                        "prompt_requirement_f1": 0.25,
                    },
                },
            ],
        )
        return dataset_path, requirements_path, run_dir

    def _candidate(self, candidate_key: str, *, score: float, prompt_f1: float, gate: bool) -> dict[str, object]:
        return {
            "candidate_key": candidate_key,
            "preference_score": score,
            "hard_invalid": False,
            "generation_ok": True,
            "metrics": {
                "yaml_parse_ok": True,
                "block_parse_ok": True,
                "prompt_requirement_f1": prompt_f1,
                "kubernetes_domain_gate_pass": gate,
                "kubernetes_domain_validity_score": 1.0 if gate else 0.6,
                "required_field_complete_resource_rate": 1.0,
            },
        }


if __name__ == "__main__":
    unittest.main()
