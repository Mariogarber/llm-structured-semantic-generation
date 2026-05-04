from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.evaluation import evaluate_blocks_prediction, summarize_evaluations
from llm_structured_semantic_generation.latent import (
    mean_pool_generate_hidden_states,
    mean_pool_generated_hidden_states,
)
from llm_structured_semantic_generation.sft_serialization import (
    deserialize_training_blocks,
    serialize_blocks_for_training,
)
from llm_structured_semantic_generation.structure import (
    YAMLBlock,
    blocks_to_yaml,
    validate_round_trip,
    yaml_to_blocks,
)


class StructuralBlocksTest(unittest.TestCase):
    def test_simple_yaml_round_trip_preserves_parsed_documents(self) -> None:
        yaml_text = (
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "metadata:\n"
            "  name: game-demo\n"
        )

        result = validate_round_trip(yaml_text)

        self.assertTrue(result.yaml_parse_ok)
        self.assertTrue(result.semantics_preserved)
        self.assertEqual(
            [block.to_dict() for block in result.blocks],
            [
                {"line_index": 0, "line_text": "apiVersion: v1", "level": 0, "document_index": 0},
                {"line_index": 1, "line_text": "kind: ConfigMap", "level": 0, "document_index": 0},
                {"line_index": 2, "line_text": "metadata:", "level": 0, "document_index": 0},
                {"line_index": 3, "line_text": "name: game-demo", "level": 1, "document_index": 0},
            ],
        )

    def test_multidocument_yaml_uses_document_index(self) -> None:
        yaml_text = (
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "---\n"
            "apiVersion: v1\n"
            "kind: Service\n"
        )

        blocks = yaml_to_blocks(yaml_text)
        reconstruction = blocks_to_yaml(blocks)

        self.assertEqual([block.document_index for block in blocks], [0, 0, 1, 1])
        self.assertIn("---\n", reconstruction.yaml_text)
        self.assertEqual(tuple(yaml.safe_load_all(reconstruction.yaml_text)), tuple(yaml.safe_load_all(yaml_text)))

    def test_blank_lines_inside_scalar_are_preserved_by_semantics(self) -> None:
        yaml_text = (
            "apiVersion: v1\n"
            "data:\n"
            "  game.properties: 'enemy.types=aliens,monsters\n"
            "\n"
            "    player.maximum-lives=5'\n"
            "kind: ConfigMap\n"
            "metadata:\n"
            "  name: game-demo\n"
        )

        result = validate_round_trip(yaml_text)

        self.assertTrue(result.yaml_parse_ok)
        self.assertTrue(result.semantics_preserved)
        self.assertIn(YAMLBlock(line_index=3, line_text="", level=0, document_index=0), result.blocks)

    def test_invalid_line_index_sequence_is_rejected(self) -> None:
        blocks = [
            {"document_index": 0, "line_index": 0, "level": 0, "line_text": "metadata:"},
            {"document_index": 0, "line_index": 2, "level": 1, "line_text": "name: app"},
        ]

        result = blocks_to_yaml(blocks)

        self.assertFalse(result.yaml_parse_ok)
        self.assertIn("unexpected_line_index:2:expected:1", ";".join(result.errors))

    def test_evaluation_reports_wrong_level_without_hidden_repair(self) -> None:
        reference_yaml = "metadata:\n  name: app\n"
        predicted_blocks = [
            {"document_index": 0, "line_index": 0, "level": 0, "line_text": "metadata:"},
            {"document_index": 0, "line_index": 1, "level": 0, "line_text": "name: app"},
        ]

        evaluation = evaluate_blocks_prediction(reference_yaml, predicted_blocks)

        self.assertTrue(evaluation.yaml_parse_ok)
        self.assertFalse(evaluation.parsed_equal_to_reference)
        self.assertEqual(evaluation.level_exact_match_rate, 0.5)
        self.assertEqual(evaluation.line_text_f1, 1.0)
        self.assertEqual(evaluation.level_mae, 0.5)
        self.assertEqual(evaluation.indentation_leak_rate, 0.0)

    def test_raw_line_text_recovery_accepts_yaml_style_blocks(self) -> None:
        reference_yaml = (
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "metadata:\n"
            "  name: game-demo\n"
            "data:\n"
            "  key: value\n"
        )
        predicted_blocks = [
            {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
            {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            {"document_index": 0, "line_index": 2, "level": 1, "line_text": "metadata:"},
            {"document_index": 0, "line_index": 3, "level": 2, "line_text": "  name: game-demo"},
            {"document_index": 0, "line_index": 4, "level": 1, "line_text": "data:"},
            {"document_index": 0, "line_index": 5, "level": 2, "line_text": "  key: value"},
        ]

        strict_result = blocks_to_yaml(predicted_blocks)
        recovered_result = blocks_to_yaml(predicted_blocks, recovery_mode="raw_line_text")
        recovered_evaluation = evaluate_blocks_prediction(
            reference_yaml,
            predicted_blocks,
            recovery_mode="raw_line_text",
        )

        self.assertFalse(strict_result.yaml_parse_ok)
        self.assertTrue(recovered_result.yaml_parse_ok)
        self.assertEqual(tuple(yaml.safe_load_all(recovered_result.yaml_text)), tuple(yaml.safe_load_all(reference_yaml)))
        self.assertTrue(recovered_evaluation.yaml_parse_ok)
        self.assertTrue(recovered_evaluation.parsed_equal_to_reference)
        self.assertEqual(recovered_evaluation.primary_kind_match, True)
        self.assertEqual(recovered_evaluation.primary_api_version_match, True)
        self.assertEqual(recovered_evaluation.primary_metadata_name_match, True)
        self.assertEqual(recovered_evaluation.semantic_key_f1, 1.0)

    def test_sft_serialization_round_trip(self) -> None:
        blocks = [
            YAMLBlock(document_index=0, line_index=0, level=0, line_text="apiVersion: v1"),
            YAMLBlock(document_index=0, line_index=1, level=0, line_text="kind: Pod"),
        ]

        serialized = serialize_blocks_for_training(blocks)
        restored = deserialize_training_blocks(serialized)

        self.assertEqual(restored, blocks)

    def test_detailed_metrics_capture_format_and_semantics(self) -> None:
        reference_yaml = (
            "apiVersion: v1\n"
            "kind: Pod\n"
            "metadata:\n"
            "  name: app\n"
            "spec:\n"
            "  containers:\n"
            "  - name: main\n"
            "    image: nginx:latest\n"
            "    volumeMounts:\n"
            "    - name: cache\n"
            "      mountPath: /cache\n"
            "  volumes:\n"
            "  - name: cache\n"
            "    emptyDir: {}\n"
        )
        predicted_blocks = [
            {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
            {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: Pod"},
            {"document_index": 0, "line_index": 2, "level": 0, "line_text": "metadata:"},
            {"document_index": 0, "line_index": 3, "level": 1, "line_text": "name: app"},
            {"document_index": 0, "line_index": 4, "level": 0, "line_text": "spec:"},
            {"document_index": 0, "line_index": 5, "level": 1, "line_text": "containers:"},
            {"document_index": 0, "line_index": 6, "level": 1, "line_text": "- name: main"},
            {"document_index": 0, "line_index": 7, "level": 2, "line_text": "image: nginx:latest"},
            {"document_index": 0, "line_index": 8, "level": 2, "line_text": "volumeMounts:"},
            {"document_index": 0, "line_index": 9, "level": 2, "line_text": "- name: cache"},
            {"document_index": 0, "line_index": 10, "level": 3, "line_text": "mountPath: /cache"},
            {"document_index": 0, "line_index": 11, "level": 1, "line_text": "volumes:"},
            {"document_index": 0, "line_index": 12, "level": 1, "line_text": "- name: cache"},
            {"document_index": 0, "line_index": 13, "level": 2, "line_text": "emptyDir: {}"},
        ]

        evaluation = evaluate_blocks_prediction(reference_yaml, predicted_blocks)
        summary = summarize_evaluations([evaluation])

        self.assertTrue(evaluation.yaml_parse_ok)
        self.assertEqual(evaluation.valid_block_ratio, 1.0)
        self.assertEqual(evaluation.line_text_f1, 1.0)
        self.assertEqual(evaluation.primary_kind_match, True)
        self.assertEqual(evaluation.primary_api_version_match, True)
        self.assertEqual(evaluation.primary_metadata_name_match, True)
        self.assertEqual(evaluation.semantic_key_f1, 1.0)
        self.assertEqual(evaluation.volume_mount_consistency, 1.0)
        self.assertEqual(summary["average_line_text_f1"], 1.0)
        self.assertEqual(summary["average_volume_mount_consistency"], 1.0)

    def test_mean_pool_generated_hidden_states_uses_generated_suffix_only(self) -> None:
        hidden = torch.tensor(
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [5.0, 7.0],
                [9.0, 11.0],
            ]
        )

        pooled = mean_pool_generated_hidden_states(hidden, prompt_token_count=2)

        self.assertTrue(torch.equal(pooled, torch.tensor([7.0, 9.0])))

    def test_mean_pool_generated_hidden_states_returns_none_when_no_generated_tokens(self) -> None:
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        pooled = mean_pool_generated_hidden_states(hidden, prompt_token_count=2)

        self.assertIsNone(pooled)

    def test_mean_pool_generate_hidden_states_averages_generated_steps(self) -> None:
        generate_hidden_states = (
            (
                torch.zeros(1, 3, 2),
                torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 4.0]]]),
            ),
            (
                torch.zeros(1, 1, 2),
                torch.tensor([[[6.0, 8.0]]]),
            ),
        )

        pooled = mean_pool_generate_hidden_states(generate_hidden_states)

        self.assertTrue(torch.equal(pooled, torch.tensor([4.0, 6.0])))


if __name__ == "__main__":
    unittest.main()
