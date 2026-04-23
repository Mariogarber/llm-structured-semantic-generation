from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.evaluation import evaluate_blocks_prediction
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

    def test_sft_serialization_round_trip(self) -> None:
        blocks = [
            YAMLBlock(document_index=0, line_index=0, level=0, line_text="apiVersion: v1"),
            YAMLBlock(document_index=0, line_index=1, level=0, line_text="kind: Pod"),
        ]

        serialized = serialize_blocks_for_training(blocks)
        restored = deserialize_training_blocks(serialized)

        self.assertEqual(restored, blocks)


if __name__ == "__main__":
    unittest.main()

