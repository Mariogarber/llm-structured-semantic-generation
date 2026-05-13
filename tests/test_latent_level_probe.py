from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_jsonl
from llm_structured_semantic_generation.latent_level_probe import (
    build_content_only_example,
    chunk_path_for_unit,
    completed_unit_ids_from_chunks,
    evaluate_predictions,
    features_for_hidden_states,
    read_sft_rows,
    synthetic_hidden_for_example,
)
from llm_structured_semantic_generation.sft_serialization import serialize_blocks_for_training
from llm_structured_semantic_generation.structure import YAMLBlock


def load_probe_script():
    module_name = "test_run_kubernetes_latent_level_probe"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "run_kubernetes_latent_level_probe.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not load latent probe script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_sft_row(sample_id: str, split: str) -> dict[str, object]:
    blocks = [
        YAMLBlock(document_index=0, line_index=0, level=0, line_text="apiVersion: v1"),
        YAMLBlock(document_index=0, line_index=1, level=0, line_text="kind: ConfigMap"),
        YAMLBlock(document_index=0, line_index=2, level=0, line_text="metadata:"),
        YAMLBlock(document_index=0, line_index=3, level=1, line_text=f"name: {sample_id}"),
        YAMLBlock(document_index=1, line_index=0, level=0, line_text=""),
        YAMLBlock(document_index=1, line_index=1, level=0, line_text="apiVersion: v1"),
        YAMLBlock(document_index=1, line_index=2, level=0, line_text="kind: Service"),
    ]
    prompt = (
        "You generate Kubernetes manifests through an explicit structural representation. "
        "Return only line blocks; each block must include document_index, line_index, level, and line_text.\n\n"
        "Natural-language request:\n"
        f"Create resources for {sample_id}.\n\n"
        "Return the structural block sequence now."
    )
    return {
        "sample_id": sample_id,
        "prompt_variant": "question",
        "split": split,
        "prompt": prompt,
        "target": serialize_blocks_for_training(blocks),
        "target_yaml_normalized": "",
        "round_trip_yaml": "",
    }


class LatentLevelProbeTest(unittest.TestCase):
    def test_content_only_example_removes_gold_level_column(self) -> None:
        row = build_sft_row("q1", "train")
        example = build_content_only_example(row)

        self.assertIn("<content_blocks>", example.content_text)
        self.assertNotIn("level", example.content_text.lower())
        for line in example.content_text.splitlines():
            if line in {"<content_blocks>", "</content_blocks>"} or not line:
                continue
            self.assertEqual(len(line.split("\t")), 3)

    def test_line_alignment_extracts_all_feature_strategies_for_blank_and_multidoc_lines(self) -> None:
        row = build_sft_row("q2", "validation")
        example = build_content_only_example(row)
        hidden, offsets = synthetic_hidden_for_example(example, hidden_dim=6)
        rows_by_strategy = features_for_hidden_states(
            example=example,
            hidden=hidden,
            offsets=offsets,
            feature_strategies=[
                "record_prefix_state",
                "line_mean",
                "line_first_token",
                "line_last_token",
                "line_prefix_state",
            ],
        )

        self.assertEqual(
            set(rows_by_strategy),
            {"record_prefix_state", "line_mean", "line_first_token", "line_last_token", "line_prefix_state"},
        )
        for rows in rows_by_strategy.values():
            self.assertEqual(len(rows), len(example.line_spans))
            self.assertTrue(all(row["feature_dim"] == 6 for row in rows))
            self.assertTrue(any(row["line_text"] == "" for row in rows))

    def test_resume_progress_ignores_tmp_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            valid = chunk_path_for_unit(run_dir, "q1::question")
            valid.parent.mkdir(parents=True, exist_ok=True)
            valid.write_text(json.dumps({"unit_id": "q1::question"}), encoding="utf-8")
            (valid.parent / "q2-question.json.tmp").write_text(json.dumps({"unit_id": "q2::question"}), encoding="utf-8")

            self.assertEqual(completed_unit_ids_from_chunks(run_dir), {"q1::question"})

    def test_metric_computation_handles_rare_levels(self) -> None:
        y_true = np.asarray([0, 0, 1, 4])
        y_pred = np.asarray([0, 1, 1, 1])

        metrics = evaluate_predictions(y_true, y_pred)

        self.assertIn("macro_f1", metrics)
        self.assertEqual(metrics["labels"], [0, 1, 4])
        self.assertGreaterEqual(metrics["level_mae"], 0.0)

    def test_read_sft_rows_max_samples_is_applied_per_split_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            validation = tmp_path / "validation.jsonl"
            write_jsonl(train, [build_sft_row("train-a", "train"), build_sft_row("train-b", "train")])
            write_jsonl(validation, [build_sft_row("val-a", "validation"), build_sft_row("val-b", "validation")])

            rows = read_sft_rows([train, validation], max_samples=1)

            self.assertEqual([row["sample_id"] for row in rows], ["train-a", "val-a"])

    def test_cli_dry_run_all_is_resumable_and_writes_final_metrics_only_on_success(self) -> None:
        script = load_probe_script()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            validation = tmp_path / "validation.jsonl"
            test = tmp_path / "test.jsonl"
            write_jsonl(train, [build_sft_row("train-a", "train"), build_sft_row("train-b", "train")])
            write_jsonl(validation, [build_sft_row("val-a", "validation"), build_sft_row("val-b", "validation")])
            write_jsonl(test, [build_sft_row("test-a", "test")])

            argv = [
                "run_kubernetes_latent_level_probe.py",
                "--stage",
                "all",
                "--train-file",
                str(train),
                "--validation-file",
                str(validation),
                "--test-file",
                str(test),
                "--output-dir",
                str(tmp_path / "results"),
                "--run-id",
                "probe-smoke",
                "--batch-size",
                "1",
                "--max-samples",
                "2",
                "--feature-strategies",
                "record_prefix_state,line_mean,line_prefix_state",
                "--probe-types",
                "majority,previous_level,linear",
                "--probe-max-iter",
                "20",
                "--wandb-mode",
                "disabled",
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv):
                script.main()
            with mock.patch.object(sys, "argv", argv + ["--resume"]):
                script.main()

            run_dir = tmp_path / "results" / "probe-smoke"
            self.assertTrue((run_dir / "metrics.json").exists())
            metadata = read_jsonl(run_dir / "line_metadata.jsonl")
            line_ids = [row["line_id"] for row in metadata]
            self.assertEqual(len(line_ids), len(set(line_ids)))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["completed_row_count"], 4)
            self.assertIn("line_mean__majority", metrics["probe_ids"])
            self.assertIn("record_prefix_state__majority", metrics["probe_ids"])


if __name__ == "__main__":
    unittest.main()
