from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_jsonl
from llm_structured_semantic_generation.resumable_run import (
    ArtifactConsistencyError,
    ResumableRun,
    RunCompatibilityError,
)
from llm_structured_semantic_generation.sft_serialization import serialize_blocks_for_training
from llm_structured_semantic_generation.structure import yaml_to_blocks


def load_baseline_module():
    module_name = "test_run_kubernetes_baseline"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "scripts" / "run_kubernetes_baseline.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not load baseline script module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResumableRunStateTest(unittest.TestCase):
    def test_initialization_creates_state_and_resume_recovers_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "baseline" / "run-001"
            config = {"run_id": "run-001", "resume_signature": {"dataset": "dataset-a"}}

            run = ResumableRun.initialize(
                run_dir=run_dir,
                config=config,
                total_units=2,
                unit_id_field="unit_id",
                primary_artifact_name="predictions",
                artifact_paths={"predictions": "predictions.jsonl", "latent": "latent.jsonl"},
            )
            self.assertEqual(run.state["processed_units"], 0)
            self.assertEqual(run.state["status"], "running")

            run.record_batch([{"unit_id": "sample-1", "value": 1}])

            resumed = ResumableRun.initialize(
                run_dir=run_dir,
                config=config,
                total_units=2,
                unit_id_field="unit_id",
                primary_artifact_name="predictions",
                artifact_paths={"predictions": "predictions.jsonl", "latent": "latent.jsonl"},
            )
            self.assertEqual(resumed.completed_unit_ids, ["sample-1"])
            self.assertEqual(resumed.state["processed_units"], 1)
            self.assertEqual(resumed.state["remaining_units"], 1)

    def test_initialization_reconciles_state_from_primary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "baseline" / "run-002"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = {"run_id": "run-002", "resume_signature": {"dataset": "dataset-a"}}
            (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            write_jsonl(run_dir / "predictions.jsonl", [{"unit_id": "sample-1", "value": 1}])
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-002",
                        "status": "running",
                        "processed_units": 0,
                        "remaining_units": 2,
                        "completed_unit_ids": [],
                    }
                ),
                encoding="utf-8",
            )

            resumed = ResumableRun.initialize(
                run_dir=run_dir,
                config=config,
                total_units=2,
                unit_id_field="unit_id",
                primary_artifact_name="predictions",
                artifact_paths={"predictions": "predictions.jsonl", "latent": "latent.jsonl"},
            )

            self.assertEqual(resumed.state["processed_units"], 1)
            self.assertEqual(resumed.state["completed_unit_ids"], ["sample-1"])

    def test_initialize_fails_when_resume_signature_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "baseline" / "run-003"
            config_a = {"run_id": "run-003", "resume_signature": {"dataset": "dataset-a"}}
            config_b = {"run_id": "run-003", "resume_signature": {"dataset": "dataset-b"}}

            ResumableRun.initialize(
                run_dir=run_dir,
                config=config_a,
                total_units=1,
                unit_id_field="unit_id",
                primary_artifact_name="predictions",
                artifact_paths={"predictions": "predictions.jsonl"},
            )

            with self.assertRaises(RunCompatibilityError):
                ResumableRun.initialize(
                    run_dir=run_dir,
                    config=config_b,
                    total_units=1,
                    unit_id_field="unit_id",
                    primary_artifact_name="predictions",
                    artifact_paths={"predictions": "predictions.jsonl"},
                )

    def test_record_batch_rejects_duplicate_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "baseline" / "run-004"
            config = {"run_id": "run-004", "resume_signature": {"dataset": "dataset-a"}}
            run = ResumableRun.initialize(
                run_dir=run_dir,
                config=config,
                total_units=2,
                unit_id_field="unit_id",
                primary_artifact_name="predictions",
                artifact_paths={"predictions": "predictions.jsonl"},
            )

            run.record_batch([{"unit_id": "sample-1", "value": 1}])

            with self.assertRaises(ArtifactConsistencyError):
                run.record_batch([{"unit_id": "sample-1", "value": 2}])


class BaselineResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = load_baseline_module()

    def build_dataset_rows(self) -> list[dict[str, object]]:
        rows = []
        for index in range(3):
            yaml_text = (
                "apiVersion: v1\n"
                "kind: ConfigMap\n"
                "metadata:\n"
                f"  name: app-{index}\n"
            )
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "prompt_variant": "canonical",
                    "split": "validation",
                    "structural_target_status": "ok",
                    "prompt_text": f"create configmap {index}",
                    "target_yaml_normalized": yaml_text,
                }
            )
        return rows

    def build_args(
        self,
        *,
        dataset_path: Path,
        output_dir: Path,
        run_id: str,
        collect_latent_means: bool,
        output_format: str = "blocks_tsv_compact_v1",
        dry_run: bool = False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            dataset=dataset_path,
            model_path=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit",
            split="validation",
            max_samples=None,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            output_format=output_format,
            recovery_mode="strict",
            gpu_memory="4GiB",
            cpu_memory="16GiB",
            output_dir=output_dir,
            run_id=run_id,
            batch_size=1,
            collect_latent_means=collect_latent_means,
            dry_run=dry_run,
        )

    def fake_generate_completion_factory(
        self,
        dataset_rows: list[dict[str, object]],
        output_format: str,
        *,
        fail_on_call: int | None = None,
    ):
        completions = {}
        for index, row in enumerate(dataset_rows):
            blocks = [block.to_dict() for block in yaml_to_blocks(str(row["target_yaml_normalized"]))]
            if output_format == "blocks_tsv_compact_v1":
                raw_text = "\n".join(
                    [
                        "<blocks>",
                        *[
                            f"{block['document_index']}\t{block['level']}\t{block['line_text']}"
                            for block in blocks
                        ],
                        "</blocks>",
                    ]
                )
            elif output_format == "blocks_tsv_v1":
                raw_text = serialize_blocks_for_training(blocks)
            else:
                raw_text = json.dumps(blocks, ensure_ascii=False)
            completions[str(row["prompt_text"])] = {
                "raw_text": raw_text,
                "prompt_token_count": 10,
                "generated_token_ids": [1, 2, 3],
                "generated_token_count": 3,
                "latent_dim": 2,
                "latent_mean": [float(index), float(index + 1)],
            }

        state = {"call_count": 0}

        def fake_generate_completion(tokenizer, model, prompt: str, args):
            state["call_count"] += 1
            if fail_on_call is not None and state["call_count"] == fail_on_call:
                raise KeyboardInterrupt("simulated interruption")
            return completions[prompt]

        return fake_generate_completion

    def run_baseline_with_patches(
        self,
        *,
        args: argparse.Namespace,
        dataset_rows: list[dict[str, object]],
        fail_on_call: int | None = None,
    ) -> None:
        fake_model_checks = {
            "model_path_exists": True,
            "has_config": True,
            "has_generation_config": True,
            "has_weights": True,
            "has_tokenizer_files": True,
            "installed_transformers": True,
            "installed_torch": True,
            "installed_bitsandbytes": True,
            "quant_method": "bitsandbytes",
            "warnings": [],
            "ready_for_full_run": True,
        }

        with mock.patch.object(self.baseline, "parse_args", return_value=args), mock.patch.object(
            self.baseline, "inspect_model_path", return_value=fake_model_checks
        ), mock.patch.object(
            self.baseline, "load_model", return_value=("tokenizer", object())
        ), mock.patch.object(
            self.baseline,
            "render_chat_prompt",
            side_effect=lambda tokenizer, prompt_text, output_format: prompt_text,
        ), mock.patch.object(
            self.baseline,
            "generate_completion",
            side_effect=self.fake_generate_completion_factory(
                dataset_rows,
                args.output_format,
                fail_on_call=fail_on_call,
            ),
        ):
            self.baseline.main()

    def test_extract_blocks_tsv_round_trip(self) -> None:
        blocks = [
            {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
            {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
        ]
        serialized = serialize_blocks_for_training(blocks)

        extracted = self.baseline.extract_blocks_tsv(serialized)

        self.assertEqual(extracted, blocks)

    def test_extract_blocks_tsv_accepts_labeled_fields(self) -> None:
        serialized = (
            "<blocks>\n"
            "document_index\t0\tline_index\t0\tlevel\t0\tline_text\tapiVersion: v1\n"
            "document_index\t0\tline_index\t1\tlevel\t0\tline_text\tkind: ConfigMap\n"
            "</blocks>"
        )

        extracted = self.baseline.extract_blocks_tsv(serialized)

        self.assertEqual(
            extracted,
            [
                {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
                {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            ],
        )

    def test_extract_blocks_tsv_compact_round_trip(self) -> None:
        serialized = (
            "<blocks>\n"
            "0\t0\tapiVersion: v1\n"
            "0\t0\tkind: ConfigMap\n"
            "</blocks>"
        )

        extracted = self.baseline.extract_blocks_tsv_compact(serialized)

        self.assertEqual(
            extracted,
            [
                {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
                {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            ],
        )

    def test_build_prompt_for_compact_tsv_mentions_mapping_and_list_rules(self) -> None:
        _, user_prompt = self.baseline.build_prompt("create a deployment", "blocks_tsv_compact_v1")

        self.assertIn("do not prefix mapping children with '-'", user_prompt)
        self.assertIn("use '-' only for real YAML list items", user_prompt)
        self.assertIn("top-level YAML keys such as apiVersion, kind, metadata, and spec must stay at level 0", user_prompt)
        self.assertIn("metadata:", user_prompt)
        self.assertIn("containers:", user_prompt)

    def test_extract_blocks_tsv_accepts_literal_tab_markers_and_truncated_tail(self) -> None:
        serialized = (
            "<blocks>\n"
            "0<TAB>0<TAB>0<TAB>apiVersion: v1\n"
            "0<TAB>1<TAB>0<TAB>kind: ConfigMap\n"
            "0<TAB>2<TAB>"
        )

        extracted = self.baseline.extract_blocks_tsv(serialized)

        self.assertEqual(
            extracted,
            [
                {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
                {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            ],
        )

    def test_extract_blocks_tsv_accepts_lowercase_tab_markers(self) -> None:
        serialized = (
            "<blocks>\n"
            "0<tab>0<tab>0<tab>apiVersion: v1\n"
            "0<tab>1<tab>0<tab>kind: ConfigMap\n"
            "</blocks>"
        )

        extracted = self.baseline.extract_blocks_tsv(serialized)

        self.assertEqual(
            extracted,
            [
                {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
                {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            ],
        )

    def test_extract_blocks_tsv_accepts_vertical_tab_separators(self) -> None:
        serialized = (
            "<blocks>\n"
            "0\v0\v0\vapiVersion: v1\n"
            "0\v1\v0\vkind: ConfigMap\n"
            "</blocks>"
        )

        extracted = self.baseline.extract_blocks_tsv(serialized)

        self.assertEqual(
            extracted,
            [
                {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
                {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            ],
        )

    def test_extract_blocks_tsv_compact_accepts_lowercase_tab_markers_and_truncated_tail(self) -> None:
        serialized = (
            "<blocks>\n"
            "0<tab>0<tab>apiVersion: v1\n"
            "0<tab>0<tab>kind: ConfigMap\n"
            "0<tab>"
        )

        extracted = self.baseline.extract_blocks_tsv_compact(serialized)

        self.assertEqual(
            extracted,
            [
                {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
                {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            ],
        )

    def test_extract_blocks_tsv_compact_accepts_vertical_tab_separators(self) -> None:
        serialized = (
            "<blocks>\n"
            "0\v0\vapiVersion: v1\n"
            "0\v0\vkind: ConfigMap\n"
            "</blocks>"
        )

        extracted = self.baseline.extract_blocks_tsv_compact(serialized)

        self.assertEqual(
            extracted,
            [
                {"document_index": 0, "line_index": 0, "level": 0, "line_text": "apiVersion: v1"},
                {"document_index": 0, "line_index": 1, "level": 0, "line_text": "kind: ConfigMap"},
            ],
        )

    def test_baseline_resume_matches_continuous_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_rows = self.build_dataset_rows()
            dataset_path = tmp_path / "dataset.jsonl"
            write_jsonl(dataset_path, dataset_rows)

            continuous_args = self.build_args(
                dataset_path=dataset_path,
                output_dir=tmp_path / "continuous",
                run_id="continuous-run",
                collect_latent_means=True,
            )
            resumed_args = self.build_args(
                dataset_path=dataset_path,
                output_dir=tmp_path / "resumed",
                run_id="resumed-run",
                collect_latent_means=True,
            )

            self.run_baseline_with_patches(args=continuous_args, dataset_rows=dataset_rows)

            with self.assertRaises(KeyboardInterrupt):
                self.run_baseline_with_patches(args=resumed_args, dataset_rows=dataset_rows, fail_on_call=2)

            partial_predictions = read_jsonl(
                tmp_path / "resumed" / "resumed-run" / "predictions.jsonl",
                allow_truncated_last_line=True,
            )
            self.assertEqual(len(partial_predictions), 1)

            self.run_baseline_with_patches(args=resumed_args, dataset_rows=dataset_rows)

            continuous_predictions = read_jsonl(
                tmp_path / "continuous" / "continuous-run" / "predictions.jsonl"
            )
            resumed_predictions = read_jsonl(tmp_path / "resumed" / "resumed-run" / "predictions.jsonl")
            self.assertEqual(resumed_predictions, continuous_predictions)

            continuous_metrics = json.loads(
                (tmp_path / "continuous" / "continuous-run" / "metrics.json").read_text(encoding="utf-8")
            )
            resumed_metrics = json.loads(
                (tmp_path / "resumed" / "resumed-run" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {key: value for key, value in resumed_metrics.items() if key != "run_id"},
                {key: value for key, value in continuous_metrics.items() if key != "run_id"},
            )

            resumed_state = json.loads(
                (tmp_path / "resumed" / "resumed-run" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_state["status"], "completed")
            self.assertEqual(resumed_state["processed_units"], 3)

            latent_rows = read_jsonl(tmp_path / "resumed" / "resumed-run" / "latent_mean_vectors.jsonl")
            self.assertEqual(len(latent_rows), 3)

    def test_dry_run_writes_only_inspection_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.jsonl"
            write_jsonl(dataset_path, self.build_dataset_rows())
            args = self.build_args(
                dataset_path=dataset_path,
                output_dir=tmp_path / "dry-run",
                run_id="dry-run-001",
                collect_latent_means=False,
                dry_run=True,
            )

            fake_model_checks = {
                "model_path_exists": True,
                "has_config": True,
                "has_generation_config": True,
                "has_weights": True,
                "has_tokenizer_files": True,
                "installed_transformers": True,
                "installed_torch": True,
                "installed_bitsandbytes": True,
                "quant_method": "bitsandbytes",
                "warnings": [],
                "ready_for_full_run": True,
            }

            with mock.patch.object(self.baseline, "parse_args", return_value=args), mock.patch.object(
                self.baseline, "inspect_model_path", return_value=fake_model_checks
            ):
                self.baseline.main()

            run_dir = tmp_path / "dry-run" / "dry-run-001"
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertFalse((run_dir / "state.json").exists())
            self.assertFalse((run_dir / "predictions.jsonl").exists())
            self.assertFalse((run_dir / "latent_mean_vectors.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
