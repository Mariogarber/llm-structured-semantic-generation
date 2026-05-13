from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import write_jsonl
from llm_structured_semantic_generation.evaluation import evaluate_blocks_prediction
from llm_structured_semantic_generation.resumable_run import RunCompatibilityError
from llm_structured_semantic_generation.sft_serialization import (
    deserialize_training_blocks,
    serialize_blocks_for_training,
)
from llm_structured_semantic_generation.structure import yaml_to_blocks


def load_sft_module():
    module_name = "test_train_kubernetes_sft"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "train_kubernetes_sft.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not load SFT trainer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, **kwargs):
        add_special_tokens = kwargs.get("add_special_tokens", False)
        ids = [ord(character) % 251 + 1 for character in text]
        if add_special_tokens:
            ids = [252] + ids
        return {"input_ids": ids}


def build_sft_row(sample_id: str = "sample-1") -> dict[str, object]:
    yaml_text = (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: app\n"
    )
    blocks = [block.to_dict() for block in yaml_to_blocks(yaml_text)]
    return {
        "sample_id": sample_id,
        "prompt_variant": "question",
        "split": "validation",
        "prompt": "Natural-language request:\nCreate a ConfigMap named app.\n\nReturn the structural block sequence now.",
        "target": serialize_blocks_for_training(blocks),
        "target_yaml_normalized": yaml_text,
        "round_trip_yaml": yaml_text,
    }


class KubernetesSFTTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sft = load_sft_module()

    def test_build_training_text_preserves_blocks_tsv_v1_target(self) -> None:
        row = build_sft_row()

        prompt, target, training_text = self.sft.build_training_text(row)

        self.assertIn("Create a ConfigMap named app", prompt)
        self.assertEqual(deserialize_training_blocks(target), deserialize_training_blocks(str(row["target"])))
        self.assertIn("<blocks>", training_text)
        self.assertIn("</blocks>", training_text)

    def test_tokenize_masks_prompt_and_supervises_target_only(self) -> None:
        row = build_sft_row()
        tokenizer = FakeTokenizer()

        tokenized = self.sft.tokenize_sft_row(row, tokenizer, max_seq_length=4096)

        prompt_ids = tokenizer(
            f"{row['prompt'].rstrip()}{self.sft.PROMPT_TARGET_SEPARATOR}",
            add_special_tokens=True,
        )["input_ids"]
        self.assertTrue(all(label == -100 for label in tokenized["labels"][: len(prompt_ids)]))
        self.assertTrue(any(label != -100 for label in tokenized["labels"][len(prompt_ids) :]))
        self.assertEqual(len(tokenized["input_ids"]), len(tokenized["labels"]))

    def test_initialize_run_resumes_state_and_rejects_changed_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "serialized-sft-a"
            config_a = {"run_id": "serialized-sft-a", "resume_signature": {"train_file": "train-a"}}
            config_b = {"run_id": "serialized-sft-a", "resume_signature": {"train_file": "train-b"}}

            state = self.sft.initialize_run(run_dir, config_a, dry_run=False)
            self.sft.update_state(run_dir, state, global_step=3, epoch=1, next_batch_index=2)

            resumed = self.sft.initialize_run(run_dir, config_a, dry_run=False)
            self.assertEqual(resumed["global_step"], 3)
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(resumed["next_batch_index"], 2)

            with self.assertRaises(RunCompatibilityError):
                self.sft.initialize_run(run_dir, config_b, dry_run=False)

    def test_extract_blocks_tsv_prediction_and_evaluate_with_parser(self) -> None:
        row = build_sft_row()

        predicted_blocks = self.sft.extract_blocks_tsv_prediction(str(row["target"]))
        evaluation = evaluate_blocks_prediction(
            str(row["target_yaml_normalized"]),
            predicted_blocks,
            prompt_text=str(row["prompt"]),
        )

        self.assertTrue(evaluation.yaml_parse_ok)
        self.assertTrue(evaluation.parsed_equal_to_reference)
        self.assertEqual(evaluation.line_text_f1, 1.0)
        self.assertEqual(evaluation.level_exact_match_rate, 1.0)

    def test_intermediate_validation_predictions_are_scoped_by_checkpoint(self) -> None:
        row = build_sft_row()
        completion = {"raw_text": str(row["target"]), "generated_token_count": 12}
        with tempfile.TemporaryDirectory() as tmp:
            predictions_path = Path(tmp) / "intermediate_validation_predictions.jsonl"
            with mock.patch.object(
                self.sft,
                "generate_validation_completion",
                return_value=completion,
            ):
                first_metrics = self.sft.evaluate_validation(
                    run_id="serialized-sft-a",
                    run_dir=Path(tmp),
                    validation_rows=[row],
                    tokenizer=mock.Mock(),
                    model=mock.Mock(),
                    max_new_tokens=128,
                    checkpoint_name="checkpoint-step-5",
                    predictions_path=predictions_path,
                    resume_scope="checkpoint",
                )
                second_metrics = self.sft.evaluate_validation(
                    run_id="serialized-sft-a",
                    run_dir=Path(tmp),
                    validation_rows=[row],
                    tokenizer=mock.Mock(),
                    model=mock.Mock(),
                    max_new_tokens=128,
                    checkpoint_name="checkpoint-step-10",
                    predictions_path=predictions_path,
                    resume_scope="checkpoint",
                )
                repeated_metrics = self.sft.evaluate_validation(
                    run_id="serialized-sft-a",
                    run_dir=Path(tmp),
                    validation_rows=[row],
                    tokenizer=mock.Mock(),
                    model=mock.Mock(),
                    max_new_tokens=128,
                    checkpoint_name="checkpoint-step-10",
                    predictions_path=predictions_path,
                    resume_scope="checkpoint",
                )

            predictions = self.sft.read_jsonl(predictions_path)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(first_metrics["row_count"], 1)
            self.assertEqual(second_metrics["row_count"], 1)
            self.assertEqual(repeated_metrics["row_count"], 1)
            self.assertEqual(
                [row["checkpoint"] for row in predictions],
                ["checkpoint-step-5", "checkpoint-step-10"],
            )

    def test_validation_progress_metrics_include_structural_prompt_and_domain_signals(self) -> None:
        row = build_sft_row()
        evaluation = evaluate_blocks_prediction(
            str(row["target_yaml_normalized"]),
            [block.to_dict() for block in yaml_to_blocks(str(row["target_yaml_normalized"]))],
            prompt_text=str(row["prompt"]),
        )
        prediction = {
            "unit_id": "sample-1::question",
            "sample_id": "sample-1",
            "prompt_variant": "question",
            "split": "validation",
            "checkpoint": "checkpoint-step-final",
            "generated_token_count": 32,
            "parser_errors": [],
            "evaluation": evaluation.to_dict(),
        }

        metrics = self.sft.derive_validation_metrics(
            run_id="serialized-sft-a",
            predictions=[prediction],
            checkpoint="checkpoint-step-final",
        )
        enriched = self.sft.enrich_validation_progress_metrics(
            metrics,
            predictions=[prediction],
            total_count=70,
        )
        example = self.sft.validation_example_metrics(
            prediction,
            completed_count=1,
            total_count=70,
        )

        self.assertEqual(enriched["completed_count"], 1)
        self.assertEqual(enriched["remaining_count"], 69)
        self.assertIn("yaml_parse_success_rate", enriched)
        self.assertIn("average_level_exact_match_rate", enriched)
        self.assertIn("average_prompt_requirement_f1", enriched)
        self.assertIn("average_required_field_complete_resource_rate", enriched)
        self.assertIn("average_semantic_key_f1", enriched)
        self.assertEqual(example["yaml_parse_ok"], 1.0)
        self.assertEqual(example["line_text_f1"], 1.0)
        self.assertIn("required_field_complete_resource_rate", example)

    def test_intermediate_validation_is_capped_to_ten_samples(self) -> None:
        rows = [build_sft_row(f"validation-{index}") for index in range(12)]
        args = argparse.Namespace(
            run_id="serialized-sft-a",
            eval_max_samples=70,
            eval_sample_strategy="random",
            max_new_tokens=512,
            seed=42,
        )
        fake_metrics = {
            "yaml_parse_success_rate": 1.0,
            "average_prompt_requirement_f1": 1.0,
            "average_required_field_complete_resource_rate": 1.0,
            "average_line_text_f1": 1.0,
        }

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            self.sft,
            "evaluate_validation",
            return_value=fake_metrics,
        ) as evaluate_validation:
            metrics = self.sft.run_intermediate_validation(
                args=args,
                run_dir=Path(tmp),
                validation_rows=rows,
                tokenizer=mock.Mock(),
                model=mock.Mock(),
                wandb_run=None,
                global_step=10,
                checkpoint_name="checkpoint-step-10",
            )

        used_rows = evaluate_validation.call_args.kwargs["validation_rows"]
        self.assertEqual(len(used_rows), 10)
        self.assertEqual(metrics["eval_requested_max_samples"], 70)
        self.assertEqual(metrics["eval_max_samples_limit"], 10)
        self.assertEqual(metrics["eval_max_samples"], 10)
        self.assertEqual(metrics["eval_sample_strategy"], "random")
        self.assertEqual(len(metrics["eval_sample_unit_ids"]), 10)

    def test_intermediate_validation_random_sampling_is_deterministic_per_step(self) -> None:
        rows = [build_sft_row(f"validation-{index}") for index in range(20)]

        first_step_a = self.sft.select_intermediate_validation_rows(
            rows,
            max_samples=10,
            sample_strategy="random",
            seed=42,
            global_step=10,
        )
        first_step_b = self.sft.select_intermediate_validation_rows(
            rows,
            max_samples=10,
            sample_strategy="random",
            seed=42,
            global_step=10,
        )
        second_step = self.sft.select_intermediate_validation_rows(
            rows,
            max_samples=10,
            sample_strategy="random",
            seed=42,
            global_step=20,
        )

        first_ids_a = [self.sft.build_unit_id(row) for row in first_step_a]
        first_ids_b = [self.sft.build_unit_id(row) for row in first_step_b]
        second_ids = [self.sft.build_unit_id(row) for row in second_step]
        first_ten_ids = [self.sft.build_unit_id(row) for row in rows[:10]]

        self.assertEqual(first_ids_a, first_ids_b)
        self.assertNotEqual(first_ids_a, second_ids)
        self.assertNotEqual(first_ids_a, first_ten_ids)

    def test_log_validation_progress_writes_local_jsonl_and_wandb_payload(self) -> None:
        row = build_sft_row()
        evaluation = evaluate_blocks_prediction(
            str(row["target_yaml_normalized"]),
            [block.to_dict() for block in yaml_to_blocks(str(row["target_yaml_normalized"]))],
            prompt_text=str(row["prompt"]),
        )
        prediction = {
            "unit_id": "sample-1::question",
            "sample_id": "sample-1",
            "prompt_variant": "question",
            "split": "validation",
            "checkpoint": "checkpoint-step-final",
            "generated_token_count": 32,
            "parser_errors": [],
            "evaluation": evaluation.to_dict(),
        }
        fake_run = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            self.sft.log_validation_example(
                run_dir=run_dir,
                wandb_run=fake_run,
                prediction=prediction,
                completed_count=1,
                total_count=70,
            )
            self.sft.log_validation_progress(
                run_dir=run_dir,
                wandb_run=fake_run,
                run_id="serialized-sft-a",
                predictions=[prediction],
                checkpoint_name="checkpoint-step-final",
                total_count=70,
            )

            self.assertTrue((run_dir / "validation_example_metrics.jsonl").exists())
            self.assertTrue((run_dir / "validation_metrics_progress.jsonl").exists())
            progress = self.sft.read_jsonl(run_dir / "validation_metrics_progress.jsonl")[0]
            self.assertEqual(progress["completed_count"], 1)
            logged_payloads = [call.args[0] for call in fake_run.log.call_args_list]
            merged_keys = set().union(*(payload.keys() for payload in logged_payloads))
            self.assertIn("validation_example/yaml_parse_ok", merged_keys)
            self.assertIn("validation_progress/yaml_parse_success_rate", merged_keys)
            self.assertIn("validation_progress/average_prompt_requirement_f1", merged_keys)

    def test_dry_run_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_file = tmp_path / "train.jsonl"
            validation_file = tmp_path / "validation.jsonl"
            write_jsonl(train_file, [build_sft_row("train-1")])
            write_jsonl(validation_file, [build_sft_row("validation-1")])
            args = argparse.Namespace(
                model_variant="serialized_sft",
                serialization="blocks_tsv_v1",
                train_file=train_file,
                validation_file=validation_file,
                base_model_path=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit",
                output_dir=tmp_path / "runs",
                run_id="dry-run",
                batch_size=1,
                epochs=3,
                learning_rate=2e-4,
                gradient_accumulation_steps=8,
                max_seq_length=2048,
                max_new_tokens=1024,
                checkpoint_steps=25,
                checkpoint_keep_last=0,
                eval_checkpoint_steps=0,
                eval_max_samples=5,
                eval_sample_strategy="random",
                validation_log_every=1,
                oom_recovery="fail",
                max_oom_skips=0,
                max_train_samples=None,
                max_validation_samples=None,
                seed=42,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                lora_target_modules="q_proj,k_proj,v_proj,o_proj",
                warmup_ratio=0.03,
                weight_decay=0.0,
                gpu_memory="4.8GiB",
                cpu_memory="32GiB",
                wandb_mode="disabled",
                wandb_project="llm-structured-semantic-generation",
                wandb_entity=None,
                wandb_run_name=None,
                wandb_tags="",
                wandb_log_artifacts=False,
                dry_run=True,
                skip_final_eval=False,
            )
            fake_model_checks = {
                "ready_for_full_run": True,
                "warnings": [],
            }

            with mock.patch.object(self.sft, "parse_args", return_value=args), mock.patch.object(
                self.sft,
                "inspect_model_path",
                return_value=fake_model_checks,
            ):
                self.sft.main()

            run_dir = tmp_path / "runs" / "dry-run"
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "state.json").exists())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["dry_run"])
            self.assertEqual(metrics["row_count_train"], 1)
            self.assertEqual(metrics["row_count_validation"], 1)
            self.assertEqual(config["wandb"]["mode"], "disabled")
            self.assertEqual(config["wandb"]["project"], "llm-structured-semantic-generation")

    def test_prune_old_checkpoints_keeps_latest_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for step in (1, 2, 3, 5):
                checkpoint = run_dir / "checkpoints" / f"checkpoint-step-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "marker.txt").write_text(str(step), encoding="utf-8")

            removed = self.sft.prune_old_checkpoints(run_dir, keep_last=2)

            self.assertEqual(
                [path.name for path in self.sft.sorted_checkpoints(run_dir)],
                ["checkpoint-step-3", "checkpoint-step-5"],
            )
            self.assertEqual(
                [Path(path).name for path in removed],
                ["checkpoint-step-1", "checkpoint-step-2"],
            )

    def test_prune_old_checkpoints_keep_zero_preserves_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for step in (1, 2):
                (run_dir / "checkpoints" / f"checkpoint-step-{step}").mkdir(parents=True)

            removed = self.sft.prune_old_checkpoints(run_dir, keep_last=0)

            self.assertEqual(removed, [])
            self.assertEqual(len(self.sft.sorted_checkpoints(run_dir)), 2)

    def test_prune_old_checkpoints_logs_and_continues_when_delete_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for step in (1, 2, 3):
                (run_dir / "checkpoints" / f"checkpoint-step-{step}").mkdir(parents=True)

            real_rmtree = shutil.rmtree

            def fake_rmtree(path):
                if Path(path).name == "checkpoint-step-1":
                    raise PermissionError("locked")
                real_rmtree(path)

            with mock.patch.object(self.sft.time, "sleep"), mock.patch.object(
                self.sft.shutil,
                "rmtree",
                side_effect=fake_rmtree,
            ):
                removed = self.sft.prune_old_checkpoints(run_dir, keep_last=1)

            self.assertEqual([Path(path).name for path in removed], ["checkpoint-step-2"])
            self.assertTrue((run_dir / "checkpoints" / "checkpoint-step-1").exists())
            self.assertFalse((run_dir / "checkpoints" / "checkpoint-step-2").exists())
            errors = self.sft.read_jsonl(run_dir / "checkpoint_prune_errors.jsonl")
            self.assertEqual(errors[0]["checkpoint"], str(run_dir / "checkpoints" / "checkpoint-step-1"))
            self.assertIn("PermissionError", errors[0]["error"])

    def test_log_oom_skipped_batch_is_resumable_by_unit_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            batch = {
                "unit_ids": ["q229::question"],
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "labels": [[-100, 2, 3]],
            }

            row = self.sft.log_oom_skipped_batch(
                run_dir=run_dir,
                run_id="serialized-sft-a",
                epoch=0,
                batch_index=322,
                global_step=40,
                batch=batch,
                error=RuntimeError("CUDA out of memory"),
                cuda_memory={"allocated_bytes": 123},
            )

            self.assertEqual(row["unit_ids"], ["q229::question"])
            self.assertEqual(row["batch_index"], 322)
            self.assertEqual(
                self.sft.read_oom_skipped_unit_ids(run_dir),
                {"q229::question"},
            )

    def test_wandb_disabled_does_not_import_or_start_run(self) -> None:
        args = argparse.Namespace(
            wandb_mode="disabled",
            wandb_project="llm-structured-semantic-generation",
            wandb_entity=None,
            wandb_run_name=None,
            wandb_tags="",
            run_id="no-wandb",
        )

        run = self.sft.init_wandb_run(args=args, config={"run_id": "no-wandb"}, run_dir=Path("."))

        self.assertIsNone(run)

    def test_wandb_log_forwards_payload_when_run_is_present(self) -> None:
        fake_run = mock.Mock()

        self.sft.wandb_log(fake_run, {"train/loss": 1.25}, step=3)

        fake_run.log.assert_called_once_with({"train/loss": 1.25}, step=3)

    def test_load_env_file_sets_missing_values_without_overriding_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# local secrets",
                        "WANDB_API_KEY=from-file",
                        "EXISTING_VALUE=from-file",
                        "QUOTED_VALUE='hello'",
                    ]
                ),
                encoding="utf-8",
            )
            old_existing = os.environ.get("EXISTING_VALUE")
            old_key = os.environ.get("WANDB_API_KEY")
            old_quoted = os.environ.get("QUOTED_VALUE")
            os.environ["EXISTING_VALUE"] = "from-env"
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("QUOTED_VALUE", None)
            try:
                loaded = self.sft.load_env_file(env_path)

                self.assertEqual(os.environ["WANDB_API_KEY"], "from-file")
                self.assertEqual(os.environ["EXISTING_VALUE"], "from-env")
                self.assertEqual(os.environ["QUOTED_VALUE"], "hello")
                self.assertEqual(loaded, {"WANDB_API_KEY": "from-file", "QUOTED_VALUE": "hello"})
            finally:
                if old_existing is None:
                    os.environ.pop("EXISTING_VALUE", None)
                else:
                    os.environ["EXISTING_VALUE"] = old_existing
                if old_key is None:
                    os.environ.pop("WANDB_API_KEY", None)
                else:
                    os.environ["WANDB_API_KEY"] = old_key
                if old_quoted is None:
                    os.environ.pop("QUOTED_VALUE", None)
                else:
                    os.environ["QUOTED_VALUE"] = old_quoted


if __name__ == "__main__":
    unittest.main()
