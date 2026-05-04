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
                eval_checkpoint_steps=0,
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
            self.assertTrue(metrics["dry_run"])
            self.assertEqual(metrics["row_count_train"], 1)
            self.assertEqual(metrics["row_count_validation"], 1)


if __name__ == "__main__":
    unittest.main()
