import argparse
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import write_jsonl
from llm_structured_semantic_generation.evaluation import evaluate_blocks_prediction
from llm_structured_semantic_generation.sft_serialization import serialize_blocks_for_training
from llm_structured_semantic_generation.structure import yaml_to_blocks


def load_two_head_module():
    module_name = "test_train_kubernetes_two_head_sft"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "train_kubernetes_two_head_sft.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Could not load two-head SFT trainer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeOffsetTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, **kwargs):
        add_special_tokens = kwargs.get("add_special_tokens", False)
        ids = [ord(character) % 251 + 1 for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids = [252] + ids
            offsets = [(0, 0)] + offsets
        output = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = offsets
        return output


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


class TinyBackbone:
    def __init__(self):
        import torch

        class _Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)
                self.embed = torch.nn.Embedding(300, 4)
                self.lm = torch.nn.Linear(4, 300)

            def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True):
                hidden = self.embed(input_ids)
                logits = self.lm(hidden)
                loss = logits[..., 0].mean() * 0.0 + 0.7 if labels is not None else None
                return SimpleNamespace(loss=loss, logits=logits, hidden_states=(hidden,))

            def generate(self, *args, **kwargs):  # pragma: no cover - not needed here
                raise NotImplementedError

            def save_pretrained(self, path):  # pragma: no cover - not needed here
                Path(path).mkdir(parents=True, exist_ok=True)

        self.module = _Backbone()


class KubernetesTwoHeadSFTTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.two_head = load_two_head_module()

    def test_content_only_target_removes_gold_level_column(self) -> None:
        row = build_sft_row()

        example = self.two_head.build_content_only_example(row)

        self.assertIn("<content_blocks>", example.content_text)
        self.assertNotIn("<blocks>", example.content_text)
        data_lines = [
            line
            for line in example.content_text.splitlines()
            if line and line not in {"<content_blocks>", "</content_blocks>"}
        ]
        self.assertTrue(data_lines)
        for line in data_lines:
            self.assertEqual(len(line.split("\t", maxsplit=3)), 3)
        self.assertEqual([span.level for span in example.line_spans], [0, 0, 0, 1])

    def test_tokenize_aligns_one_record_prefix_level_per_line(self) -> None:
        row = build_sft_row()
        tokenizer = FakeOffsetTokenizer()

        tokenized = self.two_head.tokenize_two_head_row(row, tokenizer, max_seq_length=4096)

        self.assertEqual(tokenized["level_labels"], [0, 0, 0, 1])
        self.assertEqual(len(tokenized["level_label_positions"]), 4)
        active_token_labels = [
            value for value in tokenized["level_token_labels"] if value != self.two_head.IGNORE_INDEX
        ]
        self.assertEqual(active_token_labels, [0, 0, 0, 1])
        self.assertEqual(
            tokenized["level_label_positions"],
            [metadata["record_prefix_token_index"] for metadata in tokenized["level_metadata"]],
        )
        self.assertTrue(
            all(
                metadata["record_prefix_token_index"] < len(tokenized["input_ids"])
                for metadata in tokenized["level_metadata"]
            )
        )

    def test_collator_masks_non_line_positions_and_pads_line_labels(self) -> None:
        row_a = build_sft_row("sample-a")
        row_b = build_sft_row("sample-b")
        tokenizer = FakeOffsetTokenizer()
        item_a = self.two_head.tokenize_two_head_row(row_a, tokenizer, max_seq_length=4096)
        item_b = self.two_head.tokenize_two_head_row(row_b, tokenizer, max_seq_length=4096)
        item_b["level_labels"] = item_b["level_labels"][:-1]
        item_b["level_label_positions"] = item_b["level_label_positions"][:-1]

        batch = self.two_head.TwoHeadSFTCollator(tokenizer)([item_a, item_b])

        self.assertEqual(batch["level_labels"].shape[0], 2)
        self.assertEqual(batch["level_labels"][1, -1].item(), self.two_head.IGNORE_INDEX)
        self.assertEqual(batch["level_label_positions"][1, -1].item(), -1)
        self.assertGreater((batch["level_token_labels"] == self.two_head.IGNORE_INDEX).sum().item(), 0)

    def test_two_head_forward_returns_total_lm_and_level_losses(self) -> None:
        import torch

        backbone = TinyBackbone().module
        model = self.two_head.TwoHeadQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=3,
            level_head_hidden_dim=5,
            level_head_dropout=0.0,
            lambda_level=2.0,
        ).module
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        labels = torch.tensor([[-100, 2, 3, 4]], dtype=torch.long)
        positions = torch.tensor([[1, 3]], dtype=torch.long)
        level_labels = torch.tensor([[0, 1]], dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            level_label_positions=positions,
            level_labels=level_labels,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.lm_loss)
        self.assertIsNotNone(outputs.level_loss)
        self.assertAlmostEqual(
            outputs.loss.item(),
            outputs.lm_loss.item() + 2.0 * outputs.level_loss.item(),
            places=5,
        )
        self.assertEqual(tuple(outputs.level_logits.shape), (1, 2, 3))

    def test_extract_content_prediction_and_validation_metric_shape(self) -> None:
        content = (
            "<content_blocks>\n"
            "0\t0\tapiVersion: v1\n"
            "0\t1\tkind: ConfigMap\n"
            "0\t2\tmetadata:\n"
            "0\t3\tname: app\n"
            "</content_blocks>"
        )

        content_blocks, content_text, spans = self.two_head.extract_content_blocks_prediction(content)

        self.assertEqual(len(content_blocks), 4)
        self.assertIn("<content_blocks>", content_text)
        self.assertEqual(len(spans), 4)

        predicted_blocks = [
            {**block, "level": level}
            for block, level in zip(content_blocks, [0, 0, 0, 1], strict=True)
        ]
        evaluation = evaluate_blocks_prediction(
            str(build_sft_row()["target_yaml_normalized"]),
            predicted_blocks,
            prompt_text=str(build_sft_row()["prompt"]),
        )
        prediction = {
            "unit_id": "sample-1::question",
            "evaluation": evaluation.to_dict(),
        }
        metrics = self.two_head.derive_validation_metrics(
            run_id="two-head",
            predictions=[prediction],
            checkpoint="checkpoint-step-final",
        )

        self.assertEqual(metrics["model_variant"], "two_head_sft")
        self.assertEqual(metrics["serialization"], "content_blocks_v1")
        self.assertIn("average_level_exact_match_rate", metrics)

    def test_extract_content_prediction_repairs_missing_document_index_and_renumbers(self) -> None:
        content = (
            "<content_blocks>\n"
            "0\t0\tapiVersion: v1\n"
            "4\tkind: ConfigMap\n"
            "0\t9\tmetadata:\n"
            "0\t10\tname: app\n"
            "</content_blocks>"
        )

        content_blocks, content_text, spans = self.two_head.extract_content_blocks_prediction(content)

        self.assertEqual(
            content_blocks,
            [
                {"document_index": 0, "line_index": 0, "line_text": "apiVersion: v1"},
                {
                    "document_index": 0,
                    "line_index": 1,
                    "line_text": "kind: ConfigMap",
                    "surface_repair": "missing_document_index",
                    "line_index_normalized_from": 4,
                },
                {
                    "document_index": 0,
                    "line_index": 2,
                    "line_text": "metadata:",
                    "line_index_normalized_from": 9,
                },
                {
                    "document_index": 0,
                    "line_index": 3,
                    "line_text": "name: app",
                    "line_index_normalized_from": 10,
                },
            ],
        )
        self.assertIn("0\t1\tkind: ConfigMap", content_text)
        self.assertEqual([span.line_index for span in spans], [0, 1, 2, 3])

    def test_extract_content_prediction_rejects_unrepairable_malformed_rows(self) -> None:
        content = (
            "<content_blocks>\n"
            "0\t0\tapiVersion: v1\n"
            "---\n"
            "</content_blocks>"
        )

        with self.assertRaisesRegex(ValueError, "not_enough_content_tsv_fields"):
            self.two_head.extract_content_blocks_prediction(content)

    def test_dry_run_writes_expected_artifacts_and_wandb_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_file = tmp_path / "train.jsonl"
            validation_file = tmp_path / "validation.jsonl"
            write_jsonl(train_file, [build_sft_row("train-1")])
            write_jsonl(validation_file, [build_sft_row("validation-1")])
            args = argparse.Namespace(
                model_variant="two_head_sft",
                serialization="content_blocks_v1",
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
                lambda_level=1.0,
                level_class_count=9,
                level_head_hidden_dim=256,
                level_head_dropout=0.05,
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

            with mock.patch.object(self.two_head, "parse_args", return_value=args), mock.patch.object(
                self.two_head.serialized_sft,
                "inspect_model_path",
                return_value=fake_model_checks,
            ):
                self.two_head.main()

            run_dir = tmp_path / "runs" / "dry-run"
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "state.json").exists())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["dry_run"])
            self.assertEqual(config["model_variant"], "two_head_sft")
            self.assertEqual(config["level_head"]["loss"], "cross_entropy")
            self.assertEqual(config["wandb"]["mode"], "disabled")

    def test_wandb_disabled_does_not_import_or_start_run(self) -> None:
        args = argparse.Namespace(
            wandb_mode="disabled",
            wandb_project="llm-structured-semantic-generation",
            wandb_entity=None,
            wandb_run_name=None,
            wandb_tags="",
            run_id="no-wandb",
        )

        run = self.two_head.init_wandb_run(args=args, config={"run_id": "no-wandb"}, run_dir=Path("."))

        self.assertIsNone(run)

    def test_wandb_online_or_offline_mode_forwards_config_and_tags(self) -> None:
        fake_run = mock.Mock()
        fake_run.id = "two-head"
        fake_wandb = SimpleNamespace(
            init=mock.Mock(return_value=fake_run),
            define_metric=mock.Mock(),
        )
        args = argparse.Namespace(
            wandb_mode="offline",
            wandb_project="llm-structured-semantic-generation",
            wandb_entity=None,
            wandb_run_name=None,
            wandb_tags="custom",
            run_id="two-head",
        )
        old_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = fake_wandb
        try:
            with mock.patch.object(self.two_head, "find_spec", return_value=True):
                run = self.two_head.init_wandb_run(
                    args=args,
                    config={"run_id": "two-head"},
                    run_dir=Path("."),
                )
        finally:
            if old_wandb is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old_wandb

        self.assertIs(run, fake_run)
        kwargs = fake_wandb.init.call_args.kwargs
        self.assertEqual(kwargs["mode"], "offline")
        self.assertIn("two_head_sft", kwargs["tags"])
        self.assertIn("content_blocks_v1", kwargs["tags"])
        self.assertIn("record_prefix_state", kwargs["tags"])
        self.assertIn("custom", kwargs["tags"])


if __name__ == "__main__":
    unittest.main()
