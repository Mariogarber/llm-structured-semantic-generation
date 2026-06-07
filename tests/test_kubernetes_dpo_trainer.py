from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_dpo_trainer_module():
    module_name = "test_train_kubernetes_dpo"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "train_kubernetes_dpo.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Could not load DPO trainer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> dict[str, list[int]]:
        del truncation
        ids = [ord(char) % 97 + 3 for char in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


class KubernetesDPOTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_dpo_trainer_module()

    def test_output_dir_inside_model_or_sft_run_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe_dpo_output_dir"):
            self.module.assert_output_dir_safe(
                self.module.REPO_ROOT / "model" / "bad-run",
                base_model_path=self.module.DEFAULT_BASE_MODEL,
                sft_checkpoint_root=self.module.DEFAULT_SFT_CHECKPOINT,
            )

        with self.assertRaisesRegex(ValueError, "unsafe_dpo_output_dir"):
            self.module.assert_output_dir_safe(
                self.module.DEFAULT_SFT_CHECKPOINT / "bad-dpo-output",
                base_model_path=self.module.DEFAULT_BASE_MODEL,
                sft_checkpoint_root=self.module.DEFAULT_SFT_CHECKPOINT,
            )

    def test_preference_validation_allows_malformed_rejected_text(self) -> None:
        rows = [
            {
                "preference_id": "q1::question::p1",
                "split": "train",
                "prompt": "Create a ConfigMap.",
                "chosen": "<blocks>\n0\t0\t0\tapiVersion: v1\n</blocks>",
                "rejected": "<blocks>\n0\t0\t0\tapiVersion: v1\n0\t1 broken line",
            }
        ]

        self.module.validate_preference_rows(rows)

    def test_tokenize_completion_pair_masks_prompt_and_keeps_metadata(self) -> None:
        row = {
            "preference_id": "q1::question::p1",
            "unit_id": "q1::question",
            "prompt": "Prompt",
            "chosen": "<blocks>\n0\t0\t0\tkind: ConfigMap\n</blocks>",
            "rejected": "bad blocks",
            "pair_type": "strong_score_margin",
            "score_margin": 0.5,
        }

        item = self.module.tokenize_completion_pair(row, TinyTokenizer(), max_seq_length=512)

        self.assertEqual(item["preference_id"], "q1::question::p1")
        self.assertEqual(item["metadata"]["pair_type"], "strong_score_margin")
        self.assertTrue(any(label != -100 for label in item["chosen_labels"]))
        first_target = next(index for index, label in enumerate(item["chosen_labels"]) if label != -100)
        self.assertTrue(all(label == -100 for label in item["chosen_labels"][:first_target]))

    def test_collator_pads_pairs_and_reference_logps(self) -> None:
        rows = [
            {
                "preference_id": "p1",
                "unit_id": "u1",
                "prompt": "A",
                "chosen": "good",
                "rejected": "bad",
            },
            {
                "preference_id": "p2",
                "unit_id": "u2",
                "prompt": "Longer",
                "chosen": "better",
                "rejected": "worse",
            },
        ]
        reference = {
            "p1": {"chosen_reference_logp": -1.0, "rejected_reference_logp": -2.0},
            "p2": {"chosen_reference_logp": -3.0, "rejected_reference_logp": -4.0},
        }
        dataset = self.module.DPODataset(rows, TinyTokenizer(), max_seq_length=512, reference_logps=reference)
        batch = self.module.DPOCollator(TinyTokenizer())([dataset[0], dataset[1]])

        self.assertEqual(batch["chosen_input_ids"].shape[0], 2)
        self.assertEqual(batch["rejected_input_ids"].shape[0], 2)
        self.assertEqual(batch["preference_ids"], ["p1", "p2"])
        self.assertEqual(batch["chosen_reference_logps"].tolist(), [-1.0, -3.0])

    def test_dpo_loss_metrics_reflect_reward_margin(self) -> None:
        import torch

        loss, metrics = self.module.dpo_loss_from_logps(
            policy_chosen_logps=torch.tensor([-1.0, -2.0]),
            policy_rejected_logps=torch.tensor([-3.0, -4.0]),
            reference_chosen_logps=torch.tensor([-2.0, -2.0]),
            reference_rejected_logps=torch.tensor([-2.5, -2.5]),
            beta=0.1,
        )

        self.assertLess(float(loss.item()), 0.6932)
        self.assertAlmostEqual(float(metrics["reward_accuracy"].item()), 1.0)
        self.assertGreater(float(metrics["reward_margin"].item()), 0.0)

    def test_train_log_row_contains_wandb_core_metrics(self) -> None:
        args = argparse.Namespace(run_id="dpo-test")

        row = self.module.train_log_row(
            args=args,
            epoch=0,
            batch_index=2,
            global_step=1,
            metrics={
                "loss": 0.2,
                "reward_margin": 0.3,
                "reward_accuracy": 1.0,
                "chosen_reward": 0.4,
                "rejected_reward": 0.1,
                "chosen_logp": -10.0,
                "rejected_logp": -12.0,
                "logp_margin": 2.0,
                "reference_logp_margin": 1.0,
            },
            learning_rate=5e-6,
            grad_norm=0.7,
            skipped_batches=0,
        )

        self.assertEqual(row["run_id"], "dpo-test")
        self.assertEqual(row["reward_accuracy"], 1.0)
        self.assertEqual(row["grad_norm"], 0.7)
        self.assertEqual(row["learning_rate"], 5e-6)

    def test_two_thirds_validation_step_rounds_up(self) -> None:
        self.assertEqual(self.module.two_thirds_validation_step(1), 1)
        self.assertEqual(self.module.two_thirds_validation_step(3), 2)
        self.assertEqual(self.module.two_thirds_validation_step(57), 38)

    def test_select_two_thirds_validation_rows_is_deterministic(self) -> None:
        rows = [{"sample_id": f"s{index}"} for index in range(20)]

        first = self.module.select_two_thirds_validation_rows(
            rows,
            max_samples=10,
            sample_strategy="random",
            seed=42,
            global_step=38,
        )
        second = self.module.select_two_thirds_validation_rows(
            rows,
            max_samples=10,
            sample_strategy="random",
            seed=42,
            global_step=38,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)

    def test_wandb_disabled_does_not_require_api_key(self) -> None:
        args = argparse.Namespace(wandb_mode="disabled")

        self.module.validate_wandb_request(args)

    def test_wandb_online_requires_api_key(self) -> None:
        args = argparse.Namespace(wandb_mode="online")
        old_key = os.environ.get("WANDB_API_KEY")
        os.environ.pop("WANDB_API_KEY", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "wandb_online_requires_wandb_api_key"):
                self.module.validate_wandb_request(args)
        finally:
            if old_key is not None:
                os.environ["WANDB_API_KEY"] = old_key

    def test_wandb_summary_metrics_use_metric_axis_without_global_step_argument(self) -> None:
        class FakeWandbRun:
            def __init__(self) -> None:
                self.calls = []

            def log(self, payload, step=None) -> None:
                self.calls.append((payload, step))

        run = FakeWandbRun()

        self.module.wandb_log_numeric_metrics_on_metric_axis(
            run,
            {"global_step": 57, "yaml_parse_success_rate": 0.9, "status": "completed", "flag": True},
            prefix="validation",
        )

        self.assertEqual(run.calls[0][1], None)
        self.assertEqual(
            run.calls[0][0],
            {
                "validation/global_step": 57,
                "validation/yaml_parse_success_rate": 0.9,
            },
        )

    def test_load_reference_logps_rejects_duplicate_ids(self) -> None:
        from llm_structured_semantic_generation.dataset_io import write_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference_logps.jsonl"
            write_jsonl(
                path,
                [
                    {"preference_id": "p1", "chosen_reference_logp": -1.0, "rejected_reference_logp": -2.0},
                    {"preference_id": "p1", "chosen_reference_logp": -1.5, "rejected_reference_logp": -2.5},
                ],
            )

            with self.assertRaisesRegex(ValueError, "reference_logps_duplicate_preference_id"):
                self.module.load_reference_logps(path)


if __name__ == "__main__":
    unittest.main()
