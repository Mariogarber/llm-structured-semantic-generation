import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_regression_module():
    module_name = "test_train_kubernetes_two_head_regression_sft"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "train_kubernetes_two_head_regression_sft.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Could not load regression two-head SFT trainer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, **kwargs):
        ids = [ord(character) % 251 + 1 for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if kwargs.get("add_special_tokens", False):
            ids = [252] + ids
            offsets = [(0, 0)] + offsets
        output = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = offsets
        return output

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


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

            def generate(self, *args, **kwargs):  # pragma: no cover
                raise NotImplementedError

            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        self.module = _Backbone()


class KubernetesTwoHeadRegressionSFTTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.regression = load_regression_module()

    def test_huber_loss_uses_quadratic_then_linear_regions(self) -> None:
        import torch

        prediction = torch.tensor([0.5, 3.0])
        target = torch.tensor([0.0, 0.0])

        loss = self.regression.huber_loss_per_item(prediction, target, delta=1.0)

        self.assertTrue(torch.allclose(loss, torch.tensor([0.125, 2.5])))

    def test_forward_returns_regression_outputs_and_total_loss(self) -> None:
        import torch

        backbone = TinyBackbone().module
        model = self.regression.TwoHeadRegressionQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=9,
            regression_head_hidden_dims=(5,),
            regression_head_dropouts=(0.0,),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
            regression_huber_delta=1.0,
        ).module
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        labels = torch.tensor([[-100, 2, 3, 4]], dtype=torch.long)
        positions = torch.tensor([[1, 3]], dtype=torch.long)
        level_labels = torch.tensor([[0, 6]], dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            level_label_positions=positions,
            level_labels=level_labels,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.regression_level_loss)
        self.assertEqual(tuple(outputs.level_scores.shape), (1, 2))
        self.assertEqual(tuple(outputs.level_logits.shape), (1, 2, 9))
        self.assertIsNone(outputs.thresholds)
        self.assertIsNone(outputs.ordinal_logits)
        self.assertAlmostEqual(
            outputs.loss.item(),
            outputs.lm_loss.item() + 2.0 * outputs.regression_level_loss.item(),
            places=5,
        )

    def test_save_and_load_checkpoint_uses_regression_head_file(self) -> None:
        import torch

        backbone = TinyBackbone().module
        model = self.regression.TwoHeadRegressionQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=9,
            regression_head_hidden_dims=(5,),
            regression_head_dropouts=(0.0,),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
            regression_huber_delta=1.0,
        ).module
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        state = {"global_step": 8, "epoch": 0, "next_batch_index": 0}

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = self.regression.save_checkpoint(
                run_dir=run_dir,
                model=model,
                tokenizer=FakeTokenizer(),
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
            )

            self.assertTrue((path / "regression_level_head.pt").exists())
            self.assertTrue((path / "training_state.pt").exists())

            original = model.regression_level_head.projector[-1].bias.detach().clone()
            with torch.no_grad():
                model.regression_level_head.projector[-1].bias.add_(10.0)
            self.regression.load_checkpoint_state(
                checkpoint_path=path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            self.assertTrue(torch.allclose(model.regression_level_head.projector[-1].bias, original))

    def test_optimizer_can_boost_regression_head_learning_rate(self) -> None:
        backbone = TinyBackbone().module
        model = self.regression.TwoHeadRegressionQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=9,
            regression_head_hidden_dims=(5,),
            regression_head_dropouts=(0.0,),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
            regression_huber_delta=1.0,
        ).module
        args = argparse.Namespace(
            learning_rate=2e-4,
            regression_head_learning_rate_multiplier=3.0,
            weight_decay=0.01,
            warmup_ratio=0.0,
        )

        optimizer, scheduler = self.regression.build_optimizer_and_scheduler(args, model, total_optimizer_steps=10)
        base_group = next(group for group in optimizer.param_groups if group.get("name") == "base_lora")
        regression_group = next(group for group in optimizer.param_groups if group.get("name") == "regression_head")
        regression_parameter_ids = {id(parameter) for parameter in regression_group["params"]}

        self.assertAlmostEqual(base_group["lr"], 2e-4)
        self.assertAlmostEqual(regression_group["lr"], 6e-4)
        self.assertEqual(regression_group["weight_decay"], 0.01)
        self.assertIn(id(model.regression_level_head.projector[1].weight), regression_parameter_ids)

        rates = self.regression.optimizer_learning_rates_by_group(optimizer, scheduler)
        self.assertAlmostEqual(rates["base_lora"], 2e-4)
        self.assertAlmostEqual(rates["regression_head"], 6e-4)

    def test_dry_run_config_records_regression_contract(self) -> None:
        row = {
            "sample_id": "sample-1",
            "prompt_variant": "question",
            "split": "validation",
            "prompt": "Natural-language request:\nCreate a ConfigMap named app.\n\nReturn the structural block sequence now.",
            "target": "<blocks>\n0\t0\t0\tapiVersion: v1\n0\t1\t0\tkind: ConfigMap\n</blocks>",
            "target_yaml_normalized": "apiVersion: v1\nkind: ConfigMap\n",
            "round_trip_yaml": "apiVersion: v1\nkind: ConfigMap\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_file = tmp_path / "train.jsonl"
            validation_file = tmp_path / "validation.jsonl"
            train_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
            validation_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
            argv = [
                "trainer",
                "--run-id",
                "regression-dry-run",
                "--train-file",
                str(train_file),
                "--validation-file",
                str(validation_file),
                "--output-dir",
                str(tmp_path),
                "--dry-run",
                "--regression-head-learning-rate-multiplier",
                "3",
                "--regression-huber-delta",
                "1.5",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.regression.main()

            config = json.loads((tmp_path / "regression-dry-run" / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["model_variant"], "two_head_level_regression_huber_v1")
            self.assertEqual(config["level_regression_head"]["loss"], "density_weighted_huber")
            self.assertEqual(config["level_regression_head"]["huber_delta"], 1.5)
            self.assertEqual(config["regression_head_learning_rate_multiplier"], 3.0)
            self.assertEqual(config["wandb_metric_policy"]["logs_regression_scores"], True)


if __name__ == "__main__":
    unittest.main()
