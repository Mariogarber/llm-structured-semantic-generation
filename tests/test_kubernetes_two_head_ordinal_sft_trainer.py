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
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import write_jsonl
from llm_structured_semantic_generation.sft_serialization import serialize_blocks_for_training
from llm_structured_semantic_generation.structure import yaml_to_blocks


def load_ordinal_module():
    module_name = "test_train_kubernetes_two_head_ordinal_sft"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "train_kubernetes_two_head_ordinal_sft.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Could not load ordinal two-head SFT trainer module")
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

            def generate(self, *args, **kwargs):  # pragma: no cover
                raise NotImplementedError

            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        self.module = _Backbone()


class KubernetesTwoHeadOrdinalSFTTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ordinal = load_ordinal_module()

    def test_ordinal_targets_encode_greater_than_thresholds(self) -> None:
        import torch

        labels = torch.tensor([[0, 1, 3, 8]], dtype=torch.long)
        targets = self.ordinal.ordinal_targets(labels, level_class_count=9)

        self.assertEqual(tuple(targets.shape), (1, 4, 8))
        self.assertEqual(targets[0, 0].tolist(), [0.0] * 8)
        self.assertEqual(targets[0, 1].tolist(), [1.0] + [0.0] * 7)
        self.assertEqual(targets[0, 3].tolist(), [1.0] * 8)

    def test_thresholds_are_strictly_ordered(self) -> None:
        import torch

        head = self.ordinal.OrdinalLevelHead(
            hidden_size=4,
            hidden_dims=(5,),
            dropouts=(0.0,),
            level_class_count=9,
            level_weights=[1.0] * 9,
        ).module

        thresholds = head.thresholds().detach()

        self.assertEqual(tuple(thresholds.shape), (8,))
        self.assertTrue(bool(torch.all(thresholds[1:] > thresholds[:-1])))
        self.assertAlmostEqual(float(thresholds.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(thresholds[0]), -3.5, places=5)
        self.assertAlmostEqual(float(thresholds[-1]), 3.5, places=5)

    def test_threshold_initialization_can_use_centered_half_gap(self) -> None:
        import torch

        head = self.ordinal.OrdinalLevelHead(
            hidden_size=4,
            hidden_dims=(5,),
            dropouts=(0.0,),
            level_class_count=9,
            level_weights=[1.0] * 9,
            initial_threshold_center=0.0,
            initial_threshold_gap=0.5,
        ).module

        thresholds = head.thresholds().detach()
        expected = torch.tensor([-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75])

        self.assertTrue(bool(torch.allclose(thresholds, expected, atol=1e-5)))

    def test_forward_returns_ordinal_outputs_and_total_loss(self) -> None:
        import torch

        backbone = TinyBackbone().module
        model = self.ordinal.TwoHeadOrdinalQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=9,
            ordinal_head_hidden_dims=(5,),
            ordinal_head_dropouts=(0.0,),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
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
        self.assertIsNotNone(outputs.ordinal_level_loss)
        self.assertEqual(tuple(outputs.ordinal_z.shape), (1, 2))
        self.assertEqual(tuple(outputs.ordinal_logits.shape), (1, 2, 8))
        self.assertEqual(tuple(outputs.thresholds.shape), (8,))
        self.assertEqual(tuple(outputs.level_logits.shape), (1, 2, 9))
        self.assertAlmostEqual(
            outputs.loss.item(),
            outputs.lm_loss.item() + 2.0 * outputs.ordinal_level_loss.item(),
            places=5,
        )

    def test_density_weights_are_smoothed_and_capped(self) -> None:
        rows = [build_sft_row("sample-a"), build_sft_row("sample-b")]

        weights = self.ordinal.compute_level_density_weights(
            rows,
            level_class_count=9,
            kernel="triangular",
            radius=1,
            max_weight=3.0,
        )

        self.assertEqual(len(weights["weights"]), 9)
        self.assertLessEqual(max(weights["weights"]), 3.0)
        self.assertGreater(weights["weights"][8], weights["weights"][0])

    def test_save_and_load_checkpoint_uses_ordinal_head_file(self) -> None:
        import torch

        backbone = TinyBackbone().module
        model = self.ordinal.TwoHeadOrdinalQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=9,
            ordinal_head_hidden_dims=(5,),
            ordinal_head_dropouts=(0.0,),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
        ).module
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        state = {"global_step": 8, "epoch": 0, "next_batch_index": 0}

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = self.ordinal.save_checkpoint(
                run_dir=run_dir,
                model=model,
                tokenizer=FakeTokenizer(),
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
            )

            self.assertTrue((path / "ordinal_level_head.pt").exists())
            self.assertTrue((path / "training_state.pt").exists())

            with torch.no_grad():
                model.ordinal_level_head.raw_tau0.add_(10.0)
            self.ordinal.load_checkpoint_state(
                checkpoint_path=path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            self.assertLess(model.ordinal_level_head.raw_tau0.item(), 5.0)

    def test_optimizer_can_boost_threshold_learning_rate(self) -> None:
        backbone = TinyBackbone().module
        model = self.ordinal.TwoHeadOrdinalQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=9,
            ordinal_head_hidden_dims=(5,),
            ordinal_head_dropouts=(0.0,),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
        ).module
        args = argparse.Namespace(
            learning_rate=2e-4,
            ordinal_mlp_learning_rate_multiplier=3.0,
            threshold_learning_rate_multiplier=25.0,
            weight_decay=0.01,
            warmup_ratio=0.0,
        )

        optimizer, _scheduler = self.ordinal.build_optimizer_and_scheduler(args, model, total_optimizer_steps=10)
        base_group = next(group for group in optimizer.param_groups if group.get("name") == "base_lora")
        ordinal_mlp_group = next(group for group in optimizer.param_groups if group.get("name") == "ordinal_mlp")
        threshold_group = next(group for group in optimizer.param_groups if group.get("name") == "ordinal_thresholds")
        ordinal_mlp_parameter_ids = {id(parameter) for parameter in ordinal_mlp_group["params"]}
        threshold_parameter_ids = {id(parameter) for parameter in threshold_group["params"]}

        self.assertAlmostEqual(base_group["lr"], 2e-4)
        self.assertAlmostEqual(ordinal_mlp_group["lr"], 6e-4)
        self.assertEqual(ordinal_mlp_group["weight_decay"], 0.01)
        self.assertAlmostEqual(threshold_group["lr"], 5e-3)
        self.assertEqual(threshold_group["weight_decay"], 0.0)
        self.assertIn(id(model.ordinal_level_head.projector[1].weight), ordinal_mlp_parameter_ids)
        self.assertIn(id(model.ordinal_level_head.raw_tau0), threshold_parameter_ids)
        self.assertIn(id(model.ordinal_level_head.raw_deltas), threshold_parameter_ids)

        rates = self.ordinal.optimizer_learning_rates_by_group(optimizer, _scheduler)
        self.assertAlmostEqual(rates["base_lora"], 2e-4)
        self.assertAlmostEqual(rates["ordinal_mlp"], 6e-4)
        self.assertAlmostEqual(rates["ordinal_thresholds"], 5e-3)

    def test_gradient_diagnostics_measure_projector_thresholds_and_effective_shift(self) -> None:
        import torch

        backbone = TinyBackbone().module
        model = self.ordinal.TwoHeadOrdinalQwenForSFT(
            backbone,
            hidden_size=4,
            level_class_count=9,
            ordinal_head_hidden_dims=(5,),
            ordinal_head_dropouts=(0.0,),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
        ).module
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.tensor([[-100, 2, 3, 4]], dtype=torch.long),
            "level_label_positions": torch.tensor([[1, 3]], dtype=torch.long),
            "level_labels": torch.tensor([[0, 6]], dtype=torch.long),
        }

        outputs = model(**batch)
        outputs.loss.backward()
        before = self.ordinal.begin_ordinal_step_diagnostics(
            model=model,
            outputs=outputs,
            batch_on_device=batch,
        )
        optimizer.step()
        diagnostics = self.ordinal.complete_ordinal_step_diagnostics(model=model, before=before)
        payload = self.ordinal.gradient_diagnostics_wandb_payload(diagnostics)

        self.assertGreater(diagnostics["grad_element_count_ordinal_mlp"], 0)
        self.assertGreater(diagnostics["grad_element_count_ordinal_threshold_raw"], 0)
        self.assertGreater(diagnostics["update_rms_ordinal_mlp"], 0.0)
        self.assertGreater(diagnostics["update_rms_ordinal_threshold_raw"], 0.0)
        self.assertIn("effective_z_to_tau_mean_shift_ratio", diagnostics)
        self.assertIn("train/grad_rms/ordinal_mlp", payload)
        self.assertIn("train/effective_shift/z_to_tau_mean_ratio", payload)

    def test_resume_signature_includes_ordinal_hyperparameters(self) -> None:
        args = argparse.Namespace(
            model_variant="two_head_ordinal_density_v2",
            serialization="content_blocks_v1",
            train_file=Path("train.jsonl"),
            validation_file=Path("validation.jsonl"),
            base_model_path=Path("model"),
            batch_size=1,
            epochs=3,
            learning_rate=2e-4,
            gradient_accumulation_steps=8,
            max_seq_length=2048,
            max_new_tokens=1024,
            checkpoint_keep_last=0,
            max_train_samples=None,
            max_validation_samples=None,
            seed=42,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_target_modules="q_proj,k_proj,v_proj,o_proj",
            lambda_level=2.0,
            level_class_count=9,
            ordinal_head_hidden_dims="512,256",
            ordinal_head_dropouts="0.1,0.05",
            ordinal_mlp_learning_rate_multiplier=3.0,
            threshold_learning_rate_multiplier=25.0,
            initial_threshold_center=0.0,
            initial_threshold_gap=0.5,
            density_kernel="triangular",
            density_kernel_radius=1,
            max_density_weight=12.0,
        )

        signature = self.ordinal.build_resume_signature(args)

        self.assertEqual(signature["ordinal_head_hidden_dims"], [512, 256])
        self.assertEqual(signature["density_kernel"], "triangular")
        self.assertEqual(signature["max_density_weight"], 12.0)
        self.assertEqual(signature["ordinal_mlp_learning_rate_multiplier"], 3.0)
        self.assertEqual(signature["threshold_learning_rate_multiplier"], 25.0)
        self.assertEqual(signature["initial_threshold_center"], 0.0)
        self.assertEqual(signature["initial_threshold_gap"], 0.5)

    def test_curated_validation_payload_excludes_auxiliary_text_metrics(self) -> None:
        metrics = {
            "yaml_parse_success_rate": 0.5,
            "average_level_exact_match_rate": 0.6,
            "average_bleu_score": 0.9,
            "predicted_level_count_5": 3,
        }

        payload = self.ordinal.curated_validation_payload("validation", metrics)

        self.assertIn("validation/yaml_parse_success_rate", payload)
        self.assertIn("validation/predicted_level_count_5", payload)
        self.assertNotIn("validation/average_bleu_score", payload)

    def test_dry_run_writes_ordinal_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_file = tmp_path / "train.jsonl"
            validation_file = tmp_path / "validation.jsonl"
            write_jsonl(train_file, [build_sft_row("train-1")])
            write_jsonl(validation_file, [build_sft_row("validation-1")])
            args = argparse.Namespace(
                model_variant="two_head_ordinal_density_v2",
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
                checkpoint_steps=8,
                checkpoint_keep_last=0,
                eval_checkpoint_steps=8,
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
                lambda_level=2.0,
                level_class_count=9,
                ordinal_head_hidden_dims="512,256",
                ordinal_head_dropouts="0.1,0.05",
                ordinal_mlp_learning_rate_multiplier=1.0,
                threshold_learning_rate_multiplier=1.0,
                ordinal_gradient_diagnostics=True,
                density_kernel="triangular",
                density_kernel_radius=1,
                max_density_weight=12.0,
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

            with mock.patch.object(self.ordinal, "parse_args", return_value=args), mock.patch.object(
                self.ordinal.serialized_sft,
                "inspect_model_path",
                return_value=fake_model_checks,
            ):
                self.ordinal.main()

            run_dir = tmp_path / "runs" / "dry-run"
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

            self.assertTrue(metrics["dry_run"])
            self.assertEqual(config["model_variant"], "two_head_ordinal_density_v2")
            self.assertEqual(config["ordinal_mlp_learning_rate_multiplier"], 1.0)
            self.assertEqual(config["threshold_learning_rate_multiplier"], 1.0)
            self.assertTrue(config["ordinal_gradient_diagnostics"])
            self.assertEqual(config["initial_threshold_center"], 0.0)
            self.assertEqual(config["initial_threshold_gap"], 1.0)
            self.assertAlmostEqual(sum(config["initial_thresholds"]) / len(config["initial_thresholds"]), 0.0)
            self.assertEqual(config["ordinal_level_head"]["projector_optimizer"]["learning_rate_multiplier"], 1.0)
            self.assertEqual(config["ordinal_level_head"]["threshold_initialization"]["center"], 0.0)
            self.assertEqual(config["ordinal_level_head"]["threshold_initialization"]["gap"], 1.0)
            self.assertTrue(config["ordinal_level_head"]["threshold_initialization"]["trainable_after_initialization"])
            self.assertEqual(config["ordinal_level_head"]["loss"], "density_weighted_ordinal_bce")
            self.assertEqual(config["wandb_metric_policy"]["dashboard"], "curated")
            self.assertTrue(config["wandb_metric_policy"]["logs_ordinal_gradient_diagnostics"])


if __name__ == "__main__":
    unittest.main()
