from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import train_kubernetes_sft as serialized_sft
import train_kubernetes_two_head_ordinal_sft as ordinal_sft
from llm_structured_semantic_generation.dataset_io import append_jsonl, write_json
from llm_structured_semantic_generation.resumable_run import utc_now_iso


TWO_HEAD_REGRESSION_SFT = "two_head_level_regression_huber_v1"
CONTENT_BLOCKS_V1 = ordinal_sft.CONTENT_BLOCKS_V1
RECORD_PREFIX_STATE = ordinal_sft.RECORD_PREFIX_STATE
IGNORE_INDEX = ordinal_sft.IGNORE_INDEX
DEFAULT_LEVEL_CLASS_COUNT = ordinal_sft.DEFAULT_LEVEL_CLASS_COUNT
DEFAULT_REGRESSION_HEAD_HIDDEN_DIMS = ordinal_sft.DEFAULT_ORDINAL_HEAD_HIDDEN_DIMS
DEFAULT_REGRESSION_HEAD_DROPOUTS = ordinal_sft.DEFAULT_ORDINAL_HEAD_DROPOUTS
DEFAULT_LAMBDA_LEVEL = ordinal_sft.DEFAULT_LAMBDA_LEVEL
DEFAULT_DENSITY_KERNEL = ordinal_sft.DEFAULT_DENSITY_KERNEL
DEFAULT_DENSITY_KERNEL_RADIUS = ordinal_sft.DEFAULT_DENSITY_KERNEL_RADIUS
DEFAULT_MAX_DENSITY_WEIGHT = ordinal_sft.DEFAULT_MAX_DENSITY_WEIGHT
DEFAULT_REGRESSION_HUBER_DELTA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Kubernetes v1 Architecture B-regression: content_blocks_v1 "
            "plus a Huber level-regression head."
        )
    )
    parser.add_argument("--model-variant", choices=[TWO_HEAD_REGRESSION_SFT], default=TWO_HEAD_REGRESSION_SFT)
    parser.add_argument("--serialization", choices=[CONTENT_BLOCKS_V1], default=CONTENT_BLOCKS_V1)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "train.jsonl",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "validation.jsonl",
    )
    parser.add_argument("--base-model-path", type=Path, default=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "two_head_regression_sft_kubernetes_v1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=serialized_sft.positive_int, default=1)
    parser.add_argument("--epochs", type=serialized_sft.positive_int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=serialized_sft.positive_int, default=8)
    parser.add_argument("--max-seq-length", type=serialized_sft.positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=serialized_sft.positive_int, default=1024)
    parser.add_argument("--checkpoint-steps", type=serialized_sft.positive_int, default=25)
    parser.add_argument("--checkpoint-keep-last", type=serialized_sft.non_negative_int, default=0)
    parser.add_argument("--eval-checkpoint-steps", type=serialized_sft.non_negative_int, default=0)
    parser.add_argument("--eval-max-samples", type=serialized_sft.positive_int, default=10)
    parser.add_argument("--eval-sample-strategy", choices=["random", "first"], default="random")
    parser.add_argument("--validation-log-every", type=serialized_sft.non_negative_int, default=1)
    parser.add_argument("--oom-recovery", choices=["fail", "skip_batch"], default="fail")
    parser.add_argument("--max-oom-skips", type=serialized_sft.non_negative_int, default=0)
    parser.add_argument("--max-train-samples", type=serialized_sft.positive_int, default=None)
    parser.add_argument("--max-validation-samples", type=serialized_sft.positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=serialized_sft.positive_int, default=8)
    parser.add_argument("--lora-alpha", type=serialized_sft.positive_int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=",".join(serialized_sft.DEFAULT_LORA_TARGET_MODULES))
    parser.add_argument("--lambda-level", type=float, default=DEFAULT_LAMBDA_LEVEL)
    parser.add_argument("--level-class-count", type=serialized_sft.positive_int, default=DEFAULT_LEVEL_CLASS_COUNT)
    parser.add_argument(
        "--regression-head-hidden-dims",
        default=",".join(str(value) for value in DEFAULT_REGRESSION_HEAD_HIDDEN_DIMS),
    )
    parser.add_argument(
        "--regression-head-dropouts",
        default=",".join(str(value) for value in DEFAULT_REGRESSION_HEAD_DROPOUTS),
    )
    parser.add_argument(
        "--regression-head-learning-rate-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied only to the level-regression head.",
    )
    parser.add_argument(
        "--regression-huber-delta",
        type=float,
        default=DEFAULT_REGRESSION_HUBER_DELTA,
        help="Huber delta used by the level-regression loss.",
    )
    parser.add_argument(
        "--regression-gradient-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log curated gradient, optimizer-update, and same-hidden score-shift diagnostics for the regression head.",
    )
    parser.add_argument("--density-kernel", choices=["triangular", "uniform"], default=DEFAULT_DENSITY_KERNEL)
    parser.add_argument("--density-kernel-radius", type=serialized_sft.non_negative_int, default=DEFAULT_DENSITY_KERNEL_RADIUS)
    parser.add_argument("--max-density-weight", type=float, default=DEFAULT_MAX_DENSITY_WEIGHT)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="llm-structured-semantic-generation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def huber_loss_per_item(prediction: Any, target: Any, *, delta: float) -> Any:
    import torch

    if delta <= 0:
        raise ValueError("regression_huber_delta must be positive")
    diff = prediction - target
    abs_diff = diff.abs()
    delta_tensor = torch.as_tensor(float(delta), device=prediction.device, dtype=prediction.dtype)
    quadratic = 0.5 * diff * diff
    linear = delta_tensor * (abs_diff - 0.5 * delta_tensor)
    return torch.where(abs_diff <= delta_tensor, quadratic, linear)


class RegressionLevelHead:
    def __init__(
        self,
        *,
        hidden_size: int,
        hidden_dims: tuple[int, ...],
        dropouts: tuple[float, ...],
        level_class_count: int,
        level_weights: list[float],
    ) -> None:
        import torch

        class _RegressionLevelHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[torch.nn.Module] = [torch.nn.LayerNorm(hidden_size)]
                input_dim = hidden_size
                for index, output_dim in enumerate(hidden_dims):
                    layers.append(torch.nn.Linear(input_dim, output_dim))
                    layers.append(torch.nn.GELU())
                    dropout = dropouts[index] if index < len(dropouts) else 0.0
                    if dropout > 0:
                        layers.append(torch.nn.Dropout(dropout))
                    input_dim = output_dim
                layers.append(torch.nn.Linear(input_dim, 1))
                self.projector = torch.nn.Sequential(*layers)
                self.level_class_count = int(level_class_count)
                self.register_buffer("level_weights", torch.tensor(level_weights, dtype=torch.float32))

            def forward(self, selected_hidden: Any) -> Any:
                level_scores = self.projector(selected_hidden).squeeze(-1)
                predicted_levels = torch.round(level_scores).clamp(0, self.level_class_count - 1).to(dtype=torch.long)
                level_grid = torch.arange(
                    self.level_class_count,
                    device=level_scores.device,
                    dtype=level_scores.dtype,
                )
                level_logits = -torch.abs(level_scores.unsqueeze(-1) - level_grid)
                return SimpleNamespace(
                    level_scores=level_scores,
                    z=level_scores,
                    predicted_levels=predicted_levels,
                    level_logits=level_logits,
                    ordinal_logits=None,
                    thresholds=None,
                )

        self.module = _RegressionLevelHead()


class TwoHeadRegressionQwenForSFT:
    def __init__(
        self,
        backbone: Any,
        *,
        hidden_size: int,
        level_class_count: int,
        regression_head_hidden_dims: tuple[int, ...],
        regression_head_dropouts: tuple[float, ...],
        level_weights: list[float],
        lambda_level: float,
        regression_huber_delta: float,
    ) -> None:
        import torch

        class _TwoHeadRegressionModule(torch.nn.Module):
            def __init__(self, outer: "TwoHeadRegressionQwenForSFT") -> None:
                super().__init__()
                self.outer = outer
                self.backbone = outer.backbone
                self.regression_level_head = outer.regression_level_head

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self.outer.forward(*args, **kwargs)

            def generate(self, *args: Any, **kwargs: Any) -> Any:
                return self.backbone.generate(*args, **kwargs)

            @property
            def config(self) -> Any:
                return self.backbone.config

        self.backbone = backbone
        self.lambda_level = float(lambda_level)
        self.level_class_count = int(level_class_count)
        self.regression_huber_delta = float(regression_huber_delta)
        self.regression_level_head = RegressionLevelHead(
            hidden_size=hidden_size,
            hidden_dims=regression_head_hidden_dims,
            dropouts=regression_head_dropouts,
            level_class_count=level_class_count,
            level_weights=level_weights,
        ).module
        self.module = _TwoHeadRegressionModule(self)

    def __getattr__(self, name: str) -> Any:
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def parameters(self) -> Any:
        return self.module.parameters()

    def train(self, mode: bool = True) -> Any:
        return self.module.train(mode)

    def eval(self) -> Any:
        return self.module.eval()

    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
        labels: Any | None = None,
        level_label_positions: Any | None = None,
        level_labels: Any | None = None,
        level_token_labels: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        level_scores = None
        level_logits = None
        predicted_levels = None
        level_loss = None
        if level_label_positions is not None:
            safe_positions = level_label_positions.clamp(min=0)
            gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            selected_hidden = hidden.gather(1, gather_index)
            head_device = next(self.regression_level_head.parameters()).device
            if head_device != selected_hidden.device:
                self.regression_level_head.to(selected_hidden.device)
            head_dtype = next(self.regression_level_head.parameters()).dtype
            selected_hidden = selected_hidden.to(dtype=head_dtype)
            head_outputs = self.regression_level_head(selected_hidden)
            level_scores = head_outputs.level_scores
            level_logits = head_outputs.level_logits
            predicted_levels = head_outputs.predicted_levels
            if level_labels is not None:
                valid = (level_label_positions >= 0) & (level_labels != IGNORE_INDEX)
                if bool(valid.any()):
                    safe_level_labels = level_labels.clamp(min=0, max=self.level_class_count - 1)
                    targets = safe_level_labels.to(device=level_scores.device, dtype=level_scores.dtype)
                    per_item_loss = huber_loss_per_item(
                        level_scores,
                        targets,
                        delta=self.regression_huber_delta,
                    )
                    weights = self.regression_level_head.level_weights.to(
                        device=level_scores.device,
                        dtype=level_scores.dtype,
                    )[safe_level_labels]
                    level_loss = (per_item_loss[valid] * weights[valid]).mean()
                else:
                    level_loss = level_scores.sum() * 0.0

        lm_loss = outputs.loss if labels is not None else None
        if lm_loss is not None and level_loss is not None:
            loss = lm_loss + self.lambda_level * level_loss
        elif lm_loss is not None:
            loss = lm_loss
        elif level_loss is not None:
            loss = self.lambda_level * level_loss
        else:
            loss = None
        return SimpleNamespace(
            loss=loss,
            lm_loss=lm_loss,
            level_loss=level_loss,
            regression_level_loss=level_loss,
            ordinal_level_loss=level_loss,
            level_scores=level_scores,
            ordinal_z=level_scores,
            level_logits=level_logits,
            ordinal_logits=None,
            predicted_levels=predicted_levels,
            thresholds=None,
            logits=getattr(outputs, "logits", None),
            hidden_states=getattr(outputs, "hidden_states", None),
        )


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.base_model_path, local_files_only=True)
    quantization_config = getattr(config, "quantization_config", None)
    is_bitsandbytes_4bit = isinstance(quantization_config, dict) and bool(quantization_config.get("load_in_4bit"))
    load_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available() and is_bitsandbytes_4bit:
        load_kwargs["device_map"] = {"": 0}
    else:
        max_memory = {"cpu": args.cpu_memory}
        if torch.cuda.is_available():
            max_memory[0] = args.gpu_memory
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
    if is_bitsandbytes_4bit:
        backbone = prepare_model_for_kbit_training(backbone)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(ordinal_sft.split_csv(args.lora_target_modules)),
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.config.use_cache = False
    model = TwoHeadRegressionQwenForSFT(
        backbone,
        hidden_size=int(config.hidden_size),
        level_class_count=args.level_class_count,
        regression_head_hidden_dims=ordinal_sft.split_int_csv(args.regression_head_hidden_dims),
        regression_head_dropouts=ordinal_sft.split_float_csv(args.regression_head_dropouts),
        level_weights=list(getattr(args, "level_density_weights", [1.0] * args.level_class_count)),
        lambda_level=args.lambda_level,
        regression_huber_delta=args.regression_huber_delta,
    ).module
    return tokenizer, model


def save_checkpoint(
    *,
    run_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    state: dict[str, Any],
) -> Path:
    import torch

    path = serialized_sft.checkpoint_dir(run_dir, int(state["global_step"]))
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "tokenizer")
    torch.save(model.regression_level_head.state_dict(), path / "regression_level_head.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
        },
        path / "training_state.pt",
    )
    return path


def load_checkpoint_state(
    *,
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import set_peft_model_state_dict

    adapter_path = checkpoint_path / "adapter"
    if adapter_path.exists():
        if (adapter_path / "adapter_model.safetensors").exists():
            from safetensors.torch import load_file

            adapter_state = load_file(adapter_path / "adapter_model.safetensors")
        elif (adapter_path / "adapter_model.bin").exists():
            adapter_state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
        else:
            adapter_state = None
        if adapter_state is not None:
            set_peft_model_state_dict(model.backbone, adapter_state, adapter_name="default")
    level_head_path = checkpoint_path / "regression_level_head.pt"
    if level_head_path.exists():
        model.regression_level_head.load_state_dict(torch.load(level_head_path, map_location="cpu"))
    payload = torch.load(checkpoint_path / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    ordinal_sft.move_optimizer_state_to_active_devices(optimizer)
    scheduler.load_state_dict(payload["scheduler"])
    return model, payload["state"]


def predict_levels_for_content(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    content_blocks: list[dict[str, Any]],
    content_text: str,
    spans: list[ordinal_sft.ContentLineSpan],
) -> list[dict[str, Any]]:
    import torch

    aligned = ordinal_sft.positions_for_content_prediction(
        tokenizer=tokenizer,
        prompt=prompt,
        content_text=content_text,
        spans=spans,
    )
    device = ordinal_sft.model_input_device(model)
    input_ids = torch.tensor([aligned["input_ids"]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([aligned["attention_mask"]], dtype=torch.long).to(device)
    positions = torch.tensor([aligned["level_label_positions"]], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            level_label_positions=positions,
        )
    predicted_levels = outputs.predicted_levels[0].detach().cpu().tolist()
    level_scores = outputs.level_scores[0].detach().cpu().tolist()
    return [
        {
            "document_index": int(block["document_index"]),
            "line_index": int(block["line_index"]),
            "level": int(predicted_levels[index]),
            "level_score": float(level_scores[index]),
            "ordinal_score": float(level_scores[index]),
            "line_text": str(block["line_text"]),
        }
        for index, block in enumerate(content_blocks)
    ]


def regression_history_snapshot(*, outputs: Any | None = None, valid_mask: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if outputs is not None and getattr(outputs, "level_scores", None) is not None:
        scores = outputs.level_scores.detach().float()
        if valid_mask is not None:
            scores = scores[valid_mask]
        scores = scores.cpu()
        if scores.numel():
            payload["level_score_mean"] = float(scores.mean())
            payload["level_score_min"] = float(scores.min())
            payload["level_score_max"] = float(scores.max())
    if outputs is not None and getattr(outputs, "predicted_levels", None) is not None:
        predicted_tensor = outputs.predicted_levels.detach()
        if valid_mask is not None:
            predicted_tensor = predicted_tensor[valid_mask]
        predicted = predicted_tensor.cpu().reshape(-1).tolist()
        counts = {level: 0 for level in range(DEFAULT_LEVEL_CLASS_COUNT)}
        for level in predicted:
            counts[int(level)] = counts.get(int(level), 0) + 1
        for level in range(DEFAULT_LEVEL_CLASS_COUNT):
            payload[f"batch_predicted_level_count_{level}"] = counts[level]
    return payload


def regression_history_wandb_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in snapshot.items():
        if key == "level_score_mean":
            payload["train/regression_score/mean"] = value
        elif key == "level_score_min":
            payload["train/regression_score/min"] = value
        elif key == "level_score_max":
            payload["train/regression_score/max"] = value
        elif key.startswith("batch_predicted_level_count_"):
            payload[f"train/batch_predicted_level/count_{key.rsplit('_', 1)[-1]}"] = value
    return payload


def regression_diagnostic_parameter_groups(model: Any) -> dict[str, list[Any]]:
    named_parameters = model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()
    groups: dict[str, list[Any]] = {"regression_head": []}
    seen_parameter_ids: set[int] = set()
    for name, parameter in named_parameters:
        if not getattr(parameter, "requires_grad", False):
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        if ".regression_level_head.projector." in f".{name}":
            groups["regression_head"].append(parameter)
    return groups


def selected_hidden_for_level_head(*, model: Any, outputs: Any, level_label_positions: Any) -> Any | None:
    if getattr(outputs, "hidden_states", None) is None:
        return None
    hidden = outputs.hidden_states[-1]
    safe_positions = level_label_positions.clamp(min=0)
    gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    selected_hidden = hidden.gather(1, gather_index)
    head_dtype = next(model.regression_level_head.parameters()).dtype
    return selected_hidden.detach().to(dtype=head_dtype)


def regression_head_probe_snapshot(*, model: Any, selected_hidden: Any, valid_mask: Any | None) -> dict[str, Any]:
    import torch

    was_training = model.regression_level_head.training
    model.regression_level_head.eval()
    try:
        with torch.no_grad():
            head_device = next(model.regression_level_head.parameters()).device
            head_outputs = model.regression_level_head(selected_hidden.to(device=head_device))
            scores = head_outputs.level_scores.detach().float()
            if valid_mask is not None:
                scores = scores[valid_mask.to(device=scores.device)]
    finally:
        model.regression_level_head.train(was_training)
    return {"scores": scores.cpu()}


def begin_regression_step_diagnostics(*, model: Any, outputs: Any, batch_on_device: dict[str, Any]) -> dict[str, Any]:
    groups = regression_diagnostic_parameter_groups(model)
    gradients = {
        group_name: ordinal_sft.tensor_collection_stats([parameter.grad for parameter in parameters if parameter.grad is not None])
        for group_name, parameters in groups.items()
    }
    parameter_snapshots = {
        group_name: ordinal_sft.parameter_snapshot(parameters)
        for group_name, parameters in groups.items()
    }
    valid_mask = (batch_on_device["level_label_positions"] >= 0) & (batch_on_device["level_labels"] >= 0)
    selected_hidden = selected_hidden_for_level_head(
        model=model,
        outputs=outputs,
        level_label_positions=batch_on_device["level_label_positions"],
    )
    probe_before = None
    if selected_hidden is not None:
        probe_before = regression_head_probe_snapshot(
            model=model,
            selected_hidden=selected_hidden,
            valid_mask=valid_mask,
        )
    return {
        "groups": groups,
        "gradients": gradients,
        "parameter_snapshots": parameter_snapshots,
        "selected_hidden": selected_hidden,
        "valid_mask": valid_mask,
        "probe_before": probe_before,
    }


def complete_regression_step_diagnostics(*, model: Any, before: dict[str, Any]) -> dict[str, Any]:
    import torch

    row: dict[str, Any] = {}
    groups: dict[str, list[Any]] = before["groups"]
    for group_name, stats in before["gradients"].items():
        row[f"grad_norm_{group_name}"] = stats["norm"]
        row[f"grad_rms_{group_name}"] = stats["rms"]
        row[f"grad_max_abs_{group_name}"] = stats["max_abs"]
        row[f"grad_element_count_{group_name}"] = stats["element_count"]
    for group_name, parameters in groups.items():
        stats = ordinal_sft.parameter_update_stats(parameters, before["parameter_snapshots"][group_name])
        row[f"update_norm_{group_name}"] = stats["norm"]
        row[f"update_rms_{group_name}"] = stats["rms"]
        row[f"update_max_abs_{group_name}"] = stats["max_abs"]
        row[f"update_element_count_{group_name}"] = stats["element_count"]

    probe_before = before.get("probe_before")
    selected_hidden = before.get("selected_hidden")
    if probe_before is not None and selected_hidden is not None:
        probe_after = regression_head_probe_snapshot(
            model=model,
            selected_hidden=selected_hidden,
            valid_mask=before.get("valid_mask"),
        )
        scores_before = probe_before["scores"]
        scores_after = probe_after["scores"]
        if scores_before.numel() and scores_after.shape == scores_before.shape:
            score_delta = scores_after - scores_before
            row["effective_level_score_mean_abs_shift"] = float(score_delta.abs().mean())
            row["effective_level_score_rms_shift"] = float(torch.sqrt(torch.mean(score_delta * score_delta)))
            row["effective_level_score_max_abs_shift"] = float(score_delta.abs().max())
            row["probe_level_score_before_mean"] = float(scores_before.mean())
            row["probe_level_score_after_mean"] = float(scores_after.mean())
    return row


def gradient_diagnostics_wandb_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "grad_norm_regression_head": "train/grad_norm/regression_head",
        "grad_rms_regression_head": "train/grad_rms/regression_head",
        "update_norm_regression_head": "train/update_norm/regression_head",
        "update_rms_regression_head": "train/update_rms/regression_head",
        "effective_level_score_mean_abs_shift": "train/effective_shift/level_score_mean_abs",
        "effective_level_score_rms_shift": "train/effective_shift/level_score_rms",
        "effective_level_score_max_abs_shift": "train/effective_shift/level_score_max_abs",
    }
    return {
        wandb_key: snapshot[key]
        for key, wandb_key in key_map.items()
        if key in snapshot
    }


def init_wandb_run(*, args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    if ordinal_sft.find_spec("wandb") is None:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. "
            "Install optional LLM dependencies with `uv sync --extra llm`, or run with --wandb-mode disabled."
        )
    import wandb

    tags = list(ordinal_sft.split_csv(args.wandb_tags))
    for tag in (TWO_HEAD_REGRESSION_SFT, CONTENT_BLOCKS_V1, RECORD_PREFIX_STATE, "level_regression_huber"):
        if tag not in tags:
            tags.append(tag)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.run_id,
        id=args.run_id,
        resume="allow",
        mode=args.wandb_mode,
        dir=str(run_dir),
        tags=tags,
        config=config,
    )
    wandb.define_metric("validation_progress/completed_count")
    wandb.define_metric("validation_progress/*", step_metric="validation_progress/completed_count")
    wandb.define_metric("validation_example/completed_count")
    wandb.define_metric("validation_example/*", step_metric="validation_example/completed_count")
    wandb.define_metric("train/regression_score/*")
    wandb.define_metric("train/batch_predicted_level/*")
    wandb.define_metric("train/grad_norm/*")
    wandb.define_metric("train/grad_rms/*")
    wandb.define_metric("train/update_norm/*")
    wandb.define_metric("train/update_rms/*")
    wandb.define_metric("train/effective_shift/*")
    return run


def wandb_log_artifacts(
    wandb_run: Any | None,
    *,
    run_dir: Path,
    final_checkpoint: str | None,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb
    except ImportError:  # pragma: no cover
        return
    artifact = wandb.Artifact(name=f"{wandb_run.id}-two-head-regression-sft-artifacts", type="two-head-regression-sft-run")
    for filename in (
        "config.json",
        "state.json",
        "metrics.json",
        "train_log.jsonl",
        "regression_history.jsonl",
        "gradient_diagnostics.jsonl",
        "validation_predictions.jsonl",
        "intermediate_validation_predictions.jsonl",
        "intermediate_validation_metrics.jsonl",
        "validation_metrics_progress.jsonl",
        "validation_example_metrics.jsonl",
        "oom_skipped_batches.jsonl",
        "checkpoint_retention_log.jsonl",
        "checkpoint_prune_errors.jsonl",
    ):
        path = run_dir / filename
        if path.exists():
            artifact.add_file(str(path))
    if final_checkpoint:
        checkpoint_path = Path(final_checkpoint)
        if checkpoint_path.exists():
            artifact.add_dir(str(checkpoint_path), name=checkpoint_path.name)
    wandb_run.log_artifact(artifact)


def build_resume_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_variant": args.model_variant,
        "serialization": args.serialization,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "base_model_path": str(args.base_model_path),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_keep_last": args.checkpoint_keep_last,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": list(ordinal_sft.split_csv(args.lora_target_modules)),
        "lambda_level": args.lambda_level,
        "level_class_count": args.level_class_count,
        "regression_head_hidden_dims": list(ordinal_sft.split_int_csv(args.regression_head_hidden_dims)),
        "regression_head_dropouts": list(ordinal_sft.split_float_csv(args.regression_head_dropouts)),
        "regression_huber_delta": args.regression_huber_delta,
        "density_kernel": args.density_kernel,
        "density_kernel_radius": args.density_kernel_radius,
        "max_density_weight": args.max_density_weight,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "target_contract": "prompt -> content_blocks_v1 + Huber level-regression head -> blocks_tsv_v1 -> parser -> YAML",
        "regression_head_learning_rate_multiplier": args.regression_head_learning_rate_multiplier,
    }


def build_optimizer_and_scheduler(args: argparse.Namespace, model: Any, total_optimizer_steps: int):
    import torch
    from transformers import get_linear_schedule_with_warmup

    regression_lr_multiplier = float(getattr(args, "regression_head_learning_rate_multiplier", 1.0))
    if regression_lr_multiplier <= 0:
        raise ValueError("regression_head_learning_rate_multiplier must be positive")

    named_parameters = model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()
    base_parameters = []
    regression_parameters = []
    seen_parameter_ids: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        if ".regression_level_head.projector." in f".{name}":
            regression_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if base_parameters:
        parameter_groups.append(
            {
                "params": base_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "base_lora",
            }
        )
    if regression_parameters:
        parameter_groups.append(
            {
                "params": regression_parameters,
                "lr": args.learning_rate * regression_lr_multiplier,
                "weight_decay": args.weight_decay,
                "name": "regression_head",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    return optimizer, scheduler


def optimizer_learning_rates_by_group(optimizer: Any, scheduler: Any) -> dict[str, float]:
    scheduler_lrs = list(scheduler.get_last_lr())
    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("name") or f"group_{index}"
        rates[str(name)] = float(scheduler_lrs[index]) if index < len(scheduler_lrs) else float(group.get("lr", 0.0))
    if "base_lora" not in rates and scheduler_lrs:
        rates["base_lora"] = float(scheduler_lrs[0])
    if "regression_head" not in rates:
        rates["regression_head"] = rates.get("base_lora", float(scheduler_lrs[0]) if scheduler_lrs else 0.0)
    return rates


def install_evaluation_hooks() -> None:
    ordinal_sft.predict_levels_for_content = predict_levels_for_content


def train(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    install_evaluation_hooks()
    torch.manual_seed(args.seed)
    run_dir = args.output_dir / args.run_id
    wandb_run = init_wandb_run(args=args, config=config, run_dir=run_dir)
    density_weights = ordinal_sft.compute_level_density_weights(
        train_rows,
        level_class_count=args.level_class_count,
        kernel=args.density_kernel,
        radius=args.density_kernel_radius,
        max_weight=args.max_density_weight,
    )
    args.level_density_weights = density_weights["weights"]
    tokenizer, model = load_model_and_tokenizer(args)
    train_dataset = ordinal_sft.TwoHeadSFTDataset(train_rows, tokenizer, max_seq_length=args.max_seq_length)
    collator = ordinal_sft.TwoHeadSFTCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_optimizer_steps = max(optimizer_steps_per_epoch * args.epochs, 1)
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_optimizer_steps)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    checkpoint_path = serialized_sft.latest_checkpoint(run_dir)
    if checkpoint_path is not None and (checkpoint_path / "training_state.pt").exists():
        model, checkpoint_state = load_checkpoint_state(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        state.update(checkpoint_state)
        serialized_sft.update_state(run_dir, state)

    device = ordinal_sft.model_input_device(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_lm_loss = 0.0
    rolling_regression_level_loss = 0.0
    rolling_count = 0
    oom_skipped_unit_ids = serialized_sft.read_oom_skipped_unit_ids(run_dir) if args.oom_recovery == "skip_batch" else set()
    oom_skip_count = len(oom_skipped_unit_ids)

    for epoch in range(int(state.get("epoch", 0)), args.epochs):
        start_batch_index = int(state.get("next_batch_index", 0)) if epoch == int(state.get("epoch", 0)) else 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < start_batch_index:
                continue
            unit_ids = [str(unit_id) for unit_id in batch.get("unit_ids", [])]
            next_epoch = epoch
            next_batch_index = batch_index + 1
            if next_batch_index >= len(train_loader):
                next_epoch = epoch + 1
                next_batch_index = 0
            if args.oom_recovery == "skip_batch" and any(unit_id in oom_skipped_unit_ids for unit_id in unit_ids):
                state = serialized_sft.update_state(
                    run_dir,
                    state,
                    status="running",
                    epoch=next_epoch,
                    next_batch_index=next_batch_index,
                    oom_skipped_batches=oom_skip_count,
                )
                continue
            batch_on_device = None
            outputs = None
            loss = None
            try:
                batch_on_device = ordinal_sft.move_batch_to_device(batch, device)
                outputs = model(**batch_on_device)
                loss = outputs.loss
                (loss / args.gradient_accumulation_steps).backward()
            except torch.OutOfMemoryError as exc:
                if args.oom_recovery != "skip_batch":
                    raise
                cuda_memory = serialized_sft.cuda_memory_snapshot(torch)
                optimizer.zero_grad(set_to_none=True)
                rolling_loss = 0.0
                rolling_lm_loss = 0.0
                rolling_regression_level_loss = 0.0
                rolling_count = 0
                del batch_on_device, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                oom_row = serialized_sft.log_oom_skipped_batch(
                    run_dir=run_dir,
                    run_id=args.run_id,
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=int(state.get("global_step", 0)),
                    batch=batch,
                    error=exc,
                    cuda_memory=cuda_memory,
                )
                oom_skipped_unit_ids.update(unit_ids)
                oom_skip_count += 1
                state = serialized_sft.update_state(
                    run_dir,
                    state,
                    status="running",
                    epoch=next_epoch,
                    next_batch_index=next_batch_index,
                    oom_skipped_batches=oom_skip_count,
                    last_oom_skipped_batch=oom_row,
                )
                serialized_sft.wandb_log(
                    wandb_run,
                    {
                        "train/oom_skipped_batches": oom_skip_count,
                        "train/oom_skipped_samples": len(oom_skipped_unit_ids),
                    },
                    step=int(state.get("global_step", 0)),
                )
                if args.max_oom_skips and oom_skip_count > args.max_oom_skips:
                    raise RuntimeError(f"max_oom_skips_exceeded:{oom_skip_count}>{args.max_oom_skips}") from exc
                continue
            rolling_loss += float(loss.detach().cpu())
            rolling_lm_loss += float(outputs.lm_loss.detach().cpu()) if outputs.lm_loss is not None else 0.0
            rolling_regression_level_loss += (
                float(outputs.regression_level_loss.detach().cpu()) if outputs.regression_level_loss is not None else 0.0
            )
            rolling_count += 1

            should_step = (
                rolling_count >= args.gradient_accumulation_steps
                or (batch_index + 1 == len(train_loader) and rolling_count > 0)
            )
            if not should_step:
                continue

            diagnostic_before = None
            if getattr(args, "regression_gradient_diagnostics", True):
                diagnostic_before = begin_regression_step_diagnostics(
                    model=model,
                    outputs=outputs,
                    batch_on_device=batch_on_device,
                )
            ordinal_sft.move_optimizer_state_to_active_devices(optimizer)
            optimizer.step()
            gradient_diagnostics = (
                complete_regression_step_diagnostics(model=model, before=diagnostic_before)
                if diagnostic_before is not None
                else {}
            )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step = int(state.get("global_step", 0)) + 1
            average_loss = rolling_loss / max(rolling_count, 1)
            average_lm_loss = rolling_lm_loss / max(rolling_count, 1)
            average_regression_level_loss = rolling_regression_level_loss / max(rolling_count, 1)
            state = serialized_sft.update_state(
                run_dir,
                state,
                status="running",
                global_step=global_step,
                epoch=next_epoch,
                next_batch_index=next_batch_index,
                last_loss=average_loss,
                last_lm_loss=average_lm_loss,
                last_regression_level_loss=average_regression_level_loss,
                last_ordinal_level_loss=average_regression_level_loss,
                oom_skipped_batches=oom_skip_count,
            )
            valid_mask = (batch_on_device["level_label_positions"] >= 0) & (batch_on_device["level_labels"] >= 0)
            regression_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "updated_at": utc_now_iso(),
                **regression_history_snapshot(outputs=outputs, valid_mask=valid_mask),
            }
            append_jsonl(run_dir / "regression_history.jsonl", [regression_row])
            gradient_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "updated_at": utc_now_iso(),
                **gradient_diagnostics,
            }
            if gradient_diagnostics:
                append_jsonl(run_dir / "gradient_diagnostics.jsonl", [gradient_row])
            learning_rates = optimizer_learning_rates_by_group(optimizer, scheduler)
            log_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "loss": average_loss,
                "lm_loss": average_lm_loss,
                "regression_level_loss": average_regression_level_loss,
                "ordinal_level_loss": average_regression_level_loss,
                "lambda_level": args.lambda_level,
                "learning_rate": learning_rates["base_lora"],
                "regression_head_learning_rate": learning_rates["regression_head"],
                "updated_at": utc_now_iso(),
            }
            for key in (
                "grad_rms_regression_head",
                "update_rms_regression_head",
                "effective_level_score_mean_abs_shift",
            ):
                if key in gradient_diagnostics:
                    log_row[key] = gradient_diagnostics[key]
            append_jsonl(run_dir / "train_log.jsonl", [log_row])
            train_payload = {
                "train/loss": average_loss,
                "train/lm_loss": average_lm_loss,
                "train/regression_level_loss": average_regression_level_loss,
                "train/ordinal_level_loss": average_regression_level_loss,
                "train/lambda_level": args.lambda_level,
                "train/learning_rate": log_row["learning_rate"],
                "train/regression_head/learning_rate": log_row["regression_head_learning_rate"],
                "train/epoch": epoch,
                "train/batch_index": batch_index,
            }
            train_payload.update(regression_history_wandb_payload(regression_row))
            train_payload.update(gradient_diagnostics_wandb_payload(gradient_row))
            serialized_sft.wandb_log(wandb_run, train_payload, step=global_step)
            rolling_loss = 0.0
            rolling_lm_loss = 0.0
            rolling_regression_level_loss = 0.0
            rolling_count = 0

            saved_checkpoint: Path | None = None
            if global_step % args.checkpoint_steps == 0 or next_epoch >= args.epochs:
                saved_checkpoint = save_checkpoint(
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                )
                removed_checkpoints = serialized_sft.prune_old_checkpoints(
                    run_dir,
                    keep_last=args.checkpoint_keep_last,
                )
                if removed_checkpoints:
                    append_jsonl(
                        run_dir / "checkpoint_retention_log.jsonl",
                        [
                            {
                                "run_id": args.run_id,
                                "global_step": global_step,
                                "saved_checkpoint": str(saved_checkpoint),
                                "removed_checkpoints": removed_checkpoints,
                                "keep_last": args.checkpoint_keep_last,
                                "updated_at": utc_now_iso(),
                            }
                        ],
                    )

            if args.eval_checkpoint_steps and global_step % args.eval_checkpoint_steps == 0:
                checkpoint_name = (
                    saved_checkpoint.name
                    if saved_checkpoint is not None and saved_checkpoint.name.endswith(str(global_step))
                    else f"in-memory-step-{global_step}"
                )
                ordinal_sft.run_intermediate_validation(
                    args=args,
                    run_dir=run_dir,
                    validation_rows=validation_rows,
                    tokenizer=tokenizer,
                    model=model,
                    wandb_run=wandb_run,
                    global_step=global_step,
                    checkpoint_name=checkpoint_name,
                )

    state = serialized_sft.update_state(run_dir, state, status="trained")
    final_checkpoint = save_checkpoint(
        run_dir=run_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
    )
    removed_checkpoints = serialized_sft.prune_old_checkpoints(
        run_dir,
        keep_last=args.checkpoint_keep_last,
    )
    if removed_checkpoints:
        append_jsonl(
            run_dir / "checkpoint_retention_log.jsonl",
            [
                {
                    "run_id": args.run_id,
                    "global_step": int(state.get("global_step", 0)),
                    "saved_checkpoint": str(final_checkpoint),
                    "removed_checkpoints": removed_checkpoints,
                    "keep_last": args.checkpoint_keep_last,
                    "updated_at": utc_now_iso(),
                }
            ],
        )

    if args.skip_final_eval:
        metrics = {
            "run_id": args.run_id,
            "model_variant": args.model_variant,
            "serialization": args.serialization,
            "status": "trained_without_final_eval",
            "final_checkpoint": str(final_checkpoint),
        }
    else:
        metrics = ordinal_sft.evaluate_validation(
            run_id=args.run_id,
            run_dir=run_dir,
            validation_rows=validation_rows,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            checkpoint_name=final_checkpoint.name,
            wandb_run=wandb_run,
            validation_log_every=args.validation_log_every,
        )
        metrics["selection_key"] = list(ordinal_sft.structural_selection_key(metrics))
        metrics["best_checkpoint"] = str(final_checkpoint)
        serialized_sft.wandb_log(
            wandb_run,
            ordinal_sft.curated_validation_payload("validation", metrics),
            step=int(state.get("global_step", 0)),
        )

    write_json(run_dir / "metrics.json", metrics)
    serialized_sft.update_state(
        run_dir,
        state,
        status="completed",
        completed_at=utc_now_iso(),
        best_checkpoint=metrics.get("best_checkpoint"),
        best_validation_score=metrics.get("selection_key"),
    )
    serialized_sft.wandb_log(
        wandb_run,
        ordinal_sft.curated_validation_payload("final", metrics),
        step=int(state.get("global_step", 0)),
    )
    if args.wandb_log_artifacts:
        wandb_log_artifacts(
            wandb_run,
            run_dir=run_dir,
            final_checkpoint=metrics.get("best_checkpoint") or metrics.get("final_checkpoint"),
        )
    serialized_sft.wandb_finish(wandb_run)
    return metrics


def main() -> None:
    loaded_env_keys = sorted(serialized_sft.load_env_file().keys())
    args = parse_args()
    run_dir = args.output_dir / args.run_id
    train_rows = ordinal_sft.load_sft_rows(args.train_file, max_samples=args.max_train_samples, split_name="train")
    validation_rows = ordinal_sft.load_sft_rows(
        args.validation_file,
        max_samples=args.max_validation_samples,
        split_name="validation",
    )
    density_weights = ordinal_sft.compute_level_density_weights(
        train_rows,
        level_class_count=args.level_class_count,
        kernel=args.density_kernel,
        radius=args.density_kernel_radius,
        max_weight=args.max_density_weight,
    )
    args.level_density_weights = density_weights["weights"]
    config = {
        "run_id": args.run_id,
        "model_variant": args.model_variant,
        "serialization": args.serialization,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "base_model_path": str(args.base_model_path),
        "row_count_train": len(train_rows),
        "row_count_validation": len(validation_rows),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "base_learning_rate": args.learning_rate,
        "regression_head_learning_rate_multiplier": args.regression_head_learning_rate_multiplier,
        "regression_head_learning_rate": args.learning_rate * args.regression_head_learning_rate_multiplier,
        "regression_huber_delta": args.regression_huber_delta,
        "regression_gradient_diagnostics": bool(getattr(args, "regression_gradient_diagnostics", True)),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_steps": args.checkpoint_steps,
        "checkpoint_keep_last": args.checkpoint_keep_last,
        "eval_checkpoint_steps": args.eval_checkpoint_steps,
        "eval_max_samples": args.eval_max_samples,
        "eval_sample_strategy": args.eval_sample_strategy,
        "validation_log_every": args.validation_log_every,
        "oom_recovery": args.oom_recovery,
        "max_oom_skips": args.max_oom_skips,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "seed": args.seed,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(ordinal_sft.split_csv(args.lora_target_modules)),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "level_regression_head": {
            "type": "scalar_level_regression",
            "class_count": args.level_class_count,
            "hidden_dims": list(ordinal_sft.split_int_csv(args.regression_head_hidden_dims)),
            "dropouts": list(ordinal_sft.split_float_csv(args.regression_head_dropouts)),
            "loss": "density_weighted_huber",
            "prediction_policy": "round_and_clamp_to_level_range",
            "lambda_level": args.lambda_level,
            "huber_delta": args.regression_huber_delta,
            "head_optimizer": {
                "learning_rate_multiplier": args.regression_head_learning_rate_multiplier,
                "learning_rate": args.learning_rate * args.regression_head_learning_rate_multiplier,
                "weight_decay": args.weight_decay,
            },
            "density_weights": density_weights,
            "lora_target": False,
        },
        "training_policy": {
            "shuffle_train": False,
            "label_policy": "prompt tokens masked with -100; content_blocks_v1 target tokens supervised; level labels supervised only at record_prefix_state positions",
            "checkpoint_policy": "save adapter, regression level head, optimizer, and scheduler state at checkpoint_steps and completion",
            "checkpoint_retention_policy": "checkpoint_keep_last=0 keeps all; otherwise older local checkpoints are deleted after a new checkpoint is saved",
            "oom_recovery_policy": "fail by default; oom_recovery=skip_batch logs CUDA OOM microbatches to oom_skipped_batches.jsonl and excludes them from optimizer updates",
            "regression_gradient_diagnostics_policy": "when enabled, log raw gradient RMS/norm, optimizer-update RMS/norm, and same-hidden score movement for the regression head",
        },
        "wandb_metric_policy": {
            "dashboard": "curated",
            "logs_regression_scores": True,
            "logs_regression_gradient_diagnostics": bool(getattr(args, "regression_gradient_diagnostics", True)),
            "logs_regression_head_learning_rate": True,
            "keeps_auxiliary_text_metrics_local": True,
        },
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name or args.run_id,
            "tags": list(ordinal_sft.split_csv(args.wandb_tags)),
            "log_artifacts": args.wandb_log_artifacts,
            "env_file_loaded_keys": loaded_env_keys,
            "api_key_available": bool(os.environ.get("WANDB_API_KEY")),
        },
        "dry_run": args.dry_run,
        "skip_final_eval": args.skip_final_eval,
        "model_checks": serialized_sft.inspect_model_path(args.base_model_path),
        "resume_signature": build_resume_signature(args),
    }
    state = serialized_sft.initialize_run(run_dir, config, dry_run=args.dry_run)
    if args.dry_run:
        write_json(
            run_dir / "metrics.json",
            {
                "run_id": args.run_id,
                "model_variant": args.model_variant,
                "serialization": args.serialization,
                "row_count_train": len(train_rows),
                "row_count_validation": len(validation_rows),
                "status": "dry_run",
            },
        )
        write_json(run_dir / "state.json", state)
        print({"output_dir": str(run_dir), "dry_run": True, **json.loads((run_dir / "metrics.json").read_text())})
        return
    if args.regression_huber_delta <= 0:
        raise ValueError("regression_huber_delta must be positive")
    metrics = train(args, train_rows, validation_rows, config=config)
    print({"output_dir": str(run_dir), **metrics})


if __name__ == "__main__":
    main()
