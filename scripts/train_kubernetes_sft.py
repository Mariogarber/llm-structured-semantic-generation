from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import append_jsonl, read_jsonl, write_json
from llm_structured_semantic_generation.evaluation import (
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)
from llm_structured_semantic_generation.resumable_run import RunCompatibilityError, utc_now_iso
from llm_structured_semantic_generation.sft_serialization import (
    BLOCKS_TSV_V1,
    deserialize_training_blocks,
)


SERIALIZED_SFT = "serialized_sft"
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
PROMPT_TARGET_SEPARATOR = "\n\n"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_INTERMEDIATE_EVAL_MAX_SAMPLES = 10
INTERMEDIATE_EVAL_MAX_SAMPLES_LIMIT = 10


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", maxsplit=1)
    key = key.strip()
    value = value.strip().strip("\"'")
    if not key:
        return None
    return key, value


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """Load simple KEY=VALUE entries without overriding existing environment."""

    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Kubernetes v1 Architecture A: serialized_sft with LoRA."
    )
    parser.add_argument(
        "--model-variant",
        choices=[SERIALIZED_SFT],
        default=SERIALIZED_SFT,
        help="Only Architecture A / serialized_sft is implemented here.",
    )
    parser.add_argument(
        "--serialization",
        choices=[BLOCKS_TSV_V1],
        default=BLOCKS_TSV_V1,
        help="SFT target serialization. Architecture A trains on blocks_tsv_v1.",
    )
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
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "sft_kubernetes_v1",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=8)
    parser.add_argument("--max-seq-length", type=positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--checkpoint-steps", type=positive_int, default=25)
    parser.add_argument(
        "--checkpoint-keep-last",
        type=non_negative_int,
        default=0,
        help="Keep only the last N checkpoints after each save. Use 0 to keep all checkpoints.",
    )
    parser.add_argument("--eval-checkpoint-steps", type=non_negative_int, default=0)
    parser.add_argument(
        "--eval-max-samples",
        type=positive_int,
        default=DEFAULT_INTERMEDIATE_EVAL_MAX_SAMPLES,
        help=(
            "Validation examples to generate for each intermediate eval. "
            f"Capped at {INTERMEDIATE_EVAL_MAX_SAMPLES_LIMIT} to keep training checks lightweight. "
            "Final eval always uses validation-file."
        ),
    )
    parser.add_argument(
        "--eval-sample-strategy",
        choices=["random", "first"],
        default="random",
        help=(
            "How to choose examples for intermediate validation. "
            "random is deterministic per run seed and global step; final eval always uses validation-file."
        ),
    )
    parser.add_argument(
        "--validation-log-every",
        type=non_negative_int,
        default=1,
        help="During final validation, log cumulative metrics every N completed predictions. Use 0 to disable streaming logs.",
    )
    parser.add_argument(
        "--oom-recovery",
        choices=["fail", "skip_batch"],
        default="fail",
        help="When set to skip_batch, CUDA OOM microbatches are logged, skipped, and training continues.",
    )
    parser.add_argument(
        "--max-oom-skips",
        type=non_negative_int,
        default=0,
        help="Maximum skipped microbatches allowed in oom-recovery=skip_batch. Use 0 for no explicit limit.",
    )
    parser.add_argument("--max-train-samples", type=positive_int, default=None)
    parser.add_argument("--max-validation-samples", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=positive_int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="Comma-separated LoRA module names.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
        help="Optional Weights & Biases logging mode. Defaults to disabled for reproducible local runs.",
    )
    parser.add_argument(
        "--wandb-project",
        default="llm-structured-semantic-generation",
        help="Weights & Biases project name used when wandb logging is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run display name. Defaults to --run-id.",
    )
    parser.add_argument(
        "--wandb-tags",
        default="",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Upload config.json, metrics.json, and the final adapter checkpoint as W&B artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files/config and write config only; do not load or train the model.",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Train and write checkpoints, but do not generate validation predictions.",
    )
    return parser.parse_args()


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['prompt_variant']}"


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


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
        "lora_target_modules": list(split_csv(args.lora_target_modules)),
        "target_contract": "prompt -> blocks_tsv_v1 -> parser -> YAML",
    }


def inspect_model_path(model_path: Path) -> dict[str, Any]:
    files = {path.name for path in model_path.glob("*") if path.is_file()} if model_path.exists() else set()
    tokenizer_files = {
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    }
    checks: dict[str, Any] = {
        "model_path_exists": model_path.exists(),
        "has_config": "config.json" in files,
        "has_generation_config": "generation_config.json" in files,
        "has_weights": any(name.endswith((".safetensors", ".bin")) for name in files),
        "has_tokenizer_files": bool(files & tokenizer_files),
        "installed_transformers": find_spec("transformers") is not None,
        "installed_torch": find_spec("torch") is not None,
        "installed_bitsandbytes": find_spec("bitsandbytes") is not None,
        "installed_peft": find_spec("peft") is not None,
        "installed_accelerate": find_spec("accelerate") is not None,
        "installed_wandb": find_spec("wandb") is not None,
        "quant_method": None,
        "warnings": [],
    }
    if checks["has_config"]:
        try:
            config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
            checks["quant_method"] = config.get("quantization_config", {}).get("quant_method")
        except json.JSONDecodeError:
            checks["warnings"].append("config_json_not_parseable")
    if not checks["has_tokenizer_files"]:
        checks["warnings"].append("missing_local_tokenizer_files")
    if checks["quant_method"] == "bitsandbytes" and not checks["installed_bitsandbytes"]:
        checks["warnings"].append("model_quantization_requires_bitsandbytes")
    if not checks["installed_wandb"]:
        checks["warnings"].append("wandb_not_installed_optional")

    required = [
        "model_path_exists",
        "has_config",
        "has_weights",
        "has_tokenizer_files",
        "installed_transformers",
        "installed_torch",
        "installed_peft",
        "installed_accelerate",
    ]
    checks["ready_for_full_run"] = all(bool(checks[item]) for item in required) and not (
        checks["quant_method"] == "bitsandbytes" and not checks["installed_bitsandbytes"]
    )
    return checks


def validate_sft_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    required = {
        "sample_id",
        "prompt_variant",
        "split",
        "prompt",
        "target",
        "target_yaml_normalized",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{split_name}_row_{index}_missing_fields:{missing}")
        try:
            deserialize_training_blocks(str(row["target"]))
        except ValueError as exc:
            raise ValueError(f"{split_name}_row_{index}_invalid_blocks_tsv_v1:{exc}") from exc


def load_sft_rows(path: Path, *, max_samples: int | None, split_name: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    validate_sft_rows(rows, split_name)
    return rows


def build_training_text(row: dict[str, Any]) -> tuple[str, str, str]:
    prompt = str(row["prompt"]).rstrip()
    target = str(row["target"]).strip()
    return prompt, target, f"{prompt}{PROMPT_TARGET_SEPARATOR}{target}"


def tokenize_sft_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    prompt, target, _ = build_training_text(row)
    prompt_ids = tokenizer(
        f"{prompt}{PROMPT_TARGET_SEPARATOR}",
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    target_ids = tokenizer(
        target + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]
    if not any(label != -100 for label in labels):
        raise ValueError(f"row_has_no_supervised_target_after_truncation:{build_unit_id(row)}")
    return {
        "unit_id": build_unit_id(row),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class SFTDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, *, max_seq_length: int) -> None:
        self.items = [
            tokenize_sft_row(row, tokenizer, max_seq_length=max_seq_length)
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class SFTCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        unit_ids = []
        for item in batch:
            pad_count = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_count)
            attention_mask.append(item["attention_mask"] + [0] * pad_count)
            labels.append(item["labels"] + [-100] * pad_count)
            unit_ids.append(item["unit_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "unit_ids": unit_ids,
        }


def initialize_run(run_dir: Path, config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    state_path = run_dir / "state.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("resume_signature") != config.get("resume_signature"):
            raise RunCompatibilityError(
                "sft_run_resume_signature_mismatch:"
                f"{run_dir}:existing={existing_config.get('resume_signature')}:"
                f"new={config.get('resume_signature')}"
            )
    write_json(config_path, config)
    if dry_run:
        return {
            "run_id": config["run_id"],
            "status": "dry_run",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "resume_signature": config["resume_signature"],
        }
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    state = {
        "run_id": config["run_id"],
        "status": "running",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "completed_at": None,
        "global_step": 0,
        "epoch": 0,
        "next_batch_index": 0,
        "best_checkpoint": None,
        "best_validation_score": None,
        "resume_signature": config["resume_signature"],
    }
    write_json(state_path, state)
    return state


def update_state(run_dir: Path, state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    state.update(updates)
    state["updated_at"] = utc_now_iso()
    write_json(run_dir / "state.json", state)
    return state


def checkpoint_dir(run_dir: Path, global_step: int) -> Path:
    return run_dir / "checkpoints" / f"checkpoint-step-{global_step}"


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_root = run_dir / "checkpoints"
    if not checkpoints_root.exists():
        return None
    candidates = []
    for path in checkpoints_root.glob("checkpoint-step-*"):
        match = re.fullmatch(r"checkpoint-step-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def sorted_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints_root = run_dir / "checkpoints"
    if not checkpoints_root.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints_root.glob("checkpoint-step-*"):
        match = re.fullmatch(r"checkpoint-step-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    return [path for _, path in sorted(candidates)]


def prune_old_checkpoints(run_dir: Path, *, keep_last: int) -> list[str]:
    if keep_last <= 0:
        return []
    checkpoints = sorted_checkpoints(run_dir)
    stale = checkpoints[:-keep_last]
    removed: list[str] = []
    for path in stale:
        last_error: str | None = None
        for attempt in range(3):
            try:
                shutil.rmtree(path)
                removed.append(str(path))
                last_error = None
                break
            except OSError as exc:
                last_error = f"{exc.__class__.__name__}:{exc}"
                time.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            append_jsonl(
                run_dir / "checkpoint_prune_errors.jsonl",
                [
                    {
                        "checkpoint": str(path),
                        "keep_last": keep_last,
                        "error": last_error,
                        "updated_at": utc_now_iso(),
                    }
                ],
            )
    return removed


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

    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
    if is_bitsandbytes_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(split_csv(args.lora_target_modules)),
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return tokenizer, model


def model_input_device(model: Any) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for module_name in ("model.embed_tokens", "transformer.wte", ""):
            device = device_map.get(module_name)
            if device not in (None, "disk", "cpu"):
                return str(device)
    return str(getattr(model, "device", "cpu"))


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key != "unit_ids"
    }


def tensor_count(value: Any) -> int | None:
    try:
        return int(value.sum().item())
    except AttributeError:
        return None


def oom_skipped_batches_path(run_dir: Path) -> Path:
    return run_dir / "oom_skipped_batches.jsonl"


def read_oom_skipped_unit_ids(run_dir: Path) -> set[str]:
    path = oom_skipped_batches_path(run_dir)
    if not path.exists():
        return set()
    skipped: set[str] = set()
    for row in read_jsonl(path, allow_truncated_last_line=True):
        skipped.update(str(unit_id) for unit_id in row.get("unit_ids", []))
    return skipped


def cuda_memory_snapshot(torch_module: Any) -> dict[str, int] | None:
    if not torch_module.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "reserved_bytes": int(torch_module.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    }


def log_oom_skipped_batch(
    *,
    run_dir: Path,
    run_id: str,
    epoch: int,
    batch_index: int,
    global_step: int,
    batch: dict[str, Any],
    error: BaseException,
    cuda_memory: dict[str, int] | None,
    previously_recorded: bool = False,
) -> dict[str, Any]:
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    supervised_token_count = tensor_count(labels.ne(-100)) if hasattr(labels, "ne") else None
    attention_token_count = tensor_count(attention_mask) if attention_mask is not None else None
    row = {
        "run_id": run_id,
        "epoch": epoch,
        "batch_index": batch_index,
        "global_step": global_step,
        "unit_ids": list(batch.get("unit_ids", [])),
        "input_shape": list(batch["input_ids"].shape) if hasattr(batch.get("input_ids"), "shape") else None,
        "attention_token_count": attention_token_count,
        "supervised_token_count": supervised_token_count,
        "error_type": error.__class__.__name__,
        "error": str(error),
        "cuda_memory": cuda_memory,
        "previously_recorded": previously_recorded,
        "updated_at": utc_now_iso(),
    }
    append_jsonl(oom_skipped_batches_path(run_dir), [row])
    return row


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

    path = checkpoint_dir(run_dir, int(state["global_step"]))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "tokenizer")
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
        else:
            adapter_state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
        set_peft_model_state_dict(model, adapter_state, adapter_name="default")
    payload = torch.load(checkpoint_path / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return model, payload["state"]


def build_optimizer_and_scheduler(args: argparse.Namespace, model: Any, total_optimizer_steps: int):
    import torch
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    return optimizer, scheduler


def init_wandb_run(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    if find_spec("wandb") is None:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. "
            "Install optional LLM dependencies with `uv sync --extra llm`, or run with --wandb-mode disabled."
        )

    import wandb

    tags = list(split_csv(args.wandb_tags))
    if args.model_variant not in tags:
        tags.append(args.model_variant)
    if args.serialization not in tags:
        tags.append(args.serialization)

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
    return run


def wandb_log(wandb_run: Any | None, payload: dict[str, Any], *, step: int | None = None) -> None:
    if wandb_run is None:
        return
    wandb_run.log(payload, step=step)


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
    except ImportError:  # pragma: no cover - guarded by init_wandb_run
        return

    artifact = wandb.Artifact(name=f"{wandb_run.id}-sft-artifacts", type="sft-run")
    for filename in (
        "config.json",
        "state.json",
        "metrics.json",
        "train_log.jsonl",
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


def wandb_finish(wandb_run: Any | None) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def normalize_structured_field_separators(text: str) -> str:
    normalized = re.sub(r"<tab>", "\t", text, flags=re.IGNORECASE)
    normalized = re.sub(r"<vt>", "\t", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("\v", "\t").replace("\f", "\t")
    return normalized


def extract_blocks_tsv_prediction(serialized: str) -> list[dict[str, Any]]:
    candidate = serialized.strip()
    fenced = re.search(r"```(?:text|tsv)?\s*(<blocks>.*?</blocks>)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        wrapped = re.search(r"<blocks>.*?</blocks>", candidate, flags=re.DOTALL)
        if wrapped:
            candidate = wrapped.group(0)
        else:
            opened = re.search(r"<blocks>.*", candidate, flags=re.DOTALL)
            if opened:
                candidate = opened.group(0)
    candidate = normalize_structured_field_separators(candidate)
    rows: list[str] = []
    for raw_line in candidate.splitlines():
        line = normalize_structured_field_separators(raw_line)
        stripped = line.strip()
        if not stripped or stripped in {"<blocks>", "</blocks>"}:
            continue
        parts = line.split("\t", maxsplit=3)
        if len(parts) != 4:
            if rows:
                break
            raise ValueError("not_enough_tsv_fields")
        document_index, line_index, level, line_text = parts
        try:
            int(document_index)
            int(line_index)
            int(level)
        except ValueError as exc:
            if rows:
                break
            raise ValueError(f"invalid_tsv_numeric_field:{exc}") from exc
        rows.append(f"{document_index}\t{line_index}\t{level}\t{line_text}")
    if not rows:
        raise ValueError("no_valid_blocks_found")
    return [
        block.to_dict()
        for block in deserialize_training_blocks("\n".join(["<blocks>", *rows, "</blocks>"]))
    ]


def generate_validation_completion(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    prompt_text = f"{prompt.rstrip()}{PROMPT_TARGET_SEPARATOR}"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_input_device(model))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_token_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return {
        "raw_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "generated_token_count": int(generated_token_ids.shape[-1]),
    }


def derive_validation_metrics(
    *,
    run_id: str,
    predictions: list[dict[str, Any]],
    checkpoint: str,
) -> dict[str, Any]:
    evaluated_results = [
        StructuralEvaluation(**row["evaluation"])
        for row in predictions
        if row.get("evaluation") is not None
    ]
    metrics = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "row_count": len(predictions),
        "evaluated_count": len(evaluated_results),
        "model_variant": SERIALIZED_SFT,
        "serialization": BLOCKS_TSV_V1,
        "structured_output_parse_success_rate": len(evaluated_results) / len(predictions) if predictions else 0.0,
    }
    metrics.update(summarize_evaluations(evaluated_results))
    return metrics


def enrich_validation_progress_metrics(
    metrics: dict[str, Any],
    *,
    predictions: list[dict[str, Any]],
    total_count: int,
) -> dict[str, Any]:
    generated_counts = [
        int(row["generated_token_count"])
        for row in predictions
        if isinstance(row.get("generated_token_count"), int)
    ]
    parser_error_sample_count = sum(1 for row in predictions if row.get("parser_errors"))
    evaluated_count = int(metrics.get("evaluated_count") or 0)
    row_count = int(metrics.get("row_count") or 0)
    yaml_parse_success_rate = metrics.get("yaml_parse_success_rate")
    yaml_parse_failure_count = None
    if isinstance(yaml_parse_success_rate, (int, float)):
        yaml_parse_failure_count = evaluated_count - int(round(float(yaml_parse_success_rate) * evaluated_count))

    enriched = dict(metrics)
    enriched.update(
        {
            "completed_count": row_count,
            "total_count": total_count,
            "remaining_count": max(total_count - row_count, 0),
            "completion_rate": safe_divide(float(row_count), float(total_count)),
            "structured_output_parse_failure_count": row_count - evaluated_count,
            "parser_error_sample_count": parser_error_sample_count,
            "yaml_parse_failure_count": yaml_parse_failure_count,
            "average_generated_token_count": (
                sum(generated_counts) / len(generated_counts) if generated_counts else 0.0
            ),
            "min_generated_token_count": min(generated_counts) if generated_counts else 0,
            "max_generated_token_count": max(generated_counts) if generated_counts else 0,
            "total_generated_token_count": sum(generated_counts),
        }
    )
    return enriched


def validation_example_metrics(prediction: dict[str, Any], *, completed_count: int, total_count: int) -> dict[str, Any]:
    evaluation = prediction.get("evaluation") or {}
    return {
        "completed_count": completed_count,
        "total_count": total_count,
        "completion_rate": safe_divide(float(completed_count), float(total_count)),
        "generated_token_count": prediction.get("generated_token_count", 0),
        "structured_output_parse_ok": 1.0 if prediction.get("evaluation") is not None else 0.0,
        "parser_error_count": len(prediction.get("parser_errors") or []),
        "yaml_parse_ok": 1.0 if evaluation.get("yaml_parse_ok") else 0.0,
        "parsed_equal_to_reference": 1.0 if evaluation.get("parsed_equal_to_reference") else 0.0,
        "line_text_f1": evaluation.get("line_text_f1"),
        "line_text_precision": evaluation.get("line_text_precision"),
        "line_text_recall": evaluation.get("line_text_recall"),
        "level_exact_match_rate": evaluation.get("level_exact_match_rate"),
        "level_mae": evaluation.get("level_mae"),
        "content_exact_match_rate": evaluation.get("content_exact_match_rate"),
        "valid_block_ratio": evaluation.get("valid_block_ratio"),
        "indentation_leak_rate": evaluation.get("indentation_leak_rate"),
        "document_count_match": (
            1.0 if evaluation.get("document_count_match") else 0.0
            if "document_count_match" in evaluation
            else None
        ),
        "line_count_match": (
            1.0 if evaluation.get("line_count_match") else 0.0
            if "line_count_match" in evaluation
            else None
        ),
        "kind_sequence_match_rate": evaluation.get("kind_sequence_match_rate"),
        "primary_kind_match": (
            1.0 if evaluation.get("primary_kind_match") else 0.0
            if evaluation.get("primary_kind_match") is not None
            else None
        ),
        "primary_api_version_match": (
            1.0 if evaluation.get("primary_api_version_match") else 0.0
            if evaluation.get("primary_api_version_match") is not None
            else None
        ),
        "primary_metadata_name_match": (
            1.0 if evaluation.get("primary_metadata_name_match") else 0.0
            if evaluation.get("primary_metadata_name_match") is not None
            else None
        ),
        "semantic_key_f1": evaluation.get("semantic_key_f1"),
        "semantic_key_precision": evaluation.get("semantic_key_precision"),
        "semantic_key_recall": evaluation.get("semantic_key_recall"),
        "prompt_requirement_f1": evaluation.get("prompt_requirement_f1"),
        "prompt_requirement_precision": evaluation.get("prompt_requirement_precision"),
        "prompt_requirement_recall": evaluation.get("prompt_requirement_recall"),
        "prompt_requirement_exact_match": (
            1.0 if evaluation.get("prompt_requirement_exact_match") else 0.0
            if evaluation.get("prompt_requirement_exact_match") is not None
            else None
        ),
        "required_field_presence_rate": evaluation.get("required_field_presence_rate"),
        "required_field_complete_resource_rate": evaluation.get("required_field_complete_resource_rate"),
        "required_field_complete_sample": (
            1.0 if evaluation.get("required_field_complete_sample") else 0.0
            if evaluation.get("required_field_complete_sample") is not None
            else None
        ),
        "workload_selector_template_consistency": evaluation.get("workload_selector_template_consistency"),
        "service_selector_match_rate": evaluation.get("service_selector_match_rate"),
        "volume_mount_consistency": evaluation.get("volume_mount_consistency"),
    }


def structural_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("yaml_parse_success_rate") or 0.0),
        float(metrics.get("average_prompt_requirement_f1") or 0.0),
        float(metrics.get("average_required_field_complete_resource_rate") or 0.0),
        float(metrics.get("average_line_text_f1") or 0.0),
    )


def evaluate_validation(
    *,
    run_id: str,
    run_dir: Path,
    validation_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    checkpoint_name: str,
    predictions_path: Path | None = None,
    resume_scope: str = "unit",
    wandb_run: Any | None = None,
    validation_log_every: int = 0,
) -> dict[str, Any]:
    if resume_scope not in {"unit", "checkpoint"}:
        raise ValueError(f"unsupported_validation_resume_scope:{resume_scope}")
    predictions_path = predictions_path or run_dir / "validation_predictions.jsonl"
    if predictions_path.exists():
        existing = read_jsonl(predictions_path, allow_truncated_last_line=True)
    else:
        existing = []
    if resume_scope == "checkpoint":
        completed = {(row.get("checkpoint"), row["unit_id"]) for row in existing}
        metric_predictions = [row for row in existing if row.get("checkpoint") == checkpoint_name]
    else:
        completed = {row["unit_id"] for row in existing}
        metric_predictions = list(existing)

    for row in validation_rows:
        unit_id = build_unit_id(row)
        completed_key = (checkpoint_name, unit_id) if resume_scope == "checkpoint" else unit_id
        if completed_key in completed:
            continue
        completion = generate_validation_completion(
            tokenizer=tokenizer,
            model=model,
            prompt=str(row["prompt"]),
            max_new_tokens=max_new_tokens,
        )
        parse_errors: list[str] = []
        try:
            predicted_blocks = extract_blocks_tsv_prediction(completion["raw_text"])
        except ValueError as exc:
            predicted_blocks = []
            parse_errors.append(f"structured_output_parse_error:blocks_tsv_v1:{exc.__class__.__name__}:{exc}")
        evaluation = (
            evaluate_blocks_prediction(
                str(row["target_yaml_normalized"]),
                predicted_blocks,
                prompt_text=str(row["prompt"]),
            )
            if predicted_blocks
            else None
        )
        prediction = {
            "unit_id": unit_id,
            "sample_id": row["sample_id"],
            "prompt_variant": row["prompt_variant"],
            "split": row["split"],
            "checkpoint": checkpoint_name,
            "output_format": BLOCKS_TSV_V1,
            "raw_model_output": completion["raw_text"],
            "generated_token_count": completion["generated_token_count"],
            "predicted_blocks": predicted_blocks,
            "parser_errors": parse_errors,
            "evaluation": evaluation.to_dict() if evaluation else None,
        }
        append_jsonl(predictions_path, [prediction])
        existing.append(prediction)
        metric_predictions.append(prediction)
        completed.add(completed_key)
        completed_count = len(metric_predictions)
        if validation_log_every > 0:
            log_validation_example(
                run_dir=run_dir,
                wandb_run=wandb_run,
                prediction=prediction,
                completed_count=completed_count,
                total_count=len(validation_rows),
            )
        if validation_log_every > 0 and (
            completed_count % validation_log_every == 0
            or completed_count == len(validation_rows)
        ):
            log_validation_progress(
                run_dir=run_dir,
                wandb_run=wandb_run,
                run_id=run_id,
                predictions=metric_predictions,
                checkpoint_name=checkpoint_name,
                total_count=len(validation_rows),
            )

    return derive_validation_metrics(
        run_id=run_id,
        predictions=metric_predictions,
        checkpoint=checkpoint_name,
    )


def wandb_log_numeric_metrics(
    wandb_run: Any | None,
    metrics: dict[str, Any],
    *,
    prefix: str,
    step: int,
) -> None:
    wandb_log(
        wandb_run,
        {
            f"{prefix}/{key}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        step=step,
    )


def numeric_payload(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def log_validation_progress(
    *,
    run_dir: Path,
    wandb_run: Any | None,
    run_id: str,
    predictions: list[dict[str, Any]],
    checkpoint_name: str,
    total_count: int,
) -> dict[str, Any]:
    metrics = derive_validation_metrics(
        run_id=run_id,
        predictions=predictions,
        checkpoint=checkpoint_name,
    )
    metrics = enrich_validation_progress_metrics(
        metrics,
        predictions=predictions,
        total_count=total_count,
    )
    metrics["updated_at"] = utc_now_iso()
    append_jsonl(run_dir / "validation_metrics_progress.jsonl", [metrics])
    wandb_log(wandb_run, numeric_payload("validation_progress", metrics))
    return metrics


def log_validation_example(
    *,
    run_dir: Path,
    wandb_run: Any | None,
    prediction: dict[str, Any],
    completed_count: int,
    total_count: int,
) -> dict[str, Any]:
    metrics = validation_example_metrics(
        prediction,
        completed_count=completed_count,
        total_count=total_count,
    )
    local_row = {
        "unit_id": prediction["unit_id"],
        "sample_id": prediction["sample_id"],
        "prompt_variant": prediction["prompt_variant"],
        "checkpoint": prediction["checkpoint"],
        "updated_at": utc_now_iso(),
        **metrics,
    }
    append_jsonl(run_dir / "validation_example_metrics.jsonl", [local_row])
    wandb_log(
        wandb_run,
        {
            key: value
            for key, value in numeric_payload("validation_example", metrics).items()
            if value is not None
        },
    )
    return local_row


def intermediate_eval_seed(*, base_seed: int, global_step: int) -> int:
    return int(base_seed) + int(global_step) * 1_000_003


def select_intermediate_validation_rows(
    validation_rows: list[dict[str, Any]],
    *,
    max_samples: int,
    sample_strategy: str,
    seed: int,
    global_step: int,
) -> list[dict[str, Any]]:
    if max_samples >= len(validation_rows):
        return list(validation_rows)
    if sample_strategy == "first":
        return validation_rows[:max_samples]
    if sample_strategy != "random":
        raise ValueError(f"unsupported_intermediate_eval_sample_strategy:{sample_strategy}")

    rng = random.Random(intermediate_eval_seed(base_seed=seed, global_step=global_step))
    selected_indices = sorted(rng.sample(range(len(validation_rows)), max_samples))
    return [validation_rows[index] for index in selected_indices]


def run_intermediate_validation(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    validation_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    wandb_run: Any | None,
    global_step: int,
    checkpoint_name: str,
) -> dict[str, Any]:
    eval_requested_max_samples = int(args.eval_max_samples)
    eval_effective_max_samples = min(
        eval_requested_max_samples,
        INTERMEDIATE_EVAL_MAX_SAMPLES_LIMIT,
    )
    eval_sample_strategy = getattr(args, "eval_sample_strategy", "random")
    eval_sample_seed = intermediate_eval_seed(
        base_seed=int(args.seed),
        global_step=global_step,
    )
    eval_rows = select_intermediate_validation_rows(
        validation_rows,
        max_samples=eval_effective_max_samples,
        sample_strategy=eval_sample_strategy,
        seed=int(args.seed),
        global_step=global_step,
    )
    metrics = evaluate_validation(
        run_id=args.run_id,
        run_dir=run_dir,
        validation_rows=eval_rows,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        checkpoint_name=checkpoint_name,
        predictions_path=run_dir / "intermediate_validation_predictions.jsonl",
        resume_scope="checkpoint",
    )
    metrics["global_step"] = global_step
    metrics["eval_requested_max_samples"] = eval_requested_max_samples
    metrics["eval_max_samples_limit"] = INTERMEDIATE_EVAL_MAX_SAMPLES_LIMIT
    metrics["eval_max_samples"] = len(eval_rows)
    metrics["eval_sample_strategy"] = eval_sample_strategy
    metrics["eval_sample_seed"] = eval_sample_seed
    metrics["eval_sample_unit_ids"] = [build_unit_id(row) for row in eval_rows]
    metrics["selection_key"] = list(structural_selection_key(metrics))
    append_jsonl(run_dir / "intermediate_validation_metrics.jsonl", [metrics])
    wandb_log_numeric_metrics(wandb_run, metrics, prefix="validation_sample", step=global_step)
    model.train()
    return metrics


def train(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    run_dir = args.output_dir / args.run_id
    wandb_run = init_wandb_run(args=args, config=config, run_dir=run_dir)
    tokenizer, model = load_model_and_tokenizer(args)
    train_dataset = SFTDataset(train_rows, tokenizer, max_seq_length=args.max_seq_length)
    collator = SFTCollator(tokenizer)
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
    checkpoint_path = latest_checkpoint(run_dir)
    if checkpoint_path is not None and (checkpoint_path / "training_state.pt").exists():
        model, checkpoint_state = load_checkpoint_state(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        state.update(checkpoint_state)
        update_state(run_dir, state)

    device = model_input_device(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_count = 0
    oom_skipped_unit_ids = read_oom_skipped_unit_ids(run_dir) if args.oom_recovery == "skip_batch" else set()
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
                state = update_state(
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
                batch_on_device = move_batch_to_device(batch, device)
                outputs = model(**batch_on_device)
                loss = outputs.loss
                (loss / args.gradient_accumulation_steps).backward()
            except torch.OutOfMemoryError as exc:
                if args.oom_recovery != "skip_batch":
                    raise
                cuda_memory = cuda_memory_snapshot(torch)
                optimizer.zero_grad(set_to_none=True)
                rolling_loss = 0.0
                rolling_count = 0
                del batch_on_device, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                oom_row = log_oom_skipped_batch(
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
                state = update_state(
                    run_dir,
                    state,
                    status="running",
                    epoch=next_epoch,
                    next_batch_index=next_batch_index,
                    oom_skipped_batches=oom_skip_count,
                    last_oom_skipped_batch=oom_row,
                )
                wandb_log(
                    wandb_run,
                    {
                        "train/oom_skipped_batches": oom_skip_count,
                        "train/oom_skipped_samples": len(oom_skipped_unit_ids),
                    },
                    step=int(state.get("global_step", 0)),
                )
                if args.max_oom_skips and oom_skip_count > args.max_oom_skips:
                    raise RuntimeError(
                        f"max_oom_skips_exceeded:{oom_skip_count}>{args.max_oom_skips}"
                    ) from exc
                continue
            rolling_loss += float(loss.detach().cpu())
            rolling_count += 1

            should_step = (
                rolling_count >= args.gradient_accumulation_steps
                or (batch_index + 1 == len(train_loader) and rolling_count > 0)
            )
            if not should_step:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step = int(state.get("global_step", 0)) + 1
            average_loss = rolling_loss / max(rolling_count, 1)
            state = update_state(
                run_dir,
                state,
                status="running",
                global_step=global_step,
                epoch=next_epoch,
                next_batch_index=next_batch_index,
                last_loss=average_loss,
                oom_skipped_batches=oom_skip_count,
            )
            log_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "loss": average_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "updated_at": utc_now_iso(),
            }
            append_jsonl(
                run_dir / "train_log.jsonl",
                [log_row],
            )
            wandb_log(
                wandb_run,
                {
                    "train/loss": average_loss,
                    "train/learning_rate": log_row["learning_rate"],
                    "train/epoch": epoch,
                    "train/batch_index": batch_index,
                },
                step=global_step,
            )
            rolling_loss = 0.0
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
                removed_checkpoints = prune_old_checkpoints(
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
                run_intermediate_validation(
                    args=args,
                    run_dir=run_dir,
                    validation_rows=validation_rows,
                    tokenizer=tokenizer,
                    model=model,
                    wandb_run=wandb_run,
                    global_step=global_step,
                    checkpoint_name=checkpoint_name,
                )

    state = update_state(run_dir, state, status="trained")
    final_checkpoint = save_checkpoint(
        run_dir=run_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
    )
    removed_checkpoints = prune_old_checkpoints(
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
        metrics = evaluate_validation(
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
        metrics["selection_key"] = list(structural_selection_key(metrics))
        metrics["best_checkpoint"] = str(final_checkpoint)
        wandb_log_numeric_metrics(
            wandb_run,
            metrics,
            prefix="validation",
            step=int(state.get("global_step", 0)),
        )

    write_json(run_dir / "metrics.json", metrics)
    update_state(
        run_dir,
        state,
        status="completed",
        completed_at=utc_now_iso(),
        best_checkpoint=metrics.get("best_checkpoint"),
        best_validation_score=metrics.get("selection_key"),
    )
    wandb_log_numeric_metrics(
        wandb_run,
        metrics,
        prefix="final",
        step=int(state.get("global_step", 0)),
    )
    if args.wandb_log_artifacts:
        wandb_log_artifacts(
            wandb_run,
            run_dir=run_dir,
            final_checkpoint=metrics.get("best_checkpoint") or metrics.get("final_checkpoint"),
        )
    wandb_finish(wandb_run)
    return metrics


def main() -> None:
    loaded_env_keys = sorted(load_env_file().keys())
    args = parse_args()
    run_dir = args.output_dir / args.run_id
    train_rows = load_sft_rows(args.train_file, max_samples=args.max_train_samples, split_name="train")
    validation_rows = load_sft_rows(
        args.validation_file,
        max_samples=args.max_validation_samples,
        split_name="validation",
    )
    config = {
        "run_id": args.run_id,
        "model_variant": args.model_variant,
        "serialization": args.serialization,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "base_model_path": str(args.base_model_path),
        "row_count_train": len(train_rows),
        "row_count_validation": len(validation_rows),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
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
            "target_modules": list(split_csv(args.lora_target_modules)),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training_policy": {
            "shuffle_train": False,
            "label_policy": "prompt tokens masked with -100; blocks_tsv_v1 target tokens supervised",
            "checkpoint_policy": "save adapter plus optimizer and scheduler state at checkpoint_steps and completion",
            "checkpoint_retention_policy": "checkpoint_keep_last=0 keeps all; otherwise older local checkpoints are deleted after a new checkpoint is saved",
            "oom_recovery_policy": "fail by default; oom_recovery=skip_batch logs CUDA OOM microbatches to oom_skipped_batches.jsonl and excludes them from optimizer updates",
        },
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name or args.run_id,
            "tags": list(split_csv(args.wandb_tags)),
            "log_artifacts": args.wandb_log_artifacts,
            "env_file_loaded_keys": loaded_env_keys,
            "api_key_available": bool(os.environ.get("WANDB_API_KEY")),
        },
        "dry_run": args.dry_run,
        "skip_final_eval": args.skip_final_eval,
        "model_checks": inspect_model_path(args.base_model_path),
        "resume_signature": build_resume_signature(args),
    }
    state = initialize_run(run_dir, config, dry_run=args.dry_run)
    if args.dry_run:
        write_json(
            run_dir / "metrics.json",
            {
                "dry_run": True,
                "row_count_train": len(train_rows),
                "row_count_validation": len(validation_rows),
                "ready_for_full_run": config["model_checks"]["ready_for_full_run"],
                "model_warnings": config["model_checks"]["warnings"],
            },
        )
        write_json(run_dir / "state.json", state)
        print({"output_dir": str(run_dir), "dry_run": True, **json.loads((run_dir / "metrics.json").read_text())})
        return

    if not config["model_checks"]["ready_for_full_run"]:
        raise RuntimeError(f"SFT full run is not ready. See model_checks: {config['model_checks']}")
    metrics = train(args, train_rows, validation_rows, config=config)
    print({"output_dir": str(run_dir), **metrics})


if __name__ == "__main__":
    main()
