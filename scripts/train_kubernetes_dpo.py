from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from llm_structured_semantic_generation.dataset_io import append_jsonl, read_jsonl, write_json
from llm_structured_semantic_generation.resumable_run import RunCompatibilityError, utc_now_iso
from llm_structured_semantic_generation.sft_serialization import BLOCKS_TSV_V1

from train_kubernetes_sft import (
    DEFAULT_ENV_FILE,
    DEFAULT_LORA_TARGET_MODULES,
    PROMPT_TARGET_SEPARATOR,
    build_unit_id,
    cuda_memory_snapshot,
    evaluate_validation,
    initialize_run,
    latest_checkpoint,
    load_checkpoint_state,
    load_env_file,
    load_sft_rows,
    model_input_device,
    numeric_payload,
    prune_old_checkpoints,
    save_checkpoint,
    split_csv,
    structural_selection_key,
    update_state,
    wandb_finish,
    wandb_log,
    wandb_log_numeric_metrics,
)


SERIALIZED_SFT_DPO = "serialized_sft_dpo"
SERIALIZED_SFT = "serialized_sft"
DEFAULT_PREFERENCE_FILE = (
    REPO_ROOT
    / "results"
    / "dpo_kubernetes_v1"
    / "preference_annotation"
    / "agent-full-auto-v1"
    / "preferences_final.jsonl"
)
DEFAULT_SFT_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "sft_kubernetes_v1"
    / "serialized-sft-a-v1-20260505-171226"
    / "checkpoints"
    / "checkpoint-step-159"
)
DEFAULT_BASE_MODEL = REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dpo_kubernetes_v1" / "training"
DEFAULT_VALIDATION_FILE = REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "validation.jsonl"
DEFAULT_TWO_THIRDS_VALIDATION_SAMPLES = 10
FORBIDDEN_OUTPUT_ROOTS = (
    REPO_ROOT / "model",
    REPO_ROOT / "results" / "sft_kubernetes_v1" / "serialized-sft-a-v1-20260505-171226",
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Kubernetes v1 serialized_sft_dpo with offline Direct Preference Optimization "
            "from automatic preference pairs."
        )
    )
    parser.add_argument("--preference-file", type=Path, default=DEFAULT_PREFERENCE_FILE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--sft-adapter-path",
        type=Path,
        default=DEFAULT_SFT_CHECKPOINT,
        help="SFT checkpoint root or adapter directory used to initialize policy and frozen reference.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=8)
    parser.add_argument("--max-seq-length", type=positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--checkpoint-steps", type=positive_int, default=25)
    parser.add_argument(
        "--checkpoint-keep-last",
        type=non_negative_int,
        default=0,
        help="Keep only the last N DPO checkpoints. Use 0 to keep all.",
    )
    parser.add_argument("--max-train-samples", type=positive_int, default=None)
    parser.add_argument("--max-validation-samples", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing for the trainable DPO policy.",
    )
    parser.add_argument(
        "--oom-recovery",
        choices=["fail", "skip_batch"],
        default="fail",
        help="When set to skip_batch, CUDA OOM preference batches are logged, skipped, and training continues.",
    )
    parser.add_argument("--max-oom-skips", type=non_negative_int, default=0)
    parser.add_argument(
        "--reference-logps-file",
        type=Path,
        default=None,
        help="Optional persisted reference log-prob JSONL. Defaults to <run-dir>/reference_logps.jsonl.",
    )
    parser.add_argument(
        "--force-reference-logps-recompute",
        action="store_true",
        help="Ignore existing reference_logps.jsonl and recompute fixed pi_ref log-probs.",
    )
    parser.add_argument(
        "--validation-log-every",
        type=non_negative_int,
        default=1,
        help="During final validation, log cumulative metrics every N completed predictions. Use 0 to disable.",
    )
    parser.add_argument(
        "--two-thirds-validation-samples",
        type=non_negative_int,
        default=0,
        help=(
            "Run one lightweight validation after reaching two thirds of optimizer steps. "
            f"Use {DEFAULT_TWO_THIRDS_VALIDATION_SAMPLES} for the canonical beta=0.10 run; 0 disables it."
        ),
    )
    parser.add_argument(
        "--two-thirds-validation-sample-strategy",
        choices=["random", "first"],
        default="random",
        help="How to select examples for the two-thirds validation sample.",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Train and write checkpoints, but do not generate validation predictions.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
        help="Weights & Biases logging mode. Use online for real DPO runs.",
    )
    parser.add_argument("--wandb-project", default="llm-structured-semantic-generation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files/config and write config only; do not load or train the model.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_output_dir_safe(run_dir: Path, *, base_model_path: Path, sft_checkpoint_root: Path) -> None:
    resolved = run_dir.resolve()
    forbidden_roots = [root.resolve() for root in FORBIDDEN_OUTPUT_ROOTS]
    forbidden_roots.extend([base_model_path.resolve(), sft_checkpoint_root.resolve()])
    for root in forbidden_roots:
        if resolved == root or is_relative_to(resolved, root):
            raise ValueError(f"unsafe_dpo_output_dir_inside_source_artifact:{resolved}:forbidden_root={root}")


def resolve_sft_adapter_and_checkpoint_root(path: Path) -> tuple[Path, Path]:
    resolved = resolve_project_path(path).resolve()
    if (resolved / "adapter" / "adapter_config.json").exists():
        return resolved / "adapter", resolved
    if (resolved / "adapter_config.json").exists():
        checkpoint_root = resolved.parent
        if checkpoint_root.name == "adapter":
            checkpoint_root = checkpoint_root.parent
        return resolved, checkpoint_root
    raise FileNotFoundError(f"missing_sft_adapter_config:{resolved}")


def resolve_tokenizer_path(*, checkpoint_root: Path, base_model_path: Path) -> Path:
    checkpoint_tokenizer = checkpoint_root / "tokenizer"
    if (checkpoint_tokenizer / "tokenizer.json").exists() or (checkpoint_tokenizer / "tokenizer_config.json").exists():
        return checkpoint_tokenizer
    return base_model_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_artifact_fingerprint(base_model_path: Path, sft_adapter_path: Path, checkpoint_root: Path) -> dict[str, Any]:
    adapter_files = []
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        path = sft_adapter_path / name
        if path.exists():
            adapter_files.append(
                {
                    "path": project_path(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    base_config = base_model_path / "config.json"
    return {
        "base_model_path": project_path(base_model_path),
        "sft_checkpoint_root": project_path(checkpoint_root),
        "sft_adapter_path": project_path(sft_adapter_path),
        "read_only_policy": "DPO writes only under its run directory; source model and SFT checkpoint are never save targets.",
        "base_config": (
            {
                "path": project_path(base_config),
                "size_bytes": base_config.stat().st_size,
                "sha256": file_sha256(base_config),
            }
            if base_config.exists()
            else None
        ),
        "adapter_files": adapter_files,
    }


def collect_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "wandb"):
        if find_spec(module_name) is None:
            versions[module_name] = "not_installed"
            continue
        try:
            module = import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - defensive environment reporting
            versions[module_name] = f"import_error:{exc.__class__.__name__}:{exc}"
    return versions


def inspect_cuda_environment() -> dict[str, Any]:
    if find_spec("torch") is None:
        return {"torch_installed": False, "cuda_available": False}
    import torch

    payload: dict[str, Any] = {
        "torch_installed": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        payload.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            payload["free_bytes"] = int(free_bytes)
            payload["total_bytes"] = int(total_bytes)
        except RuntimeError:
            pass
    return payload


def inspect_base_model_path(model_path: Path) -> dict[str, Any]:
    files = {path.name for path in model_path.glob("*") if path.is_file()} if model_path.exists() else set()
    tokenizer_files = {"tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "tokenizer_config.json"}
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


def validate_preference_text(text: str, *, row_index: int, field_name: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"preference_row_{row_index}_empty_{field_name}")


def validate_preference_rows(rows: list[dict[str, Any]]) -> None:
    required = {"preference_id", "split", "prompt", "chosen", "rejected"}
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"preference_row_{index}_missing_fields:{missing}")
        preference_id = row.get("preference_id")
        if not isinstance(preference_id, str) or not preference_id:
            raise ValueError(f"preference_row_{index}_invalid_preference_id")
        if preference_id in seen_ids:
            raise ValueError(f"duplicate_preference_id:{preference_id}")
        seen_ids.add(preference_id)
        if row.get("split") != "train":
            raise ValueError(f"preference_row_{index}_not_train_split:{row.get('split')}")
        validate_preference_text(str(row["chosen"]), row_index=index, field_name="chosen")
        validate_preference_text(str(row["rejected"]), row_index=index, field_name="rejected")


def load_preference_rows(path: Path, *, max_samples: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    validate_preference_rows(rows)
    return rows


def build_resume_signature(args: argparse.Namespace, *, sft_adapter_path: Path, checkpoint_root: Path) -> dict[str, Any]:
    return {
        "model_variant": SERIALIZED_SFT_DPO,
        "source_model_variant": SERIALIZED_SFT,
        "serialization": BLOCKS_TSV_V1,
        "preference_file": project_path(resolve_project_path(args.preference_file)),
        "validation_file": project_path(resolve_project_path(args.validation_file)),
        "base_model_path": project_path(resolve_project_path(args.base_model_path)),
        "sft_adapter_path": project_path(sft_adapter_path),
        "sft_checkpoint_root": project_path(checkpoint_root),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "two_thirds_validation_samples": args.two_thirds_validation_samples,
        "two_thirds_validation_sample_strategy": args.two_thirds_validation_sample_strategy,
        "seed": args.seed,
        "target_contract": "prompt + chosen/rejected blocks_tsv_v1 -> offline DPO -> serialized_sft_dpo",
        "reference_policy": "frozen serialized_sft checkpoint; reference log-probs are precomputed and persisted",
    }


def build_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    preference_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    sft_adapter_path: Path,
    checkpoint_root: Path,
    tokenizer_path: Path,
    loaded_env_keys: Iterable[str],
) -> dict[str, Any]:
    base_model_path = resolve_project_path(args.base_model_path).resolve()
    return {
        "run_id": args.run_id,
        "stage": "dpo_training_v1",
        "model_variant": SERIALIZED_SFT_DPO,
        "source_model_variant": SERIALIZED_SFT,
        "serialization": BLOCKS_TSV_V1,
        "preference_file": project_path(resolve_project_path(args.preference_file)),
        "validation_file": project_path(resolve_project_path(args.validation_file)),
        "output_dir": project_path(run_dir),
        "row_count_preference_train": len(preference_rows),
        "row_count_validation": len(validation_rows),
        "base_model_path": project_path(base_model_path),
        "sft_adapter_path": project_path(sft_adapter_path),
        "sft_checkpoint_root": project_path(checkpoint_root),
        "tokenizer_path": project_path(tokenizer_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_steps": args.checkpoint_steps,
        "checkpoint_keep_last": args.checkpoint_keep_last,
        "two_thirds_validation_samples": args.two_thirds_validation_samples,
        "two_thirds_validation_sample_strategy": args.two_thirds_validation_sample_strategy,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "oom_recovery": args.oom_recovery,
        "max_oom_skips": args.max_oom_skips,
        "skip_final_eval": args.skip_final_eval,
        "training_policy": {
            "loss": "DPO logistic loss with fixed pi_ref sequence log-probs",
            "reference_logps": "precomputed before trainable policy loading to avoid two 7B models in GPU memory",
            "target_surface": "blocks_tsv_v1",
            "checkpoint_policy": "save DPO adapter plus optimizer/scheduler state only under this run directory",
        },
        "source_artifacts": source_artifact_fingerprint(base_model_path, sft_adapter_path, checkpoint_root),
        "runtime_versions": collect_runtime_versions(),
        "cuda": inspect_cuda_environment(),
        "model_checks": inspect_base_model_path(base_model_path),
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name or args.run_id,
            "tags": build_wandb_tags(args),
            "log_artifacts": args.wandb_log_artifacts,
            "env_file_loaded_keys": list(loaded_env_keys),
            "api_key_available": bool(os.environ.get("WANDB_API_KEY")),
        },
        "dry_run": args.dry_run,
        "resume_signature": build_resume_signature(
            args,
            sft_adapter_path=sft_adapter_path,
            checkpoint_root=checkpoint_root,
        ),
    }


def build_wandb_tags(args: argparse.Namespace) -> list[str]:
    tags = list(split_csv(args.wandb_tags))
    required = [
        "dpo",
        SERIALIZED_SFT_DPO,
        "kubernetes_v1",
        BLOCKS_TSV_V1,
        f"beta_{args.beta:g}",
    ]
    for tag in required:
        if tag not in tags:
            tags.append(tag)
    return tags


def init_wandb_run(*, args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    if find_spec("wandb") is None:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. "
            "Install optional LLM dependencies with `uv sync --extra llm`, or run with --wandb-mode disabled."
        )
    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb_online_requires_wandb_api_key: set WANDB_API_KEY in the environment or .env, "
            "or use --wandb-mode offline/disabled."
        )

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.run_id,
        id=args.run_id,
        resume="allow",
        mode=args.wandb_mode,
        dir=str(run_dir),
        tags=build_wandb_tags(args),
        config=config,
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("validation_progress/completed_count")
    wandb.define_metric("validation_progress/*", step_metric="validation_progress/completed_count")
    wandb.define_metric("validation_example/completed_count")
    wandb.define_metric("validation_example/*", step_metric="validation_example/completed_count")
    wandb.define_metric("validation_two_thirds/global_step")
    wandb.define_metric("validation_two_thirds/*", step_metric="validation_two_thirds/global_step")
    wandb.define_metric("validation/global_step")
    wandb.define_metric("validation/*", step_metric="validation/global_step")
    wandb.define_metric("final/global_step")
    wandb.define_metric("final/*", step_metric="final/global_step")
    return run


def wandb_log_numeric_metrics_on_metric_axis(
    wandb_run: Any | None,
    metrics: dict[str, Any],
    *,
    prefix: str,
) -> None:
    # Validation progress advances W&B's internal step, so final summaries use
    # explicit metric axes instead of the global W&B step argument.
    wandb_log(wandb_run, numeric_payload(prefix, metrics))


def validate_wandb_request(args: argparse.Namespace) -> None:
    if args.wandb_mode == "disabled":
        return
    if find_spec("wandb") is None:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. "
            "Install optional LLM dependencies with `uv sync --extra llm`, or run with --wandb-mode disabled."
        )
    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb_online_requires_wandb_api_key: set WANDB_API_KEY in the environment or .env, "
            "or use --wandb-mode offline/disabled."
        )


def wandb_log_dpo_artifacts(
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

    artifact = wandb.Artifact(name=f"{wandb_run.id}-dpo-artifacts", type="dpo-run")
    for filename in (
        "config.json",
        "state.json",
        "metrics.json",
        "train_log.jsonl",
        "reference_logps.jsonl",
        "reference_logps_summary.json",
        "validation_predictions.jsonl",
        "validation_metrics_progress.jsonl",
        "validation_example_metrics.jsonl",
        "two_thirds_validation_predictions.jsonl",
        "two_thirds_validation_metrics.jsonl",
        "oom_skipped_batches.jsonl",
        "checkpoint_retention_log.jsonl",
    ):
        path = run_dir / filename
        if path.exists():
            artifact.add_file(str(path))
    if final_checkpoint:
        checkpoint_path = Path(final_checkpoint)
        if checkpoint_path.exists():
            artifact.add_dir(str(checkpoint_path), name=checkpoint_path.name)
    wandb_run.log_artifact(artifact)


def build_optimizer_and_scheduler(args: argparse.Namespace, model: Any, total_optimizer_steps: int) -> tuple[Any, Any]:
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


def load_model_and_tokenizer(
    *,
    args: argparse.Namespace,
    sft_adapter_path: Path,
    tokenizer_path: Path,
    trainable: bool,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base_model_path = resolve_project_path(args.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(base_model_path, local_files_only=True)
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

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    if trainable and is_bitsandbytes_4bit:
        base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        local_files_only=True,
        is_trainable=trainable,
    )
    if trainable and args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.config.use_cache = False
    return tokenizer, model


def tokenize_completion_pair(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    prompt = str(row["prompt"]).rstrip()
    prompt_ids = tokenizer(
        f"{prompt}{PROMPT_TARGET_SEPARATOR}",
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    def encode_target(text: str, field_name: str) -> tuple[list[int], list[int], list[int]]:
        target_ids = tokenizer(
            text.strip() + (tokenizer.eos_token or ""),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
        if not any(label != -100 for label in labels):
            raise ValueError(f"preference_row_has_no_{field_name}_target_after_truncation:{row['preference_id']}")
        return input_ids, [1] * len(input_ids), labels

    chosen_input_ids, chosen_attention_mask, chosen_labels = encode_target(str(row["chosen"]), "chosen")
    rejected_input_ids, rejected_attention_mask, rejected_labels = encode_target(str(row["rejected"]), "rejected")
    return {
        "preference_id": row["preference_id"],
        "unit_id": row.get("unit_id"),
        "prompt": row["prompt"],
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
        "metadata": {
            "pair_type": row.get("pair_type"),
            "score_margin": row.get("score_margin"),
            "chosen_candidate_key": row.get("chosen_candidate_key"),
            "rejected_candidate_key": row.get("rejected_candidate_key"),
        },
    }


class DPODataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_seq_length: int,
        reference_logps: dict[str, dict[str, float]] | None = None,
    ) -> None:
        items = [
            tokenize_completion_pair(row, tokenizer, max_seq_length=max_seq_length)
            for row in rows
        ]
        if reference_logps is not None:
            for item in items:
                ref = reference_logps.get(item["preference_id"])
                if ref is None:
                    raise ValueError(f"missing_reference_logps:{item['preference_id']}")
                item["chosen_reference_logp"] = float(ref["chosen_reference_logp"])
                item["rejected_reference_logp"] = float(ref["rejected_reference_logp"])
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class DPOCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def _pad(self, batch: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        import torch

        input_key = f"{prefix}_input_ids"
        mask_key = f"{prefix}_attention_mask"
        labels_key = f"{prefix}_labels"
        max_length = max(len(item[input_key]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            pad_count = max_length - len(item[input_key])
            input_ids.append(item[input_key] + [self.pad_token_id] * pad_count)
            attention_mask.append(item[mask_key] + [0] * pad_count)
            labels.append(item[labels_key] + [-100] * pad_count)
        return {
            f"{prefix}_input_ids": torch.tensor(input_ids, dtype=torch.long),
            f"{prefix}_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            f"{prefix}_labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        payload = {}
        payload.update(self._pad(batch, "chosen"))
        payload.update(self._pad(batch, "rejected"))
        payload["preference_ids"] = [item["preference_id"] for item in batch]
        payload["unit_ids"] = [item.get("unit_id") for item in batch]
        payload["chosen_reference_logps"] = torch.tensor(
            [float(item["chosen_reference_logp"]) for item in batch],
            dtype=torch.float32,
        )
        payload["rejected_reference_logps"] = torch.tensor(
            [float(item["rejected_reference_logp"]) for item in batch],
            dtype=torch.float32,
        )
        return payload


def move_dpo_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key not in {"preference_ids", "unit_ids"}
    } | {
        "preference_ids": batch.get("preference_ids", []),
        "unit_ids": batch.get("unit_ids", []),
    }


def sequence_logps(model: Any, *, input_ids: Any, attention_mask: Any, labels: Any) -> Any:
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask.to(token_logps.dtype)).sum(dim=-1)


def dpo_loss_from_logps(
    *,
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    beta: float,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    policy_logp_margin = policy_chosen_logps - policy_rejected_logps
    reference_logp_margin = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_logp_margin - reference_logp_margin)
    losses = -F.logsigmoid(logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    reward_margins = chosen_rewards - rejected_rewards
    metrics = {
        "loss": losses.mean(),
        "reward_margin": reward_margins.mean().detach(),
        "reward_accuracy": (reward_margins > 0).float().mean().detach(),
        "chosen_reward": chosen_rewards.mean().detach(),
        "rejected_reward": rejected_rewards.mean().detach(),
        "chosen_logp": policy_chosen_logps.mean().detach(),
        "rejected_logp": policy_rejected_logps.mean().detach(),
        "logp_margin": policy_logp_margin.mean().detach(),
        "reference_logp_margin": reference_logp_margin.mean().detach(),
        "preference_count": torch.tensor(float(policy_chosen_logps.shape[0])),
    }
    return losses.mean(), metrics


def tensor_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().float().cpu().item())
    return float(value)


def numeric_metrics_to_floats(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: tensor_to_float(value) for key, value in metrics.items()}


def precompute_reference_logps(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    rows: list[dict[str, Any]],
    sft_adapter_path: Path,
    tokenizer_path: Path,
    reference_logps_path: Path,
) -> dict[str, dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader

    if args.force_reference_logps_recompute and reference_logps_path.exists():
        reference_logps_path.unlink()
    existing = load_reference_logps(reference_logps_path) if reference_logps_path.exists() else {}
    missing_rows = [row for row in rows if row["preference_id"] not in existing]
    if not missing_rows:
        write_json(
            run_dir / "reference_logps_summary.json",
            {
                "reference_logps_path": project_path(reference_logps_path),
                "row_count": len(existing),
                "updated_at": utc_now_iso(),
            },
        )
        return existing

    tokenizer, reference_model = load_model_and_tokenizer(
        args=args,
        sft_adapter_path=sft_adapter_path,
        tokenizer_path=tokenizer_path,
        trainable=False,
    )
    reference_model.eval()
    dataset = DPODataset(missing_rows, tokenizer, max_seq_length=args.max_seq_length)
    collator = DPOCollatorForReference(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    device = model_input_device(reference_model)
    for batch_index, batch in enumerate(loader):
        model_batch = move_dpo_batch_to_device(batch, device)
        with torch.no_grad():
            chosen_logps = sequence_logps(
                reference_model,
                input_ids=model_batch["chosen_input_ids"],
                attention_mask=model_batch["chosen_attention_mask"],
                labels=model_batch["chosen_labels"],
            )
            rejected_logps = sequence_logps(
                reference_model,
                input_ids=model_batch["rejected_input_ids"],
                attention_mask=model_batch["rejected_attention_mask"],
                labels=model_batch["rejected_labels"],
            )
        rows_to_append = []
        for item_index, preference_id in enumerate(batch["preference_ids"]):
            row = {
                "preference_id": preference_id,
                "chosen_reference_logp": float(chosen_logps[item_index].detach().cpu().item()),
                "rejected_reference_logp": float(rejected_logps[item_index].detach().cpu().item()),
                "reference_logp_margin": float((chosen_logps[item_index] - rejected_logps[item_index]).detach().cpu().item()),
                "computed_at": utc_now_iso(),
                "batch_index": batch_index,
            }
            rows_to_append.append(row)
            existing[preference_id] = row
        append_jsonl(reference_logps_path, rows_to_append)

    del reference_model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(
        run_dir / "reference_logps_summary.json",
        {
            "reference_logps_path": project_path(reference_logps_path),
            "row_count": len(existing),
            "updated_at": utc_now_iso(),
        },
    )
    return existing


class DPOCollatorForReference(DPOCollator):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        payload = {}
        payload.update(self._pad(batch, "chosen"))
        payload.update(self._pad(batch, "rejected"))
        payload["preference_ids"] = [item["preference_id"] for item in batch]
        payload["unit_ids"] = [item.get("unit_id") for item in batch]
        payload["chosen_reference_logps"] = torch.zeros(len(batch), dtype=torch.float32)
        payload["rejected_reference_logps"] = torch.zeros(len(batch), dtype=torch.float32)
        return payload


def load_reference_logps(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path, allow_truncated_last_line=True)
    reference_logps: dict[str, dict[str, float]] = {}
    for row in rows:
        preference_id = row.get("preference_id")
        if not isinstance(preference_id, str) or not preference_id:
            raise ValueError(f"reference_logps_invalid_preference_id:{path}")
        if preference_id in reference_logps:
            raise ValueError(f"reference_logps_duplicate_preference_id:{preference_id}")
        reference_logps[preference_id] = {
            "chosen_reference_logp": float(row["chosen_reference_logp"]),
            "rejected_reference_logp": float(row["rejected_reference_logp"]),
        }
    return reference_logps


def train_log_row(
    *,
    args: argparse.Namespace,
    epoch: int,
    batch_index: int,
    global_step: int,
    metrics: dict[str, float],
    learning_rate: float,
    grad_norm: float | None,
    skipped_batches: int,
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "epoch": epoch,
        "batch_index": batch_index,
        "global_step": global_step,
        "loss": metrics.get("loss"),
        "reward_margin": metrics.get("reward_margin"),
        "reward_accuracy": metrics.get("reward_accuracy"),
        "chosen_reward": metrics.get("chosen_reward"),
        "rejected_reward": metrics.get("rejected_reward"),
        "chosen_logp": metrics.get("chosen_logp"),
        "rejected_logp": metrics.get("rejected_logp"),
        "logp_margin": metrics.get("logp_margin"),
        "reference_logp_margin": metrics.get("reference_logp_margin"),
        "learning_rate": learning_rate,
        "grad_norm": grad_norm,
        "skipped_batches": skipped_batches,
        "updated_at": utc_now_iso(),
    }


def log_dpo_oom_skipped_batch(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    epoch: int,
    batch_index: int,
    global_step: int,
    batch: dict[str, Any],
    error: BaseException,
    cuda_memory: dict[str, int] | None,
) -> None:
    append_jsonl(
        run_dir / "oom_skipped_batches.jsonl",
        [
            {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "preference_ids": list(batch.get("preference_ids", [])),
                "unit_ids": list(batch.get("unit_ids", [])),
                "chosen_shape": list(batch["chosen_input_ids"].shape) if hasattr(batch.get("chosen_input_ids"), "shape") else None,
                "rejected_shape": list(batch["rejected_input_ids"].shape) if hasattr(batch.get("rejected_input_ids"), "shape") else None,
                "error_type": error.__class__.__name__,
                "error": str(error),
                "cuda_memory": cuda_memory,
                "updated_at": utc_now_iso(),
            }
        ],
    )


def total_optimizer_steps(args: argparse.Namespace, dataset_length: int) -> int:
    batches_per_epoch = math.ceil(dataset_length / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    return max(optimizer_steps_per_epoch * args.epochs, 1)


def two_thirds_validation_step(total_steps: int) -> int:
    return max(math.ceil(total_steps * 2 / 3), 1)


def two_thirds_validation_seed(*, base_seed: int, global_step: int) -> int:
    return int(base_seed) + int(global_step) * 1_000_003


def select_two_thirds_validation_rows(
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
        raise ValueError(f"unsupported_two_thirds_validation_sample_strategy:{sample_strategy}")

    rng = random.Random(two_thirds_validation_seed(base_seed=seed, global_step=global_step))
    selected_indices = sorted(rng.sample(range(len(validation_rows)), max_samples))
    return [validation_rows[index] for index in selected_indices]


def run_two_thirds_validation(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    validation_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    wandb_run: Any | None,
    global_step: int,
    trigger_step: int,
    checkpoint_name: str,
) -> dict[str, Any]:
    requested_samples = int(args.two_thirds_validation_samples)
    if requested_samples <= 0 or not validation_rows:
        return {}
    eval_rows = select_two_thirds_validation_rows(
        validation_rows,
        max_samples=min(requested_samples, len(validation_rows)),
        sample_strategy=args.two_thirds_validation_sample_strategy,
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
        predictions_path=run_dir / "two_thirds_validation_predictions.jsonl",
        resume_scope="checkpoint",
    )
    metrics["global_step"] = global_step
    metrics["trigger_step"] = trigger_step
    metrics["trigger_fraction"] = 2 / 3
    metrics["eval_requested_max_samples"] = requested_samples
    metrics["eval_max_samples"] = len(eval_rows)
    metrics["eval_sample_strategy"] = args.two_thirds_validation_sample_strategy
    metrics["eval_sample_seed"] = two_thirds_validation_seed(
        base_seed=int(args.seed),
        global_step=global_step,
    )
    metrics["eval_sample_unit_ids"] = [build_unit_id(row) for row in eval_rows]
    metrics["selection_key"] = list(structural_selection_key(metrics))
    append_jsonl(run_dir / "two_thirds_validation_metrics.jsonl", [metrics])
    wandb_log_numeric_metrics(wandb_run, metrics, prefix="validation_two_thirds", step=global_step)
    model.train()
    return metrics


def train(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    preference_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    config: dict[str, Any],
    sft_adapter_path: Path,
    tokenizer_path: Path,
    reference_logps_path: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    reference_logps = precompute_reference_logps(
        args=args,
        run_dir=run_dir,
        rows=preference_rows,
        sft_adapter_path=sft_adapter_path,
        tokenizer_path=tokenizer_path,
        reference_logps_path=reference_logps_path,
    )
    tokenizer, model = load_model_and_tokenizer(
        args=args,
        sft_adapter_path=sft_adapter_path,
        tokenizer_path=tokenizer_path,
        trainable=True,
    )
    dataset = DPODataset(
        preference_rows,
        tokenizer,
        max_seq_length=args.max_seq_length,
        reference_logps=reference_logps,
    )
    collator = DPOCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    total_steps = total_optimizer_steps(args, len(dataset))
    two_thirds_step = two_thirds_validation_step(total_steps)
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_steps)
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
        update_state(run_dir, state, status="running")

    wandb_run = init_wandb_run(args=args, config=config, run_dir=run_dir)
    model.train()
    device = model_input_device(model)
    global_step = int(state.get("global_step", 0))
    start_epoch = int(state.get("epoch", 0))
    next_batch_index = int(state.get("next_batch_index", 0))
    oom_skip_count = int(state.get("oom_skipped_batches", 0) or 0)
    rolling_metrics: dict[str, float] = {}
    rolling_count = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < next_batch_index:
                continue
            model_batch = move_dpo_batch_to_device(batch, device)
            try:
                policy_chosen_logps = sequence_logps(
                    model,
                    input_ids=model_batch["chosen_input_ids"],
                    attention_mask=model_batch["chosen_attention_mask"],
                    labels=model_batch["chosen_labels"],
                )
                policy_rejected_logps = sequence_logps(
                    model,
                    input_ids=model_batch["rejected_input_ids"],
                    attention_mask=model_batch["rejected_attention_mask"],
                    labels=model_batch["rejected_labels"],
                )
                loss, batch_metrics = dpo_loss_from_logps(
                    policy_chosen_logps=policy_chosen_logps,
                    policy_rejected_logps=policy_rejected_logps,
                    reference_chosen_logps=model_batch["chosen_reference_logps"],
                    reference_rejected_logps=model_batch["rejected_reference_logps"],
                    beta=args.beta,
                )
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower()
                if not is_oom or args.oom_recovery != "skip_batch":
                    raise
                oom_skip_count += 1
                log_dpo_oom_skipped_batch(
                    run_dir=run_dir,
                    args=args,
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=global_step,
                    batch=batch,
                    error=exc,
                    cuda_memory=cuda_memory_snapshot(torch),
                )
                if args.max_oom_skips and oom_skip_count > args.max_oom_skips:
                    raise RuntimeError(f"max_oom_skips_exceeded:{oom_skip_count}") from exc
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                update_state(
                    run_dir,
                    state,
                    status="running",
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    oom_skipped_batches=oom_skip_count,
                )
                continue

            batch_metrics_float = numeric_metrics_to_floats(batch_metrics)
            for key, value in batch_metrics_float.items():
                rolling_metrics[key] = rolling_metrics.get(key, 0.0) + value
            rolling_count += 1

            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if not should_step:
                update_state(
                    run_dir,
                    state,
                    status="running",
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    oom_skipped_batches=oom_skip_count,
                )
                continue

            grad_norm_value: float | None = None
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=1.0,
                )
                grad_norm_value = tensor_to_float(grad_norm)
            except RuntimeError:
                grad_norm_value = None
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            next_epoch = epoch + 1 if batch_index + 1 == len(loader) else epoch
            next_batch = 0 if batch_index + 1 == len(loader) else batch_index + 1
            averaged_metrics = {
                key: value / max(rolling_count, 1)
                for key, value in rolling_metrics.items()
            }
            learning_rate = float(scheduler.get_last_lr()[0])
            update_state(
                run_dir,
                state,
                status="running",
                global_step=global_step,
                epoch=next_epoch,
                next_batch_index=next_batch,
                last_loss=averaged_metrics.get("loss"),
                oom_skipped_batches=oom_skip_count,
            )
            log_row = train_log_row(
                args=args,
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
                metrics=averaged_metrics,
                learning_rate=learning_rate,
                grad_norm=grad_norm_value,
                skipped_batches=oom_skip_count,
            )
            append_jsonl(run_dir / "train_log.jsonl", [log_row])
            wandb_log(
                wandb_run,
                {
                    "train/global_step": global_step,
                    "train/loss": log_row["loss"],
                    "train/reward_margin": log_row["reward_margin"],
                    "train/reward_accuracy": log_row["reward_accuracy"],
                    "train/chosen_reward": log_row["chosen_reward"],
                    "train/rejected_reward": log_row["rejected_reward"],
                    "train/chosen_logp": log_row["chosen_logp"],
                    "train/rejected_logp": log_row["rejected_logp"],
                    "train/logp_margin": log_row["logp_margin"],
                    "train/reference_logp_margin": log_row["reference_logp_margin"],
                    "train/learning_rate": learning_rate,
                    "train/grad_norm": grad_norm_value,
                    "train/epoch": epoch,
                    "train/batch_index": batch_index,
                    "train/skipped_batches": oom_skip_count,
                },
                step=global_step,
            )
            rolling_metrics = {}
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
                removed = prune_old_checkpoints(run_dir, keep_last=args.checkpoint_keep_last)
                if removed:
                    append_jsonl(
                        run_dir / "checkpoint_retention_log.jsonl",
                        [
                            {
                                "run_id": args.run_id,
                                "global_step": global_step,
                                "saved_checkpoint": str(saved_checkpoint),
                                "removed_checkpoints": removed,
                                "keep_last": args.checkpoint_keep_last,
                                "updated_at": utc_now_iso(),
                            }
                        ],
                    )

            if (
                args.two_thirds_validation_samples > 0
                and global_step >= two_thirds_step
                and not state.get("two_thirds_validation_completed")
            ):
                checkpoint_name = (
                    saved_checkpoint.name
                    if saved_checkpoint is not None and saved_checkpoint.name.endswith(str(global_step))
                    else f"in-memory-two-thirds-step-{global_step}"
                )
                update_state(
                    run_dir,
                    state,
                    status="running",
                    two_thirds_validation_status="running",
                    two_thirds_validation_step=global_step,
                    two_thirds_validation_trigger_step=two_thirds_step,
                    two_thirds_validation_total_steps=total_steps,
                )
                two_thirds_metrics = run_two_thirds_validation(
                    args=args,
                    run_dir=run_dir,
                    validation_rows=validation_rows,
                    tokenizer=tokenizer,
                    model=model,
                    wandb_run=wandb_run,
                    global_step=global_step,
                    trigger_step=two_thirds_step,
                    checkpoint_name=checkpoint_name,
                )
                update_state(
                    run_dir,
                    state,
                    status="running",
                    two_thirds_validation_status="completed",
                    two_thirds_validation_completed=True,
                    two_thirds_validation_metrics=two_thirds_metrics,
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
    removed = prune_old_checkpoints(run_dir, keep_last=args.checkpoint_keep_last)
    if removed:
        append_jsonl(
            run_dir / "checkpoint_retention_log.jsonl",
            [
                {
                    "run_id": args.run_id,
                    "global_step": int(state.get("global_step", 0)),
                    "saved_checkpoint": str(final_checkpoint),
                    "removed_checkpoints": removed,
                    "keep_last": args.checkpoint_keep_last,
                    "updated_at": utc_now_iso(),
                }
            ],
        )

    if args.skip_final_eval:
        metrics = {
            "run_id": args.run_id,
            "model_variant": SERIALIZED_SFT_DPO,
            "source_model_variant": SERIALIZED_SFT,
            "serialization": BLOCKS_TSV_V1,
            "status": "trained_without_final_eval",
            "final_checkpoint": str(final_checkpoint),
            "row_count_preference_train": len(preference_rows),
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
        metrics["model_variant"] = SERIALIZED_SFT_DPO
        metrics["source_model_variant"] = SERIALIZED_SFT
        metrics["global_step"] = int(state.get("global_step", 0))
        metrics["selection_key"] = list(structural_selection_key(metrics))
        metrics["best_checkpoint"] = str(final_checkpoint)
        wandb_log_numeric_metrics_on_metric_axis(
            wandb_run,
            metrics,
            prefix="validation",
        )

    metrics["global_step"] = int(state.get("global_step", 0))
    write_json(run_dir / "metrics.json", metrics)
    update_state(
        run_dir,
        state,
        status="completed",
        completed_at=utc_now_iso(),
        best_checkpoint=metrics.get("best_checkpoint"),
        best_validation_score=metrics.get("selection_key"),
    )
    wandb_log_numeric_metrics_on_metric_axis(
        wandb_run,
        metrics,
        prefix="final",
    )
    if args.wandb_log_artifacts:
        wandb_log_dpo_artifacts(
            wandb_run,
            run_dir=run_dir,
            final_checkpoint=metrics.get("best_checkpoint") or metrics.get("final_checkpoint"),
        )
    wandb_finish(wandb_run)
    return metrics


def main() -> None:
    args = parse_args()
    loaded_env = load_env_file(DEFAULT_ENV_FILE)
    validate_wandb_request(args)
    preference_file = resolve_project_path(args.preference_file)
    validation_file = resolve_project_path(args.validation_file)
    base_model_path = resolve_project_path(args.base_model_path).resolve()
    sft_adapter_path, checkpoint_root = resolve_sft_adapter_and_checkpoint_root(args.sft_adapter_path)
    tokenizer_path = resolve_tokenizer_path(checkpoint_root=checkpoint_root, base_model_path=base_model_path)
    run_dir = resolve_project_path(args.output_dir) / args.run_id
    assert_output_dir_safe(run_dir, base_model_path=base_model_path, sft_checkpoint_root=checkpoint_root)
    reference_logps_path = (
        resolve_project_path(args.reference_logps_file)
        if args.reference_logps_file is not None
        else run_dir / "reference_logps.jsonl"
    )

    preference_rows = load_preference_rows(preference_file, max_samples=args.max_train_samples)
    validation_rows = load_sft_rows(
        validation_file,
        max_samples=args.max_validation_samples,
        split_name="validation",
    )
    config = build_config(
        args,
        run_dir=run_dir,
        preference_rows=preference_rows,
        validation_rows=validation_rows,
        sft_adapter_path=sft_adapter_path,
        checkpoint_root=checkpoint_root,
        tokenizer_path=tokenizer_path,
        loaded_env_keys=loaded_env.keys(),
    )
    state = initialize_run(run_dir, config, dry_run=args.dry_run)
    if args.dry_run:
        metrics = {
            "dry_run": True,
            "row_count_preference_train": len(preference_rows),
            "row_count_validation": len(validation_rows),
            "ready_for_full_run": config["model_checks"]["ready_for_full_run"],
            "model_warnings": config["model_checks"]["warnings"],
            "runtime_versions": config["runtime_versions"],
            "cuda": config["cuda"],
            "source_model_protected": True,
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "state.json", state)
        print(json.dumps({"output_dir": str(run_dir), **metrics}, ensure_ascii=False))
        return

    if not config["model_checks"]["ready_for_full_run"]:
        raise RuntimeError(f"DPO full run is not ready. See model_checks: {config['model_checks']}")
    metrics = train(
        args,
        run_dir=run_dir,
        preference_rows=preference_rows,
        validation_rows=validation_rows,
        config=config,
        sft_adapter_path=sft_adapter_path,
        tokenizer_path=tokenizer_path,
        reference_logps_path=reference_logps_path,
    )
    print(json.dumps({"output_dir": str(run_dir), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
