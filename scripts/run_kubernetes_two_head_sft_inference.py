from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from llm_structured_semantic_generation.dataset_io import write_json
from llm_structured_semantic_generation.evaluation import (
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)
from llm_structured_semantic_generation.resumable_run import ResumableRun, utc_now_iso
from llm_structured_semantic_generation.structure import blocks_to_yaml
from build_kubernetes_dpo_candidates import (
    maybe_clear_cuda_cache,
    resolve_checkpoint,
    resolve_checkpoint_root_and_adapter,
)
from train_kubernetes_sft import (
    build_unit_id,
    inspect_model_path,
    load_env_file,
    positive_int,
)
from train_kubernetes_two_head_sft import (
    CONTENT_BLOCKS_V1,
    RECORD_PREFIX_STATE,
    TWO_HEAD_SFT,
    extract_content_blocks_prediction,
    generate_validation_completion,
    load_model_and_tokenizer,
    load_sft_rows,
    predict_levels_for_content,
)


DEFAULT_TWO_HEAD_RUN_DIR = (
    REPO_ROOT / "results" / "two_head_sft_kubernetes_v1" / "two-head-sft-v1-20260516"
)
DEFAULT_TEST_FILE = REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "test.jsonl"
DEFAULT_BASE_MODEL_PATH = REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "two_head_sft_kubernetes_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two_head_sft Kubernetes v1 inference on an SFT split and compute parser-facing metrics."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--two-head-run-dir", type=Path, default=DEFAULT_TWO_HEAD_RUN_DIR)
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Use best, latest, a checkpoint directory name, or a checkpoint/adapter path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume or create output-dir/run-id. Defaults to a UTC timestamped two-head SFT test run.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Number of predictions to persist per batch. Use 1 for maximal resumability.",
    )
    parser.add_argument("--max-samples", type=positive_int, default=None)
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write config/metrics without loading the model.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def source_model_args(
    *,
    source_config: dict[str, Any],
    base_model_path: Path,
    gpu_memory: str,
    cpu_memory: str,
) -> SimpleNamespace:
    lora = source_config.get("lora", {})
    level_head = source_config.get("level_head", {})
    target_modules = lora.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
    return SimpleNamespace(
        base_model_path=base_model_path,
        gpu_memory=gpu_memory,
        cpu_memory=cpu_memory,
        lora_r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_target_modules=",".join(str(module) for module in target_modules),
        lambda_level=float(level_head.get("lambda_level", 1.0)),
        level_class_count=int(level_head.get("class_count", 9)),
        level_head_hidden_dim=int(level_head.get("hidden_dim", 256)),
        level_head_dropout=float(level_head.get("dropout", 0.05)),
    )


def inspect_checkpoint(checkpoint_root: Path, adapter_path: Path) -> dict[str, Any]:
    checks = {
        "checkpoint_root_exists": checkpoint_root.exists(),
        "adapter_path_exists": adapter_path.exists(),
        "has_adapter_config": (adapter_path / "adapter_config.json").exists(),
        "has_adapter_weights": any(adapter_path.glob("adapter_model.*")),
        "has_level_head": (checkpoint_root / "level_head.pt").exists(),
        "has_tokenizer": (checkpoint_root / "tokenizer").exists(),
        "warnings": [],
    }
    if not checks["has_tokenizer"]:
        checks["warnings"].append("checkpoint_tokenizer_missing_using_base_tokenizer")
    checks["ready_for_full_run"] = all(
        bool(checks[key])
        for key in (
            "checkpoint_root_exists",
            "adapter_path_exists",
            "has_adapter_config",
            "has_adapter_weights",
            "has_level_head",
        )
    )
    return checks


def load_two_head_model_and_tokenizer(
    *,
    source_config: dict[str, Any],
    base_model_path: Path,
    checkpoint_root: Path,
    adapter_path: Path,
    gpu_memory: str,
    cpu_memory: str,
) -> tuple[Any, Any]:
    import torch
    from peft import set_peft_model_state_dict

    model_args = source_model_args(
        source_config=source_config,
        base_model_path=base_model_path,
        gpu_memory=gpu_memory,
        cpu_memory=cpu_memory,
    )
    tokenizer, model = load_model_and_tokenizer(model_args)

    if (adapter_path / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file

        adapter_state = load_file(adapter_path / "adapter_model.safetensors")
    else:
        adapter_state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
    set_peft_model_state_dict(model.backbone, adapter_state, adapter_name="default")
    try:
        level_head_state = torch.load(checkpoint_root / "level_head.pt", map_location="cpu", weights_only=True)
    except TypeError:
        level_head_state = torch.load(checkpoint_root / "level_head.pt", map_location="cpu")
    model.level_head.load_state_dict(level_head_state)
    model.eval()
    model.backbone.config.use_cache = True
    return tokenizer, model


def build_resume_signature(
    args: argparse.Namespace,
    *,
    test_file: Path,
    two_head_run_dir: Path,
    checkpoint_root: Path,
    adapter_path: Path,
    base_model_path: Path,
    source_config: dict[str, Any],
) -> dict[str, Any]:
    model_args = source_model_args(
        source_config=source_config,
        base_model_path=base_model_path,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
    )
    return {
        "stage": "two_head_sft_inference_v1",
        "model_variant": TWO_HEAD_SFT,
        "serialization": CONTENT_BLOCKS_V1,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "test_file": project_path(test_file),
        "two_head_run_dir": project_path(two_head_run_dir),
        "checkpoint_root": project_path(checkpoint_root),
        "adapter_path": project_path(adapter_path),
        "base_model_path": project_path(base_model_path),
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "lora_r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "lora_target_modules": model_args.lora_target_modules.split(","),
        "lambda_level": model_args.lambda_level,
        "level_class_count": model_args.level_class_count,
        "level_head_hidden_dim": model_args.level_head_hidden_dim,
        "level_head_dropout": model_args.level_head_dropout,
        "target_contract": "prompt -> content_blocks_v1 + level head -> blocks_tsv_v1 -> parser -> YAML",
    }


def build_prediction_row(
    *,
    row: dict[str, Any],
    checkpoint_root: Path,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    completion: dict[str, Any] = {}
    generation_error: BaseException | None = None
    try:
        completion = generate_validation_completion(
            tokenizer=tokenizer,
            model=model,
            prompt=str(row["prompt"]),
            max_new_tokens=max_new_tokens,
        )
    except RuntimeError as exc:
        generation_error = exc
        maybe_clear_cuda_cache()

    raw_output = completion.get("raw_text") if generation_error is None else None
    content_blocks: list[dict[str, Any]] = []
    predicted_blocks: list[dict[str, Any]] = []
    reconstructed_yaml = ""
    parse_errors: list[str] = []
    reconstruction_errors: list[str] = []
    evaluation = None

    if generation_error is not None:
        parse_errors.append(f"generation_error:{type(generation_error).__name__}:{generation_error}")
    elif isinstance(raw_output, str):
        try:
            content_blocks, content_text, spans = extract_content_blocks_prediction(raw_output)
            predicted_blocks = predict_levels_for_content(
                tokenizer=tokenizer,
                model=model,
                prompt=str(completion["prompt"]),
                content_blocks=content_blocks,
                content_text=content_text,
                spans=spans,
            )
            reconstruction = blocks_to_yaml(predicted_blocks, recovery_mode="strict")
            reconstructed_yaml = reconstruction.yaml_text
            reconstruction_errors = list(reconstruction.errors)
            evaluation = evaluate_blocks_prediction(
                str(row["target_yaml_normalized"]),
                predicted_blocks,
                recovery_mode="strict",
                prompt_text=str(row["prompt"]),
            ).to_dict()
        except ValueError as exc:
            parse_errors.append(f"structured_output_parse_error:content_blocks_v1:{exc.__class__.__name__}:{exc}")
        except RuntimeError as exc:
            parse_errors.append(f"level_prediction_error:{exc.__class__.__name__}:{exc}")
            maybe_clear_cuda_cache()
    else:
        parse_errors.append("generation_error:missing_raw_output")

    return {
        "unit_id": build_unit_id(row),
        "sample_id": row["sample_id"],
        "prompt_variant": row["prompt_variant"],
        "split": row["split"],
        "prompt": row["prompt"],
        "reference_yaml": row["target_yaml_normalized"],
        "checkpoint": project_path(checkpoint_root),
        "output_format": CONTENT_BLOCKS_V1,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "generation_ok": generation_error is None,
        "generation_error_type": type(generation_error).__name__ if generation_error is not None else None,
        "generation_error": str(generation_error) if generation_error is not None else None,
        "two_head_prompt": completion.get("prompt"),
        "raw_model_output": raw_output,
        "generated_token_count": completion.get("generated_token_count"),
        "predicted_content_blocks": content_blocks,
        "predicted_blocks": predicted_blocks,
        "reconstructed_yaml": reconstructed_yaml,
        "parser_errors": parse_errors,
        "reconstruction_errors": reconstruction_errors,
        "structured_output_parse_success": evaluation is not None,
        "evaluation": evaluation,
        "generated_at": utc_now_iso(),
    }


def derive_metrics(
    *,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = [
        StructuralEvaluation(**row["evaluation"])
        for row in predictions
        if isinstance(row.get("evaluation"), dict)
    ]
    generated_token_counts = [
        float(row["generated_token_count"])
        for row in predictions
        if isinstance(row.get("generated_token_count"), int)
    ]
    metrics: dict[str, Any] = {
        "run_id": config["run_id"],
        "stage": "two_head_sft_inference_v1",
        "model_variant": TWO_HEAD_SFT,
        "serialization": CONTENT_BLOCKS_V1,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "split": config["split"],
        "source_two_head_run_id": config["source_two_head_run_id"],
        "checkpoint": config["checkpoint_root"],
        "row_count": len(predictions),
        "evaluated_count": len(evaluations),
        "generation_success_rate": average(
            [1.0 if row.get("generation_ok") else 0.0 for row in predictions]
        ),
        "structured_output_parse_success_rate": len(evaluations) / len(predictions) if predictions else 0.0,
        "average_generated_token_count": average(generated_token_counts),
        "perplexity_available": False,
    }
    metrics.update(summarize_evaluations(evaluations))
    return metrics


def write_dry_run_artifacts(run_dir: Path, config: dict[str, Any]) -> None:
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "state.json",
        {
            "run_id": config["run_id"],
            "status": "dry_run",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "total_units": config["row_count"],
            "processed_units": 0,
            "remaining_units": config["row_count"],
            "resume_signature": config["resume_signature"],
        },
    )
    write_json(
        run_dir / "metrics.json",
        {
            "dry_run": True,
            "row_count": config["row_count"],
            "ready_for_full_run": config["ready_for_full_run"],
            "model_warnings": config["model_checks"]["warnings"],
            "checkpoint_warnings": config["checkpoint_checks"]["warnings"],
            "checkpoint_root": config["checkpoint_root"],
            "adapter_path": config["adapter_path"],
        },
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"two-head-sft-v1-test-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    load_env_file()
    test_file = resolve_repo_path(args.test_file)
    two_head_run_dir = resolve_repo_path(args.two_head_run_dir)
    base_model_path = resolve_repo_path(args.base_model_path)
    checkpoint_root, adapter_path = resolve_checkpoint_root_and_adapter(
        resolve_checkpoint(two_head_run_dir, args.checkpoint)
    )
    source_config = read_json(two_head_run_dir / "config.json")
    rows = load_sft_rows(test_file, max_samples=args.max_samples, split_name="test")
    run_dir = resolve_repo_path(args.output_dir) / run_id
    source_two_head_run_id = two_head_run_dir.name
    model_checks = inspect_model_path(base_model_path)
    checkpoint_checks = inspect_checkpoint(checkpoint_root, adapter_path)
    ready_for_full_run = bool(model_checks["ready_for_full_run"]) and bool(
        checkpoint_checks["ready_for_full_run"]
    )

    config = {
        "run_id": run_id,
        "stage": "two_head_sft_inference_v1",
        "created_at": utc_now_iso(),
        "model_variant": TWO_HEAD_SFT,
        "serialization": CONTENT_BLOCKS_V1,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "split": "test",
        "test_file": project_path(test_file),
        "source_two_head_run_id": source_two_head_run_id,
        "two_head_run_dir": project_path(two_head_run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_root": project_path(checkpoint_root),
        "adapter_path": project_path(adapter_path),
        "base_model_path": project_path(base_model_path),
        "row_count": len(rows),
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "gpu_memory": args.gpu_memory,
        "cpu_memory": args.cpu_memory,
        "decoding": "greedy",
        "target_contract": "prompt -> content_blocks_v1 + level head -> blocks_tsv_v1 -> parser -> YAML",
        "dry_run": args.dry_run,
        "model_checks": model_checks,
        "checkpoint_checks": checkpoint_checks,
        "ready_for_full_run": ready_for_full_run,
        "artifacts": {
            "predictions": "predictions.jsonl",
            "metrics": "metrics.json",
            "state": "state.json",
            "config": "config.json",
        },
        "resume_signature": build_resume_signature(
            args,
            test_file=test_file,
            two_head_run_dir=two_head_run_dir,
            checkpoint_root=checkpoint_root,
            adapter_path=adapter_path,
            base_model_path=base_model_path,
            source_config=source_config,
        ),
    }

    if args.dry_run:
        write_dry_run_artifacts(run_dir, config)
        print(
            {
                "dry_run": True,
                "output_dir": str(run_dir),
                "row_count": len(rows),
                "ready_for_full_run": ready_for_full_run,
                "model_warnings": model_checks["warnings"],
                "checkpoint_warnings": checkpoint_checks["warnings"],
            }
        )
        return

    if not ready_for_full_run:
        raise RuntimeError(
            "Two-head inference is not ready. "
            f"model_checks={model_checks}; checkpoint_checks={checkpoint_checks}"
        )

    run = ResumableRun.initialize(
        run_dir=run_dir,
        config=config,
        total_units=len(rows),
        unit_id_field="unit_id",
        primary_artifact_name="predictions",
        artifact_paths={"predictions": "predictions.jsonl"},
    )

    if run.is_complete:
        metrics = derive_metrics(config=config, predictions=run.primary_rows)
        write_json(run_dir / "metrics.json", metrics)
        run.mark_completed()
        print({"output_dir": str(run_dir), **metrics})
        return

    tokenizer, model = load_two_head_model_and_tokenizer(
        source_config=source_config,
        base_model_path=base_model_path,
        checkpoint_root=checkpoint_root,
        adapter_path=adapter_path,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
    )
    pending_rows = [row for row in rows if build_unit_id(row) not in run.completed_unit_id_set]

    batch_predictions: list[dict[str, Any]] = []
    try:
        for row in pending_rows:
            batch_predictions.append(
                build_prediction_row(
                    row=row,
                    checkpoint_root=checkpoint_root,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            if len(batch_predictions) >= args.batch_size:
                run.record_batch(batch_predictions)
                last_unit_id = batch_predictions[-1]["unit_id"]
                print(f"progress completed={len(run.completed_unit_ids)}/{run.total_units} last_unit={last_unit_id}")
                batch_predictions = []
    except KeyboardInterrupt:
        if batch_predictions:
            run.record_batch(batch_predictions)
            print(f"interrupted_progress_saved completed={len(run.completed_unit_ids)}/{run.total_units}")
        raise

    if batch_predictions:
        run.record_batch(batch_predictions)

    metrics = derive_metrics(config=config, predictions=run.primary_rows)
    write_json(run_dir / "metrics.json", metrics)
    run.mark_completed()
    print({"output_dir": str(run_dir), **metrics})


if __name__ == "__main__":
    main()
