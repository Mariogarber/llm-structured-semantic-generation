from __future__ import annotations

import argparse
import json
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import write_json
from llm_structured_semantic_generation.latent_level_probe import (
    FEATURE_STRATEGIES,
    PROBE_TYPES,
    atomic_write_json,
    build_content_only_example,
    build_unit_id,
    chunk_path_for_unit,
    completed_probe_ids,
    completed_unit_ids_from_chunks,
    features_for_hidden_states,
    init_or_validate_config,
    metadata_from_span,
    probe_id,
    rebuild_aggregate_artifacts,
    read_sft_rows,
    synthetic_hidden_for_example,
    train_and_evaluate_probe,
    write_state,
)
from llm_structured_semantic_generation.resumable_run import RunCompatibilityError, utc_now_iso


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "latent_level_probe_kubernetes_v1"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract line-level hidden-state features and train resumable probes for YAML level prediction."
    )
    parser.add_argument("--stage", choices=["extract", "probe", "all"], default="all")
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "train.jsonl")
    parser.add_argument("--validation-file", type=Path, default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "validation.jsonl")
    parser.add_argument("--test-file", type=Path, default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "test.jsonl")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Load test rows. Keep disabled during probe selection; enable only for final candidate evaluation.",
    )
    parser.add_argument("--base-model-path", type=Path, default=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--gpu-memory", default="5GiB", help="Maximum GPU memory for auto device mapping when needed.")
    parser.add_argument("--cpu-memory", default="24GiB", help="Maximum CPU memory for auto device mapping when needed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-new-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint-every-n-batches", type=int, default=1)
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--feature-strategies", type=parse_csv, default=list(FEATURE_STRATEGIES))
    parser.add_argument("--probe-types", type=parse_csv, default=["majority", "previous_level", "linear", "mlp"])
    parser.add_argument("--eval-splits", type=parse_csv, default=["validation"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=1)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--probe-max-iter", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--wandb-project", default="llm-structured-semantic-generation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-tags", default="latent-level-probe")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic synthetic hidden states instead of loading an LLM.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint_every_n_batches <= 0:
        raise ValueError("--checkpoint-every-n-batches must be positive")
    unsupported_features = sorted(set(args.feature_strategies) - set(FEATURE_STRATEGIES))
    if unsupported_features:
        raise ValueError(f"unsupported_feature_strategies:{unsupported_features}")
    unsupported_probes = sorted(set(args.probe_types) - set(PROBE_TYPES))
    if unsupported_probes:
        raise ValueError(f"unsupported_probe_types:{unsupported_probes}")


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    resume_signature = {
        "train_file": str(args.train_file.resolve()),
        "validation_file": str(args.validation_file.resolve()),
        "test_file": str(args.test_file.resolve()) if args.include_test else None,
        "include_test": bool(args.include_test),
        "base_model_path": str(args.base_model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "gpu_memory": args.gpu_memory,
        "cpu_memory": args.cpu_memory,
        "max_samples": args.max_samples,
        "hidden_layer": args.hidden_layer,
        "feature_strategies": list(args.feature_strategies),
        "dry_run": bool(args.dry_run),
    }
    return {
        "run_id": args.run_id,
        "created_or_resumed_at": utc_now_iso(),
        "stage": args.stage,
        "batch_size": args.batch_size,
        "checkpoint_every_n_batches": args.checkpoint_every_n_batches,
        "probe_types": list(args.probe_types),
        "eval_splits": list(args.eval_splits),
        "train_split": args.train_split,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_layers": args.mlp_layers,
        "mlp_dropout": args.mlp_dropout,
        "probe_max_iter": args.probe_max_iter,
        "random_state": args.random_state,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "resume_signature": resume_signature,
    }


def init_wandb(args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    if find_spec("wandb") is None:
        raise RuntimeError("Weights & Biases requested but wandb is not installed. Use --wandb-mode disabled.")
    import wandb

    tags = parse_csv(args.wandb_tags)
    for tag in ("latent-level-probe", args.stage):
        if tag not in tags:
            tags.append(tag)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        mode=args.wandb_mode,
        dir=str(run_dir),
        tags=tags,
        config=config,
    )
    wandb.define_metric("extract/processed_units")
    wandb.define_metric("extract/*", step_metric="extract/processed_units")
    wandb.define_metric("probe/completed_probe_count")
    wandb.define_metric("probe/*", step_metric="probe/completed_probe_count")
    return run


def wandb_log(wandb_run: Any | None, payload: dict[str, Any]) -> None:
    if wandb_run is not None:
        wandb_run.log(payload)


def finish_wandb(wandb_run: Any | None, run_dir: Path) -> None:
    if wandb_run is None:
        return
    try:
        import wandb

        artifact = wandb.Artifact(name=f"{wandb_run.id}-latent-level-probe", type="latent-level-probe-run")
        for pattern in (
            "config.json",
            "state.json",
            "metrics.json",
            "line_metadata.jsonl",
            "features_*.jsonl",
            "probe_metrics_*.json",
            "probe_predictions_*.jsonl",
            "confusion_matrix_*.png",
        ):
            for path in run_dir.glob(pattern):
                artifact.add_file(str(path))
        wandb_run.log_artifact(artifact)
    finally:
        wandb_run.finish()


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    if find_spec("transformers") is None:
        raise RuntimeError("transformers is required for real extraction. Use --dry-run for smoke tests.")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for offset-based line alignment.")
    config = AutoConfig.from_pretrained(args.base_model_path, local_files_only=True, trust_remote_code=True)
    quantization_config = getattr(config, "quantization_config", None)
    is_bitsandbytes_4bit = isinstance(quantization_config, dict) and bool(quantization_config.get("load_in_4bit"))
    load_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
        "trust_remote_code": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available() and is_bitsandbytes_4bit:
        # Keep the local 4-bit checkpoint on the single CUDA device. Auto-dispatch
        # can reject this 6GB setup before loading, even though the quantized
        # checkpoint is intended to run on the laptop GPU.
        load_kwargs["device_map"] = {"": 0}
    else:
        max_memory: dict[Any, str] = {"cpu": args.cpu_memory}
        if torch.cuda.is_available():
            max_memory[0] = args.gpu_memory
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        **load_kwargs,
    )
    if args.adapter_path is not None:
        if find_spec("peft") is None:
            raise RuntimeError("peft is required to load --adapter-path")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def input_device(model: Any) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for module_name in ("model.embed_tokens", "transformer.wte", ""):
            device = device_map.get(module_name)
            if device not in (None, "disk", "cpu"):
                return str(device)
    return str(getattr(model, "device", "cpu"))


def extract_real_hidden(
    *,
    example: Any,
    tokenizer: Any,
    model: Any,
    hidden_layer: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    import torch

    encoded = tokenizer(
        example.full_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    offsets = [(int(start), int(end)) for start, end in encoded.pop("offset_mapping")[0].tolist()]
    device = input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    hidden = hidden_states[hidden_layer][0].detach().float().cpu().numpy()
    return hidden, offsets


def extract_rows(args: argparse.Namespace, run_dir: Path, rows: list[dict[str, Any]], wandb_run: Any | None, config: dict[str, Any]) -> None:
    completed = completed_unit_ids_from_chunks(run_dir)
    pending_rows = [row for row in rows if build_unit_id(row) not in completed]
    tokenizer = model = None
    if pending_rows and not args.dry_run:
        tokenizer, model = load_model_and_tokenizer(args)

    batches_since_checkpoint = 0
    for batch_start in range(0, len(pending_rows), args.batch_size):
        batch = pending_rows[batch_start : batch_start + args.batch_size]
        for row in batch:
            example = build_content_only_example(row)
            if args.dry_run:
                hidden, offsets = synthetic_hidden_for_example(example)
            else:
                hidden, offsets = extract_real_hidden(
                    example=example,
                    tokenizer=tokenizer,
                    model=model,
                    hidden_layer=args.hidden_layer,
                )
            features_by_strategy = features_for_hidden_states(
                example=example,
                hidden=hidden,
                offsets=offsets,
                feature_strategies=args.feature_strategies,
            )
            chunk = {
                "unit_id": example.unit_id,
                "sample_id": example.sample_id,
                "prompt_variant": example.prompt_variant,
                "split": example.split,
                "created_at": utc_now_iso(),
                "hidden_layer": args.hidden_layer,
                "feature_strategies": list(args.feature_strategies),
                "line_metadata": [metadata_from_span(span) for span in example.line_spans],
                "features_by_strategy": features_by_strategy,
            }
            atomic_write_json(chunk_path_for_unit(run_dir, example.unit_id), chunk)

        batches_since_checkpoint += 1
        completed_now = completed_unit_ids_from_chunks(run_dir)
        wandb_log(
            wandb_run,
            {
                "extract/processed_units": len(completed_now),
                "extract/remaining_units": max(len(rows) - len(completed_now), 0),
                "extract/current_batch_index": batch_start // args.batch_size,
            },
        )
        if batches_since_checkpoint >= args.checkpoint_every_n_batches:
            rebuild_aggregate_artifacts(run_dir, args.feature_strategies)
            write_state(run_dir, config=config, stage="extract", total_units=len(rows))
            batches_since_checkpoint = 0

    rebuild_aggregate_artifacts(run_dir, args.feature_strategies)
    write_state(run_dir, config=config, stage="extract", total_units=len(rows))


def write_confusion_matrix_png(path: Path, labels: list[int], matrix: list[list[int]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("Gold level")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp_path, dpi=150, format="png")
    plt.close(fig)
    tmp_path.replace(path)


def run_probes(args: argparse.Namespace, run_dir: Path, rows: list[dict[str, Any]], wandb_run: Any | None, config: dict[str, Any]) -> list[dict[str, Any]]:
    completed = set(completed_probe_ids(run_dir))
    metrics_rows: list[dict[str, Any]] = []
    for strategy in args.feature_strategies:
        for probe_type in args.probe_types:
            pid = probe_id(
                strategy=strategy,
                probe_type=probe_type,
                mlp_hidden_dim=args.mlp_hidden_dim,
                mlp_layers=args.mlp_layers,
                mlp_dropout=args.mlp_dropout,
            )
            if pid in completed:
                continue
            metrics = train_and_evaluate_probe(
                run_dir=run_dir,
                strategy=strategy,
                probe_type=probe_type,
                train_split=args.train_split,
                eval_splits=args.eval_splits,
                random_state=args.random_state,
                mlp_hidden_dim=args.mlp_hidden_dim,
                mlp_layers=args.mlp_layers,
                mlp_dropout=args.mlp_dropout,
                max_iter=args.probe_max_iter,
            )
            metrics_rows.append(metrics)
            for split in args.eval_splits:
                if split not in metrics:
                    continue
                split_metrics = metrics[split]
                write_confusion_matrix_png(
                    run_dir / f"confusion_matrix_{split}_{pid}.png",
                    split_metrics["labels"],
                    split_metrics["confusion_matrix"],
                    f"{pid} on {split}",
                )
                wandb_log(
                    wandb_run,
                    {
                        "probe/completed_probe_count": len(completed_probe_ids(run_dir)),
                        f"probe/{pid}/{split}/accuracy": split_metrics["accuracy"],
                        f"probe/{pid}/{split}/balanced_accuracy": split_metrics["balanced_accuracy"],
                        f"probe/{pid}/{split}/macro_f1": split_metrics["macro_f1"],
                        f"probe/{pid}/{split}/weighted_f1": split_metrics["weighted_f1"],
                        f"probe/{pid}/{split}/level_mae": split_metrics["level_mae"],
                    },
                )
            completed.add(pid)
            write_state(
                run_dir,
                config=config,
                stage="probe",
                total_units=len(rows),
                completed_probe_configs=sorted(completed),
            )
    return metrics_rows


def write_final_metrics(run_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    probe_metric_paths = sorted(run_dir.glob("probe_metrics_*.json"))
    probes = [json.loads(path.read_text(encoding="utf-8")) for path in probe_metric_paths]
    summary = {
        "run_id": args.run_id,
        "completed_at": utc_now_iso(),
        "stage": args.stage,
        "row_count": len(rows),
        "completed_row_count": len(completed_unit_ids_from_chunks(run_dir)),
        "feature_strategies": list(args.feature_strategies),
        "probe_count": len(probes),
        "probe_ids": [row["probe_id"] for row in probes],
        "probes": probes,
    }
    write_json(run_dir / "metrics.json", summary)
    write_state(
        run_dir,
        config=config,
        stage=args.stage,
        total_units=len(rows),
        completed_probe_configs=completed_probe_ids(run_dir),
        status="completed",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_dir = (args.output_dir / args.run_id).resolve()
    config = build_config(args)
    try:
        init_or_validate_config(run_dir, config, force_new_run=args.force_new_run)
    except RunCompatibilityError as exc:
        raise SystemExit(str(exc)) from exc

    input_files = [args.train_file, args.validation_file]
    if args.include_test:
        input_files.append(args.test_file)
    rows = read_sft_rows(input_files, max_samples=args.max_samples)
    if not rows:
        raise SystemExit("No SFT rows found.")
    write_state(run_dir, config=config, stage=args.stage, total_units=len(rows))
    wandb_run = init_wandb(args, config, run_dir)
    try:
        if args.stage in {"extract", "all"}:
            extract_rows(args, run_dir, rows, wandb_run, config)
        else:
            rebuild_aggregate_artifacts(run_dir, args.feature_strategies)
        if args.stage in {"probe", "all"}:
            run_probes(args, run_dir, rows, wandb_run, config)
        write_final_metrics(run_dir, args, rows, config)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "row_count": len(rows),
                    "completed_row_count": len(completed_unit_ids_from_chunks(run_dir)),
                    "completed_probe_ids": completed_probe_ids(run_dir),
                    "metrics": str(run_dir / "metrics.json"),
                },
                ensure_ascii=False,
            )
        )
    finally:
        finish_wandb(wandb_run, run_dir)


if __name__ == "__main__":
    main()
