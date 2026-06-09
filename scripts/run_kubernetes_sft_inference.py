from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json
from llm_structured_semantic_generation.evaluation import (
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)
from llm_structured_semantic_generation.resumable_run import ResumableRun, utc_now_iso
from llm_structured_semantic_generation.sft_serialization import BLOCKS_TSV_V1
from llm_structured_semantic_generation.structure import blocks_to_yaml
from build_kubernetes_dpo_candidates import (
    load_sft_model_and_tokenizer,
    maybe_clear_cuda_cache,
    resolve_checkpoint,
    resolve_checkpoint_root_and_adapter,
)
from train_kubernetes_sft import (
    SERIALIZED_SFT,
    build_unit_id,
    extract_blocks_tsv_prediction,
    generate_validation_completion,
    inspect_model_path,
    load_env_file,
    load_sft_rows,
    positive_int,
)


DEFAULT_SFT_RUN_DIR = REPO_ROOT / "results" / "sft_kubernetes_v1" / "serialized-sft-a-v1-20260505-171226"
DEFAULT_TEST_FILE = REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "test.jsonl"
DEFAULT_BASE_MODEL_PATH = REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "sft_kubernetes_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run serialized_sft Kubernetes v1 inference on an SFT split and compute parser-facing metrics."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--sft-run-dir", type=Path, default=DEFAULT_SFT_RUN_DIR)
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Use best, latest, a checkpoint directory name, or a checkpoint/adapter path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume or create output-dir/run-id. Defaults to a UTC timestamped serialized SFT test run.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Number of predictions to persist per batch. Use 1 for maximal resumability.",
    )
    parser.add_argument("--max-samples", type=positive_int, default=None)
    parser.add_argument("--max-new-tokens", type=positive_int, default=512)
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


def build_resume_signature(
    args: argparse.Namespace,
    *,
    test_file: Path,
    sft_run_dir: Path,
    checkpoint_root: Path,
    adapter_path: Path,
    base_model_path: Path,
) -> dict[str, Any]:
    return {
        "stage": "sft_inference_v1",
        "model_variant": SERIALIZED_SFT,
        "serialization": BLOCKS_TSV_V1,
        "test_file": project_path(test_file),
        "sft_run_dir": project_path(sft_run_dir),
        "checkpoint_root": project_path(checkpoint_root),
        "adapter_path": project_path(adapter_path),
        "base_model_path": project_path(base_model_path),
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "target_contract": "prompt -> blocks_tsv_v1 -> parser -> YAML",
    }


def build_prediction_row(
    *,
    row: dict[str, Any],
    checkpoint_root: Path,
    completion: dict[str, Any],
    generation_error: BaseException | None,
) -> dict[str, Any]:
    raw_output = completion.get("raw_text") if generation_error is None else None
    parse_errors: list[str] = []
    predicted_blocks: list[dict[str, Any]] = []
    reconstructed_yaml = ""
    reconstruction_errors: list[str] = []
    evaluation = None

    if generation_error is None and raw_output is not None:
        try:
            predicted_blocks = extract_blocks_tsv_prediction(raw_output)
        except ValueError as exc:
            parse_errors.append(f"structured_output_parse_error:blocks_tsv_v1:{exc.__class__.__name__}:{exc}")

        if predicted_blocks:
            reconstruction = blocks_to_yaml(predicted_blocks, recovery_mode="strict")
            reconstructed_yaml = reconstruction.yaml_text
            reconstruction_errors = list(reconstruction.errors)
            evaluation = evaluate_blocks_prediction(
                str(row["target_yaml_normalized"]),
                predicted_blocks,
                recovery_mode="strict",
                prompt_text=str(row["prompt"]),
            ).to_dict()
    else:
        parse_errors.append(
            "generation_error:"
            f"{type(generation_error).__name__ if generation_error is not None else 'unknown'}:"
            f"{generation_error}"
        )

    return {
        "unit_id": build_unit_id(row),
        "sample_id": row["sample_id"],
        "prompt_variant": row["prompt_variant"],
        "split": row["split"],
        "prompt": row["prompt"],
        "reference_yaml": row["target_yaml_normalized"],
        "checkpoint": project_path(checkpoint_root),
        "output_format": BLOCKS_TSV_V1,
        "generation_ok": generation_error is None,
        "generation_error_type": type(generation_error).__name__ if generation_error is not None else None,
        "generation_error": str(generation_error) if generation_error is not None else None,
        "raw_model_output": raw_output,
        "generated_token_count": completion.get("generated_token_count"),
        "predicted_blocks": predicted_blocks,
        "reconstructed_yaml": reconstructed_yaml,
        "parser_errors": parse_errors,
        "reconstruction_errors": reconstruction_errors,
        "structured_output_parse_success": evaluation is not None,
        "evaluation": evaluation,
        "generated_at": utc_now_iso(),
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
        "stage": "sft_inference_v1",
        "model_variant": SERIALIZED_SFT,
        "serialization": BLOCKS_TSV_V1,
        "split": config["split"],
        "source_sft_run_id": config["source_sft_run_id"],
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


def write_dry_run_metrics(run_dir: Path, config: dict[str, Any]) -> None:
    write_json(
        run_dir / "metrics.json",
        {
            "dry_run": True,
            "row_count": config["row_count"],
            "ready_for_full_run": config["model_checks"]["ready_for_full_run"],
            "model_warnings": config["model_checks"]["warnings"],
            "checkpoint_root": config["checkpoint_root"],
            "adapter_path": config["adapter_path"],
        },
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"serialized-sft-a-v1-test-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    load_env_file()
    test_file = resolve_repo_path(args.test_file)
    sft_run_dir = resolve_repo_path(args.sft_run_dir)
    base_model_path = resolve_repo_path(args.base_model_path)
    checkpoint_root, adapter_path = resolve_checkpoint_root_and_adapter(
        resolve_checkpoint(sft_run_dir, args.checkpoint)
    )
    rows = load_sft_rows(test_file, max_samples=args.max_samples, split_name="test")
    run_dir = resolve_repo_path(args.output_dir) / run_id
    source_sft_run_id = sft_run_dir.name

    config = {
        "run_id": run_id,
        "stage": "sft_inference_v1",
        "created_at": utc_now_iso(),
        "model_variant": SERIALIZED_SFT,
        "serialization": BLOCKS_TSV_V1,
        "split": "test",
        "test_file": project_path(test_file),
        "source_sft_run_id": source_sft_run_id,
        "sft_run_dir": project_path(sft_run_dir),
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
        "target_contract": "prompt -> blocks_tsv_v1 -> parser -> YAML",
        "dry_run": args.dry_run,
        "model_checks": inspect_model_path(base_model_path),
        "artifacts": {
            "predictions": "predictions.jsonl",
            "metrics": "metrics.json",
            "state": "state.json",
            "config": "config.json",
        },
        "resume_signature": build_resume_signature(
            args,
            test_file=test_file,
            sft_run_dir=sft_run_dir,
            checkpoint_root=checkpoint_root,
            adapter_path=adapter_path,
            base_model_path=base_model_path,
        ),
    }

    if args.dry_run:
        write_json(run_dir / "config.json", config)
        write_dry_run_metrics(run_dir, config)
        print(
            {
                "dry_run": True,
                "output_dir": str(run_dir),
                "row_count": len(rows),
                "ready_for_full_run": config["model_checks"]["ready_for_full_run"],
                "model_warnings": config["model_checks"]["warnings"],
            }
        )
        return

    if not config["model_checks"]["ready_for_full_run"]:
        raise RuntimeError(f"SFT inference is not ready. See model_checks: {config['model_checks']}")

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

    tokenizer, model = load_sft_model_and_tokenizer(
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
            completion: dict[str, Any] = {}
            generation_error: BaseException | None = None
            try:
                completion = generate_validation_completion(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=str(row["prompt"]),
                    max_new_tokens=args.max_new_tokens,
                )
            except RuntimeError as exc:
                generation_error = exc
                maybe_clear_cuda_cache()

            batch_predictions.append(
                build_prediction_row(
                    row=row,
                    checkpoint_root=checkpoint_root,
                    completion=completion,
                    generation_error=generation_error,
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
