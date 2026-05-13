from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json, write_jsonl
from llm_structured_semantic_generation.evaluation import (
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute SFT validation metrics from persisted validation_predictions.jsonl "
            "without rerunning model inference."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="SFT run directory containing config.json and validation_predictions.jsonl.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=None,
        help="Optional validation JSONL override. Defaults to config.json validation_file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions JSONL override. Defaults to <run-dir>/validation_predictions.jsonl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint name to filter when the predictions file contains multiple checkpoints.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <run-dir>/validation_metrics_recomputed.json.",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=None,
        help="Optional output JSONL path. Defaults to <run-dir>/validation_predictions_recomputed.jsonl.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional Markdown report path. Defaults to <run-dir>/validation_metrics_recomputed_report.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def build_unit_id(row: dict[str, Any]) -> str:
    sample_id = row.get("sample_id")
    prompt_variant = row.get("prompt_variant")
    if not isinstance(sample_id, str) or not isinstance(prompt_variant, str):
        raise ValueError(f"Cannot build unit id from row: {row}")
    return f"{sample_id}::{prompt_variant}"


def load_validation_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {build_unit_id(row): row for row in rows}


def infer_checkpoint(predictions: list[dict[str, Any]], old_metrics: dict[str, Any] | None) -> str:
    if old_metrics and isinstance(old_metrics.get("checkpoint"), str):
        return str(old_metrics["checkpoint"])
    checkpoints = [row.get("checkpoint") for row in predictions if isinstance(row.get("checkpoint"), str)]
    if checkpoints:
        return str(checkpoints[-1])
    return "unknown"


def derive_metrics(
    *,
    run_id: str,
    predictions: list[dict[str, Any]],
    checkpoint: str,
    model_variant: str,
    serialization: str,
) -> dict[str, Any]:
    evaluated_results = [
        StructuralEvaluation(**row["evaluation"])
        for row in predictions
        if row.get("evaluation") is not None
    ]
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "row_count": len(predictions),
        "evaluated_count": len(evaluated_results),
        "model_variant": model_variant,
        "serialization": serialization,
        "structured_output_parse_success_rate": len(evaluated_results) / len(predictions) if predictions else 0.0,
        "source_predictions": "validation_predictions.jsonl",
        "perplexity_available": False,
    }
    metrics.update(summarize_evaluations(evaluated_results))
    return metrics


def metric_line(name: str, old_metrics: dict[str, Any] | None, new_metrics: dict[str, Any]) -> str:
    old_value = None if old_metrics is None else old_metrics.get(name)
    return f"- `{name}`: old={old_value} new={new_metrics.get(name)}"


def render_report(
    *,
    run_dir: Path,
    validation_file: Path,
    predictions_file: Path,
    old_metrics: dict[str, Any] | None,
    new_metrics: dict[str, Any],
) -> str:
    headline_metrics = [
        "structured_output_parse_success_rate",
        "yaml_parse_success_rate",
        "parsed_equal_rate",
        "block_parse_success_rate",
        "document_count_match_rate",
        "line_count_match_rate",
        "average_line_text_f1",
        "average_level_mae",
        "average_semantic_key_f1",
        "average_prompt_requirement_f1",
        "prompt_requirement_exact_match_rate",
        "average_required_field_complete_resource_rate",
        "required_field_complete_sample_rate",
        "average_kubernetes_domain_validity_score",
        "kubernetes_domain_gate_pass_rate",
        "average_kubernetes_domain_validity_level",
        "kubernetes_level_0_pass_rate",
        "kubernetes_level_1_pass_rate",
        "kubernetes_level_2_pass_rate",
        "kubernetes_level_3_pass_rate",
        "kubernetes_level_4_pass_rate",
        "kubernetes_level_5_pass_rate",
        "average_bleu_score",
        "average_rouge1_f1",
        "average_rouge2_f1",
        "average_rougeL_f1",
        "average_perplexity",
        "perplexity_available_rate",
    ]
    lines = [
        "# SFT Validation Offline Recomputed Metrics Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Validation file: `{validation_file}`",
        f"- Predictions file: `{predictions_file}`",
        f"- Row count: `{new_metrics.get('row_count')}`",
        f"- Evaluated count: `{new_metrics.get('evaluated_count')}`",
        f"- Checkpoint: `{new_metrics.get('checkpoint')}`",
        "",
        "## Headline metrics",
        "",
    ]
    lines.extend(metric_line(name, old_metrics, new_metrics) for name in headline_metrics)
    lines.extend(
        [
            "",
            "## Error profile",
            "",
            f"- Kubernetes domain error counts: `{new_metrics.get('kubernetes_domain_error_counts')}`",
            "",
            "## Notes",
            "",
            "- These metrics were recomputed offline from persisted SFT validation predictions; no new model inference was run.",
            "- BLEU, ROUGE, and perplexity remain auxiliary signals and are not used for the Kubernetes domain gate.",
            "- Perplexity is null here because the offline artifact does not contain model log-probabilities.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config_path = run_dir / "config.json"
    config = load_json(config_path)
    old_metrics = load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None

    validation_file = resolve_repo_path(args.validation_file or Path(str(config["validation_file"])))
    predictions_file = (args.predictions or (run_dir / "validation_predictions.jsonl")).resolve()
    validation_index = load_validation_index(validation_file)
    predictions = read_jsonl(predictions_file, allow_truncated_last_line=True)

    checkpoint = args.checkpoint or infer_checkpoint(predictions, old_metrics)
    selected_predictions = [
        row for row in predictions if args.checkpoint is None or row.get("checkpoint") == args.checkpoint
    ]

    recomputed_predictions: list[dict[str, Any]] = []
    for prediction in selected_predictions:
        unit_id = prediction.get("unit_id") or build_unit_id(prediction)
        reference_row = validation_index.get(unit_id)
        if reference_row is None:
            raise KeyError(f"Reference row not found for unit_id={unit_id}")

        predicted_blocks = list(prediction.get("predicted_blocks") or [])
        prompt_text = str(reference_row.get("prompt") or prediction.get("prompt") or "")
        evaluation = (
            evaluate_blocks_prediction(
                str(reference_row["target_yaml_normalized"]),
                predicted_blocks,
                prompt_text=prompt_text,
            ).to_dict()
            if predicted_blocks
            else None
        )

        recomputed_row = dict(prediction)
        recomputed_row["evaluation"] = evaluation
        recomputed_row["evaluation_source"] = "offline_recomputed"
        recomputed_predictions.append(recomputed_row)

    metrics = derive_metrics(
        run_id=str(config.get("run_id", run_dir.name)),
        predictions=recomputed_predictions,
        checkpoint=checkpoint,
        model_variant=str(config.get("model_variant", "serialized_sft")),
        serialization=str(config.get("serialization", "blocks_tsv_v1")),
    )

    output_metrics = (args.output_metrics or (run_dir / "validation_metrics_recomputed.json")).resolve()
    output_predictions = (args.output_predictions or (run_dir / "validation_predictions_recomputed.jsonl")).resolve()
    output_report = (args.output_report or (run_dir / "validation_metrics_recomputed_report.md")).resolve()

    write_json(output_metrics, metrics)
    write_jsonl(output_predictions, recomputed_predictions)
    output_report.write_text(
        render_report(
            run_dir=run_dir,
            validation_file=validation_file,
            predictions_file=predictions_file,
            old_metrics=old_metrics,
            new_metrics=metrics,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "validation_file": str(validation_file),
                "predictions_file": str(predictions_file),
                "output_metrics": str(output_metrics),
                "output_predictions": str(output_predictions),
                "output_report": str(output_report),
                "row_count": metrics["row_count"],
                "evaluated_count": metrics["evaluated_count"],
                "average_kubernetes_domain_validity_score": metrics.get("average_kubernetes_domain_validity_score"),
                "kubernetes_domain_gate_pass_rate": metrics.get("kubernetes_domain_gate_pass_rate"),
                "average_bleu_score": metrics.get("average_bleu_score"),
                "average_rougeL_f1": metrics.get("average_rougeL_f1"),
                "perplexity_available_rate": metrics.get("perplexity_available_rate"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
