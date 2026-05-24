from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.evaluation import (  # noqa: E402
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)
from llm_structured_semantic_generation.sft_serialization import (  # noqa: E402
    deserialize_training_blocks,
)


DEEP_LEVELS = (5, 6, 7, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute diagnostic metrics for the first Kubernetes two_head_sft run."
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
        "--test-file",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "test.jsonl",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "two_head_sft_kubernetes_v1"
        / "two-head-sft-v1-20260516"
        / "validation_predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "two_head_diagnostics_v1"
        / "two-head-sft-v1-20260516",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['prompt_variant']}"


def load_split_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        blocks = deserialize_training_blocks(str(row["target"]))
        rows[unit_id(row)] = {
            "row": row,
            "blocks": [block.to_dict() for block in blocks],
            "levels": [int(block.level) for block in blocks],
            "line_texts": [str(block.line_text) for block in blocks],
        }
    return rows


def level_distribution(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[int] = Counter()
    for item in rows.values():
        counter.update(item["levels"])
    total = sum(counter.values())
    return {
        "total_lines": total,
        "counts": {str(level): counter[level] for level in sorted(counter)},
        "rates": {
            str(level): (counter[level] / total if total else 0.0)
            for level in sorted(counter)
        },
    }


def predicted_level_distribution(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[int] = Counter()
    for prediction in predictions:
        for block in prediction.get("predicted_blocks", []):
            counter[int(block["level"])] += 1
    total = sum(counter.values())
    return {
        "total_lines": total,
        "counts": {str(level): counter[level] for level in sorted(counter)},
        "rates": {
            str(level): (counter[level] / total if total else 0.0)
            for level in sorted(counter)
        },
    }


def line_features(texts: list[str]) -> dict[str, bool]:
    joined = "\n".join(texts)
    lowered = joined.lower()
    return {
        "has_container": "containers:" in joined or "initContainers:" in joined,
        "has_volume": "volume" in lowered,
        "has_command": "command:" in joined or "/bin/sh" in joined or "-c" in texts,
    }


def position_aligned_confusion(
    *,
    predictions: list[dict[str, Any]],
    validation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    gold_counts: Counter[int] = Counter()
    pred_counts: Counter[int] = Counter()
    paired_positions = 0
    total_gold_positions = 0
    total_pred_positions = 0

    for prediction in predictions:
        item = validation_rows[prediction["unit_id"]]
        gold_levels = item["levels"]
        pred_levels = [int(block["level"]) for block in prediction.get("predicted_blocks", [])]
        total_gold_positions += len(gold_levels)
        total_pred_positions += len(pred_levels)
        for gold, pred in zip(gold_levels, pred_levels):
            confusion[int(gold)][int(pred)] += 1
            gold_counts[int(gold)] += 1
            pred_counts[int(pred)] += 1
            paired_positions += 1

    per_level: dict[str, dict[str, float | int]] = {}
    for level in sorted(set(gold_counts) | set(pred_counts) | set(DEEP_LEVELS)):
        true_positive = confusion[level][level]
        support = gold_counts[level]
        predicted = pred_counts[level]
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision or recall
            else 0.0
        )
        per_level[str(level)] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    deep_support = sum(gold_counts[level] for level in DEEP_LEVELS)
    deep_true_positive = sum(confusion[level][level] for level in DEEP_LEVELS)
    deep_predicted = sum(pred_counts[level] for level in DEEP_LEVELS)
    deep_off_by_one = sum(
        count
        for gold in DEEP_LEVELS
        for pred, count in confusion[gold].items()
        if abs(gold - pred) <= 1
    )
    compressed_to_0_4 = sum(
        count
        for gold in DEEP_LEVELS
        for pred, count in confusion[gold].items()
        if pred <= 4
    )

    return {
        "paired_positions": paired_positions,
        "total_gold_positions": total_gold_positions,
        "total_pred_positions": total_pred_positions,
        "confusion": {
            str(gold): {str(pred): count for pred, count in sorted(preds.items())}
            for gold, preds in sorted(confusion.items())
        },
        "per_level": per_level,
        "deep_levels": {
            "levels": list(DEEP_LEVELS),
            "support": deep_support,
            "predicted": deep_predicted,
            "true_positive": deep_true_positive,
            "exact_recall": deep_true_positive / deep_support if deep_support else 0.0,
            "off_by_one_recall": deep_off_by_one / deep_support if deep_support else 0.0,
            "compressed_to_0_4": compressed_to_0_4,
            "compressed_to_0_4_rate": compressed_to_0_4 / deep_support if deep_support else 0.0,
        },
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parseability_groups(
    *,
    predictions: list[dict[str, Any]],
    validation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"parseable": [], "failed": []}
    for prediction in predictions:
        item = validation_rows[prediction["unit_id"]]
        pred_blocks = prediction.get("predicted_blocks", [])
        pred_levels = [int(block["level"]) for block in pred_blocks]
        features = line_features(item["line_texts"])
        record = {
            "target_lines": len(item["blocks"]),
            "pred_lines": len(pred_blocks),
            "line_delta": len(pred_blocks) - len(item["blocks"]),
            "target_max_level": max(item["levels"]) if item["levels"] else 0,
            "pred_max_level": max(pred_levels) if pred_levels else 0,
            **features,
        }
        key = "parseable" if prediction["evaluation"]["yaml_parse_ok"] else "failed"
        groups[key].append(record)

    summary: dict[str, Any] = {}
    for key, rows in groups.items():
        summary[key] = {
            "count": len(rows),
            "target_lines_mean": mean([row["target_lines"] for row in rows]),
            "pred_lines_mean": mean([row["pred_lines"] for row in rows]),
            "line_delta_mean": mean([row["line_delta"] for row in rows]),
            "target_max_level_mean": mean([row["target_max_level"] for row in rows]),
            "pred_max_level_mean": mean([row["pred_max_level"] for row in rows]),
            "has_container_rate": mean([1.0 if row["has_container"] else 0.0 for row in rows]),
            "has_volume_rate": mean([1.0 if row["has_volume"] else 0.0 for row in rows]),
            "has_command_rate": mean([1.0 if row["has_command"] else 0.0 for row in rows]),
        }
    return summary


def evaluation_from_dict(payload: dict[str, Any]) -> StructuralEvaluation:
    return StructuralEvaluation(**payload)


def attach_reference_position_levels(
    *,
    predicted_content_blocks: list[dict[str, Any]],
    reference_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_key = {
        (int(block["document_index"]), int(block["line_index"])): int(block["level"])
        for block in reference_blocks
    }
    unaligned = 0
    blocks: list[dict[str, Any]] = []
    for index, block in enumerate(predicted_content_blocks):
        key = (int(block["document_index"]), int(block["line_index"]))
        if key in by_key:
            level = by_key[key]
        elif index < len(reference_blocks):
            level = int(reference_blocks[index]["level"])
            unaligned += 1
        else:
            level = 0
            unaligned += 1
        blocks.append(
            {
                "document_index": int(block["document_index"]),
                "line_index": int(block["line_index"]),
                "level": level,
                "line_text": str(block["line_text"]),
            }
        )
    return blocks, unaligned


def attach_predicted_position_levels_to_reference_content(
    *,
    predicted_blocks: list[dict[str, Any]],
    reference_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    missing = 0
    blocks: list[dict[str, Any]] = []
    for index, block in enumerate(reference_blocks):
        if index < len(predicted_blocks):
            level = int(predicted_blocks[index]["level"])
        else:
            level = 0
            missing += 1
        blocks.append(
            {
                "document_index": int(block["document_index"]),
                "line_index": int(block["line_index"]),
                "level": level,
                "line_text": str(block["line_text"]),
            }
        )
    return blocks, missing


def run_counterfactuals(
    *,
    predictions: list[dict[str, Any]],
    validation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    actual = [evaluation_from_dict(prediction["evaluation"]) for prediction in predictions]
    reference_content_reference_levels: list[StructuralEvaluation] = []
    generated_content_gold_levels: list[StructuralEvaluation] = []
    gold_content_predicted_levels: list[StructuralEvaluation] = []
    line_count_matched_actual: list[StructuralEvaluation] = []
    line_count_matched_gold_content_predicted_levels: list[StructuralEvaluation] = []
    generated_gold_unaligned: list[int] = []
    gold_pred_missing: list[int] = []

    for prediction in predictions:
        item = validation_rows[prediction["unit_id"]]
        row = item["row"]
        reference_content_reference_levels.append(
            evaluate_blocks_prediction(
                str(row["target_yaml_normalized"]),
                item["blocks"],
                prompt_text=str(row["prompt"]),
            )
        )
        generated_blocks, unaligned = attach_reference_position_levels(
            predicted_content_blocks=prediction.get("predicted_content_blocks", []),
            reference_blocks=item["blocks"],
        )
        generated_gold_unaligned.append(unaligned)
        generated_content_gold_levels.append(
            evaluate_blocks_prediction(
                str(row["target_yaml_normalized"]),
                generated_blocks,
                prompt_text=str(row["prompt"]),
            )
        )

        gold_blocks, missing = attach_predicted_position_levels_to_reference_content(
            predicted_blocks=prediction.get("predicted_blocks", []),
            reference_blocks=item["blocks"],
        )
        gold_pred_missing.append(missing)
        gold_content_predicted_levels.append(
            evaluate_blocks_prediction(
                str(row["target_yaml_normalized"]),
                gold_blocks,
                prompt_text=str(row["prompt"]),
            )
        )
        predicted_blocks = prediction.get("predicted_blocks", [])
        if len(predicted_blocks) == len(item["blocks"]):
            line_count_matched_actual.append(evaluation_from_dict(prediction["evaluation"]))
            line_count_matched_gold_content_predicted_levels.append(
                evaluate_blocks_prediction(
                    str(row["target_yaml_normalized"]),
                    gold_blocks,
                    prompt_text=str(row["prompt"]),
                )
            )

    return {
        "reference_content_reference_levels_sanity": summarize_evaluations(
            reference_content_reference_levels
        ),
        "actual_generated_content_predicted_levels": summarize_evaluations(actual),
        "generated_content_reference_position_levels": summarize_evaluations(
            generated_content_gold_levels
        ),
        "reference_content_predicted_position_levels": summarize_evaluations(
            gold_content_predicted_levels
        ),
        "line_count_matched": {
            "row_count": len(line_count_matched_actual),
            "actual_generated_content_predicted_levels": summarize_evaluations(
                line_count_matched_actual
            ),
            "reference_content_predicted_position_levels": summarize_evaluations(
                line_count_matched_gold_content_predicted_levels
            ),
        },
        "alignment_caveats": {
            "generated_content_reference_position_levels_total_unaligned": sum(
                generated_gold_unaligned
            ),
            "generated_content_reference_position_levels_mean_unaligned": mean(
                [float(value) for value in generated_gold_unaligned]
            ),
            "reference_content_predicted_position_levels_total_missing": sum(
                gold_pred_missing
            ),
            "reference_content_predicted_position_levels_mean_missing": mean(
                [float(value) for value in gold_pred_missing]
            ),
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    train_rows = load_split_rows(args.train_file)
    validation_rows = load_split_rows(args.validation_file)
    test_rows = load_split_rows(args.test_file)
    predictions = read_jsonl(args.predictions_file)

    diagnostics = {
        "inputs": {
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "test_file": str(args.test_file),
            "predictions_file": str(args.predictions_file),
        },
        "row_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
            "predictions": len(predictions),
        },
        "target_level_distributions": {
            "train": level_distribution(train_rows),
            "validation": level_distribution(validation_rows),
            "test": level_distribution(test_rows),
        },
        "predicted_level_distribution": predicted_level_distribution(predictions),
        "position_aligned_level_confusion": position_aligned_confusion(
            predictions=predictions,
            validation_rows=validation_rows,
        ),
        "parseability_groups": parseability_groups(
            predictions=predictions,
            validation_rows=validation_rows,
        ),
        "counterfactuals": run_counterfactuals(
            predictions=predictions,
            validation_rows=validation_rows,
        ),
    }
    write_json(args.output_dir / "diagnostic_metrics.json", diagnostics)
    print(json.dumps({"output": str(args.output_dir / "diagnostic_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
