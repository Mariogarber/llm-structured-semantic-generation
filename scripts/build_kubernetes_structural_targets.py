from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json, write_jsonl
from llm_structured_semantic_generation.structure import validate_round_trip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build line-and-level structural targets for Kubernetes v1."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "dataset_train_ready.jsonl",
        help="Prompt-variant JSONL produced by the Kubernetes preprocessor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "dataset_structural_targets.jsonl",
        help="Output JSONL with derived block targets.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "structural_targets_report.json",
        help="Output JSON quality report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_rows = read_jsonl(args.input)
    output_rows = []
    split_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    total_blocks = 0
    max_blocks = 0
    errors: list[dict[str, object]] = []

    for row in input_rows:
        round_trip = validate_round_trip(row["target_yaml_normalized"])
        status = "ok" if round_trip.yaml_parse_ok and round_trip.semantics_preserved else "reject"
        blocks = [block.to_dict() for block in round_trip.blocks]
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "prompt_variant": row["prompt_variant"],
                "prompt_policy": row["prompt_policy"],
                "prompt_text": row["prompt_text"],
                "target_yaml_normalized": row["target_yaml_normalized"],
                "split": row["split"],
                "validation_status": row["validation_status"],
                "leakage_group": row["leakage_group"],
                "blocks": blocks,
                "block_count": len(blocks),
                "structural_target_status": status,
                "structural_target_errors": list(round_trip.errors),
            }
        )
        split_counts[row["split"]] += 1
        status_counts[status] += 1
        total_blocks += len(blocks)
        max_blocks = max(max_blocks, len(blocks))
        if status != "ok":
            errors.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_variant": row["prompt_variant"],
                    "errors": list(round_trip.errors),
                }
            )

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "row_count": len(output_rows),
        "split_counts": dict(split_counts),
        "status_counts": dict(status_counts),
        "ready_for_baseline": status_counts.get("reject", 0) == 0 and len(output_rows) > 0,
        "average_block_count": round(total_blocks / len(output_rows), 4) if output_rows else 0,
        "max_block_count": max_blocks,
        "error_examples": errors[:10],
    }

    write_jsonl(args.output, output_rows)
    write_json(args.report, report)
    print(report)


if __name__ == "__main__":
    main()

