from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json, write_jsonl
from llm_structured_semantic_generation.sft_serialization import build_sft_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed SFT JSONL rows from Kubernetes structural targets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "dataset_structural_targets.jsonl",
        help="Structural target JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft",
        help="Directory where train/validation/test JSONL files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    by_split: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    skipped: list[dict[str, str]] = []

    for row in rows:
        if row["structural_target_status"] != "ok":
            skipped.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_variant": row["prompt_variant"],
                    "reason": "structural_target_not_ok",
                }
            )
            continue
        split = row["split"]
        if split not in by_split:
            skipped.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_variant": row["prompt_variant"],
                    "reason": f"unknown_split:{split}",
                }
            )
            continue
        by_split[split].append(build_sft_row(row))

    for split, split_rows in by_split.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_rows)

    split_counts = Counter({split: len(split_rows) for split, split_rows in by_split.items()})
    report = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "split_counts": dict(split_counts),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:10],
        "serialization": "blocks_tsv_v1",
        "ready_for_sft": len(skipped) == 0 and split_counts["train"] > 0,
    }
    write_json(args.output_dir / "sft_dataset_report.json", report)
    print(report)


if __name__ == "__main__":
    main()

