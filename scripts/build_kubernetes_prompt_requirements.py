from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json, write_jsonl
from llm_structured_semantic_generation.prompt_requirements import (
    extract_prompt_requirements,
    evaluate_prompt_requirements,
    evaluate_required_fields,
    summarize_prompt_requirement_atoms,
)
from llm_structured_semantic_generation.structure import parse_yaml_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prompt requirement artifacts for Kubernetes v1."
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
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "prompt_requirements.jsonl",
        help="Output JSONL with extracted prompt requirements.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "prompt_requirements_report.json",
        help="Output JSON summary report.",
    )
    return parser.parse_args()


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def average_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def rate(values: list[bool]) -> float:
    return average([1.0 if value else 0.0 for value in values])


def optional_rate(values: list[bool | None]) -> float | None:
    present = [1.0 if value else 0.0 for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def main() -> None:
    args = parse_args()
    input_rows = read_jsonl(args.input)
    output_rows = []
    split_counts: Counter[str] = Counter()
    all_atom_rows = []
    prompt_evaluations = []
    required_field_evaluations = []

    for row in input_rows:
        prompt_requirements = extract_prompt_requirements(row["prompt_text"])
        reference_documents = parse_yaml_documents(row["target_yaml_normalized"])
        prompt_evaluation = evaluate_prompt_requirements(row["prompt_text"], reference_documents)
        required_field_evaluation = evaluate_required_fields(reference_documents)

        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "prompt_variant": row["prompt_variant"],
                "prompt_policy": row["prompt_policy"],
                "prompt_text": row["prompt_text"],
                "split": row["split"],
                "prompt_requirement_supported": prompt_evaluation.prompt_requirement_supported,
                "prompt_requirement_count": prompt_evaluation.prompt_requirement_count,
                "prompt_requirement_categories": sorted({atom.category for atom in prompt_requirements}),
                "prompt_requirements": [atom.to_dict() for atom in prompt_requirements],
                "reference_prompt_requirement_evaluation": prompt_evaluation.to_dict(),
                "reference_required_field_evaluation": required_field_evaluation.to_dict(),
            }
        )
        split_counts[row["split"]] += 1
        all_atom_rows.append(prompt_requirements)
        prompt_evaluations.append(prompt_evaluation)
        required_field_evaluations.append(required_field_evaluation)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "row_count": len(output_rows),
        "split_counts": dict(split_counts),
        "prompt_requirement_support_rate": rate(
            [item.prompt_requirement_supported for item in prompt_evaluations]
        ),
        "average_prompt_requirement_count": average(
            [float(item.prompt_requirement_count) for item in prompt_evaluations if item.prompt_requirement_supported]
        ),
        "prompt_requirement_exact_match_rate_against_reference": optional_rate(
            [item.prompt_requirement_exact_match for item in prompt_evaluations]
        ),
        "average_prompt_requirement_precision_against_reference": average_optional(
            [item.prompt_requirement_precision for item in prompt_evaluations]
        ),
        "average_prompt_requirement_recall_against_reference": average_optional(
            [item.prompt_requirement_recall for item in prompt_evaluations]
        ),
        "average_prompt_requirement_f1_against_reference": average_optional(
            [item.prompt_requirement_f1 for item in prompt_evaluations]
        ),
        "average_reference_required_field_presence_rate": average_optional(
            [item.required_field_presence_rate for item in required_field_evaluations]
        ),
        "average_reference_required_field_complete_resource_rate": average_optional(
            [item.required_field_complete_resource_rate for item in required_field_evaluations]
        ),
        "reference_required_field_complete_sample_rate": optional_rate(
            [item.required_field_complete_sample for item in required_field_evaluations]
        ),
        "requirement_category_counts": summarize_prompt_requirement_atoms(all_atom_rows),
    }

    write_jsonl(args.output, output_rows)
    write_json(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
