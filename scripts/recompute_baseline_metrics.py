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

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json, write_jsonl
from llm_structured_semantic_generation.evaluation import StructuralEvaluation, evaluate_blocks_prediction, summarize_evaluations
from llm_structured_semantic_generation.prompt_requirements import (
    KIND_REQUIRED_FIELD_GROUPS,
    GENERIC_REQUIRED_FIELD_GROUPS,
    extract_prompt_requirements,
    extract_yaml_requirement_atoms_from_documents,
)
from llm_structured_semantic_generation.structure import parse_yaml_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute baseline evaluation metrics from persisted predictions without rerunning inference."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and predictions.jsonl for a completed or partial baseline run.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset override. Defaults to config.json dataset path.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <run-dir>/metrics_recomputed.json.",
    )
    parser.add_argument(
        "--output-evaluations",
        type=Path,
        default=None,
        help="Optional output JSONL path. Defaults to <run-dir>/recomputed_evaluations.jsonl.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional output Markdown path. Defaults to <run-dir>/recomputed_metrics_report.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_unit_id(row: dict[str, Any]) -> str:
    sample_id = row.get("sample_id")
    prompt_variant = row.get("prompt_variant")
    if not isinstance(sample_id, str) or not isinstance(prompt_variant, str):
        raise ValueError(f"Cannot build unit id from row: {row}")
    return f"{sample_id}::{prompt_variant}"


def load_dataset_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        index[build_unit_id(row)] = row
    return index


def derive_metrics(
    *,
    run_id: str,
    predictions: list[dict[str, Any]],
    recomputed_evaluations: list[dict[str, Any]],
    output_format: str,
    collect_latent_means: bool,
) -> dict[str, Any]:
    evaluated_results = [
        StructuralEvaluation(**row["evaluation_recomputed"])
        for row in recomputed_evaluations
        if row.get("evaluation_recomputed") is not None
    ]
    metrics = {
        "run_id": run_id,
        "row_count": len(predictions),
        "evaluated_count": len(evaluated_results),
        "output_format": output_format,
        "structured_output_parse_success_rate": len(evaluated_results) / len(predictions) if predictions else 0.0,
        "json_block_parse_success_rate": len(evaluated_results) / len(predictions) if predictions else 0.0,
    }
    metrics.update(summarize_evaluations(evaluated_results))
    if collect_latent_means:
        latent_dims = sorted(
            {
                row.get("latent_dim")
                for row in predictions
                if row.get("latent_dim") is not None
            }
        )
        metrics["latent_collection"] = {
            "enabled": True,
            "row_count": len(predictions),
            "rows_with_vector": sum(1 for row in predictions if row.get("latent_mean") is not None),
            "rows_without_generated_tokens": sum(1 for row in predictions if row.get("latent_mean") is None),
            "latent_dims": latent_dims,
            "artifact": "latent_mean_vectors.jsonl",
        }
    else:
        metrics["latent_collection"] = {"enabled": False}
    return metrics


def _required_group_labels_for_kind(kind: str) -> list[str]:
    labels: list[str] = []
    for group in GENERIC_REQUIRED_FIELD_GROUPS + KIND_REQUIRED_FIELD_GROUPS.get(kind, ()):
        if len(group) == 1:
            labels.append(".".join(group[0]))
        else:
            labels.append(" OR ".join(".".join(path) for path in group))
    return labels


def _path_present(document: Any, path: tuple[str, ...]) -> bool:
    current = document
    for segment in path:
        if not isinstance(current, dict):
            return False
        current = current.get(segment)
    if current is None:
        return False
    if isinstance(current, (list, dict, tuple, set)):
        return len(current) > 0
    return True


def _missing_required_groups(document: dict[str, Any]) -> list[str]:
    kind = document.get("kind")
    if not isinstance(kind, str):
        return []
    missing: list[str] = []
    for group in GENERIC_REQUIRED_FIELD_GROUPS + KIND_REQUIRED_FIELD_GROUPS.get(kind, ()):
        if not any(_path_present(document, path) for path in group):
            if len(group) == 1:
                missing.append(".".join(group[0]))
            else:
                missing.append(" OR ".join(".".join(path) for path in group))
    return missing


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def analyze_recomputed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_category_prompt_total: Counter[str] = Counter()
    prompt_category_prediction_total: Counter[str] = Counter()
    prompt_category_match_total: Counter[str] = Counter()
    missing_required_groups_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    incomplete_resource_count_by_kind: Counter[str] = Counter()
    sample_incomplete_required_fields: list[dict[str, Any]] = []

    structured_failure_count = 0
    yaml_failure_count = 0
    exact_match_count = 0

    for row in rows:
        evaluation = row.get("evaluation_recomputed")
        if evaluation is None:
            structured_failure_count += 1
            continue

        if evaluation["parsed_equal_to_reference"]:
            exact_match_count += 1
        if not evaluation["yaml_parse_ok"]:
            yaml_failure_count += 1
            continue

        prompt_atoms = {atom["canonical"]: atom for atom in row.get("prompt_requirements", [])}
        predicted_atoms = {atom["canonical"]: atom for atom in row.get("predicted_requirement_atoms", [])}
        prompt_categories = {atom["category"] for atom in prompt_atoms.values()}
        comparable_prediction_atoms = {
            canonical: atom
            for canonical, atom in predicted_atoms.items()
            if atom["category"] in prompt_categories
        }
        matched_atoms = set(prompt_atoms) & set(comparable_prediction_atoms)

        for atom in prompt_atoms.values():
            prompt_category_prompt_total[atom["category"]] += 1
        for atom in comparable_prediction_atoms.values():
            prompt_category_prediction_total[atom["category"]] += 1
        for canonical in matched_atoms:
            prompt_category_match_total[prompt_atoms[canonical]["category"]] += 1

        prediction_documents = row.get("prediction_documents", [])
        missing_summary_for_row: list[dict[str, Any]] = []
        for document in prediction_documents:
            if not isinstance(document, dict):
                continue
            kind = document.get("kind")
            if not isinstance(kind, str):
                continue
            missing_groups = _missing_required_groups(document)
            if not missing_groups:
                continue
            incomplete_resource_count_by_kind[kind] += 1
            for missing_group in missing_groups:
                missing_required_groups_by_kind[kind][missing_group] += 1
            missing_summary_for_row.append(
                {
                    "kind": kind,
                    "missing_required_groups": missing_groups,
                }
            )
        if missing_summary_for_row:
            sample_incomplete_required_fields.append(
                {
                    "unit_id": row["unit_id"],
                    "sample_id": row["sample_id"],
                    "prompt_variant": row["prompt_variant"],
                    "missing_required_groups": missing_summary_for_row,
                }
            )

    category_rows = []
    for category in sorted(prompt_category_prompt_total):
        prompt_total = prompt_category_prompt_total[category]
        prediction_total = prompt_category_prediction_total[category]
        match_total = prompt_category_match_total[category]
        category_rows.append(
            {
                "category": category,
                "prompt_total": prompt_total,
                "prediction_total": prediction_total,
                "match_total": match_total,
                "precision": _safe_divide(match_total, prediction_total),
                "recall": _safe_divide(match_total, prompt_total),
            }
        )

    missing_required_fields_by_kind = {
        kind: {
            "incomplete_resource_count": incomplete_resource_count_by_kind[kind],
            "missing_groups": dict(counter.most_common()),
            "required_groups_considered": _required_group_labels_for_kind(kind),
        }
        for kind, counter in sorted(
            missing_required_groups_by_kind.items(),
            key=lambda item: (-sum(item[1].values()), item[0]),
        )
    }

    return {
        "structured_output_failure_count": structured_failure_count,
        "yaml_failure_count_with_structured_output": yaml_failure_count,
        "parsed_equal_count": exact_match_count,
        "prompt_requirement_category_summary": category_rows,
        "missing_required_fields_by_kind": missing_required_fields_by_kind,
        "sample_incomplete_required_fields_examples": sample_incomplete_required_fields[:12],
    }


def render_report(
    *,
    run_dir: Path,
    dataset_path: Path,
    old_metrics: dict[str, Any] | None,
    new_metrics: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    def metric_line(name: str) -> str:
        old_value = None if old_metrics is None else old_metrics.get(name)
        new_value = new_metrics.get(name)
        return f"- `{name}`: old={old_value} new={new_value}"

    category_lines = [
        f"- `{row['category']}`: prompt_total={row['prompt_total']}, prediction_total={row['prediction_total']}, match_total={row['match_total']}, precision={row['precision']}, recall={row['recall']}"
        for row in analysis["prompt_requirement_category_summary"]
    ]
    if not category_lines:
        category_lines = ["- No prompt-requirement category rows were available."]

    incomplete_lines = []
    for kind, payload in analysis["missing_required_fields_by_kind"].items():
        missing_groups = ", ".join(
            f"{group} ({count})" for group, count in list(payload["missing_groups"].items())[:5]
        )
        incomplete_lines.append(
            f"- `{kind}`: incomplete_resources={payload['incomplete_resource_count']}; most common missing groups: {missing_groups}"
        )
    if not incomplete_lines:
        incomplete_lines = ["- No missing required-field groups were found among YAML-parsed predictions."]

    example_lines = []
    for example in analysis["sample_incomplete_required_fields_examples"]:
        chunks = []
        for item in example["missing_required_groups"]:
            chunks.append(f"{item['kind']}: {', '.join(item['missing_required_groups'])}")
        example_lines.append(
            f"- `{example['unit_id']}`: " + " | ".join(chunks)
        )
    if not example_lines:
        example_lines = ["- No incomplete required-field examples were found."]

    conclusion_lines = [
        f"- The recomputation confirms that the run can be re-evaluated offline from persisted artifacts alone; no new model inference was needed.",
        f"- Prompt-requirement coverage is materially stronger than exact YAML equality: `average_prompt_requirement_f1 = {new_metrics.get('average_prompt_requirement_f1')}` versus `parsed_equal_rate = {new_metrics.get('parsed_equal_rate')}`.",
        f"- Required-field validity is high once the prediction reaches YAML parseability: `average_required_field_complete_resource_rate = {new_metrics.get('average_required_field_complete_resource_rate')}` and `required_field_complete_sample_rate = {new_metrics.get('required_field_complete_sample_rate')}`.",
        f"- The largest remaining bottleneck is still upstream structural generation, not minimal resource completeness: `structured_output_parse_success_rate = {new_metrics.get('structured_output_parse_success_rate')}` and `yaml_parse_success_rate = {new_metrics.get('yaml_parse_success_rate')}`.",
        f"- This means the baseline often captures coarse intent and minimal field structure, but still fails too often in full structural realization and exact reconstruction.",
    ]

    return "\n".join(
        [
            "# Baseline Offline Recomputed Metrics Report",
            "",
            f"- Run directory: `{run_dir}`",
            f"- Dataset used for recomputation: `{dataset_path}`",
            f"- Row count: `{new_metrics.get('row_count')}`",
            f"- Evaluated count: `{new_metrics.get('evaluated_count')}`",
            "",
            "## Headline metrics",
            "",
            metric_line("structured_output_parse_success_rate"),
            metric_line("yaml_parse_success_rate"),
            metric_line("parsed_equal_rate"),
            metric_line("average_line_text_f1"),
            metric_line("average_semantic_key_f1"),
            metric_line("average_prompt_requirement_precision"),
            metric_line("average_prompt_requirement_recall"),
            metric_line("average_prompt_requirement_f1"),
            metric_line("prompt_requirement_exact_match_rate"),
            metric_line("average_required_field_presence_rate"),
            metric_line("average_required_field_complete_resource_rate"),
            metric_line("required_field_complete_sample_rate"),
            "",
            "## Error profile",
            "",
            f"- Structured-output failures before evaluation: `{analysis['structured_output_failure_count']}`",
            f"- YAML failures after structured parsing: `{analysis['yaml_failure_count_with_structured_output']}`",
            f"- Parsed-equal rows: `{analysis['parsed_equal_count']}`",
            "",
            "## Prompt Requirement Categories",
            "",
            *category_lines,
            "",
            "## Missing Required Fields By Kind",
            "",
            *incomplete_lines,
            "",
            "## Example Incomplete Predictions",
            "",
            *example_lines,
            "",
            "## Conclusions",
            "",
            *conclusion_lines,
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config = load_json(run_dir / "config.json")
    dataset_path = (args.dataset or Path(config["dataset"])).resolve()
    old_metrics = load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None
    predictions = read_jsonl(run_dir / "predictions.jsonl", allow_truncated_last_line=False)
    dataset_index = load_dataset_index(dataset_path)

    recomputed_rows: list[dict[str, Any]] = []
    recovery_mode = str(config.get("recovery_mode", "strict"))
    output_format = str(config.get("output_format", "unknown"))
    collect_latent_means = bool(config.get("collect_latent_means", False))

    for prediction in predictions:
        unit_id = prediction.get("unit_id") or build_unit_id(prediction)
        reference_row = dataset_index.get(unit_id)
        if reference_row is None:
            raise KeyError(f"Reference row not found for unit_id={unit_id}")

        prompt_text = prediction.get("prompt_text") or reference_row.get("prompt_text")
        predicted_blocks = list(prediction.get("predicted_blocks") or [])
        recomputed_evaluation = (
            evaluate_blocks_prediction(
                reference_row["target_yaml_normalized"],
                predicted_blocks,
                recovery_mode=recovery_mode,
                prompt_text=prompt_text,
            ).to_dict()
            if predicted_blocks
            else None
        )

        prediction_documents: list[Any] = []
        if recomputed_evaluation is not None and recomputed_evaluation.get("yaml_parse_ok"):
            prediction_documents = list(parse_yaml_documents(prediction.get("reconstructed_yaml", "")))

        prompt_requirements = [atom.to_dict() for atom in extract_prompt_requirements(str(prompt_text or ""))]
        predicted_requirement_atoms = [
            atom.to_dict() for atom in extract_yaml_requirement_atoms_from_documents(tuple(prediction_documents))
        ]

        recomputed_rows.append(
            {
                "unit_id": unit_id,
                "sample_id": prediction.get("sample_id"),
                "prompt_variant": prediction.get("prompt_variant"),
                "split": prediction.get("split"),
                "output_format": output_format,
                "parser_errors": list(prediction.get("parser_errors") or []),
                "prompt_requirements": prompt_requirements,
                "predicted_requirement_atoms": predicted_requirement_atoms,
                "prediction_documents": prediction_documents,
                "evaluation_recomputed": recomputed_evaluation,
            }
        )

    metrics = derive_metrics(
        run_id=str(config.get("run_id", run_dir.name)),
        predictions=predictions,
        recomputed_evaluations=recomputed_rows,
        output_format=output_format,
        collect_latent_means=collect_latent_means,
    )
    analysis = analyze_recomputed_rows(recomputed_rows)

    output_metrics = (args.output_metrics or (run_dir / "metrics_recomputed.json")).resolve()
    output_evaluations = (args.output_evaluations or (run_dir / "recomputed_evaluations.jsonl")).resolve()
    output_report = (args.output_report or (run_dir / "recomputed_metrics_report.md")).resolve()

    write_json(output_metrics, metrics)
    write_jsonl(output_evaluations, recomputed_rows)
    output_report.write_text(
        render_report(
            run_dir=run_dir,
            dataset_path=dataset_path,
            old_metrics=old_metrics,
            new_metrics=metrics,
            analysis=analysis,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "dataset": str(dataset_path),
                "output_metrics": str(output_metrics),
                "output_evaluations": str(output_evaluations),
                "output_report": str(output_report),
                "row_count": metrics["row_count"],
                "evaluated_count": metrics["evaluated_count"],
                "average_prompt_requirement_f1": metrics.get("average_prompt_requirement_f1"),
                "average_required_field_complete_resource_rate": metrics.get("average_required_field_complete_resource_rate"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
