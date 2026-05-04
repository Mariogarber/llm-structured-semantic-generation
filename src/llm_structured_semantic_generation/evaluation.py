from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .prompt_requirements import evaluate_prompt_requirements, evaluate_required_fields
from .structure import blocks_to_yaml, coerce_block, parse_yaml_documents, yaml_to_blocks


SEMANTIC_FIELDS = (
    "metadata",
    "spec",
    "containers",
    "image",
    "ports",
    "env",
    "volumes",
    "volumeMounts",
    "selector",
    "template",
    "data",
    "rules",
    "subjects",
    "roleRef",
)

WORKLOAD_KINDS = {"Deployment", "DaemonSet", "StatefulSet", "ReplicaSet"}


@dataclass(frozen=True)
class StructuralEvaluation:
    yaml_parse_ok: bool
    parsed_equal_to_reference: bool
    block_parse_ok: bool
    valid_block_ratio: float
    document_index_monotonic_ok: bool
    line_index_sequence_ok: bool
    indentation_leak_rate: float
    reference_document_count: int
    prediction_document_count: int
    document_count_match: bool
    document_count_error: int
    line_count_reference: int
    line_count_prediction: int
    line_count_match: bool
    block_count_error: int
    content_exact_match_rate: float
    level_exact_match_rate: float
    line_text_precision: float
    line_text_recall: float
    line_text_f1: float
    level_mae: float | None
    primary_kind_match: bool | None
    primary_api_version_match: bool | None
    primary_metadata_name_match: bool | None
    kind_sequence_match_rate: float
    semantic_key_precision: float
    semantic_key_recall: float
    semantic_key_f1: float
    workload_selector_template_consistency: float | None
    service_selector_match_rate: float | None
    volume_mount_consistency: float | None
    errors: tuple[str, ...]
    prompt_requirement_supported: bool = False
    prompt_requirement_count: int = 0
    comparable_prediction_requirement_count: int = 0
    matched_prompt_requirement_count: int = 0
    prompt_requirement_precision: float | None = None
    prompt_requirement_recall: float | None = None
    prompt_requirement_f1: float | None = None
    prompt_requirement_exact_match: bool | None = None
    required_field_applicable_resource_count: int = 0
    required_field_total_count: int = 0
    required_field_present_count: int = 0
    required_field_presence_rate: float | None = None
    required_field_complete_resource_rate: float | None = None
    required_field_complete_sample: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _match_rate(reference: list[Any], prediction: list[Any]) -> float:
    if not reference and not prediction:
        return 1.0
    if not reference:
        return 0.0
    matched = sum(1 for left, right in zip(reference, prediction) if left == right)
    return matched / len(reference)


def _precision_recall_f1(reference: list[Any], prediction: list[Any]) -> tuple[float, float, float]:
    reference_counts = Counter(reference)
    prediction_counts = Counter(prediction)
    true_positive = sum(
        min(reference_counts[item], prediction_counts[item])
        for item in reference_counts.keys() | prediction_counts.keys()
    )
    precision = _safe_divide(true_positive, sum(prediction_counts.values()))
    recall = _safe_divide(true_positive, sum(reference_counts.values()))
    f1 = _safe_divide(2 * precision * recall, precision + recall) if precision or recall else 0.0
    return precision, recall, f1


def _mean_abs_error(reference: list[int], prediction: list[int]) -> float | None:
    overlap = min(len(reference), len(prediction))
    if overlap == 0:
        return None
    return sum(abs(left - right) for left, right in zip(reference[:overlap], prediction[:overlap])) / overlap


def _collect_recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_collect_recursive_keys(nested))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_collect_recursive_keys(item))
    return keys


def _semantic_key_set(documents: tuple[Any, ...]) -> set[str]:
    found: set[str] = set()
    recursive_keys = _collect_recursive_keys(documents)
    for key in SEMANTIC_FIELDS:
        if key in recursive_keys:
            found.add(key)
    return found


def _mapping_at(value: Any, *path: str) -> dict[str, Any] | None:
    current = value
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current if isinstance(current, dict) else None


def _pod_specs(document: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if not isinstance(document, dict):
        return specs

    root_spec = document.get("spec")
    if isinstance(root_spec, dict):
        if any(key in root_spec for key in ("containers", "initContainers", "ephemeralContainers", "volumes")):
            specs.append(root_spec)
        template_spec = _mapping_at(root_spec, "template", "spec")
        if template_spec is not None:
            specs.append(template_spec)
        job_template_spec = _mapping_at(root_spec, "jobTemplate", "spec", "template", "spec")
        if job_template_spec is not None:
            specs.append(job_template_spec)
    return specs


def _container_lists(pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    for key in ("containers", "initContainers", "ephemeralContainers"):
        value = pod_spec.get(key)
        if isinstance(value, list):
            containers.extend(item for item in value if isinstance(item, dict))
    return containers


def _workload_selector_template_consistency(documents: tuple[Any, ...]) -> float | None:
    applicable = 0
    satisfied = 0
    for document in documents:
        if not isinstance(document, dict):
            continue
        if document.get("kind") not in WORKLOAD_KINDS:
            continue
        selector = _mapping_at(document, "spec", "selector", "matchLabels")
        labels = _mapping_at(document, "spec", "template", "metadata", "labels")
        if selector is None or labels is None:
            continue
        applicable += 1
        if all(labels.get(key) == value for key, value in selector.items()):
            satisfied += 1
    if applicable == 0:
        return None
    return satisfied / applicable


def _service_selector_match_rate(documents: tuple[Any, ...]) -> float | None:
    workload_labels = []
    for document in documents:
        labels = _mapping_at(document, "spec", "template", "metadata", "labels")
        if labels is not None:
            workload_labels.append(labels)

    applicable = 0
    satisfied = 0
    for document in documents:
        if not isinstance(document, dict) or document.get("kind") != "Service":
            continue
        selector = _mapping_at(document, "spec", "selector")
        if selector is None:
            continue
        applicable += 1
        if any(all(labels.get(key) == value for key, value in selector.items()) for labels in workload_labels):
            satisfied += 1
    if applicable == 0:
        return None
    return satisfied / applicable


def _volume_mount_consistency(documents: tuple[Any, ...]) -> float | None:
    applicable = 0
    satisfied = 0
    for document in documents:
        for pod_spec in _pod_specs(document):
            containers = _container_lists(pod_spec)
            mount_names = []
            for container in containers:
                mounts = container.get("volumeMounts")
                if isinstance(mounts, list):
                    for mount in mounts:
                        if isinstance(mount, dict) and isinstance(mount.get("name"), str):
                            mount_names.append(mount["name"])
            if not mount_names:
                continue
            applicable += 1
            volume_names = {
                volume.get("name")
                for volume in pod_spec.get("volumes", [])
                if isinstance(volume, dict) and isinstance(volume.get("name"), str)
            }
            if all(name in volume_names for name in mount_names):
                satisfied += 1
    if applicable == 0:
        return None
    return satisfied / applicable


def _first_document_field(documents: tuple[Any, ...], *path: str) -> Any:
    if not documents:
        return None
    current: Any = documents[0]
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _kind_sequence(documents: tuple[Any, ...]) -> list[Any]:
    kinds = []
    for document in documents:
        kinds.append(document.get("kind") if isinstance(document, dict) else None)
    return kinds


def _summarize_predicted_blocks(predicted_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    total_blocks = len(predicted_blocks)
    valid_blocks = []
    indentation_leaks = 0
    for block in predicted_blocks:
        try:
            coerced = coerce_block(block)
        except ValueError:
            continue
        valid_blocks.append(coerced)
        if coerced.line_text.startswith((" ", "\t")):
            indentation_leaks += 1

    document_index_monotonic_ok = True
    line_index_sequence_ok = True
    previous_document = -1
    expected_by_document: dict[int, int] = {}
    for block in valid_blocks:
        if block.document_index < previous_document:
            document_index_monotonic_ok = False
        previous_document = max(previous_document, block.document_index)

        expected = expected_by_document.setdefault(block.document_index, 0)
        if block.line_index != expected:
            line_index_sequence_ok = False
        expected_by_document[block.document_index] = expected + 1

    prediction_document_count = len({block.document_index for block in valid_blocks})
    return {
        "valid_block_ratio": _safe_divide(len(valid_blocks), total_blocks),
        "document_index_monotonic_ok": document_index_monotonic_ok,
        "line_index_sequence_ok": line_index_sequence_ok,
        "indentation_leak_rate": _safe_divide(indentation_leaks, len(valid_blocks)),
        "prediction_document_count_from_blocks": prediction_document_count,
    }


def _build_failed_evaluation(
    reference_yaml: str,
    predicted_blocks: list[dict[str, Any]],
    *,
    errors: tuple[str, ...],
    prompt_text: str | None = None,
) -> StructuralEvaluation:
    reference_documents = parse_yaml_documents(reference_yaml)
    reference_blocks = list(yaml_to_blocks(reference_yaml))
    block_summary = _summarize_predicted_blocks(predicted_blocks)
    prompt_evaluation = evaluate_prompt_requirements(prompt_text, ())
    required_field_evaluation = evaluate_required_fields(())
    return StructuralEvaluation(
        yaml_parse_ok=False,
        parsed_equal_to_reference=False,
        block_parse_ok=False,
        valid_block_ratio=block_summary["valid_block_ratio"],
        document_index_monotonic_ok=block_summary["document_index_monotonic_ok"],
        line_index_sequence_ok=block_summary["line_index_sequence_ok"],
        indentation_leak_rate=block_summary["indentation_leak_rate"],
        reference_document_count=len(reference_documents),
        prediction_document_count=block_summary["prediction_document_count_from_blocks"],
        document_count_match=len(reference_documents) == block_summary["prediction_document_count_from_blocks"],
        document_count_error=abs(
            len(reference_documents) - block_summary["prediction_document_count_from_blocks"]
        ),
        line_count_reference=len(reference_blocks),
        line_count_prediction=len(predicted_blocks),
        line_count_match=len(reference_blocks) == len(predicted_blocks),
        block_count_error=abs(len(reference_blocks) - len(predicted_blocks)),
        content_exact_match_rate=0.0,
        level_exact_match_rate=0.0,
        line_text_precision=0.0,
        line_text_recall=0.0,
        line_text_f1=0.0,
        level_mae=None,
        primary_kind_match=None,
        primary_api_version_match=None,
        primary_metadata_name_match=None,
        kind_sequence_match_rate=0.0,
        semantic_key_precision=0.0,
        semantic_key_recall=0.0,
        semantic_key_f1=0.0,
        workload_selector_template_consistency=None,
        service_selector_match_rate=None,
        volume_mount_consistency=None,
        errors=errors,
        prompt_requirement_supported=prompt_evaluation.prompt_requirement_supported,
        prompt_requirement_count=prompt_evaluation.prompt_requirement_count,
        comparable_prediction_requirement_count=prompt_evaluation.comparable_prediction_requirement_count,
        matched_prompt_requirement_count=prompt_evaluation.matched_prompt_requirement_count,
        prompt_requirement_precision=prompt_evaluation.prompt_requirement_precision,
        prompt_requirement_recall=prompt_evaluation.prompt_requirement_recall,
        prompt_requirement_f1=prompt_evaluation.prompt_requirement_f1,
        prompt_requirement_exact_match=prompt_evaluation.prompt_requirement_exact_match,
        required_field_applicable_resource_count=required_field_evaluation.required_field_applicable_resource_count,
        required_field_total_count=required_field_evaluation.required_field_total_count,
        required_field_present_count=required_field_evaluation.required_field_present_count,
        required_field_presence_rate=required_field_evaluation.required_field_presence_rate,
        required_field_complete_resource_rate=required_field_evaluation.required_field_complete_resource_rate,
        required_field_complete_sample=required_field_evaluation.required_field_complete_sample,
    )


def evaluate_yaml_prediction(
    reference_yaml: str,
    prediction_yaml: str,
    *,
    prompt_text: str | None = None,
) -> StructuralEvaluation:
    errors: list[str] = []

    try:
        reference_documents = parse_yaml_documents(reference_yaml)
    except Exception as exc:  # pragma: no cover - reference data should already be valid
        return _build_failed_evaluation(
            "",
            [],
            errors=(f"reference_parse_error:{exc.__class__.__name__}",),
            prompt_text=prompt_text,
        )

    reference_blocks = list(yaml_to_blocks(reference_yaml))

    try:
        prediction_documents = parse_yaml_documents(prediction_yaml)
        yaml_parse_ok = True
    except Exception as exc:
        prediction_documents = ()
        yaml_parse_ok = False
        errors.append(f"prediction_parse_error:{exc.__class__.__name__}")

    try:
        prediction_blocks = list(yaml_to_blocks(prediction_yaml)) if yaml_parse_ok else []
        block_parse_ok = yaml_parse_ok
    except ValueError as exc:
        prediction_blocks = []
        block_parse_ok = False
        errors.append(f"block_parse_error:{exc}")

    if not yaml_parse_ok:
        return _build_failed_evaluation(reference_yaml, [], errors=tuple(errors), prompt_text=prompt_text)

    reference_text = [block.line_text for block in reference_blocks]
    prediction_text = [block.line_text for block in prediction_blocks]
    reference_levels = [block.level for block in reference_blocks]
    prediction_levels = [block.level for block in prediction_blocks]
    line_text_precision, line_text_recall, line_text_f1 = _precision_recall_f1(reference_text, prediction_text)
    semantic_precision, semantic_recall, semantic_f1 = _precision_recall_f1(
        sorted(_semantic_key_set(reference_documents)),
        sorted(_semantic_key_set(prediction_documents)),
    )
    kind_sequence_match_rate = _match_rate(_kind_sequence(reference_documents), _kind_sequence(prediction_documents))
    primary_kind_reference = _first_document_field(reference_documents, "kind")
    primary_kind_prediction = _first_document_field(prediction_documents, "kind")
    primary_api_reference = _first_document_field(reference_documents, "apiVersion")
    primary_api_prediction = _first_document_field(prediction_documents, "apiVersion")
    primary_name_reference = _first_document_field(reference_documents, "metadata", "name")
    primary_name_prediction = _first_document_field(prediction_documents, "metadata", "name")
    prompt_evaluation = evaluate_prompt_requirements(prompt_text, prediction_documents)
    required_field_evaluation = evaluate_required_fields(prediction_documents)

    return StructuralEvaluation(
        yaml_parse_ok=True,
        parsed_equal_to_reference=prediction_documents == reference_documents,
        block_parse_ok=block_parse_ok,
        valid_block_ratio=1.0 if prediction_blocks else 0.0,
        document_index_monotonic_ok=True,
        line_index_sequence_ok=True,
        indentation_leak_rate=0.0,
        reference_document_count=len(reference_documents),
        prediction_document_count=len(prediction_documents),
        document_count_match=len(reference_documents) == len(prediction_documents),
        document_count_error=abs(len(reference_documents) - len(prediction_documents)),
        line_count_reference=len(reference_blocks),
        line_count_prediction=len(prediction_blocks),
        line_count_match=len(reference_blocks) == len(prediction_blocks),
        block_count_error=abs(len(reference_blocks) - len(prediction_blocks)),
        content_exact_match_rate=_match_rate(reference_text, prediction_text),
        level_exact_match_rate=_match_rate(reference_levels, prediction_levels),
        line_text_precision=line_text_precision,
        line_text_recall=line_text_recall,
        line_text_f1=line_text_f1,
        level_mae=_mean_abs_error(reference_levels, prediction_levels),
        primary_kind_match=primary_kind_reference == primary_kind_prediction,
        primary_api_version_match=primary_api_reference == primary_api_prediction,
        primary_metadata_name_match=primary_name_reference == primary_name_prediction,
        kind_sequence_match_rate=kind_sequence_match_rate,
        semantic_key_precision=semantic_precision,
        semantic_key_recall=semantic_recall,
        semantic_key_f1=semantic_f1,
        workload_selector_template_consistency=_workload_selector_template_consistency(prediction_documents),
        service_selector_match_rate=_service_selector_match_rate(prediction_documents),
        volume_mount_consistency=_volume_mount_consistency(prediction_documents),
        errors=tuple(errors),
        prompt_requirement_supported=prompt_evaluation.prompt_requirement_supported,
        prompt_requirement_count=prompt_evaluation.prompt_requirement_count,
        comparable_prediction_requirement_count=prompt_evaluation.comparable_prediction_requirement_count,
        matched_prompt_requirement_count=prompt_evaluation.matched_prompt_requirement_count,
        prompt_requirement_precision=prompt_evaluation.prompt_requirement_precision,
        prompt_requirement_recall=prompt_evaluation.prompt_requirement_recall,
        prompt_requirement_f1=prompt_evaluation.prompt_requirement_f1,
        prompt_requirement_exact_match=prompt_evaluation.prompt_requirement_exact_match,
        required_field_applicable_resource_count=required_field_evaluation.required_field_applicable_resource_count,
        required_field_total_count=required_field_evaluation.required_field_total_count,
        required_field_present_count=required_field_evaluation.required_field_present_count,
        required_field_presence_rate=required_field_evaluation.required_field_presence_rate,
        required_field_complete_resource_rate=required_field_evaluation.required_field_complete_resource_rate,
        required_field_complete_sample=required_field_evaluation.required_field_complete_sample,
    )


def evaluate_blocks_prediction(
    reference_yaml: str,
    predicted_blocks: list[dict[str, Any]],
    *,
    recovery_mode: str = "strict",
    prompt_text: str | None = None,
) -> StructuralEvaluation:
    reconstruction = blocks_to_yaml(predicted_blocks, recovery_mode=recovery_mode)
    if not reconstruction.yaml_parse_ok:
        return _build_failed_evaluation(
            reference_yaml,
            predicted_blocks,
            errors=reconstruction.errors,
            prompt_text=prompt_text,
        )

    evaluation = evaluate_yaml_prediction(
        reference_yaml,
        reconstruction.yaml_text,
        prompt_text=prompt_text,
    )
    block_summary = _summarize_predicted_blocks(predicted_blocks)
    return StructuralEvaluation(
        yaml_parse_ok=evaluation.yaml_parse_ok,
        parsed_equal_to_reference=evaluation.parsed_equal_to_reference,
        block_parse_ok=evaluation.block_parse_ok,
        valid_block_ratio=block_summary["valid_block_ratio"],
        document_index_monotonic_ok=block_summary["document_index_monotonic_ok"],
        line_index_sequence_ok=block_summary["line_index_sequence_ok"],
        indentation_leak_rate=block_summary["indentation_leak_rate"],
        reference_document_count=evaluation.reference_document_count,
        prediction_document_count=evaluation.prediction_document_count,
        document_count_match=evaluation.document_count_match,
        document_count_error=evaluation.document_count_error,
        line_count_reference=evaluation.line_count_reference,
        line_count_prediction=len(predicted_blocks),
        line_count_match=evaluation.line_count_reference == len(predicted_blocks),
        block_count_error=abs(evaluation.line_count_reference - len(predicted_blocks)),
        content_exact_match_rate=evaluation.content_exact_match_rate,
        level_exact_match_rate=evaluation.level_exact_match_rate,
        line_text_precision=evaluation.line_text_precision,
        line_text_recall=evaluation.line_text_recall,
        line_text_f1=evaluation.line_text_f1,
        level_mae=evaluation.level_mae,
        primary_kind_match=evaluation.primary_kind_match,
        primary_api_version_match=evaluation.primary_api_version_match,
        primary_metadata_name_match=evaluation.primary_metadata_name_match,
        kind_sequence_match_rate=evaluation.kind_sequence_match_rate,
        semantic_key_precision=evaluation.semantic_key_precision,
        semantic_key_recall=evaluation.semantic_key_recall,
        semantic_key_f1=evaluation.semantic_key_f1,
        workload_selector_template_consistency=evaluation.workload_selector_template_consistency,
        service_selector_match_rate=evaluation.service_selector_match_rate,
        volume_mount_consistency=evaluation.volume_mount_consistency,
        errors=evaluation.errors,
        prompt_requirement_supported=evaluation.prompt_requirement_supported,
        prompt_requirement_count=evaluation.prompt_requirement_count,
        comparable_prediction_requirement_count=evaluation.comparable_prediction_requirement_count,
        matched_prompt_requirement_count=evaluation.matched_prompt_requirement_count,
        prompt_requirement_precision=evaluation.prompt_requirement_precision,
        prompt_requirement_recall=evaluation.prompt_requirement_recall,
        prompt_requirement_f1=evaluation.prompt_requirement_f1,
        prompt_requirement_exact_match=evaluation.prompt_requirement_exact_match,
        required_field_applicable_resource_count=evaluation.required_field_applicable_resource_count,
        required_field_total_count=evaluation.required_field_total_count,
        required_field_present_count=evaluation.required_field_present_count,
        required_field_presence_rate=evaluation.required_field_presence_rate,
        required_field_complete_resource_rate=evaluation.required_field_complete_resource_rate,
        required_field_complete_sample=evaluation.required_field_complete_sample,
    )


def summarize_evaluations(evaluations: list[StructuralEvaluation]) -> dict[str, Any]:
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

    return {
        "evaluated_count": len(evaluations),
        "yaml_parse_success_rate": rate([item.yaml_parse_ok for item in evaluations]),
        "parsed_equal_rate": rate([item.parsed_equal_to_reference for item in evaluations]),
        "block_parse_success_rate": rate([item.block_parse_ok for item in evaluations]),
        "document_index_monotonic_ok_rate": rate([item.document_index_monotonic_ok for item in evaluations]),
        "line_index_sequence_ok_rate": rate([item.line_index_sequence_ok for item in evaluations]),
        "document_count_match_rate": rate([item.document_count_match for item in evaluations]),
        "line_count_match_rate": rate([item.line_count_match for item in evaluations]),
        "average_valid_block_ratio": average([item.valid_block_ratio for item in evaluations]),
        "average_indentation_leak_rate": average([item.indentation_leak_rate for item in evaluations]),
        "average_document_count_error": average([float(item.document_count_error) for item in evaluations]),
        "average_block_count_error": average([float(item.block_count_error) for item in evaluations]),
        "average_content_exact_match_rate": average([item.content_exact_match_rate for item in evaluations]),
        "average_level_exact_match_rate": average([item.level_exact_match_rate for item in evaluations]),
        "average_line_text_precision": average([item.line_text_precision for item in evaluations]),
        "average_line_text_recall": average([item.line_text_recall for item in evaluations]),
        "average_line_text_f1": average([item.line_text_f1 for item in evaluations]),
        "average_level_mae": average_optional([item.level_mae for item in evaluations]),
        "primary_kind_match_rate": optional_rate([item.primary_kind_match for item in evaluations]),
        "primary_api_version_match_rate": optional_rate([item.primary_api_version_match for item in evaluations]),
        "primary_metadata_name_match_rate": optional_rate(
            [item.primary_metadata_name_match for item in evaluations]
        ),
        "average_kind_sequence_match_rate": average([item.kind_sequence_match_rate for item in evaluations]),
        "average_semantic_key_precision": average([item.semantic_key_precision for item in evaluations]),
        "average_semantic_key_recall": average([item.semantic_key_recall for item in evaluations]),
        "average_semantic_key_f1": average([item.semantic_key_f1 for item in evaluations]),
        "average_workload_selector_template_consistency": average_optional(
            [item.workload_selector_template_consistency for item in evaluations]
        ),
        "average_service_selector_match_rate": average_optional(
            [item.service_selector_match_rate for item in evaluations]
        ),
        "average_volume_mount_consistency": average_optional(
            [item.volume_mount_consistency for item in evaluations]
        ),
        "prompt_requirement_support_rate": rate([item.prompt_requirement_supported for item in evaluations]),
        "average_prompt_requirement_count": average(
            [float(item.prompt_requirement_count) for item in evaluations if item.prompt_requirement_supported]
        ),
        "average_prompt_requirement_precision": average_optional(
            [item.prompt_requirement_precision for item in evaluations]
        ),
        "average_prompt_requirement_recall": average_optional(
            [item.prompt_requirement_recall for item in evaluations]
        ),
        "average_prompt_requirement_f1": average_optional(
            [item.prompt_requirement_f1 for item in evaluations]
        ),
        "prompt_requirement_exact_match_rate": optional_rate(
            [item.prompt_requirement_exact_match for item in evaluations]
        ),
        "average_required_field_presence_rate": average_optional(
            [item.required_field_presence_rate for item in evaluations]
        ),
        "average_required_field_complete_resource_rate": average_optional(
            [item.required_field_complete_resource_rate for item in evaluations]
        ),
        "required_field_complete_sample_rate": optional_rate(
            [item.required_field_complete_sample for item in evaluations]
        ),
    }
