from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .structure import blocks_to_yaml, parse_yaml_documents, yaml_to_blocks


@dataclass(frozen=True)
class StructuralEvaluation:
    yaml_parse_ok: bool
    parsed_equal_to_reference: bool
    block_parse_ok: bool
    line_count_reference: int
    line_count_prediction: int
    line_count_match: bool
    content_exact_match_rate: float
    level_exact_match_rate: float
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _match_rate(reference: list[Any], prediction: list[Any]) -> float:
    if not reference and not prediction:
        return 1.0
    if not reference:
        return 0.0
    matched = sum(1 for left, right in zip(reference, prediction) if left == right)
    return matched / len(reference)


def evaluate_yaml_prediction(reference_yaml: str, prediction_yaml: str) -> StructuralEvaluation:
    errors: list[str] = []

    try:
        reference_documents = parse_yaml_documents(reference_yaml)
    except Exception as exc:  # pragma: no cover - reference data should already be valid
        return StructuralEvaluation(
            yaml_parse_ok=False,
            parsed_equal_to_reference=False,
            block_parse_ok=False,
            line_count_reference=0,
            line_count_prediction=0,
            line_count_match=False,
            content_exact_match_rate=0.0,
            level_exact_match_rate=0.0,
            errors=(f"reference_parse_error:{exc.__class__.__name__}",),
        )

    try:
        prediction_documents = parse_yaml_documents(prediction_yaml)
        yaml_parse_ok = True
    except Exception as exc:
        prediction_documents = ()
        yaml_parse_ok = False
        errors.append(f"prediction_parse_error:{exc.__class__.__name__}")

    try:
        reference_blocks = list(yaml_to_blocks(reference_yaml))
        prediction_blocks = list(yaml_to_blocks(prediction_yaml)) if yaml_parse_ok else []
        block_parse_ok = True
    except ValueError as exc:
        reference_blocks = []
        prediction_blocks = []
        block_parse_ok = False
        errors.append(f"block_parse_error:{exc}")

    reference_text = [block.line_text for block in reference_blocks]
    prediction_text = [block.line_text for block in prediction_blocks]
    reference_levels = [block.level for block in reference_blocks]
    prediction_levels = [block.level for block in prediction_blocks]

    return StructuralEvaluation(
        yaml_parse_ok=yaml_parse_ok,
        parsed_equal_to_reference=yaml_parse_ok and prediction_documents == reference_documents,
        block_parse_ok=block_parse_ok,
        line_count_reference=len(reference_blocks),
        line_count_prediction=len(prediction_blocks),
        line_count_match=len(reference_blocks) == len(prediction_blocks),
        content_exact_match_rate=_match_rate(reference_text, prediction_text),
        level_exact_match_rate=_match_rate(reference_levels, prediction_levels),
        errors=tuple(errors),
    )


def evaluate_blocks_prediction(reference_yaml: str, predicted_blocks: list[dict[str, Any]]) -> StructuralEvaluation:
    reconstruction = blocks_to_yaml(predicted_blocks)
    if not reconstruction.yaml_parse_ok:
        reference_blocks = list(yaml_to_blocks(reference_yaml))
        return StructuralEvaluation(
            yaml_parse_ok=False,
            parsed_equal_to_reference=False,
            block_parse_ok=False,
            line_count_reference=len(reference_blocks),
            line_count_prediction=0,
            line_count_match=False,
            content_exact_match_rate=0.0,
            level_exact_match_rate=0.0,
            errors=reconstruction.errors,
        )
    return evaluate_yaml_prediction(reference_yaml, reconstruction.yaml_text)

