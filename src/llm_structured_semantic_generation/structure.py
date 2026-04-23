from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import yaml


DEFAULT_INDENT_WIDTH = 2


@dataclass(frozen=True)
class YAMLBlock:
    """One logical YAML line with its explicit hierarchy level."""

    line_index: int
    line_text: str
    level: int
    document_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionResult:
    yaml_text: str
    yaml_parse_ok: bool
    errors: tuple[str, ...]
    parsed_documents: tuple[Any, ...] = ()


@dataclass(frozen=True)
class RoundTripResult:
    blocks: tuple[YAMLBlock, ...]
    reconstructed_yaml: str
    yaml_parse_ok: bool
    semantics_preserved: bool
    errors: tuple[str, ...]


def parse_yaml_documents(yaml_text: str) -> tuple[Any, ...]:
    return tuple(yaml.safe_load_all(yaml_text))


def yaml_to_blocks(yaml_text: str, indent_width: int = DEFAULT_INDENT_WIDTH) -> tuple[YAMLBlock, ...]:
    """Convert normalized YAML text into line-and-level blocks.

    Document separator lines are represented by the following blocks'
    ``document_index`` rather than as content blocks. Blank lines are kept
    because they can be significant inside quoted multi-line scalar renderings.
    """

    if indent_width <= 0:
        raise ValueError("indent_width must be positive")

    blocks: list[YAMLBlock] = []
    document_index = 0
    line_index_by_document: dict[int, int] = {document_index: 0}

    for physical_line in yaml_text.splitlines():
        stripped = physical_line.strip()
        if stripped in {"---", "..."}:
            if stripped == "---":
                document_index += 1
                line_index_by_document.setdefault(document_index, 0)
            continue

        leading_spaces = len(physical_line) - len(physical_line.lstrip(" "))
        if leading_spaces % indent_width != 0:
            raise ValueError(
                f"Line indentation is not divisible by {indent_width}: {physical_line!r}"
            )

        line_index = line_index_by_document[document_index]
        blocks.append(
            YAMLBlock(
                line_index=line_index,
                line_text=physical_line[leading_spaces:],
                level=leading_spaces // indent_width,
                document_index=document_index,
            )
        )
        line_index_by_document[document_index] = line_index + 1

    return tuple(blocks)


def coerce_block(block: YAMLBlock | dict[str, Any]) -> YAMLBlock:
    if isinstance(block, YAMLBlock):
        return block
    try:
        return YAMLBlock(
            line_index=int(block["line_index"]),
            line_text=str(block["line_text"]),
            level=int(block["level"]),
            document_index=int(block.get("document_index", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid YAML block payload: {block!r}") from exc


def validate_blocks(blocks: Iterable[YAMLBlock | dict[str, Any]]) -> tuple[YAMLBlock, ...]:
    coerced = tuple(coerce_block(block) for block in blocks)
    errors: list[str] = []
    previous_document = 0
    expected_line_index_by_document: dict[int, int] = {}

    for position, block in enumerate(coerced):
        if block.document_index < 0:
            errors.append(f"block_{position}:negative_document_index")
        if block.level < 0:
            errors.append(f"block_{position}:negative_level")
        if block.line_text.startswith((" ", "\t")):
            errors.append(f"block_{position}:line_text_contains_leading_indentation")
        if "\n" in block.line_text or "\r" in block.line_text:
            errors.append(f"block_{position}:line_text_contains_newline")
        if block.document_index < previous_document:
            errors.append(f"block_{position}:document_index_regression")
        previous_document = block.document_index

        expected = expected_line_index_by_document.setdefault(block.document_index, 0)
        if block.line_index != expected:
            errors.append(
                f"block_{position}:unexpected_line_index:{block.line_index}:expected:{expected}"
            )
        expected_line_index_by_document[block.document_index] = expected + 1

    if errors:
        raise ValueError(";".join(errors))
    return coerced


def blocks_to_yaml(
    blocks: Iterable[YAMLBlock | dict[str, Any]],
    indent_width: int = DEFAULT_INDENT_WIDTH,
) -> ReconstructionResult:
    if indent_width <= 0:
        return ReconstructionResult(
            yaml_text="",
            yaml_parse_ok=False,
            errors=("indent_width_must_be_positive",),
        )

    try:
        validated_blocks = validate_blocks(blocks)
    except ValueError as exc:
        return ReconstructionResult(
            yaml_text="",
            yaml_parse_ok=False,
            errors=tuple(str(exc).split(";")),
        )

    lines: list[str] = []
    current_document = validated_blocks[0].document_index if validated_blocks else 0
    for index, block in enumerate(validated_blocks):
        if index > 0 and block.document_index != current_document:
            lines.append("---")
            current_document = block.document_index
        lines.append(f"{' ' * (block.level * indent_width)}{block.line_text}")

    yaml_text = "\n".join(lines)
    if yaml_text:
        yaml_text += "\n"

    try:
        parsed_documents = parse_yaml_documents(yaml_text)
    except yaml.YAMLError as exc:
        return ReconstructionResult(
            yaml_text=yaml_text,
            yaml_parse_ok=False,
            errors=(f"yaml_parse_error:{exc.__class__.__name__}",),
        )

    return ReconstructionResult(
        yaml_text=yaml_text,
        yaml_parse_ok=True,
        errors=(),
        parsed_documents=parsed_documents,
    )


def validate_round_trip(yaml_text: str, indent_width: int = DEFAULT_INDENT_WIDTH) -> RoundTripResult:
    errors: list[str] = []
    try:
        reference_documents = parse_yaml_documents(yaml_text)
    except yaml.YAMLError as exc:
        return RoundTripResult(
            blocks=(),
            reconstructed_yaml="",
            yaml_parse_ok=False,
            semantics_preserved=False,
            errors=(f"reference_yaml_parse_error:{exc.__class__.__name__}",),
        )

    try:
        blocks = yaml_to_blocks(yaml_text, indent_width=indent_width)
    except ValueError as exc:
        return RoundTripResult(
            blocks=(),
            reconstructed_yaml="",
            yaml_parse_ok=False,
            semantics_preserved=False,
            errors=(str(exc),),
        )

    reconstruction = blocks_to_yaml(blocks, indent_width=indent_width)
    errors.extend(reconstruction.errors)
    semantics_preserved = reconstruction.yaml_parse_ok and reconstruction.parsed_documents == reference_documents
    if reconstruction.yaml_parse_ok and not semantics_preserved:
        errors.append("round_trip_semantics_changed")

    return RoundTripResult(
        blocks=blocks,
        reconstructed_yaml=reconstruction.yaml_text,
        yaml_parse_ok=reconstruction.yaml_parse_ok,
        semantics_preserved=semantics_preserved,
        errors=tuple(errors),
    )


def blocks_to_serialized_target(blocks: Iterable[YAMLBlock | dict[str, Any]]) -> str:
    """Serialize blocks as stable JSONL-like rows for supervised targets."""

    validated_blocks = validate_blocks(blocks)
    rows = []
    for block in validated_blocks:
        rows.append(json.dumps(block.to_dict(), ensure_ascii=False, sort_keys=True))
    return "\n".join(rows)
