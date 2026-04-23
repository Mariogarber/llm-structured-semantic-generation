from __future__ import annotations

from typing import Any

from .structure import YAMLBlock, blocks_to_yaml, validate_blocks


SYSTEM_PROMPT = (
    "You generate Kubernetes manifests through an explicit structural representation. "
    "Return only line blocks; each block must include document_index, line_index, level, and line_text."
)


def serialize_blocks_for_training(blocks: list[YAMLBlock | dict[str, Any]]) -> str:
    validated_blocks = validate_blocks(blocks)
    lines = ["<blocks>"]
    for block in validated_blocks:
        escaped_text = block.line_text.replace("\\", "\\\\").replace("\t", "\\t")
        lines.append(
            f"{block.document_index}\t{block.line_index}\t{block.level}\t{escaped_text}"
        )
    lines.append("</blocks>")
    return "\n".join(lines)


def deserialize_training_blocks(serialized: str) -> list[YAMLBlock]:
    lines = [line for line in serialized.splitlines() if line and line not in {"<blocks>", "</blocks>"}]
    blocks: list[YAMLBlock] = []
    for line in lines:
        document_index, line_index, level, line_text = line.split("\t", maxsplit=3)
        blocks.append(
            YAMLBlock(
                document_index=int(document_index),
                line_index=int(line_index),
                level=int(level),
                line_text=line_text.replace("\\t", "\t").replace("\\\\", "\\"),
            )
        )
    return list(validate_blocks(blocks))


def build_sft_prompt(prompt_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Natural-language request:\n"
        f"{prompt_text.strip()}\n\n"
        "Return the structural block sequence now."
    )


def build_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    blocks = [YAMLBlock(**block) for block in row["blocks"]]
    serialized_target = serialize_blocks_for_training(blocks)
    reconstruction = blocks_to_yaml(blocks)
    return {
        "sample_id": row["sample_id"],
        "prompt_variant": row["prompt_variant"],
        "split": row["split"],
        "prompt": build_sft_prompt(row["prompt_text"]),
        "target": serialized_target,
        "target_yaml_normalized": row["target_yaml_normalized"],
        "round_trip_yaml": reconstruction.yaml_text,
    }

