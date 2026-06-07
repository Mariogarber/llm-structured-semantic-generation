from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import train_kubernetes_sft as serialized_sft
from llm_structured_semantic_generation.dataset_io import append_jsonl, read_jsonl, write_json
from llm_structured_semantic_generation.evaluation import (
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)
from llm_structured_semantic_generation.resumable_run import RunCompatibilityError, utc_now_iso
from llm_structured_semantic_generation.sft_serialization import (
    BLOCKS_TSV_V1,
    deserialize_training_blocks,
)


TWO_HEAD_ORDINAL_SFT = "two_head_ordinal_film_positional_v2"
CONTENT_BLOCKS_V1 = "content_blocks_v1"
RECORD_PREFIX_STATE = "record_prefix_state"
PROMPT_TARGET_SEPARATOR = "\n\n"
IGNORE_INDEX = -100
DEFAULT_LEVEL_CLASS_COUNT = 9
DEFAULT_ORDINAL_HEAD_HIDDEN_DIMS = (512, 64)
DEFAULT_ORDINAL_HEAD_DROPOUTS = (0.10, 0.0)
DEFAULT_ORDINAL_POSITION_ENCODING = "sinusoidal_absolute"
DEFAULT_ORDINAL_POSITION_DIM = 16
DEFAULT_ORDINAL_POSITION_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
DEFAULT_ORDINAL_POSITION_INJECTION = "film_after_512"
DEFAULT_ORDINAL_FILM_HIDDEN_DIM = 128
DEFAULT_ORDINAL_FILM_IDENTITY_SCALE = 0.10
DEFAULT_LAMBDA_LEVEL = 2.0
DEFAULT_DENSITY_KERNEL = "triangular"
DEFAULT_DENSITY_KERNEL_RADIUS = 1
DEFAULT_MAX_DENSITY_WEIGHT = 12.0
DEFAULT_INITIAL_THRESHOLD_CENTER = 0.0
DEFAULT_INITIAL_THRESHOLD_GAP = 0.5
DEEP_LEVELS = (5, 6, 7, 8)


@dataclass(frozen=True)
class ContentLineSpan:
    document_index: int
    line_index: int
    line_position: int
    level: int | None
    line_text: str
    record_start: int
    record_end: int
    line_text_start: int
    line_text_end: int


@dataclass(frozen=True)
class ContentOnlyExample:
    unit_id: str
    prompt: str
    content_text: str
    full_text: str
    line_spans: list[ContentLineSpan]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Kubernetes v1 Architecture B2 FiLM positional variant: learned ordinal level "
            "thresholds with causal sinusoidal line-position conditioning."
        )
    )
    parser.add_argument("--model-variant", choices=[TWO_HEAD_ORDINAL_SFT], default=TWO_HEAD_ORDINAL_SFT)
    parser.add_argument("--serialization", choices=[CONTENT_BLOCKS_V1], default=CONTENT_BLOCKS_V1)
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
        "--base-model-path",
        type=Path,
        default=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "two_head_ordinal_film_positional_sft_kubernetes_v1",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=serialized_sft.positive_int, default=1)
    parser.add_argument("--epochs", type=serialized_sft.positive_int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=serialized_sft.positive_int, default=8)
    parser.add_argument("--max-seq-length", type=serialized_sft.positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=serialized_sft.positive_int, default=1024)
    parser.add_argument("--checkpoint-steps", type=serialized_sft.positive_int, default=25)
    parser.add_argument("--checkpoint-keep-last", type=serialized_sft.non_negative_int, default=0)
    parser.add_argument("--eval-checkpoint-steps", type=serialized_sft.non_negative_int, default=0)
    parser.add_argument("--eval-max-samples", type=serialized_sft.positive_int, default=10)
    parser.add_argument("--eval-sample-strategy", choices=["random", "first"], default="random")
    parser.add_argument("--validation-log-every", type=serialized_sft.non_negative_int, default=1)
    parser.add_argument("--oom-recovery", choices=["fail", "skip_batch"], default="fail")
    parser.add_argument("--max-oom-skips", type=serialized_sft.non_negative_int, default=0)
    parser.add_argument("--max-train-samples", type=serialized_sft.positive_int, default=None)
    parser.add_argument("--max-validation-samples", type=serialized_sft.positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=serialized_sft.positive_int, default=8)
    parser.add_argument("--lora-alpha", type=serialized_sft.positive_int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(serialized_sft.DEFAULT_LORA_TARGET_MODULES),
    )
    parser.add_argument("--lambda-level", type=float, default=DEFAULT_LAMBDA_LEVEL)
    parser.add_argument("--level-class-count", type=serialized_sft.positive_int, default=DEFAULT_LEVEL_CLASS_COUNT)
    parser.add_argument("--ordinal-head-hidden-dims", default=",".join(str(value) for value in DEFAULT_ORDINAL_HEAD_HIDDEN_DIMS))
    parser.add_argument("--ordinal-head-dropouts", default=",".join(str(value) for value in DEFAULT_ORDINAL_HEAD_DROPOUTS))
    parser.add_argument(
        "--ordinal-position-encoding",
        choices=["none", "sinusoidal_absolute"],
        default=DEFAULT_ORDINAL_POSITION_ENCODING,
        help="Causal line-position encoding consumed only by this positional ordinal head.",
    )
    parser.add_argument("--ordinal-position-dim", type=serialized_sft.positive_int, default=DEFAULT_ORDINAL_POSITION_DIM)
    parser.add_argument(
        "--ordinal-position-frequencies",
        default=",".join(format(value, "g") for value in DEFAULT_ORDINAL_POSITION_FREQUENCIES),
        help="Comma-separated sinusoidal frequency scales used as sin(t/f), cos(t/f).",
    )
    parser.add_argument(
        "--ordinal-position-injection",
        choices=["film_after_512"],
        default=DEFAULT_ORDINAL_POSITION_INJECTION,
        help="Where the positional encoding is injected into the ordinal head.",
    )
    parser.add_argument("--ordinal-film-hidden-dim", type=serialized_sft.positive_int, default=DEFAULT_ORDINAL_FILM_HIDDEN_DIM)
    parser.add_argument("--ordinal-film-identity-scale", type=float, default=DEFAULT_ORDINAL_FILM_IDENTITY_SCALE)
    parser.add_argument(
        "--threshold-learning-rate-multiplier",
        type=float,
        default=50.0,
        help=(
            "Multiplier applied only to learned ordinal threshold parameters "
            "(raw_tau0/raw_deltas). The FiLM V2 default keeps the LR50 setting "
            "from the current centered-gap ordinal experiments."
        ),
    )
    parser.add_argument(
        "--ordinal-mlp-learning-rate-multiplier",
        type=float,
        default=3.0,
        help=(
            "Multiplier applied only to the ordinal MLP projector that maps hidden states to z. "
            "The FiLM V2 default keeps the MLP3 setting from the current centered-gap "
            "ordinal experiments."
        ),
    )
    parser.add_argument(
        "--ordinal-gradient-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Log curated gradient, optimizer-update, and functional shift diagnostics "
            "for the ordinal MLP projector and learned threshold parameters."
        ),
    )
    parser.add_argument(
        "--initial-threshold-center",
        type=float,
        default=None,
        help=(
            "Center of the initial ordered ordinal thresholds. When omitted, new runs use "
            f"{DEFAULT_INITIAL_THRESHOLD_CENTER}, while old run resumes keep backward-compatible "
            "resume signatures."
        ),
    )
    parser.add_argument(
        "--initial-threshold-gap",
        type=float,
        default=None,
        help=(
            "Initial spacing between adjacent ordinal thresholds. When omitted, new runs use "
            f"{DEFAULT_INITIAL_THRESHOLD_GAP}. Thresholds remain trainable after initialization."
        ),
    )
    parser.add_argument("--density-kernel", choices=["triangular", "uniform"], default=DEFAULT_DENSITY_KERNEL)
    parser.add_argument("--density-kernel-radius", type=serialized_sft.non_negative_int, default=DEFAULT_DENSITY_KERNEL_RADIUS)
    parser.add_argument("--max-density-weight", type=float, default=DEFAULT_MAX_DENSITY_WEIGHT)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="llm-structured-semantic-generation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def build_unit_id(row: dict[str, Any]) -> str:
    return serialized_sft.build_unit_id(row)


def split_csv(value: str) -> tuple[str, ...]:
    return serialized_sft.split_csv(value)


def split_int_csv(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected_at_least_one_integer")
    if any(item <= 0 for item in items):
        raise ValueError(f"expected_positive_integers:{value}")
    return items


def split_float_csv(value: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected_at_least_one_float")
    if any(item < 0 for item in items):
        raise ValueError(f"expected_non_negative_floats:{value}")
    return items


def position_frequency_values(value: str) -> tuple[float, ...]:
    frequencies = split_float_csv(value)
    if any(frequency <= 0 for frequency in frequencies):
        raise ValueError(f"expected_positive_position_frequencies:{value}")
    return frequencies


def validate_position_config(args: argparse.Namespace) -> None:
    if args.ordinal_position_encoding == "none":
        return
    hidden_dims = split_int_csv(args.ordinal_head_hidden_dims)
    if len(hidden_dims) < 2:
        raise ValueError("film_positional_head_requires_at_least_two_hidden_dims")
    frequencies = position_frequency_values(args.ordinal_position_frequencies)
    expected_dim = 2 * len(frequencies)
    if int(args.ordinal_position_dim) != expected_dim:
        raise ValueError(
            "ordinal_position_dim_must_match_sin_cos_frequency_count:"
            f"dim={args.ordinal_position_dim}:expected={expected_dim}"
        )
    if args.ordinal_position_injection != "film_after_512":
        raise ValueError(f"unsupported_ordinal_position_injection:{args.ordinal_position_injection}")
    if float(args.ordinal_film_identity_scale) <= 0:
        raise ValueError("ordinal_film_identity_scale_must_be_positive")


def effective_initial_threshold_center(args: argparse.Namespace) -> float:
    value = getattr(args, "initial_threshold_center", None)
    return DEFAULT_INITIAL_THRESHOLD_CENTER if value is None else float(value)


def effective_initial_threshold_gap(args: argparse.Namespace) -> float:
    value = getattr(args, "initial_threshold_gap", None)
    gap = DEFAULT_INITIAL_THRESHOLD_GAP if value is None else float(value)
    if gap <= 0:
        raise ValueError("initial_threshold_gap must be positive")
    return gap


def initial_threshold_values(*, level_class_count: int, center: float, gap: float) -> list[float]:
    threshold_count = int(level_class_count) - 1
    if threshold_count <= 0:
        raise ValueError("level_class_count must be at least 2")
    start = float(center) - 0.5 * float(gap) * (threshold_count - 1)
    return [start + index * float(gap) for index in range(threshold_count)]


def ordinal_targets(level_labels: Any, *, level_class_count: int) -> Any:
    import torch

    thresholds = torch.arange(level_class_count - 1, device=level_labels.device)
    return (level_labels.unsqueeze(-1) > thresholds).to(dtype=torch.float32)


def _kernel_weight(distance: int, *, kernel: str, radius: int) -> float:
    if distance < 0:
        raise ValueError("distance_must_be_non_negative")
    if kernel == "uniform":
        return 1.0 if distance <= radius else 0.0
    if kernel == "triangular":
        return max(float(radius + 1 - distance), 0.0) if distance <= radius else 0.0
    raise ValueError(f"unsupported_density_kernel:{kernel}")


def compute_level_density_weights(
    rows: list[dict[str, Any]],
    *,
    level_class_count: int,
    kernel: str,
    radius: int,
    max_weight: float,
) -> dict[str, Any]:
    if max_weight <= 0:
        raise ValueError("max_density_weight_must_be_positive")
    counts = [0 for _ in range(level_class_count)]
    for row in rows:
        for block in deserialize_training_blocks(str(row["target"])):
            level = int(block.level)
            if 0 <= level < level_class_count:
                counts[level] += 1
    smoothed = []
    for level in range(level_class_count):
        value = 0.0
        for other_level, count in enumerate(counts):
            value += count * _kernel_weight(abs(level - other_level), kernel=kernel, radius=radius)
        smoothed.append(value)
    positive = [value for value in smoothed if value > 0]
    mean_density = sum(positive) / len(positive) if positive else 1.0
    weights = [
        min(mean_density / value, max_weight) if value > 0 else max_weight
        for value in smoothed
    ]
    return {
        "counts": counts,
        "smoothed_density": smoothed,
        "weights": weights,
        "kernel": kernel,
        "radius": radius,
        "max_weight": max_weight,
    }


def _escape_line_text(line_text: str) -> str:
    return line_text.replace("\\", "\\\\").replace("\t", "\\t")


def _unescape_line_text(line_text: str) -> str:
    return line_text.replace("\\t", "\t").replace("\\\\", "\\")


def extract_natural_language_request(prompt: str) -> str:
    match = re.search(
        r"Natural-language request:\n(?P<request>.*?)(?:\n\nReturn\b|$)",
        prompt,
        flags=re.DOTALL,
    )
    if match:
        return match.group("request").strip()
    return prompt.strip()


def build_content_only_prompt(prompt: str) -> str:
    request = extract_natural_language_request(prompt)
    return (
        "You generate Kubernetes manifests through an explicit content-line representation. "
        "Return only content line blocks; each block must include document_index, line_index, and line_text. "
        "Do not include hierarchy levels in the output surface.\n\n"
        "Natural-language request:\n"
        f"{request}\n\n"
        "Return the content line block sequence now."
    )


def build_content_text_and_spans(
    blocks: list[Any],
    *,
    include_level_labels: bool,
) -> tuple[str, list[ContentLineSpan]]:
    parts = ["<content_blocks>\n"]
    spans: list[ContentLineSpan] = []
    cursor = len(parts[0])
    for line_position, block in enumerate(blocks):
        prefix = f"{int(block.document_index)}\t{int(block.line_index)}\t"
        escaped_text = _escape_line_text(str(block.line_text))
        record = f"{prefix}{escaped_text}\n"
        record_start = cursor
        line_text_start = record_start + len(prefix)
        line_text_end = line_text_start + len(escaped_text)
        record_end = record_start + len(record)
        spans.append(
            ContentLineSpan(
                document_index=int(block.document_index),
                line_index=int(block.line_index),
                line_position=line_position,
                level=int(block.level) if include_level_labels else None,
                line_text=str(block.line_text),
                record_start=record_start,
                record_end=record_end,
                line_text_start=line_text_start,
                line_text_end=line_text_end,
            )
        )
        parts.append(record)
        cursor = record_end
    parts.append("</content_blocks>")
    return "".join(parts), spans


def build_content_only_example(row: dict[str, Any]) -> ContentOnlyExample:
    blocks = deserialize_training_blocks(str(row["target"]))
    prompt = build_content_only_prompt(str(row["prompt"]))
    content_text, spans = build_content_text_and_spans(blocks, include_level_labels=True)
    full_text = f"{prompt}{PROMPT_TARGET_SEPARATOR}{content_text}"
    prompt_offset = len(prompt) + len(PROMPT_TARGET_SEPARATOR)
    shifted_spans = [
        ContentLineSpan(
            document_index=span.document_index,
            line_index=span.line_index,
            line_position=span.line_position,
            level=span.level,
            line_text=span.line_text,
            record_start=span.record_start + prompt_offset,
            record_end=span.record_end + prompt_offset,
            line_text_start=span.line_text_start + prompt_offset,
            line_text_end=span.line_text_end + prompt_offset,
        )
        for span in spans
    ]
    return ContentOnlyExample(
        unit_id=build_unit_id(row),
        prompt=prompt,
        content_text=content_text,
        full_text=full_text,
        line_spans=shifted_spans,
    )


def gold_levels_from_row(row: dict[str, Any]) -> list[int]:
    return [int(block.level) for block in deserialize_training_blocks(str(row["target"]))]


def level_diagnostic_metrics(predictions: list[dict[str, Any]], *, level_class_count: int) -> dict[str, Any]:
    gold_counts = [0 for _ in range(level_class_count)]
    pred_counts = [0 for _ in range(level_class_count)]
    true_positive = [0 for _ in range(level_class_count)]
    predicted_max_levels: list[int] = []
    deep_support = 0
    deep_exact = 0
    deep_off_by_one = 0
    deep_compressed = 0
    target_max_ge5_total = 0
    target_max_ge5_yaml_ok = 0

    for prediction in predictions:
        gold_levels = [int(level) for level in prediction.get("gold_levels", [])]
        pred_levels = [int(block["level"]) for block in prediction.get("predicted_blocks", [])]
        if pred_levels:
            predicted_max_levels.append(max(pred_levels))
        if gold_levels and max(gold_levels) >= 5:
            target_max_ge5_total += 1
            evaluation = prediction.get("evaluation") or {}
            if evaluation.get("yaml_parse_ok"):
                target_max_ge5_yaml_ok += 1
        for gold, pred in zip(gold_levels, pred_levels):
            if 0 <= gold < level_class_count:
                gold_counts[gold] += 1
            if 0 <= pred < level_class_count:
                pred_counts[pred] += 1
            if 0 <= gold < level_class_count and gold == pred:
                true_positive[gold] += 1
            if gold in DEEP_LEVELS:
                deep_support += 1
                if pred == gold:
                    deep_exact += 1
                if abs(pred - gold) <= 1:
                    deep_off_by_one += 1
                if pred <= 4:
                    deep_compressed += 1

    metrics: dict[str, Any] = {}
    for level in range(level_class_count):
        metrics[f"gold_level_count_{level}"] = gold_counts[level]
        metrics[f"predicted_level_count_{level}"] = pred_counts[level]
        metrics[f"level_recall_{level}"] = true_positive[level] / gold_counts[level] if gold_counts[level] else 0.0
        metrics[f"level_precision_{level}"] = true_positive[level] / pred_counts[level] if pred_counts[level] else 0.0
    metrics.update(
        {
            "deep_level_exact_recall_5_8": deep_exact / deep_support if deep_support else 0.0,
            "deep_level_off_by_one_recall_5_8": deep_off_by_one / deep_support if deep_support else 0.0,
            "compressed_deep_to_0_4_rate": deep_compressed / deep_support if deep_support else 0.0,
            "deep_level_support_5_8": deep_support,
            "predicted_max_level_mean": (
                sum(predicted_max_levels) / len(predicted_max_levels) if predicted_max_levels else 0.0
            ),
            "target_max_level_ge_5_count": target_max_ge5_total,
            "target_max_level_ge_5_yaml_parse_success_rate": (
                target_max_ge5_yaml_ok / target_max_ge5_total if target_max_ge5_total else 0.0
            ),
        }
    )
    return metrics


def validate_sft_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    serialized_sft.validate_sft_rows(rows, split_name)
    for index, row in enumerate(rows):
        example = build_content_only_example(row)
        for line in example.content_text.splitlines():
            if not line or line in {"<content_blocks>", "</content_blocks>"}:
                continue
            if len(line.split("\t", maxsplit=2)) != 3:
                raise ValueError(f"{split_name}_row_{index}_invalid_content_only_line:{line!r}")


def load_sft_rows(path: Path, *, max_samples: int | None, split_name: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    validate_sft_rows(rows, split_name)
    return rows


def _prefix_index(offsets: list[tuple[int, int]], *, position: int) -> int | None:
    candidates = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start != token_end and token_end <= position
    ]
    return candidates[-1] if candidates else None


def _first_token_index_for_span(offsets: list[tuple[int, int]], span: ContentLineSpan) -> int | None:
    for index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        if token_end > span.record_start and token_start < span.record_end:
            return index
    return None


def tokenizer_call(tokenizer: Any, text: str, **kwargs: Any) -> dict[str, Any]:
    output = tokenizer(text, **kwargs)
    return dict(output)


def tokenize_two_head_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    example = build_content_only_example(row)
    eos = tokenizer.eos_token or ""
    encoded = tokenizer_call(
        tokenizer,
        example.full_text + eos,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise ValueError("tokenizer_must_return_offsets_for_two_head_alignment")
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    prompt_ids = tokenizer(
        f"{example.prompt}{PROMPT_TARGET_SEPARATOR}",
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if len(labels) < len(input_ids):
        labels.extend(input_ids[len(labels) :])
    labels = labels[: len(input_ids)]

    level_label_positions: list[int] = []
    level_line_positions: list[int] = []
    level_labels: list[int] = []
    level_token_labels = [IGNORE_INDEX] * len(input_ids)
    level_metadata: list[dict[str, Any]] = []
    for span in example.line_spans:
        position = _prefix_index(offsets, position=span.record_start)
        if position is None:
            position = _first_token_index_for_span(offsets, span)
        if position is None:
            raise ValueError(f"no_token_aligned_for_record_prefix:{example.unit_id}:{span.line_position}")
        if position >= max_seq_length:
            continue
        if span.level is None:
            raise ValueError("training_span_missing_level_label")
        level_label_positions.append(position)
        level_line_positions.append(int(span.line_position))
        level_labels.append(int(span.level))
        level_token_labels[position] = int(span.level)
        level_metadata.append(
            {
                "document_index": span.document_index,
                "line_index": span.line_index,
                "line_position": span.line_position,
                "level": int(span.level),
                "record_start": span.record_start,
                "record_prefix_token_index": position,
            }
        )

    input_ids = input_ids[:max_seq_length]
    labels = labels[:max_seq_length]
    level_token_labels = level_token_labels[:max_seq_length]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(f"row_has_no_supervised_content_after_truncation:{example.unit_id}")
    if not level_labels:
        raise ValueError(f"row_has_no_level_labels_after_truncation:{example.unit_id}")
    return {
        "unit_id": example.unit_id,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "level_label_positions": level_label_positions,
        "level_line_positions": level_line_positions,
        "level_labels": level_labels,
        "level_token_labels": level_token_labels,
        "level_metadata": level_metadata,
    }


class TwoHeadSFTDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, *, max_seq_length: int) -> None:
        self.items = [
            tokenize_two_head_row(row, tokenizer, max_seq_length=max_seq_length)
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class TwoHeadSFTCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in batch)
        max_lines = max(len(item["level_labels"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        level_token_labels = []
        level_label_positions = []
        level_line_positions = []
        level_labels = []
        unit_ids = []
        for item in batch:
            pad_count = max_length - len(item["input_ids"])
            line_pad_count = max_lines - len(item["level_labels"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_count)
            attention_mask.append(item["attention_mask"] + [0] * pad_count)
            labels.append(item["labels"] + [IGNORE_INDEX] * pad_count)
            level_token_labels.append(item["level_token_labels"] + [IGNORE_INDEX] * pad_count)
            level_label_positions.append(item["level_label_positions"] + [-1] * line_pad_count)
            level_line_positions.append(item["level_line_positions"] + [-1] * line_pad_count)
            level_labels.append(item["level_labels"] + [IGNORE_INDEX] * line_pad_count)
            unit_ids.append(item["unit_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "level_token_labels": torch.tensor(level_token_labels, dtype=torch.long),
            "level_label_positions": torch.tensor(level_label_positions, dtype=torch.long),
            "level_line_positions": torch.tensor(level_line_positions, dtype=torch.long),
            "level_labels": torch.tensor(level_labels, dtype=torch.long),
            "unit_ids": unit_ids,
        }


class OrdinalLevelHead:
    def __init__(
        self,
        *,
        hidden_size: int,
        hidden_dims: tuple[int, ...],
        dropouts: tuple[float, ...],
        level_class_count: int,
        level_weights: list[float],
        initial_threshold_center: float = DEFAULT_INITIAL_THRESHOLD_CENTER,
        initial_threshold_gap: float = DEFAULT_INITIAL_THRESHOLD_GAP,
        position_encoding: str = DEFAULT_ORDINAL_POSITION_ENCODING,
        position_dim: int = DEFAULT_ORDINAL_POSITION_DIM,
        position_frequencies: tuple[float, ...] = DEFAULT_ORDINAL_POSITION_FREQUENCIES,
        position_injection: str = DEFAULT_ORDINAL_POSITION_INJECTION,
        film_hidden_dim: int = DEFAULT_ORDINAL_FILM_HIDDEN_DIM,
        film_identity_scale: float = DEFAULT_ORDINAL_FILM_IDENTITY_SCALE,
    ) -> None:
        import torch

        class _OrdinalLevelHead(torch.nn.Module):
            def __init__(self, outer: "OrdinalLevelHead") -> None:
                super().__init__()
                threshold_count = int(level_class_count) - 1
                if threshold_count <= 0:
                    raise ValueError("level_class_count must be at least 2")
                initial_gap = float(initial_threshold_gap)
                if initial_gap <= 0:
                    raise ValueError("initial_threshold_gap must be positive")
                initial_thresholds = initial_threshold_values(
                    level_class_count=level_class_count,
                    center=float(initial_threshold_center),
                    gap=initial_gap,
                )
                self.position_encoding = str(position_encoding)
                self.position_dim = int(position_dim)
                self.position_injection = str(position_injection)
                self.position_enabled = self.position_encoding != "none"
                frequencies = tuple(float(value) for value in position_frequencies)
                if self.position_enabled:
                    if self.position_encoding != "sinusoidal_absolute":
                        raise ValueError(f"unsupported_ordinal_position_encoding:{self.position_encoding}")
                    if self.position_injection != "film_after_512":
                        raise ValueError(f"unsupported_ordinal_position_injection:{self.position_injection}")
                    if self.position_dim != 2 * len(frequencies):
                        raise ValueError(
                            "ordinal_position_dim_must_match_sin_cos_frequency_count:"
                            f"dim={self.position_dim}:expected={2 * len(frequencies)}"
                        )
                    if any(value <= 0 for value in frequencies):
                        raise ValueError(f"expected_positive_position_frequencies:{frequencies}")
                    if len(hidden_dims) < 2:
                        raise ValueError("film_positional_head_requires_at_least_two_hidden_dims")
                    if film_identity_scale <= 0:
                        raise ValueError("film_identity_scale_must_be_positive")

                first_hidden_dim = int(hidden_dims[0])
                pre_layers: list[torch.nn.Module] = [
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, first_hidden_dim),
                    torch.nn.GELU(),
                ]
                first_dropout = dropouts[0] if dropouts else 0.0
                if first_dropout > 0:
                    pre_layers.append(torch.nn.Dropout(first_dropout))
                self.pre_film_projector = torch.nn.Sequential(*pre_layers)

                post_layers: list[torch.nn.Module] = []
                input_dim = first_hidden_dim
                for index, output_dim in enumerate(hidden_dims[1:], start=1):
                    post_layers.append(torch.nn.Linear(input_dim, output_dim))
                    post_layers.append(torch.nn.GELU())
                    dropout = dropouts[index] if index < len(dropouts) else 0.0
                    if dropout > 0:
                        post_layers.append(torch.nn.Dropout(dropout))
                    input_dim = output_dim
                post_layers.append(torch.nn.LayerNorm(input_dim))
                self.post_film_projector = torch.nn.Sequential(*post_layers)
                if self.position_enabled:
                    self.position_norm = torch.nn.LayerNorm(self.position_dim)
                    self.film_identity_scale = float(film_identity_scale)
                    self.film_generator = torch.nn.Sequential(
                        torch.nn.Linear(self.position_dim, int(film_hidden_dim)),
                        torch.nn.GELU(),
                        torch.nn.Linear(int(film_hidden_dim), first_hidden_dim * 2),
                    )
                    torch.nn.init.zeros_(self.film_generator[-1].weight)
                    torch.nn.init.zeros_(self.film_generator[-1].bias)
                    self.register_buffer("position_frequencies", torch.tensor(frequencies, dtype=torch.float32))
                else:
                    self.position_norm = None
                    self.film_identity_scale = 0.0
                    self.film_generator = None
                    self.register_buffer("position_frequencies", torch.empty(0, dtype=torch.float32))
                self.final_projector = torch.nn.Linear(input_dim, 1)
                self.raw_tau0 = torch.nn.Parameter(torch.tensor(initial_thresholds[0], dtype=torch.float32))
                raw_gap = torch.expm1(torch.tensor(initial_gap, dtype=torch.float32)).log()
                self.raw_deltas = torch.nn.Parameter(raw_gap.repeat(threshold_count - 1))
                self.register_buffer("level_weights", torch.tensor(level_weights, dtype=torch.float32))

            def position_features(self, line_positions: Any, *, dtype: Any, device: Any) -> Any:
                if not self.position_enabled:
                    raise RuntimeError("position_features_called_when_position_encoding_disabled")
                positions = line_positions.to(device=device, dtype=torch.float32).clamp(min=0)
                frequencies = self.position_frequencies.to(device=device, dtype=torch.float32)
                angles = positions.unsqueeze(-1) / frequencies
                features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).reshape(*positions.shape, -1)
                return features.to(dtype=dtype)

            def thresholds(self) -> Any:
                if self.raw_deltas.numel() == 0:
                    return self.raw_tau0.reshape(1)
                gaps = torch.nn.functional.softplus(self.raw_deltas)
                return torch.cat([self.raw_tau0.reshape(1), self.raw_tau0 + torch.cumsum(gaps, dim=0)])

            def forward(self, selected_hidden: Any, line_positions: Any | None = None) -> Any:
                hidden_features = self.pre_film_projector(selected_hidden)
                if self.position_enabled:
                    if line_positions is None:
                        raise ValueError("line_positions_required_for_positional_ordinal_head")
                    position_features = self.position_features(
                        line_positions,
                        dtype=hidden_features.dtype,
                        device=hidden_features.device,
                    )
                    position_features = self.position_norm(position_features)
                    gamma, beta = self.film_generator(position_features).chunk(2, dim=-1)
                    scale = float(self.film_identity_scale)
                    hidden_features = hidden_features * (1.0 + scale * torch.tanh(gamma)) + scale * beta
                hidden_features = self.post_film_projector(hidden_features)
                z = self.final_projector(hidden_features).squeeze(-1)
                thresholds = self.thresholds().to(device=z.device, dtype=z.dtype)
                ordinal_logits = z.unsqueeze(-1) - thresholds
                predicted_levels = (ordinal_logits > 0).sum(dim=-1)
                cumulative = torch.sigmoid(ordinal_logits)
                lower = torch.ones_like(cumulative[..., :1])
                upper = torch.zeros_like(cumulative[..., :1])
                class_probs = torch.cat([lower, cumulative], dim=-1) - torch.cat([cumulative, upper], dim=-1)
                class_logits = torch.log(class_probs.clamp_min(1e-8))
                return SimpleNamespace(
                    z=z,
                    thresholds=thresholds,
                    ordinal_logits=ordinal_logits,
                    predicted_levels=predicted_levels,
                    level_logits=class_logits,
                )

        self.module = _OrdinalLevelHead(self)


class TwoHeadOrdinalQwenForSFT:
    def __init__(
        self,
        backbone: Any,
        *,
        hidden_size: int,
        level_class_count: int,
        ordinal_head_hidden_dims: tuple[int, ...],
        ordinal_head_dropouts: tuple[float, ...],
        level_weights: list[float],
        lambda_level: float,
        initial_threshold_center: float = DEFAULT_INITIAL_THRESHOLD_CENTER,
        initial_threshold_gap: float = DEFAULT_INITIAL_THRESHOLD_GAP,
        position_encoding: str = DEFAULT_ORDINAL_POSITION_ENCODING,
        position_dim: int = DEFAULT_ORDINAL_POSITION_DIM,
        position_frequencies: tuple[float, ...] = DEFAULT_ORDINAL_POSITION_FREQUENCIES,
        position_injection: str = DEFAULT_ORDINAL_POSITION_INJECTION,
        film_hidden_dim: int = DEFAULT_ORDINAL_FILM_HIDDEN_DIM,
        film_identity_scale: float = DEFAULT_ORDINAL_FILM_IDENTITY_SCALE,
    ) -> None:
        import torch

        class _TwoHeadModule(torch.nn.Module):
            def __init__(self, outer: "TwoHeadOrdinalQwenForSFT") -> None:
                super().__init__()
                self.outer = outer
                self.backbone = outer.backbone
                self.ordinal_level_head = outer.ordinal_level_head

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self.outer.forward(*args, **kwargs)

            def generate(self, *args: Any, **kwargs: Any) -> Any:
                return self.backbone.generate(*args, **kwargs)

            @property
            def config(self) -> Any:
                return self.backbone.config

        self.backbone = backbone
        self.lambda_level = float(lambda_level)
        self.level_class_count = int(level_class_count)
        self.ordinal_level_head = OrdinalLevelHead(
            hidden_size=hidden_size,
            hidden_dims=ordinal_head_hidden_dims,
            dropouts=ordinal_head_dropouts,
            level_class_count=level_class_count,
            level_weights=level_weights,
            initial_threshold_center=initial_threshold_center,
            initial_threshold_gap=initial_threshold_gap,
            position_encoding=position_encoding,
            position_dim=position_dim,
            position_frequencies=position_frequencies,
            position_injection=position_injection,
            film_hidden_dim=film_hidden_dim,
            film_identity_scale=film_identity_scale,
        ).module
        self.module = _TwoHeadModule(self)

    def __getattr__(self, name: str) -> Any:
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def parameters(self) -> Any:
        return self.module.parameters()

    def train(self, mode: bool = True) -> Any:
        return self.module.train(mode)

    def eval(self) -> Any:
        return self.module.eval()

    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
        labels: Any | None = None,
        level_label_positions: Any | None = None,
        level_line_positions: Any | None = None,
        level_labels: Any | None = None,
        level_token_labels: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        import torch

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        level_logits = None
        ordinal_logits = None
        ordinal_z = None
        predicted_levels = None
        thresholds = None
        level_loss = None
        if level_label_positions is not None:
            safe_positions = level_label_positions.clamp(min=0)
            gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            selected_hidden = hidden.gather(1, gather_index)
            head_device = next(self.ordinal_level_head.parameters()).device
            if head_device != selected_hidden.device:
                self.ordinal_level_head.to(selected_hidden.device)
            head_dtype = next(self.ordinal_level_head.parameters()).dtype
            selected_hidden = selected_hidden.to(dtype=head_dtype)
            safe_line_positions = None
            if level_line_positions is not None:
                safe_line_positions = level_line_positions.clamp(min=0)
            head_outputs = self.ordinal_level_head(selected_hidden, line_positions=safe_line_positions)
            level_logits = head_outputs.level_logits
            ordinal_logits = head_outputs.ordinal_logits
            ordinal_z = head_outputs.z
            predicted_levels = head_outputs.predicted_levels
            thresholds = head_outputs.thresholds
            if level_labels is not None:
                valid = (level_label_positions >= 0) & (level_labels != IGNORE_INDEX)
                if bool(valid.any()):
                    safe_level_labels = level_labels.clamp(min=0, max=self.level_class_count - 1)
                    targets = ordinal_targets(safe_level_labels, level_class_count=self.level_class_count).to(
                        device=ordinal_logits.device,
                        dtype=ordinal_logits.dtype,
                    )
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        ordinal_logits,
                        targets,
                        reduction="none",
                    ).mean(dim=-1)
                    weights = self.ordinal_level_head.level_weights.to(
                        device=ordinal_logits.device,
                        dtype=ordinal_logits.dtype,
                    )[safe_level_labels]
                    level_loss = (bce[valid] * weights[valid]).mean()
                else:
                    level_loss = ordinal_logits.sum() * 0.0

        lm_loss = outputs.loss if labels is not None else None
        if lm_loss is not None and level_loss is not None:
            loss = lm_loss + self.lambda_level * level_loss
        elif lm_loss is not None:
            loss = lm_loss
        elif level_loss is not None:
            loss = self.lambda_level * level_loss
        else:
            loss = None
        return SimpleNamespace(
            loss=loss,
            lm_loss=lm_loss,
            level_loss=level_loss,
            ordinal_level_loss=level_loss,
            level_logits=level_logits,
            ordinal_logits=ordinal_logits,
            ordinal_z=ordinal_z,
            predicted_levels=predicted_levels,
            thresholds=thresholds,
            logits=getattr(outputs, "logits", None),
            hidden_states=getattr(outputs, "hidden_states", None),
        )


def model_input_device(model: Any) -> str:
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        return serialized_sft.model_input_device(backbone)
    return serialized_sft.model_input_device(model)


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key != "unit_ids"
    }


def move_optimizer_state_to_active_devices(optimizer: Any) -> None:
    import torch

    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            grad = getattr(param, "grad", None)
            target_device = grad.device if grad is not None else param.device
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.device != target_device:
                    state[key] = value.to(device=target_device)


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.base_model_path, local_files_only=True)
    quantization_config = getattr(config, "quantization_config", None)
    is_bitsandbytes_4bit = isinstance(quantization_config, dict) and bool(quantization_config.get("load_in_4bit"))
    load_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available() and is_bitsandbytes_4bit:
        load_kwargs["device_map"] = {"": 0}
    else:
        max_memory = {"cpu": args.cpu_memory}
        if torch.cuda.is_available():
            max_memory[0] = args.gpu_memory
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model_path, **load_kwargs)
    if is_bitsandbytes_4bit:
        backbone = prepare_model_for_kbit_training(backbone)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(split_csv(args.lora_target_modules)),
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.config.use_cache = False
    model = TwoHeadOrdinalQwenForSFT(
        backbone,
        hidden_size=int(config.hidden_size),
        level_class_count=args.level_class_count,
        ordinal_head_hidden_dims=split_int_csv(args.ordinal_head_hidden_dims),
        ordinal_head_dropouts=split_float_csv(args.ordinal_head_dropouts),
        level_weights=list(getattr(args, "level_density_weights", [1.0] * args.level_class_count)),
        lambda_level=args.lambda_level,
        initial_threshold_center=effective_initial_threshold_center(args),
        initial_threshold_gap=effective_initial_threshold_gap(args),
        position_encoding=args.ordinal_position_encoding,
        position_dim=args.ordinal_position_dim,
        position_frequencies=position_frequency_values(args.ordinal_position_frequencies),
        position_injection=args.ordinal_position_injection,
        film_hidden_dim=args.ordinal_film_hidden_dim,
        film_identity_scale=args.ordinal_film_identity_scale,
    ).module
    return tokenizer, model


def save_checkpoint(
    *,
    run_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    state: dict[str, Any],
) -> Path:
    import torch

    path = serialized_sft.checkpoint_dir(run_dir, int(state["global_step"]))
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "tokenizer")
    torch.save(model.ordinal_level_head.state_dict(), path / "ordinal_level_head.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
        },
        path / "training_state.pt",
    )
    return path


def load_checkpoint_state(
    *,
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import set_peft_model_state_dict

    adapter_path = checkpoint_path / "adapter"
    if adapter_path.exists():
        if (adapter_path / "adapter_model.safetensors").exists():
            from safetensors.torch import load_file

            adapter_state = load_file(adapter_path / "adapter_model.safetensors")
        elif (adapter_path / "adapter_model.bin").exists():
            adapter_state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
        else:
            adapter_state = None
        if adapter_state is not None:
            set_peft_model_state_dict(model.backbone, adapter_state, adapter_name="default")
    level_head_path = checkpoint_path / "ordinal_level_head.pt"
    if level_head_path.exists():
        model.ordinal_level_head.load_state_dict(torch.load(level_head_path, map_location="cpu"))
    payload = torch.load(checkpoint_path / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state_to_active_devices(optimizer)
    scheduler.load_state_dict(payload["scheduler"])
    return model, payload["state"]


def normalize_structured_field_separators(text: str) -> str:
    return serialized_sft.normalize_structured_field_separators(text)


def extract_content_blocks_prediction(serialized: str) -> tuple[list[dict[str, Any]], str, list[ContentLineSpan]]:
    candidate = serialized.strip()
    fenced = re.search(r"```(?:text|tsv)?\s*(<content_blocks>.*?</content_blocks>)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        wrapped = re.search(r"<content_blocks>.*?</content_blocks>", candidate, flags=re.DOTALL)
        if wrapped:
            candidate = wrapped.group(0)
        else:
            opened = re.search(r"<content_blocks>.*", candidate, flags=re.DOTALL)
            if opened:
                candidate = opened.group(0)
    candidate = normalize_structured_field_separators(candidate)

    rows: list[dict[str, Any]] = []
    lightweight_blocks: list[Any] = []
    for raw_line in candidate.splitlines():
        line = normalize_structured_field_separators(raw_line)
        stripped = line.strip()
        if not stripped or stripped in {"<content_blocks>", "</content_blocks>"}:
            continue
        parts = line.split("\t", maxsplit=2)
        repaired_missing_document_index = False
        if len(parts) == 2 and rows:
            previous_document_index = int(rows[-1]["document_index"])
            parts = [str(previous_document_index), parts[0], parts[1]]
            repaired_missing_document_index = True
        elif len(parts) != 3:
            raise ValueError("not_enough_content_tsv_fields")
        document_index, line_index, line_text = parts
        try:
            document_index_int = int(document_index)
            line_index_int = int(line_index)
        except ValueError as exc:
            raise ValueError(f"invalid_content_tsv_numeric_field:{exc}") from exc
        unescaped_text = _unescape_line_text(line_text)
        row = {
            "document_index": document_index_int,
            "line_index": line_index_int,
            "line_text": unescaped_text,
        }
        if repaired_missing_document_index:
            row["surface_repair"] = "missing_document_index"
        rows.append(row)
    if not rows:
        raise ValueError("no_valid_content_blocks_found")
    next_line_index_by_document: dict[int, int] = {}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        document_index_int = int(row["document_index"])
        normalized_line_index = next_line_index_by_document.get(document_index_int, 0)
        next_line_index_by_document[document_index_int] = normalized_line_index + 1
        normalized_row = {
            "document_index": document_index_int,
            "line_index": normalized_line_index,
            "line_text": str(row["line_text"]),
        }
        if row.get("surface_repair"):
            normalized_row["surface_repair"] = row["surface_repair"]
        if int(row["line_index"]) != normalized_line_index:
            normalized_row["line_index_normalized_from"] = int(row["line_index"])
        normalized_rows.append(normalized_row)
        lightweight_blocks.append(
            SimpleNamespace(
                document_index=document_index_int,
                line_index=normalized_line_index,
                line_text=str(row["line_text"]),
                level=0,
            )
        )
    content_text, spans = build_content_text_and_spans(lightweight_blocks, include_level_labels=False)
    return normalized_rows, content_text, spans


def generate_validation_completion(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    two_head_prompt = build_content_only_prompt(prompt)
    prompt_text = f"{two_head_prompt.rstrip()}{PROMPT_TARGET_SEPARATOR}"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_input_device(model))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_token_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return {
        "prompt": two_head_prompt,
        "raw_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "generated_token_count": int(generated_token_ids.shape[-1]),
    }


def positions_for_content_prediction(
    *,
    tokenizer: Any,
    prompt: str,
    content_text: str,
    spans: list[ContentLineSpan],
) -> dict[str, Any]:
    encoded = tokenizer_call(
        tokenizer,
        f"{prompt}{PROMPT_TARGET_SEPARATOR}{content_text}",
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    prompt_offset = len(prompt) + len(PROMPT_TARGET_SEPARATOR)
    positions: list[int] = []
    shifted_spans: list[ContentLineSpan] = []
    for span in spans:
        shifted = ContentLineSpan(
            document_index=span.document_index,
            line_index=span.line_index,
            line_position=span.line_position,
            level=None,
            line_text=span.line_text,
            record_start=span.record_start + prompt_offset,
            record_end=span.record_end + prompt_offset,
            line_text_start=span.line_text_start + prompt_offset,
            line_text_end=span.line_text_end + prompt_offset,
        )
        position = _prefix_index(offsets, position=shifted.record_start)
        if position is None:
            position = _first_token_index_for_span(offsets, shifted)
        if position is None:
            raise ValueError(f"no_token_aligned_for_generated_record:{span.line_position}")
        positions.append(position)
        shifted_spans.append(shifted)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask", [1] * len(encoded["input_ids"])),
        "level_label_positions": positions,
        "level_line_positions": [int(span.line_position) for span in spans],
        "spans": shifted_spans,
    }


def predict_levels_for_content(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    content_blocks: list[dict[str, Any]],
    content_text: str,
    spans: list[ContentLineSpan],
) -> list[dict[str, Any]]:
    import torch

    aligned = positions_for_content_prediction(
        tokenizer=tokenizer,
        prompt=prompt,
        content_text=content_text,
        spans=spans,
    )
    device = model_input_device(model)
    input_ids = torch.tensor([aligned["input_ids"]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([aligned["attention_mask"]], dtype=torch.long).to(device)
    positions = torch.tensor([aligned["level_label_positions"]], dtype=torch.long).to(device)
    line_positions = torch.tensor([aligned["level_line_positions"]], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            level_label_positions=positions,
            level_line_positions=line_positions,
        )
    predicted_levels = outputs.predicted_levels[0].detach().cpu().tolist()
    ordinal_scores = outputs.ordinal_z[0].detach().cpu().tolist()
    return [
        {
            "document_index": int(block["document_index"]),
            "line_index": int(block["line_index"]),
            "level": int(predicted_levels[index]),
            "ordinal_score": float(ordinal_scores[index]),
            "line_text": str(block["line_text"]),
        }
        for index, block in enumerate(content_blocks)
    ]


def derive_validation_metrics(
    *,
    run_id: str,
    predictions: list[dict[str, Any]],
    checkpoint: str,
) -> dict[str, Any]:
    evaluated_results = [
        StructuralEvaluation(**row["evaluation"])
        for row in predictions
        if row.get("evaluation") is not None
    ]
    metrics = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "row_count": len(predictions),
        "evaluated_count": len(evaluated_results),
        "model_variant": TWO_HEAD_ORDINAL_SFT,
        "serialization": CONTENT_BLOCKS_V1,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "structured_output_parse_success_rate": len(evaluated_results) / len(predictions) if predictions else 0.0,
    }
    metrics.update(summarize_evaluations(evaluated_results))
    metrics.update(level_diagnostic_metrics(predictions, level_class_count=DEFAULT_LEVEL_CLASS_COUNT))
    return metrics


CURATED_VALIDATION_METRICS = (
    "yaml_parse_success_rate",
    "average_level_exact_match_rate",
    "average_level_mae",
    "deep_level_exact_recall_5_8",
    "deep_level_off_by_one_recall_5_8",
    "compressed_deep_to_0_4_rate",
    "predicted_max_level_mean",
    "target_max_level_ge_5_yaml_parse_success_rate",
)


def curated_validation_payload(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {
        f"{prefix}/{key}": metrics[key]
        for key in CURATED_VALIDATION_METRICS
        if isinstance(metrics.get(key), (int, float)) and not isinstance(metrics.get(key), bool)
    }
    for level in range(DEFAULT_LEVEL_CLASS_COUNT):
        key = f"predicted_level_count_{level}"
        if isinstance(metrics.get(key), (int, float)):
            payload[f"{prefix}/{key}"] = metrics[key]
    return payload


def threshold_snapshot(*, model: Any, outputs: Any | None = None, valid_mask: Any | None = None) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        thresholds = model.ordinal_level_head.thresholds().detach().float().cpu().tolist()
    payload: dict[str, Any] = {}
    for index, value in enumerate(thresholds):
        payload[f"threshold_tau_{index}"] = float(value)
    for index, (left, right) in enumerate(zip(thresholds, thresholds[1:])):
        payload[f"threshold_gap_{index}"] = float(right - left)
    if outputs is not None and getattr(outputs, "ordinal_z", None) is not None:
        z = outputs.ordinal_z.detach().float()
        if valid_mask is not None:
            z = z[valid_mask]
        z = z.cpu()
        if z.numel():
            payload["ordinal_z_mean"] = float(z.mean())
            payload["ordinal_z_min"] = float(z.min())
            payload["ordinal_z_max"] = float(z.max())
    if outputs is not None and getattr(outputs, "predicted_levels", None) is not None:
        predicted_tensor = outputs.predicted_levels.detach()
        if valid_mask is not None:
            predicted_tensor = predicted_tensor[valid_mask]
        predicted = predicted_tensor.cpu().reshape(-1).tolist()
        counts = {level: 0 for level in range(DEFAULT_LEVEL_CLASS_COUNT)}
        for level in predicted:
            counts[int(level)] = counts.get(int(level), 0) + 1
        for level in range(DEFAULT_LEVEL_CLASS_COUNT):
            payload[f"batch_predicted_level_count_{level}"] = counts[level]
    return payload


def threshold_wandb_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in snapshot.items():
        if key.startswith("threshold_tau_"):
            payload[f"train/threshold/tau_{key.rsplit('_', 1)[-1]}"] = value
        elif key.startswith("threshold_gap_"):
            payload[f"train/threshold/gap_{key.rsplit('_', 1)[-1]}"] = value
        elif key == "ordinal_z_mean":
            payload["train/ordinal_z/mean"] = value
        elif key == "ordinal_z_min":
            payload["train/ordinal_z/min"] = value
        elif key == "ordinal_z_max":
            payload["train/ordinal_z/max"] = value
    return payload


def ordinal_diagnostic_parameter_groups(model: Any) -> dict[str, list[Any]]:
    named_parameters = model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()
    groups: dict[str, list[Any]] = {
        "ordinal_mlp": [],
        "ordinal_threshold_raw": [],
    }
    seen_parameter_ids: set[int] = set()
    for name, parameter in named_parameters:
        if not getattr(parameter, "requires_grad", False):
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        if name.endswith("ordinal_level_head.raw_tau0") or name.endswith("ordinal_level_head.raw_deltas"):
            groups["ordinal_threshold_raw"].append(parameter)
        elif ".ordinal_level_head." in f".{name}":
            groups["ordinal_mlp"].append(parameter)
    return groups


def tensor_collection_stats(tensors: list[Any]) -> dict[str, Any]:
    import torch

    total_sq = 0.0
    total_numel = 0
    max_abs = 0.0
    tensor_count = 0
    for tensor in tensors:
        if tensor is None:
            continue
        detached = tensor.detach().float()
        if detached.numel() == 0:
            continue
        tensor_count += 1
        total_numel += int(detached.numel())
        total_sq += float(torch.sum(detached * detached).cpu())
        max_abs = max(max_abs, float(detached.abs().max().cpu()))
    norm = math.sqrt(total_sq) if total_numel else 0.0
    return {
        "norm": norm,
        "rms": math.sqrt(total_sq / total_numel) if total_numel else 0.0,
        "max_abs": max_abs,
        "tensor_count": tensor_count,
        "element_count": total_numel,
    }


def parameter_snapshot(parameters: list[Any]) -> dict[int, Any]:
    return {id(parameter): parameter.detach().float().cpu().clone() for parameter in parameters}


def parameter_update_stats(parameters: list[Any], before: dict[int, Any]) -> dict[str, Any]:
    updates = []
    for parameter in parameters:
        previous = before.get(id(parameter))
        if previous is None:
            continue
        updates.append(parameter.detach().float().cpu() - previous)
    return tensor_collection_stats(updates)


def selected_hidden_for_level_head(*, model: Any, outputs: Any, level_label_positions: Any) -> Any | None:
    if getattr(outputs, "hidden_states", None) is None:
        return None
    hidden = outputs.hidden_states[-1]
    safe_positions = level_label_positions.clamp(min=0)
    gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    selected_hidden = hidden.gather(1, gather_index)
    head_dtype = next(model.ordinal_level_head.parameters()).dtype
    return selected_hidden.detach().to(dtype=head_dtype)


def ordinal_head_probe_snapshot(
    *,
    model: Any,
    selected_hidden: Any,
    line_positions: Any | None,
    valid_mask: Any | None,
) -> dict[str, Any]:
    import torch

    was_training = model.ordinal_level_head.training
    model.ordinal_level_head.eval()
    try:
        with torch.no_grad():
            head_device = next(model.ordinal_level_head.parameters()).device
            head_line_positions = line_positions.to(device=head_device) if line_positions is not None else None
            head_outputs = model.ordinal_level_head(
                selected_hidden.to(device=head_device),
                line_positions=head_line_positions,
            )
            z = head_outputs.z.detach().float()
            if valid_mask is not None:
                z = z[valid_mask.to(device=z.device)]
            thresholds = head_outputs.thresholds.detach().float()
    finally:
        model.ordinal_level_head.train(was_training)
    return {
        "z": z.cpu(),
        "thresholds": thresholds.cpu(),
    }


def begin_ordinal_step_diagnostics(*, model: Any, outputs: Any, batch_on_device: dict[str, Any]) -> dict[str, Any]:
    groups = ordinal_diagnostic_parameter_groups(model)
    gradients = {
        group_name: tensor_collection_stats([parameter.grad for parameter in parameters if parameter.grad is not None])
        for group_name, parameters in groups.items()
    }
    parameter_snapshots = {
        group_name: parameter_snapshot(parameters)
        for group_name, parameters in groups.items()
    }
    valid_mask = (batch_on_device["level_label_positions"] >= 0) & (batch_on_device["level_labels"] >= 0)
    selected_hidden = selected_hidden_for_level_head(
        model=model,
        outputs=outputs,
        level_label_positions=batch_on_device["level_label_positions"],
    )
    line_positions = batch_on_device.get("level_line_positions")
    probe_before = None
    if selected_hidden is not None:
        probe_before = ordinal_head_probe_snapshot(
            model=model,
            selected_hidden=selected_hidden,
            line_positions=line_positions,
            valid_mask=valid_mask,
        )
    return {
        "groups": groups,
        "gradients": gradients,
        "parameter_snapshots": parameter_snapshots,
        "selected_hidden": selected_hidden,
        "line_positions": line_positions,
        "valid_mask": valid_mask,
        "probe_before": probe_before,
    }


def complete_ordinal_step_diagnostics(*, model: Any, before: dict[str, Any]) -> dict[str, Any]:
    import torch

    row: dict[str, Any] = {}
    groups: dict[str, list[Any]] = before["groups"]
    for group_name, stats in before["gradients"].items():
        row[f"grad_norm_{group_name}"] = stats["norm"]
        row[f"grad_rms_{group_name}"] = stats["rms"]
        row[f"grad_max_abs_{group_name}"] = stats["max_abs"]
        row[f"grad_element_count_{group_name}"] = stats["element_count"]
    for group_name, parameters in groups.items():
        stats = parameter_update_stats(parameters, before["parameter_snapshots"][group_name])
        row[f"update_norm_{group_name}"] = stats["norm"]
        row[f"update_rms_{group_name}"] = stats["rms"]
        row[f"update_max_abs_{group_name}"] = stats["max_abs"]
        row[f"update_element_count_{group_name}"] = stats["element_count"]

    probe_before = before.get("probe_before")
    selected_hidden = before.get("selected_hidden")
    if probe_before is not None and selected_hidden is not None:
        probe_after = ordinal_head_probe_snapshot(
            model=model,
            selected_hidden=selected_hidden,
            line_positions=before.get("line_positions"),
            valid_mask=before.get("valid_mask"),
        )
        z_before = probe_before["z"]
        z_after = probe_after["z"]
        tau_before = probe_before["thresholds"]
        tau_after = probe_after["thresholds"]
        if z_before.numel() and z_after.shape == z_before.shape:
            z_delta = z_after - z_before
            row["effective_z_mean_abs_shift"] = float(z_delta.abs().mean())
            row["effective_z_rms_shift"] = float(torch.sqrt(torch.mean(z_delta * z_delta)))
            row["effective_z_max_abs_shift"] = float(z_delta.abs().max())
            row["probe_z_before_mean"] = float(z_before.mean())
            row["probe_z_after_mean"] = float(z_after.mean())
        tau_delta = tau_after - tau_before
        if tau_delta.numel():
            tau_mean_abs_shift = float(tau_delta.abs().mean())
            row["effective_tau_mean_abs_shift"] = tau_mean_abs_shift
            row["effective_tau_rms_shift"] = float(torch.sqrt(torch.mean(tau_delta * tau_delta)))
            row["effective_tau_max_abs_shift"] = float(tau_delta.abs().max())
            if tau_mean_abs_shift > 0 and "effective_z_mean_abs_shift" in row:
                row["effective_z_to_tau_mean_shift_ratio"] = row["effective_z_mean_abs_shift"] / tau_mean_abs_shift
    return row


def gradient_diagnostics_wandb_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "grad_norm_ordinal_mlp": "train/grad_norm/ordinal_mlp",
        "grad_rms_ordinal_mlp": "train/grad_rms/ordinal_mlp",
        "grad_norm_ordinal_threshold_raw": "train/grad_norm/threshold_raw",
        "grad_rms_ordinal_threshold_raw": "train/grad_rms/threshold_raw",
        "update_norm_ordinal_mlp": "train/update_norm/ordinal_mlp",
        "update_rms_ordinal_mlp": "train/update_rms/ordinal_mlp",
        "update_norm_ordinal_threshold_raw": "train/update_norm/threshold_raw",
        "update_rms_ordinal_threshold_raw": "train/update_rms/threshold_raw",
        "effective_z_mean_abs_shift": "train/effective_shift/z_mean_abs",
        "effective_z_rms_shift": "train/effective_shift/z_rms",
        "effective_z_max_abs_shift": "train/effective_shift/z_max_abs",
        "effective_tau_mean_abs_shift": "train/effective_shift/tau_mean_abs",
        "effective_tau_rms_shift": "train/effective_shift/tau_rms",
        "effective_tau_max_abs_shift": "train/effective_shift/tau_max_abs",
        "effective_z_to_tau_mean_shift_ratio": "train/effective_shift/z_to_tau_mean_ratio",
    }
    return {
        wandb_key: snapshot[source_key]
        for source_key, wandb_key in key_map.items()
        if isinstance(snapshot.get(source_key), (int, float)) and not isinstance(snapshot.get(source_key), bool)
    }


def log_validation_progress(
    *,
    run_dir: Path,
    wandb_run: Any | None,
    run_id: str,
    predictions: list[dict[str, Any]],
    checkpoint_name: str,
    total_count: int,
) -> dict[str, Any]:
    metrics = derive_validation_metrics(
        run_id=run_id,
        predictions=predictions,
        checkpoint=checkpoint_name,
    )
    metrics = serialized_sft.enrich_validation_progress_metrics(
        metrics,
        predictions=predictions,
        total_count=total_count,
    )
    metrics["updated_at"] = utc_now_iso()
    append_jsonl(run_dir / "validation_metrics_progress.jsonl", [metrics])
    serialized_sft.wandb_log(wandb_run, curated_validation_payload("validation_progress", metrics))
    return metrics


def evaluate_validation(
    *,
    run_id: str,
    run_dir: Path,
    validation_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    checkpoint_name: str,
    predictions_path: Path | None = None,
    resume_scope: str = "unit",
    wandb_run: Any | None = None,
    validation_log_every: int = 0,
) -> dict[str, Any]:
    if resume_scope not in {"unit", "checkpoint"}:
        raise ValueError(f"unsupported_validation_resume_scope:{resume_scope}")
    predictions_path = predictions_path or run_dir / "validation_predictions.jsonl"
    existing = read_jsonl(predictions_path, allow_truncated_last_line=True) if predictions_path.exists() else []
    if resume_scope == "checkpoint":
        completed = {(row.get("checkpoint"), row["unit_id"]) for row in existing}
        metric_predictions = [row for row in existing if row.get("checkpoint") == checkpoint_name]
    else:
        completed = {row["unit_id"] for row in existing}
        metric_predictions = list(existing)

    for row in validation_rows:
        unit_id = build_unit_id(row)
        completed_key = (checkpoint_name, unit_id) if resume_scope == "checkpoint" else unit_id
        if completed_key in completed:
            continue
        completion = generate_validation_completion(
            tokenizer=tokenizer,
            model=model,
            prompt=str(row["prompt"]),
            max_new_tokens=max_new_tokens,
        )
        parse_errors: list[str] = []
        try:
            content_blocks, content_text, spans = extract_content_blocks_prediction(completion["raw_text"])
            predicted_blocks = predict_levels_for_content(
                tokenizer=tokenizer,
                model=model,
                prompt=completion["prompt"],
                content_blocks=content_blocks,
                content_text=content_text,
                spans=spans,
            )
        except ValueError as exc:
            content_blocks = []
            predicted_blocks = []
            parse_errors.append(f"structured_output_parse_error:content_blocks_v1:{exc.__class__.__name__}:{exc}")
        evaluation = (
            evaluate_blocks_prediction(
                str(row["target_yaml_normalized"]),
                predicted_blocks,
                prompt_text=str(row["prompt"]),
            )
            if predicted_blocks
            else None
        )
        prediction = {
            "unit_id": unit_id,
            "sample_id": row["sample_id"],
            "prompt_variant": row["prompt_variant"],
            "split": row["split"],
            "checkpoint": checkpoint_name,
            "output_format": CONTENT_BLOCKS_V1,
            "level_alignment_policy": RECORD_PREFIX_STATE,
            "raw_model_output": completion["raw_text"],
            "generated_token_count": completion["generated_token_count"],
            "gold_levels": gold_levels_from_row(row),
            "predicted_content_blocks": content_blocks,
            "predicted_blocks": predicted_blocks,
            "parser_errors": parse_errors,
            "evaluation": evaluation.to_dict() if evaluation else None,
        }
        append_jsonl(predictions_path, [prediction])
        existing.append(prediction)
        metric_predictions.append(prediction)
        completed.add(completed_key)
        completed_count = len(metric_predictions)
        if validation_log_every > 0:
            serialized_sft.log_validation_example(
                run_dir=run_dir,
                wandb_run=wandb_run,
                prediction=prediction,
                completed_count=completed_count,
                total_count=len(validation_rows),
            )
        if validation_log_every > 0 and (
            completed_count % validation_log_every == 0
            or completed_count == len(validation_rows)
        ):
            log_validation_progress(
                run_dir=run_dir,
                wandb_run=wandb_run,
                run_id=run_id,
                predictions=metric_predictions,
                checkpoint_name=checkpoint_name,
                total_count=len(validation_rows),
            )

    return derive_validation_metrics(
        run_id=run_id,
        predictions=metric_predictions,
        checkpoint=checkpoint_name,
    )


def structural_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return serialized_sft.structural_selection_key(metrics)


def select_intermediate_validation_rows(
    validation_rows: list[dict[str, Any]],
    *,
    max_samples: int,
    sample_strategy: str,
    seed: int,
    global_step: int,
) -> list[dict[str, Any]]:
    return serialized_sft.select_intermediate_validation_rows(
        validation_rows,
        max_samples=max_samples,
        sample_strategy=sample_strategy,
        seed=seed,
        global_step=global_step,
    )


def run_intermediate_validation(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    validation_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    wandb_run: Any | None,
    global_step: int,
    checkpoint_name: str,
) -> dict[str, Any]:
    eval_effective_max_samples = min(int(args.eval_max_samples), serialized_sft.INTERMEDIATE_EVAL_MAX_SAMPLES_LIMIT)
    eval_rows = select_intermediate_validation_rows(
        validation_rows,
        max_samples=eval_effective_max_samples,
        sample_strategy=getattr(args, "eval_sample_strategy", "random"),
        seed=int(args.seed),
        global_step=global_step,
    )
    metrics = evaluate_validation(
        run_id=args.run_id,
        run_dir=run_dir,
        validation_rows=eval_rows,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        checkpoint_name=checkpoint_name,
        predictions_path=run_dir / "intermediate_validation_predictions.jsonl",
        resume_scope="checkpoint",
    )
    metrics["global_step"] = global_step
    metrics["eval_requested_max_samples"] = int(args.eval_max_samples)
    metrics["eval_max_samples_limit"] = serialized_sft.INTERMEDIATE_EVAL_MAX_SAMPLES_LIMIT
    metrics["eval_max_samples"] = len(eval_rows)
    metrics["eval_sample_strategy"] = getattr(args, "eval_sample_strategy", "random")
    metrics["eval_sample_seed"] = serialized_sft.intermediate_eval_seed(
        base_seed=int(args.seed),
        global_step=global_step,
    )
    metrics["eval_sample_unit_ids"] = [build_unit_id(row) for row in eval_rows]
    metrics["selection_key"] = list(structural_selection_key(metrics))
    append_jsonl(run_dir / "intermediate_validation_metrics.jsonl", [metrics])
    serialized_sft.wandb_log(wandb_run, curated_validation_payload("validation_sample", metrics), step=global_step)
    model.train()
    return metrics


def init_wandb_run(*, args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    if find_spec("wandb") is None:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. "
            "Install optional LLM dependencies with `uv sync --extra llm`, or run with --wandb-mode disabled."
        )
    import wandb

    tags = list(split_csv(args.wandb_tags))
    for tag in (TWO_HEAD_ORDINAL_SFT, CONTENT_BLOCKS_V1, RECORD_PREFIX_STATE, "ordinal_thresholds"):
        if tag not in tags:
            tags.append(tag)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.run_id,
        id=args.run_id,
        resume="allow",
        mode=args.wandb_mode,
        dir=str(run_dir),
        tags=tags,
        config=config,
    )
    wandb.define_metric("validation_progress/completed_count")
    wandb.define_metric("validation_progress/*", step_metric="validation_progress/completed_count")
    wandb.define_metric("validation_example/completed_count")
    wandb.define_metric("validation_example/*", step_metric="validation_example/completed_count")
    wandb.define_metric("train/threshold/*")
    wandb.define_metric("train/ordinal_z/*")
    wandb.define_metric("train/grad_norm/*")
    wandb.define_metric("train/grad_rms/*")
    wandb.define_metric("train/update_norm/*")
    wandb.define_metric("train/update_rms/*")
    wandb.define_metric("train/effective_shift/*")
    return run


def wandb_log_artifacts(
    wandb_run: Any | None,
    *,
    run_dir: Path,
    final_checkpoint: str | None,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb
    except ImportError:  # pragma: no cover
        return
    artifact = wandb.Artifact(name=f"{wandb_run.id}-two-head-ordinal-sft-artifacts", type="two-head-ordinal-sft-run")
    for filename in (
        "config.json",
        "state.json",
        "metrics.json",
        "train_log.jsonl",
        "threshold_history.jsonl",
        "gradient_diagnostics.jsonl",
        "validation_predictions.jsonl",
        "intermediate_validation_predictions.jsonl",
        "intermediate_validation_metrics.jsonl",
        "validation_metrics_progress.jsonl",
        "validation_example_metrics.jsonl",
        "oom_skipped_batches.jsonl",
        "checkpoint_retention_log.jsonl",
        "checkpoint_prune_errors.jsonl",
    ):
        path = run_dir / filename
        if path.exists():
            artifact.add_file(str(path))
    if final_checkpoint:
        checkpoint_path = Path(final_checkpoint)
        if checkpoint_path.exists():
            artifact.add_dir(str(checkpoint_path), name=checkpoint_path.name)
    wandb_run.log_artifact(artifact)


def build_resume_signature(args: argparse.Namespace) -> dict[str, Any]:
    signature = {
        "model_variant": args.model_variant,
        "serialization": args.serialization,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "base_model_path": str(args.base_model_path),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_keep_last": args.checkpoint_keep_last,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": list(split_csv(args.lora_target_modules)),
        "lambda_level": args.lambda_level,
        "level_class_count": args.level_class_count,
        "ordinal_head_hidden_dims": list(split_int_csv(args.ordinal_head_hidden_dims)),
        "ordinal_head_dropouts": list(split_float_csv(args.ordinal_head_dropouts)),
        "ordinal_position_encoding": args.ordinal_position_encoding,
        "ordinal_position_dim": args.ordinal_position_dim,
        "ordinal_position_frequencies": list(position_frequency_values(args.ordinal_position_frequencies)),
        "ordinal_position_injection": args.ordinal_position_injection,
        "ordinal_film_hidden_dim": args.ordinal_film_hidden_dim,
        "ordinal_film_identity_scale": args.ordinal_film_identity_scale,
        "density_kernel": args.density_kernel,
        "density_kernel_radius": args.density_kernel_radius,
        "max_density_weight": args.max_density_weight,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "target_contract": "prompt -> content_blocks_v1 + FiLM positional ordinal density level head -> blocks_tsv_v1 -> parser -> YAML",
    }
    threshold_lr_multiplier = float(getattr(args, "threshold_learning_rate_multiplier", 1.0))
    if threshold_lr_multiplier != 1.0:
        signature["threshold_learning_rate_multiplier"] = threshold_lr_multiplier
    ordinal_mlp_lr_multiplier = float(getattr(args, "ordinal_mlp_learning_rate_multiplier", 1.0))
    if ordinal_mlp_lr_multiplier != 1.0:
        signature["ordinal_mlp_learning_rate_multiplier"] = ordinal_mlp_lr_multiplier
    if getattr(args, "initial_threshold_center", None) is not None or getattr(args, "initial_threshold_gap", None) is not None:
        signature["initial_threshold_center"] = effective_initial_threshold_center(args)
        signature["initial_threshold_gap"] = effective_initial_threshold_gap(args)
    return signature


def build_optimizer_and_scheduler(args: argparse.Namespace, model: Any, total_optimizer_steps: int):
    import torch
    from transformers import get_linear_schedule_with_warmup

    threshold_lr_multiplier = float(getattr(args, "threshold_learning_rate_multiplier", 1.0))
    ordinal_mlp_lr_multiplier = float(getattr(args, "ordinal_mlp_learning_rate_multiplier", 1.0))
    if threshold_lr_multiplier == 1.0 and ordinal_mlp_lr_multiplier == 1.0:
        return serialized_sft.build_optimizer_and_scheduler(args, model, total_optimizer_steps)
    if threshold_lr_multiplier <= 0:
        raise ValueError("threshold_learning_rate_multiplier must be positive")
    if ordinal_mlp_lr_multiplier <= 0:
        raise ValueError("ordinal_mlp_learning_rate_multiplier must be positive")

    named_parameters = model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()
    base_parameters = []
    ordinal_mlp_parameters = []
    threshold_parameters = []
    seen_parameter_ids: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        if name.endswith("ordinal_level_head.raw_tau0") or name.endswith("ordinal_level_head.raw_deltas"):
            threshold_parameters.append(parameter)
        elif ".ordinal_level_head." in f".{name}":
            ordinal_mlp_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if base_parameters:
        parameter_groups.append(
            {
                "params": base_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "base_lora",
            }
        )
    if ordinal_mlp_parameters:
        parameter_groups.append(
            {
                "params": ordinal_mlp_parameters,
                "lr": args.learning_rate * ordinal_mlp_lr_multiplier,
                "weight_decay": args.weight_decay,
                "name": "ordinal_mlp",
            }
        )
    if threshold_parameters:
        parameter_groups.append(
            {
                "params": threshold_parameters,
                "lr": args.learning_rate * threshold_lr_multiplier,
                "weight_decay": 0.0,
                "name": "ordinal_thresholds",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    return optimizer, scheduler


def optimizer_learning_rates_by_group(optimizer: Any, scheduler: Any) -> dict[str, float]:
    scheduler_lrs = list(scheduler.get_last_lr())
    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("name") or f"group_{index}"
        if index < len(scheduler_lrs):
            rates[str(name)] = float(scheduler_lrs[index])
        else:
            rates[str(name)] = float(group.get("lr", 0.0))
    if "base_lora" not in rates and scheduler_lrs:
        rates["base_lora"] = float(scheduler_lrs[0])
    if "ordinal_mlp" not in rates:
        rates["ordinal_mlp"] = rates.get("base_lora", float(scheduler_lrs[0]) if scheduler_lrs else 0.0)
    if "ordinal_thresholds" not in rates:
        rates["ordinal_thresholds"] = rates.get("base_lora", float(scheduler_lrs[0]) if scheduler_lrs else 0.0)
    return rates


def train(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    run_dir = args.output_dir / args.run_id
    wandb_run = init_wandb_run(args=args, config=config, run_dir=run_dir)
    density_weights = compute_level_density_weights(
        train_rows,
        level_class_count=args.level_class_count,
        kernel=args.density_kernel,
        radius=args.density_kernel_radius,
        max_weight=args.max_density_weight,
    )
    args.level_density_weights = density_weights["weights"]
    tokenizer, model = load_model_and_tokenizer(args)
    train_dataset = TwoHeadSFTDataset(train_rows, tokenizer, max_seq_length=args.max_seq_length)
    collator = TwoHeadSFTCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_optimizer_steps = max(optimizer_steps_per_epoch * args.epochs, 1)
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_optimizer_steps)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    checkpoint_path = serialized_sft.latest_checkpoint(run_dir)
    if checkpoint_path is not None and (checkpoint_path / "training_state.pt").exists():
        model, checkpoint_state = load_checkpoint_state(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        state.update(checkpoint_state)
        serialized_sft.update_state(run_dir, state)

    device = model_input_device(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_lm_loss = 0.0
    rolling_ordinal_level_loss = 0.0
    rolling_count = 0
    oom_skipped_unit_ids = serialized_sft.read_oom_skipped_unit_ids(run_dir) if args.oom_recovery == "skip_batch" else set()
    oom_skip_count = len(oom_skipped_unit_ids)

    for epoch in range(int(state.get("epoch", 0)), args.epochs):
        start_batch_index = int(state.get("next_batch_index", 0)) if epoch == int(state.get("epoch", 0)) else 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < start_batch_index:
                continue
            unit_ids = [str(unit_id) for unit_id in batch.get("unit_ids", [])]
            next_epoch = epoch
            next_batch_index = batch_index + 1
            if next_batch_index >= len(train_loader):
                next_epoch = epoch + 1
                next_batch_index = 0
            if args.oom_recovery == "skip_batch" and any(unit_id in oom_skipped_unit_ids for unit_id in unit_ids):
                state = serialized_sft.update_state(
                    run_dir,
                    state,
                    status="running",
                    epoch=next_epoch,
                    next_batch_index=next_batch_index,
                    oom_skipped_batches=oom_skip_count,
                )
                continue
            batch_on_device = None
            outputs = None
            loss = None
            try:
                batch_on_device = move_batch_to_device(batch, device)
                outputs = model(**batch_on_device)
                loss = outputs.loss
                (loss / args.gradient_accumulation_steps).backward()
            except torch.OutOfMemoryError as exc:
                if args.oom_recovery != "skip_batch":
                    raise
                cuda_memory = serialized_sft.cuda_memory_snapshot(torch)
                optimizer.zero_grad(set_to_none=True)
                rolling_loss = 0.0
                rolling_lm_loss = 0.0
                rolling_ordinal_level_loss = 0.0
                rolling_count = 0
                del batch_on_device, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                oom_row = serialized_sft.log_oom_skipped_batch(
                    run_dir=run_dir,
                    run_id=args.run_id,
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=int(state.get("global_step", 0)),
                    batch=batch,
                    error=exc,
                    cuda_memory=cuda_memory,
                )
                oom_skipped_unit_ids.update(unit_ids)
                oom_skip_count += 1
                state = serialized_sft.update_state(
                    run_dir,
                    state,
                    status="running",
                    epoch=next_epoch,
                    next_batch_index=next_batch_index,
                    oom_skipped_batches=oom_skip_count,
                    last_oom_skipped_batch=oom_row,
                )
                serialized_sft.wandb_log(
                    wandb_run,
                    {
                        "train/oom_skipped_batches": oom_skip_count,
                        "train/oom_skipped_samples": len(oom_skipped_unit_ids),
                    },
                    step=int(state.get("global_step", 0)),
                )
                if args.max_oom_skips and oom_skip_count > args.max_oom_skips:
                    raise RuntimeError(f"max_oom_skips_exceeded:{oom_skip_count}>{args.max_oom_skips}") from exc
                continue
            rolling_loss += float(loss.detach().cpu())
            rolling_lm_loss += float(outputs.lm_loss.detach().cpu()) if outputs.lm_loss is not None else 0.0
            rolling_ordinal_level_loss += float(outputs.ordinal_level_loss.detach().cpu()) if outputs.ordinal_level_loss is not None else 0.0
            rolling_count += 1

            should_step = (
                rolling_count >= args.gradient_accumulation_steps
                or (batch_index + 1 == len(train_loader) and rolling_count > 0)
            )
            if not should_step:
                continue

            diagnostic_before = None
            if getattr(args, "ordinal_gradient_diagnostics", True):
                diagnostic_before = begin_ordinal_step_diagnostics(
                    model=model,
                    outputs=outputs,
                    batch_on_device=batch_on_device,
                )
            move_optimizer_state_to_active_devices(optimizer)
            optimizer.step()
            gradient_diagnostics = (
                complete_ordinal_step_diagnostics(model=model, before=diagnostic_before)
                if diagnostic_before is not None
                else {}
            )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step = int(state.get("global_step", 0)) + 1
            average_loss = rolling_loss / max(rolling_count, 1)
            average_lm_loss = rolling_lm_loss / max(rolling_count, 1)
            average_ordinal_level_loss = rolling_ordinal_level_loss / max(rolling_count, 1)
            state = serialized_sft.update_state(
                run_dir,
                state,
                status="running",
                global_step=global_step,
                epoch=next_epoch,
                next_batch_index=next_batch_index,
                last_loss=average_loss,
                last_lm_loss=average_lm_loss,
                last_ordinal_level_loss=average_ordinal_level_loss,
                oom_skipped_batches=oom_skip_count,
            )
            threshold_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "updated_at": utc_now_iso(),
                **threshold_snapshot(
                    model=model,
                    outputs=outputs,
                    valid_mask=(batch_on_device["level_label_positions"] >= 0)
                    & (batch_on_device["level_labels"] >= 0),
                ),
            }
            append_jsonl(run_dir / "threshold_history.jsonl", [threshold_row])
            gradient_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "updated_at": utc_now_iso(),
                **gradient_diagnostics,
            }
            if gradient_diagnostics:
                append_jsonl(run_dir / "gradient_diagnostics.jsonl", [gradient_row])
            learning_rates = optimizer_learning_rates_by_group(optimizer, scheduler)
            log_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "loss": average_loss,
                "lm_loss": average_lm_loss,
                "ordinal_level_loss": average_ordinal_level_loss,
                "lambda_level": args.lambda_level,
                "learning_rate": learning_rates["base_lora"],
                "ordinal_mlp_learning_rate": learning_rates["ordinal_mlp"],
                "threshold_learning_rate": learning_rates["ordinal_thresholds"],
                "updated_at": utc_now_iso(),
            }
            for key in (
                "grad_rms_ordinal_mlp",
                "grad_rms_ordinal_threshold_raw",
                "update_rms_ordinal_mlp",
                "update_rms_ordinal_threshold_raw",
                "effective_z_mean_abs_shift",
                "effective_tau_mean_abs_shift",
                "effective_z_to_tau_mean_shift_ratio",
            ):
                if key in gradient_diagnostics:
                    log_row[key] = gradient_diagnostics[key]
            append_jsonl(run_dir / "train_log.jsonl", [log_row])
            train_payload = {
                    "train/loss": average_loss,
                    "train/lm_loss": average_lm_loss,
                    "train/ordinal_level_loss": average_ordinal_level_loss,
                    "train/lambda_level": args.lambda_level,
                    "train/learning_rate": log_row["learning_rate"],
                    "train/ordinal_mlp/learning_rate": log_row["ordinal_mlp_learning_rate"],
                    "train/threshold/learning_rate": log_row["threshold_learning_rate"],
                    "train/epoch": epoch,
                    "train/batch_index": batch_index,
                }
            train_payload.update(threshold_wandb_payload(threshold_row))
            train_payload.update(gradient_diagnostics_wandb_payload(gradient_row))
            serialized_sft.wandb_log(wandb_run, train_payload, step=global_step)
            rolling_loss = 0.0
            rolling_lm_loss = 0.0
            rolling_ordinal_level_loss = 0.0
            rolling_count = 0

            saved_checkpoint: Path | None = None
            if global_step % args.checkpoint_steps == 0 or next_epoch >= args.epochs:
                saved_checkpoint = save_checkpoint(
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                )
                removed_checkpoints = serialized_sft.prune_old_checkpoints(
                    run_dir,
                    keep_last=args.checkpoint_keep_last,
                )
                if removed_checkpoints:
                    append_jsonl(
                        run_dir / "checkpoint_retention_log.jsonl",
                        [
                            {
                                "run_id": args.run_id,
                                "global_step": global_step,
                                "saved_checkpoint": str(saved_checkpoint),
                                "removed_checkpoints": removed_checkpoints,
                                "keep_last": args.checkpoint_keep_last,
                                "updated_at": utc_now_iso(),
                            }
                        ],
                    )

            if args.eval_checkpoint_steps and global_step % args.eval_checkpoint_steps == 0:
                checkpoint_name = (
                    saved_checkpoint.name
                    if saved_checkpoint is not None and saved_checkpoint.name.endswith(str(global_step))
                    else f"in-memory-step-{global_step}"
                )
                run_intermediate_validation(
                    args=args,
                    run_dir=run_dir,
                    validation_rows=validation_rows,
                    tokenizer=tokenizer,
                    model=model,
                    wandb_run=wandb_run,
                    global_step=global_step,
                    checkpoint_name=checkpoint_name,
                )

    state = serialized_sft.update_state(run_dir, state, status="trained")
    final_checkpoint = save_checkpoint(
        run_dir=run_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
    )
    removed_checkpoints = serialized_sft.prune_old_checkpoints(
        run_dir,
        keep_last=args.checkpoint_keep_last,
    )
    if removed_checkpoints:
        append_jsonl(
            run_dir / "checkpoint_retention_log.jsonl",
            [
                {
                    "run_id": args.run_id,
                    "global_step": int(state.get("global_step", 0)),
                    "saved_checkpoint": str(final_checkpoint),
                    "removed_checkpoints": removed_checkpoints,
                    "keep_last": args.checkpoint_keep_last,
                    "updated_at": utc_now_iso(),
                }
            ],
        )

    if args.skip_final_eval:
        metrics = {
            "run_id": args.run_id,
            "model_variant": args.model_variant,
            "serialization": args.serialization,
            "status": "trained_without_final_eval",
            "final_checkpoint": str(final_checkpoint),
        }
    else:
        metrics = evaluate_validation(
            run_id=args.run_id,
            run_dir=run_dir,
            validation_rows=validation_rows,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            checkpoint_name=final_checkpoint.name,
            wandb_run=wandb_run,
            validation_log_every=args.validation_log_every,
        )
        metrics["selection_key"] = list(structural_selection_key(metrics))
        metrics["best_checkpoint"] = str(final_checkpoint)
        serialized_sft.wandb_log(
            wandb_run,
            curated_validation_payload("validation", metrics),
            step=int(state.get("global_step", 0)),
        )

    write_json(run_dir / "metrics.json", metrics)
    serialized_sft.update_state(
        run_dir,
        state,
        status="completed",
        completed_at=utc_now_iso(),
        best_checkpoint=metrics.get("best_checkpoint"),
        best_validation_score=metrics.get("selection_key"),
    )
    serialized_sft.wandb_log(
        wandb_run,
        curated_validation_payload("final", metrics),
        step=int(state.get("global_step", 0)),
    )
    if args.wandb_log_artifacts:
        wandb_log_artifacts(
            wandb_run,
            run_dir=run_dir,
            final_checkpoint=metrics.get("best_checkpoint") or metrics.get("final_checkpoint"),
        )
    serialized_sft.wandb_finish(wandb_run)
    return metrics


def main() -> None:
    loaded_env_keys = sorted(serialized_sft.load_env_file().keys())
    args = parse_args()
    validate_position_config(args)
    run_dir = args.output_dir / args.run_id
    train_rows = load_sft_rows(args.train_file, max_samples=args.max_train_samples, split_name="train")
    validation_rows = load_sft_rows(
        args.validation_file,
        max_samples=args.max_validation_samples,
        split_name="validation",
    )
    density_weights = compute_level_density_weights(
        train_rows,
        level_class_count=args.level_class_count,
        kernel=args.density_kernel,
        radius=args.density_kernel_radius,
        max_weight=args.max_density_weight,
    )
    args.level_density_weights = density_weights["weights"]
    initial_threshold_center = effective_initial_threshold_center(args)
    initial_threshold_gap = effective_initial_threshold_gap(args)
    initial_thresholds = initial_threshold_values(
        level_class_count=args.level_class_count,
        center=initial_threshold_center,
        gap=initial_threshold_gap,
    )
    config = {
        "run_id": args.run_id,
        "model_variant": args.model_variant,
        "serialization": args.serialization,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "base_model_path": str(args.base_model_path),
        "row_count_train": len(train_rows),
        "row_count_validation": len(validation_rows),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "base_learning_rate": args.learning_rate,
        "ordinal_mlp_learning_rate_multiplier": args.ordinal_mlp_learning_rate_multiplier,
        "ordinal_mlp_learning_rate": args.learning_rate * args.ordinal_mlp_learning_rate_multiplier,
        "threshold_learning_rate_multiplier": args.threshold_learning_rate_multiplier,
        "threshold_learning_rate": args.learning_rate * args.threshold_learning_rate_multiplier,
        "ordinal_gradient_diagnostics": bool(getattr(args, "ordinal_gradient_diagnostics", True)),
        "initial_threshold_center": initial_threshold_center,
        "initial_threshold_gap": initial_threshold_gap,
        "initial_thresholds": initial_thresholds,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_steps": args.checkpoint_steps,
        "checkpoint_keep_last": args.checkpoint_keep_last,
        "eval_checkpoint_steps": args.eval_checkpoint_steps,
        "eval_max_samples": args.eval_max_samples,
        "eval_sample_strategy": args.eval_sample_strategy,
        "validation_log_every": args.validation_log_every,
        "oom_recovery": args.oom_recovery,
        "max_oom_skips": args.max_oom_skips,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "seed": args.seed,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(split_csv(args.lora_target_modules)),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "ordinal_level_head": {
            "type": "learned_threshold_ordinal_film_positional",
            "class_count": args.level_class_count,
            "hidden_dims": list(split_int_csv(args.ordinal_head_hidden_dims)),
            "dropouts": list(split_float_csv(args.ordinal_head_dropouts)),
            "position_encoding": {
                "type": args.ordinal_position_encoding,
                "dim": args.ordinal_position_dim,
                "frequencies": list(position_frequency_values(args.ordinal_position_frequencies)),
                "injection": args.ordinal_position_injection,
                "causal": True,
                "uses_final_document_length": False,
                "dropout": 0.0,
                "film_hidden_dim": args.ordinal_film_hidden_dim,
                "film_identity_scale": args.ordinal_film_identity_scale,
                "film_initialization": "last_linear_zero_identity_start",
            },
            "loss": "density_weighted_ordinal_bce",
            "lambda_level": args.lambda_level,
            "threshold_count": args.level_class_count - 1,
            "threshold_parameterization": "tau_0_plus_cumulative_softplus_deltas",
            "threshold_initialization": {
                "policy": "centered_equidistant_trainable_thresholds",
                "center": initial_threshold_center,
                "gap": initial_threshold_gap,
                "thresholds": initial_thresholds,
                "trainable_after_initialization": True,
            },
            "projector_optimizer": {
                "learning_rate_multiplier": args.ordinal_mlp_learning_rate_multiplier,
                "learning_rate": args.learning_rate * args.ordinal_mlp_learning_rate_multiplier,
                "weight_decay": args.weight_decay,
            },
            "threshold_optimizer": {
                "learning_rate_multiplier": args.threshold_learning_rate_multiplier,
                "learning_rate": args.learning_rate * args.threshold_learning_rate_multiplier,
                "weight_decay": 0.0,
            },
            "density_weights": density_weights,
            "huber_auxiliary": False,
            "lora_target": False,
        },
        "training_policy": {
            "shuffle_train": False,
            "label_policy": "prompt tokens masked with -100; content_blocks_v1 target tokens supervised; level labels supervised only at record_prefix_state positions; causal line_position is passed only to the ordinal head",
            "checkpoint_policy": "save adapter, ordinal level head, optimizer, and scheduler state at checkpoint_steps and completion",
            "checkpoint_retention_policy": "checkpoint_keep_last=0 keeps all; otherwise older local checkpoints are deleted after a new checkpoint is saved",
            "oom_recovery_policy": "fail by default; oom_recovery=skip_batch logs CUDA OOM microbatches to oom_skipped_batches.jsonl and excludes them from optimizer updates",
            "ordinal_gradient_diagnostics_policy": "when enabled, log raw gradient RMS/norm, effective parameter-update RMS/norm, and same-hidden z-vs-tau movement for the ordinal projector and threshold parameters",
        },
        "wandb_metric_policy": {
            "dashboard": "curated",
            "logs_thresholds": True,
            "logs_ordinal_gradient_diagnostics": bool(getattr(args, "ordinal_gradient_diagnostics", True)),
            "logs_ordinal_mlp_learning_rate": True,
            "keeps_auxiliary_text_metrics_local": True,
        },
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name or args.run_id,
            "tags": list(split_csv(args.wandb_tags)),
            "log_artifacts": args.wandb_log_artifacts,
            "env_file_loaded_keys": loaded_env_keys,
            "api_key_available": bool(os.environ.get("WANDB_API_KEY")),
        },
        "dry_run": args.dry_run,
        "skip_final_eval": args.skip_final_eval,
        "model_checks": serialized_sft.inspect_model_path(args.base_model_path),
        "resume_signature": build_resume_signature(args),
    }
    state = serialized_sft.initialize_run(run_dir, config, dry_run=args.dry_run)
    if args.dry_run:
        write_json(
            run_dir / "metrics.json",
            {
                "dry_run": True,
                "row_count_train": len(train_rows),
                "row_count_validation": len(validation_rows),
                "ready_for_full_run": config["model_checks"]["ready_for_full_run"],
                "model_warnings": config["model_checks"]["warnings"],
            },
        )
        write_json(run_dir / "state.json", state)
        print({"output_dir": str(run_dir), "dry_run": True, **json.loads((run_dir / "metrics.json").read_text())})
        return

    if not config["model_checks"]["ready_for_full_run"]:
        raise RuntimeError(f"Two-head SFT full run is not ready. See model_checks: {config['model_checks']}")
    metrics = train(args, train_rows, validation_rows, config=config)
    print({"output_dir": str(run_dir), **metrics})


if __name__ == "__main__":
    main()
