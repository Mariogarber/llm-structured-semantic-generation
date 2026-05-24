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


TWO_HEAD_SFT = "two_head_sft"
CONTENT_BLOCKS_V1 = "content_blocks_v1"
RECORD_PREFIX_STATE = "record_prefix_state"
PROMPT_TARGET_SEPARATOR = "\n\n"
IGNORE_INDEX = -100
DEFAULT_LEVEL_CLASS_COUNT = 9
DEFAULT_LEVEL_HEAD_HIDDEN_DIM = 256
DEFAULT_LEVEL_HEAD_DROPOUT = 0.05


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
        description="Train Kubernetes v1 Architecture B: two_head_sft with LoRA and an explicit level head."
    )
    parser.add_argument("--model-variant", choices=[TWO_HEAD_SFT], default=TWO_HEAD_SFT)
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
        default=REPO_ROOT / "results" / "two_head_sft_kubernetes_v1",
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
    parser.add_argument("--lambda-level", type=float, default=1.0)
    parser.add_argument("--level-class-count", type=serialized_sft.positive_int, default=DEFAULT_LEVEL_CLASS_COUNT)
    parser.add_argument("--level-head-hidden-dim", type=serialized_sft.positive_int, default=DEFAULT_LEVEL_HEAD_HIDDEN_DIM)
    parser.add_argument("--level-head-dropout", type=float, default=DEFAULT_LEVEL_HEAD_DROPOUT)
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
            level_labels.append(item["level_labels"] + [IGNORE_INDEX] * line_pad_count)
            unit_ids.append(item["unit_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "level_token_labels": torch.tensor(level_token_labels, dtype=torch.long),
            "level_label_positions": torch.tensor(level_label_positions, dtype=torch.long),
            "level_labels": torch.tensor(level_labels, dtype=torch.long),
            "unit_ids": unit_ids,
        }


class TwoHeadQwenForSFT:
    def __init__(
        self,
        backbone: Any,
        *,
        hidden_size: int,
        level_class_count: int,
        level_head_hidden_dim: int,
        level_head_dropout: float,
        lambda_level: float,
    ) -> None:
        import torch

        class _TwoHeadModule(torch.nn.Module):
            def __init__(self, outer: "TwoHeadQwenForSFT") -> None:
                super().__init__()
                self.outer = outer
                self.backbone = outer.backbone
                self.level_head = outer.level_head

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self.outer.forward(*args, **kwargs)

            def generate(self, *args: Any, **kwargs: Any) -> Any:
                return self.backbone.generate(*args, **kwargs)

            @property
            def config(self) -> Any:
                return self.backbone.config

        self.backbone = backbone
        self.lambda_level = float(lambda_level)
        self.level_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, level_head_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(level_head_dropout),
            torch.nn.Linear(level_head_hidden_dim, level_class_count),
        )
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
        level_loss = None
        if level_label_positions is not None:
            safe_positions = level_label_positions.clamp(min=0)
            gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            selected_hidden = hidden.gather(1, gather_index)
            head_device = next(self.level_head.parameters()).device
            if head_device != selected_hidden.device:
                self.level_head.to(selected_hidden.device)
            head_dtype = next(self.level_head.parameters()).dtype
            selected_hidden = selected_hidden.to(dtype=head_dtype)
            level_logits = self.level_head(selected_hidden)
            if level_labels is not None:
                valid = (level_label_positions >= 0) & (level_labels != IGNORE_INDEX)
                if bool(valid.any()):
                    level_loss = torch.nn.functional.cross_entropy(
                        level_logits[valid],
                        level_labels[valid],
                    )
                else:
                    level_loss = level_logits.sum() * 0.0

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
            level_logits=level_logits,
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
    model = TwoHeadQwenForSFT(
        backbone,
        hidden_size=int(config.hidden_size),
        level_class_count=args.level_class_count,
        level_head_hidden_dim=args.level_head_hidden_dim,
        level_head_dropout=args.level_head_dropout,
        lambda_level=args.lambda_level,
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
    torch.save(model.level_head.state_dict(), path / "level_head.pt")
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
        else:
            adapter_state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
        set_peft_model_state_dict(model.backbone, adapter_state, adapter_name="default")
    level_head_path = checkpoint_path / "level_head.pt"
    if level_head_path.exists():
        model.level_head.load_state_dict(torch.load(level_head_path, map_location="cpu"))
    payload = torch.load(checkpoint_path / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
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
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            level_label_positions=positions,
        )
    predicted_levels = outputs.level_logits.argmax(dim=-1)[0].detach().cpu().tolist()
    return [
        {
            "document_index": int(block["document_index"]),
            "line_index": int(block["line_index"]),
            "level": int(predicted_levels[index]),
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
        "model_variant": TWO_HEAD_SFT,
        "serialization": CONTENT_BLOCKS_V1,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "structured_output_parse_success_rate": len(evaluated_results) / len(predictions) if predictions else 0.0,
    }
    metrics.update(summarize_evaluations(evaluated_results))
    return metrics


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
    serialized_sft.wandb_log(wandb_run, serialized_sft.numeric_payload("validation_progress", metrics))
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
    serialized_sft.wandb_log_numeric_metrics(wandb_run, metrics, prefix="validation_sample", step=global_step)
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
    for tag in (TWO_HEAD_SFT, CONTENT_BLOCKS_V1, RECORD_PREFIX_STATE):
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
    artifact = wandb.Artifact(name=f"{wandb_run.id}-two-head-sft-artifacts", type="two-head-sft-run")
    for filename in (
        "config.json",
        "state.json",
        "metrics.json",
        "train_log.jsonl",
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
    return {
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
        "level_head_hidden_dim": args.level_head_hidden_dim,
        "level_head_dropout": args.level_head_dropout,
        "level_alignment_policy": RECORD_PREFIX_STATE,
        "target_contract": "prompt -> content_blocks_v1 + level head -> blocks_tsv_v1 -> parser -> YAML",
    }


def build_optimizer_and_scheduler(args: argparse.Namespace, model: Any, total_optimizer_steps: int):
    return serialized_sft.build_optimizer_and_scheduler(args, model, total_optimizer_steps)


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
    rolling_level_loss = 0.0
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
                rolling_level_loss = 0.0
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
            rolling_level_loss += float(outputs.level_loss.detach().cpu()) if outputs.level_loss is not None else 0.0
            rolling_count += 1

            should_step = (
                rolling_count >= args.gradient_accumulation_steps
                or (batch_index + 1 == len(train_loader) and rolling_count > 0)
            )
            if not should_step:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step = int(state.get("global_step", 0)) + 1
            average_loss = rolling_loss / max(rolling_count, 1)
            average_lm_loss = rolling_lm_loss / max(rolling_count, 1)
            average_level_loss = rolling_level_loss / max(rolling_count, 1)
            state = serialized_sft.update_state(
                run_dir,
                state,
                status="running",
                global_step=global_step,
                epoch=next_epoch,
                next_batch_index=next_batch_index,
                last_loss=average_loss,
                last_lm_loss=average_lm_loss,
                last_level_loss=average_level_loss,
                oom_skipped_batches=oom_skip_count,
            )
            log_row = {
                "run_id": args.run_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "loss": average_loss,
                "lm_loss": average_lm_loss,
                "level_loss": average_level_loss,
                "lambda_level": args.lambda_level,
                "learning_rate": scheduler.get_last_lr()[0],
                "updated_at": utc_now_iso(),
            }
            append_jsonl(run_dir / "train_log.jsonl", [log_row])
            serialized_sft.wandb_log(
                wandb_run,
                {
                    "train/loss": average_loss,
                    "train/lm_loss": average_lm_loss,
                    "train/level_loss": average_level_loss,
                    "train/lambda_level": args.lambda_level,
                    "train/learning_rate": log_row["learning_rate"],
                    "train/epoch": epoch,
                    "train/batch_index": batch_index,
                },
                step=global_step,
            )
            rolling_loss = 0.0
            rolling_lm_loss = 0.0
            rolling_level_loss = 0.0
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
        serialized_sft.wandb_log_numeric_metrics(
            wandb_run,
            metrics,
            prefix="validation",
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
    serialized_sft.wandb_log_numeric_metrics(
        wandb_run,
        metrics,
        prefix="final",
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
    run_dir = args.output_dir / args.run_id
    train_rows = load_sft_rows(args.train_file, max_samples=args.max_train_samples, split_name="train")
    validation_rows = load_sft_rows(
        args.validation_file,
        max_samples=args.max_validation_samples,
        split_name="validation",
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
        "level_head": {
            "type": "mlp_classifier",
            "class_count": args.level_class_count,
            "hidden_dim": args.level_head_hidden_dim,
            "dropout": args.level_head_dropout,
            "loss": "cross_entropy",
            "lambda_level": args.lambda_level,
            "lora_target": False,
        },
        "training_policy": {
            "shuffle_train": False,
            "label_policy": "prompt tokens masked with -100; content_blocks_v1 target tokens supervised; level labels supervised only at record_prefix_state positions",
            "checkpoint_policy": "save adapter, level head, optimizer, and scheduler state at checkpoint_steps and completion",
            "checkpoint_retention_policy": "checkpoint_keep_last=0 keeps all; otherwise older local checkpoints are deleted after a new checkpoint is saved",
            "oom_recovery_policy": "fail by default; oom_recovery=skip_batch logs CUDA OOM microbatches to oom_skipped_batches.jsonl and excludes them from optimizer updates",
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
