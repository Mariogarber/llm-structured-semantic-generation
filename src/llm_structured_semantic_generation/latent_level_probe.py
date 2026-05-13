from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset_io import read_jsonl, write_json, write_jsonl
from .resumable_run import RunCompatibilityError, utc_now_iso
from .sft_serialization import deserialize_training_blocks
from .structure import YAMLBlock


FEATURE_STRATEGIES = (
    "record_prefix_state",
    "line_prefix_state",
    "line_first_token",
    "line_last_token",
    "line_mean",
)
PROBE_TYPES = ("majority", "previous_level", "linear", "mlp")


@dataclass(frozen=True)
class LineSpan:
    line_id: str
    unit_id: str
    sample_id: str
    prompt_variant: str
    split: str
    document_index: int
    line_index: int
    level: int
    line_text: str
    record_start: int
    record_end: int
    line_text_start: int
    line_text_end: int
    line_position: int


@dataclass(frozen=True)
class ContentOnlyExample:
    unit_id: str
    sample_id: str
    prompt_variant: str
    split: str
    prompt: str
    content_text: str
    full_text: str
    line_spans: list[LineSpan]


def build_unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['prompt_variant']}"


def safe_unit_filename(unit_id: str) -> str:
    digest = hashlib.sha1(unit_id.encode("utf-8")).hexdigest()[:16]
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", unit_id).strip("_")
    return f"{normalized[:80]}-{digest}.json"


def read_sft_rows(paths: Iterable[Path], *, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        path_rows = read_jsonl(path)
        if max_samples is not None:
            path_rows = path_rows[:max_samples]
        rows.extend(path_rows)
    return rows


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


def _escape_line_text(line_text: str) -> str:
    return line_text.replace("\\", "\\\\").replace("\t", "\\t")


def _unescape_line_text(line_text: str) -> str:
    return line_text.replace("\\t", "\t").replace("\\\\", "\\")


def build_content_only_example(row: dict[str, Any]) -> ContentOnlyExample:
    blocks = deserialize_training_blocks(row["target"])
    unit_id = build_unit_id(row)
    prompt = build_content_only_prompt(row["prompt"])

    content_parts = ["<content_blocks>\n"]
    spans: list[LineSpan] = []
    cursor = len(content_parts[0])
    for line_position, block in enumerate(blocks):
        prefix = f"{block.document_index}\t{block.line_index}\t"
        escaped_text = _escape_line_text(block.line_text)
        record = f"{prefix}{escaped_text}\n"
        record_start = cursor
        text_start = record_start + len(prefix)
        text_end = text_start + len(escaped_text)
        record_end = record_start + len(record)
        line_id = f"{unit_id}::{block.document_index}::{block.line_index}"
        spans.append(
            LineSpan(
                line_id=line_id,
                unit_id=unit_id,
                sample_id=str(row["sample_id"]),
                prompt_variant=str(row["prompt_variant"]),
                split=str(row["split"]),
                document_index=int(block.document_index),
                line_index=int(block.line_index),
                level=int(block.level),
                line_text=block.line_text,
                record_start=record_start,
                record_end=record_end,
                line_text_start=text_start,
                line_text_end=text_end,
                line_position=line_position,
            )
        )
        content_parts.append(record)
        cursor = record_end
    content_parts.append("</content_blocks>")
    content_text = "".join(content_parts)
    full_text = f"{prompt}\n\n{content_text}"
    prompt_offset = len(prompt) + 2
    shifted_spans = [
        LineSpan(
            **{
                **span.__dict__,
                "record_start": span.record_start + prompt_offset,
                "record_end": span.record_end + prompt_offset,
                "line_text_start": span.line_text_start + prompt_offset,
                "line_text_end": span.line_text_end + prompt_offset,
            }
        )
        for span in spans
    ]
    return ContentOnlyExample(
        unit_id=unit_id,
        sample_id=str(row["sample_id"]),
        prompt_variant=str(row["prompt_variant"]),
        split=str(row["split"]),
        prompt=prompt,
        content_text=content_text,
        full_text=full_text,
        line_spans=shifted_spans,
    )


def assert_no_gold_level_leakage(example: ContentOnlyExample, blocks: list[YAMLBlock] | None = None) -> None:
    body = example.content_text
    for line in body.splitlines():
        if not line or line in {"<content_blocks>", "</content_blocks>"}:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"content_only_line_must_have_3_fields:{line!r}")


def metadata_from_span(span: LineSpan) -> dict[str, Any]:
    return {
        "line_id": span.line_id,
        "unit_id": span.unit_id,
        "sample_id": span.sample_id,
        "prompt_variant": span.prompt_variant,
        "split": span.split,
        "document_index": span.document_index,
        "line_index": span.line_index,
        "line_position": span.line_position,
        "level": span.level,
        "line_text": span.line_text,
    }


def _token_indices_for_span(
    offsets: list[tuple[int, int]],
    *,
    start: int,
    end: int,
) -> list[int]:
    indices: list[int] = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        if token_end > start and token_start < end:
            indices.append(index)
    return indices


def _prefix_index(offsets: list[tuple[int, int]], *, position: int) -> int | None:
    candidates = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start != token_end and token_end <= position
    ]
    return candidates[-1] if candidates else None


def _record_token_indices(offsets: list[tuple[int, int]], span: LineSpan) -> list[int]:
    return _token_indices_for_span(offsets, start=span.record_start, end=span.record_end)


def select_line_feature(
    hidden: np.ndarray,
    offsets: list[tuple[int, int]],
    span: LineSpan,
    strategy: str,
) -> np.ndarray:
    if strategy not in FEATURE_STRATEGIES:
        raise ValueError(f"unsupported_feature_strategy:{strategy}")

    text_indices = _token_indices_for_span(
        offsets,
        start=span.line_text_start,
        end=span.line_text_end,
    )
    fallback_indices = _record_token_indices(offsets, span)
    indices = text_indices or fallback_indices
    if not indices:
        raise ValueError(f"no_tokens_aligned_for_line:{span.line_id}")

    if strategy == "line_mean":
        return hidden[indices].mean(axis=0)
    if strategy == "line_first_token":
        return hidden[indices[0]]
    if strategy == "line_last_token":
        return hidden[indices[-1]]

    prefix_position = span.record_start if strategy == "record_prefix_state" else span.line_text_start
    prefix = _prefix_index(offsets, position=prefix_position)
    if prefix is not None:
        return hidden[prefix]
    return hidden[indices[0]]


def features_for_hidden_states(
    *,
    example: ContentOnlyExample,
    hidden: np.ndarray,
    offsets: list[tuple[int, int]],
    feature_strategies: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_strategy: dict[str, list[dict[str, Any]]] = {strategy: [] for strategy in feature_strategies}
    for span in example.line_spans:
        metadata = metadata_from_span(span)
        for strategy in feature_strategies:
            vector = select_line_feature(hidden, offsets, span, strategy)
            rows_by_strategy[strategy].append(
                {
                    **metadata,
                    "feature_strategy": strategy,
                    "feature_dim": int(vector.shape[0]),
                    "feature": [float(value) for value in vector.tolist()],
                }
            )
    return rows_by_strategy


def synthetic_hidden_for_example(example: ContentOnlyExample, *, hidden_dim: int = 8) -> tuple[np.ndarray, list[tuple[int, int]]]:
    tokens: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+|\s", example.full_text):
        tokens.append((match.start(), match.end()))
    hidden = np.zeros((len(tokens), hidden_dim), dtype=np.float32)
    for index, (start, end) in enumerate(tokens):
        hidden[index, 0] = float(index % 11)
        hidden[index, 1] = float(end - start)
        hidden[index, 2] = float(start % 17)
        hidden[index, 3] = float(end % 19)
    return hidden, tokens


def chunk_path_for_unit(run_dir: Path, unit_id: str) -> Path:
    return run_dir / "chunks" / safe_unit_filename(unit_id)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def existing_chunk_paths(run_dir: Path) -> list[Path]:
    chunk_dir = run_dir / "chunks"
    if not chunk_dir.exists():
        return []
    return sorted(path for path in chunk_dir.glob("*.json") if not path.name.endswith(".tmp"))


def load_chunk(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_unit_ids_from_chunks(run_dir: Path) -> set[str]:
    completed: set[str] = set()
    for path in existing_chunk_paths(run_dir):
        chunk = load_chunk(path)
        unit_id = chunk.get("unit_id")
        if isinstance(unit_id, str) and unit_id:
            completed.add(unit_id)
    return completed


def write_state(
    run_dir: Path,
    *,
    config: dict[str, Any],
    stage: str,
    total_units: int,
    completed_probe_configs: list[str] | None = None,
    status: str = "running",
) -> dict[str, Any]:
    completed_units = sorted(completed_unit_ids_from_chunks(run_dir))
    completed_features = sorted(features_present_in_chunks(run_dir))
    state = {
        "run_id": config["run_id"],
        "status": status,
        "stage": stage,
        "updated_at": utc_now_iso(),
        "total_units": total_units,
        "processed_units": len(completed_units),
        "remaining_units": max(total_units - len(completed_units), 0),
        "completed_row_ids": completed_units,
        "completed_feature_strategies": completed_features,
        "completed_probe_configs": completed_probe_configs or completed_probe_ids(run_dir),
        "resume_signature": config.get("resume_signature"),
    }
    write_json(run_dir / "state.json", state)
    return state


def features_present_in_chunks(run_dir: Path) -> set[str]:
    present: set[str] = set()
    for path in existing_chunk_paths(run_dir):
        chunk = load_chunk(path)
        features = chunk.get("features_by_strategy", {})
        if isinstance(features, dict):
            present.update(str(key) for key in features)
    return present


def rebuild_aggregate_artifacts(run_dir: Path, feature_strategies: Iterable[str]) -> None:
    chunks = [load_chunk(path) for path in existing_chunk_paths(run_dir)]
    metadata_rows: list[dict[str, Any]] = []
    feature_rows_by_strategy: dict[str, list[dict[str, Any]]] = {strategy: [] for strategy in feature_strategies}
    seen_lines: set[str] = set()
    for chunk in chunks:
        for metadata in chunk.get("line_metadata", []):
            line_id = metadata["line_id"]
            if line_id not in seen_lines:
                metadata_rows.append(metadata)
                seen_lines.add(line_id)
        features_by_strategy = chunk.get("features_by_strategy", {})
        for strategy in feature_rows_by_strategy:
            feature_rows_by_strategy[strategy].extend(features_by_strategy.get(strategy, []))
    write_jsonl_atomic(run_dir / "line_metadata.jsonl", metadata_rows)
    for strategy, rows in feature_rows_by_strategy.items():
        write_jsonl_atomic(run_dir / f"features_{strategy}.jsonl", rows)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    write_jsonl(tmp_path, rows)
    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    raise last_error or PermissionError(f"could_not_replace:{path}")


def init_or_validate_config(run_dir: Path, config: dict[str, Any], *, force_new_run: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("resume_signature") != config.get("resume_signature"):
            if not force_new_run:
                raise RunCompatibilityError(
                    "latent_probe_resume_signature_mismatch:"
                    f"{run_dir}:existing={existing.get('resume_signature')}:"
                    f"new={config.get('resume_signature')}"
                )
            if existing_chunk_paths(run_dir):
                raise RunCompatibilityError("force_new_run_refuses_to_overwrite_existing_chunks")
        else:
            return
    write_json(config_path, config)


def load_feature_rows(run_dir: Path, strategy: str) -> list[dict[str, Any]]:
    path = run_dir / f"features_{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    return read_jsonl(path)


def split_feature_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == split]


def rows_to_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("no_feature_rows_for_split")
    x = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["level"]) for row in rows], dtype=np.int64)
    if not np.isfinite(x).all():
        raise ValueError("feature_matrix_contains_non_finite_values")
    return x, y


def previous_level_predictions(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> np.ndarray:
    majority = majority_level(train_rows)
    predictions: list[int] = []
    last_by_unit: dict[str, int] = {}
    for row in sorted(eval_rows, key=lambda item: (str(item["unit_id"]), int(item["line_position"]))):
        unit_id = str(row["unit_id"])
        predictions.append(last_by_unit.get(unit_id, majority))
        last_by_unit[unit_id] = int(row["level"])
    by_line = {str(row["line_id"]): pred for row, pred in zip(sorted(eval_rows, key=lambda item: (str(item["unit_id"]), int(item["line_position"]))), predictions)}
    return np.asarray([by_line[str(row["line_id"])] for row in eval_rows], dtype=np.int64)


def majority_level(rows: list[dict[str, Any]]) -> int:
    counts: dict[int, int] = {}
    for row in rows:
        level = int(row["level"])
        counts[level] = counts.get(level, 0) + 1
    if not counts:
        return 0
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int] | None = None) -> dict[str, Any]:
    used_labels = labels or sorted({int(value) for value in y_true.tolist()} | {int(value) for value in y_pred.tolist()})
    report = classification_report(
        y_true,
        y_pred,
        labels=used_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "level_mae": float(mean_absolute_error(y_true, y_pred)),
        "labels": used_labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=used_labels).tolist(),
    }


def probe_id(*, strategy: str, probe_type: str, mlp_hidden_dim: int, mlp_layers: int, mlp_dropout: float) -> str:
    if probe_type != "mlp":
        return f"{strategy}__{probe_type}"
    dropout = str(mlp_dropout).replace(".", "p")
    return f"{strategy}__mlp_h{mlp_hidden_dim}_l{mlp_layers}_d{dropout}"


def completed_probe_ids(run_dir: Path) -> list[str]:
    return sorted(path.stem.removeprefix("probe_metrics_") for path in run_dir.glob("probe_metrics_*.json"))


def train_and_evaluate_probe(
    *,
    run_dir: Path,
    strategy: str,
    probe_type: str,
    train_split: str = "train",
    eval_splits: Iterable[str] = ("validation",),
    random_state: int = 42,
    mlp_hidden_dim: int = 64,
    mlp_layers: int = 1,
    mlp_dropout: float = 0.0,
    max_iter: int = 300,
) -> dict[str, Any]:
    feature_rows = load_feature_rows(run_dir, strategy)
    train_rows = split_feature_rows(feature_rows, train_split)
    train_x, train_y = rows_to_xy(train_rows)
    labels = sorted({int(value) for value in train_y.tolist()})
    pid = probe_id(
        strategy=strategy,
        probe_type=probe_type,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout,
    )

    model: Any | None = None
    if probe_type == "majority" or len(labels) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(train_x, train_y)
    elif probe_type == "linear":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=random_state, class_weight="balanced"),
        )
        model.fit(train_x, train_y)
    elif probe_type == "mlp":
        hidden_layers = tuple([mlp_hidden_dim] * mlp_layers)
        mlp_kwargs: dict[str, Any] = {}
        if _mlp_supports_dropout():
            mlp_kwargs["dropout"] = mlp_dropout
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                alpha=0.0001,
                batch_size="auto",
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=False,
                **mlp_kwargs,
            ),
        )
        model.fit(train_x, train_y)
    elif probe_type != "previous_level":
        raise ValueError(f"unsupported_probe_type:{probe_type}")

    metrics: dict[str, Any] = {
        "probe_id": pid,
        "feature_strategy": strategy,
        "probe_type": probe_type,
        "train_split": train_split,
        "train_line_count": len(train_rows),
        "train_level_labels": labels,
        "created_at": utc_now_iso(),
    }
    for split in eval_splits:
        eval_rows = split_feature_rows(feature_rows, split)
        if not eval_rows:
            continue
        eval_x, eval_y = rows_to_xy(eval_rows)
        if probe_type == "previous_level":
            pred = previous_level_predictions(train_rows, eval_rows)
        else:
            pred = np.asarray(model.predict(eval_x), dtype=np.int64)
        split_metrics = evaluate_predictions(eval_y, pred, labels=sorted(set(labels) | set(eval_y.tolist()) | set(pred.tolist())))
        metrics[split] = {
            **split_metrics,
            "line_count": len(eval_rows),
        }
        write_jsonl_atomic(
            run_dir / f"probe_predictions_{split}_{pid}.jsonl",
            [
                {
                    "line_id": row["line_id"],
                    "unit_id": row["unit_id"],
                    "split": split,
                    "level": int(row["level"]),
                    "predicted_level": int(value),
                    "feature_strategy": strategy,
                    "probe_type": probe_type,
                    "probe_id": pid,
                }
                for row, value in zip(eval_rows, pred.tolist())
            ],
        )
    write_json(run_dir / f"probe_metrics_{pid}.json", metrics)
    return metrics


def _mlp_supports_dropout() -> bool:
    try:
        import inspect

        return "dropout" in inspect.signature(MLPClassifier).parameters
    except Exception:
        return False


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
