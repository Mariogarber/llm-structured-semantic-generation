from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.util import find_spec
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json
from llm_structured_semantic_generation.evaluation import (
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)
from llm_structured_semantic_generation.latent import mean_pool_generate_hidden_states
from llm_structured_semantic_generation.resumable_run import ResumableRun
from llm_structured_semantic_generation.sft_serialization import (
    BLOCKS_TSV_V1,
    SYSTEM_PROMPT,
    deserialize_training_blocks,
)
from llm_structured_semantic_generation.structure import blocks_to_yaml


JSON_ARRAY_FORMAT = "json_array"
BLOCKS_TSV_COMPACT_V1 = "blocks_tsv_compact_v1"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero-shot Kubernetes v1 block-generation baseline."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "dataset_structural_targets.jsonl",
        help="Structural target JSONL.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit",
        help="Local Hugging Face model path.",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--output-format",
        choices=[BLOCKS_TSV_COMPACT_V1, BLOCKS_TSV_V1, JSON_ARRAY_FORMAT],
        default=BLOCKS_TSV_COMPACT_V1,
        help="Structured text format requested from the model. blocks_tsv_compact_v1 is the shortest and recommended for the baseline.",
    )
    parser.add_argument(
        "--recovery-mode",
        choices=["strict", "raw_line_text"],
        default="strict",
        help="How the parser reconstructs YAML from predicted blocks during evaluation.",
    )
    parser.add_argument(
        "--gpu-memory",
        default="4.8GiB",
        help="Maximum GPU memory available to accelerate device_map. Leave headroom on 6GB GPUs.",
    )
    parser.add_argument(
        "--cpu-memory",
        default="32GiB",
        help="Maximum CPU RAM available for model offload when the quantized model does not fit fully on GPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "baseline_kubernetes_v1",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Resume or create a run in output-dir/run-id. If omitted, a new timestamp-based run id is created.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Number of samples processed and persisted per batch.",
    )
    parser.add_argument(
        "--collect-latent-means",
        action="store_true",
        help=(
            "Store one mean-pooled final-layer latent vector per sample, computed over generated tokens only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write config only; do not load the model.",
    )
    return parser.parse_args()


def build_unit_id(row: dict[str, Any]) -> str:
    return f"{row['sample_id']}::{row['prompt_variant']}"


def target_contract(output_format: str) -> str:
    return f"prompt -> {output_format} -> parser -> YAML"


def build_prompt(prompt_text: str, output_format: str) -> tuple[str, str]:
    system_prompt = SYSTEM_PROMPT
    if output_format == BLOCKS_TSV_V1:
        user_prompt = (
            "Natural-language request:\n"
            f"{prompt_text.strip()}\n\n"
            "Return only the structural block sequence in blocks_tsv_v1 format:\n"
            "<blocks>\n"
            "0<TAB>0<TAB>0<TAB>apiVersion: v1\n"
            "0<TAB>1<TAB>0<TAB>kind: ConfigMap\n"
            "...\n"
            "</blocks>\n"
            "Rules:\n"
            "- one block per line inside <blocks> and </blocks>\n"
            "- each line must contain exactly 4 TAB-separated fields: document_index, line_index, level, line_text\n"
            "- use real tab characters as separators, not the literal text <TAB>\n"
            "- use only integer values for the first 3 fields\n"
            "- do not print field names such as document_index, line_index, level, or line_text\n"
            "- do not print headers, comments, explanations, bullets, or prose\n"
            "- line_text must contain the YAML content for that line without leading spaces\n"
            "- indentation is represented only by level\n"
            "- line_index must be consecutive within each document, starting at 0\n"
            "- do not wrap the answer in Markdown"
        )
    elif output_format == BLOCKS_TSV_COMPACT_V1:
        user_prompt = (
            "Natural-language request:\n"
            f"{prompt_text.strip()}\n\n"
            "Return only the structural block sequence in blocks_tsv_compact_v1 format:\n"
            "<blocks>\n"
            "0<TAB>0<TAB>apiVersion: v1\n"
            "0<TAB>0<TAB>kind: ConfigMap\n"
            "0<TAB>0<TAB>metadata:\n"
            "0<TAB>1<TAB>name: demo-config\n"
            "0<TAB>0<TAB>spec:\n"
            "0<TAB>1<TAB>selector:\n"
            "0<TAB>2<TAB>matchLabels:\n"
            "0<TAB>3<TAB>app: demo\n"
            "0<TAB>1<TAB>containers:\n"
            "0<TAB>2<TAB>- name: app\n"
            "0<TAB>3<TAB>image: nginx:latest\n"
            "...\n"
            "</blocks>\n"
            "Rules:\n"
            "- one block per line inside <blocks> and </blocks>\n"
            "- each line must contain exactly 3 TAB-separated fields: document_index, level, line_text\n"
            "- use real tab characters as separators, not the literal text <TAB>\n"
            "- use only integer values for document_index and level\n"
            "- do not print line_index, field names, headers, comments, explanations, bullets, or prose\n"
            "- line order is the true order; do not skip lines\n"
            "- line_text must contain the YAML content for that line without leading spaces\n"
            "- indentation is represented only by level\n"
            "- top-level YAML keys such as apiVersion, kind, metadata, and spec must stay at level 0\n"
            "- children of metadata and spec must be deeper than their parent; do not indent the parent keys themselves\n"
            "- use plain mapping lines such as metadata:, spec:, name: value, image: value unless the YAML line is truly a list item\n"
            "- do not prefix mapping children with '-' just because they are nested; nested mappings stay as normal key lines\n"
            "- use '-' only for real YAML list items such as entries under containers, volumes, volumeMounts, env, ports, rules, command, or args\n"
            "- after a list item like '- name: app', its child fields continue on later lines at a deeper level without another '-'\n"
            "- do not wrap the answer in Markdown"
        )
    else:
        user_prompt = (
            "Natural-language request:\n"
            f"{prompt_text.strip()}\n\n"
            "Return only a JSON array. Each item must have integer document_index, "
            "integer line_index, integer level, and string line_text.\n"
            "Rules:\n"
            "- level starts at 0 for top-level YAML lines.\n"
            "- line_text must contain the YAML content for that line without any leading spaces.\n"
            "- Indentation is represented only by level, never by spaces inside line_text.\n"
            "- line_index must be consecutive within each document, starting at 0.\n"
            "- Do not wrap the answer in Markdown."
        )
    return system_prompt, user_prompt


def render_chat_prompt(tokenizer: Any, prompt_text: str, output_format: str) -> str:
    system_prompt, user_prompt = build_prompt(prompt_text, output_format)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def inspect_model_path(model_path: Path) -> dict[str, Any]:
    files = {path.name for path in model_path.glob("*") if path.is_file()} if model_path.exists() else set()
    tokenizer_files = {
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    }
    checks: dict[str, Any] = {
        "model_path_exists": model_path.exists(),
        "has_config": "config.json" in files,
        "has_generation_config": "generation_config.json" in files,
        "has_weights": any(name.endswith((".safetensors", ".bin")) for name in files),
        "has_tokenizer_files": bool(files & tokenizer_files),
        "installed_transformers": find_spec("transformers") is not None,
        "installed_torch": find_spec("torch") is not None,
        "installed_bitsandbytes": find_spec("bitsandbytes") is not None,
        "quant_method": None,
        "warnings": [],
    }

    if checks["has_config"]:
        try:
            config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
            checks["quant_method"] = config.get("quantization_config", {}).get("quant_method")
        except json.JSONDecodeError:
            checks["warnings"].append("config_json_not_parseable")

    if not checks["has_tokenizer_files"]:
        checks["warnings"].append("missing_local_tokenizer_files")
    if checks["quant_method"] == "bitsandbytes" and not checks["installed_bitsandbytes"]:
        checks["warnings"].append("model_quantization_requires_bitsandbytes")

    required = [
        "model_path_exists",
        "has_config",
        "has_weights",
        "has_tokenizer_files",
        "installed_transformers",
        "installed_torch",
    ]
    checks["ready_for_full_run"] = all(bool(checks[item]) for item in required) and not (
        checks["quant_method"] == "bitsandbytes" and not checks["installed_bitsandbytes"]
    )
    return checks


def extract_json_array(text: str) -> list[dict[str, Any]]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        bracketed = re.search(r"\[.*\]", candidate, flags=re.DOTALL)
        if bracketed:
            candidate = bracketed.group(0)
    parsed = json.loads(candidate)
    if not isinstance(parsed, list):
        raise ValueError("model_output_is_not_a_json_array")
    return parsed


def normalize_structured_field_separators(text: str) -> str:
    normalized = re.sub(r"<tab>", "\t", text, flags=re.IGNORECASE)
    normalized = re.sub(r"<vt>", "\t", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("\v", "\t").replace("\f", "\t")
    return normalized


def extract_blocks_tsv(serialized: str) -> list[dict[str, Any]]:
    candidate = serialized.strip()
    fenced = re.search(r"```(?:text|tsv)?\s*(<blocks>.*?</blocks>)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        wrapped = re.search(r"<blocks>.*?</blocks>", candidate, flags=re.DOTALL)
        if wrapped:
            candidate = wrapped.group(0)
        else:
            opened = re.search(r"<blocks>.*", candidate, flags=re.DOTALL)
            if opened:
                candidate = opened.group(0)
    candidate = normalize_structured_field_separators(candidate)
    sanitized_lines: list[str] = []
    for raw_line in candidate.splitlines():
        normalized_line = normalize_structured_field_separators(raw_line)
        stripped = normalized_line.strip()
        if not stripped or stripped in {"<blocks>", "</blocks>"}:
            sanitized_lines.append(stripped)
            continue

        parts = normalized_line.split("\t")
        if len(parts) >= 8 and parts[0] == "document_index":
            normalized = "\t".join([parts[1], parts[3], parts[5], "\t".join(parts[7:])])
            sanitized_lines.append(normalized)
            continue
        if len(parts) >= 4 and parts[0] == "document_index":
            sanitized_lines.append("\t".join(parts[1:]))
            continue
        sanitized_lines.append(normalized_line)

    blocks: list[dict[str, Any]] = []
    for line in sanitized_lines:
        stripped = line.strip()
        if not stripped or stripped in {"<blocks>", "</blocks>"}:
            continue
        parts = line.split("\t", maxsplit=3)
        if len(parts) != 4:
            if blocks:
                break
            raise ValueError("not_enough_tsv_fields")
        document_index, line_index, level, line_text = parts
        try:
            blocks.append(
                {
                    "document_index": int(document_index),
                    "line_index": int(line_index),
                    "level": int(level),
                    "line_text": line_text,
                }
            )
        except ValueError as exc:
            if blocks:
                break
            raise ValueError(f"invalid_tsv_numeric_field:{exc}") from exc
    if not blocks:
        raise ValueError("no_valid_blocks_found")
    return [block.to_dict() for block in deserialize_training_blocks("\n".join(["<blocks>", *[
        f"{block['document_index']}\t{block['line_index']}\t{block['level']}\t{block['line_text']}" for block in blocks
    ], "</blocks>"]))]


def extract_blocks_tsv_compact(serialized: str) -> list[dict[str, Any]]:
    candidate = serialized.strip()
    fenced = re.search(r"```(?:text|tsv)?\s*(<blocks>.*?</blocks>)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        wrapped = re.search(r"<blocks>.*?</blocks>", candidate, flags=re.DOTALL)
        if wrapped:
            candidate = wrapped.group(0)
        else:
            opened = re.search(r"<blocks>.*", candidate, flags=re.DOTALL)
            if opened:
                candidate = opened.group(0)
    candidate = normalize_structured_field_separators(candidate)

    normalized_by_document: dict[int, int] = {}
    blocks: list[dict[str, Any]] = []
    for raw_line in candidate.splitlines():
        normalized_line = normalize_structured_field_separators(raw_line)
        stripped = normalized_line.strip()
        if not stripped or stripped in {"<blocks>", "</blocks>"}:
            continue

        parts = normalized_line.split("\t", maxsplit=2)
        if len(parts) != 3:
            if blocks:
                break
            raise ValueError("not_enough_compact_tsv_fields")

        document_index_text, level_text, line_text = parts
        try:
            document_index = int(document_index_text)
            level = int(level_text)
        except ValueError as exc:
            if blocks:
                break
            raise ValueError(f"invalid_compact_tsv_numeric_field:{exc}") from exc

        line_index = normalized_by_document.get(document_index, 0)
        blocks.append(
            {
                "document_index": document_index,
                "line_index": line_index,
                "level": level,
                "line_text": line_text,
            }
        )
        normalized_by_document[document_index] = line_index + 1

    if not blocks:
        raise ValueError("no_valid_blocks_found")
    return [block.to_dict() for block in deserialize_training_blocks("\n".join(["<blocks>", *[
        f"{block['document_index']}\t{block['line_index']}\t{block['level']}\t{block['line_text']}" for block in blocks
    ], "</blocks>"]))]


def parse_structured_output(raw_text: str, output_format: str) -> list[dict[str, Any]]:
    if output_format == BLOCKS_TSV_COMPACT_V1:
        return extract_blocks_tsv_compact(raw_text)
    if output_format == BLOCKS_TSV_V1:
        return extract_blocks_tsv(raw_text)
    if output_format == JSON_ARRAY_FORMAT:
        return extract_json_array(raw_text)
    raise ValueError(f"unsupported_output_format:{output_format}")


def load_model(model_path: Path, args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The baseline requires optional LLM dependencies. Install them with "
            "`uv sync --extra llm` before running without --dry-run."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    quantization_config = getattr(config, "quantization_config", None)
    is_bitsandbytes_4bit = isinstance(quantization_config, dict) and bool(quantization_config.get("load_in_4bit"))
    load_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if torch.cuda.is_available() and is_bitsandbytes_4bit:
        # On Windows with recent transformers/accelerate, auto-dispatch for this
        # local 4-bit checkpoint fails during hook installation. Loading the
        # quantized model directly on the single laptop GPU avoids that path and
        # keeps inference on CUDA.
        load_kwargs["device_map"] = {"": 0}
    else:
        max_memory = {"cpu": args.cpu_memory}
        if torch.cuda.is_available():
            max_memory[0] = args.gpu_memory
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return tokenizer, model


def input_device(model: Any) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for module_name in ("model.embed_tokens", "transformer.wte", ""):
            device = device_map.get(module_name)
            if device not in (None, "disk", "cpu"):
                return str(device)
    return str(getattr(model, "device", "cpu"))


def generate_completion(tokenizer: Any, model: Any, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(input_device(model))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_hidden_states": args.collect_latent_means,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    outputs = model.generate(**inputs, **generation_kwargs)
    full_sequence = outputs.sequences[0]
    prompt_token_count = int(inputs["input_ids"].shape[-1])
    generated_token_ids = full_sequence[prompt_token_count:]
    latent_mean = None
    latent_dim = None
    if args.collect_latent_means:
        pooled = mean_pool_generate_hidden_states(outputs.hidden_states)
        if pooled is not None:
            pooled_cpu = pooled.detach().to(dtype=torch.float32).cpu()
            latent_mean = pooled_cpu.tolist()
            latent_dim = int(pooled_cpu.shape[0])
    return {
        "raw_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "prompt_token_count": prompt_token_count,
        "generated_token_ids": generated_token_ids.detach().cpu().tolist(),
        "generated_token_count": len(generated_token_ids),
        "latent_dim": latent_dim,
        "latent_mean": latent_mean,
    }


def build_resume_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": str(args.dataset),
        "model_path": str(args.model_path),
        "split": args.split,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "output_format": args.output_format,
        "recovery_mode": args.recovery_mode,
        "collect_latent_means": args.collect_latent_means,
        "target_contract": target_contract(args.output_format),
    }


def latent_row_from_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": prediction["unit_id"],
        "sample_id": prediction["sample_id"],
        "prompt_variant": prediction["prompt_variant"],
        "split": prediction["split"],
        "generated_token_count": prediction["generated_token_count"],
        "latent_dim": prediction["latent_dim"],
        "latent_mean": prediction["latent_mean"],
    }


def derive_metrics(
    *,
    run_id: str,
    predictions: list[dict[str, Any]],
    output_format: str,
    collect_latent_means: bool,
) -> dict[str, Any]:
    evaluated_results = [
        StructuralEvaluation(**row["evaluation"])
        for row in predictions
        if row.get("evaluation") is not None
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
        latent_rows = [latent_row_from_prediction(row) for row in predictions]
        latent_dims = sorted({row["latent_dim"] for row in latent_rows if row["latent_dim"] is not None})
        metrics["latent_collection"] = {
            "enabled": True,
            "row_count": len(latent_rows),
            "rows_with_vector": sum(1 for row in latent_rows if row["latent_mean"] is not None),
            "rows_without_generated_tokens": sum(1 for row in latent_rows if row["latent_mean"] is None),
            "latent_dims": latent_dims,
            "artifact": "latent_mean_vectors.jsonl",
        }
    else:
        metrics["latent_collection"] = {"enabled": False}
    return metrics


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in read_jsonl(args.dataset)
        if row["split"] == args.split and row["structural_target_status"] == "ok"
    ]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    config = {
        "run_id": run_id,
        "dataset": str(args.dataset),
        "model_path": str(args.model_path),
        "split": args.split,
        "row_count": len(rows),
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "output_format": args.output_format,
        "recovery_mode": args.recovery_mode,
        "batch_size": args.batch_size,
        "collect_latent_means": args.collect_latent_means,
        "gpu_memory": args.gpu_memory,
        "cpu_memory": args.cpu_memory,
        "target_contract": target_contract(args.output_format),
        "dry_run": args.dry_run,
        "model_checks": inspect_model_path(args.model_path),
        "resume_signature": build_resume_signature(args),
    }

    if args.dry_run:
        write_json(output_dir / "config.json", config)
        write_json(
            output_dir / "metrics.json",
            {
                "dry_run": True,
                "row_count": len(rows),
                "ready_for_full_run": config["model_checks"]["ready_for_full_run"],
                "model_warnings": config["model_checks"]["warnings"],
            },
        )
        print(
            {
                "dry_run": True,
                "output_dir": str(output_dir),
                "row_count": len(rows),
                "ready_for_full_run": config["model_checks"]["ready_for_full_run"],
                "model_warnings": config["model_checks"]["warnings"],
            }
        )
        return

    if not config["model_checks"]["ready_for_full_run"]:
        raise RuntimeError(
            "Baseline full run is not ready. See config.json model_checks: "
            f"{config['model_checks']}"
        )

    run = ResumableRun.initialize(
        run_dir=output_dir,
        config=config,
        total_units=len(rows),
        unit_id_field="unit_id",
        primary_artifact_name="predictions",
        artifact_paths={
            "predictions": "predictions.jsonl",
            "latent_mean_vectors": "latent_mean_vectors.jsonl",
        },
    )

    if args.collect_latent_means:
        run.reconcile_secondary_artifact(
            "latent_mean_vectors",
            unit_id_field="unit_id",
            expected_rows=[latent_row_from_prediction(row) for row in run.primary_rows],
        )

    pending_rows = [row for row in rows if build_unit_id(row) not in run.completed_unit_id_set]

    if pending_rows:
        tokenizer, model = load_model(args.model_path, args)
        for batch_start in range(0, len(pending_rows), args.batch_size):
            batch_rows = pending_rows[batch_start : batch_start + args.batch_size]
            batch_predictions: list[dict[str, Any]] = []

            for row in batch_rows:
                prompt = render_chat_prompt(tokenizer, row["prompt_text"], args.output_format)
                completion = generate_completion(tokenizer, model, prompt, args)
                raw_output = completion["raw_text"]
                parse_errors: list[str] = []
                try:
                    predicted_blocks = parse_structured_output(raw_output, args.output_format)
                except (json.JSONDecodeError, ValueError) as exc:
                    predicted_blocks = []
                    parse_errors.append(
                        f"structured_output_parse_error:{args.output_format}:{exc.__class__.__name__}:{exc}"
                    )

                reconstruction = blocks_to_yaml(predicted_blocks, recovery_mode=args.recovery_mode)
                evaluation = (
                    evaluate_blocks_prediction(
                        row["target_yaml_normalized"],
                        predicted_blocks,
                        recovery_mode=args.recovery_mode,
                        prompt_text=row["prompt_text"],
                    )
                    if predicted_blocks
                    else None
                )

                batch_predictions.append(
                    {
                        "unit_id": build_unit_id(row),
                        "sample_id": row["sample_id"],
                        "prompt_variant": row["prompt_variant"],
                        "split": row["split"],
                        "prompt_text": row["prompt_text"],
                        "output_format": args.output_format,
                        "raw_model_output": raw_output,
                        "generated_token_count": completion["generated_token_count"],
                        "predicted_blocks": predicted_blocks,
                        "reconstructed_yaml": reconstruction.yaml_text,
                        "parser_errors": list(reconstruction.errors) + parse_errors,
                        "evaluation": evaluation.to_dict() if evaluation else None,
                        "latent_dim": completion["latent_dim"],
                        "latent_mean": completion["latent_mean"],
                    }
                )

            run.record_batch(
                batch_predictions,
                secondary_rows_by_name={
                    "latent_mean_vectors": [latent_row_from_prediction(row) for row in batch_predictions]
                }
                if args.collect_latent_means
                else None,
            )

    metrics = derive_metrics(
        run_id=run_id,
        predictions=run.primary_rows,
        output_format=args.output_format,
        collect_latent_means=args.collect_latent_means,
    )
    write_json(output_dir / "metrics.json", metrics)
    run.mark_completed()
    print({"output_dir": str(output_dir), **metrics})


if __name__ == "__main__":
    main()
