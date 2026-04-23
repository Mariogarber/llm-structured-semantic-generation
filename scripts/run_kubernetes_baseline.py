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

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json, write_jsonl
from llm_structured_semantic_generation.evaluation import evaluate_blocks_prediction
from llm_structured_semantic_generation.sft_serialization import SYSTEM_PROMPT
from llm_structured_semantic_generation.structure import blocks_to_yaml


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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "baseline_kubernetes_v1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write config only; do not load the model.",
    )
    return parser.parse_args()


def build_prompt(prompt_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Natural-language request:\n"
        f"{prompt_text.strip()}\n\n"
        "Return only a JSON array. Each item must have integer document_index, "
        "integer line_index, integer level, and string line_text. Do not wrap the "
        "answer in Markdown."
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


def load_model(model_path: Path):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The baseline requires optional LLM dependencies. Install them with "
            "`uv sync --extra llm` before running without --dry-run."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return tokenizer, model


def generate_text(tokenizer: Any, model: Any, prompt: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in read_jsonl(args.dataset)
        if row["split"] == args.split and row["structural_target_status"] == "ok"
    ]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        "target_contract": "prompt -> JSON blocks(document_index,line_index,level,line_text) -> parser -> YAML",
        "dry_run": args.dry_run,
        "model_checks": inspect_model_path(args.model_path),
    }
    write_json(output_dir / "config.json", config)

    if args.dry_run:
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

    tokenizer, model = load_model(args.model_path)
    predictions: list[dict[str, Any]] = []

    for row in rows:
        prompt = build_prompt(row["prompt_text"])
        raw_output = generate_text(tokenizer, model, prompt, args)
        parse_errors: list[str] = []
        try:
            predicted_blocks = extract_json_array(raw_output)
        except (json.JSONDecodeError, ValueError) as exc:
            predicted_blocks = []
            parse_errors.append(f"json_block_parse_error:{exc.__class__.__name__}:{exc}")

        reconstruction = blocks_to_yaml(predicted_blocks)
        evaluation = (
            evaluate_blocks_prediction(row["target_yaml_normalized"], predicted_blocks)
            if predicted_blocks
            else None
        )

        predictions.append(
            {
                "sample_id": row["sample_id"],
                "prompt_variant": row["prompt_variant"],
                "split": row["split"],
                "prompt_text": row["prompt_text"],
                "raw_model_output": raw_output,
                "predicted_blocks": predicted_blocks,
                "reconstructed_yaml": reconstruction.yaml_text,
                "parser_errors": list(reconstruction.errors) + parse_errors,
                "evaluation": evaluation.to_dict() if evaluation else None,
            }
        )

    evaluated = [row["evaluation"] for row in predictions if row["evaluation"] is not None]
    metrics = {
        "run_id": run_id,
        "row_count": len(predictions),
        "evaluated_count": len(evaluated),
        "json_block_parse_success_rate": len(evaluated) / len(predictions) if predictions else 0.0,
        "yaml_parse_success_rate": (
            sum(1 for item in evaluated if item["yaml_parse_ok"]) / len(evaluated) if evaluated else 0.0
        ),
        "parsed_equal_rate": (
            sum(1 for item in evaluated if item["parsed_equal_to_reference"]) / len(evaluated)
            if evaluated
            else 0.0
        ),
        "average_content_exact_match_rate": (
            sum(item["content_exact_match_rate"] for item in evaluated) / len(evaluated)
            if evaluated
            else 0.0
        ),
        "average_level_exact_match_rate": (
            sum(item["level_exact_match_rate"] for item in evaluated) / len(evaluated)
            if evaluated
            else 0.0
        ),
    }

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "metrics.json", metrics)
    print({"output_dir": str(output_dir), **metrics})


if __name__ == "__main__":
    main()
