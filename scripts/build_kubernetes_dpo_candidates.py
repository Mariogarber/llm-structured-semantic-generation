from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json
from llm_structured_semantic_generation.evaluation import (
    StructuralEvaluation,
    evaluate_blocks_prediction,
    summarize_evaluations,
)
from llm_structured_semantic_generation.resumable_run import ResumableRun, utc_now_iso
from llm_structured_semantic_generation.sft_serialization import BLOCKS_TSV_V1
from llm_structured_semantic_generation.structure import blocks_to_yaml
from train_kubernetes_sft import (
    PROMPT_TARGET_SEPARATOR,
    SERIALIZED_SFT,
    build_unit_id,
    extract_blocks_tsv_prediction,
    latest_checkpoint,
    load_env_file,
    load_sft_rows,
    model_input_device,
    non_negative_int,
    positive_int,
)


DEFAULT_SFT_RUN_DIR = REPO_ROOT / "results" / "sft_kubernetes_v1" / "serialized-sft-a-v1-20260505-171226"
DEFAULT_TRAIN_FILE = REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "train.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dpo_kubernetes_v1" / "candidate_generation"

CANDIDATES_ARTIFACT = "candidates.jsonl"
CANDIDATE_METRICS_ARTIFACT = "candidate_metrics.jsonl"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_index: int
    temperature: float
    seed: int
    top_p: float


def parse_float_csv(value: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parsed = float(stripped)
        if parsed < 0:
            raise argparse.ArgumentTypeError("temperatures must be non-negative")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one temperature is required")
    return tuple(values)


def parse_int_csv(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_checkpoint(sft_run_dir: Path, checkpoint: str) -> Path:
    sft_run_dir = resolve_project_path(sft_run_dir)
    if checkpoint == "best":
        state_path = sft_run_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"missing_sft_state:{state_path}")
        best_checkpoint = read_json(state_path).get("best_checkpoint")
        if not isinstance(best_checkpoint, str) or not best_checkpoint:
            raise ValueError(f"missing_best_checkpoint:{state_path}")
        return resolve_project_path(Path(best_checkpoint))

    if checkpoint == "latest":
        path = latest_checkpoint(sft_run_dir)
        if path is None:
            raise FileNotFoundError(f"missing_latest_checkpoint:{sft_run_dir / 'checkpoints'}")
        return path

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    by_name = sft_run_dir / "checkpoints" / checkpoint
    if by_name.exists():
        return by_name
    return REPO_ROOT / checkpoint_path


def resolve_checkpoint_root_and_adapter(checkpoint_path: Path) -> tuple[Path, Path]:
    checkpoint_path = checkpoint_path.resolve()
    if (checkpoint_path / "adapter" / "adapter_config.json").exists():
        return checkpoint_path, checkpoint_path / "adapter"
    if (checkpoint_path / "adapter_config.json").exists():
        return checkpoint_path.parent, checkpoint_path
    raise FileNotFoundError(f"missing_adapter_config:{checkpoint_path}")


def build_candidate_specs(
    *,
    num_candidates: int,
    temperatures: tuple[float, ...],
    top_p: float,
    base_seed: int,
    candidate_seeds: tuple[int, ...] = (),
) -> list[CandidateSpec]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    seeds = candidate_seeds or tuple(base_seed + index for index in range(num_candidates))
    return [
        CandidateSpec(
            candidate_id=f"c{index:02d}",
            candidate_index=index,
            temperature=temperatures[index % len(temperatures)],
            seed=seeds[index % len(seeds)],
            top_p=top_p,
        )
        for index in range(num_candidates)
    ]


def build_candidate_uid(unit_id: str, candidate_id: str) -> str:
    return f"{unit_id}::{candidate_id}"


def build_candidate_tasks(rows: list[dict[str, Any]], specs: list[CandidateSpec]) -> list[tuple[dict[str, Any], CandidateSpec, str]]:
    tasks: list[tuple[dict[str, Any], CandidateSpec, str]] = []
    for row in rows:
        unit_id = build_unit_id(row)
        for spec in specs:
            tasks.append((row, spec, build_candidate_uid(unit_id, spec.candidate_id)))
    return tasks


def select_sft_rows(
    rows: list[dict[str, Any]],
    *,
    sample_offset: int,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if sample_offset >= len(rows):
        raise ValueError(f"sample_offset_out_of_range:{sample_offset}:row_count:{len(rows)}")
    selected = rows[sample_offset:]
    if max_samples is not None:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError("no_sft_rows_selected")
    return selected


def build_resume_signature(args: argparse.Namespace, checkpoint_root: Path, adapter_path: Path) -> dict[str, Any]:
    return {
        "stage": "dpo_candidate_generation_v1",
        "source_model_variant": args.source_model_variant,
        "serialization": BLOCKS_TSV_V1,
        "train_file": project_path(resolve_project_path(args.train_file)),
        "sft_run_dir": project_path(resolve_project_path(args.sft_run_dir)),
        "checkpoint_root": project_path(checkpoint_root),
        "adapter_path": project_path(adapter_path),
        "base_model_path": project_path(resolve_project_path(args.base_model_path)),
        "sample_offset": args.sample_offset,
        "max_train_samples": args.max_train_samples,
        "num_candidates": args.num_candidates,
        "temperatures": list(args.temperatures),
        "candidate_seeds": list(args.candidate_seeds),
        "seed": args.seed,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "target_contract": "prompt -> sampled blocks_tsv_v1 candidates -> parser/evaluation -> DPO preferences",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate sampled candidate completions from the serialized_sft Kubernetes model "
            "as the first artifact for an offline DPO preference dataset."
        )
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--base-model-path", type=Path, default=REPO_ROOT / "model" / "qwen2.5-7b-instruct-4bit")
    parser.add_argument("--sft-run-dir", type=Path, default=DEFAULT_SFT_RUN_DIR)
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Use best, latest, a checkpoint directory name, or a checkpoint/adapter path.",
    )
    parser.add_argument(
        "--source-model-variant",
        default=SERIALIZED_SFT,
        help=(
            "Model family used to generate the candidates. Keep the default for "
            "serialized SFT checkpoints; use serialized_sft_dpo for DPO checkpoints."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help=(
            "Completed candidates to append per checkpoint. The default, 1, "
            "persists progress after every generated candidate."
        ),
    )
    parser.add_argument("--num-candidates", type=positive_int, default=4)
    parser.add_argument(
        "--temperatures",
        type=parse_float_csv,
        default=parse_float_csv("0.2"),
        help="Comma-separated temperatures. 0 uses greedy decoding; values >0 use sampling.",
    )
    parser.add_argument(
        "--candidate-seeds",
        type=parse_int_csv,
        default=(),
        help="Optional comma-separated seed list. Defaults to seed, seed+1, ...",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument(
        "--sample-offset",
        type=non_negative_int,
        default=0,
        help="Number of validated SFT rows to skip before selecting --max-train-samples.",
    )
    parser.add_argument("--max-train-samples", type=positive_int, default=None)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write config/state without loading the model or generating candidates.",
    )
    return parser.parse_args()


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def load_sft_model_and_tokenizer(
    *,
    base_model_path: Path,
    checkpoint_root: Path,
    adapter_path: Path,
    gpu_memory: str,
    cpu_memory: str,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = checkpoint_root / "tokenizer"
    tokenizer_source = tokenizer_path if tokenizer_path.exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(base_model_path, local_files_only=True)
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
        max_memory: dict[Any, str] = {"cpu": cpu_memory}
        if torch.cuda.is_available():
            max_memory[0] = gpu_memory
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def generate_completion(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    spec: CandidateSpec,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    set_generation_seed(spec.seed)
    prompt_text = f"{prompt.rstrip()}{PROMPT_TARGET_SEPARATOR}"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_input_device(model))
    input_token_count = int(inputs["input_ids"].shape[-1])

    class StopOnGeneratedSubsequence(StoppingCriteria):
        def __init__(self, stop_sequences: list[list[int]], prompt_length: int) -> None:
            self.stop_sequences = [sequence for sequence in stop_sequences if sequence]
            self.prompt_length = prompt_length

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0][self.prompt_length:].tolist()
            return any(
                len(generated) >= len(sequence) and generated[-len(sequence) :] == sequence
                for sequence in self.stop_sequences
            )

    stop_sequences = [
        tokenizer.encode("</blocks>", add_special_tokens=False),
        tokenizer.encode("\n</blocks>", add_special_tokens=False),
    ]
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": spec.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList(
            [StopOnGeneratedSubsequence(stop_sequences, prompt_length=input_token_count)]
        ),
    }
    if spec.temperature > 0:
        generation_kwargs["temperature"] = spec.temperature
        generation_kwargs["top_p"] = spec.top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_token_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return {
        "raw_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "input_token_count": input_token_count,
        "generated_token_count": int(generated_token_ids.shape[-1]),
    }


def maybe_clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def metric_float(evaluation: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = evaluation.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return default


def metric_bool(evaluation: dict[str, Any], key: str) -> bool:
    return bool(evaluation.get(key))


def compute_preference_score(evaluation: dict[str, Any] | None) -> dict[str, Any]:
    if evaluation is None or not metric_bool(evaluation, "yaml_parse_ok") or not metric_bool(evaluation, "block_parse_ok"):
        return {
            "preference_score": 0.0,
            "hard_invalid": True,
            "components": {},
            "penalties": {},
            "formula": "hard_invalid -> 0",
        }

    prompt_requirement_f1 = metric_float(evaluation, "prompt_requirement_f1")
    kubernetes_domain_validity_score = metric_float(evaluation, "kubernetes_domain_validity_score")
    required_field_complete_resource_rate = metric_float(evaluation, "required_field_complete_resource_rate")
    level_exact_match_rate = metric_float(evaluation, "level_exact_match_rate")
    kubernetes_domain_gate_pass = 1.0 if metric_bool(evaluation, "kubernetes_domain_gate_pass") else 0.0

    reference_documents = int(metric_float(evaluation, "reference_document_count"))
    prediction_documents = int(metric_float(evaluation, "prediction_document_count"))
    kind_sequence_match_rate = metric_float(evaluation, "kind_sequence_match_rate")
    invented_resource_penalty = (
        1.0
        if prediction_documents > reference_documents and kind_sequence_match_rate < 1.0
        else 0.0
    )

    reference_lines = int(metric_float(evaluation, "line_count_reference"))
    prediction_lines = int(metric_float(evaluation, "line_count_prediction"))
    severe_line_count_penalty = 0.0
    if reference_lines > 0 and (prediction_lines < 0.5 * reference_lines or prediction_lines > 2.0 * reference_lines):
        severe_line_count_penalty = 1.0

    weighted_components = {
        "prompt_requirement_f1": 1.00 * prompt_requirement_f1,
        "kubernetes_domain_validity_score": 0.75 * kubernetes_domain_validity_score,
        "required_field_complete_resource_rate": 0.50 * required_field_complete_resource_rate,
        "level_exact_match_rate": 0.25 * level_exact_match_rate,
        "kubernetes_domain_gate_pass": 0.25 * kubernetes_domain_gate_pass,
    }
    penalties = {
        "invented_resource_penalty": invented_resource_penalty,
        "severe_line_count_penalty": severe_line_count_penalty,
    }
    score = max(sum(weighted_components.values()) - sum(penalties.values()), 0.0)
    return {
        "preference_score": round(score, 6),
        "hard_invalid": False,
        "components": {
            "prompt_requirement_f1": prompt_requirement_f1,
            "kubernetes_domain_validity_score": kubernetes_domain_validity_score,
            "required_field_complete_resource_rate": required_field_complete_resource_rate,
            "level_exact_match_rate": level_exact_match_rate,
            "kubernetes_domain_gate_pass": kubernetes_domain_gate_pass,
        },
        "weighted_components": weighted_components,
        "penalties": penalties,
        "formula": (
            "1.00*prompt_requirement_f1 + 0.75*kubernetes_domain_validity_score + "
            "0.50*required_field_complete_resource_rate + 0.25*level_exact_match_rate + "
            "0.25*kubernetes_domain_gate_pass - penalties"
        ),
    }


def evaluate_candidate(row: dict[str, Any], raw_text: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if raw_text is None:
        return (
            {
                "structured_output_parse_success": False,
                "predicted_blocks": [],
                "reconstructed_yaml": "",
                "parser_errors": ["generation_failed"],
                "reconstruction_errors": [],
            },
            None,
        )

    try:
        predicted_blocks = extract_blocks_tsv_prediction(raw_text)
    except ValueError as exc:
        return (
            {
                "structured_output_parse_success": False,
                "predicted_blocks": [],
                "reconstructed_yaml": "",
                "parser_errors": [str(exc)],
                "reconstruction_errors": [],
            },
            None,
        )

    reconstruction = blocks_to_yaml(predicted_blocks, recovery_mode="strict")
    evaluation = evaluate_blocks_prediction(
        str(row["target_yaml_normalized"]),
        predicted_blocks,
        prompt_text=str(row["prompt"]),
    )
    return (
        {
            "structured_output_parse_success": True,
            "predicted_blocks": predicted_blocks,
            "reconstructed_yaml": reconstruction.yaml_text,
            "parser_errors": [],
            "reconstruction_errors": list(reconstruction.errors),
        },
        evaluation.to_dict(),
    )


def build_candidate_rows(
    *,
    row: dict[str, Any],
    spec: CandidateSpec,
    candidate_uid: str,
    checkpoint_root: Path,
    generation: dict[str, Any],
    generation_error: BaseException | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    unit_id = build_unit_id(row)
    raw_text = generation.get("raw_text") if generation_error is None else None
    parsed_payload, evaluation = evaluate_candidate(row, raw_text)
    score_payload = compute_preference_score(evaluation)
    generated_at = utc_now_iso()

    candidate_row = {
        "candidate_uid": candidate_uid,
        "unit_id": unit_id,
        "candidate_id": spec.candidate_id,
        "candidate_index": spec.candidate_index,
        "sample_id": row["sample_id"],
        "prompt_variant": row["prompt_variant"],
        "split": row.get("split", "train"),
        "prompt": row["prompt"],
        "reference_yaml": row["target_yaml_normalized"],
        "checkpoint": project_path(checkpoint_root),
        "generation_ok": generation_error is None,
        "generation_error_type": type(generation_error).__name__ if generation_error is not None else None,
        "generation_error": str(generation_error) if generation_error is not None else None,
        "generation_config": asdict(spec),
        "input_token_count": generation.get("input_token_count"),
        "generated_token_count": generation.get("generated_token_count"),
        "model_output_text": raw_text,
        **parsed_payload,
        "generated_at": generated_at,
    }
    metric_row = {
        "candidate_uid": candidate_uid,
        "unit_id": unit_id,
        "candidate_id": spec.candidate_id,
        "sample_id": row["sample_id"],
        "prompt_variant": row["prompt_variant"],
        "split": row.get("split", "train"),
        "checkpoint": project_path(checkpoint_root),
        "generation_ok": generation_error is None,
        "structured_output_parse_success": parsed_payload["structured_output_parse_success"],
        "generation_config": asdict(spec),
        "evaluation": evaluation,
        **score_payload,
        "evaluated_at": generated_at,
    }
    return candidate_row, metric_row


def build_candidate_metric_row(candidate_row: dict[str, Any], evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    if evaluation is None and candidate_row.get("structured_output_parse_success"):
        evaluation = evaluate_blocks_prediction(
            str(candidate_row["reference_yaml"]),
            list(candidate_row.get("predicted_blocks", [])),
            prompt_text=str(candidate_row["prompt"]),
        ).to_dict()

    score_payload = compute_preference_score(evaluation)
    return {
        "candidate_uid": candidate_row["candidate_uid"],
        "unit_id": candidate_row["unit_id"],
        "candidate_id": candidate_row["candidate_id"],
        "sample_id": candidate_row["sample_id"],
        "prompt_variant": candidate_row["prompt_variant"],
        "split": candidate_row.get("split", "train"),
        "checkpoint": candidate_row["checkpoint"],
        "generation_ok": candidate_row["generation_ok"],
        "structured_output_parse_success": candidate_row["structured_output_parse_success"],
        "generation_config": candidate_row["generation_config"],
        "evaluation": evaluation,
        **score_payload,
        "evaluated_at": utc_now_iso(),
    }


def reconcile_candidate_metrics(run: ResumableRun) -> None:
    expected_rows = [build_candidate_metric_row(candidate_row) for candidate_row in run.primary_rows]
    run.reconcile_secondary_artifact(
        "candidate_metrics",
        unit_id_field="candidate_uid",
        expected_rows=expected_rows,
    )


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def derive_run_metrics(
    *,
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    candidate_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = [
        StructuralEvaluation(**row["evaluation"])
        for row in candidate_metrics
        if isinstance(row.get("evaluation"), dict)
    ]
    scores = [
        float(row["preference_score"])
        for row in candidate_metrics
        if isinstance(row.get("preference_score"), (int, float))
    ]
    candidates_by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_metrics:
        unit_id = str(row["unit_id"])
        candidates_by_unit.setdefault(unit_id, []).append(row)

    best_scores = [
        max(float(row["preference_score"]) for row in rows if isinstance(row.get("preference_score"), (int, float)))
        for rows in candidates_by_unit.values()
        if any(isinstance(row.get("preference_score"), (int, float)) for row in rows)
    ]
    duplicate_groups = 0
    for rows in candidates_by_unit.values():
        outputs = [
            row.get("model_output_text")
            for row in candidates
            if row.get("unit_id") == rows[0].get("unit_id") and row.get("model_output_text") is not None
        ]
        if len(outputs) != len(set(outputs)):
            duplicate_groups += 1

    metrics = {
        "run_id": config["run_id"],
        "stage": "dpo_candidate_generation_v1",
        "created_at": utc_now_iso(),
        "source_model_variant": config.get("source_model_variant", SERIALIZED_SFT),
        "serialization": BLOCKS_TSV_V1,
        "row_count": len(candidates),
        "prompt_count": len({row["unit_id"] for row in candidates}),
        "generation_success_rate": average([1.0 if row.get("generation_ok") else 0.0 for row in candidates]),
        "structured_output_parse_success_rate": average(
            [1.0 if row.get("structured_output_parse_success") else 0.0 for row in candidates]
        ),
        "average_generated_token_count": average(
            [float(row["generated_token_count"]) for row in candidates if isinstance(row.get("generated_token_count"), int)]
        ),
        "average_preference_score": average(scores),
        "average_best_preference_score_per_prompt": average(best_scores),
        "duplicate_output_prompt_rate": duplicate_groups / len(candidates_by_unit) if candidates_by_unit else 0.0,
    }
    metrics.update(summarize_evaluations(evaluations))
    return metrics


def finalize_run(run: ResumableRun, config: dict[str, Any]) -> None:
    reconcile_candidate_metrics(run)
    metrics_rows = read_jsonl(run.artifact_paths["candidate_metrics"], allow_truncated_last_line=True)
    metrics = derive_run_metrics(
        config=config,
        candidates=run.primary_rows,
        candidate_metrics=metrics_rows,
    )
    write_json(run.run_dir / "metrics.json", metrics)
    run.mark_completed()


def main() -> None:
    args = parse_args()
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    load_env_file()
    train_file = resolve_project_path(args.train_file)
    base_model_path = resolve_project_path(args.base_model_path)
    checkpoint_root, adapter_path = resolve_checkpoint_root_and_adapter(
        resolve_checkpoint(args.sft_run_dir, args.checkpoint)
    )
    all_rows = load_sft_rows(train_file, max_samples=None, split_name="train")
    rows = select_sft_rows(
        all_rows,
        sample_offset=args.sample_offset,
        max_samples=args.max_train_samples,
    )
    specs = build_candidate_specs(
        num_candidates=args.num_candidates,
        temperatures=args.temperatures,
        top_p=args.top_p,
        base_seed=args.seed,
        candidate_seeds=args.candidate_seeds,
    )
    tasks = build_candidate_tasks(rows, specs)
    run_dir = resolve_project_path(args.output_dir) / args.run_id
    config = {
        "run_id": args.run_id,
        "stage": "dpo_candidate_generation_v1",
        "created_at": utc_now_iso(),
        "source_model_variant": args.source_model_variant,
        "train_file": project_path(train_file),
        "sft_run_dir": project_path(resolve_project_path(args.sft_run_dir)),
        "checkpoint_root": project_path(checkpoint_root),
        "adapter_path": project_path(adapter_path),
        "base_model_path": project_path(base_model_path),
        "sample_offset": args.sample_offset,
        "max_train_samples": args.max_train_samples,
        "batch_size": args.batch_size,
        "num_candidates": args.num_candidates,
        "temperatures": list(args.temperatures),
        "candidate_seeds": list(args.candidate_seeds),
        "seed": args.seed,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "artifacts": {
            "candidates": CANDIDATES_ARTIFACT,
            "candidate_metrics": CANDIDATE_METRICS_ARTIFACT,
            "metrics": "metrics.json",
        },
        "resume_signature": build_resume_signature(args, checkpoint_root, adapter_path),
    }

    run = ResumableRun.initialize(
        run_dir=run_dir,
        config=config,
        total_units=len(tasks),
        unit_id_field="candidate_uid",
        primary_artifact_name="candidates",
        artifact_paths={
            "candidates": CANDIDATES_ARTIFACT,
            "candidate_metrics": CANDIDATE_METRICS_ARTIFACT,
        },
    )
    reconcile_candidate_metrics(run)

    if args.dry_run:
        print(f"dry_run_ok run_dir={run_dir} total_candidates={len(tasks)}")
        return

    if run.is_complete:
        finalize_run(run, config)
        print(f"run_already_complete run_dir={run_dir}")
        return

    tokenizer, model = load_sft_model_and_tokenizer(
        base_model_path=base_model_path,
        checkpoint_root=checkpoint_root,
        adapter_path=adapter_path,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
    )

    pending = [
        (row, spec, candidate_uid)
        for row, spec, candidate_uid in tasks
        if candidate_uid not in run.completed_unit_id_set
    ]
    candidate_batch: list[dict[str, Any]] = []
    metric_batch: list[dict[str, Any]] = []
    try:
        for completed_index, (row, spec, candidate_uid) in enumerate(pending, start=1):
            generation: dict[str, Any] = {}
            generation_error: BaseException | None = None
            try:
                generation = generate_completion(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=str(row["prompt"]),
                    spec=spec,
                    max_new_tokens=args.max_new_tokens,
                )
            except RuntimeError as exc:
                generation_error = exc
                maybe_clear_cuda_cache()

            candidate_row, metric_row = build_candidate_rows(
                row=row,
                spec=spec,
                candidate_uid=candidate_uid,
                checkpoint_root=checkpoint_root,
                generation=generation,
                generation_error=generation_error,
            )
            candidate_batch.append(candidate_row)
            metric_batch.append(metric_row)

            if len(candidate_batch) >= args.batch_size:
                run.record_batch(candidate_batch, secondary_rows_by_name={"candidate_metrics": metric_batch})
                print(
                    "progress "
                    f"completed={len(run.completed_unit_ids)}/{run.total_units} "
                    f"last_candidate={candidate_uid} "
                    f"batch_index={completed_index}"
                )
                candidate_batch = []
                metric_batch = []
    except KeyboardInterrupt:
        if candidate_batch:
            run.record_batch(candidate_batch, secondary_rows_by_name={"candidate_metrics": metric_batch})
            print(
                "interrupted_progress_saved "
                f"completed={len(run.completed_unit_ids)}/{run.total_units}"
            )
        raise

    if candidate_batch:
        run.record_batch(candidate_batch, secondary_rows_by_name={"candidate_metrics": metric_batch})

    finalize_run(run, config)
    print(f"completed run_dir={run_dir} candidates={run.total_units}")


if __name__ == "__main__":
    main()
