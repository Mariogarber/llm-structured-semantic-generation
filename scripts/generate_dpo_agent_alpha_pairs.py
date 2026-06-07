from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_structured_semantic_generation.dataset_io import write_json, write_jsonl
from llm_structured_semantic_generation.dpo_preference_annotation import (
    AGENT_SUGGESTIONS_ARTIFACT,
    AnnotationPaths,
    append_agent_suggestion,
    discover_candidate_run_dirs,
    get_labeling_guide,
    initialize_annotation_run,
    load_review_units,
    read_jsonl_if_exists,
    write_annotation_state,
)
from llm_structured_semantic_generation.resumable_run import utc_now_iso


AGENT_POLICY_VERSION = "agent_alpha_metric_labeler_v2"
ALPHA_PAIRS_ARTIFACT = "agent_alpha_pairs.jsonl"
ALPHA_REPORT_ARTIFACT = "agent_alpha_pair_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate alpha agent preference suggestions for DPO annotation review.",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=REPO_ROOT / "results" / "dpo_kubernetes_v1" / "candidate_generation",
    )
    parser.add_argument("--candidate-run-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "dataset_train_ready.jsonl",
    )
    parser.add_argument(
        "--prompt-requirements-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "prompt_requirements.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dpo_kubernetes_v1" / "preference_annotation" / "manual-v1",
    )
    parser.add_argument("--run-id", default="dpo-agent-alpha-first50-v1")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=4)
    parser.add_argument("--min-strong-margin", type=float, default=0.25)
    parser.add_argument("--min-intermediate-margin", type=float, default=0.15)
    parser.add_argument("--min-low-margin", type=float, default=0.05)
    parser.add_argument("--prompt-f1-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_run_dirs = args.candidate_run_dir or discover_candidate_run_dirs(args.candidate_root)
    if not candidate_run_dirs:
        raise RuntimeError(f"No candidate runs found under {args.candidate_root}")
    paths = initialize_annotation_run(
        output_dir=args.output_dir,
        run_id=args.run_id,
        candidate_run_dirs=candidate_run_dirs,
        dataset_path=args.dataset_path,
        prompt_requirements_path=args.prompt_requirements_path,
        split=args.split,
        batch_size=args.batch_size,
    )
    units = load_review_units(
        candidate_run_dirs=candidate_run_dirs,
        dataset_path=args.dataset_path,
        prompt_requirements_path=args.prompt_requirements_path,
        split=args.split,
    )
    selected_units = units[: args.max_prompts]
    existing_ids = {
        str(row.get("annotation_id"))
        for row in read_jsonl_if_exists(paths.agent_suggestions, allow_truncated_last_line=True)
        if row.get("annotation_id")
    }

    generated_events: list[dict[str, Any]] = []
    skipped_existing = 0
    empty_units = 0
    for unit in selected_units:
        pair_payloads = select_alpha_pairs(
            unit,
            max_pairs_per_prompt=args.max_pairs_per_prompt,
            min_strong_margin=args.min_strong_margin,
            min_intermediate_margin=args.min_intermediate_margin,
            min_low_margin=args.min_low_margin,
            prompt_f1_tolerance=args.prompt_f1_tolerance,
        )
        if not pair_payloads:
            empty_units += 1
        for payload in pair_payloads:
            payload["annotation_id"] = deterministic_annotation_id(unit["unit_id"], payload)
            if payload["annotation_id"] in existing_ids:
                skipped_existing += 1
                continue
            event = append_agent_suggestion(
                paths=paths,
                unit=unit,
                payload=payload,
                agent_name=AGENT_POLICY_VERSION,
            )
            generated_events.append(event)
            existing_ids.add(event["annotation_id"])

    report = build_report(
        args=args,
        units=selected_units,
        generated_events=generated_events,
        skipped_existing=skipped_existing,
        empty_units=empty_units,
    )
    write_jsonl(args.output_dir / ALPHA_PAIRS_ARTIFACT, generated_events)
    write_json(args.output_dir / ALPHA_REPORT_ARTIFACT, report)
    write_annotation_state(paths=paths, units=units)
    print(f"Generated {len(generated_events)} alpha agent suggestions for {len(selected_units)} prompts.")
    print(f"Skipped {skipped_existing} duplicate suggestions.")
    print(f"Wrote {args.output_dir / ALPHA_PAIRS_ARTIFACT}")
    print(f"Wrote {args.output_dir / ALPHA_REPORT_ARTIFACT}")
    return 0


def select_alpha_pairs(
    unit: dict[str, Any],
    *,
    max_pairs_per_prompt: int,
    min_strong_margin: float,
    min_intermediate_margin: float,
    min_low_margin: float,
    prompt_f1_tolerance: float,
) -> list[dict[str, Any]]:
    candidates = unit.get("candidates", [])
    chosen_pool = [candidate for candidate in candidates if can_be_chosen(candidate)]
    parseable_pool = [candidate for candidate in candidates if is_parseable(candidate)]
    invalid_pool = [candidate for candidate in candidates if not is_parseable(candidate)]
    if not chosen_pool:
        return []

    chosen_pool.sort(key=candidate_rank, reverse=True)
    parseable_pool.sort(key=candidate_rank, reverse=True)
    invalid_pool.sort(key=candidate_rank)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    best = chosen_pool[0]
    worst_parseable = next((candidate for candidate in reversed(parseable_pool) if candidate is not best), None)
    if worst_parseable is not None and score_margin(best, worst_parseable) >= min_strong_margin:
        add_pair(
            pairs,
            seen,
            unit=unit,
            chosen=best,
            rejected=worst_parseable,
            pair_type="strong_score_margin",
            confidence="high",
        )

    gate_pair = best_gate_pair(
        chosen_pool=chosen_pool,
        parseable_pool=parseable_pool,
        prompt_f1_tolerance=prompt_f1_tolerance,
        min_gate_margin=min_strong_margin,
    )
    if gate_pair is not None:
        add_pair(
            pairs,
            seen,
            unit=unit,
            chosen=gate_pair[0],
            rejected=gate_pair[1],
            pair_type="gate_practice",
            confidence="high" if prompt_f1(gate_pair[0]) >= prompt_f1(gate_pair[1]) else "medium",
        )

    for rejected in reversed(parseable_pool):
        if len(pairs) >= max_pairs_per_prompt:
            break
        if rejected["candidate_key"] == best["candidate_key"]:
            continue
        if score_margin(best, rejected) < min_intermediate_margin:
            continue
        add_pair(
            pairs,
            seen,
            unit=unit,
            chosen=best,
            rejected=rejected,
            pair_type="intermediate_hard_negative",
            confidence="medium",
        )

    for rejected in reversed(parseable_pool):
        if len(pairs) >= max_pairs_per_prompt:
            break
        if rejected["candidate_key"] == best["candidate_key"]:
            continue
        if score_margin(best, rejected) < min_low_margin:
            continue
        add_pair(
            pairs,
            seen,
            unit=unit,
            chosen=best,
            rejected=rejected,
            pair_type="low_margin_alpha",
            confidence="low",
        )

    if len(pairs) < max_pairs_per_prompt and invalid_pool:
        add_pair(
            pairs,
            seen,
            unit=unit,
            chosen=best,
            rejected=invalid_pool[0],
            pair_type="invalid_negative_alpha",
            confidence="medium",
        )

    return pairs[:max_pairs_per_prompt]


def add_pair(
    pairs: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    *,
    unit: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    confidence: str,
) -> None:
    pair_key = (chosen["candidate_key"], rejected["candidate_key"])
    if pair_key in seen:
        return
    seen.add(pair_key)
    margin = score_margin(chosen, rejected)
    flags = pair_flags(chosen, rejected, pair_type)
    pairs.append(
        {
            "decision": "preference",
            "chosen_candidate_key": chosen["candidate_key"],
            "rejected_candidate_key": rejected["candidate_key"],
            "confidence": confidence,
            "rationale": rationale(unit, chosen, rejected, pair_type, margin),
            "metric_flags": flags,
            "review_status": "pending",
            "pair_type": pair_type,
            "score_margin": round(margin, 6),
            "agent_policy_version": AGENT_POLICY_VERSION,
        }
    )


def best_gate_pair(
    *,
    chosen_pool: list[dict[str, Any]],
    parseable_pool: list[dict[str, Any]],
    prompt_f1_tolerance: float,
    min_gate_margin: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    gate_candidates = [candidate for candidate in chosen_pool if gate_pass(candidate)]
    non_gate_candidates = [candidate for candidate in parseable_pool if not gate_pass(candidate)]
    if not gate_candidates or not non_gate_candidates:
        return None
    gate_candidates.sort(key=candidate_rank, reverse=True)
    non_gate_candidates.sort(key=candidate_rank, reverse=True)
    for chosen in gate_candidates:
        for rejected in non_gate_candidates:
            if chosen["candidate_key"] == rejected["candidate_key"]:
                continue
            if score_margin(chosen, rejected) < min_gate_margin:
                continue
            if required_field_complete(chosen) < required_field_complete(rejected):
                continue
            if prompt_f1(chosen) + prompt_f1_tolerance < prompt_f1(rejected):
                continue
            return chosen, rejected
    return None


def can_be_chosen(candidate: dict[str, Any]) -> bool:
    return is_parseable(candidate) and not bool(candidate.get("hard_invalid"))


def is_parseable(candidate: dict[str, Any]) -> bool:
    metrics = candidate.get("metrics", {})
    yaml_ok = metrics.get("yaml_parse_ok")
    block_ok = metrics.get("block_parse_ok")
    if yaml_ok is not True or block_ok is not True:
        return False
    if candidate.get("hard_invalid") is True:
        return False
    return bool(candidate.get("generation_ok", True))


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = candidate.get("metrics", {})
    return (
        1.0 if gate_pass(candidate) else 0.0,
        numeric(candidate.get("preference_score")),
        numeric(metrics.get("kubernetes_domain_validity_score")),
        prompt_f1(candidate),
        numeric(metrics.get("required_field_complete_resource_rate")),
    )


def score_margin(chosen: dict[str, Any], rejected: dict[str, Any]) -> float:
    return numeric(chosen.get("preference_score")) - numeric(rejected.get("preference_score"))


def gate_pass(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("metrics", {}).get("kubernetes_domain_gate_pass"))


def prompt_f1(candidate: dict[str, Any]) -> float:
    return numeric(candidate.get("metrics", {}).get("prompt_requirement_f1"))


def required_field_complete(candidate: dict[str, Any]) -> float:
    return numeric(candidate.get("metrics", {}).get("required_field_complete_resource_rate"))


def numeric(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def pair_flags(chosen: dict[str, Any], rejected: dict[str, Any], pair_type: str) -> list[str]:
    flags = [pair_type]
    if pair_type == "gate_practice":
        flags.append("security_boilerplate_ok")
    if pair_type == "intermediate_hard_negative":
        flags.append("hard_negative")
    if pair_type == "invalid_negative_alpha":
        flags.append("invalid_rejected_alpha")
    if pair_type == "low_margin_alpha":
        flags.append("alpha_low_margin")
    if prompt_f1(chosen) < prompt_f1(rejected):
        flags.append("gate_pass_prompt_drift")
    return flags


def rationale(
    unit: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    margin: float,
) -> str:
    chosen_metrics = chosen.get("metrics", {})
    rejected_metrics = rejected.get("metrics", {})
    return (
        f"Alpha agent suggestion for {unit['unit_id']} ({pair_type}). "
        f"Chosen score={numeric(chosen.get('preference_score')):.3f}, "
        f"rejected score={numeric(rejected.get('preference_score')):.3f}, margin={margin:.3f}. "
        f"Chosen gate={bool(chosen_metrics.get('kubernetes_domain_gate_pass'))}, "
        f"rejected gate={bool(rejected_metrics.get('kubernetes_domain_gate_pass'))}; "
        f"chosen prompt_f1={numeric(chosen_metrics.get('prompt_requirement_f1')):.3f}, "
        f"rejected prompt_f1={numeric(rejected_metrics.get('prompt_requirement_f1')):.3f}. "
        "Pending human review before final DPO export."
    )


def deterministic_annotation_id(unit_id: str, payload: dict[str, Any]) -> str:
    raw = "|".join(
        [
            AGENT_POLICY_VERSION,
            unit_id,
            str(payload.get("pair_type") or ""),
            str(payload.get("chosen_candidate_key") or ""),
            str(payload.get("rejected_candidate_key") or ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{AGENT_POLICY_VERSION}::{unit_id}::{digest}"


def build_report(
    *,
    args: argparse.Namespace,
    units: list[dict[str, Any]],
    generated_events: list[dict[str, Any]],
    skipped_existing: int,
    empty_units: int,
) -> dict[str, Any]:
    pair_type_counts: dict[str, int] = {}
    for event in generated_events:
        pair_type = str(event.get("pair_type") or "unknown")
        pair_type_counts[pair_type] = pair_type_counts.get(pair_type, 0) + 1
    return {
        "run_id": args.run_id,
        "agent_policy_version": AGENT_POLICY_VERSION,
        "labeling_guide_version": get_labeling_guide()["version"],
        "generated_at": utc_now_iso(),
        "split": args.split,
        "max_prompts": args.max_prompts,
        "prompt_count": len(units),
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "generated_pair_count": len(generated_events),
        "skipped_existing_pair_count": skipped_existing,
        "empty_unit_count": empty_units,
        "pair_type_counts": pair_type_counts,
        "parameters": {
            "min_strong_margin": args.min_strong_margin,
            "min_intermediate_margin": args.min_intermediate_margin,
            "min_low_margin": args.min_low_margin,
            "prompt_f1_tolerance": args.prompt_f1_tolerance,
        },
        "artifacts": {
            "agent_suggestions": AGENT_SUGGESTIONS_ARTIFACT,
            "alpha_pairs": ALPHA_PAIRS_ARTIFACT,
            "alpha_report": ALPHA_REPORT_ARTIFACT,
        },
        "limitations": [
            "Alpha suggestions are metric-guided and pending human review.",
            "Prompt F1 is approximate and can under-extract or misread prompt requirements.",
            "Gate-pass candidates are preferred only when prompt F1 does not indicate worse prompt adequacy.",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
