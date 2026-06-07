from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_structured_semantic_generation.dataset_io import write_json, write_jsonl
from llm_structured_semantic_generation.dpo_preference_annotation import load_review_units
from llm_structured_semantic_generation.resumable_run import utc_now_iso


POLICY_VERSION = "dpo_v2_metric_hard_negative_selector_v1"
LABELING_GUIDE_VERSION = "dpo_kubernetes_labeling_guide_v2"

DEFAULT_V2_RUN_IDS = (
    "dpo-candidates-v2-beta010-offset000-n20-c6-hot-20260531",
    "dpo-candidates-v2-beta010-offset020-n50-c6-hot-20260531",
    "dpo-candidates-v2-beta010-offset070-n100-c6-hot-20260531",
    "dpo-candidates-v2-beta010-offset170-n256-c6-hot-20260602",
)

PAIR_TYPE_PRIORITY = {
    "gate_crossing": 50,
    "level5_practice": 40,
    "domain_invariant": 30,
    "prompt_fidelity": 20,
    "structural_fidelity": 10,
}

REQUIRED_GUARDRAIL_METRICS = (
    "prompt_requirement_f1",
    "line_text_f1",
    "level_exact_match_rate",
    "required_field_presence_rate",
)

LEVEL5_PRACTICE_CATEGORIES = {
    "missing_resource_requirement",
    "latest_image_tag",
    "missing_run_as_non_root",
    "missing_read_only_root_filesystem",
    "privileged_container",
    "host_network",
    "host_pid",
    "host_ipc",
}

DOMAIN_INVARIANT_CATEGORIES = {
    "service_selector_without_workload",
    "selector_template_mismatch",
    "volume_mount_without_volume",
    "configmap_reference_missing",
    "secret_reference_missing",
    "rbac_incomplete",
    "invalid_port",
    "container_missing_image",
    "required_field",
    "kubernetes_identity",
}


@dataclass(frozen=True)
class PairProposal:
    unit: dict[str, Any]
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    pair_type: str
    confidence: str
    score_margin: float
    pair_score: float
    metric_flags: list[str]
    rationale: str
    diagnostics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the automatic DPO preference dataset v2 from DPO-generated Kubernetes candidates.",
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
        default=REPO_ROOT / "results" / "dpo_kubernetes_v1" / "preference_annotation" / "agent-full-auto-v2",
    )
    parser.add_argument("--run-id", default="agent-full-auto-v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-pairs-per-unit", type=int, default=4)
    parser.add_argument("--score-margin", type=float, default=0.25)
    parser.add_argument("--prompt-f1-tolerance", type=float, default=0.05)
    parser.add_argument("--line-f1-tolerance", type=float, default=0.10)
    parser.add_argument("--level-tolerance", type=float, default=0.15)
    parser.add_argument("--required-presence-tolerance", type=float, default=0.05)
    parser.add_argument("--near-duplicate-ratio", type=float, default=0.985)
    parser.add_argument("--ceiling-final-pairs", type=int, default=1500)
    parser.add_argument("--target-min-pairs", type=int, default=800)
    parser.add_argument("--target-max-pairs", type=int, default=1200)
    parser.add_argument(
        "--v1-report-path",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "dpo_kubernetes_v1"
        / "preference_annotation"
        / "agent-full-auto-v1"
        / "combined_preference_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_run_dirs = args.candidate_run_dir or default_v2_candidate_run_dirs(args.candidate_root)
    missing = [path for path in candidate_run_dirs if not (path / "candidates.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"missing_candidate_runs:{missing}")

    units = load_review_units(
        candidate_run_dirs=candidate_run_dirs,
        dataset_path=args.dataset_path,
        prompt_requirements_path=args.prompt_requirements_path,
        split=args.split,
    )
    proposals = select_v2_pairs(units, args=args)
    deduped, duplicate_count = dedupe_proposals(proposals)
    final_proposals = sorted(deduped, key=proposal_sort_key, reverse=True)[: args.ceiling_final_pairs]
    final_rows = [final_preference_row(proposal) for proposal in sort_final_rows(final_proposals)]
    final_rows_with_source = [{**row, "source_preference_dataset": "v2"} for row in final_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "preferences_final.jsonl", final_rows)
    write_jsonl(args.output_dir / "preferences_final_with_source.jsonl", final_rows_with_source)
    report = build_report(
        args=args,
        units=units,
        candidate_run_dirs=candidate_run_dirs,
        proposals=proposals,
        final_proposals=final_proposals,
        duplicate_count=duplicate_count,
    )
    write_json(args.output_dir / "v2_preference_report.json", report)
    (args.output_dir / "v2_preference_analysis.md").write_text(render_analysis(report), encoding="utf-8")
    print(f"Generated {len(final_rows)} v2 preference pairs for {report['unit_with_pair_count']} train units.")
    print(f"Wrote {args.output_dir / 'preferences_final.jsonl'}")
    print(f"Wrote {args.output_dir / 'v2_preference_report.json'}")
    return 0


def default_v2_candidate_run_dirs(candidate_root: Path) -> list[Path]:
    return [candidate_root / run_id for run_id in DEFAULT_V2_RUN_IDS]


def select_v2_pairs(units: Iterable[dict[str, Any]], *, args: argparse.Namespace) -> list[PairProposal]:
    proposals: list[PairProposal] = []
    for unit in units:
        unit_proposals = select_unit_pairs(unit, args=args)
        proposals.extend(unit_proposals)
    return proposals


def select_unit_pairs(unit: dict[str, Any], *, args: argparse.Namespace) -> list[PairProposal]:
    candidates = [candidate for candidate in unit.get("candidates", []) if eligible_for_pair(candidate)]
    candidates.sort(key=candidate_rank, reverse=True)
    raw: list[PairProposal] = []
    for chosen in candidates:
        for rejected in candidates:
            if chosen["candidate_key"] == rejected["candidate_key"]:
                continue
            proposal = build_pair_proposal(unit, chosen, rejected, args=args)
            if proposal is not None:
                raw.append(proposal)
    raw.sort(key=proposal_sort_key, reverse=True)

    selected: list[PairProposal] = []
    used_types: set[str] = set()
    used_rejected_outputs: set[str] = set()
    for proposal in raw:
        if len(selected) >= args.max_pairs_per_unit:
            break
        if proposal.pair_type in used_types:
            continue
        rejected_fingerprint = normalize_text(proposal.rejected.get("reconstructed_yaml") or proposal.rejected.get("model_output_text"))
        if rejected_fingerprint in used_rejected_outputs:
            continue
        selected.append(proposal)
        used_types.add(proposal.pair_type)
        used_rejected_outputs.add(rejected_fingerprint)
    return selected


def build_pair_proposal(
    unit: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> PairProposal | None:
    diagnostics = pair_diagnostics(chosen, rejected)
    if diagnostics["score_margin"] < args.score_margin:
        return None
    if not guardrails_pass(diagnostics, args=args):
        return None
    if near_duplicate(chosen, rejected, threshold=args.near_duplicate_ratio):
        return None
    pair_type = classify_pair(diagnostics)
    if pair_type is None:
        return None

    flags = metric_flags(pair_type, diagnostics)
    confidence = "high" if diagnostics["score_margin"] >= 0.50 and "gate_pass_prompt_drift" not in flags else "medium"
    return PairProposal(
        unit=unit,
        chosen=chosen,
        rejected=rejected,
        pair_type=pair_type,
        confidence=confidence,
        score_margin=round(diagnostics["score_margin"], 6),
        pair_score=pair_score(pair_type, diagnostics),
        metric_flags=flags,
        rationale=rationale(unit, chosen, rejected, pair_type, diagnostics),
        diagnostics=diagnostics,
    )


def eligible_for_pair(candidate: dict[str, Any]) -> bool:
    if candidate.get("generation_ok") is False or candidate.get("hard_invalid") is True:
        return False
    return bool(metric(candidate, "yaml_parse_ok") is True and metric(candidate, "block_parse_ok") is True)


def guardrails_pass(diagnostics: dict[str, Any], *, args: argparse.Namespace) -> bool:
    for field in REQUIRED_GUARDRAIL_METRICS:
        if diagnostics[f"chosen_{field}"] is None or diagnostics[f"rejected_{field}"] is None:
            return False
    if diagnostics["prompt_requirement_f1_delta"] < -args.prompt_f1_tolerance:
        return False
    if diagnostics["line_text_f1_delta"] < -args.line_f1_tolerance:
        return False
    if diagnostics["level_exact_match_rate_delta"] < -args.level_tolerance:
        return False
    if diagnostics["required_field_presence_rate_delta"] < -args.required_presence_tolerance:
        return False
    if diagnostics["required_field_complete_resource_rate_delta"] < -0.05:
        return False
    return True


def classify_pair(diagnostics: dict[str, Any]) -> str | None:
    if diagnostics["chosen_gate_pass"] and not diagnostics["rejected_gate_pass"]:
        return "gate_crossing"
    if not diagnostics["chosen_gate_pass"] and not diagnostics["rejected_gate_pass"]:
        if diagnostics["level5_error_delta"] >= 1 or diagnostics["level5_score_delta"] > 0.0:
            return "level5_practice"
    if (
        diagnostics["domain_error_delta"] >= 1
        or diagnostics["kubernetes_domain_validity_score_delta"] >= 0.10
        or diagnostics["kubernetes_domain_validity_level_delta"] >= 1
    ):
        return "domain_invariant"
    if diagnostics["prompt_requirement_f1_delta"] >= 0.20:
        return "prompt_fidelity"
    if diagnostics["level_exact_match_rate_delta"] >= 0.20 or diagnostics["line_text_f1_delta"] >= 0.15:
        return "structural_fidelity"
    return None


def pair_diagnostics(chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "chosen_gate_pass": gate_pass(chosen),
        "rejected_gate_pass": gate_pass(rejected),
        "score_margin": score(chosen) - score(rejected),
        "chosen_kubernetes_domain_validity_level": numeric(metric(chosen, "kubernetes_domain_validity_level")),
        "rejected_kubernetes_domain_validity_level": numeric(metric(rejected, "kubernetes_domain_validity_level")),
        "chosen_level5_score": level_score(chosen, "level_5_static_quality_smells"),
        "rejected_level5_score": level_score(rejected, "level_5_static_quality_smells"),
        "chosen_level5_error_count": len(level_errors(chosen, levels={5}, categories=LEVEL5_PRACTICE_CATEGORIES)),
        "rejected_level5_error_count": len(level_errors(rejected, levels={5}, categories=LEVEL5_PRACTICE_CATEGORIES)),
        "chosen_domain_error_count": len(level_errors(chosen, levels={3, 4}, categories=DOMAIN_INVARIANT_CATEGORIES)),
        "rejected_domain_error_count": len(level_errors(rejected, levels={3, 4}, categories=DOMAIN_INVARIANT_CATEGORIES)),
    }
    for field in (
        "prompt_requirement_f1",
        "line_text_f1",
        "level_exact_match_rate",
        "required_field_presence_rate",
        "required_field_complete_resource_rate",
        "kubernetes_domain_validity_score",
    ):
        chosen_value = optional_numeric(metric(chosen, field))
        rejected_value = optional_numeric(metric(rejected, field))
        diagnostics[f"chosen_{field}"] = chosen_value
        diagnostics[f"rejected_{field}"] = rejected_value
        diagnostics[f"{field}_delta"] = delta(chosen_value, rejected_value)
    diagnostics["kubernetes_domain_validity_level_delta"] = (
        diagnostics["chosen_kubernetes_domain_validity_level"] - diagnostics["rejected_kubernetes_domain_validity_level"]
    )
    diagnostics["level5_score_delta"] = diagnostics["chosen_level5_score"] - diagnostics["rejected_level5_score"]
    diagnostics["level5_error_delta"] = diagnostics["rejected_level5_error_count"] - diagnostics["chosen_level5_error_count"]
    diagnostics["domain_error_delta"] = diagnostics["rejected_domain_error_count"] - diagnostics["chosen_domain_error_count"]
    return diagnostics


def pair_score(pair_type: str, diagnostics: dict[str, Any]) -> float:
    return (
        PAIR_TYPE_PRIORITY[pair_type]
        + diagnostics["score_margin"]
        + max(diagnostics["kubernetes_domain_validity_score_delta"], 0.0)
        + max(diagnostics["prompt_requirement_f1_delta"], 0.0)
        + max(diagnostics["level_exact_match_rate_delta"], 0.0) * 0.5
        + max(diagnostics["line_text_f1_delta"], 0.0) * 0.5
        + max(diagnostics["level5_error_delta"], 0.0) * 0.1
        + max(diagnostics["domain_error_delta"], 0.0) * 0.1
    )


def metric_flags(pair_type: str, diagnostics: dict[str, Any]) -> list[str]:
    flags = [pair_type, "hard_negative"]
    if pair_type == "gate_crossing":
        flags.append("security_boilerplate_ok")
    if pair_type == "level5_practice":
        flags.append("level5_practice_gain")
    if pair_type == "domain_invariant":
        flags.append("domain_invariant_gain")
    if pair_type == "prompt_fidelity":
        flags.append("prompt_metric_supported")
    if pair_type == "structural_fidelity":
        flags.append("structural_metric_gain")
    if diagnostics["prompt_requirement_f1_delta"] < 0:
        flags.append("gate_pass_prompt_drift")
    return flags


def rationale(
    unit: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    diagnostics: dict[str, Any],
) -> str:
    return (
        f"Automatic v2 preference for {unit['unit_id']} ({pair_type}). "
        f"Chosen score={score(chosen):.3f}, rejected score={score(rejected):.3f}, "
        f"margin={diagnostics['score_margin']:.3f}. "
        f"Gate {diagnostics['chosen_gate_pass']} vs {diagnostics['rejected_gate_pass']}; "
        f"prompt_f1_delta={diagnostics['prompt_requirement_f1_delta']:.3f}, "
        f"kdv_delta={diagnostics['kubernetes_domain_validity_score_delta']:.3f}, "
        f"level5_error_delta={diagnostics['level5_error_delta']}, "
        f"domain_error_delta={diagnostics['domain_error_delta']}. "
        "The pair passed the v2 automatic guardrails and keeps both sides parseable."
    )


def dedupe_proposals(proposals: list[PairProposal]) -> tuple[list[PairProposal], int]:
    best_by_key: dict[tuple[str, str, str, str], PairProposal] = {}
    duplicate_count = 0
    for proposal in proposals:
        key = (
            proposal.unit["unit_id"],
            normalize_text(proposal.chosen.get("model_output_text")),
            normalize_text(proposal.rejected.get("model_output_text")),
            proposal.pair_type,
        )
        existing = best_by_key.get(key)
        if existing is None or proposal_sort_key(proposal) > proposal_sort_key(existing):
            if existing is not None:
                duplicate_count += 1
            best_by_key[key] = proposal
        else:
            duplicate_count += 1
    return list(best_by_key.values()), duplicate_count


def proposal_sort_key(proposal: PairProposal) -> tuple[float, float, str, str]:
    return (
        proposal.pair_score,
        proposal.score_margin,
        proposal.unit["unit_id"],
        proposal.chosen["candidate_key"],
    )


def sort_final_rows(proposals: list[PairProposal]) -> list[PairProposal]:
    return sorted(
        proposals,
        key=lambda proposal: (
            unit_numeric_key(proposal.unit["unit_id"]),
            proposal.unit["unit_id"],
            -PAIR_TYPE_PRIORITY[proposal.pair_type],
            proposal.rejected["candidate_key"],
        ),
    )


def final_preference_row(proposal: PairProposal) -> dict[str, Any]:
    unit = proposal.unit
    chosen = proposal.chosen
    rejected = proposal.rejected
    annotation_id = deterministic_annotation_id(proposal)
    return {
        "preference_id": f"{unit['unit_id']}::{annotation_id}",
        "unit_id": unit["unit_id"],
        "sample_id": unit["sample_id"],
        "prompt_variant": unit["prompt_variant"],
        "split": unit["split"],
        "natural_language_prompt": unit.get("prompt_text"),
        "prompt": unit.get("sft_prompt"),
        "reference_yaml": unit.get("reference_yaml"),
        "chosen_candidate_key": chosen["candidate_key"],
        "rejected_candidate_key": rejected["candidate_key"],
        "chosen": chosen.get("model_output_text"),
        "rejected": rejected.get("model_output_text"),
        "chosen_reconstructed_yaml": chosen.get("reconstructed_yaml"),
        "rejected_reconstructed_yaml": rejected.get("reconstructed_yaml"),
        "chosen_metrics": chosen.get("metrics"),
        "rejected_metrics": rejected.get("metrics"),
        "annotation_id": annotation_id,
        "label_source": "agent",
        "labeling_guide_version": LABELING_GUIDE_VERSION,
        "pair_type": proposal.pair_type,
        "score_margin": proposal.score_margin,
        "agent_policy_version": POLICY_VERSION,
        "source_annotation_id": None,
        "confidence": proposal.confidence,
        "rationale": proposal.rationale,
        "metric_flags": proposal.metric_flags,
        "created_at": utc_now_iso(),
    }


def deterministic_annotation_id(proposal: PairProposal) -> str:
    raw = "|".join(
        [
            POLICY_VERSION,
            proposal.unit["unit_id"],
            proposal.pair_type,
            proposal.chosen["candidate_key"],
            proposal.rejected["candidate_key"],
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{POLICY_VERSION}::{proposal.unit['unit_id']}::{digest}"


def build_report(
    *,
    args: argparse.Namespace,
    units: list[dict[str, Any]],
    candidate_run_dirs: list[Path],
    proposals: list[PairProposal],
    final_proposals: list[PairProposal],
    duplicate_count: int,
) -> dict[str, Any]:
    unit_with_pair = {proposal.unit["unit_id"] for proposal in final_proposals}
    pair_type_counts = Counter(proposal.pair_type for proposal in final_proposals)
    confidence_counts = Counter(proposal.confidence for proposal in final_proposals)
    flag_counts = Counter(flag for proposal in final_proposals for flag in proposal.metric_flags)
    pairs_per_unit = Counter(proposal.unit["unit_id"] for proposal in final_proposals)
    chosen_levels = Counter(
        str(int(proposal.diagnostics["chosen_kubernetes_domain_validity_level"])) for proposal in final_proposals
    )
    rejected_levels = Counter(
        str(int(proposal.diagnostics["rejected_kubernetes_domain_validity_level"])) for proposal in final_proposals
    )
    return {
        "run_id": args.run_id,
        "document_type": "run result",
        "generated_at": utc_now_iso(),
        "labeling_guide_version": LABELING_GUIDE_VERSION,
        "agent_policy_version": POLICY_VERSION,
        "split": args.split,
        "candidate_run_dirs": [str(path) for path in candidate_run_dirs],
        "candidate_run_ids": [path.name for path in candidate_run_dirs],
        "unit_count": len(units),
        "candidate_count": sum(len(unit.get("candidates", [])) for unit in units),
        "raw_pair_count": len(proposals),
        "deduped_pair_count": len(proposals) - duplicate_count,
        "duplicate_count": duplicate_count,
        "final_pair_count": len(final_proposals),
        "target_final_pairs": {
            "min": args.target_min_pairs,
            "max": args.target_max_pairs,
            "ceiling": args.ceiling_final_pairs,
            "status": target_status(len(final_proposals), args=args),
        },
        "unit_with_pair_count": len(unit_with_pair),
        "empty_unit_count": len(units) - len(unit_with_pair),
        "pairs_per_unit_distribution": dict(Counter(pairs_per_unit.values())),
        "pair_type_counts": dict(pair_type_counts),
        "confidence_counts": dict(confidence_counts),
        "metric_flag_counts": dict(flag_counts),
        "chosen_kubernetes_domain_validity_level_counts": dict(chosen_levels),
        "rejected_kubernetes_domain_validity_level_counts": dict(rejected_levels),
        "gate_crossing_count": pair_type_counts.get("gate_crossing", 0),
        "average_score_margin": safe_mean([proposal.score_margin for proposal in final_proposals]),
        "metric_delta_summary": metric_delta_summary(final_proposals),
        "parseability_checks": parseability_checks(final_proposals),
        "v1_comparison": load_v1_comparison(args.v1_report_path, final_proposals=final_proposals),
        "limiting_reasons": limiting_reasons(
            final_pair_count=len(final_proposals),
            unit_count=len(units),
            unit_with_pair_count=len(unit_with_pair),
            args=args,
        ),
        "guardrails": {
            "score_margin": args.score_margin,
            "prompt_f1_tolerance": args.prompt_f1_tolerance,
            "line_f1_tolerance": args.line_f1_tolerance,
            "level_tolerance": args.level_tolerance,
            "required_presence_tolerance": args.required_presence_tolerance,
            "near_duplicate_ratio": args.near_duplicate_ratio,
            "max_pairs_per_unit": args.max_pairs_per_unit,
            "ceiling_final_pairs": args.ceiling_final_pairs,
        },
        "artifacts": {
            "preferences_final": str(args.output_dir / "preferences_final.jsonl"),
            "preferences_final_with_source": str(args.output_dir / "preferences_final_with_source.jsonl"),
            "report": str(args.output_dir / "v2_preference_report.json"),
            "analysis_md": str(args.output_dir / "v2_preference_analysis.md"),
        },
        "limitations": [
            "This is an automatic proxy preference dataset, not a human preference dataset.",
            "Prompt adequacy still uses the approximate prompt_requirement_f1 metric as a guardrail.",
            "Security-boilerplate intent changes are approximated through prompt, line, level, and required-field guardrails.",
            "Kubernetes-domain gate pass is a partial static validator, not full Kubernetes schema validation.",
        ],
    }


def metric_delta_summary(proposals: list[PairProposal]) -> dict[str, Any]:
    fields = (
        "score_margin",
        "prompt_requirement_f1_delta",
        "line_text_f1_delta",
        "level_exact_match_rate_delta",
        "required_field_presence_rate_delta",
        "required_field_complete_resource_rate_delta",
        "kubernetes_domain_validity_score_delta",
        "kubernetes_domain_validity_level_delta",
        "level5_error_delta",
        "domain_error_delta",
    )
    return {field: describe([proposal.diagnostics[field] for proposal in proposals]) for field in fields}


def parseability_checks(proposals: list[PairProposal]) -> dict[str, Any]:
    return {
        "chosen_yaml_parse_ok_rate": rate(metric(proposal.chosen, "yaml_parse_ok") is True for proposal in proposals),
        "chosen_block_parse_ok_rate": rate(metric(proposal.chosen, "block_parse_ok") is True for proposal in proposals),
        "rejected_yaml_parse_ok_rate": rate(metric(proposal.rejected, "yaml_parse_ok") is True for proposal in proposals),
        "rejected_block_parse_ok_rate": rate(metric(proposal.rejected, "block_parse_ok") is True for proposal in proposals),
    }


def render_analysis(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Automatic DPO Preference Dataset v2",
            "",
            "Document type: run result",
            "",
            "## Scope",
            "",
            "This artifact exports automatic v2 DPO preference pairs from the DPO beta=0.10 candidate runs.",
            "",
            "Candidate runs:",
            *[f"- `{run_id}`" for run_id in report["candidate_run_ids"]],
            "",
            "## Final Dataset",
            "",
            f"- Train units considered: {report['unit_count']}",
            f"- Candidate count: {report['candidate_count']}",
            f"- Raw eligible pair proposals: {report['raw_pair_count']}",
            f"- Duplicate proposals removed: {report['duplicate_count']}",
            f"- Final pairs: {report['final_pair_count']}",
            f"- Target status: {json.dumps(report['target_final_pairs'], sort_keys=True)}",
            f"- Units with at least one pair: {report['unit_with_pair_count']}",
            f"- Pair types: {json.dumps(report['pair_type_counts'], sort_keys=True)}",
            f"- Confidence: {json.dumps(report['confidence_counts'], sort_keys=True)}",
            f"- Gate-crossing pairs: {report['gate_crossing_count']}",
            f"- Limiting reasons: {json.dumps(report['limiting_reasons'], sort_keys=True)}",
            "",
            "## Quality Summary",
            "",
            f"- Average score margin: {report['average_score_margin']}",
            f"- Prompt F1 delta: {json.dumps(report['metric_delta_summary']['prompt_requirement_f1_delta'], sort_keys=True)}",
            f"- KDV delta: {json.dumps(report['metric_delta_summary']['kubernetes_domain_validity_score_delta'], sort_keys=True)}",
            f"- Level delta: {json.dumps(report['metric_delta_summary']['level_exact_match_rate_delta'], sort_keys=True)}",
            f"- Line F1 delta: {json.dumps(report['metric_delta_summary']['line_text_f1_delta'], sort_keys=True)}",
            f"- Level 5 error delta: {json.dumps(report['metric_delta_summary']['level5_error_delta'], sort_keys=True)}",
            "",
            "## v1 Comparison",
            "",
            f"- v1 comparison: {json.dumps(report['v1_comparison'], sort_keys=True)}",
            "",
            "## Interpretation",
            "",
            "The v2 selector keeps both chosen and rejected candidates parseable and applies the v2 guardrails before export. "
            "Pairs are intentionally stricter than the v1 automatic dataset and should be treated as proxy preferences, not human labels.",
            "",
        ]
    )


def target_status(final_pair_count: int, *, args: argparse.Namespace) -> str:
    if final_pair_count < args.target_min_pairs:
        return "below_target"
    if final_pair_count > args.target_max_pairs:
        return "above_target"
    return "within_target"


def limiting_reasons(
    *,
    final_pair_count: int,
    unit_count: int,
    unit_with_pair_count: int,
    args: argparse.Namespace,
) -> list[str]:
    if final_pair_count >= args.target_min_pairs:
        return []
    empty_units = unit_count - unit_with_pair_count
    return [
        f"{empty_units} train units had no v2 pair that passed all automatic guardrails.",
        "Both chosen and rejected candidates were required to parse as YAML and satisfy the block contract.",
        "Near-duplicate candidate outputs were skipped instead of counted as additional preference evidence.",
        "The selector keeps at most one pair per failure category per unit, so it does not inflate the dataset with redundant comparisons.",
    ]


def load_v1_comparison(v1_report_path: Path, *, final_proposals: list[PairProposal]) -> dict[str, Any]:
    if not v1_report_path.exists():
        return {"available": False, "path": str(v1_report_path)}
    report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    v2_pair_types = Counter(proposal.pair_type for proposal in final_proposals)
    v2_flags = Counter(flag for proposal in final_proposals for flag in proposal.metric_flags)
    return {
        "available": True,
        "path": str(v1_report_path),
        "v1_pair_count": report.get("deduped_pair_count"),
        "v2_pair_count": len(final_proposals),
        "v1_unit_with_pair_count": report.get("unit_with_pair_count"),
        "v2_unit_with_pair_count": len({proposal.unit["unit_id"] for proposal in final_proposals}),
        "v1_pair_type_counts": report.get("pair_type_counts"),
        "v2_pair_type_counts": dict(v2_pair_types),
        "v1_average_score_margin": (report.get("margin_distribution") or {}).get("mean"),
        "v2_average_score_margin": safe_mean([proposal.score_margin for proposal in final_proposals]),
        "v1_prompt_delta_mean": ((report.get("metric_summary") or {}).get("prompt_requirement_f1") or {})
        .get("delta_chosen_minus_rejected", {})
        .get("mean"),
        "v2_prompt_delta_mean": describe(
            [proposal.diagnostics["prompt_requirement_f1_delta"] for proposal in final_proposals]
        ).get("mean"),
        "v1_kdv_delta_mean": ((report.get("metric_summary") or {}).get("kubernetes_domain_validity_score") or {})
        .get("delta_chosen_minus_rejected", {})
        .get("mean"),
        "v2_kdv_delta_mean": describe(
            [proposal.diagnostics["kubernetes_domain_validity_score_delta"] for proposal in final_proposals]
        ).get("mean"),
        "v1_gate_pair_count": report.get("gate_practice_count"),
        "v2_gate_pair_count": v2_pair_types.get("gate_crossing", 0),
        "v1_gate_prompt_drift_count": report.get("gate_prompt_drift_count"),
        "v2_gate_prompt_drift_count": v2_flags.get("gate_pass_prompt_drift", 0),
    }


def metric(candidate: dict[str, Any], field: str) -> Any:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    if field in metrics:
        return metrics[field]
    evaluation = candidate.get("evaluation") if isinstance(candidate.get("evaluation"), dict) else {}
    return evaluation.get(field)


def score(candidate: dict[str, Any]) -> float:
    return numeric(candidate.get("preference_score") or metric(candidate, "preference_score"))


def gate_pass(candidate: dict[str, Any]) -> bool:
    return bool(metric(candidate, "kubernetes_domain_gate_pass"))


def level_score(candidate: dict[str, Any], key: str) -> float:
    evaluation = candidate.get("evaluation") if isinstance(candidate.get("evaluation"), dict) else {}
    scores = evaluation.get("kubernetes_domain_level_scores")
    if not isinstance(scores, dict):
        return 0.0
    return numeric(scores.get(key))


def level_errors(candidate: dict[str, Any], *, levels: set[int], categories: set[str]) -> list[dict[str, Any]]:
    evaluation = candidate.get("evaluation") if isinstance(candidate.get("evaluation"), dict) else {}
    errors = evaluation.get("kubernetes_domain_errors")
    if not isinstance(errors, list):
        return []
    selected = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        if int(error.get("level") or -1) in levels and str(error.get("category") or "") in categories:
            selected.append(error)
    return selected


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        1.0 if gate_pass(candidate) else 0.0,
        score(candidate),
        numeric(metric(candidate, "kubernetes_domain_validity_score")),
        numeric(metric(candidate, "prompt_requirement_f1")),
        numeric(metric(candidate, "required_field_complete_resource_rate")),
        numeric(metric(candidate, "level_exact_match_rate")),
    )


def near_duplicate(chosen: dict[str, Any], rejected: dict[str, Any], *, threshold: float) -> bool:
    chosen_text = normalize_text(chosen.get("reconstructed_yaml") or chosen.get("model_output_text"))
    rejected_text = normalize_text(rejected.get("reconstructed_yaml") or rejected.get("model_output_text"))
    if not chosen_text or not rejected_text:
        return True
    if chosen_text == rejected_text:
        return True
    return SequenceMatcher(None, chosen_text, rejected_text).ratio() >= threshold


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines() if line.strip())


def unit_numeric_key(unit_id: str) -> tuple[int, str]:
    sample_id = unit_id.split("::", 1)[0]
    if sample_id.startswith("q") and sample_id[1:].isdigit():
        return int(sample_id[1:]), unit_id
    return 10**9, unit_id


def optional_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def numeric(value: Any) -> float:
    result = optional_numeric(value)
    return 0.0 if result is None else result


def delta(chosen_value: float | None, rejected_value: float | None) -> float:
    if chosen_value is None or rejected_value is None:
        return 0.0
    return chosen_value - rejected_value


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": round(sorted_values[0], 6),
        "p25": round(percentile(sorted_values, 0.25), 6),
        "median": round(percentile(sorted_values, 0.50), 6),
        "p75": round(percentile(sorted_values, 0.75), 6),
        "max": round(sorted_values[-1], 6),
        "mean": round(mean(sorted_values), 6),
        "negative_delta_count": sum(1 for value in sorted_values if value < 0),
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = q * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def rate(values: Iterable[bool]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return sum(1 for value in materialized if value) / len(materialized)


if __name__ == "__main__":
    raise SystemExit(main())
