from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dataset_io import append_jsonl, read_jsonl, write_json, write_jsonl
from .prompt_requirements import extract_prompt_requirements
from .resumable_run import utc_now_iso


HUMAN_PREFERENCES_ARTIFACT = "preferences_human.jsonl"
AGENT_SUGGESTIONS_ARTIFACT = "preferences_agent_suggestions.jsonl"
FINAL_PREFERENCES_ARTIFACT = "preferences_final.jsonl"
ANNOTATION_STATE_ARTIFACT = "annotation_state.json"
CONFIG_ARTIFACT = "config.json"

DECISION_PREFERENCE = "preference"
DECISION_TIE = "tie"
DECISION_SKIP = "skip"
VALID_DECISIONS = {DECISION_PREFERENCE, DECISION_TIE, DECISION_SKIP}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_REVIEW_STATUS = {"approved", "pending", "rejected"}
LABELING_GUIDE_VERSION = "dpo_kubernetes_labeling_guide_v1"

METRIC_SUMMARY_FIELDS = (
    "yaml_parse_ok",
    "block_parse_ok",
    "structured_output_parse_success",
    "prompt_requirement_f1",
    "prompt_requirement_precision",
    "prompt_requirement_recall",
    "kubernetes_domain_validity_score",
    "kubernetes_domain_gate_pass",
    "required_field_complete_resource_rate",
    "required_field_presence_rate",
    "level_exact_match_rate",
    "line_text_f1",
    "line_count_reference",
    "line_count_prediction",
    "kind_sequence_match_rate",
)

LABELING_GUIDE: dict[str, Any] = {
    "version": LABELING_GUIDE_VERSION,
    "title": "DPO Kubernetes preference labeling guide",
    "objective": [
        "Prefer candidates that parse correctly and preserve the block contract.",
        "Increase kubernetes_domain_gate_pass_rate when prompt adequacy is preserved.",
        "Keep high prompt adequacy and avoid unsupported changes to the requested manifest.",
        "Reward Kubernetes safety and quality practices only when they do not distort the request.",
    ],
    "decision_rules": [
        "Do not mark a non-parseable YAML or broken block output as chosen for DPO.",
        "A gate-pass candidate may be preferred with prompt_f1_drop <= 0.05 when score_margin >= 0.25, required-field completeness is not worse, and no central prompt requirement is visibly lost.",
        "A prompt_f1_drop tolerance up to 0.10 is only for separately reported sensitivity runs.",
        "Extra security fields are positive when they do not change names, resources, ports, images, relations, or intent.",
        "If gate_pass conflicts with clear prompt fidelity, choose the prompt-faithful candidate or mark tie/skip.",
    ],
    "review_order": [
        "YAML and Blocks: failing candidates are normally rejected only.",
        "Gate pass: a true value is a strong chosen signal when prompt adequacy is preserved or only slightly worse under approximate Prompt F1.",
        "KDV score and level: use as a gradual Kubernetes-quality signal when nobody passes the full gate.",
        "Prompt F1: use as an alarm and verify by reading prompt and YAML.",
        "Req fields, Level, Line F1, and line counts: use as structural tie-breakers.",
    ],
    "kubernetes_levels": [
        "Level 0: YAML parses.",
        "Level 1: parser-facing block contract and reconstruction are satisfied.",
        "Level 2: minimal Kubernetes identity: apiVersion, kind, metadata.name, known kind, required fields.",
        "Level 3: intra-resource invariants: selectors, labels, ports, volumes, images, schedules, replica ranges.",
        "Level 4: inter-resource invariants: Service/workload matches and local references to ConfigMap, Secret, PVC, ServiceAccount, Role.",
        "Level 5: static quality and security smells: requests/limits, no latest, runAsNonRoot, readOnlyRootFilesystem, no privileged or host namespaces.",
    ],
    "pair_types": [
        "Strong pair: chosen is best score, rejected is worse score, with score_margin >= 0.25.",
        "Intermediate pair: both parse, but rejected is worse on prompt, KDV, gate, required fields, or hierarchy.",
        "Gate/practice pair: chosen has gate_pass true and rejected has gate_pass false; default automatic tolerance is prompt_f1_drop <= 0.05 with score_margin >= 0.25, required-field completeness not worse, and no visible loss of central prompt requirements.",
        "Prioritize hard negatives: rejected should be parseable and informative when possible.",
        "Keep at most 3 useful pairs per prompt and avoid near-duplicate comparisons.",
    ],
    "prompt_f1_rules": [
        "Use high Prompt F1 as support only when it matches the visible YAML behavior.",
        "Do not punish a candidate automatically when Prompt F1 is low but the YAML satisfies the prompt.",
        "Do not trust high Prompt F1 if the extractor captured only a trivial atom such as kind.",
        "For gate/practice pairs, flag gate_pass_prompt_drift and use at most medium confidence when chosen Prompt F1 is lower than rejected Prompt F1.",
        "Flag doubtful cases instead of hiding metric weaknesses.",
    ],
    "tie_skip_rules": [
        "Use tie when candidates are effectively equivalent and the pair teaches no clear preference.",
        "Use skip when all candidates are poor, drift from the prompt, or require unreliable prompt metrics to decide.",
        "Use low confidence when the pair depends on semantic judgement not captured by current metrics.",
    ],
    "recommended_metric_flags": [
        "prompt_metric_unreliable",
        "under_extracted_prompt",
        "false_positive_requirement",
        "wrong_resource_context",
        "gate_pass_prompt_drift",
        "security_boilerplate_ok",
        "security_boilerplate_changes_intent",
        "hard_negative",
    ],
    "final_criterion": (
        "A good chosen is the safest and most valid candidate that still answers the prompt. "
        "A good rejected is worse in an informative way: less gate, worse KDV, lower prompt fidelity, "
        "worse structure, or a clear combination of those signals."
    ),
}


class PreferenceAnnotationError(ValueError):
    """Raised when an annotation payload cannot be accepted."""


@dataclass(frozen=True)
class AnnotationPaths:
    output_dir: Path

    @property
    def config(self) -> Path:
        return self.output_dir / CONFIG_ARTIFACT

    @property
    def state(self) -> Path:
        return self.output_dir / ANNOTATION_STATE_ARTIFACT

    @property
    def human_preferences(self) -> Path:
        return self.output_dir / HUMAN_PREFERENCES_ARTIFACT

    @property
    def agent_suggestions(self) -> Path:
        return self.output_dir / AGENT_SUGGESTIONS_ARTIFACT

    @property
    def final_preferences(self) -> Path:
        return self.output_dir / FINAL_PREFERENCES_ARTIFACT


def read_jsonl_if_exists(path: Path, *, allow_truncated_last_line: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path, allow_truncated_last_line=allow_truncated_last_line)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def initialize_annotation_run(
    *,
    output_dir: Path,
    run_id: str,
    candidate_run_dirs: Iterable[Path],
    dataset_path: Path,
    prompt_requirements_path: Path | None,
    split: str,
    batch_size: int,
) -> AnnotationPaths:
    paths = AnnotationPaths(output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": run_id,
        "stage": "dpo_preference_annotation",
        "created_or_resumed_at": utc_now_iso(),
        "candidate_run_dirs": [str(path) for path in candidate_run_dirs],
        "dataset_path": str(dataset_path),
        "prompt_requirements_path": str(prompt_requirements_path) if prompt_requirements_path else None,
        "split": split,
        "batch_size": batch_size,
        "artifacts": {
            "human_preferences": HUMAN_PREFERENCES_ARTIFACT,
            "agent_suggestions": AGENT_SUGGESTIONS_ARTIFACT,
            "final_preferences": FINAL_PREFERENCES_ARTIFACT,
            "state": ANNOTATION_STATE_ARTIFACT,
        },
        "resume_signature": {
            "stage": "dpo_preference_annotation",
            "dataset_path": str(dataset_path),
            "prompt_requirements_path": str(prompt_requirements_path) if prompt_requirements_path else None,
            "split": split,
        },
    }
    existing = load_json_if_exists(paths.config)
    existing_signature = existing.get("annotation_resume_signature")
    if existing_signature is None and existing.get("stage") == "dpo_preference_annotation":
        existing_signature = existing.get("resume_signature")
    if existing_signature is not None and existing_signature != config["resume_signature"]:
        raise PreferenceAnnotationError(
            "annotation_resume_signature_mismatch:"
            f"{paths.config}:existing={existing_signature}:"
            f"new={config['resume_signature']}"
        )
    config["annotation_resume_signature"] = config["resume_signature"]
    write_json(paths.config, {**existing, **config} if existing else config)
    return paths


def load_dataset_index(dataset_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    return {
        (str(row["sample_id"]), str(row["prompt_variant"])): row
        for row in rows
        if "sample_id" in row and "prompt_variant" in row
    }


def load_prompt_requirement_index(prompt_requirements_path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if prompt_requirements_path is None or not prompt_requirements_path.exists():
        return {}
    rows = read_jsonl(prompt_requirements_path)
    return {
        (str(row["sample_id"]), str(row["prompt_variant"])): row
        for row in rows
        if "sample_id" in row and "prompt_variant" in row
    }


def discover_candidate_run_dirs(candidate_root: Path) -> list[Path]:
    if not candidate_root.exists():
        return []
    return sorted(
        path
        for path in candidate_root.iterdir()
        if path.is_dir() and (path / "candidates.jsonl").exists()
    )


def load_review_units(
    *,
    candidate_run_dirs: Iterable[Path],
    dataset_path: Path,
    prompt_requirements_path: Path | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    dataset_index = load_dataset_index(dataset_path)
    prompt_requirement_index = load_prompt_requirement_index(prompt_requirements_path)
    units: dict[str, dict[str, Any]] = {}

    for run_dir in candidate_run_dirs:
        run_id = _load_candidate_run_id(run_dir)
        metric_by_uid = {
            str(row.get("candidate_uid")): row
            for row in read_jsonl_if_exists(run_dir / "candidate_metrics.jsonl", allow_truncated_last_line=True)
            if row.get("candidate_uid")
        }
        for candidate in read_jsonl_if_exists(run_dir / "candidates.jsonl", allow_truncated_last_line=True):
            if split and candidate.get("split") != split:
                continue
            sample_id = str(candidate.get("sample_id", ""))
            prompt_variant = str(candidate.get("prompt_variant", ""))
            if not sample_id or not prompt_variant:
                continue
            unit_id = str(candidate.get("unit_id") or f"{sample_id}::{prompt_variant}")
            key = (sample_id, prompt_variant)
            dataset_row = dataset_index.get(key, {})
            requirement_row = prompt_requirement_index.get(key, {})
            unit = units.setdefault(
                unit_id,
                _build_unit_base(
                    unit_id=unit_id,
                    candidate=candidate,
                    dataset_row=dataset_row,
                    requirement_row=requirement_row,
                ),
            )
            metric_row = metric_by_uid.get(str(candidate.get("candidate_uid")), {})
            unit["candidates"].append(_build_candidate_payload(candidate, metric_row, run_id, run_dir))

    for unit in units.values():
        unit["candidates"].sort(key=_candidate_sort_key)
    return sorted(units.values(), key=_unit_sort_key)


def build_decision_packet(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "labeling_guide": get_labeling_guide(),
        "unit_id": unit["unit_id"],
        "sample_id": unit["sample_id"],
        "prompt_variant": unit["prompt_variant"],
        "split": unit["split"],
        "natural_language_prompt": unit.get("prompt_text"),
        "sft_prompt": unit.get("sft_prompt"),
        "reference_yaml": unit.get("reference_yaml"),
        "prompt_requirements": unit.get("prompt_requirements", []),
        "reference_prompt_requirement_evaluation": unit.get("reference_prompt_requirement_evaluation"),
        "instructions": (
            "Choose the better DPO candidate for the same prompt using labeling_guide. Return JSON with "
            "decision, chosen_candidate_key, rejected_candidate_key, confidence, rationale, and metric_flags. "
            "If no informative pair exists, return tie or skip."
        ),
        "candidates": [
            {
                "candidate_key": candidate["candidate_key"],
                "candidate_uid": candidate["candidate_uid"],
                "candidate_id": candidate["candidate_id"],
                "source_run_id": candidate["source_run_id"],
                "generation_config": candidate.get("generation_config"),
                "metrics": candidate.get("metrics", {}),
                "score": candidate.get("preference_score"),
                "hard_invalid": candidate.get("hard_invalid"),
                "model_output_text": candidate.get("model_output_text"),
                "reconstructed_yaml": candidate.get("reconstructed_yaml"),
                "parser_errors": candidate.get("parser_errors", []),
                "reconstruction_errors": candidate.get("reconstruction_errors", []),
            }
            for candidate in unit.get("candidates", [])
        ],
    }


def get_labeling_guide() -> dict[str, Any]:
    return LABELING_GUIDE


def append_human_preference(
    *,
    paths: AnnotationPaths,
    unit: dict[str, Any],
    payload: dict[str, Any],
    annotator: str = "human",
) -> dict[str, Any]:
    event = build_preference_event(
        unit=unit,
        payload=payload,
        label_source="human",
        review_status="approved",
        annotator=annotator,
    )
    append_jsonl(paths.human_preferences, [event])
    return event


def append_agent_suggestion(
    *,
    paths: AnnotationPaths,
    unit: dict[str, Any],
    payload: dict[str, Any],
    agent_name: str = "codex",
) -> dict[str, Any]:
    event = build_preference_event(
        unit=unit,
        payload=payload,
        label_source="agent",
        review_status=str(payload.get("review_status") or "pending"),
        annotator=agent_name,
    )
    append_jsonl(paths.agent_suggestions, [event])
    return event


def build_preference_event(
    *,
    unit: dict[str, Any],
    payload: dict[str, Any],
    label_source: str,
    review_status: str,
    annotator: str,
) -> dict[str, Any]:
    candidate_by_key = {candidate["candidate_key"]: candidate for candidate in unit.get("candidates", [])}
    decision = str(payload.get("decision") or DECISION_PREFERENCE)
    confidence = str(payload.get("confidence") or "medium")
    review_status = str(review_status or "pending")
    chosen_key = _optional_string(payload.get("chosen_candidate_key"))
    rejected_key = _optional_string(payload.get("rejected_candidate_key"))

    if decision not in VALID_DECISIONS:
        raise PreferenceAnnotationError(f"invalid_decision:{decision}")
    if confidence not in VALID_CONFIDENCE:
        raise PreferenceAnnotationError(f"invalid_confidence:{confidence}")
    if review_status not in VALID_REVIEW_STATUS:
        raise PreferenceAnnotationError(f"invalid_review_status:{review_status}")
    if decision == DECISION_PREFERENCE:
        if not chosen_key or not rejected_key:
            raise PreferenceAnnotationError("preference_requires_chosen_and_rejected")
        if chosen_key == rejected_key:
            raise PreferenceAnnotationError("chosen_and_rejected_must_differ")
        missing = [key for key in (chosen_key, rejected_key) if key not in candidate_by_key]
        if missing:
            raise PreferenceAnnotationError(f"unknown_candidate_keys:{missing}")

    created_at = utc_now_iso()
    event = {
        "annotation_id": str(payload.get("annotation_id") or uuid.uuid4()),
        "unit_id": unit["unit_id"],
        "sample_id": unit["sample_id"],
        "prompt_variant": unit["prompt_variant"],
        "split": unit["split"],
        "decision": decision,
        "chosen_candidate_key": chosen_key,
        "rejected_candidate_key": rejected_key,
        "confidence": confidence,
        "rationale": str(payload.get("rationale") or ""),
        "metric_flags": _string_list(payload.get("metric_flags")),
        "label_source": label_source,
        "review_status": review_status,
        "annotator": annotator,
        "created_at": created_at,
        "candidate_snapshots": {},
    }
    event["labeling_guide_version"] = LABELING_GUIDE_VERSION
    event["pair_type"] = _optional_string(payload.get("pair_type"))
    event["score_margin"] = _optional_number(payload.get("score_margin"))
    event["agent_policy_version"] = _optional_string(payload.get("agent_policy_version"))
    event["source_annotation_id"] = _optional_string(payload.get("source_annotation_id"))
    if chosen_key:
        event["candidate_snapshots"]["chosen"] = _candidate_snapshot(candidate_by_key[chosen_key])
    if rejected_key:
        event["candidate_snapshots"]["rejected"] = _candidate_snapshot(candidate_by_key[rejected_key])
    return event


def load_annotation_events(paths: AnnotationPaths) -> list[dict[str, Any]]:
    return [
        *read_jsonl_if_exists(paths.agent_suggestions, allow_truncated_last_line=True),
        *read_jsonl_if_exists(paths.human_preferences, allow_truncated_last_line=True),
    ]


def latest_decisions_by_unit(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=lambda row: str(row.get("created_at") or "")):
        unit_id = str(event.get("unit_id") or "")
        if not unit_id or event.get("review_status") != "approved":
            continue
        existing = latest.get(unit_id)
        if existing is None or _event_priority(event) >= _event_priority(existing):
            latest[unit_id] = event
    return latest


def approved_preference_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_pair: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in sorted(events, key=lambda row: str(row.get("created_at") or "")):
        if event.get("review_status") != "approved" or event.get("decision") != DECISION_PREFERENCE:
            continue
        unit_id = str(event.get("unit_id") or "")
        chosen_key = str(event.get("chosen_candidate_key") or "")
        rejected_key = str(event.get("rejected_candidate_key") or "")
        if not unit_id or not chosen_key or not rejected_key:
            continue
        pair_key = (unit_id, chosen_key, rejected_key)
        existing = best_by_pair.get(pair_key)
        if existing is None or _event_priority(event) >= _event_priority(existing):
            best_by_pair[pair_key] = event
    return sorted(
        best_by_pair.values(),
        key=lambda row: (
            str(row.get("unit_id") or ""),
            -_event_priority(row),
            str(row.get("created_at") or ""),
            str(row.get("annotation_id") or ""),
        ),
    )


def export_final_preferences(
    *,
    paths: AnnotationPaths,
    units_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    for event in approved_preference_events(load_annotation_events(paths)):
        unit_id = str(event.get("unit_id") or "")
        unit = units_by_id.get(unit_id)
        if unit is None:
            continue
        candidate_by_key = {candidate["candidate_key"]: candidate for candidate in unit.get("candidates", [])}
        chosen = candidate_by_key.get(str(event.get("chosen_candidate_key")))
        rejected = candidate_by_key.get(str(event.get("rejected_candidate_key")))
        if chosen is None or rejected is None:
            continue
        final_rows.append(_final_preference_row(unit, event, chosen, rejected))
    write_jsonl(paths.final_preferences, final_rows)
    return final_rows


def write_annotation_state(
    *,
    paths: AnnotationPaths,
    units: Iterable[dict[str, Any]],
    final_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    unit_list = list(units)
    events = load_annotation_events(paths)
    latest = latest_decisions_by_unit(events)
    final_count = len(final_rows) if final_rows is not None else len(approved_preference_events(events))
    state = {
        "status": "running",
        "updated_at": utc_now_iso(),
        "total_units": len(unit_list),
        "candidate_count": sum(len(unit.get("candidates", [])) for unit in unit_list),
        "event_count": len(events),
        "approved_unit_count": len(latest),
        "approved_pair_count": final_count,
        "tie_count": sum(1 for row in latest.values() if row.get("decision") == DECISION_TIE),
        "skip_count": sum(1 for row in latest.values() if row.get("decision") == DECISION_SKIP),
        "artifacts": {
            "human_preferences": HUMAN_PREFERENCES_ARTIFACT,
            "agent_suggestions": AGENT_SUGGESTIONS_ARTIFACT,
            "final_preferences": FINAL_PREFERENCES_ARTIFACT,
        },
    }
    write_json(paths.state, state)
    return state


def _build_unit_base(
    *,
    unit_id: str,
    candidate: dict[str, Any],
    dataset_row: dict[str, Any],
    requirement_row: dict[str, Any],
) -> dict[str, Any]:
    prompt_text = dataset_row.get("prompt_text") or candidate.get("prompt")
    prompt_requirements = requirement_row.get("prompt_requirements")
    if prompt_requirements is None and isinstance(prompt_text, str):
        prompt_requirements = [atom.to_dict() for atom in extract_prompt_requirements(prompt_text)]
    return {
        "unit_id": unit_id,
        "sample_id": str(candidate.get("sample_id")),
        "prompt_variant": str(candidate.get("prompt_variant")),
        "split": str(candidate.get("split")),
        "prompt_text": prompt_text,
        "sft_prompt": candidate.get("prompt"),
        "reference_yaml": dataset_row.get("target_yaml_normalized") or candidate.get("reference_yaml"),
        "prompt_requirement_supported": requirement_row.get("prompt_requirement_supported"),
        "prompt_requirements": prompt_requirements or [],
        "reference_prompt_requirement_evaluation": requirement_row.get("reference_prompt_requirement_evaluation"),
        "reference_required_field_evaluation": requirement_row.get("reference_required_field_evaluation"),
        "candidates": [],
    }


def _build_candidate_payload(
    candidate: dict[str, Any],
    metric_row: dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    candidate_uid = str(candidate.get("candidate_uid"))
    candidate_key = f"{run_id}::{candidate_uid}"
    evaluation = metric_row.get("evaluation") if isinstance(metric_row.get("evaluation"), dict) else {}
    metrics = {field: _first_present(metric_row, evaluation, field) for field in METRIC_SUMMARY_FIELDS}
    metrics["preference_score"] = metric_row.get("preference_score")
    metrics["hard_invalid"] = metric_row.get("hard_invalid")
    return {
        "candidate_key": candidate_key,
        "candidate_uid": candidate_uid,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_index": candidate.get("candidate_index"),
        "source_run_id": run_id,
        "source_run_dir": str(run_dir),
        "checkpoint": candidate.get("checkpoint") or metric_row.get("checkpoint"),
        "generation_ok": candidate.get("generation_ok"),
        "structured_output_parse_success": candidate.get("structured_output_parse_success"),
        "generation_config": candidate.get("generation_config"),
        "model_output_text": candidate.get("model_output_text"),
        "reconstructed_yaml": candidate.get("reconstructed_yaml"),
        "reference_yaml": candidate.get("reference_yaml"),
        "predicted_blocks": candidate.get("predicted_blocks") or [],
        "parser_errors": candidate.get("parser_errors") or [],
        "reconstruction_errors": candidate.get("reconstruction_errors") or [],
        "preference_score": metric_row.get("preference_score"),
        "hard_invalid": metric_row.get("hard_invalid"),
        "components": metric_row.get("components") or {},
        "weighted_components": metric_row.get("weighted_components") or {},
        "penalties": metric_row.get("penalties") or {},
        "formula": metric_row.get("formula"),
        "metrics": metrics,
        "evaluation": evaluation,
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
    hard_invalid = bool(candidate.get("hard_invalid"))
    score = candidate.get("preference_score")
    numeric_score = float(score) if isinstance(score, (int, float)) else -1.0
    return (1 if hard_invalid else 0, -numeric_score, str(candidate.get("candidate_key")))


def _unit_sort_key(unit: dict[str, Any]) -> tuple[int, str, str]:
    sample_id = str(unit.get("sample_id") or "")
    numeric = int(sample_id[1:]) if sample_id.startswith("q") and sample_id[1:].isdigit() else 10**9
    return (numeric, sample_id, str(unit.get("prompt_variant") or ""))


def _load_candidate_run_id(run_dir: Path) -> str:
    config = load_json_if_exists(run_dir / "config.json")
    run_id = config.get("run_id")
    return str(run_id or run_dir.name)


def _first_present(primary: dict[str, Any], secondary: dict[str, Any], key: str) -> Any:
    if key in primary:
        return primary.get(key)
    return secondary.get(key)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise PreferenceAnnotationError("metric_flags_must_be_list_or_comma_string")


def _candidate_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_key": candidate.get("candidate_key"),
        "candidate_uid": candidate.get("candidate_uid"),
        "candidate_id": candidate.get("candidate_id"),
        "source_run_id": candidate.get("source_run_id"),
        "generation_config": candidate.get("generation_config"),
        "metrics": candidate.get("metrics"),
        "preference_score": candidate.get("preference_score"),
        "hard_invalid": candidate.get("hard_invalid"),
    }


def _event_priority(event: dict[str, Any]) -> int:
    source = event.get("label_source")
    if source == "human":
        return 30
    if source == "agent":
        return 20
    if source == "automatic":
        return 10
    return 0


def _final_preference_row(
    unit: dict[str, Any],
    event: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "preference_id": f"{unit['unit_id']}::{event['annotation_id']}",
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
        "annotation_id": event["annotation_id"],
        "label_source": event.get("label_source"),
        "labeling_guide_version": event.get("labeling_guide_version"),
        "pair_type": event.get("pair_type"),
        "score_margin": event.get("score_margin"),
        "agent_policy_version": event.get("agent_policy_version"),
        "source_annotation_id": event.get("source_annotation_id"),
        "confidence": event.get("confidence"),
        "rationale": event.get("rationale"),
        "metric_flags": event.get("metric_flags", []),
        "created_at": event.get("created_at"),
    }
