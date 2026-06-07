from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .dataset_io import append_jsonl, read_jsonl, write_json, write_jsonl
from .resumable_run import utc_now_iso


PROMPT_REQUIREMENT_GOLD_ARTIFACT = "prompt_requirement_gold.jsonl"
PROMPT_REQUIREMENT_AUDIT_REPORT_ARTIFACT = "prompt_requirement_audit_report.json"


def select_prompt_requirement_audit_cases(
    *,
    prompt_requirement_rows: Iterable[dict[str, Any]],
    dataset_index: dict[tuple[str, str], dict[str, Any]],
    split: str = "train",
    sample_size: int = 40,
    seed: int = 13,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in prompt_requirement_rows
        if (not split or row.get("split") == split)
    ]
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {
        "unsupported": [],
        "low_reference_f1": [],
        "exact_but_thin": [],
        "multi_resource": [],
        "random": [],
    }

    for row in rows:
        evaluation = row.get("reference_prompt_requirement_evaluation") or {}
        f1 = evaluation.get("prompt_requirement_f1")
        requirements = row.get("prompt_requirements") or []
        kind_count = sum(1 for atom in requirements if atom.get("category") == "kind")
        literal_score = _literal_score(str(row.get("prompt_text") or ""))

        if not row.get("prompt_requirement_supported"):
            buckets["unsupported"].append(row)
        if isinstance(f1, (int, float)) and f1 < 0.5:
            buckets["low_reference_f1"].append(row)
        if evaluation.get("prompt_requirement_exact_match") and len(requirements) <= 1 and literal_score >= 8:
            buckets["exact_but_thin"].append(row)
        if kind_count > 1:
            buckets["multi_resource"].append(row)
        buckets["random"].append(row)

    target_by_bucket = _bucket_targets(sample_size)
    selected: dict[str, dict[str, Any]] = {}
    selected_bucket: dict[str, str] = {}
    for bucket_name, target_count in target_by_bucket.items():
        bucket_rows = list(buckets[bucket_name])
        rng.shuffle(bucket_rows)
        for row in bucket_rows:
            if len(selected) >= sample_size:
                break
            case_id = _case_id(row)
            if case_id in selected:
                continue
            selected[case_id] = row
            selected_bucket[case_id] = bucket_name
            target_count -= 1
            if target_count <= 0:
                break

    if len(selected) < sample_size:
        remaining = [row for row in rows if _case_id(row) not in selected]
        rng.shuffle(remaining)
        for row in remaining[: sample_size - len(selected)]:
            case_id = _case_id(row)
            selected[case_id] = row
            selected_bucket[case_id] = "random"

    return [
        _audit_case_payload(row, selected_bucket[_case_id(row)], dataset_index)
        for row in sorted(selected.values(), key=_row_sort_key)
    ]


def seed_prompt_requirement_gold_file(
    *,
    output_dir: Path,
    audit_cases: list[dict[str, Any]],
    overwrite: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROMPT_REQUIREMENT_GOLD_ARTIFACT
    if path.exists() and not overwrite:
        existing = read_jsonl(path, allow_truncated_last_line=True)
        existing_case_ids = {row.get("audit_case_id") for row in existing}
        new_rows = [row for row in audit_cases if row.get("audit_case_id") not in existing_case_ids]
        if new_rows:
            append_jsonl(path, new_rows)
    else:
        write_jsonl(path, audit_cases)
    return path


def append_prompt_requirement_gold_annotation(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    annotator: str = "human",
) -> dict[str, Any]:
    case_id = str(payload.get("audit_case_id") or "")
    if not case_id:
        raise ValueError("audit_case_id is required")
    gold_requirements = payload.get("gold_requirements")
    if not isinstance(gold_requirements, list):
        raise ValueError("gold_requirements must be a list")
    status = str(payload.get("status") or "reviewed")
    if status == "pending" and gold_requirements:
        status = "reviewed"
    row = {
        **payload,
        "audit_case_id": case_id,
        "status": status,
        "annotator": annotator,
        "updated_at": utc_now_iso(),
    }
    append_jsonl(output_dir / PROMPT_REQUIREMENT_GOLD_ARTIFACT, [row])
    return row


def load_latest_gold_cases(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return []
    for row in read_jsonl(path, allow_truncated_last_line=True):
        case_id = str(row.get("audit_case_id") or "")
        if case_id:
            latest[case_id] = row
    return [latest[key] for key in sorted(latest)]


def compute_prompt_requirement_audit_report(gold_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(gold_rows)
    reviewed = [
        row
        for row in rows
        if row.get("status") == "reviewed" and isinstance(row.get("gold_requirements"), list)
    ]
    per_case: list[dict[str, Any]] = []
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    aggregate = Counter()

    for row in reviewed:
        extracted_atoms = row.get("extracted_prompt_requirements") or row.get("prompt_requirements") or []
        gold_atoms = row.get("gold_requirements") or []
        extracted = {_canonical(atom) for atom in extracted_atoms if _canonical(atom)}
        gold = {_canonical(atom) for atom in gold_atoms if _canonical(atom)}
        matched = extracted & gold
        metrics = _prf(gold, extracted)
        per_case.append(
            {
                "audit_case_id": row.get("audit_case_id"),
                "sample_id": row.get("sample_id"),
                "prompt_variant": row.get("prompt_variant"),
                "audit_bucket": row.get("audit_bucket"),
                "gold_count": len(gold),
                "extracted_count": len(extracted),
                "matched_count": len(matched),
                **metrics,
            }
        )
        aggregate["gold"] += len(gold)
        aggregate["extracted"] += len(extracted)
        aggregate["matched"] += len(matched)
        for category in _categories(gold_atoms, extracted_atoms):
            gold_for_category = {
                _canonical(atom) for atom in gold_atoms if _atom_category(atom) == category and _canonical(atom)
            }
            extracted_for_category = {
                _canonical(atom) for atom in extracted_atoms if _atom_category(atom) == category and _canonical(atom)
            }
            matched_for_category = gold_for_category & extracted_for_category
            category_stats[category]["gold"] += len(gold_for_category)
            category_stats[category]["extracted"] += len(extracted_for_category)
            category_stats[category]["matched"] += len(matched_for_category)

    by_category = {
        category: {
            **_prf_from_counts(
                gold_count=counts["gold"],
                extracted_count=counts["extracted"],
                matched_count=counts["matched"],
            ),
            "gold_count": counts["gold"],
            "extracted_count": counts["extracted"],
            "matched_count": counts["matched"],
        }
        for category, counts in sorted(category_stats.items())
    }
    return {
        "generated_at": utc_now_iso(),
        "case_count": len(rows),
        "reviewed_case_count": len(reviewed),
        "pending_case_count": len(rows) - len(reviewed),
        "overall": {
            **_prf_from_counts(
                gold_count=aggregate["gold"],
                extracted_count=aggregate["extracted"],
                matched_count=aggregate["matched"],
            ),
            "gold_count": aggregate["gold"],
            "extracted_count": aggregate["extracted"],
            "matched_count": aggregate["matched"],
        },
        "by_category": by_category,
        "per_case": per_case,
        "known_limitations": [
            "Regex extraction covers only documented atom categories.",
            "Multi-resource prompts can match atoms in the wrong resource context.",
            "Unsupported or missed categories do not become comparable prompt atoms.",
            "A prompt can score well if the extractor captured only an easy atom such as kind.",
        ],
    }


def write_prompt_requirement_audit_report(output_dir: Path) -> dict[str, Any]:
    gold_path = output_dir / PROMPT_REQUIREMENT_GOLD_ARTIFACT
    report = compute_prompt_requirement_audit_report(load_latest_gold_cases(gold_path))
    write_json(output_dir / PROMPT_REQUIREMENT_AUDIT_REPORT_ARTIFACT, report)
    return report


def _bucket_targets(sample_size: int) -> dict[str, int]:
    if sample_size <= 0:
        return {key: 0 for key in ("unsupported", "low_reference_f1", "exact_but_thin", "multi_resource", "random")}
    return {
        "unsupported": max(2, round(sample_size * 0.15)),
        "low_reference_f1": max(4, round(sample_size * 0.30)),
        "exact_but_thin": max(4, round(sample_size * 0.25)),
        "multi_resource": max(3, round(sample_size * 0.15)),
        "random": max(4, round(sample_size * 0.15)),
    }


def _audit_case_payload(
    row: dict[str, Any],
    bucket: str,
    dataset_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (str(row.get("sample_id")), str(row.get("prompt_variant")))
    dataset_row = dataset_index.get(key, {})
    return {
        "audit_case_id": _case_id(row),
        "audit_bucket": bucket,
        "status": "pending",
        "sample_id": row.get("sample_id"),
        "prompt_variant": row.get("prompt_variant"),
        "split": row.get("split"),
        "prompt_text": row.get("prompt_text"),
        "target_yaml_normalized": dataset_row.get("target_yaml_normalized"),
        "extracted_prompt_requirements": row.get("prompt_requirements") or [],
        "gold_requirements": [],
        "reference_prompt_requirement_evaluation": row.get("reference_prompt_requirement_evaluation"),
        "notes": "",
        "created_at": utc_now_iso(),
    }


def _case_id(row: dict[str, Any]) -> str:
    return f"{row.get('sample_id')}::{row.get('prompt_variant')}"


def _row_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    sample_id = str(row.get("sample_id") or "")
    numeric = int(sample_id[1:]) if sample_id.startswith("q") and sample_id[1:].isdigit() else 10**9
    return (numeric, sample_id, str(row.get("prompt_variant") or ""))


def _literal_score(prompt_text: str) -> int:
    return prompt_text.count(":") + prompt_text.count("=") + prompt_text.count('"') + prompt_text.count("'")


def _canonical(atom: dict[str, Any]) -> str:
    canonical = atom.get("canonical")
    if isinstance(canonical, str) and canonical:
        return canonical
    category = atom.get("category")
    value = atom.get("value")
    key = atom.get("key")
    if not category or value is None:
        return ""
    if key is not None:
        return f"{category}:{key}={value}"
    return f"{category}={value}"


def _atom_category(atom: dict[str, Any]) -> str:
    category = atom.get("category")
    if isinstance(category, str):
        return category
    canonical = _canonical(atom)
    if ":" in canonical and "=" in canonical:
        return canonical.split(":", 1)[0]
    if "=" in canonical:
        return canonical.split("=", 1)[0]
    return "unknown"


def _categories(*atom_groups: list[dict[str, Any]]) -> set[str]:
    return {
        _atom_category(atom)
        for atoms in atom_groups
        for atom in atoms
        if _canonical(atom)
    }


def _prf(gold: set[str], extracted: set[str]) -> dict[str, float]:
    return _prf_from_counts(
        gold_count=len(gold),
        extracted_count=len(extracted),
        matched_count=len(gold & extracted),
    )


def _prf_from_counts(*, gold_count: int, extracted_count: int, matched_count: int) -> dict[str, float]:
    precision = matched_count / extracted_count if extracted_count else 0.0
    recall = matched_count / gold_count if gold_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision or recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }
