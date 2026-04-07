from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DOMAIN = "Kubernetes"
PROMPT_POLICY = "retain_both_variants_shared_split"
DATASET_VERSION = "kubernetes_v1"
TARGET_FILE_NAME = "labeled_code.yaml"
PROMPT_FILES = {
    "question": "question.txt",
    "question_simplified": "question_simplified.txt",
}
SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}
DEFAULT_SEED = 42


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt_text(text: str) -> str:
    return " ".join(text.lower().split())


def prompt_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.lower())


def safe_read_text(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except UnicodeDecodeError as exc:
        return "", f"utf8_decode_error:{exc.reason}"


def yaml_depth(value: Any) -> int:
    if isinstance(value, dict):
        if not value:
            return 1
        return 1 + max(yaml_depth(child) for child in value.values())
    if isinstance(value, list):
        if not value:
            return 1
        return 1 + max(yaml_depth(child) for child in value)
    return 1


def yaml_node_counts(value: Any) -> dict[str, int]:
    counts = {
        "mapping_nodes": 0,
        "list_nodes": 0,
        "scalar_nodes": 0,
    }

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            counts["mapping_nodes"] += 1
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            counts["list_nodes"] += 1
            for child in node:
                walk(child)
        else:
            counts["scalar_nodes"] += 1

    walk(value)
    counts["total_nodes"] = sum(counts.values())
    return counts


def canonicalize_yaml(value: Any) -> str:
    text = yaml.safe_dump(
        value,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
        indent=2,
        width=4096,
    )
    return text if text.endswith("\n") else f"{text}\n"


def parse_yaml_documents(text: str) -> list[Any]:
    return list(yaml.safe_load_all(text))


def canonicalize_yaml_documents(documents: list[Any]) -> str:
    rendered_documents = [canonicalize_yaml(document).rstrip("\n") for document in documents]
    if not rendered_documents:
        return ""
    return "\n---\n".join(rendered_documents) + "\n"


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


@dataclass
class SampleRecord:
    sample_id: str
    domain: str
    prompt_policy: str
    question_path: str
    question_simplified_path: str
    target_path: str
    target_logical_name: str
    question_text: str
    question_simplified_text: str
    target_yaml_raw: str
    target_yaml_normalized: str
    question_char_count: int
    question_simplified_char_count: int
    target_yaml_char_count: int
    question_word_count: int
    question_simplified_word_count: int
    prompt_pair_similarity: float
    yaml_parse_ok: bool
    yaml_document_count: int
    yaml_top_level_key_count: int
    yaml_max_depth: int
    yaml_mapping_nodes: int
    yaml_list_nodes: int
    yaml_scalar_nodes: int
    yaml_total_nodes: int
    normalization_semantics_preserved: bool
    validation_status: str
    validation_notes: str
    yaml_hash: str
    duplicate_yaml_group_size: int = 1
    duplicate_prompt_group_size: int = 1
    near_duplicate_group_size: int = 1
    leakage_group: str = ""
    leakage_reasons: str = ""
    split: str = ""

    @property
    def complexity_score(self) -> float:
        return float(
            self.yaml_total_nodes
            + self.yaml_max_depth * 5
            + max(self.question_word_count, self.question_simplified_word_count)
        )


@dataclass
class PromptVariantRecord:
    sample_id: str
    domain: str
    prompt_variant: str
    prompt_path: str
    prompt_text: str
    prompt_text_normalized: str
    prompt_hash: str
    prompt_char_count: int
    prompt_word_count: int
    target_path: str
    target_logical_name: str
    target_yaml_raw: str
    target_yaml_normalized: str
    yaml_parse_ok: bool
    prompt_policy: str
    validation_status: str
    validation_notes: str
    leakage_group: str = ""
    leakage_reasons: str = ""
    split: str = ""


def sorted_sample_ids(question_root: Path, target_root: Path) -> list[str]:
    question_ids = {path.name for path in question_root.iterdir() if path.is_dir()}
    target_ids = {path.name for path in target_root.iterdir() if path.is_dir()}
    missing_question = sorted(target_ids - question_ids)
    missing_target = sorted(question_ids - target_ids)
    if missing_question or missing_target:
        raise ValueError(
            "Sample ID mismatch between prompt and target directories: "
            f"missing_question={missing_question[:5]}, missing_target={missing_target[:5]}"
        )
    return sorted(question_ids, key=lambda item: int(item.removeprefix("q")))


def build_records(repo_root: Path) -> tuple[list[SampleRecord], list[PromptVariantRecord]]:
    question_root = repo_root / "data" / "question" / DOMAIN
    target_root = repo_root / "data" / "docker_compose" / DOMAIN
    sample_ids = sorted_sample_ids(question_root, target_root)

    sample_records: list[SampleRecord] = []
    prompt_records: list[PromptVariantRecord] = []

    for sample_id in sample_ids:
        prompt_dir = question_root / sample_id
        target_dir = target_root / sample_id

        notes: list[str] = []
        reject_reasons: list[str] = []

        question_path = prompt_dir / PROMPT_FILES["question"]
        question_simplified_path = prompt_dir / PROMPT_FILES["question_simplified"]
        target_path = target_dir / TARGET_FILE_NAME

        for expected_path in (question_path, question_simplified_path, target_path):
            if not expected_path.exists():
                reject_reasons.append(f"missing_file:{expected_path.name}")

        question_text, question_error = safe_read_text(question_path) if question_path.exists() else ("", None)
        simplified_text, simplified_error = (
            safe_read_text(question_simplified_path) if question_simplified_path.exists() else ("", None)
        )
        target_yaml_raw, target_error = safe_read_text(target_path) if target_path.exists() else ("", None)

        for error in (question_error, simplified_error, target_error):
            if error:
                reject_reasons.append(error)

        if not question_text.strip():
            reject_reasons.append("question_empty")
        if not simplified_text.strip():
            reject_reasons.append("question_simplified_empty")
        if not target_yaml_raw.strip():
            reject_reasons.append("target_yaml_empty")

        yaml_parse_ok = False
        yaml_document_count = 0
        yaml_top_level_key_count = 0
        yaml_max_depth = 0
        yaml_counts = {
            "mapping_nodes": 0,
            "list_nodes": 0,
            "scalar_nodes": 0,
            "total_nodes": 0,
        }
        target_yaml_normalized = ""
        normalization_semantics_preserved = False

        if not reject_reasons:
            try:
                yaml_documents = parse_yaml_documents(target_yaml_raw)
                yaml_parse_ok = True
                yaml_document_count = len(yaml_documents)
                yaml_top_level_key_count = sum(
                    len(document) if isinstance(document, dict) else 0 for document in yaml_documents
                )
                yaml_max_depth = max((yaml_depth(document) for document in yaml_documents), default=0)
                yaml_counts = {
                    "mapping_nodes": 0,
                    "list_nodes": 0,
                    "scalar_nodes": 0,
                    "total_nodes": 0,
                }
                for document in yaml_documents:
                    document_counts = yaml_node_counts(document)
                    for key, value in document_counts.items():
                        yaml_counts[key] += value
                target_yaml_normalized = canonicalize_yaml_documents(yaml_documents)
                normalization_semantics_preserved = parse_yaml_documents(target_yaml_normalized) == yaml_documents
                if not normalization_semantics_preserved:
                    reject_reasons.append("normalization_semantics_changed")
                if canonicalize_yaml_documents(parse_yaml_documents(target_yaml_normalized)) != target_yaml_normalized:
                    reject_reasons.append("normalization_not_deterministic")
            except yaml.YAMLError as exc:
                reject_reasons.append(f"yaml_parse_error:{exc.__class__.__name__}")

        validation_status = "ok"
        if reject_reasons:
            validation_status = "reject"
            notes.extend(reject_reasons)
        elif notes:
            validation_status = "warning"

        pair_similarity = difflib.SequenceMatcher(
            None,
            normalize_prompt_text(question_text),
            normalize_prompt_text(simplified_text),
        ).ratio()

        sample_record = SampleRecord(
            sample_id=sample_id,
            domain=DOMAIN,
            prompt_policy=PROMPT_POLICY,
            question_path=str(question_path.relative_to(repo_root)).replace("\\", "/"),
            question_simplified_path=str(question_simplified_path.relative_to(repo_root)).replace("\\", "/"),
            target_path=str(target_path.relative_to(repo_root)).replace("\\", "/"),
            target_logical_name="target_yaml",
            question_text=question_text,
            question_simplified_text=simplified_text,
            target_yaml_raw=target_yaml_raw,
            target_yaml_normalized=target_yaml_normalized,
            question_char_count=len(question_text),
            question_simplified_char_count=len(simplified_text),
            target_yaml_char_count=len(target_yaml_raw),
            question_word_count=len(prompt_tokens(question_text)),
            question_simplified_word_count=len(prompt_tokens(simplified_text)),
            prompt_pair_similarity=round(pair_similarity, 6),
            yaml_parse_ok=yaml_parse_ok,
            yaml_document_count=yaml_document_count,
            yaml_top_level_key_count=yaml_top_level_key_count,
            yaml_max_depth=yaml_max_depth,
            yaml_mapping_nodes=yaml_counts["mapping_nodes"],
            yaml_list_nodes=yaml_counts["list_nodes"],
            yaml_scalar_nodes=yaml_counts["scalar_nodes"],
            yaml_total_nodes=yaml_counts["total_nodes"],
            normalization_semantics_preserved=normalization_semantics_preserved,
            validation_status=validation_status,
            validation_notes=";".join(notes),
            yaml_hash=sha256_text(target_yaml_normalized) if target_yaml_normalized else "",
        )
        sample_records.append(sample_record)

        for variant_name, variant_text, variant_path in (
            ("question", question_text, question_path),
            ("question_simplified", simplified_text, question_simplified_path),
        ):
            normalized_prompt = normalize_prompt_text(variant_text)
            prompt_records.append(
                PromptVariantRecord(
                    sample_id=sample_id,
                    domain=DOMAIN,
                    prompt_variant=variant_name,
                    prompt_path=str(variant_path.relative_to(repo_root)).replace("\\", "/"),
                    prompt_text=variant_text,
                    prompt_text_normalized=normalized_prompt,
                    prompt_hash=sha256_text(normalized_prompt),
                    prompt_char_count=len(variant_text),
                    prompt_word_count=len(prompt_tokens(variant_text)),
                    target_path=sample_record.target_path,
                    target_logical_name="target_yaml",
                    target_yaml_raw=target_yaml_raw,
                    target_yaml_normalized=target_yaml_normalized,
                    yaml_parse_ok=yaml_parse_ok,
                    prompt_policy=PROMPT_POLICY,
                    validation_status=validation_status,
                    validation_notes=sample_record.validation_notes,
                )
            )

    return sample_records, prompt_records


def annotate_duplicates(
    sample_records: list[SampleRecord],
    prompt_records: list[PromptVariantRecord],
) -> None:
    sample_map = {record.sample_id: record for record in sample_records}
    union_find = UnionFind([record.sample_id for record in sample_records])
    leakage_reasons: dict[str, set[str]] = defaultdict(set)

    yaml_groups: dict[str, list[str]] = defaultdict(list)
    for record in sample_records:
        if record.yaml_hash:
            yaml_groups[record.yaml_hash].append(record.sample_id)
    for members in yaml_groups.values():
        if len(members) < 2:
            continue
        for sample_id in members:
            sample_map[sample_id].duplicate_yaml_group_size = len(members)
            leakage_reasons[sample_id].add("exact_yaml_duplicate")
        base = members[0]
        for member in members[1:]:
            union_find.union(base, member)

    prompt_groups: dict[str, set[str]] = defaultdict(set)
    for record in prompt_records:
        if record.prompt_text_normalized:
            prompt_groups[record.prompt_hash].add(record.sample_id)
    for members in prompt_groups.values():
        if len(members) < 2:
            continue
        sorted_members = sorted(members, key=lambda item: int(item.removeprefix("q")))
        for sample_id in sorted_members:
            sample_map[sample_id].duplicate_prompt_group_size = len(sorted_members)
            leakage_reasons[sample_id].add("exact_prompt_duplicate")
        base = sorted_members[0]
        for member in sorted_members[1:]:
            union_find.union(base, member)

    near_duplicate_sizes: dict[str, set[str]] = defaultdict(set)
    ordered_samples = sorted(sample_records, key=lambda item: int(item.sample_id.removeprefix("q")))
    sample_prompt_lookup: dict[str, list[str]] = defaultdict(list)
    for record in prompt_records:
        sample_prompt_lookup[record.sample_id].append(record.prompt_text_normalized)

    for index, left in enumerate(ordered_samples):
        left_prompts = sample_prompt_lookup[left.sample_id]
        for right in ordered_samples[index + 1 :]:
            right_prompts = sample_prompt_lookup[right.sample_id]
            best_ratio = 0.0
            for left_prompt in left_prompts:
                for right_prompt in right_prompts:
                    if not left_prompt or not right_prompt:
                        continue
                    length_ratio = min(len(left_prompt), len(right_prompt)) / max(len(left_prompt), len(right_prompt))
                    if length_ratio < 0.85:
                        continue
                    best_ratio = max(best_ratio, difflib.SequenceMatcher(None, left_prompt, right_prompt).ratio())
            if best_ratio >= 0.97:
                union_find.union(left.sample_id, right.sample_id)
                near_duplicate_sizes[left.sample_id].add(right.sample_id)
                near_duplicate_sizes[right.sample_id].add(left.sample_id)
                leakage_reasons[left.sample_id].add("near_prompt_duplicate")
                leakage_reasons[right.sample_id].add("near_prompt_duplicate")

    for sample_id, neighbors in near_duplicate_sizes.items():
        sample_map[sample_id].near_duplicate_group_size = len(neighbors) + 1

    grouped_members: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_map:
        grouped_members[union_find.find(sample_id)].append(sample_id)

    group_names: dict[str, str] = {}
    for index, root in enumerate(sorted(grouped_members, key=lambda item: int(item.removeprefix("q"))), start=1):
        group_names[root] = f"group_{index:03d}"

    for sample_id, record in sample_map.items():
        root = union_find.find(sample_id)
        record.leakage_group = group_names[root]
        record.leakage_reasons = ",".join(sorted(leakage_reasons.get(sample_id, set()))) or "sample_id_only"

    for record in prompt_records:
        sample_record = sample_map[record.sample_id]
        record.leakage_group = sample_record.leakage_group
        record.leakage_reasons = sample_record.leakage_reasons


def integer_targets(total_items: int, ratios: dict[str, float]) -> dict[str, int]:
    targets: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for split_name, ratio in ratios.items():
        exact = total_items * ratio
        whole = math.floor(exact)
        targets[split_name] = whole
        assigned += whole
        remainders.append((exact - whole, split_name))

    for _, split_name in sorted(remainders, reverse=True)[: total_items - assigned]:
        targets[split_name] += 1
    return targets


def assign_splits(sample_records: list[SampleRecord], prompt_records: list[PromptVariantRecord], seed: int) -> None:
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in sample_records:
        groups[record.leakage_group].append(record)

    total_samples = len(sample_records)
    target_sample_counts = integer_targets(total_samples, SPLIT_RATIOS)
    total_complexity = sum(record.complexity_score for record in sample_records)
    target_complexity = {
        split_name: total_complexity * ratio for split_name, ratio in SPLIT_RATIOS.items()
    }

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(
        key=lambda item: (
            -len(item[1]),
            -sum(member.complexity_score for member in item[1]),
            int(min(member.sample_id.removeprefix("q") for member in item[1])),
        )
    )

    current_sample_counts = {split_name: 0 for split_name in SPLIT_RATIOS}
    current_complexity = {split_name: 0.0 for split_name in SPLIT_RATIOS}
    group_split_assignment: dict[str, str] = {}

    for group_name, members in group_items:
        group_sample_count = len(members)
        group_complexity = sum(member.complexity_score for member in members)

        best_split = None
        best_score = None
        for split_name in SPLIT_RATIOS:
            next_count = current_sample_counts[split_name] + group_sample_count
            next_complexity = current_complexity[split_name] + group_complexity
            count_error = abs(next_count - target_sample_counts[split_name]) / max(target_sample_counts[split_name], 1)
            complexity_error = abs(next_complexity - target_complexity[split_name]) / max(target_complexity[split_name], 1.0)
            overflow_penalty = 0.0
            if next_count > target_sample_counts[split_name]:
                overflow_penalty = (next_count - target_sample_counts[split_name]) * 0.15
            score = count_error + complexity_error + overflow_penalty
            if best_score is None or score < best_score or (
                score == best_score and current_sample_counts[split_name] < current_sample_counts[best_split]  # type: ignore[index]
            ):
                best_score = score
                best_split = split_name

        assert best_split is not None
        group_split_assignment[group_name] = best_split
        current_sample_counts[best_split] += group_sample_count
        current_complexity[best_split] += group_complexity

    sample_lookup = {record.sample_id: record for record in sample_records}
    for group_name, members in groups.items():
        split_name = group_split_assignment[group_name]
        for member in members:
            sample_lookup[member.sample_id].split = split_name

    for record in prompt_records:
        record.split = sample_lookup[record.sample_id].split


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def generate_quality_report(
    sample_records: list[SampleRecord],
    prompt_records: list[PromptVariantRecord],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    split_counts = defaultdict(int)
    prompt_split_counts = defaultdict(int)
    leakage_groups = defaultdict(list)
    for record in sample_records:
        split_counts[record.split] += 1
        leakage_groups[record.leakage_group].append(record.sample_id)
    for record in prompt_records:
        prompt_split_counts[record.split] += 1

    validation_counts = defaultdict(int)
    for record in sample_records:
        validation_counts[record.validation_status] += 1

    all_yaml_parse = all(record.yaml_parse_ok for record in sample_records)
    deterministic_normalization = all(
        record.normalization_semantics_preserved and record.target_yaml_normalized
        for record in sample_records
    )
    shared_split_ok = all(
        len({record.split for record in prompt_records if record.sample_id == sample_id}) == 1
        for sample_id in {record.sample_id for record in prompt_records}
    )
    ready_for_next_step = (
        len(sample_records) == 283
        and len(prompt_records) == 566
        and validation_counts["reject"] == 0
        and all_yaml_parse
        and deterministic_normalization
        and shared_split_ok
    )

    largest_leakage_group = max((len(members) for members in leakage_groups.values()), default=0)
    report = {
        "dataset_version": DATASET_VERSION,
        "domain": DOMAIN,
        "seed": seed,
        "output_dir": str(output_dir),
        "sample_count": len(sample_records),
        "prompt_variant_count": len(prompt_records),
        "expected_sample_count": 283,
        "expected_prompt_variant_count": 566,
        "validation_status_counts": dict(validation_counts),
        "sample_split_counts": dict(split_counts),
        "prompt_variant_split_counts": dict(prompt_split_counts),
        "all_yaml_parse_ok": all_yaml_parse,
        "normalization_is_deterministic": deterministic_normalization,
        "shared_split_per_sample_ok": shared_split_ok,
        "largest_leakage_group_size": largest_leakage_group,
        "ready_for_next_step": ready_for_next_step,
        "readiness_gates": {
            "manifest_complete": len(sample_records) == 283 and len(prompt_records) == 566,
            "yaml_validation_100_percent": all_yaml_parse,
            "no_reject_rows": validation_counts["reject"] == 0,
            "variants_share_split": shared_split_ok,
            "export_train_ready_exists": True,
        },
    }
    return report


def output_rows(sample_records: list[SampleRecord], prompt_records: list[PromptVariantRecord]) -> dict[str, list[dict[str, Any]]]:
    sample_rows = [record.__dict__.copy() for record in sample_records]
    prompt_rows = [record.__dict__.copy() for record in prompt_records]
    split_rows = [
        {
            "sample_id": record.sample_id,
            "domain": record.domain,
            "leakage_group": record.leakage_group,
            "leakage_reasons": record.leakage_reasons,
            "split": record.split,
            "validation_status": record.validation_status,
        }
        for record in sample_records
    ]
    train_ready_rows = [
        {
            "sample_id": record.sample_id,
            "domain": record.domain,
            "prompt_variant": record.prompt_variant,
            "prompt_policy": record.prompt_policy,
            "prompt_text": record.prompt_text,
            "target_yaml_raw": record.target_yaml_raw,
            "target_yaml_normalized": record.target_yaml_normalized,
            "split": record.split,
            "validation_status": record.validation_status,
            "leakage_group": record.leakage_group,
        }
        for record in prompt_records
    ]
    train_ready_sample_rows = [
        {
            "sample_id": record.sample_id,
            "domain": record.domain,
            "prompt_policy": record.prompt_policy,
            "prompt_original": record.question_text,
            "prompt_simplified": record.question_simplified_text,
            "target_yaml_raw": record.target_yaml_raw,
            "target_yaml_normalized": record.target_yaml_normalized,
            "split": record.split,
            "validation_status": record.validation_status,
            "leakage_group": record.leakage_group,
        }
        for record in sample_records
    ]
    return {
        "samples": sample_rows,
        "prompts": prompt_rows,
        "splits": split_rows,
        "train_ready": train_ready_rows,
        "train_ready_samples": train_ready_sample_rows,
    }


def write_run_summary(output_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {DATASET_VERSION}",
        "",
        f"- Domain: {report['domain']}",
        f"- Samples: {report['sample_count']}",
        f"- Prompt variants: {report['prompt_variant_count']}",
        f"- Splits: {report['sample_split_counts']}",
        f"- Ready for next step: {report['ready_for_next_step']}",
        f"- Seed: {report['seed']}",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-ready Kubernetes dataset from data/.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for processed artifacts.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed for split assignment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or repo_root / "data" / "processed" / DATASET_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_records, prompt_records = build_records(repo_root)
    annotate_duplicates(sample_records, prompt_records)
    assign_splits(sample_records, prompt_records, seed=args.seed)

    rows = output_rows(sample_records, prompt_records)
    write_csv(output_dir / "dataset_manifest_samples.csv", rows["samples"])
    write_csv(output_dir / "dataset_manifest_prompt_variants.csv", rows["prompts"])
    write_csv(output_dir / "dataset_splits.csv", rows["splits"])
    write_jsonl(output_dir / "dataset_train_ready.jsonl", rows["train_ready"])
    write_jsonl(output_dir / "dataset_train_ready_samples.jsonl", rows["train_ready_samples"])

    quality_report = generate_quality_report(sample_records, prompt_records, output_dir, seed=args.seed)
    write_json(output_dir / "quality_report.json", quality_report)
    write_run_summary(output_dir, quality_report)

    print(json.dumps(quality_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
