from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json
from llm_structured_semantic_generation.dpo_preference_annotation import load_dataset_index
from llm_structured_semantic_generation.prompt_requirement_audit import (
    PROMPT_REQUIREMENT_AUDIT_REPORT_ARTIFACT,
    PROMPT_REQUIREMENT_GOLD_ARTIFACT,
    seed_prompt_requirement_gold_file,
    select_prompt_requirement_audit_cases,
    write_prompt_requirement_audit_report,
)
from llm_structured_semantic_generation.resumable_run import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small stratified prompt-requirement gold-set seed for human audit.",
    )
    parser.add_argument(
        "--prompt-requirements-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "prompt_requirements.jsonl",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "dataset_train_ready.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dpo_kubernetes_v1" / "preference_annotation" / "manual-v1",
    )
    parser.add_argument("--run-id", default="prompt-requirement-gold-v1")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_index = load_dataset_index(args.dataset_path)
    prompt_requirement_rows = read_jsonl(args.prompt_requirements_path)
    audit_cases = select_prompt_requirement_audit_cases(
        prompt_requirement_rows=prompt_requirement_rows,
        dataset_index=dataset_index,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _config_payload(args)
    config_path = args.output_dir / "config.json"
    existing_config = _load_json(config_path)
    if existing_config:
        write_json(
            config_path,
            {
                **existing_config,
                "prompt_requirement_gold_set": config,
                "updated_at": utc_now_iso(),
            },
        )
    else:
        write_json(config_path, {**config, "prompt_requirement_gold_set": config})
    gold_path = seed_prompt_requirement_gold_file(
        output_dir=args.output_dir,
        audit_cases=audit_cases,
        overwrite=args.overwrite,
    )
    report = write_prompt_requirement_audit_report(args.output_dir)
    state = {
        "run_id": args.run_id,
        "status": "completed",
        "updated_at": utc_now_iso(),
        "total_units": len(audit_cases),
        "processed_units": len(audit_cases),
        "remaining_units": 0,
        "artifacts": {
            "prompt_requirement_gold": PROMPT_REQUIREMENT_GOLD_ARTIFACT,
            "prompt_requirement_audit_report": PROMPT_REQUIREMENT_AUDIT_REPORT_ARTIFACT,
        },
        "reviewed_case_count": report["reviewed_case_count"],
    }
    write_json(args.output_dir / "state.json", state)
    print(f"Wrote {len(audit_cases)} audit cases to {gold_path}")
    print(f"Wrote audit report to {args.output_dir / PROMPT_REQUIREMENT_AUDIT_REPORT_ARTIFACT}")
    return 0


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "stage": "prompt_requirement_gold_set",
        "created_at": utc_now_iso(),
        "prompt_requirements_path": str(args.prompt_requirements_path),
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "resume_signature": {
            "stage": "prompt_requirement_gold_set",
            "prompt_requirements_path": str(args.prompt_requirements_path),
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
