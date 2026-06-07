from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
STATIC_DIR = Path(__file__).resolve().parent / "dpo_preference_ui_static"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_structured_semantic_generation.dataset_io import read_jsonl
from llm_structured_semantic_generation.dpo_preference_annotation import (
    AnnotationPaths,
    append_agent_suggestion,
    append_human_preference,
    approved_preference_events,
    build_decision_packet,
    discover_candidate_run_dirs,
    export_final_preferences,
    get_labeling_guide,
    initialize_annotation_run,
    latest_decisions_by_unit,
    load_annotation_events,
    load_dataset_index,
    load_review_units,
    write_annotation_state,
)
from llm_structured_semantic_generation.prompt_requirement_audit import (
    append_prompt_requirement_gold_annotation,
    compute_prompt_requirement_audit_report,
    load_latest_gold_cases,
    seed_prompt_requirement_gold_file,
    select_prompt_requirement_audit_cases,
    write_prompt_requirement_audit_report,
)


class PreferenceApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        candidate_run_dirs = args.candidate_run_dir or discover_candidate_run_dirs(args.candidate_root)
        if not candidate_run_dirs:
            raise RuntimeError(f"No candidate runs found under {args.candidate_root}")
        self.paths = initialize_annotation_run(
            output_dir=args.output_dir,
            run_id=args.run_id,
            candidate_run_dirs=candidate_run_dirs,
            dataset_path=args.dataset_path,
            prompt_requirements_path=args.prompt_requirements_path,
            split=args.split,
            batch_size=args.batch_size,
        )
        self.units = load_review_units(
            candidate_run_dirs=candidate_run_dirs,
            dataset_path=args.dataset_path,
            prompt_requirements_path=args.prompt_requirements_path,
            split=args.split,
        )
        self.units_by_id = {unit["unit_id"]: unit for unit in self.units}
        if args.prompt_requirements_path and args.prompt_requirements_path.exists():
            self._seed_audit_cases()
        write_annotation_state(paths=self.paths, units=self.units)

    def summary(self) -> dict[str, Any]:
        latest = latest_decisions_by_unit(load_annotation_events(self.paths))
        events = load_annotation_events(self.paths)
        candidates = [
            candidate
            for unit in self.units
            for candidate in unit.get("candidates", [])
        ]
        gate_values = [
            candidate.get("metrics", {}).get("kubernetes_domain_gate_pass")
            for candidate in candidates
            if candidate.get("metrics", {}).get("kubernetes_domain_gate_pass") is not None
        ]
        gate_pass_count = sum(1 for value in gate_values if bool(value))
        return {
            "run_id": self.args.run_id,
            "split": self.args.split,
            "output_dir": str(self.args.output_dir),
            "unit_count": len(self.units),
            "candidate_count": len(candidates),
            "kubernetes_domain_gate_pass_count": gate_pass_count,
            "kubernetes_domain_gate_evaluable_count": len(gate_values),
            "kubernetes_domain_gate_pass_rate": (
                gate_pass_count / len(gate_values) if gate_values else None
            ),
            "approved_unit_count": len(latest),
            "approved_pair_count": len(approved_preference_events(events)),
        }

    def list_units(self, *, offset: int = 0, limit: int = 50, status: str | None = None) -> dict[str, Any]:
        latest = latest_decisions_by_unit(load_annotation_events(self.paths))
        approved_pair_counts: dict[str, int] = {}
        for event in approved_preference_events(load_annotation_events(self.paths)):
            unit_id = str(event.get("unit_id") or "")
            if unit_id:
                approved_pair_counts[unit_id] = approved_pair_counts.get(unit_id, 0) + 1
        summaries = []
        for unit in self.units:
            unit_status = _unit_status(unit["unit_id"], latest)
            if status and unit_status != status:
                continue
            candidates = unit.get("candidates", [])
            best_score = next(
                (
                    candidate.get("preference_score")
                    for candidate in candidates
                    if isinstance(candidate.get("preference_score"), (int, float))
                ),
                None,
            )
            summaries.append(
                {
                    "unit_id": unit["unit_id"],
                    "sample_id": unit["sample_id"],
                    "prompt_variant": unit["prompt_variant"],
                    "split": unit["split"],
                    "candidate_count": len(candidates),
                    "best_score": best_score,
                    "status": unit_status,
                    "approved_pair_count": approved_pair_counts.get(unit["unit_id"], 0),
                    "prompt_requirement_count": len(unit.get("prompt_requirements", [])),
                }
            )
        return {
            "total": len(summaries),
            "offset": offset,
            "limit": limit,
            "units": summaries[offset : offset + limit],
        }

    def get_unit(self, unit_id: str) -> dict[str, Any]:
        unit = self._unit(unit_id)
        events = load_annotation_events(self.paths)
        latest = latest_decisions_by_unit(events).get(unit_id)
        suggestions = [
            event
            for event in events
            if event.get("unit_id") == unit_id and event.get("label_source") == "agent"
        ]
        return {
            **unit,
            "latest_decision": latest,
            "agent_suggestions": suggestions,
        }

    def save_preference(self, payload: dict[str, Any]) -> dict[str, Any]:
        unit = self._unit(str(payload.get("unit_id") or ""))
        event = append_human_preference(paths=self.paths, unit=unit, payload=payload)
        state = write_annotation_state(paths=self.paths, units=self.units)
        return {"event": event, "state": state}

    def save_agent_suggestion(self, payload: dict[str, Any]) -> dict[str, Any]:
        unit = self._unit(str(payload.get("unit_id") or ""))
        event = append_agent_suggestion(paths=self.paths, unit=unit, payload=payload)
        state = write_annotation_state(paths=self.paths, units=self.units)
        return {"event": event, "state": state}

    def export_final(self) -> dict[str, Any]:
        final_rows = export_final_preferences(paths=self.paths, units_by_id=self.units_by_id)
        state = write_annotation_state(paths=self.paths, units=self.units, final_rows=final_rows)
        return {
            "path": str(self.paths.final_preferences),
            "count": len(final_rows),
            "state": state,
        }

    def decision_packet(self, unit_id: str) -> dict[str, Any]:
        return build_decision_packet(self._unit(unit_id))

    def labeling_guide(self) -> dict[str, Any]:
        return get_labeling_guide()

    def audit_cases(self) -> dict[str, Any]:
        rows = load_latest_gold_cases(self.args.output_dir / "prompt_requirement_gold.jsonl")
        return {"total": len(rows), "cases": rows}

    def save_gold_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = append_prompt_requirement_gold_annotation(output_dir=self.args.output_dir, payload=payload)
        report = write_prompt_requirement_audit_report(self.args.output_dir)
        return {"case": row, "report": report}

    def audit_report(self) -> dict[str, Any]:
        rows = load_latest_gold_cases(self.args.output_dir / "prompt_requirement_gold.jsonl")
        report = compute_prompt_requirement_audit_report(rows)
        return report

    def _unit(self, unit_id: str) -> dict[str, Any]:
        unit = self.units_by_id.get(unit_id)
        if unit is None:
            raise KeyError(f"unknown_unit_id:{unit_id}")
        return unit

    def _seed_audit_cases(self) -> None:
        gold_path = self.args.output_dir / "prompt_requirement_gold.jsonl"
        if gold_path.exists():
            return
        dataset_index = load_dataset_index(self.args.dataset_path)
        rows = read_jsonl(self.args.prompt_requirements_path)
        audit_cases = select_prompt_requirement_audit_cases(
            prompt_requirement_rows=rows,
            dataset_index=dataset_index,
            split=self.args.split,
            sample_size=self.args.audit_sample_size,
            seed=self.args.audit_seed,
        )
        seed_prompt_requirement_gold_file(output_dir=self.args.output_dir, audit_cases=audit_cases)
        write_prompt_requirement_audit_report(self.args.output_dir)


class PreferenceRequestHandler(BaseHTTPRequestHandler):
    app: PreferenceApp

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_static("index.html")
            elif parsed.path in ("/app.js", "/style.css"):
                self._send_static(parsed.path.lstrip("/"))
            elif parsed.path == "/api/summary":
                self._send_json(self.app.summary())
            elif parsed.path == "/api/labeling-guide":
                self._send_json(self.app.labeling_guide())
            elif parsed.path == "/api/units":
                query = parse_qs(parsed.query)
                self._send_json(
                    self.app.list_units(
                        offset=_int_query(query, "offset", 0),
                        limit=_int_query(query, "limit", 50),
                        status=_str_query(query, "status"),
                    )
                )
            elif parsed.path.startswith("/api/units/") and parsed.path.endswith("/decision-packet"):
                unit_id = unquote(parsed.path.removeprefix("/api/units/").removesuffix("/decision-packet"))
                self._send_json(self.app.decision_packet(unit_id))
            elif parsed.path.startswith("/api/units/"):
                unit_id = unquote(parsed.path.removeprefix("/api/units/"))
                self._send_json(self.app.get_unit(unit_id))
            elif parsed.path == "/api/audit-cases":
                self._send_json(self.app.audit_cases())
            elif parsed.path == "/api/audit-report":
                self._send_json(self.app.audit_report())
            else:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {parsed.path}")
        except Exception as exc:  # pragma: no cover - exercised by browser use
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/api/preferences":
                self._send_json(self.app.save_preference(payload))
            elif parsed.path == "/api/agent-suggestions":
                self._send_json(self.app.save_agent_suggestion(payload))
            elif parsed.path == "/api/export-final":
                self._send_json(self.app.export_final())
            elif parsed.path == "/api/audit-gold":
                self._send_json(self.app.save_gold_case(payload))
            else:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {parsed.path}")
        except Exception as exc:  # pragma: no cover - exercised by browser use
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length") or "0")
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, name: str) -> None:
        path = STATIC_DIR / name
        if not path.exists():
            self._send_error(HTTPStatus.NOT_FOUND, f"Missing static asset: {name}")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local DPO preference annotation UI.")
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
    parser.add_argument("--run-id", default="dpo-preference-ui-v1")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--split", default="train")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--audit-sample-size", type=int, default=40)
    parser.add_argument("--audit-seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = PreferenceApp(args)
    PreferenceRequestHandler.app = app
    server = ThreadingHTTPServer((args.host, args.port), PreferenceRequestHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving DPO preference UI at {url}")
    print(f"Output dir: {args.output_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        server.server_close()
    return 0


def _unit_status(unit_id: str, latest: dict[str, dict[str, Any]]) -> str:
    event = latest.get(unit_id)
    if event is None:
        return "pending"
    return str(event.get("decision") or "annotated")


def _int_query(query: dict[str, list[str]], key: str, default: int) -> int:
    values = query.get(key)
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        return default


def _str_query(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = values[0].strip()
    return value or None


if __name__ == "__main__":
    raise SystemExit(main())
