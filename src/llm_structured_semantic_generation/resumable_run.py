from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset_io import append_jsonl, read_jsonl, write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunCompatibilityError(RuntimeError):
    """Raised when a run cannot be safely resumed."""


class ArtifactConsistencyError(RuntimeError):
    """Raised when append-only artifacts are inconsistent with each other."""


class ResumableRun:
    """Manage append-only run artifacts and a resumable state file."""

    def __init__(
        self,
        *,
        run_dir: Path,
        config: dict[str, Any],
        total_units: int,
        unit_id_field: str,
        primary_artifact_name: str,
        artifact_paths: dict[str, str],
        primary_rows: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.total_units = total_units
        self.unit_id_field = unit_id_field
        self.primary_artifact_name = primary_artifact_name
        self.artifact_paths = {name: run_dir / relative_path for name, relative_path in artifact_paths.items()}
        self.primary_rows = primary_rows
        self.state = state
        self.completed_unit_ids = [self._extract_unit_id(row) for row in primary_rows]
        self.completed_unit_id_set = set(self.completed_unit_ids)

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def is_complete(self) -> bool:
        return len(self.completed_unit_id_set) >= self.total_units

    @classmethod
    def initialize(
        cls,
        *,
        run_dir: Path,
        config: dict[str, Any],
        total_units: int,
        unit_id_field: str,
        primary_artifact_name: str,
        artifact_paths: dict[str, str],
    ) -> ResumableRun:
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        existing_config = cls._load_json(config_path) if config_path.exists() else None
        cls._validate_resume_signature(existing_config, config, run_dir)
        write_json(config_path, config)

        primary_path = run_dir / artifact_paths[primary_artifact_name]
        primary_rows = read_jsonl(primary_path, allow_truncated_last_line=True) if primary_path.exists() else []
        cls._assert_unique_unit_ids(primary_rows, unit_id_field, primary_path)

        existing_state = cls._load_json(run_dir / "state.json") if (run_dir / "state.json").exists() else None
        state = cls._build_state_payload(
            config=config,
            total_units=total_units,
            unit_id_field=unit_id_field,
            primary_artifact_name=primary_artifact_name,
            artifact_paths=artifact_paths,
            primary_rows=primary_rows,
            existing_state=existing_state,
        )
        write_json(run_dir / "state.json", state)
        return cls(
            run_dir=run_dir,
            config=config,
            total_units=total_units,
            unit_id_field=unit_id_field,
            primary_artifact_name=primary_artifact_name,
            artifact_paths=artifact_paths,
            primary_rows=primary_rows,
            state=state,
        )

    def record_batch(
        self,
        primary_rows: list[dict[str, Any]],
        *,
        secondary_rows_by_name: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        if not primary_rows:
            return
        batch_unit_ids = [self._extract_unit_id(row) for row in primary_rows]
        duplicates = sorted(set(unit_id for unit_id in batch_unit_ids if batch_unit_ids.count(unit_id) > 1))
        if duplicates:
            raise ArtifactConsistencyError(f"duplicate_unit_ids_in_batch:{duplicates}")
        repeated = sorted(self.completed_unit_id_set.intersection(batch_unit_ids))
        if repeated:
            raise ArtifactConsistencyError(f"batch_contains_completed_units:{repeated}")

        append_jsonl(self.artifact_paths[self.primary_artifact_name], primary_rows)
        self.primary_rows.extend(primary_rows)
        self.completed_unit_ids.extend(batch_unit_ids)
        self.completed_unit_id_set.update(batch_unit_ids)

        for artifact_name, rows in (secondary_rows_by_name or {}).items():
            append_jsonl(self.artifact_paths[artifact_name], rows)

        self.state = self._build_state_payload(
            config=self.config,
            total_units=self.total_units,
            unit_id_field=self.unit_id_field,
            primary_artifact_name=self.primary_artifact_name,
            artifact_paths={name: path.name for name, path in self.artifact_paths.items()},
            primary_rows=self.primary_rows,
            existing_state=self.state,
        )
        write_json(self.state_path, self.state)

    def reconcile_secondary_artifact(
        self,
        artifact_name: str,
        *,
        unit_id_field: str,
        expected_rows: list[dict[str, Any]],
    ) -> None:
        artifact_path = self.artifact_paths[artifact_name]
        existing_rows = read_jsonl(artifact_path, allow_truncated_last_line=True) if artifact_path.exists() else []
        self._assert_unique_unit_ids(existing_rows, unit_id_field, artifact_path)

        expected_by_id = {self._extract_unit_id(row, field_name=unit_id_field): row for row in expected_rows}
        existing_ids = {self._extract_unit_id(row, field_name=unit_id_field) for row in existing_rows}
        unexpected = sorted(existing_ids - set(expected_by_id))
        if unexpected:
            raise ArtifactConsistencyError(
                f"artifact_contains_rows_not_backed_by_primary:{artifact_name}:{unexpected}"
            )

        missing_ids = [unit_id for unit_id in self.completed_unit_ids if unit_id in expected_by_id and unit_id not in existing_ids]
        if not missing_ids:
            return
        append_jsonl(artifact_path, [expected_by_id[unit_id] for unit_id in missing_ids])
        self.state = self._build_state_payload(
            config=self.config,
            total_units=self.total_units,
            unit_id_field=self.unit_id_field,
            primary_artifact_name=self.primary_artifact_name,
            artifact_paths={name: path.name for name, path in self.artifact_paths.items()},
            primary_rows=self.primary_rows,
            existing_state=self.state,
        )
        write_json(self.state_path, self.state)

    def mark_completed(self) -> None:
        completed_at = utc_now_iso()
        self.state["status"] = "completed"
        self.state["updated_at"] = completed_at
        self.state["completed_at"] = completed_at
        self.state["processed_units"] = len(self.completed_unit_ids)
        self.state["remaining_units"] = max(self.total_units - len(self.completed_unit_ids), 0)
        write_json(self.state_path, self.state)

    def _extract_unit_id(self, row: dict[str, Any], *, field_name: str | None = None) -> str:
        key = field_name or self.unit_id_field
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ArtifactConsistencyError(f"missing_or_invalid_unit_id:{key}")
        return value

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactConsistencyError(f"invalid_json_file:{path}") from exc

    @staticmethod
    def _validate_resume_signature(
        existing_config: dict[str, Any] | None,
        new_config: dict[str, Any],
        run_dir: Path,
    ) -> None:
        if existing_config is None:
            return
        if existing_config.get("resume_signature") != new_config.get("resume_signature"):
            raise RunCompatibilityError(
                "run_resume_signature_mismatch:"
                f"{run_dir}:existing={existing_config.get('resume_signature')}:"
                f"new={new_config.get('resume_signature')}"
            )

    @classmethod
    def _build_state_payload(
        cls,
        *,
        config: dict[str, Any],
        total_units: int,
        unit_id_field: str,
        primary_artifact_name: str,
        artifact_paths: dict[str, str],
        primary_rows: list[dict[str, Any]],
        existing_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        completed_unit_ids = [cls._extract_unit_id_static(row, unit_id_field) for row in primary_rows]
        created_at = (
            existing_state.get("created_at")
            if existing_state and isinstance(existing_state.get("created_at"), str)
            else utc_now_iso()
        )
        completed = len(completed_unit_ids) >= total_units
        return {
            "run_id": config["run_id"],
            "status": "completed" if completed else "running",
            "created_at": created_at,
            "updated_at": utc_now_iso(),
            "completed_at": existing_state.get("completed_at") if completed and existing_state else None,
            "total_units": total_units,
            "processed_units": len(completed_unit_ids),
            "remaining_units": max(total_units - len(completed_unit_ids), 0),
            "unit_id_field": unit_id_field,
            "completed_unit_ids": completed_unit_ids,
            "primary_artifact": primary_artifact_name,
            "artifact_paths": artifact_paths,
            "resume_signature": config.get("resume_signature"),
        }

    @staticmethod
    def _assert_unique_unit_ids(rows: list[dict[str, Any]], unit_id_field: str, path: Path) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for row in rows:
            unit_id = ResumableRun._extract_unit_id_static(row, unit_id_field)
            if unit_id in seen:
                duplicates.add(unit_id)
            seen.add(unit_id)
        if duplicates:
            duplicates_list = ",".join(sorted(duplicates))
            raise ArtifactConsistencyError(f"duplicate_unit_ids_in_artifact:{path}:{duplicates_list}")

    @staticmethod
    def _extract_unit_id_static(row: dict[str, Any], unit_id_field: str) -> str:
        value = row.get(unit_id_field)
        if not isinstance(value, str) or not value:
            raise ArtifactConsistencyError(f"missing_or_invalid_unit_id:{unit_id_field}")
        return value
