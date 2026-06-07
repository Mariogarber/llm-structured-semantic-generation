from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "latent_level_probe_kubernetes_v1"
    / "latent-level-probe-real-full-20260513-1528"
)
QUARTILE_NAMES = ("Q1_0_25", "Q2_25_50", "Q3_50_75", "Q4_75_100")


@dataclass
class VectorStats:
    count: int = 0
    vector_sum: np.ndarray | None = None
    squared_norm_sum: float = 0.0

    def add(self, vector: np.ndarray) -> None:
        if self.vector_sum is None:
            self.vector_sum = np.zeros_like(vector, dtype=np.float64)
        self.count += 1
        self.vector_sum += vector
        self.squared_norm_sum += float(np.dot(vector, vector))

    def centroid(self) -> np.ndarray:
        if self.count <= 0 or self.vector_sum is None:
            raise ValueError("empty_vector_stats")
        return self.vector_sum / self.count

    def within_rms(self) -> float:
        centroid = self.centroid()
        variance = (self.squared_norm_sum / self.count) - float(np.dot(centroid, centroid))
        return math.sqrt(max(variance, 0.0))


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure positional drift in line-level latent features by comparing "
            "centroids across YAML length quartiles."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-strategies",
        default="all",
        help="Comma-separated feature strategies, or 'all' to use every features_*.jsonl file.",
    )
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated splits to include. The aggregate scope 'all' is always written.",
    )
    parser.add_argument("--min-group-count", type=int, default=5)
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Use raw feature vectors instead of global per-dimension standardization per strategy.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc


def resolve_feature_strategies(run_dir: Path, value: str) -> list[str]:
    if value == "all":
        strategies: list[str] = []
        for path in sorted(run_dir.glob("features_*.jsonl")):
            strategies.append(path.stem.removeprefix("features_"))
        if not strategies:
            raise FileNotFoundError(f"no feature files found under {run_dir}")
        return strategies
    return parse_csv_list(value)


def read_unit_lengths(metadata_path: Path, allowed_splits: set[str]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for row in iter_jsonl(metadata_path):
        split = str(row["split"])
        if split not in allowed_splits:
            continue
        unit_id = str(row["unit_id"])
        line_position = int(row["line_position"])
        lengths[unit_id] = max(lengths.get(unit_id, 0), line_position + 1)
    return lengths


def quartile_for_position(line_position: int, line_count: int) -> int:
    if line_count <= 0:
        raise ValueError("line_count_must_be_positive")
    return min((line_position * 4) // line_count, 3)


def compute_scaler(feature_path: Path, allowed_splits: set[str]) -> tuple[np.ndarray, np.ndarray, int]:
    vector_sum: np.ndarray | None = None
    squared_sum: np.ndarray | None = None
    count = 0

    for row in iter_jsonl(feature_path):
        if str(row["split"]) not in allowed_splits:
            continue
        vector = np.asarray(row["feature"], dtype=np.float64)
        if vector_sum is None:
            vector_sum = np.zeros_like(vector, dtype=np.float64)
            squared_sum = np.zeros_like(vector, dtype=np.float64)
        vector_sum += vector
        squared_sum += vector * vector
        count += 1

    if count == 0 or vector_sum is None or squared_sum is None:
        raise ValueError(f"no_vectors_for_splits:{feature_path}")

    mean = vector_sum / count
    variance = np.maximum((squared_sum / count) - (mean * mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-12] = 1.0
    return mean, std, count


def collect_group_stats(
    *,
    feature_path: Path,
    unit_lengths: dict[str, int],
    allowed_splits: set[str],
    mean: np.ndarray,
    std: np.ndarray,
    standardize: bool,
) -> dict[tuple[str, int, int], VectorStats]:
    stats: dict[tuple[str, int, int], VectorStats] = {}

    for row in iter_jsonl(feature_path):
        split = str(row["split"])
        if split not in allowed_splits:
            continue
        unit_id = str(row["unit_id"])
        line_count = unit_lengths.get(unit_id)
        if not line_count:
            continue
        line_position = int(row["line_position"])
        quartile = quartile_for_position(line_position, line_count)
        level = int(row["level"])
        vector = np.asarray(row["feature"], dtype=np.float64)
        if standardize:
            vector = (vector - mean) / std

        for scope in ("all", split):
            key = (scope, level, quartile)
            stats.setdefault(key, VectorStats()).add(vector)

    return stats


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def finite_mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def add_distance_row(
    *,
    rows: list[dict[str, Any]],
    strategy: str,
    scope: str,
    level_a: int,
    level_b: int,
    quartile_a: int,
    quartile_b: int,
    stats_a: VectorStats,
    stats_b: VectorStats,
    distance_type: str,
) -> None:
    centroid_a = stats_a.centroid()
    centroid_b = stats_b.centroid()
    rows.append(
        {
            "feature_strategy": strategy,
            "scope": scope,
            "distance_type": distance_type,
            "level_a": level_a,
            "level_b": level_b,
            "quartile_a": QUARTILE_NAMES[quartile_a],
            "quartile_b": QUARTILE_NAMES[quartile_b],
            "count_a": stats_a.count,
            "count_b": stats_b.count,
            "cosine_distance": cosine_distance(centroid_a, centroid_b),
            "euclidean_distance": euclidean_distance(centroid_a, centroid_b),
            "within_rms_a": stats_a.within_rms(),
            "within_rms_b": stats_b.within_rms(),
            "centroid_norm_a": float(np.linalg.norm(centroid_a)),
            "centroid_norm_b": float(np.linalg.norm(centroid_b)),
        }
    )


def build_rows_for_strategy(
    *,
    strategy: str,
    stats: dict[tuple[str, int, int], VectorStats],
    min_group_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows: list[dict[str, Any]] = []
    same_level_rows: list[dict[str, Any]] = []
    interlevel_rows: list[dict[str, Any]] = []

    scopes = sorted({key[0] for key in stats})
    levels = sorted({key[1] for key in stats})
    quartiles = range(len(QUARTILE_NAMES))

    for scope, level, quartile in sorted(stats):
        group = stats[(scope, level, quartile)]
        centroid = group.centroid()
        group_rows.append(
            {
                "feature_strategy": strategy,
                "scope": scope,
                "level": level,
                "quartile": QUARTILE_NAMES[quartile],
                "count": group.count,
                "within_rms": group.within_rms(),
                "centroid_norm": float(np.linalg.norm(centroid)),
            }
        )

    for scope in scopes:
        for level in levels:
            available = [
                quartile
                for quartile in quartiles
                if stats.get((scope, level, quartile), VectorStats()).count >= min_group_count
            ]
            for quartile_a, quartile_b in itertools.combinations(available, 2):
                add_distance_row(
                    rows=same_level_rows,
                    strategy=strategy,
                    scope=scope,
                    level_a=level,
                    level_b=level,
                    quartile_a=quartile_a,
                    quartile_b=quartile_b,
                    stats_a=stats[(scope, level, quartile_a)],
                    stats_b=stats[(scope, level, quartile_b)],
                    distance_type="same_level_across_quartiles",
                )

        for quartile in quartiles:
            available_levels = [
                level
                for level in levels
                if stats.get((scope, level, quartile), VectorStats()).count >= min_group_count
            ]
            for level_a, level_b in itertools.combinations(available_levels, 2):
                add_distance_row(
                    rows=interlevel_rows,
                    strategy=strategy,
                    scope=scope,
                    level_a=level_a,
                    level_b=level_b,
                    quartile_a=quartile,
                    quartile_b=quartile,
                    stats_a=stats[(scope, level_a, quartile)],
                    stats_b=stats[(scope, level_b, quartile)],
                    distance_type="different_level_same_quartile",
                )

    return group_rows, same_level_rows, interlevel_rows


def build_summary_rows(
    *,
    same_level_rows: list[dict[str, Any]],
    interlevel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (str(row["feature_strategy"]), str(row["scope"]))
            for row in same_level_rows + interlevel_rows
        }
    )
    for strategy, scope in keys:
        same = [
            row
            for row in same_level_rows
            if row["feature_strategy"] == strategy and row["scope"] == scope
        ]
        inter = [
            row
            for row in interlevel_rows
            if row["feature_strategy"] == strategy and row["scope"] == scope
        ]
        same_cos = finite_mean(row["cosine_distance"] for row in same)
        same_euc = finite_mean(row["euclidean_distance"] for row in same)
        inter_cos = finite_mean(row["cosine_distance"] for row in inter)
        inter_euc = finite_mean(row["euclidean_distance"] for row in inter)
        q1_q4 = [
            row
            for row in same
            if row["quartile_a"] == QUARTILE_NAMES[0] and row["quartile_b"] == QUARTILE_NAMES[3]
        ]
        q1_q4_cos = finite_mean(row["cosine_distance"] for row in q1_q4)
        q1_q4_euc = finite_mean(row["euclidean_distance"] for row in q1_q4)
        summary_rows.append(
            {
                "feature_strategy": strategy,
                "scope": scope,
                "same_level_pair_count": len(same),
                "interlevel_pair_count": len(inter),
                "mean_same_level_quartile_cosine_distance": same_cos,
                "mean_interlevel_same_quartile_cosine_distance": inter_cos,
                "cosine_positional_to_level_ratio": (
                    None if same_cos is None or inter_cos in (None, 0.0) else same_cos / inter_cos
                ),
                "mean_same_level_quartile_euclidean_distance": same_euc,
                "mean_interlevel_same_quartile_euclidean_distance": inter_euc,
                "euclidean_positional_to_level_ratio": (
                    None if same_euc is None or inter_euc in (None, 0.0) else same_euc / inter_euc
                ),
                "mean_q1_q4_same_level_cosine_distance": q1_q4_cos,
                "mean_q1_q4_same_level_euclidean_distance": q1_q4_euc,
                "q1_q4_pair_count": len(q1_q4),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "positional_drift_analysis")).resolve()
    allowed_splits = set(parse_csv_list(args.splits))
    strategies = resolve_feature_strategies(run_dir, args.feature_strategies)

    if args.min_group_count <= 0:
        raise ValueError("--min-group-count must be positive")

    metadata_path = run_dir / "line_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    unit_lengths = read_unit_lengths(metadata_path, allowed_splits)
    if not unit_lengths:
        raise ValueError(f"no_units_for_splits:{sorted(allowed_splits)}")

    all_group_rows: list[dict[str, Any]] = []
    all_same_level_rows: list[dict[str, Any]] = []
    all_interlevel_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        feature_path = run_dir / f"features_{strategy}.jsonl"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        mean, std, vector_count = compute_scaler(feature_path, allowed_splits)
        scaler_rows.append(
            {
                "feature_strategy": strategy,
                "vector_count": vector_count,
                "feature_dim": int(mean.shape[0]),
                "standardized": not args.no_standardize,
                "mean_std": float(np.mean(std)),
                "min_std": float(np.min(std)),
                "max_std": float(np.max(std)),
            }
        )
        stats = collect_group_stats(
            feature_path=feature_path,
            unit_lengths=unit_lengths,
            allowed_splits=allowed_splits,
            mean=mean,
            std=std,
            standardize=not args.no_standardize,
        )
        group_rows, same_level_rows, interlevel_rows = build_rows_for_strategy(
            strategy=strategy,
            stats=stats,
            min_group_count=args.min_group_count,
        )
        all_group_rows.extend(group_rows)
        all_same_level_rows.extend(same_level_rows)
        all_interlevel_rows.extend(interlevel_rows)

    summary_rows = build_summary_rows(
        same_level_rows=all_same_level_rows,
        interlevel_rows=all_interlevel_rows,
    )

    write_csv(output_dir / "group_stats.csv", all_group_rows)
    write_csv(output_dir / "same_level_quartile_distances.csv", all_same_level_rows)
    write_csv(output_dir / "interlevel_same_quartile_distances.csv", all_interlevel_rows)
    write_csv(output_dir / "drift_summary.csv", summary_rows)
    write_csv(output_dir / "scaler_summary.csv", scaler_rows)

    summary_payload = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "feature_strategies": strategies,
        "splits": sorted(allowed_splits),
        "unit_count": len(unit_lengths),
        "min_group_count": args.min_group_count,
        "standardized": not args.no_standardize,
        "artifacts": {
            "group_stats": str(output_dir / "group_stats.csv"),
            "same_level_quartile_distances": str(output_dir / "same_level_quartile_distances.csv"),
            "interlevel_same_quartile_distances": str(output_dir / "interlevel_same_quartile_distances.csv"),
            "drift_summary": str(output_dir / "drift_summary.csv"),
            "scaler_summary": str(output_dir / "scaler_summary.csv"),
        },
        "summary_rows": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
