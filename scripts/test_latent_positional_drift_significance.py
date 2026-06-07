from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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


@dataclass(frozen=True)
class FeatureRow:
    vector: np.ndarray
    unit_id: str
    split: str
    level: int
    quartile: int


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run permutation significance tests for positional latent drift."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--feature-strategy", default="record_prefix_state")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--scopes", default="all,train,validation")
    parser.add_argument("--min-group-count", type=int, default=8)
    parser.add_argument("--max-per-group", type=int, default=160)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--one-row-per-unit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer at most one sampled row from each unit_id inside each level/quartile group.",
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


def quartile_for_position(line_position: int, line_count: int) -> int:
    if line_count <= 0:
        raise ValueError("line_count_must_be_positive")
    return min((line_position * 4) // line_count, 3)


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


def compute_scaler(feature_path: Path, allowed_splits: set[str]) -> tuple[np.ndarray, np.ndarray]:
    vector_sum: np.ndarray | None = None
    squared_sum: np.ndarray | None = None
    count = 0
    for row in iter_jsonl(feature_path):
        if str(row["split"]) not in allowed_splits:
            continue
        vector = np.asarray(row["feature"], dtype=np.float64)
        if vector_sum is None:
            vector_sum = np.zeros_like(vector)
            squared_sum = np.zeros_like(vector)
        vector_sum += vector
        squared_sum += vector * vector
        count += 1
    if count == 0 or vector_sum is None or squared_sum is None:
        raise ValueError(f"no_vectors_for_splits:{feature_path}")
    mean = vector_sum / count
    variance = np.maximum((squared_sum / count) - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-12] = 1.0
    return mean, std


def load_grouped_features(
    *,
    feature_path: Path,
    unit_lengths: dict[str, int],
    allowed_splits: set[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, dict[int, dict[int, list[FeatureRow]]]]:
    grouped: dict[str, dict[int, dict[int, list[FeatureRow]]]] = {
        "all": defaultdict(lambda: defaultdict(list))
    }
    for split in allowed_splits:
        grouped[split] = defaultdict(lambda: defaultdict(list))

    for row in iter_jsonl(feature_path):
        split = str(row["split"])
        if split not in allowed_splits:
            continue
        unit_id = str(row["unit_id"])
        line_count = unit_lengths.get(unit_id)
        if not line_count:
            continue
        quartile = quartile_for_position(int(row["line_position"]), line_count)
        level = int(row["level"])
        vector = (np.asarray(row["feature"], dtype=np.float64) - mean) / std
        feature_row = FeatureRow(
            vector=vector.astype(np.float32),
            unit_id=unit_id,
            split=split,
            level=level,
            quartile=quartile,
        )
        grouped["all"][level][quartile].append(feature_row)
        grouped[split][level][quartile].append(feature_row)

    return grouped


def sample_rows(
    rows: list[FeatureRow],
    *,
    n: int,
    rng: np.random.Generator,
    one_row_per_unit: bool,
) -> list[FeatureRow]:
    if len(rows) <= n:
        return list(rows)
    order = rng.permutation(len(rows))
    if not one_row_per_unit:
        return [rows[int(index)] for index in order[:n]]

    selected: list[FeatureRow] = []
    seen_units: set[str] = set()
    leftovers: list[FeatureRow] = []
    for index in order:
        row = rows[int(index)]
        if row.unit_id not in seen_units and len(selected) < n:
            selected.append(row)
            seen_units.add(row.unit_id)
        else:
            leftovers.append(row)
    if len(selected) < n:
        selected.extend(leftovers[: n - len(selected)])
    return selected


def stack_vectors(rows: list[FeatureRow]) -> np.ndarray:
    return np.stack([row.vector for row in rows], axis=0).astype(np.float64, copy=False)


def squared_euclidean_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    y = x if y is None else y
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    distances = x_norm + y_norm - 2.0 * x @ y.T
    return np.maximum(distances, 0.0)


def median_bandwidth_gamma(x: np.ndarray, *, rng: np.random.Generator, max_points: int = 300) -> float:
    if x.shape[0] > max_points:
        indices = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[indices]
    distances = squared_euclidean_matrix(x)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    positive = upper[upper > 1e-12]
    median_sq = float(np.median(positive)) if positive.size else 1.0
    return 1.0 / (2.0 * median_sq)


def biased_mmd2_from_kernel(kernel: np.ndarray, mask_a: np.ndarray) -> float:
    mask_b = ~mask_a
    k_aa = kernel[np.ix_(mask_a, mask_a)].mean()
    k_bb = kernel[np.ix_(mask_b, mask_b)].mean()
    k_ab = kernel[np.ix_(mask_a, mask_b)].mean()
    return float(k_aa + k_bb - 2.0 * k_ab)


def mmd_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    pooled = np.vstack([x, y])
    n_x = x.shape[0]
    gamma = median_bandwidth_gamma(pooled, rng=rng)
    kernel = np.exp(-gamma * squared_euclidean_matrix(pooled))
    mask = np.zeros(pooled.shape[0], dtype=bool)
    mask[:n_x] = True
    observed = biased_mmd2_from_kernel(kernel, mask)
    null_values = []
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(pooled.shape[0])
        perm_mask = np.zeros(pooled.shape[0], dtype=bool)
        perm_mask[permuted[:n_x]] = True
        value = biased_mmd2_from_kernel(kernel, perm_mask)
        null_values.append(value)
        if value >= observed:
            exceedances += 1
    p_value = (exceedances + 1.0) / (permutations + 1.0)
    null = np.asarray(null_values, dtype=np.float64)
    return {
        "mmd2_biased": observed,
        "p_value": p_value,
        "gamma": gamma,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if null.size > 1 else 0.0,
    }


def permanova_f_from_distance(distance_sq: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    n = distance_sq.shape[0]
    groups = np.unique(labels)
    k = len(groups)
    if k < 2 or n <= k:
        return float("nan"), float("nan"), float("nan")
    total_ss = float(np.sum(np.triu(distance_sq, k=1)) / n)
    within_ss = 0.0
    for group in groups:
        idx = np.flatnonzero(labels == group)
        if idx.size <= 1:
            continue
        sub = distance_sq[np.ix_(idx, idx)]
        within_ss += float(np.sum(np.triu(sub, k=1)) / idx.size)
    between_ss = total_ss - within_ss
    f_stat = (between_ss / (k - 1)) / (within_ss / (n - k)) if within_ss > 0 else float("inf")
    r2 = between_ss / total_ss if total_ss > 0 else float("nan")
    return f_stat, r2, between_ss


def permanova_permutation_test(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    distance_sq = squared_euclidean_matrix(x)
    observed_f, r2, between_ss = permanova_f_from_distance(distance_sq, labels)
    exceedances = 0
    null_values = []
    for _ in range(permutations):
        permuted_labels = rng.permutation(labels)
        value, _, _ = permanova_f_from_distance(distance_sq, permuted_labels)
        null_values.append(value)
        if value >= observed_f:
            exceedances += 1
    p_value = (exceedances + 1.0) / (permutations + 1.0)
    null = np.asarray(null_values, dtype=np.float64)
    return {
        "pseudo_f": observed_f,
        "r2": r2,
        "between_ss": between_ss,
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if null.size > 1 else 0.0,
    }


def one_way_f(values: np.ndarray, labels: np.ndarray) -> float:
    groups = np.unique(labels)
    n = values.size
    k = len(groups)
    grand = float(values.mean())
    ss_between = 0.0
    ss_within = 0.0
    for group in groups:
        group_values = values[labels == group]
        mean = float(group_values.mean())
        ss_between += group_values.size * (mean - grand) ** 2
        ss_within += float(np.sum((group_values - mean) ** 2))
    if n <= k or ss_within <= 0:
        return float("nan")
    return (ss_between / (k - 1)) / (ss_within / (n - k))


def distances_to_group_centroids(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = np.zeros(x.shape[0], dtype=np.float64)
    for group in np.unique(labels):
        idx = np.flatnonzero(labels == group)
        centroid = x[idx].mean(axis=0)
        distances[idx] = np.linalg.norm(x[idx] - centroid, axis=1)
    return distances


def permdisp_permutation_test(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    observed_distances = distances_to_group_centroids(x, labels)
    observed_f = one_way_f(observed_distances, labels)
    exceedances = 0
    null_values = []
    for _ in range(permutations):
        permuted_labels = rng.permutation(labels)
        distances = distances_to_group_centroids(x, permuted_labels)
        value = one_way_f(distances, permuted_labels)
        null_values.append(value)
        if value >= observed_f:
            exceedances += 1
    p_value = (exceedances + 1.0) / (permutations + 1.0)
    null = np.asarray(null_values, dtype=np.float64)
    return {
        "dispersion_f": observed_f,
        "p_value": p_value,
        "mean_distance_to_centroid": float(observed_distances.mean()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if null.size > 1 else 0.0,
    }


def benjamini_hochberg(rows: list[dict[str, Any]], p_key: str = "p_value", q_key: str = "q_value") -> None:
    valid = [(idx, float(row[p_key])) for idx, row in enumerate(rows) if row.get(p_key) is not None]
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    previous = 1.0
    q_values: dict[int, float] = {}
    for rank_from_end, (idx, p_value) in enumerate(reversed(valid), start=1):
        rank = m - rank_from_end + 1
        q_value = min(previous, p_value * m / rank)
        previous = q_value
        q_values[idx] = q_value
    for idx, row in enumerate(rows):
        row[q_key] = q_values.get(idx)


def run_mmd_tests(
    *,
    grouped: dict[str, dict[int, dict[int, list[FeatureRow]]]],
    scopes: list[str],
    min_group_count: int,
    max_per_group: int,
    permutations: int,
    rng: np.random.Generator,
    one_row_per_unit: bool,
    feature_strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for level in sorted(grouped[scope]):
            q1 = grouped[scope][level].get(0, [])
            q4 = grouped[scope][level].get(3, [])
            if len(q1) < min_group_count or len(q4) < min_group_count:
                continue
            n = min(max_per_group, len(q1), len(q4))
            sampled_q1 = sample_rows(q1, n=n, rng=rng, one_row_per_unit=one_row_per_unit)
            sampled_q4 = sample_rows(q4, n=n, rng=rng, one_row_per_unit=one_row_per_unit)
            x = stack_vectors(sampled_q1)
            y = stack_vectors(sampled_q4)
            result = mmd_permutation_test(x, y, permutations=permutations, rng=rng)
            rows.append(
                {
                    "feature_strategy": feature_strategy,
                    "scope": scope,
                    "level": level,
                    "quartile_a": QUARTILE_NAMES[0],
                    "quartile_b": QUARTILE_NAMES[3],
                    "sampled_count_per_group": n,
                    "available_count_a": len(q1),
                    "available_count_b": len(q4),
                    "permutations": permutations,
                    **result,
                }
            )
    benjamini_hochberg(rows)
    return rows


def run_permanova_tests(
    *,
    grouped: dict[str, dict[int, dict[int, list[FeatureRow]]]],
    scopes: list[str],
    min_group_count: int,
    max_per_group: int,
    permutations: int,
    rng: np.random.Generator,
    one_row_per_unit: bool,
    feature_strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for level in sorted(grouped[scope]):
            quartile_groups = {
                quartile: rows_for_group
                for quartile, rows_for_group in grouped[scope][level].items()
                if len(rows_for_group) >= min_group_count
            }
            if len(quartile_groups) < 2:
                continue
            n = min(max_per_group, min(len(value) for value in quartile_groups.values()))
            sampled_rows: list[FeatureRow] = []
            labels: list[int] = []
            for quartile in sorted(quartile_groups):
                sampled = sample_rows(
                    quartile_groups[quartile],
                    n=n,
                    rng=rng,
                    one_row_per_unit=one_row_per_unit,
                )
                sampled_rows.extend(sampled)
                labels.extend([quartile] * len(sampled))
            x = stack_vectors(sampled_rows)
            label_array = np.asarray(labels, dtype=np.int64)
            permanova = permanova_permutation_test(
                x,
                label_array,
                permutations=permutations,
                rng=rng,
            )
            permdisp = permdisp_permutation_test(
                x,
                label_array,
                permutations=permutations,
                rng=rng,
            )
            rows.append(
                {
                    "feature_strategy": feature_strategy,
                    "scope": scope,
                    "level": level,
                    "quartiles_tested": "|".join(QUARTILE_NAMES[q] for q in sorted(quartile_groups)),
                    "quartile_count": len(quartile_groups),
                    "sampled_count_per_quartile": n,
                    "sampled_total_count": x.shape[0],
                    "available_counts": json.dumps(
                        {QUARTILE_NAMES[q]: len(v) for q, v in sorted(quartile_groups.items())},
                        sort_keys=True,
                    ),
                    "permutations": permutations,
                    **permanova,
                    "permdisp_f": permdisp["dispersion_f"],
                    "permdisp_p_value": permdisp["p_value"],
                    "permdisp_mean_distance_to_centroid": permdisp["mean_distance_to_centroid"],
                    "permdisp_null_mean": permdisp["null_mean"],
                    "permdisp_null_std": permdisp["null_std"],
                }
            )
    benjamini_hochberg(rows)
    benjamini_hochberg(rows, p_key="permdisp_p_value", q_key="permdisp_q_value")
    return rows


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
    output_dir = (
        args.output_dir
        or (run_dir / "positional_drift_significance" / args.feature_strategy)
    ).resolve()
    allowed_splits = set(parse_csv_list(args.splits))
    scopes = parse_csv_list(args.scopes)
    if "all" not in scopes:
        scopes.insert(0, "all")
    unsupported_scopes = sorted(set(scopes) - (allowed_splits | {"all"}))
    if unsupported_scopes:
        raise ValueError(f"unsupported_scopes:{unsupported_scopes}")
    if args.min_group_count < 2:
        raise ValueError("--min-group-count must be at least 2")
    if args.max_per_group < args.min_group_count:
        raise ValueError("--max-per-group must be >= --min-group-count")
    if args.permutations <= 0:
        raise ValueError("--permutations must be positive")

    metadata_path = run_dir / "line_metadata.jsonl"
    feature_path = run_dir / f"features_{args.feature_strategy}.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    unit_lengths = read_unit_lengths(metadata_path, allowed_splits)
    mean, std = compute_scaler(feature_path, allowed_splits)
    grouped = load_grouped_features(
        feature_path=feature_path,
        unit_lengths=unit_lengths,
        allowed_splits=allowed_splits,
        mean=mean,
        std=std,
    )

    rng = np.random.default_rng(args.random_state)
    mmd_rows = run_mmd_tests(
        grouped=grouped,
        scopes=scopes,
        min_group_count=args.min_group_count,
        max_per_group=args.max_per_group,
        permutations=args.permutations,
        rng=rng,
        one_row_per_unit=args.one_row_per_unit,
        feature_strategy=args.feature_strategy,
    )
    permanova_rows = run_permanova_tests(
        grouped=grouped,
        scopes=scopes,
        min_group_count=args.min_group_count,
        max_per_group=args.max_per_group,
        permutations=args.permutations,
        rng=rng,
        one_row_per_unit=args.one_row_per_unit,
        feature_strategy=args.feature_strategy,
    )

    write_csv(output_dir / "mmd_q1_q4_tests.csv", mmd_rows)
    write_csv(output_dir / "permanova_quartile_tests.csv", permanova_rows)
    summary = {
        "run_dir": str(run_dir),
        "feature_strategy": args.feature_strategy,
        "output_dir": str(output_dir),
        "splits": sorted(allowed_splits),
        "scopes": scopes,
        "min_group_count": args.min_group_count,
        "max_per_group": args.max_per_group,
        "permutations": args.permutations,
        "random_state": args.random_state,
        "one_row_per_unit": args.one_row_per_unit,
        "mmd_test_count": len(mmd_rows),
        "permanova_test_count": len(permanova_rows),
        "mmd_significant_q_lt_0_05": sum(
            1 for row in mmd_rows if row.get("q_value") is not None and float(row["q_value"]) < 0.05
        ),
        "permanova_significant_q_lt_0_05": sum(
            1
            for row in permanova_rows
            if row.get("q_value") is not None and float(row["q_value"]) < 0.05
        ),
        "permdisp_significant_q_lt_0_05": sum(
            1
            for row in permanova_rows
            if row.get("permdisp_q_value") is not None and float(row["permdisp_q_value"]) < 0.05
        ),
        "artifacts": {
            "mmd_q1_q4_tests": str(output_dir / "mmd_q1_q4_tests.csv"),
            "permanova_quartile_tests": str(output_dir / "permanova_quartile_tests.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
