"""Generate chapter 7 figures for the DPO v2 lr=1e-5 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "latex" / "figures" / "chapter7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = ROOT / "results" / "baseline_kubernetes_v1" / "compact-validation70-320-vtfix" / "metrics_recomputed.json"
DPO_V1 = ROOT / "results" / "dpo_kubernetes_v1" / "training" / "dpo-beta030-full-20260530-130338" / "metrics.json"
DPO_V2 = (
    ROOT
    / "results"
    / "dpo_kubernetes_v1"
    / "training"
    / "dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609"
    / "metrics.json"
)
TRAIN_LOG = DPO_V2.parent / "train_log.jsonl"

# Manual recomputation checked after the run report was written. These values
# replace the stale KDV fields present in metrics.json for the lr=1e-5 run.
DPO_V2_MANUAL_KDV = {
    "average_kubernetes_domain_validity_score": 0.8632,
    "average_kubernetes_domain_validity_level": 4.1944,
    "kubernetes_domain_gate_pass_rate": 15 / 70,
    "kubernetes_level_5_pass_rate": 15 / 70,
}


COLORS = {
    "baseline": "#6D7782",
    "v1": "#2E6F9E",
    "v2": "#3A8F6B",
    "orange": "#D08A2E",
    "red": "#B65A4A",
    "purple": "#7A5CA8",
    "grid": "#D9DEE3",
    "text": "#1D252C",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": COLORS["baseline"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def grouped_bar(ax: plt.Axes, labels: list[str], series: dict[str, list[float]], ylim: tuple[float, float]) -> None:
    x = np.arange(len(labels))
    width = 0.25
    offsets = [-width, 0, width]
    colors = [COLORS["baseline"], COLORS["v1"], COLORS["v2"]]
    for offset, (name, values), color in zip(offsets, series.items(), colors):
        ax.bar(x + offset, values, width, label=name, color=color)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left")


def fig_headline_metrics(metrics: dict[str, dict]) -> None:
    metric_keys = [
        ("YAML parse", "yaml_parse_success_rate"),
        ("Exact YAML", "parsed_equal_rate"),
        ("Line F1", "average_line_text_f1"),
        ("Prompt F1", "average_prompt_requirement_f1"),
        ("Semantic F1", "average_semantic_key_f1"),
        ("KDV score", "average_kubernetes_domain_validity_score"),
    ]
    labels = [label for label, _ in metric_keys]
    series = {
        "Baseline": [metrics["baseline"][key] for _, key in metric_keys],
        "DPO v1 beta=0.30": [metrics["dpo_v1"][key] for _, key in metric_keys],
        "DPO v2 lr=1e-5": [metrics["dpo_v2"][key] for _, key in metric_keys],
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    grouped_bar(ax, labels, series, (0, 1.05))
    ax.set_ylabel("valor")
    ax.set_title("Comparación de métricas principales en validación", fontsize=14, weight="bold")
    save(fig, "fig7_9_dpo_v2_lr1e5_headline_metrics.png")


def fig_kubernetes_levels(metrics: dict[str, dict]) -> None:
    levels = np.arange(0, 6)
    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    for name, key, color, marker in [
        ("Baseline", "baseline", COLORS["baseline"], "o"),
        ("DPO v1 beta=0.30", "dpo_v1", COLORS["v1"], "s"),
        ("DPO v2 lr=1e-5", "dpo_v2", COLORS["v2"], "^"),
    ]:
        values = [metrics[key][f"kubernetes_level_{level}_pass_rate"] for level in levels]
        ax.plot(levels, values, marker=marker, linewidth=2.2, markersize=6, label=name, color=color)
    ax.set_xticks(levels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("nivel KDV")
    ax.set_ylabel("pass rate")
    ax.set_title("Validez Kubernetes por niveles", fontsize=14, weight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig7_10_dpo_v2_lr1e5_kdv_levels.png")


def fig_error_profile(metrics: dict[str, dict]) -> None:
    errors = [
        ("YAML parse", "yaml_parse"),
        ("Resources", "missing_resource_requirement"),
        ("runAsNonRoot", "missing_run_as_non_root"),
        ("readOnlyRootFS", "missing_read_only_root_filesystem"),
        ("latest tag", "latest_image_tag"),
        ("required field", "required_field"),
    ]
    labels = [label for label, _ in errors]
    series = {
        "Baseline": [metrics["baseline"].get("kubernetes_domain_error_counts", {}).get(key, 0) for _, key in errors],
        "DPO v1 beta=0.30": [
            metrics["dpo_v1"].get("kubernetes_domain_error_counts", {}).get(key, 0) for _, key in errors
        ],
        "DPO v2 lr=1e-5": [metrics["dpo_v2"].get("kubernetes_domain_error_counts", {}).get(key, 0) for _, key in errors],
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    grouped_bar(ax, labels, series, (0, max(max(v) for v in series.values()) * 1.18))
    ax.set_ylabel("conteo")
    ax.set_title("Perfil de errores de dominio detectados", fontsize=14, weight="bold")
    save(fig, "fig7_11_dpo_v2_lr1e5_error_profile.png")


def rolling_mean(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def fig_training_dynamics(train_log: list[dict]) -> None:
    steps = [row["global_step"] for row in train_log]
    panels = [
        ("loss", "Loss", COLORS["red"]),
        ("reward_margin", "Reward margin", COLORS["v2"]),
        ("reward_accuracy", "Reward accuracy", COLORS["v1"]),
        ("grad_norm", "Grad norm", COLORS["purple"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True)
    for ax, (key, title, color) in zip(axes.flatten(), panels):
        values = [row[key] for row in train_log]
        ax.plot(steps, values, color=color, alpha=0.28, linewidth=1)
        ax.plot(steps, rolling_mean(values), color=color, linewidth=2.1)
        for boundary in [29, 58]:
            ax.axvline(boundary, color=COLORS["grid"], linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("global step")
    axes[1, 1].set_xlabel("global step")
    fig.suptitle("Dinámica de entrenamiento de DPO v2 lr=1e-5", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_12_dpo_v2_lr1e5_training_dynamics.png")


def main() -> None:
    metrics = {
        "baseline": load_json(BASELINE),
        "dpo_v1": load_json(DPO_V1),
        "dpo_v2": load_json(DPO_V2),
    }
    metrics["dpo_v2"].update(DPO_V2_MANUAL_KDV)
    train_log = load_train_log(TRAIN_LOG)

    fig_headline_metrics(metrics)
    fig_kubernetes_levels(metrics)
    fig_error_profile(metrics)
    fig_training_dynamics(train_log)


if __name__ == "__main__":
    main()
