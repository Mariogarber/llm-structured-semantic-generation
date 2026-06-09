"""Generate chapter 7 figures for the DPO v2 dataset section.

The values are taken from the repository experiment reports under:
results/dpo_kubernetes_v1/preference_annotation/agent-full-auto-v1
results/dpo_kubernetes_v1/preference_annotation/agent-full-auto-v2
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "latex" / "figures" / "chapter7"
OUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "blue": "#2E6F9E",
    "green": "#3A8F6B",
    "orange": "#D08A2E",
    "red": "#B65A4A",
    "purple": "#7A5CA8",
    "slate": "#596673",
    "light": "#EEF2F5",
    "grid": "#D9DEE3",
    "text": "#1D252C",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": COLORS["slate"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_iterative_cycle() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    nodes = [
        ("SFT\nserializado", (0.8, 2.2), COLORS["blue"]),
        ("DPO v1\nbeta=0.10", (2.8, 2.2), COLORS["green"]),
        ("Candidatos\nv2", (4.8, 2.2), COLORS["orange"]),
        ("Selector\nde preferencias", (6.8, 2.2), COLORS["purple"]),
        ("Dataset\nDPO v2", (8.8, 2.2), COLORS["red"]),
    ]

    for label, (x, y), color in nodes:
        box = FancyBboxPatch(
            (x - 0.78, y - 0.48),
            1.56,
            0.96,
            boxstyle="round,pad=0.045,rounding_size=0.10",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, weight="bold")

    for i in range(len(nodes) - 1):
        x0, y0 = nodes[i][1]
        x1, y1 = nodes[i + 1][1]
        arrow = FancyArrowPatch(
            (x0 + 0.84, y0),
            (x1 - 0.84, y1),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.7,
            color=COLORS["slate"],
        )
        ax.add_patch(arrow)

    loop = FancyArrowPatch(
        (8.8, 1.56),
        (2.8, 1.56),
        connectionstyle="arc3,rad=-0.27",
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.5,
        linestyle="--",
        color=COLORS["slate"],
    )
    ax.add_patch(loop)
    ax.text(
        5.8,
        0.58,
        "Nueva politica ajustada como punto de partida de la siguiente iteracion",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["slate"],
    )
    ax.text(
        5,
        3.45,
        "Construccion iterativa de preferencias automaticas",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )

    save(fig, "fig7_5_dpo_iterative_cycle.png")


def fig_pair_types() -> None:
    v1 = {
        "strong_score_margin": 230,
        "intermediate_hard_negative": 184,
        "gate_practice": 40,
    }
    v2 = {
        "domain_invariant": 91,
        "gate_crossing": 62,
        "prompt_fidelity": 38,
        "structural_fidelity": 25,
        "level5_practice": 13,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=False)
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]]

    for ax, data, title in zip(axes, [v1, v2], ["DPO v1", "DPO v2"]):
        labels = list(data.keys())[::-1]
        values = [data[k] for k in labels]
        colors = palette[: len(labels)][::-1]
        bars = ax.barh(labels, values, color=colors, height=0.58)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("pares")
        ax.tick_params(axis="y", labelsize=9)
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * 0.025, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
        ax.set_xlim(0, max(values) * 1.22)

    fig.suptitle("Composicion de pares de preferencia por iteracion", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_6_dpo_v1_v2_pair_types.png")


def fig_kdv_levels() -> None:
    levels = np.array([1, 2, 3, 4, 5])
    chosen = np.array([12, 6, 10, 118, 83])
    rejected = np.array([145, 8, 21, 29, 26])

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    width = 0.35
    ax.bar(levels - width / 2, chosen, width=width, label="chosen", color=COLORS["green"])
    ax.bar(levels + width / 2, rejected, width=width, label="rejected", color=COLORS["red"])
    ax.set_xticks(levels)
    ax.set_xlabel("nivel KDV")
    ax.set_ylabel("numero de pares")
    ax.set_title("Distribucion de niveles KDV en el dataset DPO v2", fontsize=14, weight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    for x, y in zip(levels - width / 2, chosen):
        ax.text(x, y + 3, str(y), ha="center", va="bottom", fontsize=9)
    for x, y in zip(levels + width / 2, rejected):
        ax.text(x, y + 3, str(y), ha="center", va="bottom", fontsize=9)

    save(fig, "fig7_7_dpo_v2_kdv_levels.png")


def fig_metric_deltas() -> None:
    labels = [
        "prompt F1",
        "KDV score",
        "required fields",
        "level exact",
        "line F1",
    ]
    values = np.array([0.173241, 0.077875, 0.495279, 0.312390, 0.266520])
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["slate"]]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.58)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("diferencia media chosen - rejected")
    ax.set_title("Separacion media de las preferencias DPO v2", fontsize=14, weight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 0.56)
    for bar, value in zip(bars, values):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"+{value:.3f}", va="center", fontsize=9)

    save(fig, "fig7_8_dpo_v2_metric_deltas.png")


def main() -> None:
    fig_iterative_cycle()
    fig_pair_types()
    fig_kdv_levels()
    fig_metric_deltas()


if __name__ == "__main__":
    main()
