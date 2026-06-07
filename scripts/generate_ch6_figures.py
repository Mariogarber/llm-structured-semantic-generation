"""Generate Chapter 6 figures for the TFM thesis.

Outputs 11 PNG figures to latex/figures/chapter6/:
  fig6_1_architecture.png          - Diagrama de arquitectura two-head
  fig6_2_probe_recall_by_level.png - Recall del latent probe por nivel
  fig6_3_train_loss_twohead.png    - Curvas de pérdida two_head_sft_v1 (LM + nivel)
  fig6_4_comparison_v1.png         - Comparativa baseline zero-shot vs two_head_v1
  fig6_5_level_distribution.png    - Distribución de niveles: gold vs predicted (colapso)
  fig6_6_ordinal_variants.png      - Comparativa 4 variantes ordinal
  fig6_7_threshold_drift.png       - Deriva de umbrales τ_k durante el entrenamiento
  fig6_8_structural_tradeoff.png   - Trade-off: parseo YAML vs recall niveles profundos
  fig6_9_positional_v1_quartiles.png - Diagnostico por cuartiles del positional V1
  fig6_10_thop_metrics_comparison.png - Comparativa THO centered vs THOP
  fig6_11_positional_parser_progression.png - Progresion parser-facing THO/THOP/THOP-FiLM

Usage:
    uv run python scripts/generate_ch6_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch  # noqa: F811 – used via alias below
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "figures" / "chapter6"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = ROOT / "results"

TWO_HEAD_V1_DIR = RESULTS / "two_head_sft_kubernetes_v1" / "two-head-sft-v1-20260516"
DIAGNOSTICS_DIR = RESULTS / "two_head_diagnostics_v1" / "two-head-sft-v1-20260516"
PROBE_DIR = (
    RESULTS
    / "latent_level_probe_kubernetes_v1"
    / "latent-level-probe-real-full-20260513-1528"
)
ORDINAL_DIR = RESULTS / "two_head_ordinal_sft_kubernetes_v1"

ORDINAL_RUNS = {
    "initial": ORDINAL_DIR / "two-head-ordinal-density-v2-20260519" / "metrics.json",
    "lr25": (
        ORDINAL_DIR
        / "two-head-ordinal-density-v2-threshold-lr25-20260520"
        / "metrics.json"
    ),
    "mlp3": (
        ORDINAL_DIR
        / "two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522"
        / "metrics.json"
    ),
    "centered": (
        ORDINAL_DIR
        / "two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523"
        / "metrics.json"
    ),
}

CENTERED_THRESHOLD_LOG = (
    ORDINAL_DIR
    / "two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523"
    / "threshold_history.jsonl"
)

# ---------------------------------------------------------------------------
# Palette / style
# ---------------------------------------------------------------------------

C_BASELINE = "#4878CF"    # blue  — baseline zero-shot
C_SERIAL = "#6ACC65"      # green — serialized_sft (reference)
C_TWOHEAD = "#E24A33"     # red   — two_head
C_GOLD = "#4878CF"        # blue  — gold / ground truth
C_LM = "#6ACC65"          # green — LM loss component
C_LEVEL = "#E24A33"       # red   — level loss component
C_TOTAL = "#555555"       # grey  — total loss
C_ORDINAL_VARIANTS = ["#FAA43A", "#988ED5", "#E24A33", "#4878CF"]
# ^ initial, lr25, mlp3, centered
C_THRESH = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
C_FILM = "#4C9F70"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")


# ---------------------------------------------------------------------------
# Figure 6.1 — Architecture diagram
# ---------------------------------------------------------------------------


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel="", fc="#D9EAF7", ec="#3A7FBF", fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.4,
        )
        ax.add_patch(rect)
        if sublabel:
            ax.text(
                x, y + 0.13, label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#222222",
            )
            ax.text(
                x, y - 0.18, sublabel,
                ha="center", va="center", fontsize=7.5, color="#555555",
            )
        else:
            ax.text(
                x, y, label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#222222",
            )

    def arrow(x1, y1, x2, y2, label="", lw=1.4, color="#555555"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=lw,
                connectionstyle="arc3,rad=0.0",
            ),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.07, my + 0.12, label, fontsize=7.5, color="#555555", ha="left")

    # ── Prompt box ──
    box(1.1, 2.5, 1.5, 0.7, "Prompt", "instrucción NL", fc="#FFF3CD", ec="#D0A010")

    # ── Backbone ──
    box(3.5, 2.5, 2.0, 1.0, "Backbone", "Qwen2.5-7B + LoRA", fc="#D9EAF7", ec="#3A7FBF", fontsize=9)

    # ── Hidden states ──
    box(6.0, 2.5, 1.6, 0.7, "Estados ocultos", r"$h_t \in \mathbb{R}^{3584}$", fc="#EDE8F5", ec="#7B5EA7")

    # ── LM head (top branch) ──
    box(8.5, 3.8, 2.0, 0.7, "LM Head", "autoregresivo", fc="#DFF0D8", ec="#4CAF50")

    # ── Level head (bottom branch) ──
    box(8.5, 1.2, 2.0, 0.7, "Level Head", "MLP(256), 9 clases", fc="#FCDEDE", ec="#D32F2F")

    # ── Output text ──
    box(9.5, 3.8, 0.8, 0.55, "texto", "", fc="#F9F9F9", ec="#AAAAAA", fontsize=8)

    # ── Output level ──
    box(9.5, 1.2, 0.8, 0.55, "nivel", "", fc="#F9F9F9", ec="#AAAAAA", fontsize=8)

    # ── Parser ──
    box(9.0, 2.5, 1.4, 0.65, "Parser", "→ YAML final", fc="#FFF3CD", ec="#D0A010")

    # ── Arrows ──
    arrow(1.85, 2.5, 2.5, 2.5)           # prompt → backbone
    arrow(4.5, 2.5, 5.2, 2.5)            # backbone → hidden states
    arrow(6.8, 2.75, 7.5, 3.8)           # hidden → LM head (upper branch)
    arrow(6.8, 2.25, 7.5, 1.2)           # hidden → Level head (lower branch), "record_prefix_state"
    arrow(9.5, 3.45, 9.2, 2.83)          # LM head → parser
    arrow(9.5, 1.55, 9.2, 2.17)          # Level head → parser

    # ── Labels on diagonal arrows ──
    ax.text(7.0, 3.3, "tokens", fontsize=7.5, color="#555555", ha="center", style="italic")
    ax.text(7.0, 1.7, "record_prefix_state", fontsize=7.0, color="#888888", ha="center", style="italic")

    # ── Loss labels ──
    ax.text(8.5, 4.32, r"$\mathcal{L}_{\mathrm{LM}}$", fontsize=9, color="#4CAF50", ha="center")
    ax.text(8.5, 0.68, r"$\mathcal{L}_{\mathrm{nivel}}$", fontsize=9, color="#D32F2F", ha="center")
    ax.text(8.4, 2.5, r"$\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \lambda\,\mathcal{L}_{\mathrm{nivel}}$",
            fontsize=8, color="#555555", ha="center")

    ax.set_title(
        "Arquitectura two-head: cabezal de lenguaje y cabezal de nivel explícito",
        fontsize=11,
        pad=10,
    )
    fig.tight_layout()
    _save(fig, "fig6_1_architecture.png")


# ---------------------------------------------------------------------------
# Figure 6.2 — Latent probe recall by level
# ---------------------------------------------------------------------------
# Source: latent-level-probe-real-full-20260513-1528/metrics.json
# Best probe: line_first_token__linear (validation classification_report)

PROBE_RECALL = {
    0: 1.0000,
    1: 0.9320,
    2: 0.9478,
    3: 0.7744,
    4: 0.6089,
    5: 0.3158,
    6: 0.6136,
    7: 1.0000,  # 4 samples — unstable
    8: 0.6667,  # 12 samples — unstable
}
PROBE_SUPPORT = {0: 324, 1: 294, 2: 230, 3: 328, 4: 450, 5: 114, 6: 44, 7: 4, 8: 12}


def fig_probe_recall_by_level() -> None:
    levels = list(range(9))
    recalls = [PROBE_RECALL[l] for l in levels]
    supports = [PROBE_SUPPORT[l] for l in levels]

    # Colour by reliability: levels 7-8 have very few samples → grey them
    colors = []
    for l, s in zip(levels, supports):
        if s < 10:
            colors.append("#CCCCCC")
        elif l >= 5:
            colors.append(C_TWOHEAD)
        else:
            colors.append(C_GOLD)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(
        [str(l) for l in levels],
        recalls,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.65,
    )

    for bar, r, s in zip(bars, recalls, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{r:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.03,
            f"n={s}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#777777",
        )

    ax.axvline(4.5, color="#AAAAAA", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(4.6, 0.88, "Límite nivel 5", fontsize=7.5, color="#888888", va="top")

    ax.set_xlabel("Nivel de indentación (level)")
    ax.set_ylabel("Recall del probe lineal (validación)")
    ax.set_title(
        "Recall por nivel — sonda latente lineal (line_first_token)\n"
        "sobre el modelo base Qwen2.5-7B",
        pad=8,
    )
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elems = [
        mpatches.Patch(facecolor=C_GOLD, label="Niveles 0–4 (señal estable)"),
        mpatches.Patch(facecolor=C_TWOHEAD, label="Nivel 5–6 (señal débil)"),
        mpatches.Patch(facecolor="#CCCCCC", label="Nivel 7–8 (n < 10, inestable)"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig6_2_probe_recall_by_level.png")


# ---------------------------------------------------------------------------
# Figure 6.3 — Training loss curves two_head_sft_v1
# ---------------------------------------------------------------------------
# train_log.jsonl fields: global_step, loss, lm_loss, level_loss


def fig_train_loss_twohead() -> None:
    steps, total, lm, level = [], [], [], []
    with open(TWO_HEAD_V1_DIR / "train_log.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line.strip())
            steps.append(rec["global_step"])
            total.append(rec["loss"])
            lm.append(rec["lm_loss"])
            level.append(rec["level_loss"])

    epoch_steps = [53, 106, 159]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    ax.plot(steps, total, color=C_TOTAL, linewidth=1.6, label="Pérdida total", alpha=0.9)
    ax.plot(steps, lm, color=C_LM, linewidth=1.3, linestyle="-", label=r"$\mathcal{L}_\mathrm{LM}$", alpha=0.85)
    ax.plot(steps, level, color=C_LEVEL, linewidth=1.3, linestyle="--", label=r"$\mathcal{L}_\mathrm{nivel}$", alpha=0.85)

    for i, ep in enumerate(epoch_steps, 1):
        ax.axvline(ep, color="#AAAAAA", linestyle=":", linewidth=0.9)
        ax.text(ep + 1.5, max(total) * 0.85, f"Época {i}", fontsize=8, color="#666666", va="top")

    ax.set_xlabel("Paso global (global_step)")
    ax.set_ylabel("Pérdida")
    ax.set_title(
        r"Curvas de pérdida — two\_head\_sft\_v1 (3 épocas, 159 pasos)"
        "\nPérdida LM y pérdida de nivel (λ = 1.0)",
        pad=8,
    )
    ax.set_xlim(0, 163)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig6_3_train_loss_twohead.png")


# ---------------------------------------------------------------------------
# Figure 6.4 — Grouped bar chart: baseline zero-shot vs two_head_v1
# ---------------------------------------------------------------------------
# Numbers from:
#   baseline:      chapter 5 validation metrics / baseline recomputation
#   two_head_v1:    two-head-sft-v1-20260516/metrics.json

BASELINE_METRICS = {
    "yaml_parse": 0.3077,
    "level_exact": 0.1131,
    "level_mae": 0.7896,
    "line_f1": 0.2086,
    "semantic_f1": 0.2778,
    "prompt_f1": 0.2213,
}

TH_V1_METRICS = {
    "yaml_parse": 0.4857,
    "level_exact": 0.3504,
    "level_mae": 0.2392,
    "line_f1": 0.3743,
    "semantic_f1": 0.4592,
    "prompt_f1": 0.4206,
}


def fig_comparison_v1() -> None:
    metric_labels = [
        "Parseo YAML\n(yaml_parse)",
        "Nivel exacto\n(level_exact)",
        "MAE nivel\n(level_mae)",
        "F1 línea\n(line_f1)",
        "F1 semántica\n(semantic_key_f1)",
        "F1 prompt\n(prompt_f1)",
    ]
    baseline_vals = [
        BASELINE_METRICS["yaml_parse"],
        BASELINE_METRICS["level_exact"],
        BASELINE_METRICS["level_mae"],
        BASELINE_METRICS["line_f1"],
        BASELINE_METRICS["semantic_f1"],
        BASELINE_METRICS["prompt_f1"],
    ]
    th_vals = [
        TH_V1_METRICS["yaml_parse"],
        TH_V1_METRICS["level_exact"],
        TH_V1_METRICS["level_mae"],
        TH_V1_METRICS["line_f1"],
        TH_V1_METRICS["semantic_f1"],
        TH_V1_METRICS["prompt_f1"],
    ]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b_base = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="baseline zero-shot",
        color=C_BASELINE,
        edgecolor="white",
        linewidth=0.5,
    )
    b_th = ax.bar(
        x + width / 2,
        th_vals,
        width,
        label="THM v1",
        color=C_TWOHEAD,
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in (b_base, b_th):
        for metric_i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.3f}" if metric_i == 2 else f"{h:.0%}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

    # Special annotation for MAE (lower is better)
    ax.text(
        x[2] + width / 2 + 0.18,
        th_vals[2] + 0.055,
        "↓ mejor",
        fontsize=7.0,
        color="#555555",
        ha="left",
    )

    ax.set_ylabel("Tasas (%) y MAE (0-1)")
    ax.set_title(
        "Comparativa baseline zero-shot vs. THM v1\n(split de validación, 70 muestras)",
        pad=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig6_4_comparison_v1.png")


# ---------------------------------------------------------------------------
# Figure 6.5 — Level distribution: gold vs predicted (collapse)
# ---------------------------------------------------------------------------
# Gold: from diagnostic_metrics.json > target_level_distributions > validation
# Pred: from diagnostic_metrics.json > predicted_level_distribution

GOLD_VALIDATION_COUNTS = {0: 324, 1: 294, 2: 230, 3: 328, 4: 450, 5: 114, 6: 44, 7: 4, 8: 12}
PRED_COUNTS = {0: 300, 1: 278, 2: 304, 3: 431, 4: 411}  # levels 5-8: 0 predictions


def fig_level_distribution() -> None:
    levels = list(range(9))
    total_gold = sum(GOLD_VALIDATION_COUNTS.values())
    total_pred = sum(PRED_COUNTS.values())

    gold_rates = [GOLD_VALIDATION_COUNTS.get(l, 0) / total_gold for l in levels]
    pred_rates = [PRED_COUNTS.get(l, 0) / total_pred for l in levels]

    x = np.arange(len(levels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    b_gold = ax.bar(
        x - width / 2,
        gold_rates,
        width,
        label="Distribución gold (validación)",
        color=C_GOLD,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    b_pred = ax.bar(
        x + width / 2,
        pred_rates,
        width,
        label="Distribución predicha (two_head_v1)",
        color=C_TWOHEAD,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    # Highlight collapse region
    ax.axvspan(4.5, 8.5, color="#FFEEEE", alpha=0.5, zorder=0)
    ax.text(5.2, max(gold_rates) * 0.88, "Sin predicciones\n≥ nivel 5",
            fontsize=8, color="#CC3333", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CC3333", alpha=0.8))

    ax.set_xlabel("Nivel de indentación (level)")
    ax.set_ylabel("Proporción de líneas")
    ax.set_title(
        "Distribución de niveles: ground truth vs. predicciones del two\\_head\\_v1\n"
        "(split de validación — líneas alineadas por posición)",
        pad=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_ylim(0, max(gold_rates) * 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig6_5_level_distribution.png")


# ---------------------------------------------------------------------------
# Figure 6.6 — Ordinal variants comparison (4 runs)
# ---------------------------------------------------------------------------
# Metrics: yaml_parse, level_exact_match, level_mae, line_f1 for 4 variants

ORDINAL_DATA = {
    "initial": {
        "yaml_parse": 0.2571,
        "level_exact": 0.0554,
        "level_mae": 0.7695,
        "line_f1": 0.1899,
        "deep_off1": 0.2190,
    },
    "lr25": {
        "yaml_parse": 0.2857,
        "level_exact": 0.0693,
        "level_mae": 0.8062,
        "line_f1": 0.2269,
        "deep_off1": 0.2536,
    },
    "mlp3": {
        "yaml_parse": 0.1429,
        "level_exact": 0.0409,
        "level_mae": 0.9182,
        "line_f1": 0.1153,
        "deep_off1": 0.0000,
    },
    "centered": {
        "yaml_parse": 0.0286,
        "level_exact": 0.0107,
        "level_mae": 0.4559,
        "line_f1": 0.0145,
        "deep_off1": 0.5652,
    },
}

VARIANT_LABELS = [
    "initial\n(ordinal v2)",
    "lr25\n(thresh. LR ×25)",
    "mlp3\n(MLP 3-cap.)",
    "centered\n(gap=0.5, centrado)",
]


def fig_ordinal_variants() -> None:
    metrics = ["yaml_parse", "level_exact", "line_f1", "deep_off1"]
    metric_labels_short = [
        "Parseo YAML",
        "Nivel exacto",
        "F1 línea",
        "Off-by-1 profundo",
    ]
    colors_per_metric = [C_GOLD, "#6AAED6", "#8EBA42", "#FAA43A"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 4.2), sharey=False)

    for ax, metric, mlabel, color in zip(axes, metrics, metric_labels_short, colors_per_metric):
        variants = list(ORDINAL_DATA.keys())
        vals = [ORDINAL_DATA[v][metric] for v in variants]
        vlabels = VARIANT_LABELS

        bars = ax.bar(
            range(len(variants)),
            vals,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            width=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(vlabels, fontsize=7.2, rotation=0)
        ax.set_title(mlabel, fontsize=9, pad=5)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Valor de la métrica (0 – 1)")

    fig.suptitle(
        "Comparativa de variantes ordinal (split de validación, 70 muestras)\n"
        "Two-head con regresión ordinal y umbrales adaptativos",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig6_6_ordinal_variants.png")


# ---------------------------------------------------------------------------
# Figure 6.7 — Threshold drift over training (centered variant)
# ---------------------------------------------------------------------------


def fig_threshold_drift() -> None:
    steps = []
    thresholds = {k: [] for k in range(8)}  # tau_0 … tau_7

    with open(CENTERED_THRESHOLD_LOG, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line.strip())
            steps.append(rec["global_step"])
            for k in range(8):
                key = f"threshold_tau_{k}"
                if key in rec:
                    thresholds[k].append(rec[key])
                else:
                    thresholds[k].append(None)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for k in range(8):
        vals = thresholds[k]
        valid_steps = [s for s, v in zip(steps, vals) if v is not None]
        valid_vals = [v for v in vals if v is not None]
        if valid_vals:
            ax.plot(
                valid_steps,
                valid_vals,
                color=C_THRESH[k],
                linewidth=1.3,
                label=rf"$\tau_{k}$",
                alpha=0.85,
            )

    epoch_steps = [53, 106, 159]
    for i, ep in enumerate(epoch_steps, 1):
        ax.axvline(ep, color="#CCCCCC", linestyle=":", linewidth=0.8)
        ax.text(ep + 1, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1.0,
                f"Época {i}", fontsize=7.5, color="#888888", va="top")

    ax.set_xlabel("Paso global (global_step)")
    ax.set_ylabel(r"Valor de $\tau_k$")
    ax.set_title(
        r"Evolución de los umbrales $\tau_k$ durante el entrenamiento"
        "\n(variante centered-gap05, λ·LR = ×50)",
        pad=8,
    )
    ax.set_xlim(0, 163)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig6_7_threshold_drift.png")


# ---------------------------------------------------------------------------
# Figure 6.8 — Structural tradeoff: YAML parse vs deep level recall
# ---------------------------------------------------------------------------
# All 5 variants: two_head_v1, initial, lr25, mlp3, centered

TRADEOFF_DATA = [
    # (label, yaml_parse, deep_exact_recall_5_8)
    ("two\\_head\\_v1\n(baseline)", 0.4857, 0.0),
    ("initial\n(ordinal)", 0.2571, 0.0),
    ("lr25\n(thresh. ×25)", 0.2857, 0.0),
    ("mlp3\n(MLP 3-cap.)", 0.1429, 0.0),
    ("centered\n(gap=0.5)", 0.0286, 0.2174),
]


def fig_structural_tradeoff() -> None:
    labels = [d[0] for d in TRADEOFF_DATA]
    yaml_vals = [d[1] for d in TRADEOFF_DATA]
    deep_vals = [d[2] for d in TRADEOFF_DATA]

    x = np.arange(len(labels))
    width = 0.38

    colors_yaml = [C_GOLD] * 4 + ["#AAAAAA"]   # centered in grey (near zero)
    colors_deep = [C_TWOHEAD] * 4 + [C_TWOHEAD]

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    b_yaml = ax.bar(
        x - width / 2,
        yaml_vals,
        width,
        label="Tasa de parseo YAML",
        color=colors_yaml,
        edgecolor="white",
        linewidth=0.5,
    )
    b_deep = ax.bar(
        x + width / 2,
        deep_vals,
        width,
        label="Recall exacto niveles 5–8",
        color=colors_deep,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    for bar, val in zip(list(b_yaml) + list(b_deep), yaml_vals + deep_vals):
        if val > 0.005:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.010,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    # Annotation: only centered breaks zero deep recall
    ax.annotate(
        "Primera variante con\npredicciones nivel ≥ 5",
        xy=(x[-1] + width / 2, deep_vals[-1]),
        xytext=(x[-1] + 0.6, deep_vals[-1] + 0.10),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
        fontsize=8,
        color="#555555",
        ha="left",
    )

    ax.set_ylabel("Valor de la métrica (0 – 1)")
    ax.set_title(
        "Trade-off estructural: parseo YAML vs. recall de niveles profundos (5–8)\n"
        "Cinco variantes two-head supervisado (split de validación)",
        pad=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, 0.62)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig6_8_structural_tradeoff.png")


# ---------------------------------------------------------------------------
# Figure 6.9 — Positional V1 quartile diagnosis
# ---------------------------------------------------------------------------
# Source:
# docs/experiments/two_head_sft/runs/
# TWO_HEAD_ORDINAL_POSITIONAL_V1_FINAL_CONCAT_RUN_20260527_ANALYSIS.md

POSITIONAL_V1_QUARTILES = {
    "Q1": {
        "lines": 455,
        "pred0": 0.9473,
        "gold0": 0.6220,
        "mean_pred": 0.0725,
        "mean_gold": 0.5187,
        "mae": 0.4813,
    },
    "Q2": {
        "lines": 423,
        "pred0": 0.5177,
        "gold0": 0.0662,
        "mean_pred": 1.0071,
        "mean_gold": 2.2861,
        "mae": 1.3830,
    },
    "Q3": {
        "lines": 436,
        "pred0": 0.1583,
        "gold0": 0.0093,
        "mean_pred": 2.6193,
        "mean_gold": 3.4720,
        "mae": 1.1776,
    },
    "Q4": {
        "lines": 396,
        "pred0": 0.0581,
        "gold0": 0.0123,
        "mean_pred": 3.1616,
        "mean_gold": 3.8246,
        "mae": 1.1846,
    },
}


def fig_positional_v1_quartiles() -> None:
    quartiles = list(POSITIONAL_V1_QUARTILES.keys())
    x = np.arange(len(quartiles))
    width = 0.36

    pred0 = [POSITIONAL_V1_QUARTILES[q]["pred0"] for q in quartiles]
    gold0 = [POSITIONAL_V1_QUARTILES[q]["gold0"] for q in quartiles]
    mean_pred = [POSITIONAL_V1_QUARTILES[q]["mean_pred"] for q in quartiles]
    mean_gold = [POSITIONAL_V1_QUARTILES[q]["mean_gold"] for q in quartiles]
    lines = [POSITIONAL_V1_QUARTILES[q]["lines"] for q in quartiles]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax0, ax1 = axes

    b_gold = ax0.bar(
        x - width / 2,
        gold0,
        width,
        label="Gold level=0",
        color=C_GOLD,
        edgecolor="white",
        linewidth=0.6,
    )
    b_pred = ax0.bar(
        x + width / 2,
        pred0,
        width,
        label="Pred. level=0",
        color=C_TWOHEAD,
        edgecolor="white",
        linewidth=0.6,
    )

    for bars in (b_gold, b_pred):
        for bar in bars:
            h = bar.get_height()
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.018,
                f"{h:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    ax0.annotate(
        "Retraso de entrada\nen jerarquía",
        xy=(1 + width / 2, pred0[1]),
        xytext=(1.55, 0.78),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
        fontsize=8,
        color="#555555",
        ha="left",
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{q}\n(n={n})" for q, n in zip(quartiles, lines)])
    ax0.set_ylim(0, 1.12)
    ax0.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax0.set_ylabel("Proporción de líneas")
    ax0.set_title("Predicción excesiva de level=0", pad=8)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1.plot(
        x,
        mean_gold,
        marker="o",
        linewidth=2.0,
        color=C_GOLD,
        label="Media gold",
    )
    ax1.plot(
        x,
        mean_pred,
        marker="o",
        linewidth=2.0,
        color=C_TWOHEAD,
        label="Media predicha",
    )
    ax1.fill_between(
        x,
        mean_pred,
        mean_gold,
        color=C_TWOHEAD,
        alpha=0.12,
        label="Brecha de profundidad",
    )
    for xi, gp, pp in zip(x, mean_gold, mean_pred):
        ax1.vlines(xi, pp, gp, color="#999999", linewidth=0.9, linestyle=":")
        ax1.text(xi, gp + 0.12, f"{gp:.1f}", ha="center", fontsize=8, color=C_GOLD)
        ax1.text(xi, pp - 0.22, f"{pp:.1f}", ha="center", fontsize=8, color=C_TWOHEAD)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{q}\n(n={n})" for q, n in zip(quartiles, lines)])
    ax1.set_ylim(0, 4.55)
    ax1.set_ylabel("Nivel medio")
    ax1.set_title("Profundidad media por tramo", pad=8)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle(
        "Diagnóstico temporal del THOP final-concat: el cabezal entra tarde en la jerarquía",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_9_positional_v1_quartiles.png")


# ---------------------------------------------------------------------------
# Figure 6.10 — THOP metric comparison
# ---------------------------------------------------------------------------
# Centered values: TWO_HEAD_ORDINAL_DENSITY_V2_CENTERED_GAP05_...
# THOP values: TWO_HEAD_ORDINAL_POSITIONAL_V1_FINAL_CONCAT_RUN_20260527_ANALYSIS.md

THOP_COMPARISON = {
    "centered": {
        "yaml_parse": 0.0286,
        "level_exact": 0.0107,
        "line_f1": 0.0145,
        "prompt_f1": 0.0143,
        "kdv_score": 0.0238,
        "deep_exact": 0.2174,
        "deep_off1": 0.5652,
        "deep_compression": 0.6014,
    },
    "thop": {
        "yaml_parse": 0.0725,
        "level_exact": 0.0223,
        "line_f1": 0.0455,
        "prompt_f1": 0.0476,
        "kdv_score": 0.0507,
        "deep_exact": 0.1379,
        "deep_off1": 0.4966,
        "deep_compression": 0.6966,
    },
    "film": {
        # Corrected final evaluation values supplied from the completed manual audit.
        # The local metrics.json for this run is stale for yaml_parse and kdv_score.
        "yaml_parse": 0.2857,
        "level_exact": 0.0278,
        "line_f1": 0.0782,
        "prompt_f1": 0.0799,
        "kdv_score": 0.1250,
        "deep_exact": 0.1214,
        "deep_off1": 0.4143,
        "deep_compression": 0.7429,
    },
}


def fig_thop_metrics_comparison() -> None:
    labels = ["THO centered", "THOP final-concat"]
    colors = ["#FAA43A", "#7A68A6"]

    metric_groups = [
        (
            "Validez y contenido",
            [
                ("Parseo YAML", "yaml_parse"),
                ("Nivel exacto", "level_exact"),
                ("F1 línea", "line_f1"),
                ("KDV score", "kdv_score"),
            ],
        ),
        (
            "Niveles profundos",
            [
                ("Recall exacto\n5--8", "deep_exact"),
                ("Off-by-one\n5--8", "deep_off1"),
                ("Compresión\n5--8 a 0--4", "deep_compression"),
            ],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))

    for ax, (title, metrics) in zip(axes, metric_groups):
        x = np.arange(len(metrics))
        width = 0.34

        centered_vals = [THOP_COMPARISON["centered"][key] for _, key in metrics]
        thop_vals = [THOP_COMPARISON["thop"][key] for _, key in metrics]

        bars_a = ax.bar(
            x - width / 2,
            centered_vals,
            width,
            label=labels[0],
            color=colors[0],
            edgecolor="white",
            linewidth=0.6,
        )
        bars_b = ax.bar(
            x + width / 2,
            thop_vals,
            width,
            label=labels[1],
            color=colors[1],
            edgecolor="white",
            linewidth=0.6,
        )

        for bars in (bars_a, bars_b):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )

        ax.set_title(title, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _ in metrics], fontsize=8)
        ax.set_ylim(0, 0.78 if title == "Niveles profundos" else 0.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Valor de la métrica")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "THOP frente al THO centrado: mejora parcial de superficie, no solución estructural",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_10_thop_metrics_comparison.png")


# ---------------------------------------------------------------------------
# Figure 6.11 — Parser-facing progression with positional conditioning
# ---------------------------------------------------------------------------


def fig_positional_parser_progression() -> None:
    labels = ["THO\ncentrado", "THOP\nfinal-concat", "THOP-FiLM\nFiLM after 512"]
    colors = ["#FAA43A", "#7A68A6", C_FILM]

    metric_panels = [
        (
            "Parseo estructural",
            [
                ("Parseo YAML", "yaml_parse"),
                ("KDV score", "kdv_score"),
            ],
            0.34,
        ),
        (
            "Alineamiento de superficie",
            [
                ("F1 línea", "line_f1"),
                ("F1 requisitos", "prompt_f1"),
            ],
            0.12,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    model_keys = ["centered", "thop", "film"]

    for ax, (title, metrics, ymax) in zip(axes, metric_panels):
        x = np.arange(len(metrics))
        width = 0.24
        offsets = [-width, 0, width]

        for key, label, color, offset in zip(model_keys, labels, colors, offsets):
            values = [THOP_COMPARISON[key].get(metric_key, 0.0) for _, metric_key in metrics]
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.025,
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )

        ax.set_title(title, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _ in metrics], fontsize=8)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Valor de la métrica")
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle(
        "Progresión de métricas estructurales al introducir codificación posicional",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_11_positional_parser_progression.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating Chapter 6 figures → {OUT}\n")
    fig_architecture()
    fig_probe_recall_by_level()
    fig_train_loss_twohead()
    fig_comparison_v1()
    fig_level_distribution()
    fig_ordinal_variants()
    fig_threshold_drift()
    fig_structural_tradeoff()
    fig_positional_v1_quartiles()
    fig_thop_metrics_comparison()
    fig_positional_parser_progression()
    print(f"\nDone. 11 figures saved to {OUT.relative_to(ROOT)}")
