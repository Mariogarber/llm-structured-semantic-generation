"""Generate Chapter 5 figures for the TFM thesis.

Outputs 4 PNG figures to latex/figures/chapter5/:
  fig5_1_baseline_states.png    - Estado de predicciones baseline (validation split)
  fig5_2_training_loss.png      - Curva de pérdida SFT serializado
  fig5_3_metrics_comparison.png - Comparativa Baseline vs SFT (grouped bar chart)
  fig5_4_domain_errors.png      - Perfil de errores de dominio SFT

Usage:
    uv run python scripts/generate_ch5_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "figures" / "chapter5"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_LOG = (
    ROOT
    / "results"
    / "sft_kubernetes_v1"
    / "serialized-sft-a-v1-20260505-171226"
    / "train_log.jsonl"
)

# ---------------------------------------------------------------------------
# Palette / style
# ---------------------------------------------------------------------------

C_BASELINE = "#4878CF"   # blue
C_SFT = "#6ACC65"        # green
C_FAIL = "#E24A33"       # red
C_WARN = "#FAA43A"       # orange
C_OK = "#6ACC65"         # green (same as SFT)
C_SEC = "#E24A33"        # security errors
C_SEM = "#8EBA42"        # semantic errors

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
# Figure 5.1 — Baseline state distribution (validation split)
# ---------------------------------------------------------------------------
# Numbers derived from compact-validation70-320-vtfix/metrics.json:
#   evaluated_count = 65  →  5 rows failed structured output parse
#   yaml_parse_success_rate = 0.3077  →  0.3077 × 65 ≈ 20 valid YAML
#   65 − 20 = 45 rows: structured OK, YAML failed


def fig_baseline_states() -> None:
    labels = ["Parse\nestructurado\nfallido", "YAML\ninválido", "YAML\nválido"]
    counts = [5, 45, 20]
    colors = [C_FAIL, C_WARN, C_OK]

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )

    ax.set_ylabel("Número de muestras (n = 70)")
    ax.set_title(
        "Estado de las predicciones — baseline zero-shot\n(split de validación, 70 muestras)",
        pad=8,
    )
    ax.set_ylim(0, 58)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUT / "fig5_1_baseline_states.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ---------------------------------------------------------------------------
# Figure 5.2 — Training loss curve (serialized_sft)
# ---------------------------------------------------------------------------
# Epochs end at steps ≈ 53, 106, 159  (426 rows / grad_accum=8 = 53.25 steps/epoch)


def fig_training_loss() -> None:
    steps, losses = [], []
    with open(TRAIN_LOG, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line.strip())
            steps.append(rec["global_step"])
            losses.append(rec["loss"])

    epoch_steps = [53, 106, 159]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(steps, losses, color=C_SFT, linewidth=1.5, label="Pérdida de entrenamiento")

    for i, ep_step in enumerate(epoch_steps, 1):
        ax.axvline(ep_step, color="gray", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.text(
            ep_step + 1.5,
            max(losses) * 0.88,
            f"Época {i}",
            fontsize=8,
            color="#555555",
            va="top",
        )

    ax.set_xlabel("Paso global (global_step)")
    ax.set_ylabel("Pérdida (loss)")
    ax.set_title("Curva de pérdida — serialized_sft (LoRA, 3 épocas, 159 pasos)")
    ax.set_xlim(0, 163)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUT / "fig5_2_training_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ---------------------------------------------------------------------------
# Figure 5.3 — Grouped bar chart: Baseline vs SFT (key metrics)
# ---------------------------------------------------------------------------
# Metrics (validation split, both runs):
#   yaml_parse_success_rate:         0.3077  →  0.9857
#   average_level_exact_match_rate:  0.1131  →  0.7578
#   average_semantic_key_f1:         0.2778  →  0.9552
#   average_prompt_requirement_f1:   0.2213  →  0.8531


def fig_metrics_comparison() -> None:
    metric_labels = [
        "Parseo YAML\n(yaml_parse)",
        "Nivel exacto\n(level_exact)",
        "Clave semántica\n(semantic_key_f1)",
        "Requisito prompt\n(prompt_req_f1)",
    ]
    baseline_vals = [0.3077, 0.1131, 0.2778, 0.2213]
    sft_vals = [0.9857, 0.7578, 0.9552, 0.8531]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    b_base = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline zero-shot",
        color=C_BASELINE,
        edgecolor="white",
        linewidth=0.5,
    )
    b_sft = ax.bar(
        x + width / 2,
        sft_vals,
        width,
        label="Serialized SFT (val.)",
        color=C_SFT,
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in (b_base, b_sft):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.0%}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#333333",
            )

    ax.set_ylabel("Valor de la métrica (0 – 1)")
    ax.set_title(
        "Comparativa de métricas clave — Baseline vs. Serialized SFT\n(split de validación)",
        pad=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=8.8)
    ax.set_ylim(0, 1.14)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUT / "fig5_3_metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ---------------------------------------------------------------------------
# Figure 5.4 — Domain error profile (serialized_sft, validation split)
# ---------------------------------------------------------------------------
# From validation_metrics_recomputed.json > kubernetes_domain_error_counts


def fig_domain_errors() -> None:
    # Show top-5 errors; remaining (kubernetes_identity=1, yaml_parse=1) are omitted
    cat_raw = [
        "missing_resource_requirement",
        "missing_run_as_non_root",
        "missing_read_only_root_filesystem",
        "latest_image_tag",
        "volume_mount_without_volume",
    ]
    counts_raw = [261, 71, 71, 67, 5]

    # Spanish labels for readability in the thesis
    cat_labels = [
        "Sin límites de recursos\n(missing_resource_requirement)",
        "Contenedor como root\n(missing_run_as_non_root)",
        "FS raíz no read-only\n(missing_read_only_root_filesystem)",
        "Imagen con tag flotante\n(latest_image_tag)",
        "Volumen no declarado\n(volume_mount_without_volume)",
    ]
    colors_raw = [C_SEC, C_SEC, C_SEC, C_SEC, C_SEM]

    # Reverse so highest count is at top of horizontal chart
    cat_labels_rev = list(reversed(cat_labels))
    counts_rev = list(reversed(counts_raw))
    colors_rev = list(reversed(colors_raw))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.barh(
        cat_labels_rev,
        counts_rev,
        color=colors_rev,
        edgecolor="white",
        linewidth=0.5,
        height=0.52,
    )

    for bar, cnt in zip(bars, counts_rev):
        ax.text(
            bar.get_width() + 4,
            bar.get_y() + bar.get_height() / 2,
            str(cnt),
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xlabel("Instancias de error (acumuladas, 70 muestras del split de validación)")
    ax.set_title(
        "Perfil de errores de dominio — Serialized SFT\n(compuerta de validez Kubernetes, nivel 5)",
        pad=8,
    )
    ax.set_xlim(0, 310)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elems = [
        Patch(facecolor=C_SEC, label="Antipatrón de seguridad"),
        Patch(facecolor=C_SEM, label="Semántica de dominio"),
    ]
    ax.legend(handles=legend_elems, loc="lower right")

    fig.tight_layout()
    path = OUT / "fig5_4_domain_errors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating Chapter 5 figures → {OUT}\n")
    fig_baseline_states()
    fig_training_loss()
    fig_metrics_comparison()
    fig_domain_errors()
    print(f"\nDone. 4 figures saved to {OUT.relative_to(ROOT)}")
