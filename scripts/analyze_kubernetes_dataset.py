from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl, write_json


SEMANTIC_FIELDS = (
    "metadata",
    "spec",
    "containers",
    "image",
    "ports",
    "env",
    "volumes",
    "volumeMounts",
    "selector",
    "template",
    "data",
    "rules",
    "subjects",
    "roleRef",
)


TERMINOLOGY = (
    {
        "term": "primary_kind",
        "definition": "First Kubernetes kind found among parsed mapping documents in a sample.",
    },
    {
        "term": "yaml_max_depth",
        "definition": "Maximum recursive depth of the parsed YAML object tree. This is not the same as block level.",
    },
    {
        "term": "yaml_total_nodes",
        "definition": "Total parsed YAML mapping, list, and scalar nodes. This does not mean Kubernetes Node resources.",
    },
    {
        "term": "block_count",
        "definition": "Number of line-and-level blocks derived from normalized YAML.",
    },
    {
        "term": "level",
        "definition": "Indentation level of a block, using two spaces per level in v1.",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exploratory analysis report for Kubernetes v1."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "kubernetes_v1",
        help="Directory containing Kubernetes v1 processed artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset_analysis_kubernetes_v1",
        help="Directory where the report and figures will be written.",
    )
    parser.add_argument("--top-kinds", type=int, default=15)
    return parser.parse_args()


def load_yaml_documents(yaml_text: str) -> list[Any]:
    return list(yaml.safe_load_all(yaml_text))


def walk_mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                keys.add(str(key))
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return keys


def build_resource_rows(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in samples.to_dict(orient="records"):
        documents = load_yaml_documents(row["target_yaml_normalized"])
        all_keys = set()
        for document_index, document in enumerate(documents):
            document_keys = walk_mapping_keys(document)
            all_keys.update(document_keys)
            if isinstance(document, dict):
                kind = str(document.get("kind", "<missing>"))
                api_version = str(document.get("apiVersion", "<missing>"))
            else:
                kind = "<non_mapping>"
                api_version = "<non_mapping>"
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "document_index": document_index,
                    "kind": kind,
                    "apiVersion": api_version,
                    "yaml_max_depth": int(row["yaml_max_depth"]),
                    "yaml_total_nodes": int(row["yaml_total_nodes"]),
                    "question_word_count": int(row["question_word_count"]),
                    "question_simplified_word_count": int(row["question_simplified_word_count"]),
                    "target_yaml_char_count": int(row["target_yaml_char_count"]),
                    "leakage_reasons": row["leakage_reasons"],
                    "duplicate_yaml_group_size": int(row["duplicate_yaml_group_size"]),
                    "keys": sorted(document_keys),
                }
            )
        for field in SEMANTIC_FIELDS:
            row[field] = field in all_keys
    return pd.DataFrame(rows)


def build_sample_features(samples: pd.DataFrame, structural_rows: list[dict[str, Any]]) -> pd.DataFrame:
    structural_by_sample: dict[str, dict[str, Any]] = {}
    for row in structural_rows:
        if row["prompt_variant"] != "question":
            continue
        levels = [block["level"] for block in row["blocks"]]
        structural_by_sample[row["sample_id"]] = {
            "block_count": int(row["block_count"]),
            "max_block_level": max(levels) if levels else 0,
        }

    feature_rows: list[dict[str, Any]] = []
    for row in samples.to_dict(orient="records"):
        documents = load_yaml_documents(row["target_yaml_normalized"])
        kinds = [
            str(document.get("kind", "<missing>"))
            for document in documents
            if isinstance(document, dict)
        ]
        keys = set()
        for document in documents:
            keys.update(walk_mapping_keys(document))

        structural = structural_by_sample.get(row["sample_id"], {"block_count": 0, "max_block_level": 0})
        feature_row = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "primary_kind": kinds[0] if kinds else "<missing>",
            "resource_count": len(documents),
            "question_word_count": int(row["question_word_count"]),
            "question_simplified_word_count": int(row["question_simplified_word_count"]),
            "prompt_pair_similarity": float(row["prompt_pair_similarity"]),
            "yaml_max_depth": int(row["yaml_max_depth"]),
            "yaml_mapping_nodes": int(row["yaml_mapping_nodes"]),
            "yaml_list_nodes": int(row["yaml_list_nodes"]),
            "yaml_scalar_nodes": int(row["yaml_scalar_nodes"]),
            "yaml_total_nodes": int(row["yaml_total_nodes"]),
            "target_yaml_char_count": int(row["target_yaml_char_count"]),
            "leakage_reasons": row["leakage_reasons"],
            "duplicate_yaml_group_size": int(row["duplicate_yaml_group_size"]),
            **structural,
        }
        for field in SEMANTIC_FIELDS:
            feature_row[field] = field in keys
        feature_rows.append(feature_row)
    return pd.DataFrame(feature_rows)


def build_level_rows(structural_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in structural_rows:
        if row["prompt_variant"] != "question":
            continue
        for block in row["blocks"]:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "level": int(block["level"]),
                }
            )
    return pd.DataFrame(rows)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def save_barplot(data: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=data.values, y=data.index, ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_histograms(sample_features: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    columns = [
        ("yaml_max_depth", "YAML max depth"),
        ("yaml_total_nodes", "YAML total nodes"),
        ("block_count", "Structural block count"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, title) in zip(axes, columns):
        sns.histplot(sample_features[column], kde=False, bins=18, ax=ax, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("samples")
    fig.tight_layout()
    path = figures_dir / "complexity_histograms.png"
    fig.savefig(path)
    plt.close(fig)
    output["complexity_histograms"] = path.name

    fig, ax = plt.subplots(figsize=(8, 5))
    top_kinds = sample_features["primary_kind"].value_counts().head(12).index
    subset = sample_features[sample_features["primary_kind"].isin(top_kinds)]
    sns.boxplot(data=subset, y="primary_kind", x="yaml_total_nodes", ax=ax, color="#72B7B2")
    ax.set_title("YAML complexity by primary kind")
    ax.set_xlabel("yaml_total_nodes")
    ax.set_ylabel("primary_kind")
    fig.tight_layout()
    path = figures_dir / "complexity_by_kind_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    output["complexity_by_kind_boxplot"] = path.name
    return output


def plot_prompt_figures(sample_features: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    prompt_lengths = sample_features[
        ["sample_id", "question_word_count", "question_simplified_word_count"]
    ].melt(
        id_vars="sample_id",
        var_name="prompt_variant",
        value_name="word_count",
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(
        data=prompt_lengths,
        x="word_count",
        hue="prompt_variant",
        bins=24,
        alpha=0.55,
        ax=ax,
    )
    ax.set_title("Prompt length distribution")
    ax.set_xlabel("word count")
    ax.set_ylabel("samples")
    fig.tight_layout()
    path = figures_dir / "prompt_length_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    output["prompt_length_distribution"] = path.name

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=sample_features,
        x="question_word_count",
        y="yaml_total_nodes",
        hue="split",
        ax=ax,
    )
    ax.set_title("Prompt length vs YAML complexity")
    ax.set_xlabel("question word count")
    ax.set_ylabel("yaml_total_nodes")
    fig.tight_layout()
    path = figures_dir / "prompt_length_vs_yaml_nodes.png"
    fig.savefig(path)
    plt.close(fig)
    output["prompt_length_vs_yaml_nodes"] = path.name

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(sample_features["prompt_pair_similarity"], bins=24, ax=ax, color="#F58518")
    ax.set_title("Original vs simplified prompt similarity")
    ax.set_xlabel("Sequence similarity")
    ax.set_ylabel("samples")
    fig.tight_layout()
    path = figures_dir / "prompt_pair_similarity.png"
    fig.savefig(path)
    plt.close(fig)
    output["prompt_pair_similarity"] = path.name
    return output


def plot_split_figures(sample_features: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.countplot(data=sample_features, x="split", hue="leakage_reasons", ax=axes[0])
    axes[0].set_title("Leakage reasons by split")
    axes[0].set_xlabel("split")
    axes[0].set_ylabel("samples")
    axes[0].legend(title="leakage")
    sns.boxplot(data=sample_features, x="split", y="yaml_total_nodes", ax=axes[1], color="#72B7B2")
    axes[1].set_title("YAML nodes by split")
    axes[1].set_xlabel("split")
    axes[1].set_ylabel("yaml_total_nodes")
    fig.tight_layout()
    path = figures_dir / "split_balance_and_leakage.png"
    fig.savefig(path)
    plt.close(fig)
    output["split_balance_and_leakage"] = path.name

    kind_split = pd.crosstab(sample_features["primary_kind"], sample_features["split"])
    kind_split = kind_split.loc[kind_split.sum(axis=1).sort_values(ascending=False).head(15).index]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(kind_split, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Top primary kinds by split")
    ax.set_xlabel("split")
    ax.set_ylabel("primary_kind")
    fig.tight_layout()
    path = figures_dir / "kind_by_split_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    output["kind_by_split_heatmap"] = path.name
    return output


def plot_structural_figures(
    sample_features: pd.DataFrame,
    level_rows: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, str]:
    output: dict[str, str] = {}
    level_counts = level_rows["level"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=level_counts.index, y=level_counts.values, ax=ax, color="#54A24B")
    ax.set_title("Line-and-level target distribution")
    ax.set_xlabel("level")
    ax.set_ylabel("blocks")
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)
    fig.tight_layout()
    path = figures_dir / "level_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    output["level_distribution"] = path.name

    depth_bins = pd.crosstab(sample_features["primary_kind"], sample_features["yaml_max_depth"])
    depth_bins = depth_bins.loc[depth_bins.sum(axis=1).sort_values(ascending=False).head(12).index]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(depth_bins, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title("Primary kind by YAML max depth")
    ax.set_xlabel("yaml_max_depth")
    ax.set_ylabel("primary_kind")
    fig.tight_layout()
    path = figures_dir / "kind_depth_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    output["kind_depth_heatmap"] = path.name
    return output


def plot_semantic_figures(sample_features: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    field_presence = sample_features[list(SEMANTIC_FIELDS)].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=field_presence.values * 100, y=field_presence.index, ax=ax, color="#B279A2")
    ax.set_title("Approximate semantic field coverage")
    ax.set_xlabel("% samples containing field")
    ax.set_ylabel("field")
    fig.tight_layout()
    path = figures_dir / "semantic_field_presence.png"
    fig.savefig(path)
    plt.close(fig)
    output["semantic_field_presence"] = path.name

    binary = sample_features[list(SEMANTIC_FIELDS)].astype(int)
    cooccurrence = binary.T.dot(binary)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cooccurrence, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set_title("Semantic field co-occurrence")
    fig.tight_layout()
    path = figures_dir / "semantic_field_cooccurrence.png"
    fig.savefig(path)
    plt.close(fig)
    output["semantic_field_cooccurrence"] = path.name
    return output


def image_tag(figures_dir: Path, filename: str) -> str:
    data = (figures_dir / filename).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    alt = html.escape(filename.removesuffix(".png").replace("_", " "))
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}" loading="lazy">'


def describe_numeric(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def build_summary(
    sample_features: pd.DataFrame,
    resource_rows: pd.DataFrame,
    level_rows: pd.DataFrame,
) -> dict[str, Any]:
    field_presence = {
        field: round(float(sample_features[field].mean()), 4)
        for field in SEMANTIC_FIELDS
    }
    duplicate_samples = sample_features[sample_features["leakage_reasons"] == "exact_yaml_duplicate"]
    observations = [
        "The dataset is Kubernetes-centered and ready for descriptive analysis before training.",
        "Coverage is uneven across resource kinds; top kinds should be considered separately in evaluation.",
        "Most samples contain a single Kubernetes resource, so multi-document behavior is underrepresented.",
        "Exact YAML duplicate groups exist and should remain grouped when interpreting train/test results.",
        "Structural target levels are concentrated in shallow-to-medium depths, with a small deep tail.",
    ]
    return {
        "sample_count": int(len(sample_features)),
        "resource_document_count": int(len(resource_rows)),
        "split_counts": sample_features["split"].value_counts().to_dict(),
        "resource_count_per_sample": sample_features["resource_count"].value_counts().sort_index().to_dict(),
        "top_kinds": resource_rows["kind"].value_counts().head(15).to_dict(),
        "top_api_versions": resource_rows["apiVersion"].value_counts().head(10).to_dict(),
        "leakage_reasons": sample_features["leakage_reasons"].value_counts().to_dict(),
        "exact_yaml_duplicate_sample_count": int(len(duplicate_samples)),
        "prompt_word_count": describe_numeric(sample_features["question_word_count"]),
        "simplified_prompt_word_count": describe_numeric(sample_features["question_simplified_word_count"]),
        "prompt_pair_similarity": describe_numeric(sample_features["prompt_pair_similarity"]),
        "yaml_max_depth": describe_numeric(sample_features["yaml_max_depth"]),
        "yaml_total_nodes": describe_numeric(sample_features["yaml_total_nodes"]),
        "block_count": describe_numeric(sample_features["block_count"]),
        "level_distribution": level_rows["level"].value_counts().sort_index().to_dict(),
        "semantic_field_presence": field_presence,
        "terminology": list(TERMINOLOGY),
        "observations": observations,
        "limitations": [
            "Bias here means dataset coverage bias, not social bias.",
            "Semantic fields are approximate recursive key-presence signals, not full Kubernetes schema validation.",
            "Primary kind uses the first document in a sample; multi-document samples are also counted in resource-level tables.",
        ],
    }


def write_markdown_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    top_kinds = ", ".join(f"{kind} ({count})" for kind, count in list(summary["top_kinds"].items())[:8])
    split_counts = json.dumps(summary["split_counts"], sort_keys=True)
    lines = [
        "# Kubernetes v1 Dataset Analysis",
        "",
        "This report is descriptive and is intended to inspect dataset coverage before baseline, SFT, or DPO.",
        "",
        "## Key Numbers",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Kubernetes resource documents: {summary['resource_document_count']}",
        f"- Splits: `{split_counts}`",
        f"- Top resource kinds: {top_kinds}",
        f"- Exact YAML duplicate samples: {summary['exact_yaml_duplicate_sample_count']}",
        "",
        "## Main Observations",
        "",
        *[f"- {item}" for item in summary["observations"]],
        "",
        "## Terminology",
        "",
        *[f"- `{item['term']}`: {item['definition']}" for item in summary["terminology"]],
        "",
        "## Outputs",
        "",
        "- `dataset_analysis_report.html`: navigable report with embedded figures.",
        "- `figures/*.png`: static figures for the thesis or slides.",
        "- `dataset_analysis_summary.json`: machine-readable summary.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_html_report(output_dir: Path, figures: dict[str, str], summary: dict[str, Any]) -> None:
    figures_dir = output_dir / "figures"
    cards = []
    ordered = [
        ("kind_counts", "Resource kind distribution"),
        ("api_version_counts", "API version distribution"),
        ("complexity_histograms", "Structural complexity histograms"),
        ("complexity_by_kind_boxplot", "Complexity by primary kind"),
        ("level_distribution", "Line-and-level distribution"),
        ("kind_depth_heatmap", "Primary kind by YAML depth"),
        ("prompt_length_distribution", "Prompt length distribution"),
        ("prompt_length_vs_yaml_nodes", "Prompt length vs YAML complexity"),
        ("prompt_pair_similarity", "Prompt variant similarity"),
        ("split_balance_and_leakage", "Split balance and leakage"),
        ("kind_by_split_heatmap", "Kind by split"),
        ("semantic_field_presence", "Semantic field presence"),
        ("semantic_field_cooccurrence", "Semantic field co-occurrence"),
    ]
    for key, title in ordered:
        filename = figures.get(key)
        if filename:
            cards.append(
                f"<section><h2>{html.escape(title)}</h2>{image_tag(figures_dir, filename)}</section>"
            )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kubernetes v1 Dataset Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2933; }}
    header {{ max-width: 960px; }}
    section {{ margin: 2rem 0; max-width: 1100px; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    code, pre {{ background: #f0f4f8; padding: 0.1rem 0.25rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; max-width: 900px; }}
    .metric {{ border: 1px solid #d9e2ec; padding: 1rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Kubernetes v1 Dataset Analysis</h1>
    <p>Descriptive analysis before baseline/SFT. Bias means dataset coverage bias, not social bias.</p>
    <div class="grid">
      <div class="metric"><strong>Samples</strong><br>{summary['sample_count']}</div>
      <div class="metric"><strong>Resource docs</strong><br>{summary['resource_document_count']}</div>
      <div class="metric"><strong>Duplicate YAML samples</strong><br>{summary['exact_yaml_duplicate_sample_count']}</div>
      <div class="metric"><strong>Median blocks</strong><br>{summary['block_count']['median']:.1f}</div>
    </div>
  </header>
  <section>
    <h2>Main observations</h2>
    <ul>
      {''.join(f'<li>{html.escape(item)}</li>' for item in summary['observations'])}
    </ul>
  </section>
  <section>
    <h2>Terminology</h2>
    <ul>
      {''.join(f"<li><code>{html.escape(item['term'])}</code>: {html.escape(item['definition'])}</li>" for item in summary['terminology'])}
    </ul>
  </section>
  {''.join(cards)}
</body>
</html>
"""
    (output_dir / "dataset_analysis_report.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    samples = pd.read_csv(processed_dir / "dataset_manifest_samples.csv")
    structural_rows = read_jsonl(processed_dir / "dataset_structural_targets.jsonl")
    sample_features = build_sample_features(samples, structural_rows)
    resource_rows = build_resource_rows(samples)
    level_rows = build_level_rows(structural_rows)

    figures: dict[str, str] = {}
    kind_counts = resource_rows["kind"].value_counts().head(args.top_kinds)
    kind_path = figures_dir / "kind_counts.png"
    save_barplot(kind_counts, "Kubernetes resource kind distribution", "resource documents", "kind", kind_path)
    figures["kind_counts"] = kind_path.name

    api_counts = resource_rows["apiVersion"].value_counts().head(12)
    api_path = figures_dir / "api_version_counts.png"
    save_barplot(api_counts, "Kubernetes apiVersion distribution", "resource documents", "apiVersion", api_path)
    figures["api_version_counts"] = api_path.name

    figures.update(plot_histograms(sample_features, figures_dir))
    figures.update(plot_structural_figures(sample_features, level_rows, figures_dir))
    figures.update(plot_prompt_figures(sample_features, figures_dir))
    figures.update(plot_split_figures(sample_features, figures_dir))
    figures.update(plot_semantic_figures(sample_features, figures_dir))

    sample_features.to_csv(output_dir / "dataset_analysis_sample_features.csv", index=False)
    resource_rows.drop(columns=["keys"]).to_csv(output_dir / "dataset_analysis_resource_rows.csv", index=False)

    summary = build_summary(sample_features, resource_rows, level_rows)
    summary["figures"] = figures
    write_json(output_dir / "dataset_analysis_summary.json", summary)
    write_markdown_readme(output_dir, summary)
    write_html_report(output_dir, figures, summary)
    print(
        {
            "output_dir": str(output_dir),
            "sample_count": summary["sample_count"],
            "resource_document_count": summary["resource_document_count"],
            "figure_count": len(figures),
        }
    )


if __name__ == "__main__":
    main()
