from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.dataset_io import read_jsonl


DEFAULT_OUTPUT_NAME = "baseline_dashboard.html"
SUMMARY_METRICS = [
    ("structured_output_parse_success_rate", "Structured output parse"),
    ("yaml_parse_success_rate", "YAML parse"),
    ("parsed_equal_rate", "Parsed equality"),
    ("block_parse_success_rate", "Parser control"),
    ("average_content_exact_match_rate", "Content exact match"),
    ("average_level_exact_match_rate", "Level exact match"),
    ("average_semantic_key_f1", "Semantic key F1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an interactive self-contained HTML dashboard for a baseline run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing baseline run artifacts such as config.json and predictions.jsonl.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional output HTML path. Defaults to <run-dir>/baseline_dashboard.html.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path, allow_truncated_last_line=False)


def format_metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return f"{value * 100:.1f}%"
        return f"{value:.4f}"
    return str(value)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def truncate_text(value: str, *, limit: int = 120) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def classify_error(error_text: str) -> str:
    if error_text.startswith("block_"):
        parts = error_text.split(":", 1)
        return parts[1] if len(parts) > 1 else error_text
    if error_text.startswith("json_block_parse_error:"):
        return "json_block_parse_error"
    return error_text.split(":", 1)[0]


def build_summary_cards(
    *,
    config: dict[str, Any] | None,
    state: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    cards: list[tuple[str, str]] = []
    run_id = None if config is None else config.get("run_id")
    cards.append(("Run ID", str(run_id or "unknown")))
    cards.append(("Predictions", format_metric_value(len(predictions))))
    evaluated = None if metrics is None else metrics.get("evaluated_count")
    cards.append(("Evaluated rows", format_metric_value(evaluated if evaluated is not None else len(predictions))))
    cards.append(("Latent rows", format_metric_value(len(latent_rows))))
    if config is not None:
        cards.append(("Split", str(config.get("split", "unknown"))))
        cards.append(("Output format", str(config.get("output_format", "unknown"))))
        cards.append(("Recovery mode", str(config.get("recovery_mode", "unknown"))))
        cards.append(("Collect latent means", format_metric_value(config.get("collect_latent_means"))))
    if state is not None:
        run_completed = state.get("status") == "completed" or bool(state.get("completed")) or bool(state.get("completed_at"))
        cards.append(("Run completed", format_metric_value(run_completed)))
    return cards


def build_metric_table(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "<p class='empty-state'>No metrics.json found.</p>"

    rows = []
    for key, label in SUMMARY_METRICS:
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(format_metric_value(metrics.get(key)))}</td>"
            "</tr>"
        )
    return (
        "<table class='metric-table'>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def build_error_summary(predictions: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser_counter: Counter[str] = Counter()
    sample_counter: Counter[str] = Counter()

    for prediction in predictions:
        errors = list(prediction.get("parser_errors") or [])
        evaluation = prediction.get("evaluation") or {}
        errors.extend(evaluation.get("errors") or [])
        normalized_errors = {classify_error(error) for error in errors}
        for error in normalized_errors:
            sample_counter[error] += 1
        for error in errors:
            parser_counter[classify_error(error)] += 1

    parser_df = pd.DataFrame(
        [{"error_type": key, "occurrences": value} for key, value in parser_counter.most_common()]
    )
    sample_df = pd.DataFrame(
        [{"error_type": key, "samples": value} for key, value in sample_counter.most_common()]
    )
    return parser_df, sample_df


def make_error_distribution_figure(predictions: list[dict[str, Any]]) -> str:
    parser_df, sample_df = build_error_summary(predictions)
    figure = go.Figure()

    if not parser_df.empty:
        figure.add_trace(
            go.Bar(
                x=parser_df["error_type"],
                y=parser_df["occurrences"],
                name="Occurrences",
                marker_color="#c2410c",
            )
        )
    if not sample_df.empty:
        figure.add_trace(
            go.Bar(
                x=sample_df["error_type"],
                y=sample_df["samples"],
                name="Samples affected",
                marker_color="#2563eb",
            )
        )

    figure.update_layout(
        title="Error distribution",
        barmode="group",
        margin=dict(l=24, r=24, t=56, b=140),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="", tickangle=-28),
        yaxis=dict(title="Count"),
    )

    if parser_df.empty and sample_df.empty:
        figure.add_annotation(
            text="No parser or evaluation errors found in predictions.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#334155"),
        )
    return pio.to_html(figure, include_plotlyjs=False, full_html=False, config={"responsive": True})


def prediction_status(prediction: dict[str, Any]) -> str:
    evaluation = prediction.get("evaluation")
    if not evaluation:
        return "not_evaluated"
    if evaluation.get("parsed_equal_to_reference"):
        return "parsed_equal"
    if evaluation.get("yaml_parse_ok"):
        return "yaml_parse_ok"
    return "parser_or_yaml_failure"


def build_samples(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        evaluation = prediction.get("evaluation") or {}
        parser_errors = list(prediction.get("parser_errors") or [])
        evaluation_errors = list(evaluation.get("errors") or [])
        sample = {
            "row_index": index,
            "unit_id": prediction.get("unit_id") or f"{prediction.get('sample_id', 'sample')}::{prediction.get('prompt_variant', 'variant')}",
            "sample_id": prediction.get("sample_id", ""),
            "prompt_variant": prediction.get("prompt_variant", ""),
            "split": prediction.get("split", ""),
            "status": prediction_status(prediction),
            "generated_token_count": prediction.get("generated_token_count"),
            "predicted_block_count": len(prediction.get("predicted_blocks") or []),
            "parser_error_count": len(parser_errors),
            "evaluation_error_count": len(evaluation_errors),
            "prompt_preview": truncate_text(str(prediction.get("prompt_text", ""))),
            "yaml_parse_ok": evaluation.get("yaml_parse_ok"),
            "parsed_equal_to_reference": evaluation.get("parsed_equal_to_reference"),
            "content_exact_match_rate": evaluation.get("content_exact_match_rate"),
            "level_exact_match_rate": evaluation.get("level_exact_match_rate"),
            "raw_model_output": prediction.get("raw_model_output", ""),
            "reconstructed_yaml": prediction.get("reconstructed_yaml", ""),
            "prompt_text": prediction.get("prompt_text", ""),
            "parser_errors": parser_errors,
            "evaluation": evaluation,
            "predicted_blocks_json": compact_json(prediction.get("predicted_blocks") or []),
            "evaluation_json": compact_json(evaluation),
        }
        samples.append(sample)
    return samples


def build_projection_dataframe(
    predictions: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if not latent_rows:
        return pd.DataFrame()

    prediction_by_unit: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        sample_id = prediction.get("sample_id")
        prompt_variant = prediction.get("prompt_variant")
        unit_id = prediction.get("unit_id") or f"{sample_id}::{prompt_variant}"
        prediction_by_unit[unit_id] = prediction

    records: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    for latent_row in latent_rows:
        vector = latent_row.get("latent_mean")
        if not vector:
            continue
        unit_id = latent_row.get("unit_id") or f"{latent_row.get('sample_id')}::{latent_row.get('prompt_variant')}"
        prediction = prediction_by_unit.get(unit_id, {})
        evaluation = prediction.get("evaluation") or {}
        records.append(
            {
                "unit_id": unit_id,
                "sample_id": latent_row.get("sample_id", ""),
                "prompt_variant": latent_row.get("prompt_variant", ""),
                "split": latent_row.get("split", ""),
                "generated_token_count": latent_row.get("generated_token_count"),
                "status": prediction_status(prediction),
                "yaml_parse_ok": str(evaluation.get("yaml_parse_ok")),
                "parsed_equal_to_reference": str(evaluation.get("parsed_equal_to_reference")),
                "content_exact_match_rate": evaluation.get("content_exact_match_rate"),
                "level_exact_match_rate": evaluation.get("level_exact_match_rate"),
            }
        )
        vectors.append(vector)

    if len(vectors) < 2:
        return pd.DataFrame()

    components = PCA(n_components=2).fit_transform(vectors)
    projection = pd.DataFrame(records)
    projection["x"] = components[:, 0]
    projection["y"] = components[:, 1]
    return projection


def make_projection_figure(predictions: list[dict[str, Any]], latent_rows: list[dict[str, Any]]) -> str:
    projection = build_projection_dataframe(predictions, latent_rows)
    if projection.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="Latent projection unavailable. Need at least two rows with latent_mean vectors.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#334155"),
        )
        figure.update_layout(
            title="Latent 2D projection",
            margin=dict(l=24, r=24, t=56, b=24),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return pio.to_html(figure, include_plotlyjs=False, full_html=False, config={"responsive": True})

    figure = px.scatter(
        projection,
        x="x",
        y="y",
        color="status",
        symbol="split",
        hover_data={
            "sample_id": True,
            "prompt_variant": True,
            "generated_token_count": True,
            "content_exact_match_rate": ":.3f",
            "level_exact_match_rate": ":.3f",
            "yaml_parse_ok": True,
            "parsed_equal_to_reference": True,
            "x": False,
            "y": False,
        },
        color_discrete_sequence=["#2563eb", "#0f766e", "#c2410c", "#7c3aed"],
        title="Latent 2D projection (PCA)",
    )
    figure.update_traces(marker=dict(size=10, line=dict(width=0.5, color="#ffffff")))
    figure.update_layout(
        margin=dict(l=24, r=24, t=56, b=24),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="PC1",
        yaxis_title="PC2",
    )
    return pio.to_html(figure, include_plotlyjs=False, full_html=False, config={"responsive": True})


def render_config_panel(config: dict[str, Any] | None, state: dict[str, Any] | None, metrics: dict[str, Any] | None) -> str:
    payload = {
        "config.json": config,
        "state.json": state,
        "metrics.json": metrics,
    }
    blocks = []
    for label, value in payload.items():
        blocks.append(
            "<div class='artifact-block'>"
            f"<h3>{html.escape(label)}</h3>"
            f"<pre>{html.escape(compact_json(value) if value is not None else 'Missing')}</pre>"
            "</div>"
        )
    return "".join(blocks)


def build_warnings(
    *,
    config: dict[str, Any] | None,
    state: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if config is None:
        warnings.append("Missing config.json. The dashboard is using only the artifacts it could infer from the run directory.")
    if metrics is None:
        warnings.append("Missing metrics.json. Global metric cards and tables may be partial.")
    if state is None:
        warnings.append("Missing state.json. Completion status cannot be confirmed from persisted state.")
    elif not (state.get("status") == "completed" or bool(state.get("completed")) or bool(state.get("completed_at"))):
        warnings.append("state.json marks the run as incomplete. Metrics and samples may reflect an interrupted execution.")
    if not predictions:
        warnings.append("Missing predictions.jsonl or it is empty. The sample table and error plots will stay mostly empty.")
    if config is not None and config.get("dry_run"):
        warnings.append("This run is marked as dry-run. No real generations are expected.")
    if config is not None and config.get("collect_latent_means") and not latent_rows:
        warnings.append("The run expected latent mean collection, but latent_mean_vectors.jsonl was not found.")
    return warnings


def render_html(
    *,
    run_dir: Path,
    output_html: Path,
    config: dict[str, Any] | None,
    state: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
) -> str:
    summary_cards = build_summary_cards(
        config=config,
        state=state,
        metrics=metrics,
        predictions=predictions,
        latent_rows=latent_rows,
    )
    warnings = build_warnings(
        config=config,
        state=state,
        metrics=metrics,
        predictions=predictions,
        latent_rows=latent_rows,
    )
    samples = build_samples(predictions)
    error_distribution_html = make_error_distribution_figure(predictions)
    projection_html = make_projection_figure(predictions, latent_rows)
    metric_table_html = build_metric_table(metrics)
    config_panel_html = render_config_panel(config, state, metrics)
    samples_json = json.dumps(samples, ensure_ascii=False)

    summary_cards_html = "".join(
        "<div class='summary-card'>"
        f"<span class='summary-label'>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        "</div>"
        for label, value in summary_cards
    )
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Baseline dashboard - {html.escape(run_dir.name)}</title>
  <script>{get_plotlyjs()}</script>
  <style>
    :root {{
      --bg: #f1f5f9;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --border: #dbe4ee;
      --text: #0f172a;
      --muted: #475569;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --warn: #c2410c;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      --mono: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      --sans: "Segoe UI", "Trebuchet MS", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.08), transparent 22%),
        var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }}
    .page {{
      width: min(1440px, calc(100vw - 32px));
      margin: 24px auto 40px;
      display: grid;
      gap: 20px;
    }}
    .hero {{
      background: linear-gradient(135deg, #0f172a, #134e4a);
      color: #f8fafc;
      padding: 24px;
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.8rem, 2vw + 1rem, 3rem);
    }}
    .hero p {{
      margin: 0;
      color: rgba(248, 250, 252, 0.85);
      max-width: 900px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .summary-card {{
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 18px;
      padding: 14px 16px;
      backdrop-filter: blur(10px);
    }}
    .summary-label {{
      display: block;
      color: rgba(248, 250, 252, 0.72);
      font-size: 0.85rem;
      margin-bottom: 6px;
    }}
    .summary-card strong {{
      font-size: 1.1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 1.25rem;
    }}
    .warning-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--warn);
    }}
    .two-column {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .metric-table th,
    .metric-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    .metric-table thead {{
      background: var(--panel-soft);
    }}
    .artifact-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .artifact-block {{
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
      min-width: 0;
    }}
    .artifact-block h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 0.86rem;
      color: var(--muted);
    }}
    .table-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .table-toolbar input,
    .table-toolbar select {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      background: var(--panel-soft);
      color: var(--text);
      min-width: 180px;
    }}
    .table-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 20px;
      align-items: start;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel-soft);
      min-height: 520px;
      max-height: 760px;
    }}
    table.sample-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
    }}
    .sample-table th,
    .sample-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    .sample-table thead {{
      position: sticky;
      top: 0;
      background: #e2e8f0;
      z-index: 1;
    }}
    .sample-table th {{
      cursor: pointer;
      user-select: none;
    }}
    .sample-table tbody tr {{
      cursor: pointer;
    }}
    .sample-table tbody tr:hover {{
      background: #e0f2fe;
    }}
    .sample-table tbody tr.selected {{
      background: #ccfbf1;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .badge.parsed_equal {{ background: #dcfce7; color: #166534; }}
    .badge.yaml_parse_ok {{ background: #dbeafe; color: #1d4ed8; }}
    .badge.parser_or_yaml_failure {{ background: #ffedd5; color: #c2410c; }}
    .badge.not_evaluated {{ background: #e2e8f0; color: #334155; }}
    .detail-panel {{
      position: sticky;
      top: 20px;
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      min-height: 520px;
      max-height: 760px;
      overflow: auto;
    }}
    .detail-panel h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .detail-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .detail-meta div {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px;
    }}
    .detail-meta span {{
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      margin-bottom: 4px;
    }}
    .detail-section {{
      margin-top: 14px;
    }}
    .detail-section ul {{
      margin: 8px 0 0;
      padding-left: 18px;
    }}
    .detail-section pre {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      max-height: 240px;
      overflow: auto;
    }}
    .empty-state {{
      color: var(--muted);
      margin: 0;
    }}
    @media (max-width: 1120px) {{
      .two-column,
      .table-layout {{
        grid-template-columns: 1fr;
      }}
      .detail-panel {{
        position: static;
        max-height: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Baseline Results Dashboard</h1>
      <p>
        Self-contained report for <strong>{html.escape(run_dir.name)}</strong>.
        Source run: <code>{html.escape(str(run_dir))}</code>.
        Output: <code>{html.escape(str(output_html))}</code>.
      </p>
      <div class="summary-grid">{summary_cards_html}</div>
    </section>

    <section class="panel">
      <h2>Run notes</h2>
      {"<ul class='warning-list'>" + warning_html + "</ul>" if warnings else "<p class='empty-state'>No warnings detected from the available artifacts.</p>"}
    </section>

    <section class="two-column">
      <div class="panel">
        <h2>Global metrics</h2>
        {metric_table_html}
      </div>
      <div class="panel">
        <h2>Artifacts snapshot</h2>
        <div class="artifact-grid">{config_panel_html}</div>
      </div>
    </section>

    <section class="two-column">
      <div class="panel">
        {error_distribution_html}
      </div>
      <div class="panel">
        {projection_html}
      </div>
    </section>

    <section class="panel">
      <h2>Samples explorer</h2>
      <div class="table-toolbar">
        <input id="sample-search" type="search" placeholder="Filter by sample id, prompt or unit id">
        <select id="status-filter">
          <option value="all">All statuses</option>
          <option value="parsed_equal">parsed_equal</option>
          <option value="yaml_parse_ok">yaml_parse_ok</option>
          <option value="parser_or_yaml_failure">parser_or_yaml_failure</option>
          <option value="not_evaluated">not_evaluated</option>
        </select>
        <select id="split-filter">
          <option value="all">All splits</option>
          <option value="validation">validation</option>
          <option value="test">test</option>
        </select>
      </div>
      <div class="table-layout">
        <div class="table-wrap">
          <table class="sample-table">
            <thead>
              <tr>
                <th data-sort-key="sample_id">Sample</th>
                <th data-sort-key="prompt_variant">Variant</th>
                <th data-sort-key="split">Split</th>
                <th data-sort-key="status">Status</th>
                <th data-sort-key="generated_token_count">Tokens</th>
                <th data-sort-key="predicted_block_count">Blocks</th>
                <th data-sort-key="parser_error_count">Parser errors</th>
                <th data-sort-key="content_exact_match_rate">Content match</th>
                <th data-sort-key="level_exact_match_rate">Level match</th>
                <th data-sort-key="prompt_preview">Prompt preview</th>
              </tr>
            </thead>
            <tbody id="sample-table-body"></tbody>
          </table>
        </div>
        <aside class="detail-panel" id="detail-panel">
          <p class="empty-state">Select a sample row to inspect prompt, outputs, parser errors and evaluation payload.</p>
        </aside>
      </div>
    </section>
  </div>

  <script>
    const samples = {samples_json};
    let filteredSamples = [...samples];
    let selectedUnitId = filteredSamples.length ? filteredSamples[0].unit_id : null;
    let sortState = {{ key: "sample_id", direction: "asc" }};

    const searchInput = document.getElementById("sample-search");
    const statusFilter = document.getElementById("status-filter");
    const splitFilter = document.getElementById("split-filter");
    const tableBody = document.getElementById("sample-table-body");
    const detailPanel = document.getElementById("detail-panel");

    function safeValue(value) {{
      if (value === null || value === undefined || value === "") return "N/A";
      if (typeof value === "number") {{
        if (value >= 0 && value <= 1) return (value * 100).toFixed(1) + "%";
        return value.toString();
      }}
      if (typeof value === "boolean") return value ? "Yes" : "No";
      return String(value);
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function getStatusBadge(status) {{
      return `<span class="badge ${{escapeHtml(status)}}">${{escapeHtml(status)}}</span>`;
    }}

    function compareValues(a, b) {{
      if (a === null || a === undefined) return 1;
      if (b === null || b === undefined) return -1;
      if (typeof a === "number" && typeof b === "number") return a - b;
      return String(a).localeCompare(String(b));
    }}

    function applyFilters() {{
      const query = searchInput.value.trim().toLowerCase();
      filteredSamples = samples.filter((sample) => {{
        const matchesQuery = !query || [
          sample.unit_id,
          sample.sample_id,
          sample.prompt_variant,
          sample.prompt_preview,
          sample.prompt_text
        ].some((value) => String(value || "").toLowerCase().includes(query));
        const matchesStatus = statusFilter.value === "all" || sample.status === statusFilter.value;
        const matchesSplit = splitFilter.value === "all" || sample.split === splitFilter.value;
        return matchesQuery && matchesStatus && matchesSplit;
      }});

      filteredSamples.sort((left, right) => {{
        const result = compareValues(left[sortState.key], right[sortState.key]);
        return sortState.direction === "asc" ? result : -result;
      }});

      if (!filteredSamples.find((sample) => sample.unit_id === selectedUnitId)) {{
        selectedUnitId = filteredSamples.length ? filteredSamples[0].unit_id : null;
      }}
      renderTable();
      renderDetail();
    }}

    function renderTable() {{
      if (!filteredSamples.length) {{
        tableBody.innerHTML = "<tr><td colspan='10'>No samples match the current filters.</td></tr>";
        return;
      }}
      tableBody.innerHTML = filteredSamples.map((sample) => `
        <tr data-unit-id="${{escapeHtml(sample.unit_id)}}" class="${{sample.unit_id === selectedUnitId ? "selected" : ""}}">
          <td>${{escapeHtml(sample.sample_id)}}</td>
          <td>${{escapeHtml(sample.prompt_variant)}}</td>
          <td>${{escapeHtml(sample.split)}}</td>
          <td>${{getStatusBadge(sample.status)}}</td>
          <td>${{escapeHtml(safeValue(sample.generated_token_count))}}</td>
          <td>${{escapeHtml(safeValue(sample.predicted_block_count))}}</td>
          <td>${{escapeHtml(safeValue(sample.parser_error_count))}}</td>
          <td>${{escapeHtml(safeValue(sample.content_exact_match_rate))}}</td>
          <td>${{escapeHtml(safeValue(sample.level_exact_match_rate))}}</td>
          <td>${{escapeHtml(sample.prompt_preview)}}</td>
        </tr>
      `).join("");

      tableBody.querySelectorAll("tr[data-unit-id]").forEach((row) => {{
        row.addEventListener("click", () => {{
          selectedUnitId = row.dataset.unitId;
          renderTable();
          renderDetail();
        }});
      }});
    }}

    function renderDetail() {{
      const sample = filteredSamples.find((item) => item.unit_id === selectedUnitId);
      if (!sample) {{
        detailPanel.innerHTML = "<p class='empty-state'>Select a sample row to inspect prompt, outputs, parser errors and evaluation payload.</p>";
        return;
      }}

      const parserErrors = sample.parser_errors.length
        ? `<ul>${{sample.parser_errors.map((error) => `<li>${{escapeHtml(error)}}</li>`).join("")}}</ul>`
        : "<p class='empty-state'>No parser errors recorded.</p>";

      detailPanel.innerHTML = `
        <h3>${{escapeHtml(sample.sample_id)}} <small>${{getStatusBadge(sample.status)}}</small></h3>
        <div class="detail-meta">
          <div><span>Unit ID</span><strong>${{escapeHtml(sample.unit_id)}}</strong></div>
          <div><span>Variant</span><strong>${{escapeHtml(sample.prompt_variant)}}</strong></div>
          <div><span>Split</span><strong>${{escapeHtml(sample.split)}}</strong></div>
          <div><span>Generated tokens</span><strong>${{escapeHtml(safeValue(sample.generated_token_count))}}</strong></div>
          <div><span>Predicted blocks</span><strong>${{escapeHtml(safeValue(sample.predicted_block_count))}}</strong></div>
          <div><span>Content exact match</span><strong>${{escapeHtml(safeValue(sample.content_exact_match_rate))}}</strong></div>
        </div>

        <div class="detail-section">
          <h3>Prompt</h3>
          <pre>${{escapeHtml(sample.prompt_text)}}</pre>
        </div>

        <div class="detail-section">
          <h3>Parser errors</h3>
          ${{parserErrors}}
        </div>

        <div class="detail-section">
          <h3>Reconstructed YAML</h3>
          <pre>${{escapeHtml(sample.reconstructed_yaml || "")}}</pre>
        </div>

        <div class="detail-section">
          <h3>Raw model output</h3>
          <pre>${{escapeHtml(sample.raw_model_output || "")}}</pre>
        </div>

        <div class="detail-section">
          <h3>Predicted blocks</h3>
          <pre>${{escapeHtml(sample.predicted_blocks_json)}}</pre>
        </div>

        <div class="detail-section">
          <h3>Evaluation payload</h3>
          <pre>${{escapeHtml(sample.evaluation_json)}}</pre>
        </div>
      `;
    }}

    document.querySelectorAll("th[data-sort-key]").forEach((header) => {{
      header.addEventListener("click", () => {{
        const nextKey = header.dataset.sortKey;
        if (sortState.key === nextKey) {{
          sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
        }} else {{
          sortState = {{ key: nextKey, direction: "asc" }};
        }}
        applyFilters();
      }});
    }});

    searchInput.addEventListener("input", applyFilters);
    statusFilter.addEventListener("change", applyFilters);
    splitFilter.addEventListener("change", applyFilters);

    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")

    output_html = (args.output_html or (run_dir / DEFAULT_OUTPUT_NAME)).resolve()
    config = load_json(run_dir / "config.json")
    state = load_json(run_dir / "state.json")
    metrics = load_json(run_dir / "metrics.json")
    predictions = maybe_read_jsonl(run_dir / "predictions.jsonl")
    latent_rows = maybe_read_jsonl(run_dir / "latent_mean_vectors.jsonl")

    output_html.write_text(
        render_html(
            run_dir=run_dir,
            output_html=output_html,
            config=config,
            state=state,
            metrics=metrics,
            predictions=predictions,
            latent_rows=latent_rows,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_html": str(output_html),
                "prediction_rows": len(predictions),
                "latent_rows": len(latent_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
