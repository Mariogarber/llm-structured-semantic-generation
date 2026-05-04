from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.offline import get_plotlyjs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from llm_structured_semantic_generation.dataset_io import read_jsonl


DEFAULT_COLOR_COLUMNS = (
    "cluster",
    "primary_kind",
    "split",
    "yaml_total_nodes",
    "block_count",
    "yaml_max_depth",
    "generated_token_count",
)


@dataclass(frozen=True)
class ClusterSelection:
    labels: list[int]
    selected_k: int | None
    silhouette_scores: dict[int, float]


@dataclass(frozen=True)
class ReductionResult:
    name: str
    coordinates: np.ndarray
    metadata: dict[str, Any]


def load_latent_vectors(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows = read_jsonl(path, allow_truncated_last_line=False)
    if not rows:
        raise ValueError(f"No latent rows found in {path}")

    metadata_rows: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    expected_dim: int | None = None

    for index, row in enumerate(rows, start=1):
        vector = row.get("latent_mean")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Missing non-empty latent_mean at {path}:{index}")

        numeric_vector = [float(value) for value in vector]
        if expected_dim is None:
            expected_dim = len(numeric_vector)
        elif len(numeric_vector) != expected_dim:
            raise ValueError(
                f"Inconsistent latent dimension at {path}:{index}: "
                f"expected {expected_dim}, got {len(numeric_vector)}"
            )

        metadata = {key: value for key, value in row.items() if key != "latent_mean"}
        metadata.setdefault("latent_dim", len(numeric_vector))
        metadata_rows.append(metadata)
        vectors.append(numeric_vector)

    matrix = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Latent matrix contains non-finite values: {path}")
    return pd.DataFrame(metadata_rows), matrix


def merge_feature_metadata(latent_metadata: pd.DataFrame, features_csv: Path | None) -> pd.DataFrame:
    metadata = latent_metadata.copy()
    if features_csv is None or not features_csv.exists():
        return metadata

    features = pd.read_csv(features_csv)
    if "sample_id" not in metadata.columns or "sample_id" not in features.columns:
        return metadata

    overlap = set(metadata.columns).intersection(features.columns) - {"sample_id"}
    features = features.rename(columns={column: f"dataset_{column}" for column in overlap})
    return metadata.merge(features, on="sample_id", how="left")


def standardize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 2:
        return vectors.astype(np.float32, copy=True)
    return StandardScaler().fit_transform(vectors).astype(np.float32)


def _safe_pca_components(vectors: np.ndarray, requested: int) -> int:
    if vectors.shape[0] < 2:
        return 0
    return max(1, min(requested, vectors.shape[0] - 1, vectors.shape[1]))


def pre_reduce_for_manifold(vectors: np.ndarray, pca_dims: int, random_state: int) -> tuple[np.ndarray, dict[str, Any]]:
    components = _safe_pca_components(vectors, pca_dims)
    if components <= 0 or components >= vectors.shape[1]:
        return vectors, {"pre_reduction": "none"}

    pca = PCA(n_components=components, random_state=random_state)
    reduced = pca.fit_transform(vectors)
    return reduced, {
        "pre_reduction": "pca",
        "pre_reduction_dims": components,
        "pre_reduction_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }


def compute_pca(vectors: np.ndarray, random_state: int) -> ReductionResult:
    if vectors.shape[0] < 2:
        coordinates = np.zeros((vectors.shape[0], 2), dtype=np.float32)
        return ReductionResult("pca", coordinates, {"warning": "PCA requires at least 2 samples."})

    components = min(2, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=components, random_state=random_state)
    raw_coordinates = pca.fit_transform(vectors)
    coordinates = np.zeros((vectors.shape[0], 2), dtype=np.float32)
    coordinates[:, :components] = raw_coordinates
    return ReductionResult(
        "pca",
        coordinates,
        {
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
    )


def compute_tsne(
    vectors: np.ndarray,
    *,
    pca_dims: int,
    perplexity: float | None,
    random_state: int,
) -> ReductionResult | None:
    if vectors.shape[0] < 3:
        return None

    effective_perplexity = perplexity
    if effective_perplexity is None:
        effective_perplexity = float(min(30, max(2, (vectors.shape[0] - 1) // 3)))
    effective_perplexity = min(float(effective_perplexity), float(vectors.shape[0] - 1))
    if effective_perplexity <= 0:
        return None

    manifold_input, metadata = pre_reduce_for_manifold(vectors, pca_dims, random_state)
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )
    coordinates = reducer.fit_transform(manifold_input)
    metadata.update({"perplexity": effective_perplexity})
    return ReductionResult("tsne", coordinates, metadata)


def compute_umap(
    vectors: np.ndarray,
    *,
    pca_dims: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> ReductionResult | None:
    if vectors.shape[0] < 3:
        return None

    try:
        from umap import UMAP
    except ImportError:
        return None

    effective_neighbors = max(2, min(n_neighbors, vectors.shape[0] - 1))
    manifold_input, metadata = pre_reduce_for_manifold(vectors, pca_dims, random_state)
    reducer = UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    coordinates = reducer.fit_transform(manifold_input)
    metadata.update({"n_neighbors": effective_neighbors, "min_dist": min_dist})
    return ReductionResult("umap", coordinates, metadata)


def compute_reductions(
    vectors: np.ndarray,
    *,
    methods: Iterable[str],
    pca_dims: int,
    tsne_perplexity: float | None,
    umap_neighbors: int,
    umap_min_dist: float,
    random_state: int,
) -> list[ReductionResult]:
    normalized_methods = [method.strip().lower() for method in methods if method.strip()]
    results: list[ReductionResult] = []
    for method in normalized_methods:
        if method == "pca":
            results.append(compute_pca(vectors, random_state))
        elif method == "tsne":
            result = compute_tsne(
                vectors,
                pca_dims=pca_dims,
                perplexity=tsne_perplexity,
                random_state=random_state,
            )
            if result is not None:
                results.append(result)
        elif method == "umap":
            result = compute_umap(
                vectors,
                pca_dims=pca_dims,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
            )
            if result is not None:
                results.append(result)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
    return results


def choose_kmeans_clusters(
    vectors: np.ndarray,
    *,
    cluster_count: int | None,
    max_clusters: int,
    random_state: int,
) -> ClusterSelection:
    sample_count = vectors.shape[0]
    if sample_count == 0:
        return ClusterSelection([], None, {})
    if sample_count < 3:
        return ClusterSelection([0] * sample_count, 1, {})

    if cluster_count is not None:
        selected_k = max(1, min(cluster_count, sample_count))
        labels = KMeans(n_clusters=selected_k, random_state=random_state, n_init="auto").fit_predict(vectors)
        return ClusterSelection([int(label) for label in labels], selected_k, {})

    candidate_max = min(max_clusters, sample_count - 1)
    if candidate_max < 2:
        return ClusterSelection([0] * sample_count, 1, {})

    best_k: int | None = None
    best_score = -1.0
    scores: dict[int, float] = {}
    for k in range(2, candidate_max + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(vectors, labels))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        return ClusterSelection([0] * sample_count, 1, scores)

    labels = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto").fit_predict(vectors)
    return ClusterSelection([int(label) for label in labels], best_k, scores)


def build_embedding_frame(metadata: pd.DataFrame, reductions: list[ReductionResult]) -> pd.DataFrame:
    frame = metadata.copy()
    for result in reductions:
        frame[f"{result.name}_x"] = result.coordinates[:, 0]
        frame[f"{result.name}_y"] = result.coordinates[:, 1]
    return frame


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _hover_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "unit_id",
        "sample_id",
        "prompt_variant",
        "split",
        "primary_kind",
        "cluster",
        "yaml_total_nodes",
        "block_count",
        "yaml_max_depth",
        "generated_token_count",
        "content_exact_match_rate",
        "level_exact_match_rate",
    ]
    return [column for column in candidates if column in frame.columns]


def make_scatter_html(frame: pd.DataFrame, method: str, color_column: str) -> str:
    plot_frame = frame.copy()
    if color_column not in plot_frame.columns:
        raise ValueError(f"Unknown color column: {color_column}")

    if not _is_numeric_series(plot_frame[color_column]):
        plot_frame[color_column] = plot_frame[color_column].fillna("<missing>").astype(str)

    figure = px.scatter(
        plot_frame,
        x=f"{method}_x",
        y=f"{method}_y",
        color=color_column,
        hover_data=_hover_columns(plot_frame),
        title=f"{method.upper()} latent projection colored by {color_column}",
        color_continuous_scale="Viridis",
    )
    figure.update_traces(marker={"size": 9, "opacity": 0.86, "line": {"width": 0.5, "color": "#ffffff"}})
    figure.update_layout(
        margin=dict(l=24, r=24, t=64, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return pio.to_html(figure, include_plotlyjs=False, full_html=False, config={"responsive": True})


def render_latent_report(
    *,
    title: str,
    latent_path: Path,
    output_dir: Path,
    frame: pd.DataFrame,
    reductions: list[ReductionResult],
    cluster_selection: ClusterSelection,
    color_columns: list[str],
    summary: dict[str, Any],
) -> str:
    sections: list[str] = []
    for reduction in reductions:
        available_colors = [column for column in color_columns if column in frame.columns]
        for color_column in available_colors:
            sections.append(
                "<section class='plot-card'>"
                f"<h2>{html.escape(reduction.name.upper())} by {html.escape(color_column)}</h2>"
                f"{make_scatter_html(frame, reduction.name, color_column)}"
                "</section>"
            )

    reduction_rows = "".join(
        "<tr>"
        f"<td>{html.escape(result.name)}</td>"
        f"<td><code>{html.escape(json.dumps(result.metadata, ensure_ascii=False))}</code></td>"
        "</tr>"
        for result in reductions
    )
    cluster_rows = "".join(
        f"<tr><td>{k}</td><td>{score:.4f}</td></tr>"
        for k, score in cluster_selection.silhouette_scores.items()
    )
    cluster_table = (
        "<p>No automatic silhouette search was needed.</p>"
        if not cluster_rows
        else "<table><thead><tr><th>k</th><th>silhouette</th></tr></thead>"
        f"<tbody>{cluster_rows}</tbody></table>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script>{get_plotlyjs()}</script>
  <style>
    :root {{
      --bg: #eef4f8;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #5b677a;
      --accent: #0f766e;
      --border: #d7e0e7;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 10% 5%, #d6f3ec 0, transparent 34%),
                  linear-gradient(135deg, #f8fafc 0%, var(--bg) 58%, #e7eef7 100%);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    header {{
      padding: 36px min(6vw, 72px) 18px;
    }}
    main {{
      padding: 0 min(6vw, 72px) 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 1.15rem;
    }}
    p {{
      color: var(--muted);
      line-height: 1.55;
    }}
    code {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 20px 0;
    }}
    .card, .plot-card, .meta-card {{
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: 0 18px 45px rgba(30, 41, 59, 0.08);
    }}
    .card {{
      padding: 18px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card strong {{
      display: block;
      margin-top: 6px;
      font-size: 1.5rem;
    }}
    .plot-card, .meta-card {{
      padding: 20px;
      margin: 18px 0;
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--border);
      padding: 10px;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>
      Exploratory projection of model-generated mean latent vectors. The plots are diagnostic views,
      not evidence of validated causal structure by themselves.
    </p>
  </header>
  <main>
    <div class="cards">
      <div class="card"><span>Latent rows</span><strong>{len(frame):,}</strong></div>
      <div class="card"><span>Latent dim</span><strong>{html.escape(str(summary.get("latent_dim", "unknown")))}</strong></div>
      <div class="card"><span>Selected k</span><strong>{html.escape(str(cluster_selection.selected_k))}</strong></div>
      <div class="card"><span>Reducers</span><strong>{html.escape(", ".join(result.name for result in reductions))}</strong></div>
    </div>
    <section class="meta-card">
      <h2>Run Metadata</h2>
      <table>
        <tbody>
          <tr><th>Latent source</th><td>{html.escape(str(latent_path))}</td></tr>
          <tr><th>Output directory</th><td>{html.escape(str(output_dir))}</td></tr>
          <tr><th>Color columns</th><td>{html.escape(", ".join(color_columns))}</td></tr>
        </tbody>
      </table>
    </section>
    <section class="meta-card">
      <h2>Reduction Details</h2>
      <table><thead><tr><th>method</th><th>metadata</th></tr></thead><tbody>{reduction_rows}</tbody></table>
    </section>
    <section class="meta-card">
      <h2>KMeans Selection</h2>
      {cluster_table}
    </section>
    {"".join(sections)}
  </main>
</body>
</html>
"""


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
