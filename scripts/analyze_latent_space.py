from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from llm_structured_semantic_generation.latent_analysis import (
    DEFAULT_COLOR_COLUMNS,
    build_embedding_frame,
    choose_kmeans_clusters,
    compute_reductions,
    load_latent_vectors,
    merge_feature_metadata,
    render_latent_report,
    standardize_vectors,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project latent mean vectors with PCA/t-SNE/UMAP and render an exploratory report."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing latent_mean_vectors.jsonl.",
    )
    source.add_argument(
        "--latent-jsonl",
        type=Path,
        help="Direct path to a latent_mean_vectors.jsonl file.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "dataset_analysis_kubernetes_v1"
        / "dataset_analysis_sample_features.csv",
        help="Optional sample-level feature CSV to merge by sample_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/latent_analysis or results/latent_analysis/<source-name>.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "tsne", "umap"],
        choices=["pca", "tsne", "umap"],
        help="Dimensionality reduction methods to run. UMAP is skipped if umap-learn is not installed.",
    )
    parser.add_argument(
        "--color-by",
        nargs="+",
        default=list(DEFAULT_COLOR_COLUMNS),
        help="Metadata columns to use as colors in the HTML report.",
    )
    parser.add_argument("--cluster-count", type=int, default=None, help="Fixed KMeans cluster count.")
    parser.add_argument("--max-clusters", type=int, default=10, help="Maximum k for automatic KMeans search.")
    parser.add_argument("--pca-dims", type=int, default=50, help="PCA dimensions before t-SNE/UMAP.")
    parser.add_argument("--tsne-perplexity", type=float, default=None, help="Optional fixed t-SNE perplexity.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors if UMAP is available.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist if UMAP is available.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable standardization before reduction and clustering.",
    )
    return parser.parse_args()


def resolve_latent_path(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return (args.run_dir / "latent_mean_vectors.jsonl").resolve()
    return args.latent_jsonl.resolve()


def resolve_output_dir(args: argparse.Namespace, latent_path: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    if args.run_dir is not None:
        return (args.run_dir / "latent_analysis").resolve()
    return (REPO_ROOT / "results" / "latent_analysis" / latent_path.parent.name).resolve()


def main() -> None:
    args = parse_args()
    latent_path = resolve_latent_path(args)
    if not latent_path.exists():
        raise FileNotFoundError(f"Latent JSONL does not exist: {latent_path}")

    output_dir = resolve_output_dir(args, latent_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_metadata, vectors = load_latent_vectors(latent_path)
    merged_metadata = merge_feature_metadata(latent_metadata, args.features_csv)
    analysis_vectors = vectors if args.no_standardize else standardize_vectors(vectors)

    clusters = choose_kmeans_clusters(
        analysis_vectors,
        cluster_count=args.cluster_count,
        max_clusters=args.max_clusters,
        random_state=args.random_state,
    )
    merged_metadata["cluster"] = [str(label) for label in clusters.labels]

    reductions = compute_reductions(
        analysis_vectors,
        methods=args.methods,
        pca_dims=args.pca_dims,
        tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        random_state=args.random_state,
    )
    if not reductions:
        raise RuntimeError("No reductions were produced. Try including PCA or using more samples.")

    embedding_frame = build_embedding_frame(merged_metadata, reductions)
    embeddings_csv = output_dir / "latent_embeddings_2d.csv"
    embedding_frame.to_csv(embeddings_csv, index=False)

    summary = {
        "latent_path": str(latent_path),
        "features_csv": str(args.features_csv) if args.features_csv and args.features_csv.exists() else None,
        "output_dir": str(output_dir),
        "sample_count": int(vectors.shape[0]),
        "latent_dim": int(vectors.shape[1]),
        "standardized": not args.no_standardize,
        "methods_requested": args.methods,
        "methods_rendered": [result.name for result in reductions],
        "cluster_count": clusters.selected_k,
        "silhouette_scores": {str(key): value for key, value in clusters.silhouette_scores.items()},
        "reduction_metadata": {result.name: result.metadata for result in reductions},
    }
    write_summary(output_dir / "latent_analysis_summary.json", summary)

    available_color_columns = [column for column in args.color_by if column in embedding_frame.columns]
    report_html = output_dir / "latent_space_report.html"
    report_html.write_text(
        render_latent_report(
            title="Latent Space Analysis",
            latent_path=latent_path,
            output_dir=output_dir,
            frame=embedding_frame,
            reductions=reductions,
            cluster_selection=clusters,
            color_columns=available_color_columns,
            summary=summary,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "latent_path": str(latent_path),
                "output_dir": str(output_dir),
                "embeddings_csv": str(embeddings_csv),
                "report_html": str(report_html),
                "sample_count": int(vectors.shape[0]),
                "latent_dim": int(vectors.shape[1]),
                "methods_rendered": [result.name for result in reductions],
                "cluster_count": clusters.selected_k,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
