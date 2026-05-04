from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.dataset_io import write_jsonl
from llm_structured_semantic_generation.latent_analysis import (
    build_embedding_frame,
    choose_kmeans_clusters,
    compute_reductions,
    load_latent_vectors,
    merge_feature_metadata,
    standardize_vectors,
)


class LatentAnalysisTest(unittest.TestCase):
    def test_load_merge_reduce_and_cluster_latent_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latent_path = tmp_path / "latent_mean_vectors.jsonl"
            write_jsonl(
                latent_path,
                [
                    {
                        "unit_id": "q1::question",
                        "sample_id": "q1",
                        "split": "test",
                        "generated_token_count": 8,
                        "latent_mean": [0.0, 0.1, 0.2],
                    },
                    {
                        "unit_id": "q2::question",
                        "sample_id": "q2",
                        "split": "test",
                        "generated_token_count": 10,
                        "latent_mean": [1.0, 1.1, 1.2],
                    },
                    {
                        "unit_id": "q3::question",
                        "sample_id": "q3",
                        "split": "validation",
                        "generated_token_count": 12,
                        "latent_mean": [5.0, 5.1, 5.2],
                    },
                ],
            )
            features_path = tmp_path / "features.csv"
            features_path.write_text(
                "sample_id,primary_kind,yaml_total_nodes,block_count\n"
                "q1,ConfigMap,5,4\n"
                "q2,ConfigMap,6,5\n"
                "q3,Deployment,30,20\n",
                encoding="utf-8",
            )

            metadata, vectors = load_latent_vectors(latent_path)
            merged = merge_feature_metadata(metadata, features_path)
            analysis_vectors = standardize_vectors(vectors)
            clusters = choose_kmeans_clusters(
                analysis_vectors,
                cluster_count=2,
                max_clusters=4,
                random_state=7,
            )
            reductions = compute_reductions(
                analysis_vectors,
                methods=["pca"],
                pca_dims=2,
                tsne_perplexity=None,
                umap_neighbors=3,
                umap_min_dist=0.1,
                random_state=7,
            )
            merged["cluster"] = [str(label) for label in clusters.labels]
            frame = build_embedding_frame(merged, reductions)

            self.assertEqual(vectors.shape, (3, 3))
            self.assertIn("primary_kind", frame.columns)
            self.assertIn("pca_x", frame.columns)
            self.assertIn("pca_y", frame.columns)
            self.assertEqual(clusters.selected_k, 2)
            self.assertEqual(len(frame), 3)

    def test_load_latent_vectors_rejects_inconsistent_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latent_path = Path(tmp) / "latent_mean_vectors.jsonl"
            write_jsonl(
                latent_path,
                [
                    {"unit_id": "a", "sample_id": "a", "latent_mean": [1.0, 2.0]},
                    {"unit_id": "b", "sample_id": "b", "latent_mean": [1.0]},
                ],
            )

            with self.assertRaises(ValueError):
                load_latent_vectors(latent_path)


if __name__ == "__main__":
    unittest.main()
