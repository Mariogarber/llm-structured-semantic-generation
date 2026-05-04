# Latent Space Analysis

This note defines the exploratory workflow for analyzing model-generated latent
mean vectors.

The analysis is aligned with the current Kubernetes v1 modeling contract:

- source vectors are internal model representations, currently stored as
  `latent_mean` rows in `latent_mean_vectors.jsonl`
- the analysis is diagnostic and exploratory
- visual clusters must not be treated as validated experimental results without
  later quantitative checks
- sample-level metadata should come from existing dataset analysis artifacts
  instead of invented labels

## Input artifacts

The main input is a run artifact:

```text
results/baseline_kubernetes_v1/<run-id>/latent_mean_vectors.jsonl
```

Each row is expected to include:

- `unit_id`
- `sample_id`
- `prompt_variant`
- `split`
- `generated_token_count`
- `latent_dim`
- `latent_mean`

The optional metadata input is:

```text
results/dataset_analysis_kubernetes_v1/dataset_analysis_sample_features.csv
```

This file provides structural and semantic descriptors such as `primary_kind`,
`yaml_total_nodes`, `block_count`, and `yaml_max_depth`.

## Command

Example:

```powershell
uv run python scripts/analyze_latent_space.py `
  --run-dir results/baseline_kubernetes_v1/compact-test70-320-vtfix
```

The script writes:

- `latent_embeddings_2d.csv`
- `latent_analysis_summary.json`
- `latent_space_report.html`

## Methods

The script supports:

- PCA, always available through `scikit-learn`
- t-SNE, always available through `scikit-learn`
- UMAP, only when `umap-learn` is installed

Vectors are standardized by default before dimensionality reduction and KMeans
clustering. UMAP is skipped cleanly if the optional dependency is unavailable.

## Interpretation

Recommended first views:

- color by `primary_kind` to inspect whether Kubernetes resource type is visible
  in latent geometry
- color by `yaml_total_nodes`, `block_count`, and `yaml_max_depth` to inspect
  complexity gradients
- color by `cluster` to inspect whether automatic KMeans groups align with
  interpretable YAML properties
- color by `split` as a quick sanity check for split artifacts

Important limitation: a visually clean projection does not prove that the model
has learned a robust structural latent representation. It should be used to
generate hypotheses that are later checked against structural validity metrics,
parser success, prompt adequacy, or controlled ablations.
