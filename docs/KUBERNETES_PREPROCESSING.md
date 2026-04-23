# Kubernetes Preprocessing v1

This repository now includes a reproducible preprocessing pipeline for the Kubernetes corpus stored in `data/`.

The functional modeling contract that consumes these artifacts is defined in `docs/KUBERNETES_MODEL_V1.md`.
Terminology for dataset fields, structural targets, and analysis metrics is defined in `docs/TERMINOLOGY.md`.
The dataset is not only intended for final YAML supervision. It is the experimental basis for:

- the latent intermediate representation described in the modeling document
- the explicit line-and-level block representation
- later comparative structural signals used for parser/control experiments

## Scope

- Input source: `data/question/Kubernetes`
- Target source: `data/docker_compose/Kubernetes`
- Excluded on purpose: `data/raw`
- Dataset policy: keep `question.txt` and `question_simplified.txt` as valid prompt variants, but force both variants of the same sample into the same split

## Run

```bash
python utils/kubernetes_dataset_preprocessor.py
```

Optional flags:

```bash
python utils/kubernetes_dataset_preprocessor.py --output-dir data/processed/kubernetes_v1 --seed 42
```

## Generated artifacts

The script writes the processed dataset to `data/processed/kubernetes_v1/`.

- `dataset_manifest_samples.csv`: one row per sample
- `dataset_manifest_prompt_variants.csv`: one row per prompt variant
- `dataset_splits.csv`: split assignment per sample
- `dataset_train_ready.jsonl`: training-ready rows at prompt-variant level
- `dataset_train_ready_samples.jsonl`: training-ready rows at sample level
- `quality_report.json`: validation and readiness gates

The structural target stage builds on these artifacts and writes:

- `dataset_structural_targets.jsonl`: one row per prompt variant with derived line-and-level blocks
- `structural_targets_report.json`: round-trip validation report for the structural targets
- `sft/train.jsonl`, `sft/validation.jsonl`, `sft/test.jsonl`: fixed SFT serialization derived from structural targets
- `sft/sft_dataset_report.json`: SFT export report

These artifacts are the fixed v1 basis for the main line of the project. Oversampling or synthetic enlargement, if later used, should be treated as separate experiments rather than as part of this preprocessing contract.

The accepted future oversampling branch is `kubernetes_v2`, documented in
`docs/MULTI_RESOURCE_STRATEGY_DECISION.md`. It must read from
`data/processed/kubernetes_v1/` and write to `data/processed/kubernetes_v2/`
without modifying the v1 artifacts.

## Canonicalization policy

- YAML is parsed with `yaml.safe_load_all`
- Multi-document YAML separated by `---` is supported and normalized as part of the same target sample
- Canonical YAML is rendered with sorted keys and stable indentation
- Canonical text is stored as a derived field and does not overwrite the source file
- The script checks that canonicalization is deterministic and preserves the parsed structure

This canonicalized target is meant to support more than one downstream representation. In particular, it should be possible to derive from it:

- the final YAML target
- the line-and-level structured blocks
- future intermediate or auxiliary structural representations

## Leakage policy

- Exact duplicate normalized YAML merges samples into the same leakage group
- Exact duplicate normalized prompts merge samples into the same leakage group
- Very high prompt similarity also merges samples into the same leakage group
- Split assignment is done at leakage-group level, never per prompt row

### `leakage_reasons`

The `leakage_reasons` field explains why a sample was assigned to a leakage group instead of being treated as an isolated sample.

- `sample_id_only`: no cross-sample leakage signal was found, so the sample forms its own group. This is the default safe case.
- `exact_yaml_duplicate`: two or more samples produce exactly the same normalized target YAML. These samples must stay in the same split because otherwise the model could memorize the target structure from train and get an artificially easy test example.
- `exact_prompt_duplicate`: two or more samples contain exactly the same normalized prompt text after lowercasing and whitespace normalization. These samples must stay together because they represent the same input request and would leak prompt-level information across splits.
- `near_prompt_duplicate`: two samples do not have identical normalized prompts, but they are close enough to be treated as paraphrase-level duplicates. The current heuristic requires a high sequence similarity (`>= 0.97`) and a similar prompt length (`>= 0.85` length ratio). These samples are grouped together to reduce train/test leakage through near-identical wording.

### Why these reasons are justified

- The project evaluates structured generation, so leakage is not only about repeated text; repeated YAML targets are also dangerous because they can inflate downstream metrics.
- Prompt leakage matters because the same or almost the same request in train and test makes generalization look better than it really is.
- Grouping by leakage reasons is conservative by design: it is better to keep borderline duplicates together than to overestimate model performance.
- Multiple reasons may appear at once as a comma-separated list, for example when two samples share both the same normalized YAML and the same normalized prompt.

### Current dataset state

In the current Kubernetes v1 build, the observed values are:

- `sample_id_only`
- `exact_yaml_duplicate`

The code also supports `exact_prompt_duplicate` and `near_prompt_duplicate`, even if they do not appear in the current processed version.

## Structural target stage

After preprocessing, run:

```bash
uv run python scripts/build_kubernetes_structural_targets.py
```

This derives the implemented pre-parser representation:

`target_yaml_normalized -> blocks(document_index, line_index, line_text, level) -> reconstructed YAML`

Each row must round-trip through the deterministic parser and preserve parsed YAML semantics. The parser applies indentation from `level` and validates YAML parseability; it does not invent missing Kubernetes content or silently repair semantic mistakes.

Then run:

```bash
uv run python scripts/build_kubernetes_sft_dataset.py
```

This creates the first fixed SFT-ready serialization over the structural blocks. It prepares supervised data only; it is not evidence that SFT has already been trained or evaluated.

## Dataset analysis stage

Before model training, run:

```bash
uv run python scripts/analyze_kubernetes_dataset.py
```

This writes a descriptive analysis to `results/dataset_analysis_kubernetes_v1/`:

- `dataset_analysis_report.html`: navigable report with embedded plots
- `figures/*.png`: static figures for thesis text or slides
- `dataset_analysis_summary.json`: machine-readable summary of coverage, leakage, complexity, and semantic-key presence
- `dataset_analysis_sample_features.csv`: sample-level derived features
- `dataset_analysis_resource_rows.csv`: resource-document-level derived features

This analysis is descriptive only. It is meant to expose dataset coverage bias, split balance, structural complexity, and approximate Kubernetes field coverage before baseline or SFT.

The main analysis terms are:

- `primary_kind`: first Kubernetes `kind` found among the parsed mapping documents of a sample.
- `yaml_max_depth`: maximum recursive depth of the parsed YAML object tree.
- `yaml_total_nodes`: total count of parsed YAML mapping, list, and scalar nodes.
- `block_count`: number of line-and-level blocks derived from the normalized YAML.

See `docs/TERMINOLOGY.md` for the full glossary and the distinction between YAML tree nodes, Kubernetes `Node` resources, block `level`, and parsed YAML depth.

## Future `kubernetes_v2` enrichment

`kubernetes_v2` is planned as a derived, experimental dataset for controlled
multi-resource oversampling. It is not part of the base v1 preprocessing run.

The intended source and target locations are:

- Source: `data/processed/kubernetes_v1/`
- Target: `data/processed/kubernetes_v2/`

The first accepted composition families are:

- `Deployment + Service`
- `Pod + ConfigMap`
- `Pod + Secret`
- `PersistentVolume + PersistentVolumeClaim + Pod`
- `ServiceAccount + Role + RoleBinding`
- `StatefulSet + Service`

Each synthetic row must record source sample IDs, source leakage groups, split,
composition strategy, prompt strategy, and normalized target YAML. Synthetic
examples must not mix splits.

The generated `kubernetes_v2` report must compare v1 and v2 coverage for
documents per sample, `kind` combinations, YAML depth, block count, semantic-key
presence, and leakage behavior.
