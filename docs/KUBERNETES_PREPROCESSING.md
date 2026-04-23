# Kubernetes Preprocessing v1

This repository now includes a reproducible preprocessing pipeline for the Kubernetes corpus stored in `data/`.

The functional modeling contract that consumes these artifacts is defined in `docs/KUBERNETES_MODEL_V1.md`.
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

These artifacts are the fixed v1 basis for the main line of the project. Oversampling or synthetic enlargement, if later used, should be treated as separate experiments rather than as part of this preprocessing contract.

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
