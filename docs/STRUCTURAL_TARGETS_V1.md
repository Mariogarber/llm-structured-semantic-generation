# Kubernetes Structural Targets v1

This document defines the implemented bridge between the processed Kubernetes v1
dataset and the first modeling experiments.

## Purpose

The structural target layer converts each `target_yaml_normalized` value into a
deterministic sequence of line blocks:

```json
{
  "document_index": 0,
  "line_index": 0,
  "line_text": "apiVersion: v1",
  "level": 0
}
```

This is the explicit representation immediately before parser-based structural
control. It is not the latent representation itself.

## Implemented Contract

- `line_text` stores one YAML physical line without leading indentation.
- `level` stores indentation depth using two spaces per level.
- `line_index` is zero-based inside each YAML document.
- `document_index` is zero-based and allows multi-document YAML.
- `---` document separator lines are not content blocks; they are reconstructed
  from `document_index` changes.
- Blank lines are preserved as blocks because they can appear inside rendered
  multi-line scalar values.

The parser reconstructs YAML by applying indentation from `level` and preserving
line order. It does not add missing keys, values, resources, or semantic repairs.

## Generated Artifacts

Run:

```bash
uv run python scripts/build_kubernetes_structural_targets.py
```

Outputs:

- `data/processed/kubernetes_v1/dataset_structural_targets.jsonl`
- `data/processed/kubernetes_v1/structural_targets_report.json`

The report must have `ready_for_baseline: true` before baseline generation or SFT
preparation is treated as valid.

## SFT Serialization

Run:

```bash
uv run python scripts/build_kubernetes_sft_dataset.py
```

Outputs:

- `data/processed/kubernetes_v1/sft/train.jsonl`
- `data/processed/kubernetes_v1/sft/validation.jsonl`
- `data/processed/kubernetes_v1/sft/test.jsonl`
- `data/processed/kubernetes_v1/sft/sft_dataset_report.json`

The target serialization is `blocks_tsv_v1`, wrapped in `<blocks>` and
`</blocks>` markers. This fixes the first supervised target format without
claiming that SFT has already been run.

## Validation

The structural target builder validates each row by:

- parsing the reference YAML with `yaml.safe_load_all`,
- converting normalized YAML to blocks,
- reconstructing YAML from blocks,
- parsing the reconstructed YAML,
- checking parsed-document equality with the reference.

This guards against hidden parser repair: semantic equivalence must come from the
line-and-level representation, not from invented content.

