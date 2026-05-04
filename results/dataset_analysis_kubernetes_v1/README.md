# Kubernetes v1 Dataset Analysis

This report is descriptive and is intended to inspect dataset coverage before baseline, SFT, or DPO.

## Key Numbers

- Samples: 283
- Kubernetes resource documents: 338
- Splits: `{"test": 35, "train": 213, "validation": 35}`
- Top resource kinds: Pod (56), DaemonSet (55), Service (31), Deployment (21), Job (20), PersistentVolume (16), Secret (14), StatefulSet (14)
- Exact YAML duplicate samples: 51

## Main Observations

- The dataset is Kubernetes-centered and ready for descriptive analysis before training.
- Coverage is uneven across resource kinds; top kinds should be considered separately in evaluation.
- Most samples contain a single Kubernetes resource, so multi-document behavior is underrepresented.
- Exact YAML duplicate groups exist and should remain grouped when interpreting train/test results.
- Structural target levels are concentrated in shallow-to-medium depths, with a small deep tail.

## Terminology

- `primary_kind`: First Kubernetes kind found among parsed mapping documents in a sample.
- `yaml_max_depth`: Maximum recursive depth of the parsed YAML object tree. This is not the same as block level.
- `yaml_total_nodes`: Total parsed YAML mapping, list, and scalar nodes. This does not mean Kubernetes Node resources.
- `block_count`: Number of line-and-level blocks derived from normalized YAML.
- `level`: Indentation level of a block, using two spaces per level in v1.

## Outputs

- `dataset_analysis_report.html`: navigable report with embedded figures.
- `figures/*.png`: static figures for the thesis or slides.
- `dataset_analysis_summary.json`: machine-readable summary.

## Limitations

- Bias here means dataset coverage bias, not social bias.
- Semantic fields are approximate recursive key-presence signals, not full Kubernetes schema validation.
- Primary kind uses the first document in a sample; multi-document samples are also counted in resource-level tables.
