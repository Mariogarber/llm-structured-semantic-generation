# Baseline Offline Recomputed Metrics Report

- Run directory: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\results\baseline_kubernetes_v1\compact-validation70-320-vtfix`
- Dataset used for recomputation: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\data\processed\kubernetes_v1\dataset_structural_targets.jsonl`
- Row count: `62`
- Evaluated count: `58`

## Headline metrics

- `structured_output_parse_success_rate`: old=None new=0.9354838709677419
- `yaml_parse_success_rate`: old=None new=0.25862068965517243
- `parsed_equal_rate`: old=None new=0.017241379310344827
- `average_line_text_f1`: old=None new=0.15701133683188576
- `average_semantic_key_f1`: old=None new=0.22513750617198894
- `average_prompt_requirement_precision`: old=None new=0.19827586206896552
- `average_prompt_requirement_recall`: old=None new=0.14885057471264368
- `average_prompt_requirement_f1`: old=None new=0.1617816091954023
- `prompt_requirement_exact_match_rate`: old=None new=0.10344827586206896
- `average_required_field_presence_rate`: old=None new=0.875
- `average_required_field_complete_resource_rate`: old=None new=0.5833333333333334
- `required_field_complete_sample_rate`: old=None new=0.5833333333333334

## Error profile

- Structured-output failures before evaluation: `4`
- YAML failures after structured parsing: `43`
- Parsed-equal rows: `1`

## Prompt Requirement Categories

- `image`: prompt_total=1, prediction_total=1, match_total=1, precision=1.0, recall=1.0
- `kind`: prompt_total=20, prediction_total=12, match_total=12, precision=1.0, recall=0.6
- `label`: prompt_total=2, prediction_total=2, match_total=2, precision=1.0, recall=1.0
- `metadata.name`: prompt_total=4, prediction_total=4, match_total=3, precision=0.75, recall=0.75
- `port`: prompt_total=2, prediction_total=1, match_total=1, precision=1.0, recall=0.5
- `serviceAccountName`: prompt_total=6, prediction_total=1, match_total=1, precision=1.0, recall=0.16666666666666666

## Missing Required Fields By Kind

- `DaemonSet`: incomplete_resources=5; most common missing groups: spec.template.spec.containers (5), spec.selector (2), spec.template (2)

## Example Incomplete Predictions

- `q13::question`: DaemonSet: spec.template.spec.containers
- `q13::question_simplified`: DaemonSet: spec.template.spec.containers
- `q21::question_simplified`: DaemonSet: spec.template.spec.containers
- `q48::question`: DaemonSet: spec.selector, spec.template, spec.template.spec.containers
- `q66::question`: DaemonSet: spec.selector, spec.template, spec.template.spec.containers

## Conclusions

- The recomputation confirms that the run can be re-evaluated offline from persisted artifacts alone; no new model inference was needed.
- Prompt-requirement coverage is materially stronger than exact YAML equality: `average_prompt_requirement_f1 = 0.1617816091954023` versus `parsed_equal_rate = 0.017241379310344827`.
- Required-field validity is high once the prediction reaches YAML parseability: `average_required_field_complete_resource_rate = 0.5833333333333334` and `required_field_complete_sample_rate = 0.5833333333333334`.
- The largest remaining bottleneck is still upstream structural generation, not minimal resource completeness: `structured_output_parse_success_rate = 0.9354838709677419` and `yaml_parse_success_rate = 0.25862068965517243`.
- This means the baseline often captures coarse intent and minimal field structure, but still fails too often in full structural realization and exact reconstruction.
