# Baseline Offline Recomputed Metrics Report

- Run directory: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\results\baseline_kubernetes_v1\compact-validation70-320-vtfix`
- Dataset used for recomputation: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\data\processed\kubernetes_v1\dataset_structural_targets.jsonl`
- Row count: `25`
- Evaluated count: `23`

## Headline metrics

- `structured_output_parse_success_rate`: old=None new=0.92
- `yaml_parse_success_rate`: old=None new=0.34782608695652173
- `parsed_equal_rate`: old=None new=0.043478260869565216
- `average_line_text_f1`: old=None new=0.20972365027718826
- `average_semantic_key_f1`: old=None new=0.3067367415193502
- `average_prompt_requirement_precision`: old=None new=0.2391304347826087
- `average_prompt_requirement_recall`: old=None new=0.2217391304347826
- `average_prompt_requirement_f1`: old=None new=0.22826086956521738
- `prompt_requirement_exact_match_rate`: old=None new=0.17391304347826086
- `average_required_field_presence_rate`: old=None new=0.9166666666666666
- `average_required_field_complete_resource_rate`: old=None new=0.5
- `required_field_complete_sample_rate`: old=None new=0.5

## Error profile

- Structured-output failures before evaluation: `2`
- YAML failures after structured parsing: `15`
- Parsed-equal rows: `1`

## Prompt Requirement Categories

- `image`: prompt_total=1, prediction_total=1, match_total=1, precision=1.0, recall=1.0
- `kind`: prompt_total=9, prediction_total=6, match_total=6, precision=1.0, recall=0.6666666666666666
- `label`: prompt_total=1, prediction_total=1, match_total=1, precision=1.0, recall=1.0
- `metadata.name`: prompt_total=4, prediction_total=4, match_total=3, precision=0.75, recall=0.75
- `serviceAccountName`: prompt_total=2, prediction_total=1, match_total=1, precision=1.0, recall=0.5

## Missing Required Fields By Kind

- `DaemonSet`: incomplete_resources=3; most common missing groups: spec.template.spec.containers (3)

## Example Incomplete Predictions

- `q13::question`: DaemonSet: spec.template.spec.containers
- `q13::question_simplified`: DaemonSet: spec.template.spec.containers
- `q21::question_simplified`: DaemonSet: spec.template.spec.containers

## Conclusions

- The recomputation confirms that the run can be re-evaluated offline from persisted artifacts alone; no new model inference was needed.
- Prompt-requirement coverage is materially stronger than exact YAML equality: `average_prompt_requirement_f1 = 0.22826086956521738` versus `parsed_equal_rate = 0.043478260869565216`.
- Required-field validity is high once the prediction reaches YAML parseability: `average_required_field_complete_resource_rate = 0.5` and `required_field_complete_sample_rate = 0.5`.
- The largest remaining bottleneck is still upstream structural generation, not minimal resource completeness: `structured_output_parse_success_rate = 0.92` and `yaml_parse_success_rate = 0.34782608695652173`.
- This means the baseline often captures coarse intent and minimal field structure, but still fails too often in full structural realization and exact reconstruction.
