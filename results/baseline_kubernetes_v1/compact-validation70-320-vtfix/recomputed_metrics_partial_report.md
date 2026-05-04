# Baseline Offline Recomputed Metrics Report

- Run directory: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\results\baseline_kubernetes_v1\compact-validation70-320-vtfix`
- Dataset used for recomputation: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\data\processed\kubernetes_v1\dataset_structural_targets.jsonl`
- Row count: `14`
- Evaluated count: `12`

## Headline metrics

- `structured_output_parse_success_rate`: old=None new=0.8571428571428571
- `yaml_parse_success_rate`: old=None new=0.4166666666666667
- `parsed_equal_rate`: old=None new=0.0
- `average_line_text_f1`: old=None new=0.26453947974853903
- `average_semantic_key_f1`: old=None new=0.3974358974358974
- `average_prompt_requirement_precision`: old=None new=0.3333333333333333
- `average_prompt_requirement_recall`: old=None new=0.3333333333333333
- `average_prompt_requirement_f1`: old=None new=0.3333333333333333
- `prompt_requirement_exact_match_rate`: old=None new=0.3333333333333333
- `average_required_field_presence_rate`: old=None new=0.875
- `average_required_field_complete_resource_rate`: old=None new=0.25
- `required_field_complete_sample_rate`: old=None new=0.25

## Error profile

- Structured-output failures before evaluation: `2`
- YAML failures after structured parsing: `7`
- Parsed-equal rows: `0`

## Prompt Requirement Categories

- `image`: prompt_total=1, prediction_total=1, match_total=1, precision=1.0, recall=1.0
- `kind`: prompt_total=5, prediction_total=4, match_total=4, precision=1.0, recall=0.8
- `label`: prompt_total=1, prediction_total=1, match_total=1, precision=1.0, recall=1.0
- `metadata.name`: prompt_total=2, prediction_total=2, match_total=2, precision=1.0, recall=1.0

## Missing Required Fields By Kind

- `DaemonSet`: incomplete_resources=3; most common missing groups: spec.template.spec.containers (3)

## Example Incomplete Predictions

- `q13::question`: DaemonSet: spec.template.spec.containers
- `q13::question_simplified`: DaemonSet: spec.template.spec.containers
- `q21::question_simplified`: DaemonSet: spec.template.spec.containers

## Conclusions

- The recomputation confirms that the run can be re-evaluated offline from persisted artifacts alone; no new model inference was needed.
- Prompt-requirement coverage is materially stronger than exact YAML equality: `average_prompt_requirement_f1 = 0.3333333333333333` versus `parsed_equal_rate = 0.0`.
- Required-field validity is high once the prediction reaches YAML parseability: `average_required_field_complete_resource_rate = 0.25` and `required_field_complete_sample_rate = 0.25`.
- The largest remaining bottleneck is still upstream structural generation, not minimal resource completeness: `structured_output_parse_success_rate = 0.8571428571428571` and `yaml_parse_success_rate = 0.4166666666666667`.
- This means the baseline often captures coarse intent and minimal field structure, but still fails too often in full structural realization and exact reconstruction.
