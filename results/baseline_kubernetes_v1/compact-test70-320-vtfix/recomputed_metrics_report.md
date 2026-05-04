# Baseline Offline Recomputed Metrics Report

- Run directory: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\results\baseline_kubernetes_v1\compact-test70-320-vtfix`
- Dataset used for recomputation: `C:\Users\mario\OneDrive\Documentos\UPM\Master_Data\llm-structured-semantic-generation\data\processed\kubernetes_v1\dataset_structural_targets.jsonl`
- Row count: `70`
- Evaluated count: `62`

## Headline metrics

- `structured_output_parse_success_rate`: old=0.8857142857142857 new=0.8857142857142857
- `yaml_parse_success_rate`: old=0.46774193548387094 new=0.46774193548387094
- `parsed_equal_rate`: old=0.016129032258064516 new=0.016129032258064516
- `average_line_text_f1`: old=0.3174745620804379 new=0.3174745620804377
- `average_semantic_key_f1`: old=0.40818066674827885 new=0.4081806667482789
- `average_prompt_requirement_precision`: old=None new=0.3991935483870968
- `average_prompt_requirement_recall`: old=None new=0.2889016897081414
- `average_prompt_requirement_f1`: old=None new=0.3230414746543779
- `prompt_requirement_exact_match_rate`: old=None new=0.0967741935483871
- `average_required_field_presence_rate`: old=None new=0.9028679653679654
- `average_required_field_complete_resource_rate`: old=None new=0.6964285714285714
- `required_field_complete_sample_rate`: old=None new=0.6428571428571429

## Error profile

- Structured-output failures before evaluation: `8`
- YAML failures after structured parsing: `33`
- Parsed-equal rows: `1`

## Prompt Requirement Categories

- `env`: prompt_total=2, prediction_total=2, match_total=2, precision=1.0, recall=1.0
- `image`: prompt_total=7, prediction_total=4, match_total=4, precision=1.0, recall=0.5714285714285714
- `kind`: prompt_total=52, prediction_total=32, match_total=29, precision=0.90625, recall=0.5576923076923077
- `label`: prompt_total=10, prediction_total=6, match_total=4, precision=0.6666666666666666, recall=0.4
- `metadata.name`: prompt_total=13, prediction_total=11, match_total=11, precision=1.0, recall=0.8461538461538461
- `namespace`: prompt_total=3, prediction_total=3, match_total=0, precision=0.0, recall=0.0
- `port`: prompt_total=3, prediction_total=1, match_total=1, precision=1.0, recall=0.3333333333333333
- `replicas`: prompt_total=2, prediction_total=2, match_total=2, precision=1.0, recall=1.0

## Missing Required Fields By Kind

- `DaemonSet`: incomplete_resources=5; most common missing groups: spec.selector (4), spec.template.spec.containers (3), spec.template (2)
- `StatefulSet`: incomplete_resources=1; most common missing groups: spec.serviceName (1), spec.selector (1), spec.template (1), spec.template.spec.containers (1)
- `Deployment`: incomplete_resources=1; most common missing groups: spec.template (1), spec.template.spec.containers (1)
- `Pod`: incomplete_resources=2; most common missing groups: spec.containers (2)
- `Service`: incomplete_resources=1; most common missing groups: metadata.name (1), spec.ports (1)
- `HorizontalPodAutoscaler`: incomplete_resources=1; most common missing groups: spec.maxReplicas (1)

## Example Incomplete Predictions

- `q12::question`: DaemonSet: spec.selector, spec.template, spec.template.spec.containers
- `q15::question_simplified`: DaemonSet: spec.template.spec.containers
- `q23::question`: DaemonSet: spec.selector, spec.template, spec.template.spec.containers
- `q40::question`: DaemonSet: spec.selector
- `q63::question_simplified`: DaemonSet: spec.selector
- `q83::question`: Deployment: spec.template, spec.template.spec.containers
- `q195::question`: HorizontalPodAutoscaler: spec.maxReplicas
- `q264::question`: StatefulSet: spec.serviceName, spec.selector, spec.template, spec.template.spec.containers | Service: metadata.name, spec.ports
- `q279::question`: Pod: spec.containers
- `q279::question_simplified`: Pod: spec.containers

## Conclusions

- The recomputation confirms that the run can be re-evaluated offline from persisted artifacts alone; no new model inference was needed.
- Prompt-requirement coverage is materially stronger than exact YAML equality: `average_prompt_requirement_f1 = 0.3230414746543779` versus `parsed_equal_rate = 0.016129032258064516`.
- Required-field validity is high once the prediction reaches YAML parseability: `average_required_field_complete_resource_rate = 0.6964285714285714` and `required_field_complete_sample_rate = 0.6428571428571429`.
- The largest remaining bottleneck is still upstream structural generation, not minimal resource completeness: `structured_output_parse_success_rate = 0.8857142857142857` and `yaml_parse_success_rate = 0.46774193548387094`.
- This means the baseline often captures coarse intent and minimal field structure, but still fails too often in full structural realization and exact reconstruction.
