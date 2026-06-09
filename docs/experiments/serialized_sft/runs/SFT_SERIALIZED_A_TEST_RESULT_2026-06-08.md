# Resultado de test SFT Arquitectura A - 2026-06-08

Document type: run result

## Resumen

Se ejecuto inferencia greedy sobre el split `test` completo de Kubernetes v1
para la rama `serialized_sft`.

El checkpoint usado fue el seleccionado previamente en validacion:

```text
results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/checkpoints/checkpoint-step-159
```

Por tanto, este resultado no usa `test` para seleccionar modelo. El split de
test se usa solo como evaluacion final de la rama serializada.

## Artefactos

- Run de inferencia: `serialized-sft-a-v1-test-20260608`
- Directorio:
  `results/sft_kubernetes_v1/serialized-sft-a-v1-test-20260608/`
- Predicciones: `predictions.jsonl`
- Metricas: `metrics.json`
- Estado resumible: `state.json`
- Configuracion: `config.json`
- Script: `scripts/run_kubernetes_sft_inference.py`
- Split: `test`
- Ejemplos evaluados: `70/70`
- Decodificacion: greedy
- `max_new_tokens`: `512`

## Metricas principales

| Metrica | Test |
| --- | ---: |
| `generation_success_rate` | 1.0000 |
| `structured_output_parse_success_rate` | 1.0000 |
| `yaml_parse_success_rate` | 0.9714 |
| `block_parse_success_rate` | 0.9714 |
| `parsed_equal_rate` | 0.1429 |
| `document_count_match_rate` | 0.7143 |
| `line_count_match_rate` | 0.2857 |
| `average_line_text_f1` | 0.7909 |
| `average_level_exact_match_rate` | 0.6298 |
| `average_level_mae` | 0.5206 |
| `average_prompt_requirement_f1` | 0.6932 |
| `prompt_requirement_exact_match_rate` | 0.2286 |
| `average_semantic_key_f1` | 0.9211 |
| `average_required_field_complete_resource_rate` | 0.9632 |
| `required_field_complete_sample_rate` | 0.9559 |
| `average_kubernetes_domain_validity_score` | 0.8167 |
| `kubernetes_domain_gate_pass_rate` | 0.2143 |
| `average_bleu_score` | 0.6784 |
| `average_rougeL_f1` | 0.7920 |

## Perfil de error

El parseo de bloques estructurados funciono en todos los ejemplos. El parseo
YAML fallo en:

- `q205::question`
- `q279::question`

Las categorias de error de dominio mas frecuentes fueron:

- `missing_resource_requirement`: `220`
- `missing_run_as_non_root`: `57`
- `missing_read_only_root_filesystem`: `57`
- `latest_image_tag`: `34`
- `volume_mount_without_volume`: `7`
- `kubernetes_identity`: `7`
- `required_field`: `5`
- `yaml_parse`: `2`

## Lectura

El modelo SFT serializado se mantiene como un control supervisado fuerte en
`test`: genera una superficie `blocks_tsv_v1` usable por el parser para todos
los ejemplos, y casi todos los YAML reconstruidos son sintacticamente
parseables.

El resultado es mas debil que el run documentado de validacion en metricas de
jerarquia y adecuacion al prompt, especialmente:

- `average_level_exact_match_rate`: `0.6298`
- `average_prompt_requirement_f1`: `0.6932`
- `prompt_requirement_exact_match_rate`: `0.2286`

Esto sugiere que la rama serializada generaliza estructuralmente, pero todavia
tiene dificultades con algunos requisitos complejos del prompt y con la
fidelidad jerarquica exacta en ejemplos de test no usados para seleccion.

## Limitaciones

- Las metricas de dominio Kubernetes son checks automaticos aproximados, no una
  validacion completa contra schema Kubernetes.
- El exito del parser significa que el contrato de bloques fue usable para la
  reconstruccion; no prueba correccion semantica.
- `parsed_equal_rate` sigue siendo bajo, asi que el resultado debe leerse como
  adecuacion de generacion estructurada, no como reproduccion exacta del target.
- La perplexity no se calculo en este run.
