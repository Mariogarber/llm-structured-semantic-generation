# Resultado DPO v2 lr1e-5 como segunda iteracion - 2026-06-10

Document type: run result

## Resumen

Este documento registra el segundo experimento DPO v2 sobre Kubernetes v1 usando
la misma configuracion metodologica que la run agresiva anterior, pero reduciendo
el `learning_rate` de `1e-4` a `1e-5`.

La conclusion principal es positiva y matizada. La run `lr=1e-5` conserva la
capacidad de mover el modelo sobre el dataset de preferencias v2 sin romper el
contrato estructural `blocks_tsv_v1`. Frente a la run agresiva `lr=1e-4`, mejora
de forma muy clara en parseabilidad, F1 de lineas, niveles, validez Kubernetes y
cumplimiento del prompt. Frente a SFT y DPO v1, no es una victoria global en
todas las metricas: baja algo en fidelidad de linea y exactitud de nivel, pero
mejora prompt F1, semantic key F1, parsed equal rate y el score medio de validez
Kubernetes.

Por tanto, esta run si debe considerarse un candidato experimental serio para
analisis posterior, aunque todavia no sustituye automaticamente al SFT como
referencia principal.

## Artefactos

- Run:
  `dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609`
- Directorio:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609/`
- Dataset de preferencias:
  `results/dpo_kubernetes_v1/preference_annotation/agent-full-auto-v2/preferences_final.jsonl`
- Checkpoint base y referencia:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/checkpoints/checkpoint-step-57/`
- Checkpoint final y mejor checkpoint:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609/checkpoints/checkpoint-step-87/`
- Metricas finales:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609/metrics.json`
- Predicciones de validacion:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609/validation_predictions.jsonl`
- Progreso de validacion:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609/validation_metrics_progress.jsonl`

La run termino con:

- `status`: `completed`
- `completed_at`: `2026-06-10T15:11:46Z`
- hora local aproximada en Espana: `2026-06-10 17:11`
- `global_step`: `87`
- `epoch`: `3`
- `oom_skipped_batches`: `0`
- `best_checkpoint`: `checkpoint-step-87`

## Configuracion

La configuracion efectiva fue:

| Parametro | Valor |
| --- | ---: |
| `beta` | `0.30` |
| `learning_rate` | `1e-5` |
| `epochs` | `3` |
| `batch_size` | `1` |
| `gradient_accumulation_steps` | `8` |
| `checkpoint_steps` | `29` |
| `two_thirds_validation_samples` | `0` |
| W&B | `online` |

La interpretacion metodologica sigue siendo:

```text
serialized_sft -> DPO v1 -> DPO v2 lr1e-5
```

El argumento de trainer `--sft-adapter-path` apunta al checkpoint DPO v1 por
compatibilidad con la interfaz existente. Operativamente, el punto de partida y
la politica de referencia son el checkpoint DPO v1 `checkpoint-step-57`.

## Nota Operacional

Los `reference_logps` se reutilizaron desde la run agresiva `lr=1e-4`, porque
dependen del dataset de preferencias, del checkpoint de referencia, de la
tokenizacion y de la longitud maxima, pero no del `learning_rate`.

La validacion final fue pausada despues de `6/70` ejemplos y reanudada despues
con el mismo `run_id`. La funcion de validacion es resumible por `unit_id`, por
lo que la reanudacion continuo sobre los artefactos parciales existentes y
finalizo correctamente con `70/70` ejemplos.

Los checkpoints observados fueron:

| Checkpoint | Interpretacion |
| --- | --- |
| `checkpoint-step-29` | fin aproximado de la primera epoca |
| `checkpoint-step-58` | fin aproximado de la segunda epoca |
| `checkpoint-step-87` | fin de entrenamiento y mejor checkpoint final |

## Dinamica De Entrenamiento

La curva de entrenamiento fue mucho mas contenida que en la run `lr=1e-4`. El
reward margin aumento de forma gradual y no se saturo en valores extremos.

| Epoca | Steps | Mean loss | Mean reward margin | Mean reward accuracy | Mean grad norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 29 | 0.6748 | 0.0428 | 0.6379 | 15.7842 |
| 1 | 29 | 0.4970 | 0.4931 | 0.9224 | 11.9869 |
| 2 | 29 | 0.4280 | 0.7213 | 0.9526 | 10.5762 |

Esta dinamica contrasta con la run `lr=1e-4`, donde el reward margin se disparo
rapidamente y la loss quedo casi saturada. En esta run, el modelo aprende la
preferencia sin producir el desplazamiento destructivo que se observo con el
learning rate mayor.

## Metricas Finales

La validacion completo `70/70` ejemplos.

| Metrica | Valor |
| --- | ---: |
| `structured_output_parse_success_rate` | 1.0000 |
| `yaml_parse_success_rate` | 0.9857 |
| `block_parse_success_rate` | 0.9857 |
| `parsed_equal_rate` | 0.1571 |
| `document_count_match_rate` | 0.9429 |
| `line_count_match_rate` | 0.3000 |
| `average_content_exact_match_rate` | 0.5714 |
| `average_level_exact_match_rate` | 0.7196 |
| `average_level_mae` | 0.3422 |
| `average_line_text_f1` | 0.7986 |
| `average_prompt_requirement_f1` | 0.8583 |
| `prompt_requirement_exact_match_rate` | 0.5714 |
| `average_semantic_key_f1` | 0.9596 |
| `average_required_field_complete_resource_rate` | 0.9855 |
| `average_kubernetes_domain_validity_score` | 0.8381 |
| `average_kubernetes_domain_validity_level` | 3.9429 |
| `kubernetes_domain_gate_pass_rate` | 0.1143 |

## Comparacion Principal

La comparacion usa el mismo split `validation` de `70` ejemplos y compara:

- SFT serializado:
  `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/validation_metrics_recomputed.json`
- DPO v1 beta `0.10`:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/metrics.json`
- DPO v2 agresivo `lr=1e-4`:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605/metrics.json`
- DPO v2 `lr=1e-5`:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609/metrics.json`

| Metrica | SFT | DPO v1 | DPO v2 lr1e-4 | DPO v2 lr1e-5 | Delta lr1e-5 - DPO v1 | Delta lr1e-5 - lr1e-4 | Delta lr1e-5 - SFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yaml_parse_success_rate` | 0.9857 | 0.9714 | 0.9483 | 0.9857 | +0.0143 | +0.0374 | 0.0000 |
| `structured_output_parse_success_rate` | 1.0000 | 1.0000 | 0.8286 | 1.0000 | 0.0000 | +0.1714 | 0.0000 |
| `block_parse_success_rate` | 0.9857 | 0.9714 | 0.9483 | 0.9857 | +0.0143 | +0.0374 | 0.0000 |
| `parsed_equal_rate` | 0.1143 | 0.1429 | 0.0172 | 0.1571 | +0.0143 | +0.1399 | +0.0429 |
| `document_count_match_rate` | 0.9286 | 0.9429 | 0.8621 | 0.9429 | 0.0000 | +0.0808 | +0.0143 |
| `line_count_match_rate` | 0.3571 | 0.3286 | 0.1207 | 0.3000 | -0.0286 | +0.1793 | -0.0571 |
| `average_content_exact_match_rate` | 0.6124 | 0.6060 | 0.3507 | 0.5714 | -0.0345 | +0.2208 | -0.0409 |
| `average_level_exact_match_rate` | 0.7578 | 0.7422 | 0.5358 | 0.7196 | -0.0227 | +0.1837 | -0.0382 |
| `average_line_text_f1` | 0.8206 | 0.8092 | 0.5888 | 0.7986 | -0.0106 | +0.2098 | -0.0220 |
| `average_level_mae` | 0.2723 | 0.2630 | 0.6066 | 0.3422 | +0.0792 | -0.2643 | +0.0699 |
| `average_prompt_requirement_f1` | 0.8531 | 0.8368 | 0.7918 | 0.8583 | +0.0214 | +0.0665 | +0.0052 |
| `average_semantic_key_f1` | 0.9552 | 0.9406 | 0.8924 | 0.9596 | +0.0191 | +0.0672 | +0.0044 |
| `average_required_field_complete_resource_rate` | 1.0000 | 1.0000 | 0.9455 | 0.9855 | -0.0145 | +0.0401 | -0.0145 |
| `average_kubernetes_domain_validity_score` | 0.8310 | 0.8286 | 0.7701 | 0.8381 | +0.0095 | +0.0680 | +0.0071 |
| `average_kubernetes_domain_validity_level` | 3.9000 | 3.9429 | 3.2241 | 3.9429 | 0.0000 | +0.7187 | +0.0429 |
| `kubernetes_domain_gate_pass_rate` | 0.1429 | 0.1286 | 0.0862 | 0.1143 | -0.0143 | +0.0281 | -0.0286 |
| `average_bleu_score` | 0.7327 | 0.7277 | 0.5108 | 0.7185 | -0.0093 | +0.2077 | -0.0143 |
| `average_rougeL_f1` | 0.8462 | 0.8417 | 0.6579 | 0.8272 | -0.0145 | +0.1693 | -0.0190 |

## Validez Kubernetes Por Niveles

| Nivel Kubernetes | SFT | DPO v1 | DPO v2 lr1e-4 | DPO v2 lr1e-5 | Delta lr1e-5 - DPO v1 | Delta lr1e-5 - lr1e-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kubernetes_level_0_pass_rate` | 0.9857 | 0.9714 | 0.9483 | 0.9857 | +0.0143 | +0.0374 |
| `kubernetes_level_1_pass_rate` | 0.9857 | 0.9714 | 0.9483 | 0.9857 | +0.0143 | +0.0374 |
| `kubernetes_level_2_pass_rate` | 0.9714 | 0.9714 | 0.8966 | 0.9714 | 0.0000 | +0.0749 |
| `kubernetes_level_3_pass_rate` | 0.9143 | 0.9571 | 0.6897 | 0.9571 | 0.0000 | +0.2675 |
| `kubernetes_level_4_pass_rate` | 0.9000 | 0.9429 | 0.6552 | 0.9286 | -0.0143 | +0.2734 |
| `kubernetes_level_5_pass_rate` | 0.1429 | 0.1286 | 0.0862 | 0.1143 | -0.0143 | +0.0281 |

La run `lr=1e-5` recupera la estabilidad en niveles `0`, `1` y `2`, iguala a
DPO v1 en nivel `3`, y queda solo ligeramente por debajo en niveles `4` y `5`.
Frente a `lr=1e-4`, la mejora es clara en todos los niveles.

## Perfil De Errores De Dominio

| Error Kubernetes | SFT | DPO v1 | DPO v2 lr1e-4 | DPO v2 lr1e-5 | Delta lr1e-5 - DPO v1 | Delta lr1e-5 - lr1e-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `missing_resource_requirement` | 261 | 253 | 437 | 254 | +1 | -183 |
| `missing_run_as_non_root` | 71 | 69 | 115 | 69 | 0 | -46 |
| `missing_read_only_root_filesystem` | 71 | 69 | 116 | 69 | 0 | -47 |
| `latest_image_tag` | 67 | 65 | 61 | 65 | 0 | +4 |
| `container_missing_image` | 0 | 0 | 53 | 0 | 0 | -53 |
| `volume_mount_without_volume` | 5 | 1 | 10 | 1 | 0 | -9 |
| `service_selector_without_workload` | 1 | 1 | 2 | 2 | +1 | 0 |
| `required_field` | 0 | 0 | 9 | 1 | +1 | -8 |
| `yaml_parse` | 1 | 2 | 3 | 1 | -1 | -2 |
| `kubernetes_identity` | 1 | 0 | 2 | 0 | 0 | -2 |
| `invalid_port` | 0 | 0 | 1 | 0 | 0 | -1 |

El perfil de errores confirma que `lr=1e-5` no reproduce el colapso de la run
agresiva. Los errores de recursos, seguridad y campos obligatorios vuelven a un
perfil muy parecido a DPO v1, con una mejora clara frente a `lr=1e-4`.

## Interpretacion

Este resultado separa dos efectos que estaban mezclados en la run agresiva. La
segunda iteracion DPO sobre preferencias v2 no es intrinsecamente destructiva;
lo destructivo era aplicar esa segunda iteracion con un `learning_rate` demasiado
alto. Al bajar a `1e-5`, el modelo conserva la regularidad del formato
`blocks_tsv_v1` y recupera casi por completo la parseabilidad y la coherencia de
niveles.

El resultado tampoco debe exagerarse. Aunque mejora `prompt_requirement_f1`,
`average_semantic_key_f1`, `parsed_equal_rate` y
`average_kubernetes_domain_validity_score`, queda algo por debajo de SFT y DPO
v1 en `average_line_text_f1`, `average_level_exact_match_rate`, `average_level_mae`
y `kubernetes_domain_gate_pass_rate`. Es decir, el alineamiento con preferencias
v2 parece aportar senales utiles de prompt y dominio, pero todavia tiene un coste
pequeno en fidelidad estructural fina.

La diferencia mas importante frente a `lr=1e-4` es metodologica: ahora si vemos
un desplazamiento controlado. Esto convierte a `lr=1e-5` en una configuracion
razonable para seguir explorando, mientras que `lr=1e-4` queda como limite
superior claramente demasiado agresivo.

## Conclusion Experimental

`dpo-v2-from-dpo-v1-beta030-lr1e5-e3-20260609` es el mejor resultado DPO v2
obtenido hasta este punto. No gana de forma absoluta a SFT en todas las metricas,
pero si demuestra que el dataset de preferencias v2 puede usarse sin destruir el
contrato estructural cuando el learning rate es mas conservador.

La conclusion practica es:

- `lr=1e-4` fue demasiado agresivo;
- `lr=1e-5` es una configuracion viable;
- DPO v2 mejora senales de prompt y dominio frente a DPO v1;
- el cuello principal sigue siendo el gate Kubernetes de nivel `5`;
- las proximas runs deberian explorar ajustes alrededor de este punto, no volver
  a la intensidad de `1e-4`.

Siguientes variantes razonables:

- `beta=0.30`, `lr=1e-5`, `1` epoca para comprobar si `3` epocas son necesarias;
- `beta=0.30`, `lr=2e-5`, `3` epocas como punto intermedio;
- DPO v2 desde SFT original con `lr=1e-5`, para separar el efecto del dataset v2
  del efecto de segunda iteracion sobre DPO v1;
- revision del scoring v2 para atacar mas directamente los errores de nivel `5`
  sin penalizar la fidelidad de niveles.
