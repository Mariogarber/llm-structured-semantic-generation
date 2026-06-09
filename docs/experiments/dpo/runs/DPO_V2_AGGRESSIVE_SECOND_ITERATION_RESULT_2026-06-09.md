# Resultado DPO v2 agresivo como segunda iteracion - 2026-06-09

Document type: run result

## Resumen

Este documento registra el resultado final del segundo experimento DPO sobre
Kubernetes v1 usando el dataset automatico de preferencias v2. La run se planteo
deliberadamente como una prueba agresiva: partir del checkpoint DPO v1,
aumentar la libertad relativa con `beta=0.30`, subir el `learning_rate` a
`1e-4` y entrenar durante `3` epocas.

La conclusion principal es que la ejecucion fue correcta, pero el resultado no
debe adoptarse como nuevo checkpoint principal. El modelo optimizo con fuerza el
objetivo DPO, completo el entrenamiento y la validacion, y dejo un checkpoint
final reproducible. Sin embargo, frente a `serialized_sft` y frente a DPO v1,
degrada de forma clara la estabilidad estructural, la coincidencia de niveles,
el F1 de lineas y la validez Kubernetes. Por tanto, esta run es util como medida
del limite superior de agresividad, no como mejora del pipeline.

## Artefactos

- Run:
  `dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605`
- Directorio:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605/`
- Plan previo:
  `docs/experiments/dpo/runs/DPO_V2_AGGRESSIVE_SECOND_ITERATION_PLAN_2026-06-05.md`
- Dataset de preferencias:
  `results/dpo_kubernetes_v1/preference_annotation/agent-full-auto-v2/preferences_final.jsonl`
- Checkpoint base y referencia:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/checkpoints/checkpoint-step-57/`
- Checkpoint final:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605/checkpoints/checkpoint-step-87/`
- Metricas finales:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605/metrics.json`
- Predicciones de validacion:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605/validation_predictions.jsonl`
- Progreso de validacion:
  `results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605/validation_metrics_progress.jsonl`

La run termino con:

- `status`: `completed`
- `completed_at`: `2026-06-07T18:57:10Z`
- hora local aproximada en Espana: `2026-06-07 20:57`
- `global_step`: `87`
- `oom_skipped_batches`: `0`
- `best_checkpoint`: `checkpoint-step-87`

## Configuracion

La configuracion efectiva fue:

| Parametro | Valor |
| --- | ---: |
| `beta` | `0.30` |
| `learning_rate` | `1e-4` |
| `epochs` | `3` |
| `batch_size` | `1` |
| `gradient_accumulation_steps` | `8` |
| `checkpoint_steps` | `29` |
| `two_thirds_validation_samples` | `0` |
| W&B | `online` |

El dataset v2 contiene `229` pares de preferencia. Con acumulacion de gradiente
`8`, una epoca equivale aproximadamente a `29` pasos de optimizacion, por lo que
`3` epocas producen los `87` pasos finales observados.

Como ya se explicaba en el plan, esta run debe interpretarse como:

```text
serialized_sft -> DPO v1 -> DPO v2 agresivo
```

No es una run independiente desde el SFT original. El argumento de trainer
`--sft-adapter-path` apunta al checkpoint DPO v1 por compatibilidad con la
interfaz existente, aunque metodologicamente el punto de partida es DPO v1.

## Nota Operacional

La ejecucion fue lenta e irregular. La fase de entrenamiento se interrumpio y
reanudo varias veces desde checkpoints, principalmente desde `checkpoint-step-58`.
Esto no invalida el resultado final porque el trainer persiste estado de modelo,
optimizador y scheduler en los checkpoints.

Los checkpoints observados fueron:

| Checkpoint | Interpretacion |
| --- | --- |
| `checkpoint-step-29` | fin aproximado de la primera epoca |
| `checkpoint-step-58` | fin aproximado de la segunda epoca |
| `checkpoint-step-87` | fin de entrenamiento y mejor checkpoint final |

La fuente de verdad para el resultado final es `metrics.json` junto con
`state.json`. El archivo `train_log.jsonl` incluye entradas repetidas de algunos
pasos por las reanudaciones, por lo que debe usarse con cuidado para analisis de
tiempos o curvas.

## Metricas Finales

La validacion proceso `70` filas del split `validation`. El stack de metricas
reporta `evaluated_count=58`, porque varias metricas posteriores solo se pueden
computar sobre salidas que superan la barrera de parseo estructural.

| Metrica | Valor |
| --- | ---: |
| `structured_output_parse_success_rate` | 0.8286 |
| `yaml_parse_success_rate` | 0.9483 |
| `block_parse_success_rate` | 0.9483 |
| `parsed_equal_rate` | 0.0172 |
| `document_count_match_rate` | 0.8621 |
| `line_count_match_rate` | 0.1207 |
| `average_content_exact_match_rate` | 0.3507 |
| `average_level_exact_match_rate` | 0.5358 |
| `average_level_mae` | 0.6066 |
| `average_line_text_f1` | 0.5888 |
| `average_prompt_requirement_f1` | 0.7918 |
| `prompt_requirement_exact_match_rate` | 0.5000 |
| `average_semantic_key_f1` | 0.8924 |
| `average_required_field_complete_resource_rate` | 0.9455 |
| `average_kubernetes_domain_validity_score` | 0.7701 |
| `average_kubernetes_domain_validity_level` | 3.2241 |
| `kubernetes_domain_gate_pass_rate` | 0.0862 |

## Comparacion Con SFT Y DPO v1

La comparacion principal se hace contra:

- SFT serializado:
  `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/validation_metrics_recomputed.json`
- DPO v1 beta `0.10`:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/metrics.json`

| Metrica | SFT | DPO v1 | DPO v2 agresivo | Delta v2 - DPO v1 | Delta v2 - SFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| `yaml_parse_success_rate` | 0.9857 | 0.9714 | 0.9483 | -0.0232 | -0.0374 |
| `structured_output_parse_success_rate` | 1.0000 | 1.0000 | 0.8286 | -0.1714 | -0.1714 |
| `block_parse_success_rate` | 0.9857 | 0.9714 | 0.9483 | -0.0232 | -0.0374 |
| `parsed_equal_rate` | 0.1143 | 0.1429 | 0.0172 | -0.1256 | -0.0970 |
| `document_count_match_rate` | 0.9286 | 0.9429 | 0.8621 | -0.0808 | -0.0665 |
| `line_count_match_rate` | 0.3571 | 0.3286 | 0.1207 | -0.2079 | -0.2365 |
| `average_content_exact_match_rate` | 0.6124 | 0.6060 | 0.3507 | -0.2553 | -0.2617 |
| `average_level_exact_match_rate` | 0.7578 | 0.7422 | 0.5358 | -0.2064 | -0.2220 |
| `average_line_text_f1` | 0.8206 | 0.8092 | 0.5888 | -0.2205 | -0.2318 |
| `average_level_mae` | 0.2723 | 0.2630 | 0.6066 | +0.3436 | +0.3343 |
| `average_prompt_requirement_f1` | 0.8531 | 0.8368 | 0.7918 | -0.0451 | -0.0613 |
| `average_semantic_key_f1` | 0.9552 | 0.9406 | 0.8924 | -0.0481 | -0.0628 |
| `average_required_field_complete_resource_rate` | 1.0000 | 1.0000 | 0.9455 | -0.0545 | -0.0545 |
| `average_kubernetes_domain_validity_score` | 0.8310 | 0.8286 | 0.7701 | -0.0585 | -0.0608 |
| `average_kubernetes_domain_validity_level` | 3.9000 | 3.9429 | 3.2241 | -0.7187 | -0.6759 |
| `kubernetes_domain_gate_pass_rate` | 0.1429 | 0.1286 | 0.0862 | -0.0424 | -0.0567 |
| `average_bleu_score` | 0.7327 | 0.7277 | 0.5108 | -0.2170 | -0.2219 |
| `average_rougeL_f1` | 0.8462 | 0.8417 | 0.6579 | -0.1837 | -0.1882 |

El resultado es una degradacion general, no una mezcla compensada. Las metricas
de parseo bajan, pero el dano mas importante esta en la representacion
estructural y en la fidelidad de la superficie de salida: `line_text_f1`,
`level_exact_match_rate`, `level_mae`, `line_count_match_rate` y
`parsed_equal_rate`.

## Validez Kubernetes Por Niveles

| Nivel Kubernetes | SFT | DPO v1 | DPO v2 agresivo | Delta v2 - DPO v1 |
| --- | ---: | ---: | ---: | ---: |
| `kubernetes_level_0_pass_rate` | 0.9857 | 0.9714 | 0.9483 | -0.0232 |
| `kubernetes_level_1_pass_rate` | 0.9857 | 0.9714 | 0.9483 | -0.0232 |
| `kubernetes_level_2_pass_rate` | 0.9714 | 0.9714 | 0.8966 | -0.0749 |
| `kubernetes_level_3_pass_rate` | 0.9143 | 0.9571 | 0.6897 | -0.2675 |
| `kubernetes_level_4_pass_rate` | 0.9000 | 0.9429 | 0.6552 | -0.2877 |
| `kubernetes_level_5_pass_rate` | 0.1429 | 0.1286 | 0.0862 | -0.0424 |

Esta tabla es especialmente relevante porque DPO v1 habia mostrado una mejora
local en niveles `3` y `4`. La segunda iteracion agresiva no conserva esa mejora:
los niveles intermedios caen con fuerza. Por tanto, el problema no es solo que
el gate completo siga bajo, sino que tambien se pierde parte de la coherencia
Kubernetes intermedia que DPO v1 parecia haber ganado.

## Perfil De Errores De Dominio

Errores Kubernetes finales de DPO v2:

| Error Kubernetes | Conteo |
| --- | ---: |
| `missing_resource_requirement` | 437 |
| `missing_read_only_root_filesystem` | 116 |
| `missing_run_as_non_root` | 115 |
| `latest_image_tag` | 61 |
| `container_missing_image` | 53 |
| `volume_mount_without_volume` | 10 |
| `required_field` | 9 |
| `yaml_parse` | 3 |
| `kubernetes_identity` | 2 |
| `service_selector_without_workload` | 2 |
| `invalid_port` | 1 |

El aumento de `missing_resource_requirement` frente a DPO v1 es la senal mas
llamativa. La run agresiva genera salidas que todavia pueden parsear YAML con
frecuencia razonable, pero pierden muchas propiedades de calidad estatica y de
dominio que el evaluador penaliza en niveles altos.

## Interpretacion

El experimento cumple su objetivo como prueba de agresividad. Al usar DPO v1
como punto de partida, `beta=0.30`, `learning_rate=1e-4` y `3` epocas, el modelo
se mueve mucho mas que en el primer barrido de DPO. Esta vez si se observa un
cambio claro respecto al checkpoint anterior, pero la direccion no es deseable.

La lectura mas plausible es sobreoptimizacion del proxy de preferencias v2. El
entrenamiento empuja con fuerza sobre un conjunto pequeno y estricto de pares,
pero ese desplazamiento rompe regularidades que el modelo necesitaba para
mantener estable la superficie `blocks_tsv_v1`: conteo de lineas, texto exacto,
niveles y reconstruccion parser-facing. En este proyecto, esa superficie no es
un detalle cosmetico; es la interfaz estructural que permite controlar la salida
YAML. Si se degrada, el alineamiento automatico deja de ayudar aunque el
objetivo DPO de entrenamiento parezca optimizado.

Tambien es importante que la mejora esperada en dominio Kubernetes no aparece.
El `average_kubernetes_domain_validity_score` baja de `0.8286` en DPO v1 a
`0.7701`, y el nivel medio de validez baja de `3.9429` a `3.2241`. La run no
solo sacrifica similitud textual para ganar dominio; sacrifica ambas cosas.

## Conclusion Experimental

`dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605` debe conservarse como artefacto
negativo/informativo. Demuestra que una segunda iteracion DPO demasiado
agresiva si puede mover el modelo, pero que el movimiento daña el contrato
estructural y no mejora el gate Kubernetes.

No debe reemplazar a:

- `serialized_sft` como referencia supervisada;
- DPO v1 `checkpoint-step-57` como primer resultado DPO completo.

La siguiente exploracion razonable no deberia repetir esta intensidad. Opciones
mas justificadas serian:

- mantener el dataset v2 pero reducir a `1` epoca;
- mantener `3` epocas pero bajar `learning_rate` hacia `1e-5` o `2e-5`;
- probar DPO v2 desde el SFT original para separar el efecto del dataset v2 del
  efecto de segunda iteracion sobre DPO v1;
- construir un subconjunto de preferencias v2 mas centrado en negativos duros
  que preserven la superficie `blocks_tsv_v1`.

Hasta tener una run menos agresiva que conserve parseabilidad y niveles, este
resultado debe citarse como evidencia de que el alineamiento automatico necesita
regularizacion operacional y no puede tratarse como una mejora monotona.
