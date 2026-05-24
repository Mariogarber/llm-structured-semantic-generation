# Resultado de validacion two_head_sft - 2026-05-19

## Resumen

Este documento registra las metricas finales del primer run completo
`two_head_sft` para Kubernetes v1 y las compara con el SFT serializado normal
(`serialized_sft`). La comparacion se limita aqui a metricas de validacion. No
incluye todavia auditoria cualitativa de errores, inspeccion de ejemplos ni
diagnostico causal de los fallos observados.

Nota de trazabilidad: despues de esta evaluacion se anadio una normalizacion
superficial acotada para `content_blocks_v1` en
`scripts/train_kubernetes_two_head_sft.py`. Las metricas de este documento son
las metricas originales del primer run finalizado; cualquier recomputo con esa
normalizacion debe marcarse como evaluacion postprocesada.

La comparacion principal del proyecto queda ahora materializada con dos runs
completos sobre el mismo split de validacion:

- `serialized_sft`: predice `blocks_tsv_v1`, incluyendo `level` como texto.
- `two_head_sft`: predice `content_blocks_v1` como texto y predice `level` con
  una cabeza estructural separada.

Ambos resultados deben leerse como validacion, no como resultado final sobre
`test`.

## Artefactos comparados

| Campo | serialized_sft | two_head_sft |
| --- | --- | --- |
| Run id | `serialized-sft-a-v1-20260505-171226` | `two-head-sft-v1-20260516` |
| Directorio | `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/` | `results/two_head_sft_kubernetes_v1/two-head-sft-v1-20260516/` |
| Variante | `serialized_sft` | `two_head_sft` |
| Superficie generada | `blocks_tsv_v1` | `content_blocks_v1` |
| Senal de nivel | columna `level` serializada como texto | cabeza estructural, `record_prefix_state` |
| Checkpoint | `checkpoint-step-159` | `checkpoint-step-159` |
| Split | `validation` | `validation` |
| Filas evaluadas | `70/70` | `70/70` |
| Metricas usadas | `validation_metrics_recomputed.json` | `metrics.json` |

Para `serialized_sft`, las metricas Kubernetes de dominio se toman de:

```text
results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/validation_metrics_recomputed.json
```

Ese fichero recompone las metricas con la bateria KDV, que no estaba presente
en el `metrics.json` original del primer entrenamiento serializado.

## Metricas principales

En la tabla, la columna `Delta` se calcula como:

```text
two_head_sft - serialized_sft
```

Para casi todas las metricas, mayor es mejor. La excepcion principal aqui es
`average_level_mae`, donde menor es mejor.

| Metrica | serialized_sft | two_head_sft | Delta |
| --- | ---: | ---: | ---: |
| `structured_output_parse_success_rate` | 1.0000 | 1.0000 | +0.0000 |
| `yaml_parse_success_rate` | 0.9857 | 0.4857 | -0.5000 |
| `block_parse_success_rate` | 0.9857 | 0.4857 | -0.5000 |
| `parsed_equal_rate` | 0.1143 | 0.0143 | -0.1000 |
| `average_line_text_f1` | 0.8206 | 0.3743 | -0.4462 |
| `average_content_exact_match_rate` | 0.6124 | 0.2798 | -0.3326 |
| `average_level_exact_match_rate` | 0.7578 | 0.3504 | -0.4073 |
| `average_level_mae` | 0.2723 | 0.2392 | -0.0331 |
| `line_count_match_rate` | 0.3571 | 0.2714 | -0.0857 |
| `document_count_match_rate` | 0.9286 | 0.8857 | -0.0429 |

## Metricas parser-facing y de contenido

El resultado principal de esta comparacion es que `serialized_sft` mantiene una
ventaja amplia en reconstruccion parser-facing:

- `yaml_parse_success_rate`: `0.9857` frente a `0.4857`.
- `block_parse_success_rate`: `0.9857` frente a `0.4857`.
- `average_line_text_f1`: `0.8206` frente a `0.3743`.
- `average_content_exact_match_rate`: `0.6124` frente a `0.2798`.

`two_head_sft` no mejora la exactitud discreta de `level`:

- `average_level_exact_match_rate`: `0.3504`, frente a `0.7578` en
  `serialized_sft`.

Sin embargo, su error medio absoluto de nivel es ligeramente menor:

- `average_level_mae`: `0.2392`, frente a `0.2723` en `serialized_sft`.

Esta diferencia debe registrarse con cuidado: `level_mae` y
`level_exact_match_rate` no cuentan exactamente lo mismo. La primera mide
distancia media entre niveles; la segunda exige coincidencia exacta.

## Metricas de adecuacion al prompt y senales semanticas

| Metrica | serialized_sft | two_head_sft | Delta |
| --- | ---: | ---: | ---: |
| `average_prompt_requirement_f1` | 0.8531 | 0.4206 | -0.4325 |
| `prompt_requirement_exact_match_rate` | 0.5857 | 0.2714 | -0.3143 |
| `average_semantic_key_f1` | 0.9552 | 0.4592 | -0.4960 |
| `average_kind_sequence_match_rate` | 0.9143 | 0.4357 | -0.4786 |
| `average_required_field_complete_resource_rate` | 1.0000 | 0.9412 | -0.0588 |
| `required_field_complete_sample_rate` | 1.0000 | 0.9412 | -0.0588 |

En estas metricas, `serialized_sft` tambien queda por encima de `two_head_sft`.
La diferencia es especialmente grande en requisitos del prompt, claves
semanticas aproximadas y secuencia de tipos Kubernetes.

## Metricas Kubernetes KDV

La bateria KDV es acumulativa por niveles. `kubernetes_domain_gate_pass_rate`
equivale a pasar hasta el nivel 5.

| Metrica | serialized_sft | two_head_sft | Delta |
| --- | ---: | ---: | ---: |
| `average_kubernetes_domain_validity_score` | 0.8310 | 0.4238 | -0.4071 |
| `average_kubernetes_domain_validity_level` | 3.9000 | 1.4429 | -2.4571 |
| `kubernetes_domain_gate_pass_rate` | 0.1429 | 0.1286 | -0.0143 |
| `kubernetes_level_0_pass_rate` | 0.9857 | 0.4857 | -0.5000 |
| `kubernetes_level_1_pass_rate` | 0.9857 | 0.4857 | -0.5000 |
| `kubernetes_level_2_pass_rate` | 0.9714 | 0.4571 | -0.5143 |
| `kubernetes_level_3_pass_rate` | 0.9143 | 0.4429 | -0.4714 |
| `kubernetes_level_4_pass_rate` | 0.9000 | 0.4429 | -0.4571 |
| `kubernetes_level_5_pass_rate` | 0.1429 | 0.1286 | -0.0143 |

La comparacion KDV muestra dos lecturas distintas:

- hasta nivel 4, `serialized_sft` supera ampliamente a `two_head_sft`;
- en el gate completo de nivel 5, ambos valores son bajos y cercanos:
  `0.1429` frente a `0.1286`.

Esto se debe registrar como una propiedad de la metrica: el nivel 5 incluye
checks estaticos de calidad y seguridad muy exigentes, por lo que no mide solo
parseabilidad YAML ni identidad Kubernetes minima.

## Conteos de errores KDV

### serialized_sft

```json
{
  "missing_resource_requirement": 261,
  "missing_run_as_non_root": 71,
  "missing_read_only_root_filesystem": 71,
  "latest_image_tag": 67,
  "volume_mount_without_volume": 5,
  "kubernetes_identity": 1,
  "yaml_parse": 1,
  "service_selector_without_workload": 1
}
```

### two_head_sft

```json
{
  "missing_resource_requirement": 112,
  "yaml_parse": 36,
  "missing_run_as_non_root": 28,
  "missing_read_only_root_filesystem": 28,
  "latest_image_tag": 26,
  "required_field": 2,
  "container_missing_image": 1
}
```

Estos conteos se incluyen solo como registro cuantitativo. Su interpretacion
cualitativa queda pendiente para una auditoria posterior de predicciones.

## Vista condicionada a YAML parseable

La tabla principal anterior mezcla dos efectos:

- calidad de las predicciones una vez reconstruidas;
- proporcion de predicciones que llegan a YAML parseable.

Para separar parcialmente ambos efectos, se calcula tambien una vista
condicionada a `yaml_parse_ok = true`. Esta vista no sustituye a la metrica
global, porque una salida no parseable sigue siendo un fallo real del sistema
completo. Pero ayuda a distinguir si `two_head_sft` falla solo por parseabilidad
o si su calidad tambien se degrada de forma general cuando el YAML reconstruido
si puede parsearse.

### Cada modelo sobre sus propias salidas parseables

Este primer corte usa todos los ejemplos parseables de cada modelo:

- `serialized_sft`: `69/70`;
- `two_head_sft`: `34/70`.

La comparacion no es perfectamente emparejada, porque cada modelo aporta un
subconjunto distinto de ejemplos parseables. Aun asi, muestra que la brecha se
reduce mucho cuando se excluyen las salidas no parseables.

| Metrica, solo YAML parseable | serialized_sft | two_head_sft | Delta |
| --- | ---: | ---: | ---: |
| `row_count` | 69 | 34 | -35 |
| `parsed_equal_to_reference` | 0.1159 | 0.0294 | -0.0865 |
| `line_text_f1` | 0.8325 | 0.7707 | -0.0617 |
| `content_exact_match_rate` | 0.6213 | 0.5761 | -0.0452 |
| `level_exact_match_rate` | 0.7688 | 0.7215 | -0.0473 |
| `level_mae` | 0.2723 | 0.2392 | -0.0331 |
| `kind_sequence_match_rate` | 0.9275 | 0.8971 | -0.0305 |
| `semantic_key_f1` | 0.9691 | 0.9455 | -0.0236 |
| `prompt_requirement_f1` | 0.8655 | 0.8659 | +0.0005 |
| `prompt_requirement_exact_match` | 0.5942 | 0.5588 | -0.0354 |
| `required_field_complete_resource_rate` | 1.0000 | 0.9412 | -0.0588 |
| `kubernetes_domain_validity_score` | 0.8430 | 0.8725 | +0.0296 |
| `kubernetes_domain_validity_level` | 3.9710 | 4.0294 | +0.0584 |
| `kubernetes_domain_gate_pass` | 0.1449 | 0.2647 | +0.1198 |
| `kubernetes_level_2_pass_rate` | 0.9855 | 0.9412 | -0.0443 |
| `kubernetes_level_3_pass_rate` | 0.9275 | 0.9118 | -0.0158 |
| `kubernetes_level_4_pass_rate` | 0.9130 | 0.9118 | -0.0013 |
| `kubernetes_level_5_pass_rate` | 0.1449 | 0.2647 | +0.1198 |

En esta vista, `two_head_sft` sigue por debajo en F1 de texto, exactitud de
`level` y algunas metricas de completitud. Pero la distancia es mucho menor que
en la tabla global. Ademas, `two_head_sft` queda ligeramente mejor en
`level_mae` y en las metricas KDV agregadas sobre su subconjunto parseable.

### Mismos ejemplos donde two_head_sft es parseable

Para reducir el sesgo de comparar subconjuntos distintos, se toma despues el
subconjunto de `34` `unit_id` donde `two_head_sft` produjo YAML parseable y se
evalua tambien `serialized_sft` solo en esos mismos ejemplos.

| Metrica, mismos 34 ejemplos | serialized_sft | two_head_sft | Delta |
| --- | ---: | ---: | ---: |
| `row_count` | 34 | 34 | +0 |
| `parsed_equal_to_reference` | 0.0882 | 0.0294 | -0.0588 |
| `line_text_f1` | 0.7893 | 0.7707 | -0.0186 |
| `content_exact_match_rate` | 0.5376 | 0.5761 | +0.0385 |
| `level_exact_match_rate` | 0.7341 | 0.7215 | -0.0127 |
| `level_mae` | 0.2903 | 0.2392 | -0.0511 |
| `kind_sequence_match_rate` | 0.8824 | 0.8971 | +0.0147 |
| `semantic_key_f1` | 0.9454 | 0.9455 | +0.0001 |
| `prompt_requirement_f1` | 0.8883 | 0.8659 | -0.0224 |
| `prompt_requirement_exact_match` | 0.6176 | 0.5588 | -0.0588 |
| `required_field_complete_resource_rate` | 1.0000 | 0.9412 | -0.0588 |
| `kubernetes_domain_validity_score` | 0.8676 | 0.8725 | +0.0049 |
| `kubernetes_domain_validity_level` | 4.1471 | 4.0294 | -0.1176 |
| `kubernetes_domain_gate_pass` | 0.2647 | 0.2647 | +0.0000 |
| `kubernetes_level_2_pass_rate` | 0.9706 | 0.9412 | -0.0294 |
| `kubernetes_level_3_pass_rate` | 0.9706 | 0.9118 | -0.0588 |
| `kubernetes_level_4_pass_rate` | 0.9412 | 0.9118 | -0.0294 |
| `kubernetes_level_5_pass_rate` | 0.2647 | 0.2647 | +0.0000 |

Este corte cambia de forma importante la interpretacion del resultado. En los
ejemplos donde `two_head_sft` consigue producir YAML parseable, la calidad ya no
parece degradarse de forma general. Algunas metricas siguen favoreciendo a
`serialized_sft`, pero las diferencias son pequenas en comparacion con la tabla
global:

- `line_text_f1` queda cerca: `0.7893` frente a `0.7707`;
- `semantic_key_f1` queda practicamente igual;
- `kubernetes_domain_gate_pass` queda igual en los mismos ejemplos;
- `content_exact_match_rate`, `kind_sequence_match_rate`, `level_mae` y
  `kubernetes_domain_validity_score` favorecen ligeramente a `two_head_sft`.

La lectura preliminar de esta vista es que gran parte de la caida global de
`two_head_sft` procede de las salidas no parseables. Cuando se condiciona a que
el YAML reconstruido sea parseable, el modelo no parece uniformemente peor que
`serialized_sft`; queda competitivo en varias metricas, aunque sobre un
subconjunto mucho mas pequeno.

## Lectura comparativa preliminar

En esta validacion, `serialized_sft` es claramente superior a `two_head_sft` en
la mayoria de metricas agregadas:

- reconstruccion YAML parser-facing;
- F1 de texto de linea;
- exactitud discreta de `level`;
- adecuacion al prompt;
- claves semanticas aproximadas;
- niveles KDV hasta nivel 4.

La unica excepcion destacable entre las metricas principales es
`average_level_mae`, donde `two_head_sft` queda ligeramente por debajo, que es
mejor para una metrica de error:

```text
serialized_sft: 0.2723
two_head_sft:   0.2392
```

Esta excepcion no cambia la lectura general del resultado de validacion, pero
debe conservarse porque es relevante para la pregunta especifica de si una
cabeza estructural puede aprender una senal de nivel con menor distancia media.

La vista condicionada a parseabilidad matiza esta conclusion: la brecha global
esta muy influida por el hecho de que `two_head_sft` solo produce YAML parseable
en `34/70` ejemplos. En esos `34` ejemplos, las metricas de contenido,
semantica aproximada, requisitos del prompt y KDV quedan mucho mas cerca de
`serialized_sft`.

## Implicacion para la comparacion principal

Con estos resultados, la comparacion supervisada queda provisionalmente a favor
de `serialized_sft` como sistema generativo completo en validacion. La rama
`two_head_sft` no supera al SFT normal en el objetivo parser-facing final,
principalmente porque falla muchas mas veces antes de llegar a YAML parseable.
Sin embargo, la vista sobre salidas parseables sugiere que la calidad de las
salidas validas no esta necesariamente degradada de forma general.

La conclusion prudente es:

- `serialized_sft` sigue siendo el control supervisado fuerte para Kubernetes
  v1;
- `two_head_sft` no mejora la salida final en esta primera configuracion;
- el principal cuello de botella observable en las metricas agregadas es la
  parseabilidad;
- la cabeza explicita de nivel debe analizarse despues con una auditoria de
  errores antes de decidir si el problema esta en la superficie de contenido, en
  la alineacion de `level`, en la inferencia de validacion o en la propia
  formulacion del primer experimento.

Esa auditoria no forma parte de este documento.

## Registro operacional minimo

`two_head_sft` termino correctamente:

- estado final: `completed`;
- `global_step`: `159`;
- `epoch`: `3`;
- checkpoint final: `checkpoint-step-159`;
- `metrics.json` escrito correctamente;
- artefactos sincronizados con W&B;
- microbatches omitidas por OOM durante entrenamiento: `2`.

El resultado queda listo para ser usado como primer punto de comparacion
experimental frente a `serialized_sft`.
