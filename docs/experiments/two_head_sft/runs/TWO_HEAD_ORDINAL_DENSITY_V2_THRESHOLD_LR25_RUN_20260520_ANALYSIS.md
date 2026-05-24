# Analisis del run two_head_ordinal_density_v2 threshold LR25 20260520

Document type: run result

## Resumen

El run `two-head-ordinal-density-v2-threshold-lr25-20260520` repite la
configuracion del run ordinal inicial, cambiando solo el learning rate de los
umbrales aprendidos:

```text
threshold_learning_rate_multiplier = 25
learning_rate = 2e-4
threshold_learning_rate = 5e-3
```

El entrenamiento completo 3 epocas sobre `kubernetes_v1`, con
`batch_size=1`, `gradient_accumulation_steps=8`, checkpoints cada 8 optimizer
steps y validacion intermedia cada 32 steps. El checkpoint final fue
`checkpoint-step-159`.

El run termino correctamente:

```text
status = completed
completed_at = 2026-05-21T15:58:57.139080Z
global_step = 159
oom_skipped_batches = 2
```

Los dos OOM correspondieron al mismo ejemplo largo ya observado en el run
anterior (`q229::question_simplified`) y se gestionaron con
`oom_recovery=skip_batch`. No se borraron checkpoints.

## Resultado corto

El multiplicador `25x` si consiguio mover los thresholds de forma apreciable:
`tau_5`, por ejemplo, paso de moverse `-0.0063` en el run inicial a moverse
`-0.1573`.

Sin embargo, no resolvio el fallo estructural principal. La validacion final
sigue sin emitir niveles profundos `5..8`:

```text
predicted_level_count_5 = 0
predicted_level_count_6 = 0
predicted_level_count_7 = 0
predicted_level_count_8 = 0
deep_level_exact_recall_5_8 = 0
compressed_deep_to_0_4_rate = 1.0
```

La conclusion es que `threshold_learning_rate_multiplier=25` mejora el
movimiento de los cortes, pero no basta para que la escala latente `z` alcance
las regiones profundas. El experimento refuerza la necesidad de separar tres
grupos de learning rate: LoRA/base, MLP ordinal que produce `z`, y thresholds
`tau`.

## Configuracion

| Campo | Valor |
| --- | ---: |
| Run ID | `two-head-ordinal-density-v2-threshold-lr25-20260520` |
| Modelo | `model/qwen2.5-7b-instruct-4bit` |
| Dataset | `data/processed/kubernetes_v1` |
| Epocas | 3 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Optimizer steps finales | 159 |
| Checkpoint steps | 8 |
| Eval checkpoint steps | 32 |
| Base learning rate | 0.0002 |
| Threshold LR multiplier | 25 |
| Threshold LR inicial | 0.005 |
| W&B | online, artifacts enabled |

## Comparacion con el run inicial

Run inicial:
`two-head-ordinal-density-v2-20260519`

Run LR25:
`two-head-ordinal-density-v2-threshold-lr25-20260520`

| Metrica final | Inicial | LR25 | Delta |
| --- | ---: | ---: | ---: |
| `structured_output_parse_success_rate` | 1.0000 | 1.0000 | +0.0000 |
| `yaml_parse_success_rate` | 0.2571 | 0.2857 | +0.0286 |
| `block_parse_success_rate` | 0.2571 | 0.2857 | +0.0286 |
| `average_level_exact_match_rate` | 0.0554 | 0.0693 | +0.0139 |
| `average_level_mae` | 0.7695 | 0.8062 | +0.0367 |
| `deep_level_exact_recall_5_8` | 0.0000 | 0.0000 | +0.0000 |
| `deep_level_off_by_one_recall_5_8` | 0.2190 | 0.2536 | +0.0346 |
| `compressed_deep_to_0_4_rate` | 0.9927 | 1.0000 | +0.0073 |
| `predicted_max_level_mean` | 3.6857 | 3.6857 | +0.0000 |
| `target_max_level_ge_5_yaml_parse_success_rate` | 0.2609 | 0.2391 | -0.0217 |
| `primary_kind_match_rate` | 0.8333 | 0.8500 | +0.0167 |
| `primary_api_version_match_rate` | 0.8333 | 0.8500 | +0.0167 |
| `average_kubernetes_domain_validity_score` | 0.2167 | 0.2381 | +0.0214 |
| `kubernetes_domain_gate_pass_rate` | 0.0143 | 0.0000 | -0.0143 |
| `average_bleu_score` | 0.6023 | 0.6035 | +0.0012 |
| `average_rougeL_f1` | 0.7193 | 0.7380 | +0.0187 |

La lectura es mixta. LR25 sube ligeramente la parseabilidad YAML y algunas
metricas de identidad Kubernetes, pero no mejora la variable que motivaba el
experimento: la recuperacion de niveles profundos. De hecho,
`compressed_deep_to_0_4_rate` queda en `1.0`, lo que significa que todos los
niveles gold `5..8` acabaron predichos como `0..4`.

## Distribucion final de niveles

| Nivel | Gold inicial | Pred inicial | Gold LR25 | Pred LR25 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 312 | 503 | 316 | 502 |
| 1 | 262 | 245 | 266 | 262 |
| 2 | 208 | 297 | 215 | 291 |
| 3 | 291 | 337 | 293 | 340 |
| 4 | 372 | 199 | 381 | 214 |
| 5 | 87 | 1 | 89 | 0 |
| 6 | 38 | 0 | 37 | 0 |
| 7 | 3 | 0 | 3 | 0 |
| 8 | 9 | 0 | 9 | 0 |

Aunque el run LR25 produce algo mas de nivel `4`, no cruza a `5..8` en la
validacion final. Esto confirma que el problema no era solo que los thresholds
estuvieran casi congelados.

## Fallos de parsing YAML

Los errores de parseo son casi los mismos en naturaleza, aunque no exactamente
los mismos ejemplos.

| Comparacion | Conteo |
| --- | ---: |
| Fallos YAML en run inicial | 52 / 70 |
| Fallos YAML en run LR25 | 50 / 70 |
| Fallan en ambos runs | 44 |
| Fallaban antes y LR25 arregla | 8 |
| Parseaban antes y LR25 rompe | 6 |
| Parsean en ambos runs | 12 |

Por tipo de excepcion PyYAML:

| Tipo | Inicial | LR25 |
| --- | ---: | ---: |
| `ParserError` | 37 | 34 |
| `ScannerError` | 15 | 16 |

Clasificando los mensajes:

| Patron de error | Inicial | LR25 |
| --- | ---: | ---: |
| `expected <block end>` | 37 | 34 |
| `mapping values are not allowed here` | 15 | 16 |

En los 44 ejemplos que fallan en ambos runs, la transicion de patrones fue:

| Transicion | Conteo |
| --- | ---: |
| `expected block end -> expected block end` | 26 |
| `mapping values not allowed -> mapping values not allowed` | 10 |
| `expected block end -> mapping values not allowed` | 5 |
| `mapping values not allowed -> expected block end` | 3 |

Esto indica que LR25 no cambia de forma sustancial la familia de errores. La
mayoria siguen siendo fallos de estructura YAML inducidos por niveles
insuficientes: claves que deberian estar anidadas aparecen demasiado arriba, o
elementos de lista y mappings quedan al mismo nivel cuando YAML espera cerrar
un bloque.

Ejemplos que fallan en ambos runs incluyen:

```text
q13::question
q13::question_simplified
q21::question
q126::question
q139::question
q139::question_simplified
q140::question
q140::question_simplified
```

Ejemplos que LR25 arreglo respecto al run inicial:

```text
q19::question_simplified
q22::question
q27::question
q28::question_simplified
q29::question
q33::question_simplified
q34::question
q65::question_simplified
```

Ejemplos que parseaban antes y fallan en LR25:

```text
q126::question_simplified
q24::question
q32::question
q43::question_simplified
q49::question
q49::question_simplified
```

La conclusion es que la ligera mejora de `yaml_parse_success_rate` no viene de
una correccion sistematica de los errores de jerarquia, sino de pequenos cambios
en casos concretos. La distribucion de errores sigue apuntando al mismo problema
central: el modelo no coloca niveles suficientemente profundos para sostener
listas y mappings anidados.

## Movimiento de thresholds

| Threshold | Inicio | Final inicial | Drift inicial | Final LR25 | Drift LR25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tau_0` | 0.5000 | 0.4981 | -0.0019 | 0.4512 | -0.0488 |
| `tau_1` | 1.5000 | 1.4970 | -0.0030 | 1.4235 | -0.0765 |
| `tau_2` | 2.5000 | 2.4959 | -0.0041 | 2.3956 | -0.1044 |
| `tau_3` | 3.5000 | 3.4949 | -0.0051 | 3.3695 | -0.1305 |
| `tau_4` | 4.5000 | 4.4941 | -0.0059 | 4.3512 | -0.1488 |
| `tau_5` | 5.5000 | 5.4937 | -0.0063 | 5.3427 | -0.1573 |
| `tau_6` | 6.5000 | 6.4957 | -0.0043 | 6.3969 | -0.1031 |
| `tau_7` | 7.5000 | 7.4988 | -0.0012 | 7.4776 | -0.0224 |

El cambio es real: el multiplicador de LR consiguio desplazar los cortes entre
unas 12 y 40 veces mas que el run inicial, dependiendo del umbral. Aun asi, el
umbral `tau_5` termina en `5.3427`, muy lejos del rango final observado de `z`.

## Evolucion de gaps

| Gap | Inicio | Final inicial | Final LR25 |
| --- | ---: | ---: | ---: |
| `gap_0` | 1.0000 | 0.9989 | 0.9722 |
| `gap_1` | 1.0000 | 0.9989 | 0.9721 |
| `gap_2` | 1.0000 | 0.9989 | 0.9739 |
| `gap_3` | 1.0000 | 0.9992 | 0.9818 |
| `gap_4` | 1.0000 | 0.9996 | 0.9915 |
| `gap_5` | 1.0000 | 1.0020 | 1.0542 |
| `gap_6` | 1.0000 | 1.0031 | 1.0807 |

LR25 no solo baja los cortes, tambien abre los gaps profundos (`gap_5`,
`gap_6`). Esto no ayuda al objetivo de predecir niveles altos, porque deja
`tau_6` y `tau_7` todavia mas separados de la masa de `z`.

## Evolucion de `z`

| Estadistico | Inicio comun | Final inicial | Final LR25 |
| --- | ---: | ---: | ---: |
| `z_mean` | 0.0079 | -0.1538 | -0.2160 |
| `z_min` | -0.0596 | -2.9692 | -2.6479 |
| `z_max` | 0.0454 | 1.7186 | 1.4625 |

La observacion clave es que, al final, `z_max` queda en `1.4625`, mientras
`tau_5` queda en `5.3427`. Con esa geometria, el modelo no puede emitir niveles
`5..8` salvo que algun ejemplo produzca un `z` mucho mayor en validacion. La
validacion final confirma que eso no ocurre.

## Diagnostico de gradientes

El run LR25 contiene `gradient_diagnostics.jsonl`, pero solo desde la parte
reanudadada posterior a la instrumentacion. Por tanto, debe leerse como una
senal parcial, no como una traza completa del entrenamiento.

Ultimo diagnostico registrado:

| Senal | Valor |
| --- | ---: |
| `grad_rms_ordinal_mlp` | 0.0001709 |
| `grad_rms_ordinal_threshold_raw` | 0.0078974 |
| `update_rms_ordinal_mlp` | 0.0000005121 |
| `update_rms_ordinal_threshold_raw` | 0.0000073873 |
| `effective_z_mean_abs_shift` | 0.0010048 |
| `effective_tau_mean_abs_shift` | 0.0000150 |
| `effective_z_to_tau_mean_shift_ratio` | 67.0594 |

Esto explica por que no basta mirar el gradiente bruto. El gradiente RMS de los
thresholds raw es mayor que el del MLP, y su update por parametro tambien es
mayor. Pero la decision ordinal ocurre en la escala `z - tau`. En esa escala,
el cambio funcional de `z` sobre el mismo hidden batch fue unas 67 veces mayor
que el cambio medio de `tau`.

La interpretacion es que el MLP, aunque se mueva menos por parametro, sigue
teniendo muchas mas palancas para mover la salida `z`. Los thresholds son pocos
parametros globales y ademas pasan por la parametrizacion
`tau_0 + cumsum(softplus(raw_deltas))`.

## Validaciones intermedias

El run LR25 tuvo una senal interesante en validacion intermedia:

| Checkpoint | YAML parse | Deep exact | Deep off-by-one | Compressed deep | Pred max mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-step-32` | 0.0000 | 0.0000 | 0.6667 | 0.8889 | 4.3000 |
| `checkpoint-step-64` | 0.0000 | 0.3684 | 0.8421 | 0.2632 | 5.3000 |
| `checkpoint-step-96` | 0.2000 | 0.1667 | 0.6667 | 0.7500 | 4.5000 |
| `checkpoint-step-128` | 0.2000 | 0.0000 | 0.5882 | 1.0000 | 4.3000 |

El checkpoint `64` si llego a emitir niveles profundos en la muestra
intermedia, pero sin parseabilidad YAML. La senal desaparecio despues. Esto
sugiere que el entrenamiento puede atravesar una region donde la cabeza ordinal
usa niveles profundos, pero el objetivo conjunto y la continuacion del training
la empujan de vuelta hacia una solucion comprimida.

## Interpretacion

El run responde de forma bastante clara a la pregunta experimental:

```text
Subir solo el learning rate de thresholds a 25x no basta.
```

El resultado no invalida la cabeza ordinal, pero si descarta que el problema
fuera solamente una congelacion trivial de los cortes. Hay tres hechos juntos:

- los thresholds ahora si se mueven;
- la validacion final no predice ningun nivel `5..8`;
- el rango final de `z` queda demasiado bajo respecto a `tau_5..tau_7`.

La hipotesis mas razonable pasa a ser que la escala latente `z` y los cortes
`tau` deben estudiarse como un sistema acoplado. Si solo se bajan los cortes,
pero `z` no aprende a ocupar una escala compatible con niveles profundos, el
modelo sigue colapsando. Si solo se sube el MLP, puede ocurrir lo contrario:
`z` absorbe todo el aprendizaje y los thresholds quedan como cortes casi
decorativos.

## Siguiente experimento recomendado

El siguiente cambio debe ser controlado con tres grupos de learning rate:

```text
base_lora:          learning_rate
ordinal_mlp:        learning_rate * ordinal_mlp_learning_rate_multiplier
ordinal_thresholds: learning_rate * threshold_learning_rate_multiplier
```

Una primera matriz pequena y trazable seria:

| Run | `ordinal_mlp_lr_multiplier` | `threshold_lr_multiplier` | Objetivo |
| --- | ---: | ---: | --- |
| A | 3 | 25 | comprobar si basta mover mas `z` manteniendo LR25 |
| B | 1 | 75 | aislar thresholds mas agresivos |
| C | 3 | 75 | probar ajuste conjunto de escala `z` y cortes |

Los criterios de decision no deben ser solo la loss. Deben priorizar:

- `predicted_level_count_5..8 > 0`;
- `deep_level_exact_recall_5_8 > 0`;
- bajada de `compressed_deep_to_0_4_rate`;
- aumento de `predicted_max_level_mean`;
- mantenimiento o mejora de `yaml_parse_success_rate`;
- ratio `effective_z_to_tau_mean_shift_ratio` menos extremo.

## Artefactos

Run LR25:

```text
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-threshold-lr25-20260520/
```

Artefactos principales:

- `config.json`
- `state.json`
- `metrics.json`
- `train_log.jsonl`
- `threshold_history.jsonl`
- `gradient_diagnostics.jsonl`
- `validation_predictions.jsonl`
- `validation_metrics_progress.jsonl`
- `validation_example_metrics.jsonl`
- `intermediate_validation_metrics.jsonl`
- `intermediate_validation_predictions.jsonl`
- `checkpoints/checkpoint-step-159/`

Resumen comparativo derivado:

```text
results/two_head_ordinal_sft_kubernetes_v1/lr25_comparison_summary.json
```
