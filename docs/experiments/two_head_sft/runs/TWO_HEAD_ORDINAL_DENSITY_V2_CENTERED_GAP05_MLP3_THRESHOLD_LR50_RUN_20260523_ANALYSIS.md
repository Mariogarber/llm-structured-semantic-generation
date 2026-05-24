# Analisis del run two_head_ordinal_density_v2 centered gap05 MLP3 threshold LR50 20260523

Document type: run result

## Resumen

El run `two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523`
prueba la misma separacion de learning rates del experimento anterior, pero
cambia la inicializacion de thresholds. En lugar de empezar en la escala
positiva original:

```text
[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
```

se inicializan centrados en cero y con gap `0.5`:

```text
[-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
```

La configuracion mantiene el contrato experimental de las runs anteriores:

```text
dataset = data/processed/kubernetes_v1
model = model/qwen2.5-7b-instruct-4bit
epochs = 3
batch_size = 1
gradient_accumulation_steps = 8
checkpoint_steps = 8
eval_checkpoint_steps = 32
wandb_mode = online
wandb_log_artifacts = true
base_learning_rate = 2e-4
ordinal_mlp_learning_rate_multiplier = 3
threshold_learning_rate_multiplier = 50
initial_threshold_center = 0
initial_threshold_gap = 0.5
```

El run termino correctamente:

```text
status = completed
completed_at = 2026-05-23T21:56:45.993897Z
global_step = 159
best_checkpoint = checkpoint-step-159
oom_skipped_batches = 2
```

Los dos OOM son el mismo patron ya observado y se gestionaron con
`oom_recovery=skip_batch`. No se borraron checkpoints. El run quedo
sincronizado en W&B y subio artefactos, aunque el log contiene una advertencia
de W&B sobre un intento de loguear algunos escalares finales en un step menor
que el step interno actual. Por eso, para el analisis se toma `metrics.json`
local como fuente canonica.

## Resultado corto

El experimento confirma que la inicializacion centrada si ataca una parte real
del problema: por primera vez la validacion final emite masa apreciable en
levels profundos `5` y `6`.

Sin embargo, el resultado global es negativo para la tarea final:

- `yaml_parse_success_rate` cae a `0.0286`, es decir, solo `2 / 70` ejemplos
  parsean como YAML.
- `deep_level_exact_recall_5_8` sube a `0.2174`.
- `deep_level_off_by_one_recall_5_8` sube a `0.5652`.
- `compressed_deep_to_0_4_rate` baja a `0.6014`, lejos del `1.0` de las runs
  LR25 y MLP3/THR50 no centradas.
- `predicted_level_count_5 = 121` y `predicted_level_count_6 = 12`, pero
  `predicted_level_count_7 = 0` y `predicted_level_count_8 = 0`.
- `average_line_text_f1` cae a `0.0145`, y las metricas de identidad Kubernetes
  tambien bajan con fuerza.

La lectura principal es que centrar y compactar thresholds fue util como
diagnostico de escala, pero no basta como solucion. El cabezal ordinal deja de
comprimir completamente los niveles profundos, pero introduce una geometria de
indentacion muy inestable y empeora la generacion parser-facing completa.

## Comparacion con runs anteriores

| Metrica final | Inicial | LR25 | MLP3/THR50 | Centered gap05 MLP3/THR50 |
| --- | ---: | ---: | ---: | ---: |
| `yaml_parse_success_rate` | 0.2571 | 0.2857 | 0.1429 | 0.0286 |
| `block_parse_success_rate` | 0.2571 | 0.2857 | 0.1429 | 0.0286 |
| `average_level_exact_match_rate` | 0.0554 | 0.0693 | 0.0409 | 0.0107 |
| `average_level_mae` | 0.7695 | 0.8062 | 0.9182 | 0.4559 |
| `deep_level_exact_recall_5_8` | 0.0000 | 0.0000 | 0.0000 | 0.2174 |
| `deep_level_off_by_one_recall_5_8` | 0.2190 | 0.2536 | 0.0000 | 0.5652 |
| `compressed_deep_to_0_4_rate` | 0.9927 | 1.0000 | 1.0000 | 0.6014 |
| `predicted_max_level_mean` | 3.6857 | 3.6857 | 2.9143 | 4.4857 |
| `target_max_level_ge_5_yaml_parse_success_rate` | 0.2609 | 0.2391 | 0.0435 | 0.0435 |
| `average_line_text_f1` | 0.1899 | 0.2269 | 0.1153 | 0.0145 |
| `primary_kind_match_rate` | 0.8333 | 0.8500 | 0.7000 | 0.5000 |
| `primary_api_version_match_rate` | 0.8333 | 0.8500 | 0.7000 | 0.5000 |
| `average_kubernetes_domain_validity_score` | 0.2167 | 0.2381 | 0.1190 | 0.0238 |

La mejora en `average_level_mae` y en recall profundo no se traduce en YAML
valido. Esto es importante: la run demuestra que el problema no era solo que el
modelo no pudiera cruzar cortes profundos. Puede cruzarlos, pero hacerlo con el
loss ordinal actual no preserva necesariamente la secuencia de bloques ni una
indentacion coherente.

## Distribucion final de levels

| Level | Gold centered | Pred centered |
| --- | ---: | ---: |
| 0 | 316 | 598 |
| 1 | 275 | 148 |
| 2 | 220 | 306 |
| 3 | 312 | 248 |
| 4 | 422 | 250 |
| 5 | 93 | 121 |
| 6 | 37 | 12 |
| 7 | 2 | 0 |
| 8 | 6 | 0 |

Frente a MLP3/THR50 sin centrar, donde `predicted_level_count_4..8` era cero,
esta run recupera masa en `level=4`, `level=5` y `level=6`. Ese era el efecto
esperado de usar thresholds centrados: la parte negativa de `z` deja de quedar
reservada casi por completo para `level=0`, y los cortes profundos ya no estan
en una escala tan alta.

Pero la distribucion sigue mal calibrada. El modelo sobregenera `level=0`,
subgenera `level=1`, y no llega a `level=7..8`. La mejora de niveles profundos
es real, pero todavia no es una estructura de arbol fiable.

## Evolucion intermedia

| Checkpoint | YAML parse | Deep exact 5-8 | Deep off-by-one 5-8 | Comprimido 5-8 a 0-4 | Pred max level mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-step-32` | 0.0000 | 0.0000 | 0.7273 | 1.0000 | 3.9000 |
| `checkpoint-step-64` | 0.0000 | 0.2414 | 0.5517 | 0.1034 | 6.3333 |
| `checkpoint-step-96` | 0.1000 | 0.0000 | 0.2941 | 0.8235 | 4.5000 |
| `checkpoint-step-128` | 0.1000 | 0.1333 | 0.4000 | 0.8000 | 3.8000 |
| final `checkpoint-step-159` | 0.0286 | 0.2174 | 0.5652 | 0.6014 | 4.4857 |

El mejor momento para recuperar niveles profundos fue `checkpoint-step-64`,
pero no produjo YAML parseable. La run no sigue una trayectoria monotona: la
capacidad de emitir niveles profundos aparece temprano, cae parcialmente, y
termina en una posicion intermedia. Esto apunta a una interaccion inestable
entre el objetivo ordinal, la generacion de contenido y el desbalance de levels.

## Movimiento de thresholds y escala `z`

| Threshold | Inicio centered | Final centered | Drift |
| --- | ---: | ---: | ---: |
| `tau_0` | -1.7500 | -1.7958 | -0.0458 |
| `tau_1` | -1.2500 | -1.2642 | -0.0142 |
| `tau_2` | -0.7500 | -0.6972 | +0.0528 |
| `tau_3` | -0.2500 | -0.0998 | +0.1502 |
| `tau_4` | 0.2500 | 0.5347 | +0.2847 |
| `tau_5` | 0.7500 | 1.2151 | +0.4651 |
| `tau_6` | 1.2500 | 1.9066 | +0.6566 |
| `tau_7` | 1.7500 | 2.5755 | +0.8255 |

Los thresholds altos se mueven mucho mas que en las runs no centradas. Los gaps
finales quedan entre `0.5316` y `0.6916`, por lo que el modelo aprende a
ensanchar la escala inicial:

```text
final gaps = [0.5316, 0.5669, 0.5974, 0.6345, 0.6804, 0.6916, 0.6688]
```

La ultima muestra de entrenamiento registrada tiene:

```text
ordinal_z_mean = -3.6356
ordinal_z_min = -8.8744
ordinal_z_max = -0.9327
```

Ese ultimo batch no representa toda la validacion. En el historial completo de
entrenamiento se observo:

```text
z_mean_min = -8.4161
z_mean_max = 0.7276
z_min_min = -11.1643
z_max_max = 2.8954
```

La escala centrada permite que algunos ejemplos entren en `level=5..6`, pero
los thresholds altos terminan desplazandose hacia arriba. Para llegar a
`level=7`, `z` debe superar `tau_6 = 1.9066`; para llegar a `level=8`, debe
superar `tau_7 = 2.5755`. La validacion final no produce ningun `level=7..8`.

Las diagnosticas de gradiente finales fueron:

| Magnitud | Valor |
| --- | ---: |
| `effective_z_mean_abs_shift` | 0.0090990 |
| `effective_tau_mean_abs_shift` | 0.0001194 |
| `effective_z_to_tau_mean_shift_ratio` | 76.1947 |
| `grad_rms_ordinal_mlp` | 0.0002248 |
| `grad_rms_ordinal_threshold_raw` | 0.0098860 |
| `update_rms_ordinal_mlp` | 0.0000016 |
| `update_rms_ordinal_threshold_raw` | 0.0000684 |

Aunque el update directo de thresholds es mayor que el del MLP, el efecto local
sobre `z` sigue siendo mucho mayor que el desplazamiento efectivo de los
thresholds transformados. Esto mantiene la interpretacion de las runs
anteriores: los thresholds se mueven, pero la calibracion global depende mucho
de como el MLP de `z` deforma el espacio muestra a muestra.

## Por que el YAML parse cae tanto

La metrica final es:

```text
yaml_parse_success_rate = 0.02857142857142857
kubernetes_domain_error_counts.yaml_parse = 68
```

Por tipo de excepcion PyYAML:

| Tipo | Conteo |
| --- | ---: |
| `ParserError` | 58 |
| `ScannerError` | 10 |

Comparado con runs anteriores, no aparece una familia nueva de error; aparecen
los mismos errores de YAML, pero en mas ejemplos:

| Run | ParserError | ScannerError | Fallos YAML |
| --- | ---: | ---: | ---: |
| Inicial | 37 | 15 | 52 / 70 |
| LR25 | 34 | 16 | 50 / 70 |
| MLP3/THR50 | 55 | 3 | 60 / 70 |
| Centered gap05 MLP3/THR50 | 58 | 10 | 68 / 70 |

El contrato de bloques sigue parseando internamente:

```text
structured_output_parse_success_rate = 1.0
average_valid_block_ratio = 1.0
document_index_monotonic_ok_rate = 1.0
line_index_sequence_ok_rate = 1.0
```

Por tanto, el fallo no es que el modelo deje de emitir `content_blocks_v1`.
El fallo aparece al reconstruir YAML desde bloques que tienen niveles
incorrectos o una secuencia de lineas ya poco compatible con el gold.

Se repitio el diagnostico oracle de indentacion sobre los artefactos finales.
En los 24 ejemplos donde el numero de lineas generadas coincide con el numero
de levels gold, se reconstruyo el mismo `line_text` generado de dos maneras:

1. con los levels predichos por el modelo;
2. con los levels gold, como oracle de indentacion.

| Reconstruccion en ejemplos con igual numero de lineas | Parse OK |
| --- | ---: |
| `line_text` generado + levels predichos | 0 / 24 |
| `line_text` generado + levels gold | 22 / 24 |

Ademas, 22 ejemplos que fallaban con los levels predichos pasaban a parsear al
sustituir solo los levels por los gold. Esto indica que, en los casos donde la
longitud permite comparar limpiamente, la causa dominante del parse bajo sigue
siendo la indentacion. Los levels profundos aparecen, pero no aparecen en las
lineas correctas ni con transiciones estructurales coherentes.

Ejemplos que el oracle de levels arregla:

```text
q17::question
q17::question_simplified
q19::question
q19::question_simplified
q29::question_simplified
q30::question
q30::question_simplified
q31::question
q31::question_simplified
q32::question
q32::question_simplified
q34::question
```

Tambien hay 46 / 70 ejemplos donde no se puede hacer esta prueba porque el
numero de bloques no coincide con el gold. Esa parte ya no se arregla solo con
mejores thresholds: implica omisiones, inserciones, duplicaciones o una
secuencia de lineas que se aleja del objetivo parser-facing.

## Comparacion de ejemplos parseables

La run centrada solo parsea dos ejemplos finales:

```text
q22::question
q126::question_simplified
```

Frente a LR25:

| Comparacion con LR25 | Conteo |
| --- | ---: |
| Fallan en ambas | 49 |
| Centered arregla fallo de LR25 | 1 |
| Centered rompe ejemplo que LR25 parseaba | 19 |
| Parsean en ambas | 1 |

Frente a MLP3/THR50:

| Comparacion con MLP3/THR50 | Conteo |
| --- | ---: |
| Fallan en ambas | 58 |
| Centered arregla fallo de MLP3/THR50 | 2 |
| Centered rompe ejemplo que MLP3/THR50 parseaba | 10 |
| Parsean en ambas | 0 |

Esto confirma que la caida de parse no es un simple cambio de ejemplos
equivalentes. La run centrada rompe muchos casos que antes parseaban.

## Interpretacion

El experimento responde bien a la pregunta concreta que lo motivaba: si los
thresholds empiezan centrados y mas compactos, el modelo ya no esta obligado a
llevar `z` hasta valores cercanos a `7` para expresar niveles profundos. Como
resultado, aparecen predicciones reales en `level=5` y `level=6`.

Pero esa mejora no produce una jerarquia util. El modelo parece aprender a usar
la escala profunda de forma parcial y desordenada. En otras palabras: centrar
thresholds corrige una mala condicion inicial, pero no resuelve el problema
estadistico del cabezal. Los niveles altos siguen siendo escasos, las
transiciones entre niveles importan mas que la etiqueta aislada, y el loss
ordinal BCE con thresholds globales puede castigar cruces de cortes sin
garantizar que la secuencia resultante sea una indentacion YAML valida.

El resultado tambien refuerza la hipotesis de desbalance. Los levels `5..8`
existen en el dataset, pero tienen mucho menos soporte que `0..4`. Cuando se
empuja al modelo a producirlos, aparecen, pero no necesariamente anclados a las
lineas adecuadas. Esto sugiere que la informacion latente esta parcialmente
disponible, pero la supervision actual no extrae una regla estructural estable.

## Implicacion para siguientes experimentos

La siguiente decision no deberia ser simplemente seguir subiendo learning rates
o cerrar mas el gap. Este run ya muestra que el cabezal puede emitir levels
profundos si la escala se lo permite. El problema pendiente es que esos levels
sean correctos y estables.

Las direcciones mas justificadas por este resultado son:

- probar el cabezal de regresion con Huber loss, que penaliza explicitamente la
  distancia ordinal entre nivel predicho y gold sin depender de thresholds
  globales aprendidos;
- estudiar weighting o sampling por niveles profundos, porque el desbalance
  sigue siendo una explicacion fuerte de por que `5..8` aparecen tarde y mal;
- mantener el diagnostico oracle de levels en las siguientes runs, porque
  separa bien los fallos de indentacion de los fallos de contenido;
- no interpretar BLEU/ROUGE como senal principal: en esta run suben, mientras
  las metricas parser-facing y de Kubernetes caen.

El resultado no invalida la idea de inicializar thresholds centrados. La idea
era correcta como control de escala. Lo que invalida es considerarla una
solucion suficiente para la recuperacion jerarquica.

## Artefactos consultados

```text
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/config.json
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/state.json
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/metrics.json
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/train_log.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/threshold_history.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/gradient_diagnostics.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/intermediate_validation_metrics.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/intermediate_validation_predictions.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/validation_predictions.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/validation_example_metrics.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/validation_metrics_progress.jsonl
```
