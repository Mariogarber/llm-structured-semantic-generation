# Analisis del run two_head_ordinal_density_v2 MLP3 threshold LR50 20260522

Document type: run result

## Resumen

El run `two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522`
continua la serie ordinal density v2 despues de la variante
`threshold_learning_rate_multiplier=25`. En este caso se separan tres grupos
de learning rate:

```text
base_learning_rate = 2e-4
ordinal_mlp_learning_rate_multiplier = 3
threshold_learning_rate_multiplier = 50
```

La configuracion mantiene el resto del contrato experimental:

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
```

El run termino correctamente:

```text
status = completed
completed_at = 2026-05-22T23:10:10.497861Z
global_step = 159
best_checkpoint = checkpoint-step-159
oom_skipped_batches = 2
```

Los dos OOM corresponden al mismo patron ya visto en runs anteriores y fueron
gestionados mediante `oom_recovery=skip_batch`. No se borraron checkpoints. La
run quedo sincronizada con W&B y con artefactos subidos.

## Resultado corto

El experimento no mejora la recuperacion estructural. De hecho, empeora frente
a la variante `threshold_learning_rate_multiplier=25`:

- `yaml_parse_success_rate` baja de `0.2857` a `0.1429`.
- `average_level_exact_match_rate` baja de `0.0693` a `0.0409`.
- `deep_level_exact_recall_5_8` sigue en `0`.
- `deep_level_off_by_one_recall_5_8` cae de `0.2536` a `0`.
- `compressed_deep_to_0_4_rate` sigue en `1.0`.
- `predicted_level_count_4..8` queda en cero.

La lectura principal es que subir simultaneamente el learning rate del MLP
ordinal a `3x` y el de thresholds a `50x` no resolvio el desajuste de escala.
El modelo sigue usando una region demasiado baja de la escala ordinal: la
validacion final no emite ningun nivel `4..8`, aunque el gold contiene soporte
amplio en esos niveles.

## Comparacion con runs anteriores

| Metrica final | Inicial | LR25 | MLP3/THR50 |
| --- | ---: | ---: | ---: |
| `yaml_parse_success_rate` | 0.2571 | 0.2857 | 0.1429 |
| `block_parse_success_rate` | 0.2571 | 0.2857 | 0.1429 |
| `average_level_exact_match_rate` | 0.0554 | 0.0693 | 0.0409 |
| `average_level_mae` | 0.7695 | 0.8062 | 0.9182 |
| `deep_level_exact_recall_5_8` | 0.0000 | 0.0000 | 0.0000 |
| `deep_level_off_by_one_recall_5_8` | 0.2190 | 0.2536 | 0.0000 |
| `compressed_deep_to_0_4_rate` | 0.9927 | 1.0000 | 1.0000 |
| `predicted_max_level_mean` | 3.6857 | 3.6857 | 2.9143 |
| `target_max_level_ge_5_yaml_parse_success_rate` | 0.2609 | 0.2391 | 0.0435 |
| `average_line_text_f1` | 0.1899 | 0.2269 | 0.1153 |
| `primary_kind_match_rate` | 0.8333 | 0.8500 | 0.7000 |
| `primary_api_version_match_rate` | 0.8333 | 0.8500 | 0.7000 |
| `average_kubernetes_domain_validity_score` | 0.2167 | 0.2381 | 0.1190 |

El resultado es negativo en casi todas las metricas finales. No solo no se
recuperan los niveles profundos, sino que tambien baja la calidad del contenido
generado. Esto sugiere que el objetivo ordinal mas agresivo esta interfiriendo
con el ajuste compartido de la representacion, no solo moviendo mejor los
cortes.

## Distribucion de niveles

| Nivel | Gold inicial | Pred inicial | Gold LR25 | Pred LR25 | Gold MLP3/THR50 | Pred MLP3/THR50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 312 | 503 | 316 | 502 | 315 | 497 |
| 1 | 262 | 245 | 266 | 262 | 271 | 221 |
| 2 | 208 | 297 | 215 | 291 | 213 | 558 |
| 3 | 291 | 337 | 293 | 340 | 307 | 401 |
| 4 | 372 | 199 | 381 | 214 | 424 | 0 |
| 5 | 87 | 1 | 89 | 0 | 97 | 0 |
| 6 | 38 | 0 | 37 | 0 | 38 | 0 |
| 7 | 3 | 0 | 3 | 0 | 3 | 0 |
| 8 | 9 | 0 | 9 | 0 | 9 | 0 |

El punto critico es el nivel `4`. En runs anteriores habia al menos cierta
masa predicha en `level=4`; en MLP3/THR50 desaparece por completo. Como en YAML
de Kubernetes muchos campos internos de `containers`, `env`, `command`,
`livenessProbe`, `volumeMounts` y listas anidadas viven en `level >= 4`, esta
compresion rompe directamente la reconstruccion.

## Movimiento de thresholds y escala `z`

| Threshold | Inicio | Final LR25 | Drift LR25 | Final MLP3/THR50 | Drift MLP3/THR50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tau_0` | 0.5000 | 0.4512 | -0.0488 | 0.4230 | -0.0770 |
| `tau_1` | 1.5000 | 1.4235 | -0.0765 | 1.3904 | -0.1096 |
| `tau_2` | 2.5000 | 2.3956 | -0.1044 | 2.3660 | -0.1340 |
| `tau_3` | 3.5000 | 3.3695 | -0.1305 | 3.3512 | -0.1488 |
| `tau_4` | 4.5000 | 4.3512 | -0.1488 | 4.3557 | -0.1443 |
| `tau_5` | 5.5000 | 5.3427 | -0.1573 | 5.3829 | -0.1171 |
| `tau_6` | 6.5000 | 6.3969 | -0.1031 | 6.5293 | +0.0293 |
| `tau_7` | 7.5000 | 7.4776 | -0.0224 | 7.7065 | +0.2065 |

Los thresholds se mueven mas que en el run inicial, pero no se colocan en una
escala util. La ultima muestra de entrenamiento registrada tiene:

```text
ordinal_z_mean = -0.6435
ordinal_z_min = -6.4004
ordinal_z_max = 1.9208
```

Ese `z_max` corresponde solo al ultimo batch de entrenamiento, no a toda la
validacion, pero es consistente con el resultado final: en validacion no se
predice ningun nivel `4..8`, lo que implica que la escala efectiva de `z` no
esta cruzando los cortes altos. Para predecir `level=4`, `z` tendria que
superar `tau_3 = 3.3512`; para `level=5`, tendria que superar
`tau_4 = 4.3557`.

Las diagnosticas de gradiente finales fueron:

| Magnitud | Valor |
| --- | ---: |
| `effective_z_mean_abs_shift` | 0.0034321 |
| `effective_tau_mean_abs_shift` | 0.0000871 |
| `effective_z_to_tau_mean_shift_ratio` | 39.3934 |

La proporcion indica que, paso a paso, la salida `z` sigue teniendo mas
capacidad efectiva de desplazamiento local que los thresholds transformados.
Sin embargo, eso no garantiza que la escala global quede bien calibrada: el MLP
puede mover muestras concretas, pero los cortes siguen partiendo de una
geometria demasiado alta y positiva para la activacion que realmente aprende.

## Por que el YAML parse es tan bajo

La metrica oficial final es:

```text
yaml_parse_success_rate = 0.14285714285714285
kubernetes_domain_error_counts.yaml_parse = 60
```

El fallo no viene de que el modelo no emita el formato de bloques. De hecho:

```text
structured_output_parse_success_rate = 1.0
average_valid_block_ratio = 1.0
document_index_monotonic_ok_rate = 1.0
line_index_sequence_ok_rate = 1.0
```

Es decir, el modelo produce bloques interpretables por el contrato
`content_blocks_v1`, pero los levels predichos reconstruyen un YAML mal
indentado.

Se hizo una prueba diagnostica adicional sobre los artefactos finales. En los
18 ejemplos donde el numero de lineas generadas coincide con el gold, se
reconstruyo el mismo `line_text` generado de dos formas:

1. con los levels predichos por el modelo;
2. con los levels gold, como oracle de indentacion.

El resultado fue:

| Reconstruccion en ejemplos con igual numero de lineas | Parse OK |
| --- | ---: |
| `line_text` generado + levels predichos | 3 / 18 |
| `line_text` generado + levels gold | 14 / 18 |

Ademas, 11 ejemplos que fallaban con los levels predichos pasaban a parsear
solo sustituyendo esos levels por los gold. Esto apunta con bastante fuerza a
que la causa dominante del YAML parse bajo es estructural: niveles demasiado
superficiales que colocan claves y elementos de lista en una profundidad
incorrecta.

El patron de error observado en la reconstruccion diagnostica tambien encaja
con esta lectura. La mayoria de fallos fueron `ParserError` con mensajes del
tipo `expected <block end>`, que suelen aparecer cuando un mapping o una lista
queda interrumpido por una clave al nivel equivocado.

Ejemplo conceptual:

```text
containers:
  - env:
    - name: GREETING_MESSAGE
    value: Hello
    image: busybox
```

Si `value`, `image`, `command` o los items de lista no quedan en la profundidad
esperada, PyYAML no puede decidir correctamente cuando termina la lista o el
mapping.

## Que no arreglaran automaticamente los siguientes fixes

El diagnostico oracle no dice que todo el parse bajo sea solo culpa de los
thresholds. En los 18 ejemplos con misma longitud, 4 seguian fallando incluso
con levels gold. Eso indica que tambien existen fallos locales de contenido:

- lineas de lista que deberian ser mappings compuestos;
- secuencias como `- env:` seguidas de items y claves hermanas mal ordenadas;
- valores con sintaxis YAML delicada;
- diferencias de longitud y omisiones de lineas respecto al gold.

Por tanto, centrar y compactar thresholds deberia atacar la causa dominante,
pero no puede garantizar parse perfecto. Si la siguiente run mejora los levels
pero el parse sigue bajo, habra que mirar una segunda familia de fixes centrada
en contenido y contrato de bloque: normalizacion de lineas de lista, decoding
mas conservador, o restricciones parser-facing mas fuertes.

## Implicacion para el siguiente experimento

Este run refuerza la decision de no seguir subiendo learning rates a ciegas.
El problema principal parece ser una inicializacion de escala poco natural:
los thresholds empezaban en:

```text
[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
```

Esto desperdicia la parte negativa de `z` como nivel cero y exige que el MLP
empuje `z` hasta valores cercanos a `5..7` para expresar niveles profundos.
La siguiente variante debe probar thresholds centrados en cero y mas compactos:

```text
--initial-threshold-center 0
--initial-threshold-gap 0.5
```

Con 9 niveles, esa inicializacion empieza en:

```text
[-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
```

Los thresholds siguen siendo entrenables. El cambio no introduce un parser
reparador ni redefine el objetivo; solo evita que la cabeza ordinal tenga que
aprender desde una escala inicial desalineada con la distribucion real de `z`.

La expectativa razonable es una mejora parcial de `yaml_parse_success_rate` si
el cuello principal era la indentacion. El criterio critico para validar el fix
no debe ser solo que suba el parse global, sino que aparezca masa real en:

```text
predicted_level_count_4
predicted_level_count_5
predicted_level_count_6
predicted_level_count_7
predicted_level_count_8
deep_level_exact_recall_5_8
deep_level_off_by_one_recall_5_8
target_max_level_ge_5_yaml_parse_success_rate
```

Si esos indicadores no se mueven, el problema no sera ya solo el origen y gap
inicial de los thresholds.

## Artefactos consultados

```text
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/config.json
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/state.json
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/metrics.json
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/train_log.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/threshold_history.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/gradient_diagnostics.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/validation_predictions.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/validation_example_metrics.jsonl
results/two_head_ordinal_sft_kubernetes_v1/two-head-ordinal-density-v2-mlp-lr3-threshold-lr50-20260522/validation_metrics_progress.jsonl
```
