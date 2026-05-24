# Analisis del run two_head_ordinal_density_v2 20260519

## Resumen

El run `two-head-ordinal-density-v2-20260519` completo 3 epocas sobre
`kubernetes_v1` con `batch_size=1`, `gradient_accumulation_steps=8`,
checkpoints cada 8 optimizer steps y validacion intermedia cada 32 steps. El
checkpoint final fue `checkpoint-step-159`.

La conclusion principal es que esta primera version no probo de forma fuerte la
hipotesis de umbrales aprendidos: los thresholds quedaron casi congelados. El
modelo aprendio sobre todo a mover la representacion latente `z`, mientras los
cortes globales `tau_0..tau_7` permanecieron cerca de su inicializacion
regular.

## Metricas finales

Validacion final sobre 70 ejemplos:

| Metrica | Valor |
| --- | ---: |
| `structured_output_parse_success_rate` | 1.0000 |
| `yaml_parse_success_rate` | 0.2571 |
| `block_parse_success_rate` | 0.2571 |
| `kubernetes_domain_gate_pass_rate` | 0.0143 |
| `average_level_exact_match_rate` | 0.0554 |
| `average_level_mae` | 0.7695 |
| `deep_level_exact_recall_5_8` | 0.0000 |
| `deep_level_off_by_one_recall_5_8` | 0.2190 |
| `compressed_deep_to_0_4_rate` | 0.9927 |
| `predicted_max_level_mean` | 3.6857 |
| `target_max_level_ge_5_yaml_parse_success_rate` | 0.2609 |

La metrica mas importante para este diagnostico es
`compressed_deep_to_0_4_rate = 0.9927`: casi todos los niveles gold profundos
`5..8` siguen comprimidos hacia `0..4`. La cabeza ordinal no corrigio el sesgo
que habia motivado esta version.

## Movimiento de umbrales

Los thresholds se inicializaron como cortes equiespaciados:

```text
tau = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
```

Tras 159 optimizer steps, el desplazamiento fue minimo:

| Threshold | Inicio | Final | Drift |
| --- | ---: | ---: | ---: |
| `tau_0` | 0.5000 | 0.4981 | -0.0019 |
| `tau_1` | 1.5000 | 1.4970 | -0.0030 |
| `tau_2` | 2.5000 | 2.4959 | -0.0041 |
| `tau_3` | 3.5000 | 3.4949 | -0.0051 |
| `tau_4` | 4.5000 | 4.4941 | -0.0059 |
| `tau_5` | 5.5000 | 5.4937 | -0.0063 |
| `tau_6` | 6.5000 | 6.4957 | -0.0043 |
| `tau_7` | 7.5000 | 7.4988 | -0.0012 |

Los gaps tambien cambiaron muy poco:

| Gap | Inicio | Final | Drift |
| --- | ---: | ---: | ---: |
| `gap_0` | 1.0000 | 0.9989 | -0.0011 |
| `gap_1` | 1.0000 | 0.9989 | -0.0011 |
| `gap_2` | 1.0000 | 0.9989 | -0.0011 |
| `gap_3` | 1.0000 | 0.9992 | -0.0008 |
| `gap_4` | 1.0000 | 0.9996 | -0.0004 |
| `gap_5` | 1.0000 | 1.0020 | +0.0020 |
| `gap_6` | 1.0000 | 1.0031 | +0.0031 |

Estos cambios son demasiado pequenos para considerar que el modelo haya
aprendido fronteras ordinales significativas. En la practica, los cortes se
mantuvieron casi como una cuadricula fija.

## Evolucion de `z`

La representacion latente si se movio de forma mucho mas visible:

| Estadistico de `z` | Valor |
| --- | ---: |
| `z_mean` inicial | 0.0079 |
| `z_min` inicial | -0.0596 |
| `z_max` inicial | 0.0454 |
| media de `z_mean` durante training | 1.6022 |
| minimo observado de `z` | -3.1331 |
| maximo observado de `z` | 8.8705 |
| `z_mean` final | -0.1538 |
| `z_min` final | -2.9692 |
| `z_max` final | 1.7186 |

La comparacion es clara: los thresholds se movieron en el orden de `1e-3`,
mientras que `z` llego a ocupar un rango observado aproximado de `[-3.13,
8.87]`. La perdida encontro un camino mas facil ajustando la proyeccion
`hidden -> z` que desplazando los cortes globales.

## Distribucion de niveles

Durante training, las predicciones por batch registradas en
`threshold_history.jsonl` sumaron:

| Nivel predicho | Conteo acumulado |
| --- | ---: |
| 0 | 827 |
| 1 | 567 |
| 2 | 683 |
| 3 | 490 |
| 4 | 241 |
| 5 | 128 |
| 6 | 88 |
| 7 | 40 |
| 8 | 11 |

En validacion final, sin embargo, la distribucion fue mucho mas comprimida:

| Nivel | Gold | Predicho |
| --- | ---: | ---: |
| 0 | 312 | 503 |
| 1 | 262 | 245 |
| 2 | 208 | 297 |
| 3 | 291 | 337 |
| 4 | 372 | 199 |
| 5 | 87 | 1 |
| 6 | 38 | 0 |
| 7 | 3 | 0 |
| 8 | 9 | 0 |

El modelo casi no emite niveles `5..8` en validacion. Esto explica que el
recall exacto profundo sea `0.0` y que la compresion profunda hacia `0..4`
alcance `0.9927`.

## Interpretacion

La parametrizacion ordinal era razonable para una primera prueba: un escalar
`z`, thresholds ordenados y perdida BCE acumulativa. El problema observado no
parece ser que los thresholds no puedan aprender, sino que con el mismo learning
rate que el resto de parametros apenas participan en la optimizacion.

Hay una degeneracion practica: los logits son `z - tau_k`. Si la cabeza MLP
puede mover `z` por muestra y los thresholds son solo ocho escalares globales,
la perdida puede mejorar antes ajustando `z` que recolocando `tau`. En este run,
eso es exactamente lo que indican los artefactos.

Por eso el siguiente experimento controlado es
`two-head-ordinal-density-v2-threshold-lr25-20260520`, que mantiene la misma
configuracion y solo cambia:

```text
--threshold-learning-rate-multiplier 25
```

Ese cambio no modifica la funcion objetivo ni la arquitectura. Solo da un paso
de optimizacion mayor a `raw_tau0` y `raw_deltas`, con `weight_decay=0.0`, para
comprobar si los cortes pueden salir de su inicializacion y si eso reduce la
compresion de niveles profundos.
