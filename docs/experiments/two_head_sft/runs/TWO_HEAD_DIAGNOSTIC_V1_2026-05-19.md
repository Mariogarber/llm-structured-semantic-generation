# Diagnostico two_head_sft v1 - 2026-05-19

## Objetivo

Este documento registra un diagnostico especifico del primer run completo
`two_head_sft`:

```text
results/two_head_sft_kubernetes_v1/two-head-sft-v1-20260516/
```

La pregunta no es si el modelo gana o pierde frente a `serialized_sft`, sino por
que la cabeza estructural de `level` colapsa en niveles poco profundos y como
conviene orientar el siguiente experimento.

El diagnostico se calcula con:

```powershell
uv run python scripts\diagnose_two_head_sft_v1.py
```

El artefacto numerico completo queda en:

```text
results/two_head_diagnostics_v1/two-head-sft-v1-20260516/diagnostic_metrics.json
```

## Datos usados

Entradas:

```text
data/processed/kubernetes_v1/sft/train.jsonl
data/processed/kubernetes_v1/sft/validation.jsonl
data/processed/kubernetes_v1/sft/test.jsonl
results/two_head_sft_kubernetes_v1/two-head-sft-v1-20260516/validation_predictions.jsonl
```

Conteos:

| Split | Filas |
| --- | ---: |
| train | 426 |
| validation | 70 |
| test | 70 |
| predictions | 70 |

## Distribucion de niveles

La distribucion de entrenamiento esta muy desbalanceada:

| Level | Train lines | Train % | Validation lines | Validation % |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2122 | 25.46 | 324 | 18.00 |
| 1 | 2326 | 27.90 | 294 | 16.33 |
| 2 | 1568 | 18.81 | 230 | 12.78 |
| 3 | 982 | 11.78 | 328 | 18.22 |
| 4 | 832 | 9.98 | 450 | 25.00 |
| 5 | 242 | 2.90 | 114 | 6.33 |
| 6 | 216 | 2.59 | 44 | 2.44 |
| 7 | 40 | 0.48 | 4 | 0.22 |
| 8 | 8 | 0.10 | 12 | 0.67 |

Los niveles `5..8` son solo el `6.07%` de las lineas de train, pero el `9.67%`
de validation. El caso mas extremo es `level=8`: train tiene solo `8` lineas,
mientras validation tiene `12`.

Si se aplicara una ponderacion inversa simple de cross-entropy sobre train, los
pesos relativos serian aproximadamente:

| Level | Balanced CE weight |
| ---: | ---: |
| 0 | 0.436 |
| 1 | 0.398 |
| 2 | 0.591 |
| 3 | 0.943 |
| 4 | 1.113 |
| 5 | 3.827 |
| 6 | 4.288 |
| 7 | 23.156 |
| 8 | 115.778 |

Esto no significa que deban usarse sin limite. De hecho, para `7` y `8` seria
mas prudente usar pesos suavizados o capados. Pero la tabla muestra la escala
real del desbalance.

## Colapso de la cabeza de nivel

En las predicciones finales, la cabeza no emite ningun nivel mayor que `4`:

| Level predicho | Lineas |
| ---: | ---: |
| 0 | 300 |
| 1 | 278 |
| 2 | 304 |
| 3 | 431 |
| 4 | 411 |

La matriz alineada por posicion confirma que todos los niveles profundos se
comprimen hacia `2`, `3` o `4`:

| Gold level | Support alineado | Pred 2 | Pred 3 | Pred 4 | Recall exacto |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 80 | 11 | 32 | 37 | 0.000 |
| 6 | 40 | 1 | 14 | 25 | 0.000 |
| 7 | 3 | 0 | 2 | 1 | 0.000 |
| 8 | 9 | 0 | 2 | 7 | 0.000 |

Resumen para niveles `5..8`:

```text
support alineado = 132
predicciones en 5..8 = 0
recall exacto = 0.0000
recall off-by-one = 0.2803
compresion hacia 0..4 = 1.0000
```

El valor `off-by-one = 0.2803` indica que parte de la senal ordinal existe:
algunos `level=5` se predicen como `4`. Pero el clasificador nunca cruza el
umbral de decision hacia una clase profunda.

## Parseabilidad y profundidad

La diferencia entre salidas parseables y no parseables refuerza la lectura
estructural:

| Grupo | Casos | Target lines | Pred lines | Line delta | Target max level | Pred max level |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| parseable | 34 | 25.03 | 21.38 | -3.65 | 4.85 | 3.38 |
| failed | 36 | 26.36 | 27.69 | +1.33 | 5.03 | 3.89 |

Las salidas parseables tienden a ser mas cortas que el target. Las fallidas
tienden a intentar generar mas lineas, pero siguen sin subir de profundidad. La
lectura probable es:

- cuando el modelo simplifica o omite estructura, a veces consigue YAML
  parseable;
- cuando intenta cubrir mas estructura anidada, la cabeza comprime niveles y el
  parser falla.

Tambien hay mas recursos con contenedores y volumenes entre los fallos:

| Grupo | `has_container` | `has_volume` | `has_command` |
| --- | ---: | ---: | ---: |
| parseable | 0.794 | 0.265 | 0.588 |
| failed | 0.972 | 0.361 | 0.444 |

El patron es consistente con la auditoria cualitativa: `containers`,
`volumeMounts`, `volumes` y listas internas concentran los errores.

## Counterfactuals

El script calcula tres vistas. Deben leerse como diagnostico, no como metricas
oficiales del modelo.

| Vista | YAML parse success |
| --- | ---: |
| referencia: contenido gold + niveles gold | 1.0000 |
| actual: contenido generado + niveles predichos | 0.4857 |
| contenido generado + niveles gold por posicion | 0.3857 |
| contenido gold + niveles predichos por posicion | 0.3714 |

La vista `contenido generado + niveles gold por posicion` no mejora el resultado
actual porque la alineacion por posicion es ruidosa: hay lineas generadas de mas
o de menos. El script registra `176` posiciones no alineadas en esa mezcla. Por
tanto, no debe usarse para afirmar que "los niveles gold empeoran".

La vista mas informativa es la contraria: `contenido gold + niveles predichos`.
Incluso sustituyendo el contenido por las lineas de referencia, los niveles
predichos por posicion solo producen YAML parseable en `0.3714` de los casos.
Esto apunta a que la secuencia de niveles, por si sola, ya es capaz de romper
manifiestos correctos.

Para reducir el ruido de longitud, se repite el corte solo en los `19` ejemplos
donde la prediccion y la referencia tienen el mismo numero de lineas:

| Vista, solo mismo numero de lineas | Casos | YAML parse success |
| --- | ---: | ---: |
| actual | 19 | 0.5789 |
| contenido gold + niveles predichos | 19 | 0.3158 |

Este corte es pequeno, pero importante: cuando se elimina el problema de lineas
faltantes o extra, los niveles predichos siguen degradando fuertemente la
parseabilidad del contenido gold.

## Diagnostico

El fallo no parece explicarse solo por una arquitectura MLP "pequena". Hay al
menos cuatro factores combinados:

1. **Desbalance de clases**. Los niveles `5..8` tienen poco soporte en train,
   sobre todo `7` y `8`.
2. **Objetivo nominal simple**. La cross-entropy trata `level` como clase
   nominal, aunque es una variable ordinal.
3. **Alineacion estricta**. `record_prefix_state` predice antes de ver el
   contenido de la linea actual. Esto es limpio conceptualmente, pero puede ser
   debil en listas profundas donde la linea actual ayuda a distinguir si se
   continua una lista, se abre un mapping o se vuelve hacia arriba.
4. **Competencia con la LM loss**. Con `lambda_level = 1.0`, el entrenamiento
   puede optimizar bien la generacion textual sin forzar suficiente separacion
   de clases estructurales raras.

El contraste con los probes previos es relevante. En el analisis de probes,
`record_prefix_state + MLP` ya era debil en niveles profundos, pero no colapsaba
completamente. El run integrado si colapsa a `0..4`, lo que sugiere que la senal
existe parcialmente, pero el entrenamiento conjunto no la calibra bien para
clases raras.

## Siguientes experimentos propuestos

### Experimento A: weighted ordinal level head

Mantener `record_prefix_state`, pero cambiar la supervision de nivel:

```text
level_loss = weighted_cross_entropy + alpha * ordinal_distance_loss
```

Configuracion sugerida:

- pesos por clase suavizados, no inversos puros;
- cap maximo de peso entre `8` y `12`;
- `alpha` inicial entre `0.1` y `0.3`;
- `lambda_level = 2.0`;
- metricas obligatorias por clase y `deep_level_recall_5_8`.

Objetivo minimo:

```text
predicted_count(level 5..8) > 0
deep_level_exact_recall > 0
yaml_parse_success_rate > 0.4857
```

### Experimento B: cabeza MLP mas expresiva

Sustituir la cabeza actual:

```text
Linear(hidden_size -> 256)
GELU
Dropout(0.05)
Linear(256 -> 9)
```

por una version moderadamente mas potente:

```text
LayerNorm(hidden_size)
Linear(hidden_size -> 512)
GELU
Dropout(0.10)
Linear(512 -> 256)
GELU
Dropout(0.05)
Linear(256 -> 9)
```

Esto no debe presentarse como solucion por si solo. Debe ir junto con la perdida
ponderada u ordinal, porque el problema principal observado es de decision y
calibracion sobre clases raras.

### Experimento C: alternativa de alineacion

Entrenar una variante con `line_prefix_state`:

- mismo dataset;
- misma loss ponderada/ordinal;
- misma cabeza ampliada;
- unica diferencia principal: posicion de hidden state usada para `level`.

Motivacion: los probes previos mostraron que `line_prefix_state` mejora el
comportamiento relativo en algunos niveles profundos, aunque pierde accuracy
global.

### Experimento D: fusion de estados

Si A y C no bastan, probar una cabeza que fusione:

```text
record_prefix_state
line_prefix_state
line_last_token
```

Esta variante es mas invasiva y debe ir despues de las anteriores, porque mezcla
mas causas: contexto previo, posicion de linea y contenido de linea.

## Recomendacion inmediata

Antes de lanzar otro entrenamiento completo, conviene modificar el trainer para
registrar siempre:

- distribucion de niveles predichos;
- matriz de confusion por nivel;
- recall de `level=5..8`;
- `predicted_max_level`;
- metricas separadas para salidas parseables y no parseables.

El siguiente run recomendado es:

```text
two_head_weighted_ordinal_v2
```

con `record_prefix_state`, perdida ponderada/ordinal y cabeza MLP ampliada. Esa
es la comparacion mas limpia: cambia la capacidad/calibracion de la cabeza sin
cambiar todavia la hipotesis de alineacion.
