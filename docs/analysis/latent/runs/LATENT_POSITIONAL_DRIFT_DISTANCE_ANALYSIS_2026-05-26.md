# Analisis de distancias para drift posicional latente 2026-05-26

Document type: run result

## Resumen

Este analisis comprueba una primera version cuantitativa de la hipotesis de
drift posicional documentada en:

```text
docs/analysis/latent/LATENT_POSITIONAL_DRIFT_HYPOTHESIS_2026-05-26.md
```

La pregunta es si los estados latentes usados para predecir `level` cambian de
forma apreciable segun la posicion relativa de la linea dentro del YAML. Para
ello se reutilizaron los vectores ya extraidos por el run:

```text
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/
```

No se volvio a cargar el modelo. El analisis usa los artefactos
`features_*.jsonl`, que contienen un vector de dimension `3584` por linea,
junto con `split`, `unit_id`, `line_position`, `level` y `feature_strategy`.

El resultado principal es que la senal de drift aparece ya en estos artefactos
teacher-forced del modelo base. En particular, para `record_prefix_state`, que
es la estrategia mas comparable con el cabezal de nivel del modelo
`two_head_sft`, la distancia media entre cuartiles para el mismo `level` es
practicamente igual a la distancia media entre niveles distintos dentro del
mismo cuartil.

La lectura conceptual del resultado es temporal. La generacion de tokens,
niveles y estados latentes puede verse como una serie autoregresiva:

```text
token_t -> hidden_t -> level_t
```

Desde esta perspectiva, el analisis no pregunta solo si dos niveles son
separables, sino si la regla `hidden_t -> level_t` permanece estable a medida
que avanza el YAML. Los resultados apuntan a que no: la posicion relativa de la
linea cambia la distribucion latente sobre la que opera el cabezal.

## Metodo

Cada secuencia se dividio en cuatro cuartiles por longitud relativa:

```text
Q1 = 0-25%
Q2 = 25-50%
Q3 = 50-75%
Q4 = 75-100%
```

Para cada `feature_strategy`, se estandarizaron globalmente las dimensiones del
vector usando todas las lineas incluidas (`train` + `validation`). Despues se
calcularon centroides por:

```text
scope x level x quartile
```

donde `scope` toma los valores `all`, `train` y `validation`.

Se calcularon dos familias de distancias:

1. distancias entre cuartiles para el mismo `level`;
2. distancias entre niveles distintos dentro del mismo cuartil.

El ratio principal compara ambas magnitudes:

```text
positional_to_level_ratio =
  mean_same_level_quartile_cosine_distance
  /
  mean_interlevel_same_quartile_cosine_distance
```

Un valor cercano a `1.0` indica que el desplazamiento posicional dentro de un
mismo nivel es comparable a la separacion entre niveles diferentes. Un valor por
encima de `1.0` indica que, bajo esta metrica, el drift posicional es incluso
mayor que la separacion media entre niveles.

Se uso `min_group_count = 5` para evitar centroides calculados con soporte
demasiado pequeno.

## Resultado global por estrategia

| Feature strategy | Scope | Same-level quartile cosine | Inter-level same-quartile cosine | Ratio | Q1-Q4 same-level cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| `record_prefix_state` | all | 0.6684 | 0.6499 | 1.0285 | 1.2511 |
| `record_prefix_state` | train | 0.6676 | 0.6416 | 1.0406 | 1.2345 |
| `record_prefix_state` | validation | 0.7844 | 0.7854 | 0.9988 | 1.2921 |
| `line_prefix_state` | all | 0.6740 | 0.8619 | 0.7819 | 1.1330 |
| `line_prefix_state` | train | 0.6637 | 0.8647 | 0.7675 | 1.0621 |
| `line_prefix_state` | validation | 0.7902 | 0.9208 | 0.8581 | 1.1423 |
| `line_first_token` | all | 0.7208 | 0.8783 | 0.8207 | 1.0562 |
| `line_first_token` | train | 0.7025 | 0.8814 | 0.7970 | 1.0024 |
| `line_first_token` | validation | 0.8632 | 0.9507 | 0.9079 | 1.1289 |
| `line_last_token` | all | 0.8097 | 0.7113 | 1.1383 | 1.1620 |
| `line_last_token` | train | 0.7853 | 0.7096 | 1.1067 | 1.0676 |
| `line_last_token` | validation | 0.8543 | 0.8663 | 0.9862 | 1.3813 |
| `line_mean` | all | 0.7171 | 0.8174 | 0.8773 | 1.0252 |
| `line_mean` | train | 0.6895 | 0.8203 | 0.8406 | 0.9652 |
| `line_mean` | validation | 0.8023 | 0.8924 | 0.8990 | 1.1371 |

El resultado mas importante para la hipotesis aparece en
`record_prefix_state`. En `all`, el ratio es `1.0285`; en `train`, `1.0406`;
y en `validation`, `0.9988`. Esto significa que, usando el estado
autoregresivo previo al registro de la linea, la diferencia entre posiciones
del YAML para el mismo `level` tiene un tamano comparable a la diferencia entre
niveles.

Que el patron aparezca tambien en `train` es relevante. No parece ser solo un
artefacto de la validacion ni una consecuencia de tener pocas muestras en el
split de validacion. La senal ya esta presente en el conjunto usado para ajustar
los probes originales.

## Detalle de Q1 frente a Q4 en `record_prefix_state`

La comparacion Q1-Q4 es especialmente dura, porque contrasta estados muy
tempranos con estados tardios de la secuencia. En `record_prefix_state`, las
distancias para niveles con soporte suficiente fueron:

| Scope | Level | Q1 count | Q4 count | Cosine distance |
| --- | ---: | ---: | ---: | ---: |
| all | 0 | 1810 | 96 | 1.0441 |
| all | 1 | 738 | 450 | 1.4823 |
| all | 2 | 152 | 380 | 1.3145 |
| all | 3 | 18 | 422 | 1.1636 |
| train | 0 | 1510 | 92 | 1.0541 |
| train | 1 | 606 | 418 | 1.4703 |
| train | 2 | 116 | 364 | 1.2908 |
| train | 3 | 6 | 358 | 1.1229 |
| validation | 1 | 132 | 32 | 1.3882 |
| validation | 2 | 36 | 16 | 1.1991 |
| validation | 3 | 12 | 64 | 1.2889 |

Esto apunta a una lectura concreta: incluso para el mismo `level`, el estado
latente temprano y el tardio no son intercambiables. En algunos casos, como
`level=1` y `level=2`, la distancia Q1-Q4 es muy alta. Esto encaja con la
hipotesis de que el estado autoregresivo se va transformando a medida que el
modelo acumula contexto generado.

## Separabilidad inicial frente a drift temporal

Una interpretacion tentadora seria concluir que los estados iniciales son
"mas pobres" porque el modelo ha acumulado menos contexto. Sin embargo, las
metricas actuales no demuestran eso. Lo que demuestran es que la distribucion
cambia con la posicion.

Para comprobar si el inicio parece menos separable entre niveles, se revisaron
las distancias entre centroides de distintos `level` dentro de cada cuartil en
`record_prefix_state`. Usando solo los niveles comunes `0-3`, la separacion
media entre niveles no es menor en Q1:

| Quartile | Pairs | Mean cosine distance | Mean euclidean distance |
| --- | ---: | ---: | ---: |
| Q1_0_25 | 6 | 0.8813 | 44.81 |
| Q2_25_50 | 6 | 0.9273 | 29.79 |
| Q3_50_75 | 6 | 0.7813 | 22.08 |
| Q4_75_100 | 6 | 0.2903 | 15.14 |

Esto no debe interpretarse como que Q1 sea mejor que Q4, porque los soportes,
las clases presentes y la geometria local cambian mucho entre tramos. Pero si
sirve para descartar una afirmacion demasiado fuerte: con estos resultados no
se puede decir que la representacion del inicio sea intrinsecamente menos
diversa o menos separable entre niveles.

La conclusion mas robusta es otra: el significado geometrico de un hidden state
depende del momento de generacion. Un mismo `level` puede ocupar regiones
latentes distintas al principio y al final del YAML, y un cabezal global que
ignore esa condicion temporal puede estar aprendiendo una frontera promedio mal
calibrada para algunos tramos de la serie.

## Tests de significancia

Para comprobar si las distancias anteriores podian explicarse por variabilidad
de muestreo, se anadio un segundo analisis centrado en
`record_prefix_state`. Este analisis usa permutaciones con `499` iteraciones y
submuestreo balanceado por grupo (`max_per_group = 120`). Siempre que fue
posible se tomo como maximo una fila por `unit_id` dentro de cada grupo
`level x quartile`, para reducir dependencia entre lineas del mismo YAML.

Se aplicaron dos pruebas principales:

- MMD con kernel RBF para comparar `Q1` contra `Q4` dentro del mismo `level`;
- PERMANOVA para comprobar si el cuartil explica distancias multivariantes
  dentro de un mismo `level`.

Ademas se calculo PERMDISP sobre los mismos grupos de PERMANOVA. Esto es
importante porque una PERMANOVA significativa puede reflejar diferencias de
centroide, diferencias de dispersion, o ambas cosas.

Resumen global:

| Test | Comparaciones | Significativas con q < 0.05 |
| --- | ---: | ---: |
| MMD Q1-Q4 | 10 | 10 |
| PERMANOVA por cuartil | 23 | 23 |
| PERMDISP asociado | 23 | 19 |

Con `499` permutaciones, el menor p-valor observable es `0.002`. Por eso los
resultados que aparecen como `p = 0.002` no deben leerse como p-valores exactos,
sino como evidencia de que el estadistico observado quedo por encima de todas
las permutaciones aleatorias evaluadas.

Detalle de MMD Q1-Q4:

| Scope | Level | n por grupo | MMD^2 | p | q | media nula |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0 | 96 | 0.1426 | 0.002 | 0.002 | 0.0085 |
| all | 1 | 120 | 0.2003 | 0.002 | 0.002 | 0.0063 |
| all | 2 | 120 | 0.2086 | 0.002 | 0.002 | 0.0065 |
| all | 3 | 18 | 0.3560 | 0.002 | 0.002 | 0.0429 |
| train | 0 | 92 | 0.1463 | 0.002 | 0.002 | 0.0087 |
| train | 1 | 120 | 0.2221 | 0.002 | 0.002 | 0.0063 |
| train | 2 | 116 | 0.1850 | 0.002 | 0.002 | 0.0065 |
| validation | 1 | 32 | 0.3838 | 0.002 | 0.002 | 0.0220 |
| validation | 2 | 16 | 0.5066 | 0.002 | 0.002 | 0.0473 |
| validation | 3 | 12 | 0.4076 | 0.002 | 0.002 | 0.0631 |

Estos resultados son especialmente relevantes para la hipotesis porque no
comparan niveles distintos: comparan estados del mismo `level` en posiciones
opuestas del YAML. La diferencia Q1-Q4 es significativa en todas las
comparaciones con soporte suficiente.

Detalle resumido de PERMANOVA:

| Scope | Levels significativos | R2 aproximado | Nota PERMDISP |
| --- | ---: | --- | --- |
| all | 8/8 | 0.061-0.173 | significativo en 7/8 niveles |
| train | 8/8 | 0.052-0.156 | significativo en 8/8 niveles |
| validation | 7/7 | 0.047-0.314 | significativo en 4/7 niveles |

La PERMANOVA refuerza que el cuartil tiene informacion latente medible dentro
de un mismo nivel. A la vez, PERMDISP obliga a formular la conclusion con
cuidado: en muchos niveles, el efecto posicional no es solo un desplazamiento
del centroide, sino tambien un cambio de dispersion. Para la hipotesis del
cabezal de `level`, esto no debilita el resultado; de hecho puede ser igual o
mas importante, porque un MLP global tendria que generalizar entre regiones que
cambian tanto en posicion media como en variabilidad interna.

Desde la lectura de serie temporal, MMD y PERMANOVA no localizan aun el punto
exacto donde cambia la distribucion. Los cuartiles son una aproximacion gruesa:
detectan que la serie no es estacionaria, pero no dicen si el cambio es
gradual, si ocurre alrededor de una transicion estructural concreta, o si
aparece por saltos cuando el YAML entra en bloques como `metadata`, `spec`,
`template` o `containers`.

## Interpretacion

El resultado no demuestra todavia que este drift sea la causa directa del fallo
del cabezal ordinal en el ultimo `two_head_sft`. Los vectores analizados
proceden del modelo base y de un input controlado del probe, no de la
generacion libre del modelo fine-tuned. Por tanto, la conclusion debe formularse
con cuidado.

Lo que si muestra este analisis es que la hipotesis tiene soporte empirico
inicial. El espacio latente no parece estacionario a lo largo del YAML. En la
estrategia mas autoregresiva, `record_prefix_state`, el cambio posicional
dentro del mismo nivel tiene una magnitud similar a la separacion entre niveles.
Esto hace plausible que un unico MLP global pueda estar leyendo regiones
latentes con propiedades distintas al principio y al final de la secuencia.

La formulacion mas prudente no es que el inicio del YAML sea necesariamente
menos informativo, sino que la posicion temporal actua como una condicion de
dominio para interpretar el hidden state. El cabezal de `level` no deberia
asumir sin mas que un hidden state de Q1 y uno de Q4 son muestras
intercambiables de la misma distribucion, aunque ambos compartan el mismo
`level` gold.

El analisis tambien sugiere que la posicion no actua igual en todas las
estrategias de lectura. `line_prefix_state`, `line_first_token` y `line_mean`
mantienen ratios por debajo de `1.0`, aunque sus distancias Q1-Q4 siguen siendo
altas. En cambio, `line_last_token` tambien supera `1.0` en `all` y `train`.
Esto podria indicar que los estados mas cercanos a decisiones autoregresivas
completas concentran mas drift posicional que los estados que incorporan
informacion directa de la linea.

## Siguientes comprobaciones

El siguiente paso natural es no quedarse solo en distancias de centroides. Este
resultado justifica dos experimentos complementarios:

- entrenar un clasificador `hidden_state -> quartile`, incluyendo versiones
  condicionadas a `level`, para comprobar si el cuartil se puede recuperar
  incluso dentro de un mismo nivel gold;
- hacer transferencia cruzada de probes de nivel entre cuartiles, entrenando en
  Q1, Q2, Q3 o Q4 y evaluando en el resto.

Tambien conviene repetir este analisis con hidden states del checkpoint
fine-tuned de `two_head_sft`, si los artefactos o el coste de extraccion lo
permiten. La pregunta importante seria si el SFT reduce este drift, lo mantiene
o lo amplifica durante la generacion libre.

## Implicaciones de modelado

El resultado no recomienda, por defecto, crear clases discretas combinadas del
tipo `level x position_bin`. Esa solucion codificaria explicitamente el tramo
temporal, pero multiplicaria el numero de clases y agravaria el imbalance de
niveles profundos.

La direccion mas limpia es mantener el mismo espacio de salida de `level` y
anadir informacion temporal al cabezal:

- usar `relative_position` como feature continua junto al hidden state;
- introducir embeddings posicionales o Fourier features para representar el
  momento de generacion;
- usar gating o FiLM para que la posicion module la lectura del hidden state;
- anadir un sesgo posicional sobre los logits o sobre el `ordinal_score`;
- probar calibracion por tramo solo como ablation, no como solucion principal.

Estas opciones comparten la misma intuicion: el cabezal no necesita mas clases,
sino una forma de saber desde que zona temporal de la serie esta leyendo el
estado latente.

## Artefactos generados

```text
scripts/analyze_latent_positional_drift.py
scripts/test_latent_positional_drift_significance.py
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_analysis/group_stats.csv
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_analysis/same_level_quartile_distances.csv
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_analysis/interlevel_same_quartile_distances.csv
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_analysis/drift_summary.csv
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_analysis/scaler_summary.csv
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_analysis/summary.json
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_significance/record_prefix_state/mmd_q1_q4_tests.csv
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_significance/record_prefix_state/permanova_quartile_tests.csv
results/latent_level_probe_kubernetes_v1/latent-level-probe-real-full-20260513-1528/positional_drift_significance/record_prefix_state/summary.json
```
