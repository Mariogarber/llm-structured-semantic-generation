# Propuesta two_head_ordinal_density_v2

## Objetivo

Este documento propone una segunda version experimental de `two_head_sft` para
atacar el fallo principal observado en el primer run:

```text
two-head-sft-v1-20260516
```

La auditoria y el diagnostico v1 muestran que la cabeza estructural no predice
ningun `level > 4`, aunque validation contiene niveles `5..8`. El problema no
parece ser solo de contenido generado: incluso con contenido de referencia, los
niveles predichos pueden romper la parseabilidad YAML.

La hipotesis de esta version es:

> El `level` no debe tratarse como una clase nominal independiente ni como una
> regresion continua redondeada, sino como una variable ordinal discreta
> obtenida a partir de una escala latente con umbrales aprendidos.

## Motivacion

En el primer `two_head_sft`, la cabeza de nivel es:

```text
hidden_state -> MLP -> logits(level=0..8)
```

La perdida es cross-entropy multiclase. Esta formulacion permite aprender las
clases, pero no incorpora directamente que:

- `level=6` esta mas cerca de `level=5` que de `level=2`;
- equivocarse de `5` a `4` es menos grave que comprimir `8` a `3`;
- los niveles profundos son raros y quedan subrepresentados;
- la decision entre `4` y `5` es critica para listas y mappings anidados.

Una regresion simple tampoco es suficiente. Si se entrena un escalar con MSE o
Huber y luego se redondea, el modelo puede seguir cayendo hacia la media:

```text
4.2 -> 4
4.4 -> 4
4.6 -> 5
```

Eso no resuelve necesariamente el sesgo observado. La alternativa mas adecuada
es aprender una escala continua, pero discretizarla con umbrales no uniformes.

## Cabeza ordinal con umbrales aprendidos

La formulacion propuesta usa una variable latente escalar:

```text
z = f_theta(hidden_state)
```

y un conjunto de umbrales ordenados:

```text
tau_0 < tau_1 < ... < tau_7
```

La prediccion se obtiene contando cuantos umbrales supera `z`:

```text
predicted_level = sum(z > tau_k for k in 0..7)
```

Esto permite que las regiones de la escala no tengan el mismo tamano. Por
ejemplo, si los niveles profundos necesitan mas sensibilidad, los umbrales entre
`4`, `5`, `6`, `7` y `8` pueden quedar mas juntos que los umbrales de niveles
frecuentes.

La cabeza podria implementarse como:

```text
LayerNorm(hidden_size)
Linear(hidden_size -> 512)
GELU
Dropout(0.10)
Linear(512 -> 256)
GELU
Dropout(0.05)
Linear(256 -> 1)          # z
ordered_thresholds(8)     # tau_0..tau_7
```

Para garantizar el orden de los umbrales, no se deben entrenar directamente como
ocho parametros libres. Una parametrizacion estable seria:

```text
tau_0 = raw_tau_0
tau_k = tau_{k-1} + softplus(delta_k)
```

con `delta_k` entrenable.

La inicializacion de esos umbrales no debe desperdiciar todo el semieje
negativo de `z` como nivel cero. Por defecto, los cortes iniciales se centran en
`0` con gap `1.0`:

```text
tau = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
```

Para estudiar escalas latentes mas compactas puede usarse
`--initial-threshold-gap 0.5`, que mantiene la misma parametrizacion entrenable
pero empieza en:

```text
tau = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
```

En ambos casos los thresholds siguen siendo aprendidos; solo cambia la
condicion inicial de la escala ordinal.

## Perdida ordinal

La perdida principal puede formularse como clasificacion ordinal acumulativa:

```text
target_k = 1 si level > k, si no 0
```

para cada umbral `k in 0..7`.

El modelo produce:

```text
logit_k = z - tau_k
```

y se entrena con binary cross-entropy sobre los ocho umbrales:

```text
ordinal_loss = BCEWithLogitsLoss(logit_k, target_k)
```

Esta formulacion conserva la naturaleza discreta de `level`, pero aprovecha la
ordenacion entre clases. Un error en `level=6` afecta a varios umbrales
adyacentes, no a una clase aislada.

## Correccion por densidad

Para evitar que los niveles raros sigan dominados por las clases frecuentes, la
perdida debe ponderarse por rareza. En vez de usar conteos crudos, conviene usar
una densidad suavizada de etiquetas:

```text
smoothed_density(level) = kernel_smoothing(train_level_histogram)
weight(level) = 1 / smoothed_density(level)
```

Despues, los pesos deben normalizarse y caparse para evitar que `level=8` domine
todo el entrenamiento:

```text
weight = min(weight, max_weight)
```

Configuracion inicial recomendada:

```text
max_weight = 8.0 o 12.0
kernel = triangular o gaussian pequeno sobre niveles enteros
```

Esta decision conecta con la literatura de imbalanced regression, especialmente
con label distribution smoothing y density-based weighting. En este proyecto se
usaria como una adaptacion discreta y ordinal, no como una regresion continua
generica.

Referencias utiles:

- Yang et al., 2021, "Delving into Deep Imbalanced Regression"
  - https://proceedings.mlr.press/v139/yang21m/yang21m.pdf
- Steininger et al., 2021, "Density-based weighting for imbalanced regression"
  - https://link.springer.com/article/10.1007/s10994-021-06023-5
- Vargas et al., 2020, "Cumulative link models for deep ordinal classification"
  - https://www.sciencedirect.com/science/article/abs/pii/S0925231220303805

## Regresion auxiliar opcional

Puede añadirse una perdida auxiliar sobre `z`, pero no deberia sustituir a la
perdida ordinal:

```text
loss = ordinal_loss + alpha * density_weighted_huber(z, level)
```

con:

```text
alpha = 0.1
```

Esta parte serviria como regularizador de escala. La prediccion final seguiria
dependiendo de los umbrales aprendidos, no de redondear `z`.

## Relacion con BatchNorm y LayerNorm

No se recomienda introducir BatchNorm como primera opcion. El entrenamiento
actual usa `batch_size=1` con gradient accumulation, y BatchNorm calcula
estadisticas sobre el batch real, no sobre el batch acumulado. En este contexto,
sus estadisticas pueden ser ruidosas y depender mucho del numero de lineas
supervisadas en cada muestra.

La normalizacion preferida para la cabeza es `LayerNorm`, porque opera por
ejemplo y encaja mejor con hidden states procedentes de un transformer.

## Sobre aumentar el modelo base

No se recomienda aumentar el modelo base en el mismo experimento. El siguiente
run debe aislar una pregunta:

```text
Puede una cabeza ordinal con umbrales aprendidos corregir el colapso de niveles
profundos manteniendo el mismo backbone?
```

Si se cambia a la vez:

- la cabeza;
- la perdida;
- los umbrales;
- y el tamano del modelo base;

entonces cualquier mejora seria dificil de atribuir. Para la memoria, esa
interpretabilidad importa mas que exprimir rendimiento en un solo run.

La secuencia recomendada es:

1. `two_head_ordinal_density_v2`
   - mismo Qwen2.5-7B 4bit;
   - misma LoRA;
   - misma alineacion `record_prefix_state`;
   - nueva cabeza ordinal;
   - perdida ponderada por densidad.
2. `two_head_ordinal_density_line_prefix_v2`
   - mismo diseño, pero cambiando la alineacion a `line_prefix_state`.
3. Solo despues, considerar un modelo base mayor o menos cuantizado.

## Implementacion actual

La primera implementacion vive en:

- `scripts/train_kubernetes_two_head_ordinal_sft.py`
- `tests/test_kubernetes_two_head_ordinal_sft_trainer.py`

Mantiene el backbone Qwen2.5-7B 4bit y la misma LoRA sobre
`q_proj,k_proj,v_proj,o_proj`. Esta version no aumenta el modelo base, no usa
BatchNorm y no incorpora Huber auxiliar. El objetivo es aislar el efecto de la
cabeza ordinal con umbrales aprendidos y perdida ponderada por densidad.

Comando recomendado para el primer run oficial:

```powershell
uv run python scripts\train_kubernetes_two_head_ordinal_sft.py `
  --output-dir results\two_head_ordinal_sft_kubernetes_v1 `
  --run-id two-head-ordinal-density-v2-20260519 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --epochs 3 `
  --checkpoint-steps 8 `
  --checkpoint-keep-last 0 `
  --eval-checkpoint-steps 32 `
  --eval-max-samples 10 `
  --eval-sample-strategy random `
  --validation-log-every 1 `
  --oom-recovery skip_batch `
  --max-oom-skips 0 `
  --wandb-mode online `
  --wandb-log-artifacts
```

Nota posterior al primer run oficial: la version inicial dejo
`--threshold-learning-rate-multiplier` en `1.0`, es decir, los parametros
`raw_tau0` y `raw_deltas` usaban el mismo learning rate que el resto de la
cabeza ordinal. En la practica, los umbrales apenas se movieron: la proyeccion
`hidden -> z` absorbio casi todo el ajuste. Para repetir el experimento con
aprendizaje real de umbrales, el comando debe anadir un multiplicador explicito,
por ejemplo:

```powershell
  --threshold-learning-rate-multiplier 25
```

Este multiplicador se aplica solo a `raw_tau0` y `raw_deltas`, con
`weight_decay=0.0`. El resto de la LoRA y del MLP ordinal mantiene
`--learning-rate`. Si se usa un valor distinto de `1.0`, queda registrado en la
`resume_signature`, en `config.json` y en W&B como
`train/threshold/learning_rate`.

La implementacion tambien permite separar el learning rate del MLP ordinal que
proyecta hidden states a `z`:

```powershell
  --ordinal-mlp-learning-rate-multiplier 3
```

Con esto quedan tres grupos conceptuales:

```text
base_lora:          --learning-rate
ordinal_mlp:        --learning-rate * --ordinal-mlp-learning-rate-multiplier
ordinal_thresholds: --learning-rate * --threshold-learning-rate-multiplier
```

El objetivo no es mezclar todos los cambios en un unico run, sino poder separar
dos hipotesis distintas: que los cortes `tau` se mueven demasiado poco y que la
escala latente `z` producida por el MLP no alcanza regiones profundas. La
primera se estudia subiendo `threshold-learning-rate-multiplier`; la segunda se
estudia subiendo `ordinal-mlp-learning-rate-multiplier`.

Cuando cualquiera de estos multiplicadores es distinto de `1.0`, el
optimizador crea grupos nombrados y el trainer registra en `train_log.jsonl` y
W&B:

```text
train/learning_rate
train/ordinal_mlp/learning_rate
train/threshold/learning_rate
```

El multiplicador del MLP tambien queda en `resume_signature` cuando no es
`1.0`, para evitar reanudar un run con una firma experimental distinta.

Artefactos principales por run:

- `config.json`
- `state.json`
- `train_log.jsonl`
- `threshold_history.jsonl`
- `gradient_diagnostics.jsonl`
- `validation_progress.jsonl`
- `validation_sample_metrics.jsonl`
- `metrics.json`, solo al completar correctamente
- `checkpoints/checkpoint-step-*/adapter/`
- `checkpoints/checkpoint-step-*/tokenizer/`
- `checkpoints/checkpoint-step-*/ordinal_level_head.pt`
- `checkpoints/checkpoint-step-*/training_state.pt`

`training_state.pt` guarda optimizador, scheduler y estado resumible del run.
La `resume_signature` incluye los nuevos hiperparametros ordinales: dimensiones
y dropout de la cabeza, `lambda_level`, kernel de densidad, radio, peso maximo,
numero de clases, multiplicadores de learning rate de MLP ordinal y umbrales si
no son `1.0`, politica de alineacion y contrato de target.

W&B usa logging curado. El dashboard principal recibe perdidas, learning rate,
umbrales `tau_0..tau_7`, gaps entre umbrales, estadisticos de `z` y metricas
estructurales de validacion. BLEU, ROUGE y perplexity quedan como artefactos
locales y no se suben al panel principal.

## Diagnostico de gradientes y desplazamiento efectivo

Despues del primer run y de la variante con `threshold_learning_rate_multiplier`,
la siguiente pregunta razonable no es elegir otro multiplicador a ciegas, sino
medir que parte del sistema esta absorbiendo el aprendizaje ordinal.

La comparacion ingenua seria mirar solo el gradiente bruto de los umbrales y
compararlo con el gradiente del MLP que produce `z`. Esa comparacion es util,
pero incompleta por dos razones:

1. el MLP tiene muchos mas parametros que los ocho cortes globales, por lo que
   su norma total tiende a ser mayor solo por dimensionalidad;
2. AdamW no aplica directamente el gradiente bruto, sino una actualizacion
   reescalada por sus momentos internos, el learning rate del grupo y el
   scheduler.

Por eso el trainer registra tres familias de senales:

```text
gradiente bruto:
  train/grad_norm/ordinal_mlp
  train/grad_rms/ordinal_mlp
  train/grad_norm/threshold_raw
  train/grad_rms/threshold_raw

actualizacion efectiva de parametros:
  train/update_norm/ordinal_mlp
  train/update_rms/ordinal_mlp
  train/update_norm/threshold_raw
  train/update_rms/threshold_raw

desplazamiento funcional en el espacio ordinal:
  train/effective_shift/z_mean_abs
  train/effective_shift/z_rms
  train/effective_shift/tau_mean_abs
  train/effective_shift/tau_rms
  train/effective_shift/z_to_tau_mean_ratio
```

Las metricas RMS son importantes porque dividen la norma por la raiz del numero
de elementos. Asi se evita concluir que el MLP "recibe mas gradiente" solo
porque tiene mas parametros.

La metrica mas interpretable es:

```text
train/effective_shift/z_to_tau_mean_ratio =
  mean_abs_delta_z_on_same_hidden_batch / mean_abs_delta_tau
```

Para calcularla, el trainer toma el mismo hidden state usado por la cabeza
ordinal antes del `optimizer.step()`, evalua el MLP ordinal sin dropout,
ejecuta el step, vuelve a evaluar el mismo hidden state y compara cuanto cambio
`z`. En paralelo compara cuanto se movieron los thresholds ya transformados
`tau_0..tau_7`, no solo los parametros raw `raw_tau0` y `raw_deltas`.

Esta distincion importa: la decision ordinal depende de `z > tau_k`. Por tanto,
lo que queremos equilibrar no es necesariamente la norma de gradiente en
parametros, sino la escala del movimiento en el espacio donde se decide el
nivel. Si `z` se desplaza unas veinte veces mas que `tau` por step, entonces un
multiplicador de learning rate de thresholds en el orden de `20x` o `50x` tiene
una justificacion empirica. Si el ratio cae cerca de `1`, seguir aumentando el
learning rate de thresholds dejaria de ser una correccion de escala y pasaria a
ser una nueva intervencion experimental.

El diagnostico se guarda en `gradient_diagnostics.jsonl` y se sube a W&B como
logging curado. No cambia la perdida, la arquitectura ni la politica de
checkpointing; solo anade observabilidad para decidir el siguiente
multiplicador de forma trazable.

## Criterios de exito

El experimento no debe evaluarse solo por `yaml_parse_success_rate`. Debe
registrar tambien:

- distribucion de niveles predichos;
- recall por nivel;
- recall exacto en `level=5..8`;
- recall off-by-one en `level=5..8`;
- `predicted_max_level`;
- matriz de confusion;
- parseabilidad global;
- parseabilidad en ejemplos con `target_max_level >= 5`;
- comparacion parseable-only frente al primer `two_head_sft`.

Criterio minimo para considerar que la idea funciono:

```text
predicted_count(level 5..8) > 0
deep_level_exact_recall > 0
yaml_parse_success_rate >= 0.4857
```

Criterio fuerte:

```text
deep_level_exact_recall mejora sin degradar recall de level=4 por debajo de 0.70
yaml_parse_success_rate supera claramente 0.50
```

## Decision recomendada

La siguiente version deberia priorizar umbrales aprendidos antes que aumentar el
tamano del modelo. Si esta version no consigue emitir niveles profundos, entonces
si tendra sentido probar una alineacion distinta o un backbone mas capaz. Pero
primero conviene comprobar si el fallo esta en la formulacion de la cabeza, que
es el cambio mas pequeno y mas alineado con la hipotesis estructural del
proyecto.
