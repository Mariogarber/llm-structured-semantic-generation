# Latent Level Probe v1

## Indice rapido

- [Proposito](#proposito)
- [Relacion con la comparacion SFT](#relacion-con-la-comparacion-sft)
- [Superficie de entrada y control de leakage](#superficie-de-entrada-y-control-de-leakage)
- [Estrategias de lectura del hidden state](#estrategias-de-lectura-del-hidden-state)
- [Probes y metricas](#probes-y-metricas)
- [Ejecucion](#ejecucion)
- [Guia de artefactos del run](#guia-de-artefactos-del-run)
- [Politica stop/resume](#politica-stopresume)
- [Integracion con Weights & Biases](#integracion-con-weights--biases)
- [Uso esperado en la memoria](#uso-esperado-en-la-memoria)
- [Informe de resultados](#informe-de-resultados)

## Proposito

Esta nota documenta la fase actual del proyecto: el analisis de si las
representaciones internas del modelo contienen una senal util para predecir el
`level` de cada linea YAML antes de entrenar el modelo `two_head_sft`.

La motivacion nace directamente de la pregunta central de la memoria. Un
manifiesto Kubernetes se escribe como texto YAML, pero su correccion no depende
solo de que aparezcan las palabras adecuadas. Tambien depende de que cada linea
ocupe una posicion correcta dentro de una jerarquia. En la representacion
actual del repositorio, esa jerarquia se expresa mediante la variable `level`.
Por tanto, antes de anadir una cabeza explicita para predecir `level`, conviene
preguntar si el modelo ya organiza internamente alguna informacion relacionada
con esa variable.

Este analisis no sustituye a `two_head_sft`. Es una fase diagnostica previa. Su
objetivo es medir si distintos estados ocultos del modelo permiten recuperar el
nivel jerarquico de una linea YAML mediante clasificadores ligeros. Si la senal
es facil de recuperar, la cabeza estructural puede interpretarse como una forma
de leer y supervisar una informacion que ya esta parcialmente presente. Si la
senal solo aparece con un MLP no lineal, entonces la informacion existe, pero no
esta organizada de forma simple. Si los probes fallan, el resultado tambien es
informativo: indicaria que `two_head_sft` no solo debe leer la jerarquia, sino
ayudar a formar una representacion mas estructural durante el entrenamiento.

## Relacion con la comparacion SFT

La comparacion principal del proyecto sigue siendo:

```text
serialized_sft vs two_head_sft
```

`serialized_sft` aprende a emitir `level` como texto ordinario dentro de
`blocks_tsv_v1`. Esta rama sirve como control: la jerarquia esta presente en la
superficie de salida, pero no tiene un mecanismo supervisado independiente.

`two_head_sft`, en cambio, debe separar dos predicciones coordinadas:

- el contenido de cada linea, producido por la cabeza autoregresiva del modelo;
- el `level` de cada linea, producido por una cabeza estructural adicional.

El latent level probe se situa justo entre esas dos decisiones. No entrena aun
la arquitectura final, pero examina que parte de la representacion interna puede
servir como entrada para esa futura cabeza de nivel. Por eso compara varias
formas de leer el hidden state asociado a una linea:

- `line_mean`: media de los tokens de `line_text`;
- `line_first_token`: hidden state del primer token de la linea;
- `line_last_token`: hidden state del ultimo token de la linea;
- `line_prefix_state`: hidden state inmediatamente anterior al contenido de la
  linea, despues de `document_index` y `line_index`;
- `record_prefix_state`: hidden state inmediatamente anterior al registro de la
  linea, antes de `document_index`, `line_index` y `line_text`.

Estas alternativas no son equivalentes. `line_mean` pregunta si el contenido
completo de la linea contiene pistas suficientes sobre su nivel.
`line_first_token` y `line_last_token` prueban lecturas mas simples y baratas.
`line_prefix_state` se acerca a la pregunta de si el modelo puede anticipar la
altura de la linea cuando ya conoce su posicion textual (`document_index` y
`line_index`). `record_prefix_state` es la lectura mas estricta, porque pregunta
por la senal disponible antes de cualquier informacion de la linea actual.

## Superficie de entrada y control de leakage

El probe parte de los ficheros SFT de Kubernetes v1:

```text
data/processed/kubernetes_v1/sft/train.jsonl
data/processed/kubernetes_v1/sft/validation.jsonl
data/processed/kubernetes_v1/sft/test.jsonl
```

Por defecto, el workflow carga solo `train.jsonl` y `validation.jsonl`.
`test.jsonl` queda reservado para el final del proyecto, cuando ya existan
modelos candidatos y decisiones de diseno cerradas. Para incluirlo es necesario
pasar explicitamente `--include-test`.

Cada fila contiene un target `blocks_tsv_v1`, donde aparecen
`document_index`, `line_index`, `level` y `line_text`. Sin embargo, usar ese
target directamente para extraer hidden states seria incorrecto, porque el
modelo veria el valor de `level` como texto. El clasificador podria aprender a
leer la respuesta desde la entrada, no a recuperarla desde una representacion
estructural.

Para evitar ese leakage, el workflow transforma cada target en una superficie
solo de contenido:

```text
<content_blocks>
document_index    line_index    line_text
...
</content_blocks>
```

El `level` se elimina de la entrada del modelo y se conserva solo como etiqueta
supervisada del probe. Esta separacion es una condicion metodologica esencial:
el probe debe medir informacion recuperable en los hidden states, no comprobar
si el modelo puede copiar una columna visible.

## Estrategias de lectura del hidden state

Aunque el workflow pasa al modelo la secuencia completa
`prompt + content_blocks`, el modelo causal no puede atender a tokens futuros
desde una posicion concreta. La mascara autoregresiva hace que el hidden state
de un token dependa solo del prompt, de las lineas anteriores y de los tokens ya
vistos de la linea actual. Por eso, el probe no debe interpretarse como una
prediccion de estructura con acceso a toda la salida futura, sino como una forma
controlada de observar que informacion jerarquica aparece en distintos momentos
del procesamiento de una linea.

Las estrategias comparan puntos distintos de esa progresion:

- `record_prefix_state`: usa el hidden state inmediatamente anterior al registro
  de la linea. Es la lectura mas estricta, porque pregunta si el modelo puede
  anticipar el nivel desde el prompt y las lineas previas antes de observar
  `document_index`, `line_index` o `line_text` de la linea actual.
- `line_prefix_state`: usa el hidden state inmediatamente anterior al contenido
  de la linea. En esta posicion el modelo ya ha visto `document_index` y
  `line_index`, pero todavia no ha visto `line_text`.
- `line_first_token`: usa el hidden state del primer token de `line_text`.
  Permite ver si basta una pista inicial de la linea, junto con el contexto
  anterior, para recuperar el nivel.
- `line_last_token`: usa el hidden state del ultimo token de `line_text`.
  Esta estrategia se parece a un diseno donde la cabeza de `level` predice el
  nivel cuando la linea ya ha sido generada, usando el contexto anterior y todo
  el contenido de esa linea, pero no lineas futuras.
- `line_mean`: calcula la media de los hidden states de los tokens de
  `line_text`. Resume la representacion distribuida de la linea completa y
  suele ser una lectura estable para analisis, aunque no corresponde a un unico
  instante generativo.

La comparacion entre estas estrategias tiene una lectura directa para
`two_head_sft`. Si `record_prefix_state` funciona bien, hay evidencia de que el
modelo puede anticipar parte de la jerarquia antes de generar cualquier campo de
la linea. Si `line_prefix_state` mejora claramente sobre `record_prefix_state`,
la posicion de la linea aporta informacion util. Si `line_last_token` o
`line_mean` funcionan mucho mejor, la senal de `level` parece depender mas del
contenido ya producido. En ese caso, una cabeza estructural que lea el estado al
final de la linea seria metodologicamente mas defendible que una cabeza que
intente fijar el nivel antes de generar el texto.

## Probes y metricas

El workflow entrena y evalua cuatro familias de probes:

- `majority`: baseline que predice siempre el nivel mas frecuente;
- `previous_level`: baseline estructural que predice el nivel anterior dentro
  del mismo sample;
- `linear`: regresion logistica multinomial sobre el vector latente;
- `mlp`: clasificador pequeno no lineal.

Los dos primeros no usan realmente el contenido latente. Funcionan como puntos
de comparacion para no sobreinterpretar un resultado que podria explicarse por
la distribucion de niveles o por la continuidad local de la jerarquia YAML. Los
dos ultimos si prueban si el hidden state contiene informacion util.

Las metricas principales son:

- accuracy;
- balanced accuracy;
- macro-F1;
- weighted-F1;
- MAE de nivel;
- precision, recall y F1 por nivel;
- matriz de confusion.

La interpretacion debe ser prudente. Una accuracy alta no basta si solo mejora
en los niveles frecuentes. Por eso `macro-F1`, `balanced_accuracy` y la matriz
de confusion son especialmente importantes: permiten comprobar si el probe
aprende tambien niveles menos frecuentes o si se limita a repetir el patron
dominante de niveles superficiales.

## Ejecucion

El script principal es:

```text
scripts/run_kubernetes_latent_level_probe.py
```

Ejemplo de ejecucion completa con W&B en modo offline:

```powershell
uv run python scripts/run_kubernetes_latent_level_probe.py `
  --stage all `
  --run-id latent-level-probe-v1 `
  --batch-size 1 `
  --wandb-mode offline
```

Ejemplo reducido para comprobar que el pipeline funciona sin cargar el modelo:

```powershell
uv run python scripts/run_kubernetes_latent_level_probe.py `
  --stage all `
  --run-id latent-level-probe-smoke `
  --max-samples 4 `
  --batch-size 1 `
  --wandb-mode disabled `
  --dry-run
```

Cuando llegue la evaluacion final de candidatos, el split de test puede
activarse de forma explicita:

```powershell
uv run python scripts/run_kubernetes_latent_level_probe.py `
  --stage all `
  --run-id latent-level-probe-final-test `
  --include-test `
  --batch-size 1 `
  --wandb-mode offline
```

El modo `--dry-run` usa hidden states sinteticos deterministas. Sirve para
validar resume, artefactos, metricas y tests de integracion, pero no produce un
resultado cientifico sobre el modelo.

Las fases disponibles son:

- `extract`: extrae features de hidden states y las guarda localmente;
- `probe`: entrena probes usando features ya extraidas;
- `all`: ejecuta ambas fases.

Por defecto, el run escribe en:

```text
results/latent_level_probe_kubernetes_v1/<run-id>/
```

## Guia de artefactos del run

Un run del probe genera bastantes ficheros porque cruza varias estrategias de
extraccion con varios clasificadores. Por ejemplo, con la configuracion por
defecto se prueban cinco estrategias:

```text
record_prefix_state
line_prefix_state
line_first_token
line_last_token
line_mean
```

y cuatro probes:

```text
majority
previous_level
linear
mlp
```

Eso produce 20 combinaciones. Para cada combinacion se guardan metricas,
predicciones y, cuando es posible, una matriz de confusion. Por tanto, el numero
de archivos puede parecer alto aunque el experimento sea pequeno.

Los archivos principales son:

- `config.json`: configuracion del experimento. Incluye los ficheros de entrada,
  el modelo, el `run-id`, las estrategias, los probes, `max_samples`, el modo de
  W&B y la firma usada para validar resume.
- `state.json`: estado reconstruible del run. Indica que unidades ya se
  procesaron, que estrategias y probes estan completos y si el run termino.
- `chunks/`: carpeta con un fichero por unidad procesada
  (`sample_id::prompt_variant`). Es la fuente de verdad para reanudar. En
  general no hace falta abrirla a mano.
- `line_metadata.jsonl`: indice linea a linea. Cada fila corresponde a una
  linea YAML y guarda `sample_id`, `split`, `document_index`, `line_index`,
  `line_text` y el `level` correcto.
- `features_<strategy>.jsonl`: vectores latentes extraidos con una estrategia
  concreta, por ejemplo `features_line_mean.jsonl`. Son artefactos grandes y
  normalmente no se inspeccionan manualmente.
- `probe_metrics_<probe-id>.json`: metricas de una combinacion concreta de
  estrategia y probe, por ejemplo `probe_metrics_line_mean__linear.json`.
  Estos son los archivos mas utiles para comparar resultados.
- `probe_predictions_<split>_<probe-id>.jsonl`: predicciones linea a linea para
  un split evaluado. Sirven para analisis de errores.
- `confusion_matrix_<split>_<probe-id>.png`: matriz de confusion visual para una
  combinacion de estrategia y probe.
- `metrics.json`: resumen final del run. Se escribe solo cuando el run termina
  correctamente.

Para una revision rapida del experimento, el orden recomendado es:

1. Abrir `metrics.json` para comprobar que el run completo termino.
2. Comparar varios `probe_metrics_<probe-id>.json`, especialmente los probes
   `linear` y `mlp`.
3. Revisar las matrices `confusion_matrix_*.png` para ver que niveles se
   confunden.
4. Usar `probe_predictions_*.jsonl` solo si se quiere analizar ejemplos
   concretos.
5. Consultar `line_metadata.jsonl` para volver desde una prediccion a la linea
   YAML original.

Los ficheros `features_*.jsonl` y `chunks/` son importantes para
reproducibilidad y resume, pero no suelen ser el primer sitio donde mirar
resultados.

## Politica stop/resume

Esta fase sigue la politica general del repositorio: cualquier ejecucion LLM
debe ser incremental y reanudable. La fuente de verdad para reanudar no es W&B
ni memoria de proceso, sino los artefactos locales ya escritos.

Cada muestra procesada se guarda como un chunk atomico bajo:

```text
chunks/
```

El script escribe primero un fichero temporal `*.tmp` y solo lo renombra cuando
el chunk esta completo. Asi, si el proceso se detiene a mitad de una muestra o
de un batch, el siguiente arranque no interpreta un resultado parcial como
trabajo terminado.

Los artefactos principales son:

- `config.json`: configuracion del run y firma de resume;
- `state.json`: estado reconstruido desde los chunks locales;
- `line_metadata.jsonl`: metadatos de las lineas extraidas;
- `features_<strategy>.jsonl`: features por estrategia;
- `probe_predictions_<split>_<probe-id>.jsonl`: predicciones del probe;
- `probe_metrics_<probe-id>.json`: metricas por probe;
- `metrics.json`: resumen final, escrito solo al completar correctamente.

Al reiniciar un run, el script compara la nueva configuracion con la firma
guardada en `config.json`. Si cambian parametros que afectan a la identidad del
experimento, como el modelo, los ficheros de entrada, `max_samples`, la capa
oculta o las estrategias de features, el resume se rechaza. Esta restriccion
evita mezclar features incompatibles bajo el mismo `run-id`.

## Integracion con Weights & Biases

W&B se usa como capa de observabilidad y comparacion, no como fuente de verdad
para reanudar. Cuando esta habilitado, el run se inicializa con:

```text
id = <run-id>
resume = "allow"
```

Durante la extraccion se registran contadores de progreso, como muestras
procesadas, muestras restantes y batch actual. Durante la fase de probes se
registran metricas por combinacion de estrategia y clasificador. Al final se
sube un artifact con la configuracion, estado, features agregadas, predicciones
y metricas.

Esto permite comparar experimentos del MLP, estrategias de entrada y futuros
checkpoints SFT dentro del mismo proyecto W&B, manteniendo al mismo tiempo una
ejecucion local robusta ante interrupciones.

## Uso esperado en la memoria

El resultado de esta fase debe presentarse como un analisis de sondeo
estructural. No prueba por si solo que `two_head_sft` vaya a mejorar la
generacion final, porque un probe entrenado sobre hidden states fijos no es lo
mismo que una arquitectura entrenada end to end. Sin embargo, permite responder
una pregunta metodologica importante: hasta que punto la jerarquia
YAML ya esta representada en el espacio interno del modelo antes de introducir
una cabeza explicita.

La lectura recomendada es:

- si el probe lineal funciona bien, hay evidencia de que el `level` es
  recuperable de forma simple desde la representacion;
- si solo funciona bien el MLP, hay senal, pero con una organizacion no lineal;
- si ambos fallan, la motivacion de `two_head_sft` se desplaza hacia formar una
  representacion estructural nueva durante el SFT;
- si `line_prefix_state` compite bien con features que ven el contenido de la
  linea, la cabeza de `level` podria apoyarse en el contexto generativo previo;
- si `line_mean` domina claramente, el contenido de la linea ayuda mucho a
  inferir su posicion jerarquica, pero eso no significa que la jerarquia este
  anticipada antes de generar la linea.

En todos los casos, este analisis debe conectarse despues con las metricas
finales de `two_head_sft`: `average_level_exact_match_rate`,
`yaml_parse_success_rate`, errores de reconstruccion y metricas Kubernetes
aproximadas. El probe es una herramienta para entender la representacion; la
comparacion supervisada sigue siendo la prueba principal de utilidad practica.

## Informe de resultados

El primer run completo del probe sobre Kubernetes v1 esta documentado en:

```text
docs/LATENT_LEVEL_PROBE_RESULTS_2026-05-13.md
```

Ese informe recoge la configuracion exacta del experimento, los resultados de
las 20 combinaciones de estrategia y probe, la arquitectura del MLP usado y un
analisis de errores centrado en la comparacion entre `record_prefix_state` y
`line_prefix_state`.
