# Auditoria de parseabilidad two_head_sft - 2026-05-19

## Objetivo

Este documento audita los fallos de parseabilidad del run
`two-head-sft-v1-20260516` sin cambiar codigo ni recomputar las metricas
oficiales del run. El objetivo es identificar patrones en las salidas no
parseables de `two_head_sft` y separar problemas de:

- contrato de bloques;
- reconstruccion YAML;
- prediccion de `level`;
- generacion de contenido;
- superficie `content_blocks_v1`.

No se proponen aqui correcciones definitivas. Esta auditoria sirve como base
para decidir el siguiente experimento o la siguiente revision del evaluador.

## Artefactos inspeccionados

```text
results/two_head_sft_kubernetes_v1/two-head-sft-v1-20260516/validation_predictions.jsonl
results/two_head_sft_kubernetes_v1/two-head-sft-v1-20260516/metrics.json
data/processed/kubernetes_v1/sft/validation.jsonl
```

La validacion final contiene:

```text
total rows = 70
YAML parseable = 34
YAML / parser-facing failures = 36
```

## Taxonomia de fallos

Los `36` fallos se dividen asi:

| Tipo de fallo | Casos |
| --- | ---: |
| `yaml_parse:ParserError` | 20 |
| `yaml_parse:ScannerError` | 11 |
| `block_contract:line_index_sequence` | 5 |

Los mensajes PyYAML mas frecuentes son:

| Mensaje | Casos | Lectura |
| --- | ---: | --- |
| `expected <block end>, but found '-'` mientras parsea un mapping | 16 | una lista aparece a un nivel incompatible con el mapping actual |
| `mapping values are not allowed here` | 9 | una clave aparece en un contexto donde YAML esperaba otra cosa |
| `expected <block end>, but found '?'` mientras parsea una collection | 3 | mezcla incoherente de mapping/lista |
| `could not find expected ':'` | 2 | fragmentos textuales no validos como clave YAML |
| `expected <block end>, but found '<block mapping start>'` | 1 | mapping iniciado en profundidad incoherente |

Ademas, 5 ejemplos no llegan a reconstruccion YAML por contrato de bloques:
`line_index` deja de ser consecutivo dentro del documento.

## Concentracion por tipo de recurso

| Kind principal del target | Total validation | Fallos | Tasa de fallo |
| --- | ---: | ---: | ---: |
| `DaemonSet` | 54 | 30 | 0.556 |
| `ServiceAccount` | 4 | 3 | 0.750 |
| `ConfigMap` | 4 | 2 | 0.500 |
| `PersistentVolume` | 8 | 1 | 0.125 |

La mayor parte absoluta de los fallos esta en `DaemonSet`. Esto no significa
por si solo que el modelo falle especificamente por el `kind`, porque el split
de validacion tambien esta muy cargado de `DaemonSet`. Pero si indica que los
fallos aparecen sobre todo en manifiestos con pod template, contenedores,
comandos, volumenes o listas internas.

## Patron 1: errores de listas y mappings anidados

El patron dominante es una incompatibilidad entre listas YAML y niveles
predichos. Los ejemplos fallidos suelen tener estructuras como:

```yaml
containers:
- command:
  - sh
  - -c
    - while true; do echo ...; done
```

o:

```yaml
volumeMounts:
  - mountPath: /var/log/nginx
  name: nginx-logs
  volumes:
- name: nginx-logs
    hostPath:
```

En ambos casos, el contenido generado es reconocible, pero la profundidad de
alguna linea rompe la gramatica YAML. El error no es necesariamente que el
modelo ignore completamente el recurso; con frecuencia produce claves
esperables, pero una o varias lineas de lista quedan en el nivel equivocado.

Ejemplo `q30::question`:

```yaml
containers:
- command:
  - sh
  - -c
- while true; do echo $GREETING_MESSAGE; sleep 10; done
  image: busybox
  name: ds-env-container
env:
- name: GREETING_MESSAGE
```

El item del comando queda alineado como si fuera otro item de una lista de nivel
superior, no como continuacion de `command`.

Ejemplo `q24::question`:

```yaml
volumeMounts:
  - mountPath: /var/log/nginx
  name: nginx-log
  volumes:
- hostPath:
    path: /var/log/nginx
```

Aqui `name` y `volumes` quedan en una posicion incompatible con la lista de
`volumeMounts`, y la lista de `volumes` queda demasiado arriba.

## Patron 2: la cabeza de nivel no predice profundidades mayores que 4

En toda la validacion final, `two_head_sft` no predijo ningun `level > 4`.

Distribucion agregada:

```text
gold levels:      0..8
predicted levels: 0..4
```

Conteos globales:

| Level | Gold lines | Predicted lines |
| ---: | ---: | ---: |
| 0 | 324 | 300 |
| 1 | 294 | 278 |
| 2 | 230 | 304 |
| 3 | 328 | 431 |
| 4 | 450 | 411 |
| 5 | 114 | 0 |
| 6 | 44 | 0 |
| 7 | 4 | 0 |
| 8 | 12 | 0 |

Esto es importante porque muchas estructuras validas del target usan niveles
5, 6, 7 u 8. Cuando esas lineas profundas aparecen en listas anidadas, el
modelo tiende a comprimirlas hacia niveles 3 o 4. Esa compresion puede producir
YAML todavia parseable en algunos casos, pero en otros rompe directamente la
gramatica.

La diferencia se observa tambien en los fallos:

```text
target_max_level medio en fallos = 5.03
pred_max_level medio en fallos   = 3.89
```

No todos los ejemplos con `gold level > 4` fallan, pero la ausencia total de
predicciones por encima de 4 es una senal fuerte de infraprediccion de
profundidad.

## Patron 3: comandos shell fragmentados como lineas YAML

Varios fallos aparecen en comandos de contenedor:

```yaml
- /bin/sh
- -c
- while true; do ...
```

o en salidas generadas como:

```yaml
- >-
tar -xzf ...
- && tar -xzf ...
- && nginx -g daemon off;
```

El problema aqui mezcla contenido y estructura:

- el contenido se parece a un comando shell razonable;
- pero la representacion por lineas convierte fragmentos del comando en lineas
  YAML independientes;
- si la cabeza de nivel no mantiene esas lineas dentro de la lista o del bloque
  escalar correcto, PyYAML interpreta texto shell como clave o como item de
  lista mal situado.

Esto explica errores como:

```text
could not find expected ':'
mapping values are not allowed here
```

## Patron 4: regresion o reinicio de line_index

Cinco fallos no son errores YAML propiamente dichos, sino errores del contrato
de bloques:

| Unit id | Error |
| --- | --- |
| `q21::question` | `unexpected_line_index` desde el bloque 22 |
| `q21::question_simplified` | `unexpected_line_index` desde el bloque 22 |
| `q43::question` | `unexpected_line_index` desde el bloque 24 |
| `q64::question_simplified` | `unexpected_line_index` desde el bloque 30 |
| `q66::question` | `unexpected_line_index` desde el bloque 19 |

En estos casos el modelo genera una secuencia de `line_index` que vuelve hacia
atras o deja de ser consecutiva. El parser rechaza la salida antes de intentar
parsear YAML.

Este fallo afecta a la capa de superficie `content_blocks_v1`: aunque no se
genere `level` en texto, el modelo sigue teniendo que generar indices de linea
estables.

## Patron 5: filas visibles malformadas en content_blocks

Se detectan `20` filas visibles malformadas en `raw_model_output`. No todas
producen fallo final, pero aparecen en casos relevantes:

| Unit id | Filas malformadas | Ejemplo |
| --- | ---: | --- |
| `q30::question_simplified` | 1 | `20\tvalue: Hello from the environment variable!` |
| `q31::question` | 6 | `21\t- while true; do cat /config/config-data.txt; sleep 10; done` |
| `q32::question_simplified` | 6 | `22\timage: busybox` |
| `q43::question_simplified` | 1 | `---` |

Estas filas tienen menos de tres campos TSV y por tanto no cumplen
`content_blocks_v1`.

Hay un detalle de evaluacion que conviene registrar sobre el run original: el
extractor usado durante esa evaluacion era demasiado tolerante. Si encontraba
una fila malformada despues de haber leido filas validas, cortaba la lectura en
vez de marcar necesariamente toda la salida como error de superficie. Por eso
`structured_output_parse_success_rate` pudo quedar en `1.0` aunque existieran
filas visibles malformadas en el output crudo.

Esto no cambia el hecho de que el sistema final falla, pero si sugiere que la
metrica `structured_output_parse_success_rate` del primer run era demasiado
optimista para esta arquitectura.

### Correccion aplicada despues de la auditoria

Despues de esta auditoria se implemento una normalizacion superficial acotada
en `scripts/train_kubernetes_two_head_sft.py`:

1. Si una fila de `<content_blocks>` tiene exactamente dos campos y ya existe
   un documento activo, se interpreta como:

   ```text
   line_index    line_text
   ```

   y se repara como:

   ```text
   previous_document_index    line_index    line_text
   ```

   Este caso cubre salidas como:

   ```text
   20    value: Hello from the environment variable!
   ```

   que se normalizan como:

   ```text
   0     20     value: Hello from the environment variable!
   ```

2. Despues de leer las filas, `line_index` se renumera por orden dentro de cada
   `document_index`. Esto elimina fallos donde el modelo conserva el orden de
   generacion pero emite indices repetidos, saltados o regresivos.

3. Cualquier otro caso malformado deja de cortarse silenciosamente y pasa a ser
   un error explicito de superficie.

Esta correccion no modifica el modelo ni la cabeza de nivel. Tampoco repara
listas YAML, niveles mal predichos ni contenido semantico. Es una normalizacion
del contrato de salida: `document_index` y `line_index` son variables de
posicion, y la posicion puede recuperarse de forma determinista cuando el orden
generado es claro.

Para la memoria, esta diferencia es importante: los resultados originales deben
presentarse como la evaluacion directa del primer run; la evaluacion con esta
normalizacion, si se recomputa, debe presentarse como una variante
postprocesada o parser-facing normalization.

## Patron 6: los fallos ponen a cero metricas posteriores

Cuando una prediccion no llega a YAML parseable o falla el contrato de bloques,
`_build_failed_evaluation` asigna cero a muchas metricas:

- `line_text_f1`;
- `content_exact_match_rate`;
- `level_exact_match_rate`;
- `semantic_key_f1`;
- `prompt_requirement_f1`;
- `kind_sequence_match_rate`.

Esto explica por que las metricas globales de `two_head_sft` se desploman. No
significa necesariamente que el contenido sea completamente inutil en los 36
casos fallidos; significa que el evaluador no calcula esas metricas una vez que
la reconstruccion falla.

La excepcion es `average_level_mae`: al ser opcional, se calcula solo sobre los
casos donde esta disponible. Por eso el `level_mae` final de `two_head_sft`
refleja esencialmente los casos parseables, no todos los ejemplos.

## Comparacion parseable vs no parseable

Promedios observados:

| Rasgo | Fallos | Parseables |
| --- | ---: | ---: |
| `target_lines` | 26.36 | 25.03 |
| `pred_lines` | 27.69 | 21.38 |
| `line_delta` | +1.33 | -3.65 |
| `target_max_level` | 5.03 | 4.85 |
| `pred_max_level` | 3.89 | 3.38 |
| `target_seq` | 4.67 | 4.77 |
| `pred_seq` | 4.83 | 3.71 |
| `has_volume_rate` | 0.333 | 0.059 |
| `has_container_rate` | 0.972 | 0.794 |

Los fallos tienen una presencia mucho mayor de volumenes o mounts. Tambien
tienden a generar mas lineas que el target, mientras que los casos parseables
tienden a generar menos lineas que el target. Esto sugiere dos modos distintos:

- salidas parseables pero incompletas o simplificadas;
- salidas mas largas que intentan cubrir mas estructura, pero rompen listas o
  niveles.

## Lectura preliminar

La hipotesis inicial queda reforzada: el bajo resultado global de
`two_head_sft` esta dominado por fallos de parseabilidad, no por una degradacion
uniforme de todos los aspectos de calidad.

Los patrones mas probables son:

1. La cabeza de nivel comprime la profundidad y no usa clases `5..8`.
2. Las listas YAML anidadas son el punto de ruptura principal.
3. Los comandos shell y listas de argumentos son especialmente fragiles.
4. El modelo a veces pierde el contrato de `line_index`.
5. La superficie `content_blocks_v1` tiene algunas filas malformadas que la
   extraccion actual no penaliza con suficiente claridad.

La conclusion operacional es que `two_head_sft` no parece fallar porque sea
globalmente incapaz de generar contenido Kubernetes razonable. Falla porque una
porcion grande de sus salidas no atraviesa el cuello de botella parser-facing.
Cuando ese cuello se supera, las metricas condicionadas quedan mucho mas cerca
de `serialized_sft`.

## Posibles siguientes comprobaciones

Antes de lanzar otro entrenamiento, conviene comprobar:

1. Si `level_head` esta recibiendo suficientes ejemplos efectivos de clases
   `5..8` durante entrenamiento o si hay un sesgo fuerte hacia niveles `0..4`.
2. Si la alineacion `record_prefix_state` dificulta predecir niveles profundos
   porque la cabeza no ve aun suficiente contenido de la linea actual.
3. Si conviene penalizar distancia ordinal de nivel ademas de cross-entropy.
4. Si `content_blocks_v1` deberia eliminar `line_index` de la superficie y
   derivarlo por orden de generacion, reduciendo un modo de fallo.
5. Si la evaluacion debe marcar cualquier fila visible malformada como fallo
   estricto de `structured_output_parse_success_rate`.
6. Si se necesita una validacion adicional que evalua contenido y niveles aun
   cuando YAML no parsea, para no poner automaticamente a cero todas las
   metricas posteriores.

## Separacion entre fallos corregibles y fallos de modelo

Para evitar mezclar conclusiones en la memoria, conviene separar los problemas
observados en tres grupos.

### Corregibles sin reentrenar

- Filas de dos campos donde falta `document_index`, siempre que ya haya un
  documento activo.
- `line_index` no consecutivo, si se acepta que el orden autoregresivo define
  la posicion real de la linea.
- Metricas de superficie demasiado tolerantes, como el caso donde una fila
  malformada se ignoraba despues de leer filas validas.
- Warnings de logging o serializacion que no afectan al modelo.

Estos problemas pertenecen al contrato de salida o a la evaluacion, no a la
capacidad estructural aprendida.

### Parcialmente corregibles, pero ya como repair

- Listas YAML mal niveladas.
- Fragmentos de comandos shell situados fuera de `command` o `args`.
- Campos como `name`, `volumeMounts` o `volumes` desplazados a un nivel
  incompatible.

Estos casos podrian arreglarse con heuristicas, pero eso ya seria una capa de
reparacion estructural. Si se implementa, debe evaluarse como variante separada,
no mezclarse con el resultado directo de `two_head_sft`.

### No corregibles limpiamente sin tocar el modelo

- Ausencia total de predicciones `level > 4`.
- Infraprediccion sistematica de profundidad en listas anidadas.
- Debilidad del hidden state `record_prefix_state` para niveles profundos.

Estos problemas apuntan a la formulacion del siguiente experimento: otra
alineacion de hidden state, perdida ordinal, pesos por clase o una cabeza que
pueda ver parte del contenido de la linea.
