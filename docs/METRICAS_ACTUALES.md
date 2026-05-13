# Metricas actuales del repositorio

Este documento describe las metricas que el repositorio calcula **ahora mismo**
segun la implementacion real, no segun la hoja de ruta futura.

La informacion de este documento sale principalmente de:

- `utils/kubernetes_dataset_preprocessor.py`
- `scripts/build_kubernetes_structural_targets.py`
- `scripts/analyze_kubernetes_dataset.py`
- `scripts/run_kubernetes_baseline.py`
- `scripts/train_kubernetes_sft.py`
- `scripts/recompute_sft_validation_metrics.py`
- `scripts/run_kubernetes_latent_level_probe.py`
- `src/llm_structured_semantic_generation/evaluation.py`
- `src/llm_structured_semantic_generation/kubernetes_domain.py`
- `src/llm_structured_semantic_generation/auxiliary_text_metrics.py`
- `src/llm_structured_semantic_generation/latent_level_probe.py`

## Alcance

Actualmente hay **siete familias** de metricas o checks:

1. metricas de preparacion del dataset
2. metricas de validacion de targets estructurales
3. metricas descriptivas del dataset
4. metricas de evaluacion del baseline generativo
5. metricas de validacion del SFT serializado
6. metricas de validez de dominio Kubernetes y texto auxiliar
7. metricas diagnosticas de probing latente de `level`

No todas miden lo mismo:

- unas miden **calidad del dato**
- otras miden **si el pipeline esta listo para la siguiente etapa**
- otras miden **cobertura y complejidad**
- y otras miden **calidad de generacion**

Tambien es importante dejar claro que **todavia no** se esta haciendo una
validacion completa contra el esquema oficial de Kubernetes ni una evaluacion
humana. Varias metricas semanticas actuales son aproximaciones automaticas.

## 1. Metricas de preparacion del dataset

Estas metricas se escriben en:

- `data/processed/kubernetes_v1/quality_report.json`

Su funcion no es medir el modelo, sino comprobar que el dataset procesado es
coherente y utilizable.

### `sample_count`

Que mide:

- numero total de samples del dataset

Como se calcula:

- conteo de filas a nivel `sample_id`

### `prompt_variant_count`

Que mide:

- numero total de variantes de prompt

Como se calcula:

- conteo de filas a nivel `(sample_id, prompt_variant)`

### `expected_sample_count` y `expected_prompt_variant_count`

Que miden:

- el volumen esperado del dataset tras el preprocessing

Como se calculan:

- a partir de lo que el pipeline espera exportar despues de leer y validar la
  fuente

Para que sirven:

- detectar perdidas silenciosas de muestras durante el preprocessing

### `validation_status_counts`

Que mide:

- cuantas filas quedaron en estado `ok` y cuantas en `reject`

Como se calcula:

- conteo por etiqueta de validacion final del preprocesador

Interpretacion:

- si hubiera `reject`, el dataset tendria filas descartadas por problemas de
  integridad o parseo

### `sample_split_counts`

Que mide:

- cuantos samples hay en `train`, `validation` y `test`

Como se calcula:

- conteo por split a nivel sample

### `prompt_variant_split_counts`

Que mide:

- cuantas filas de prompt-variant hay en cada split

Como se calcula:

- conteo por split a nivel variante de prompt

### `all_yaml_parse_ok`

Que mide:

- si todos los YAML objetivo del dataset se pueden parsear

Como se calcula:

- es `true` si el 100% de los targets pasan por el parser YAML sin error

### `normalization_is_deterministic`

Que mide:

- si la normalizacion canonica del YAML es estable

Como se calcula:

- el pipeline vuelve a normalizar y comprueba que el resultado no cambia

### `shared_split_per_sample_ok`

Que mide:

- si las dos variantes de prompt del mismo sample se han quedado en el mismo
  split

Como se calcula:

- comprobacion booleana sobre las asignaciones de split por `sample_id`

### `largest_leakage_group_size`

Que mide:

- el tamano del mayor grupo de leakage

Como se calcula:

- se agrupan samples que comparten YAML identico o señales fuertes de
  duplicidad y se toma el maximo

Interpretacion:

- cuanto mas alto sea, mayor es la necesidad de respetar el split por grupos

### `ready_for_next_step`

Que mide:

- si el dataset procesado puede pasar a la siguiente etapa del pipeline

Como se calcula:

- agregando varias puertas de validacion

### `readiness_gates`

Que mide:

- el detalle de las puertas de preparacion

Campos actuales:

- `manifest_complete`
- `yaml_validation_100_percent`
- `no_reject_rows`
- `variants_share_split`
- `export_train_ready_exists`

Interpretacion:

- son checks de integridad, no metricas de calidad de generacion

## 2. Metricas de validacion de targets estructurales

Estas metricas se escriben en:

- `data/processed/kubernetes_v1/structural_targets_report.json`

Miden si la transformacion:

`YAML normalizado -> bloques (document_index, line_index, level, line_text)`

ha salido bien para todas las filas.

### `row_count`

Que mide:

- numero de filas exportadas al archivo de targets estructurales

Como se calcula:

- longitud total de `dataset_structural_targets.jsonl`

### `split_counts`

Que mide:

- cuantas filas estructurales hay por split

Como se calcula:

- conteo por `split`

### `status_counts`

Que mide:

- cuantas filas tienen `structural_target_status = ok` y cuantas `reject`

Como se calcula:

- para cada fila se hace un `validate_round_trip(...)`
- el estado es `ok` si el YAML reconstruido parsea bien y preserva la semantica

### `ready_for_baseline`

Que mide:

- si los targets estructurales estan listos para alimentar el baseline

Como se calcula:

- es `true` cuando no hay rejects y existe al menos una fila

### `average_block_count`

Que mide:

- numero medio de bloques por fila

Como se calcula:

- media aritmetica de `block_count`

Formula:

```text
average_block_count = sum(block_count_i) / N
```

### `max_block_count`

Que mide:

- el sample estructuralmente mas largo, medido en bloques

Como se calcula:

- maximo de `block_count`

### `error_examples`

Que mide:

- no es una metrica resumen, sino una muestra de fallos

Como se calcula:

- se guardan algunos ejemplos de filas rechazadas con sus errores

## 3. Metricas descriptivas del dataset

Estas metricas se escriben en:

- `results/dataset_analysis_kubernetes_v1/dataset_analysis_summary.json`

Su objetivo es caracterizar el dataset antes del entrenamiento. No evalúan la
salida del modelo.

## 3.1 Cobertura y composicion

### `sample_count`

Que mide:

- numero de samples del dataset

### `resource_document_count`

Que mide:

- numero total de documentos YAML parseados

Importante:

- un sample puede contener varios documentos separados por `---`
- por eso `resource_document_count` puede ser mayor que `sample_count`

### `split_counts`

Que mide:

- distribucion de samples por split

### `resource_count_per_sample`

Que mide:

- cuantas muestras tienen 1 recurso, 2 recursos, 3 recursos, etc.

Como se calcula:

- contando el numero de documentos YAML parseados por sample

### `top_kinds`

Que mide:

- frecuencia de los `kind` de Kubernetes mas comunes

Como se calcula:

- conteo de `kind` a nivel documento YAML

### `top_api_versions`

Que mide:

- frecuencia de las `apiVersion` mas comunes

Como se calcula:

- conteo de `apiVersion` a nivel documento YAML

### `leakage_reasons`

Que mide:

- distribucion de los motivos de agrupacion por leakage

Como se calcula:

- conteo del campo `leakage_reasons`

### `exact_yaml_duplicate_sample_count`

Que mide:

- numero de samples que pertenecen a grupos con YAML exactamente duplicado

Como se calcula:

- filtrando samples con `leakage_reasons == exact_yaml_duplicate`

## 3.2 Complejidad del prompt

### `prompt_word_count`

Que mide:

- longitud en palabras del prompt original

### `simplified_prompt_word_count`

Que mide:

- longitud en palabras del prompt simplificado

### `prompt_pair_similarity`

Que mide:

- similitud entre prompt original y prompt simplificado del mismo sample

Como se calcula:

- con la heuristica de similitud ya persistida por el preprocessing

Para estas tres metricas, el script guarda:

- `min`
- `p25`
- `median`
- `p75`
- `max`
- `mean`

Es decir, se usa un resumen descriptivo de seis estadisticos.

## 3.3 Complejidad estructural del YAML

### `yaml_max_depth`

Que mide:

- profundidad maxima del arbol YAML parseado

Como se calcula:

- recursivamente sobre mappings, listas y escalares
- para multi-documento se toma el maximo entre documentos

Importante:

- **no** es lo mismo que `level`
- `yaml_max_depth` mide profundidad del arbol parseado
- `level` mide indentacion por linea en la representacion en bloques

### `yaml_total_nodes`

Que mide:

- complejidad total del YAML en numero de nodos parseados

Como se calcula:

- suma de:
  - `yaml_mapping_nodes`
  - `yaml_list_nodes`
  - `yaml_scalar_nodes`

Formula:

```text
yaml_total_nodes = yaml_mapping_nodes + yaml_list_nodes + yaml_scalar_nodes
```

### `block_count`

Que mide:

- complejidad lineal del target estructural

Como se calcula:

- numero de bloques derivados del YAML normalizado

Tambien se resume con:

- `min`
- `p25`
- `median`
- `p75`
- `max`
- `mean`

### `level_distribution`

Que mide:

- cuantas lineas-bloque hay en cada nivel de indentacion

Como se calcula:

- conteo de `level` sobre todos los bloques del dataset estructural

Interpretacion:

- describe donde se concentra la profundidad superficial de las lineas YAML

## 3.4 Cobertura semantica aproximada

### `semantic_field_presence`

Que mide:

- presencia aproximada de ciertos campos semanticamente relevantes de
  Kubernetes

Campos actuales:

- `metadata`
- `spec`
- `containers`
- `image`
- `ports`
- `env`
- `volumes`
- `volumeMounts`
- `selector`
- `template`
- `data`
- `rules`
- `subjects`
- `roleRef`

Como se calcula:

- se recorren recursivamente las claves del YAML parseado
- para cada sample se marca si cada campo aparece al menos una vez
- despues se calcula la media de ese booleano en todo el dataset

Formula:

```text
semantic_field_presence(field) =
  numero_de_samples_que_contienen_field / numero_total_de_samples
```

Limitacion importante:

- esto **no** es validacion de esquema Kubernetes
- solo es una señal de cobertura por presencia de claves

## 3.5 Medidas calculadas para visualizaciones

Ademas del resumen JSON, el script calcula tablas y conteos para las figuras del
informe HTML:

- histogramas de `yaml_max_depth`, `yaml_total_nodes` y `block_count`
- boxplots de complejidad por `primary_kind`
- distribucion de longitud de prompts
- scatter de longitud de prompt vs complejidad YAML
- balance por split y leakage
- heatmap de `primary_kind` por split
- heatmap de `primary_kind` por `yaml_max_depth`
- co-ocurrencia de campos semanticos

Estas cantidades se calculan realmente, pero varias de ellas se quedan en
figuras o tablas intermedias y no siempre aparecen como campo individual en el
JSON resumen.

## 4. Metricas de evaluacion generativa

Estas metricas se escriben en runs generativos, por ejemplo:

- `results/baseline_kubernetes_v1/<run-id>/metrics.json`
- `results/sft_kubernetes_v1/<run-id>/metrics.json`
- `results/sft_kubernetes_v1/<run-id>/validation_metrics_recomputed.json`

Su definicion implementada vive en:

- `scripts/run_kubernetes_baseline.py`
- `scripts/train_kubernetes_sft.py`
- `scripts/recompute_sft_validation_metrics.py`
- `src/llm_structured_semantic_generation/evaluation.py`

La diferencia principal entre baseline y SFT no esta en el evaluador, sino en
como se obtienen las predicciones:

- el baseline usa el modelo base sin adaptar y normalmente genera
  `blocks_tsv_compact_v1`;
- `serialized_sft` usa un adapter LoRA y genera `blocks_tsv_v1`;
- ambos se comparan mediante el mismo contrato parser-facing:
  salida estructurada -> bloques -> reconstruccion YAML -> evaluacion.

## 4.1 Como se evalua una prediccion

Para cada fila:

1. el modelo genera una salida estructurada
2. esa salida se parsea a bloques
3. los bloques se reconstruyen a YAML
4. ese YAML reconstruido se compara con el YAML de referencia

Importante:

- si falla el parseo de la salida estructurada, **no** hay evaluacion completa
- por eso algunas metricas se calculan sobre todas las filas y otras solo sobre
  las filas evaluadas

## 4.2 Metricas de cobertura de evaluacion

### `row_count`

Que mide:

- numero total de filas procesadas por el run

### `evaluated_count`

Que mide:

- numero de filas que llegaron a tener objeto `evaluation`

Como se calcula:

- conteo de predicciones con `evaluation != null`

### `structured_output_parse_success_rate`

Que mide:

- porcentaje de filas cuya salida estructurada del modelo pudo convertirse en
  bloques validos

Como se calcula:

```text
evaluated_count / row_count
```

### `json_block_parse_success_rate`

Que mide:

- hoy mide exactamente lo mismo que `structured_output_parse_success_rate`

Como se calcula:

```text
evaluated_count / row_count
```

Nota:

- el nombre se ha quedado como herencia de versiones anteriores
- en la implementacion actual es un alias redundante

## 4.3 Metricas booleanas agregadas como tasa

La funcion `summarize_evaluations(...)` convierte varias comprobaciones
booleanas en tasas medias entre `0` y `1`.

Formula general:

```text
rate = numero_de_true / numero_de_filas_evaluadas
```

### `yaml_parse_success_rate`

Que mide:

- porcentaje de filas evaluadas cuyo YAML reconstruido se puede parsear

### `parsed_equal_rate`

Que mide:

- porcentaje de filas evaluadas en las que el YAML predicho y el YAML de
  referencia son exactamente iguales **despues de parsearlos**

Como se calcula:

- se comparan directamente las estructuras parseadas, no el texto crudo

### `block_parse_success_rate`

Que mide:

- porcentaje de filas evaluadas cuyo YAML reconstruido puede volver a pasarse a
  bloques sin romper el contrato estructural

Nota:

- en la implementacion actual suele coincidir con `yaml_parse_success_rate`

### `document_index_monotonic_ok_rate`

Que mide:

- porcentaje de filas donde `document_index` no retrocede al recorrer los
  bloques

### `line_index_sequence_ok_rate`

Que mide:

- porcentaje de filas donde `line_index` es consecutivo dentro de cada
  documento

Como se calcula:

- para cada `document_index` se espera la secuencia `0, 1, 2, ...`

### `document_count_match_rate`

Que mide:

- porcentaje de filas donde el numero de documentos YAML predichos coincide con
  el de referencia

### `line_count_match_rate`

Que mide:

- porcentaje de filas donde el numero de bloques/lineas coincide exactamente
  con el de referencia

## 4.4 Metricas medias de error o calidad estructural

Estas metricas se agregan con media aritmetica sobre las filas evaluadas.

### `average_valid_block_ratio`

Que mide:

- fraccion de bloques predichos que cumplen el contrato minimo del parser

Como se calcula por fila:

```text
valid_block_ratio = bloques_validos / bloques_predichos_totales
```

### `average_indentation_leak_rate`

Que mide:

- cuanto texto de bloque trae indentacion embebida, algo que no deberia pasar

Como se calcula por fila:

```text
indentation_leak_rate = bloques_con_line_text_que_empieza_por_espacio_o_tab / bloques_validos
```

Interpretacion:

- idealmente debe ser `0.0`, porque la indentacion debe vivir en `level`, no en
  `line_text`

### `average_document_count_error`

Que mide:

- error absoluto medio en numero de documentos

Como se calcula por fila:

```text
abs(reference_document_count - prediction_document_count)
```

### `average_block_count_error`

Que mide:

- error absoluto medio en numero de bloques

Como se calcula por fila:

```text
abs(line_count_reference - line_count_prediction)
```

## 4.5 Metricas de coincidencia con la referencia

### `average_content_exact_match_rate`

Que mide:

- cuanto coincide exactamente el contenido textual de las lineas, posicion a
  posicion

Como se calcula por fila:

- se comparan `reference_text` y `prediction_text`
- se usa `_match_rate(...)`

Formula:

```text
content_exact_match_rate =
  numero_de_posiciones_iguales_en_el_prefijo_solapado / numero_de_lineas_de_referencia
```

Matiz:

- el denominador es la longitud de la referencia, no la de la prediccion

### `average_level_exact_match_rate`

Que mide:

- cuanto coincide exactamente la indentacion estructural (`level`) posicion a
  posicion

Como se calcula:

- igual que la anterior, pero usando las secuencias de niveles

### `average_line_text_precision`

Que mide:

- pureza del conjunto multiconjunto de lineas generadas

Como se calcula:

- se cuentan las ocurrencias de cada `line_text` en referencia y prediccion
- los verdaderos positivos son la suma de los minimos por item

Formula:

```text
precision = true_positive / total_lineas_predichas
```

### `average_line_text_recall`

Que mide:

- cobertura del conjunto multiconjunto de lineas de referencia

Formula:

```text
recall = true_positive / total_lineas_de_referencia
```

### `average_line_text_f1`

Que mide:

- equilibrio entre precision y recall de `line_text`

Formula:

```text
F1 = 2 * precision * recall / (precision + recall)
```

Importante:

- esta F1 no usa embeddings ni matching semantico
- opera sobre igualdad exacta de cadenas `line_text`

### `average_level_mae`

Que mide:

- error absoluto medio en niveles estructurales

Como se calcula por fila:

- se toma el prefijo solapado de referencia y prediccion
- se promedia `abs(level_ref - level_pred)`

Formula:

```text
level_mae =
  sum(abs(level_ref_i - level_pred_i)) / numero_de_posiciones_solapadas
```

Si no hay solape:

- la metrica vale `null` para esa fila

## 4.6 Metricas de identidad gruesa del recurso

### `primary_kind_match_rate`

Que mide:

- porcentaje de filas donde el `kind` del primer documento coincide con la
  referencia

### `primary_api_version_match_rate`

Que mide:

- porcentaje de filas donde el `apiVersion` del primer documento coincide con
  la referencia

### `primary_metadata_name_match_rate`

Que mide:

- porcentaje de filas donde `metadata.name` del primer documento coincide con
  la referencia

### `average_kind_sequence_match_rate`

Que mide:

- cuanto coincide la secuencia de `kind` documento a documento

Como se calcula:

- se extrae la lista de `kind` de referencia y prediccion
- se aplica `_match_rate(...)` posicion a posicion

## 4.7 Metricas semanticas aproximadas

### `average_semantic_key_precision`

### `average_semantic_key_recall`

### `average_semantic_key_f1`

Que miden:

- similitud entre el conjunto de claves semanticas relevantes presentes en la
  referencia y en la prediccion

Conjunto de claves observado:

- `metadata`
- `spec`
- `containers`
- `image`
- `ports`
- `env`
- `volumes`
- `volumeMounts`
- `selector`
- `template`
- `data`
- `rules`
- `subjects`
- `roleRef`

Como se calcula:

1. se recorren recursivamente las claves de los documentos YAML
2. se construye un conjunto de claves presentes dentro de esa lista fija
3. se aplica precision/recall/F1 por igualdad exacta de clave

Importante:

- esta metrica compara **presencia de claves**, no valores
- por tanto es una metrica semantica **aproximada**

## 4.8 Metricas de consistencia interna del YAML predicho

Estas tres metricas no comparan directamente contra la referencia. Miden si la
prediccion es internamente coherente segun reglas simples del dominio.

### `average_workload_selector_template_consistency`

Que mide:

- para workloads como `Deployment`, `DaemonSet`, `StatefulSet` o `ReplicaSet`,
  si `spec.selector.matchLabels` esta contenido en
  `spec.template.metadata.labels`

Como se calcula por documento aplicable:

```text
satisface = all(labels[key] == value for key, value in selector.items())
```

La metrica final es:

```text
documentos_que_satisfacen / documentos_aplicables
```

### `average_service_selector_match_rate`

Que mide:

- si el selector de un `Service` coincide con las labels de al menos un
  workload del mismo sample predicho

Como se calcula:

- para cada `Service` aplicable se comprueba si existe algun workload cuyas
  labels cumplan todas las entradas del selector

### `average_volume_mount_consistency`

Que mide:

- si todos los `volumeMounts.name` usados por los contenedores aparecen tambien
  definidos en `spec.volumes`

Como se calcula:

- para cada `pod_spec` aplicable:
  - se recogen los nombres montados
  - se recogen los nombres definidos en `volumes`
  - se comprueba que todos los montados existan

Si no hay casos aplicables:

- cualquiera de estas metricas queda en `null`

## 4.9 Metrica KDV de validez de dominio Kubernetes

La evaluacion generativa tambien calcula ahora una metrica por niveles llamada
`kubernetes_domain_validity_level`. Esta metrica es una bateria automatica
hibrida sin cluster. No garantiza correccion absoluta de Kubernetes, pero
separa los fallos por capas cada vez mas exigentes.

Campos por fila:

- `kubernetes_domain_validity_level`
- `kubernetes_domain_gate_pass`
- `kubernetes_domain_validity_score`
- `kubernetes_domain_level_scores`
- `kubernetes_domain_errors`

Niveles actuales:

- nivel 0: el YAML predicho parsea correctamente
- nivel 1: el contrato de bloques y reconstruccion parser-facing se satisface
- nivel 2: identidad Kubernetes minima (`apiVersion`, `kind`,
  `metadata.name`, `kind` conocido y campos requeridos por tipo)
- nivel 3: invariantes intra-recurso, como selectors frente a template labels,
  volumenes frente a `volumeMounts`, imagenes de contenedor, puertos,
  schedules y rangos de replicas
- nivel 4: invariantes entre recursos del mismo sample, como `Service`
  apuntando a workloads locales compatibles y referencias locales resolubles
  a recursos como `ConfigMap`, `Secret`, `PersistentVolumeClaim`,
  `ServiceAccount` o `Role`
- nivel 5: smells o senales de QA estatica, como falta de CPU/memory
  requests/limits, `latest` image tag, falta de `runAsNonRoot`,
  falta de `readOnlyRootFilesystem`, `allowPrivilegeEscalation`,
  contenedores privilegiados y namespaces del host activados

La metrica agregada expone:

- `average_kubernetes_domain_validity_score`
- `kubernetes_domain_gate_pass_rate`
- `average_kubernetes_domain_validity_level`
- `kubernetes_domain_error_counts`
- `kubernetes_level_0_pass_rate` hasta `kubernetes_level_5_pass_rate`

Interpretacion:

- pasar nivel 5 significa que la salida supera todos los checks automaticos
  implementados actualmente
- no significa que el manifiesto sea semanticamente perfecto ni que haya sido
  aplicado en un cluster real
- los errores se guardan por categoria para poder hacer analisis de fallos y
  construir preferencias automaticas mas adelante

## 4.10 Senales textuales auxiliares

Ademas de las metricas estructurales y de dominio, cada evaluacion incluye
`auxiliary_text_metrics`. Estas senales se guardan para comparabilidad con
literatura previa, pero no deciden si una prediccion es valida.

Campos por fila:

- `bleu_score`
- `rouge1_f1`
- `rouge2_f1`
- `rougeL_f1`
- `perplexity`
- `perplexity_available`

Como se calculan:

- BLEU y ROUGE comparan YAML de referencia y YAML predicho tras una
  normalizacion estable cuando el texto es parseable
- ROUGE se reporta como F1 para ROUGE-1, ROUGE-2 y ROUGE-L
- perplexity se calcula solo cuando el script tiene acceso real al modelo y a
  sus log-probabilities; si no esta disponible, queda en `null`

Campos agregados:

- `average_bleu_score`
- `average_rouge1_f1`
- `average_rouge2_f1`
- `average_rougeL_f1`
- `average_perplexity`
- `perplexity_available_rate`

Limitacion importante:

- estas metricas son senales debiles de similitud superficial o probabilidad
  bajo el modelo, no medidas fiables de validez Kubernetes

## 4.11 Metricas de recogida latente

Si el run se ejecuta con `--collect-latent-means`, `metrics.json` añade el
bloque `latent_collection`.

Esto no mide calidad de prediccion; mide el estado de la recogida de vectores
latentes.

### `latent_collection.enabled`

Que mide:

- si la recogida latente estaba activada

### `latent_collection.row_count`

Que mide:

- numero de filas para las que se genero entrada latente

### `latent_collection.rows_with_vector`

Que mide:

- cuantas filas acabaron con un vector medio disponible

### `latent_collection.rows_without_generated_tokens`

Que mide:

- cuantas filas no produjeron vector porque no hubo tokens generados utiles

### `latent_collection.latent_dims`

Que mide:

- dimensionalidades observadas en los vectores

### `latent_collection.artifact`

Que mide:

- no es una metrica numerica; indica el artefacto donde quedaron guardados los
  vectores

## 5. Como interpretar correctamente las metricas actuales

### 1. No todas son metricas de exito del modelo

Ejemplos:

- `ready_for_next_step`
- `ready_for_baseline`
- `manifest_complete`

Estas son puertas operativas del pipeline, no medidas de calidad generativa.

### 2. Las metricas generativas se calculan en dos niveles

- sobre **todas** las filas: por ejemplo `structured_output_parse_success_rate`
- sobre **solo las filas evaluadas**: casi todas las metricas agregadas de
  `summarize_evaluations(...)`

Esto es importante porque una F1 media buena sobre filas evaluadas no captura
los fallos previos de parseo si no se mira tambien la tasa de parseo.

### 3. Las metricas "semanticas" actuales son aproximadas

Ahora mismo no existe una validacion completa contra el esquema Kubernetes ni un
chequeo exhaustivo de coherencia semantica.

Lo que si existe es:

- presencia de claves semanticas relevantes
- consistencia simple entre selector y labels
- consistencia simple entre `volumeMounts` y `volumes`

### 4. BLEU y ROUGE son auxiliares

El repositorio ahora registra BLEU y ROUGE, pero solo como senales auxiliares.
La interpretacion principal sigue priorizando:

- parseabilidad YAML
- estructura
- reconstruccion segura desde bloques
- consistencia de dominio basica

por encima de overlap superficial de texto.

### 5. Las metricas de SFT no cierran la comparacion principal

El run `serialized_sft` ya tiene resultados de validacion fuertes, pero eso no
resuelve por si solo la pregunta central `serialized_sft vs two_head_sft`.
Mientras `two_head_sft` no este implementado y evaluado con el mismo protocolo,
las metricas de SFT serializado deben leerse como el control supervisado
alcanzado, no como el cierre de la tesis.

## 6. Metricas del probe latente de `level`

El workflow de probing se escribe en:

- `results/latent_level_probe_kubernetes_v1/<run-id>/metrics.json`
- `results/latent_level_probe_kubernetes_v1/<run-id>/probe_metrics_*.json`
- `results/latent_level_probe_kubernetes_v1/<run-id>/probe_predictions_*.jsonl`

No evalua manifiestos generados. Evalua si estados ocultos del modelo contienen
informacion recuperable sobre el `level` de cada linea YAML.

Campos principales:

- `row_count`
- `completed_row_count`
- `feature_strategies`
- `probe_count`
- `probe_ids`
- por probe: `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`,
  `level_mae`, `classification_report`, `confusion_matrix`

Estrategias actuales:

- `record_prefix_state`
- `line_prefix_state`
- `line_first_token`
- `line_last_token`
- `line_mean`

Baselines actuales:

- `majority`
- `previous_level`

Probes aprendidos actuales:

- `linear`
- `mlp_h64_l1_d0p0`

Interpretacion:

- si un probe aprendido supera claramente a `majority` y `previous_level`, hay
  evidencia diagnostica de que el `level` es recuperable desde los hidden states;
- si el probe lineal es fuerte, la senal parece bastante accesible linealmente;
- si solo el MLP es fuerte, la senal puede existir pero estar organizada de
  forma no lineal;
- ningun probe sustituye a la evaluacion generativa de `two_head_sft`.

## 7. Resumen corto

Hoy el repositorio calcula realmente:

- checks de integridad y readiness del dataset
- estadisticas descriptivas de cobertura y complejidad
- validacion de targets estructurales
- metricas de parseo, estructura, similitud por bloques, KDV por niveles y
  consistencia de dominio para baseline y SFT serializado
- senales auxiliares BLEU, ROUGE y perplexity opcional
- metricas diagnosticas de probing latente para la variable `level`

Todavia **no** calcula de forma completa:

- validacion oficial de schema Kubernetes
- metricas humanas de calidad
- metricas de adecuacion al prompt basadas en anotacion manual
- metricas RLHF o reward-model complejas
- metricas finales de `two_head_sft`
