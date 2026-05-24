# Two-Head SFT v1

Este documento describe el entrenamiento `two_head_sft` implementado para
Kubernetes v1. Su objetivo es dejar trazable que se entrena, que ve el modelo,
que se predice por la cabeza autoregresiva, que se predice por la cabeza
estructural, que artefactos se guardan y como se reanuda un run si se interrumpe.

## Proposito del run

`two_head_sft` es la rama principal de la comparacion supervisada:

```text
serialized_sft vs two_head_sft
```

La rama `serialized_sft` genera directamente `blocks_tsv_v1`, incluyendo
`level` como texto ordinario. En cambio, `two_head_sft` separa la prediccion en
dos salidas coordinadas:

- una salida textual autoregresiva que genera lineas de contenido;
- una cabeza estructural explicita que predice el `level` de cada linea.

El punto cientifico del run no es solo mejorar una metrica final, sino comprobar
si hacer de `level` una variable supervisada separada ayuda a modelar la
jerarquia YAML mejor que aprenderla como texto plano.

## Datos de entrada

El entrenamiento usa los mismos ficheros SFT ya congelados para Kubernetes v1:

```text
data/processed/kubernetes_v1/sft/train.jsonl
data/processed/kubernetes_v1/sft/validation.jsonl
```

El split `test` no se usa durante entrenamiento ni seleccion de checkpoints.
Debe reservarse para evaluar candidatos estables.

Cada fila SFT contiene, como minimo:

```json
{
  "sample_id": "...",
  "prompt_variant": "...",
  "split": "train|validation|test",
  "prompt": "...",
  "target": "<blocks>...</blocks>",
  "target_yaml_normalized": "..."
}
```

El campo `target` esta en `blocks_tsv_v1`:

```text
<blocks>
document_index    line_index    level    line_text
...
</blocks>
```

Este formato es la fuente supervisada original, pero no es la superficie textual
que ve la rama `two_head_sft`.

## Transformacion a `content_blocks_v1`

Antes de tokenizar una fila, el trainer parsea `blocks_tsv_v1` y elimina la
columna `level` de la superficie generada. La entrada supervisada para la cabeza
LM queda asi:

```text
<content_blocks>
document_index    line_index    line_text
...
</content_blocks>
```

Por ejemplo, un target original:

```text
<blocks>
0    0    0    apiVersion: v1
0    1    0    kind: ConfigMap
0    2    0    metadata:
0    3    1    name: app
</blocks>
```

se transforma en:

```text
<content_blocks>
0    0    apiVersion: v1
0    1    kind: ConfigMap
0    2    metadata:
0    3    name: app
</content_blocks>
```

El `level` no desaparece del entrenamiento: queda guardado como etiqueta interna
para la cabeza estructural. Lo importante es que no aparece en los tokens que el
modelo debe copiar o generar como texto.

## Que se predice y que no

Para cada muestra, el modelo recibe un prompt instruction-tuned que pide devolver
solo bloques de contenido:

```text
You generate Kubernetes manifests through an explicit content-line representation.
Return only content line blocks; each block must include document_index, line_index, and line_text.
Do not include hierarchy levels in the output surface.

Natural-language request:
...

Return the content line block sequence now.
```

Durante entrenamiento se supervisan dos objetivos:

- **Predicho por la cabeza LM causal**:
  - `<content_blocks>`;
  - `document_index`;
  - `line_index`;
  - `line_text`;
  - `</content_blocks>`.
- **Predicho por la cabeza estructural**:
  - un `level` por linea de contenido.
- **No predicho como texto por `two_head_sft`**:
  - la columna gold `level`;
  - la indentacion YAML final;
  - el YAML reconstruido final.

La indentacion no vive en `line_text`. `line_text` debe permanecer sin
indentacion inicial. La indentacion final la aplica el parser usando el `level`
predicho por la cabeza estructural.

## Alineacion de la cabeza de `level`

La politica de alineacion usada en el primer run es:

```text
record_prefix_state
```

Para cada linea de `content_blocks_v1`, la etiqueta `level` se asocia al hidden
state inmediatamente anterior al registro de esa linea. Es decir, la cabeza de
nivel predice la jerarquia de la siguiente linea desde:

- el prompt;
- los registros anteriores ya presentes en la secuencia autoregresiva;
- no desde la columna gold `level`, porque esa columna no esta en el texto;
- no desde el contenido textual de la linea actual, salvo en el caso de fallback
  tecnico si el tokenizer no ofrece un token anterior alineable.

El trainer usa offsets del tokenizer para mapear posiciones de caracteres a
indices de token. Todas las posiciones que no corresponden a un prefijo de
registro quedan enmascaradas con `-100` para la loss estructural.

## Modelo

El modelo base local es:

```text
model/qwen2.5-7b-instruct-4bit/
```

La arquitectura entrenada contiene:

- backbone `Qwen2ForCausalLM`;
- LoRA sobre las proyecciones de atencion:

```text
q_proj,k_proj,v_proj,o_proj
```

- cabeza LM causal estandar del modelo;
- cabeza MLP de clasificacion para `level`.

La cabeza de nivel no es un target LoRA. Es un modulo nuevo entrenable, guardado
aparte como:

```text
level_head.pt
```

Por defecto:

```text
level_class_count = 9
level_head_hidden_dim = 256
level_head_dropout = 0.05
```

Las clases corresponden a los niveles observados `0..8` en Kubernetes v1.

## Funcion de perdida

El primer run usa una perdida simple y trazable:

```text
loss = lm_loss + lambda_level * level_loss
```

con:

```text
lambda_level = 1.0
```

Donde:

- `lm_loss` es la cross-entropy causal sobre tokens de `content_blocks_v1`;
- `level_loss` es cross-entropy multiclase sobre las etiquetas `level`;
- el prompt se enmascara con `-100`;
- las posiciones que no son `record_prefix_state` se ignoran para `level_loss`.

Este run no usa:

- regresion ordinal;
- pesos por clase;
- uncertainty weighting;
- reward shaping;
- DPO;
- PPO;
- auxiliares de requisitos del prompt.

Si los errores muestran que confundir `level=4` con `level=5` y con `level=0`
debe penalizarse de forma distinta, la extension natural seria una perdida
ordinal o una mezcla de cross-entropy y penalizacion por distancia. Eso debe ser
otro experimento, no este primer run.

## Validacion durante entrenamiento

La validacion intermedia esta implementada por pasos de optimizador, no por epoca
literal. El parametro relevante es:

```text
--eval-checkpoint-steps
```

Con el dataset actual:

```text
train rows = 426
batch_size = 1
gradient_accumulation_steps = 8
optimizer steps per epoch = ceil(426 / 8) = 54
```

Por tanto, para validar aproximadamente una vez por epoca se usa:

```text
--eval-checkpoint-steps 54
```

La validacion intermedia usa:

```text
--eval-max-samples 10
--eval-sample-strategy random
```

La seleccion aleatoria es determinista por `seed` y `global_step`. Esto permite
comparar checkpoints sin depender de una muestra manual elegida a dedo.

La validacion final usa todas las filas de `validation.jsonl`, salvo que se
limite explicitamente con `--max-validation-samples`.

## Flujo de inferencia en validacion

Para cada fila de validation:

1. Se construye el prompt content-only.
2. El modelo genera texto en formato `content_blocks_v1`.
3. Se parsea la salida generada para obtener:

```text
document_index, line_index, line_text
```

4. Se vuelve a pasar por el modelo el prompt mas los `content_blocks` generados.
5. La cabeza estructural predice un `level` por registro generado.
6. Se combinan:

```text
document_index, line_index, predicted_level, line_text
```

7. Se reconstruye el contrato parser-facing equivalente a `blocks_tsv_v1`.
8. El parser reconstruye YAML aplicando indentacion desde `predicted_level`.
9. Se calculan las metricas existentes de estructura, prompt y dominio.

En validacion no se usa el `level` gold para reconstruir la prediccion. El gold
solo se usa dentro del evaluador para comparar la salida con la referencia.

## Normalizacion de superficie en validacion

La salida `content_blocks_v1` contiene dos variables posicionales:

```text
document_index, line_index
```

Estas variables no son contenido YAML. Sirven para ordenar las lineas generadas
antes de pasar por la cabeza de nivel y por el parser. Por ese motivo, la
validacion aplica una normalizacion superficial acotada antes de predecir los
niveles:

1. Si una fila visible tiene exactamente dos campos despues de que ya exista una
   fila valida en el mismo bloque, se interpreta como una omision del
   `document_index`:

```text
line_index    line_text
```

   y se normaliza a:

```text
previous_document_index    line_index    line_text
```

2. Una vez leidas las filas, `line_index` se renumera por orden de aparicion
   dentro de cada `document_index`. La generacion ya fija el orden de las
   lineas; el indice visible se trata como una ayuda de superficie, no como una
   prediccion semantica independiente.

3. Cualquier otra fila malformada se rechaza como error de superficie. En
   particular, la validacion no trunca silenciosamente el output crudo si
   aparecen filas invalidas despues de filas validas.

Esta normalizacion no repara jerarquia YAML, no cambia `line_text`, no introduce
contenido y no usa `level` gold. Solo elimina fallos de serializacion posicional
que son recuperables de forma determinista. Si se recomputan metricas con esta
normalizacion despues de un run ya finalizado, deben reportarse como metricas
postprocesadas y no como las metricas originales del entrenamiento.

## Metricas registradas

Durante entrenamiento se escriben en `train_log.jsonl` y en W&B:

```text
train/loss
train/lm_loss
train/level_loss
train/lambda_level
train/learning_rate
train/epoch
train/batch_index
```

Durante validacion se registran metricas parser-facing, incluyendo:

```text
structured_output_parse_success_rate
yaml_parse_success_rate
parsed_equal_rate
average_line_text_f1
average_level_exact_match_rate
average_level_mae
average_prompt_requirement_precision
average_prompt_requirement_recall
average_prompt_requirement_f1
average_required_field_complete_resource_rate
required_field_complete_sample_rate
average_kubernetes_domain_validity_level
```

W&B es una capa de observabilidad. La fuente de verdad para resume son los
artefactos locales.

## Artefactos del run

Cada run escribe bajo:

```text
results/two_head_sft_kubernetes_v1/<run-id>/
```

Artefactos principales:

```text
config.json
state.json
train_log.jsonl
validation_predictions.jsonl
intermediate_validation_predictions.jsonl
intermediate_validation_metrics.jsonl
validation_metrics_progress.jsonl
validation_example_metrics.jsonl
metrics.json
checkpoints/
```

Cada checkpoint incluye:

```text
adapter/
tokenizer/
level_head.pt
training_state.pt
```

`training_state.pt` contiene:

```text
optimizer state
scheduler state
run state
```

Esto permite reanudar entrenamiento desde el ultimo checkpoint local compatible.

## Resume y compatibilidad de runs

El trainer guarda una `resume_signature` en `config.json`. Si se relanza el mismo
`run-id`, el script compara la nueva configuracion con esa firma. Si cambian
parametros que alteran la identidad del entrenamiento, el resume se rechaza.

Parametros incluidos en la firma:

```text
model_variant
serialization
train_file
validation_file
base_model_path
batch_size
epochs
learning_rate
gradient_accumulation_steps
max_seq_length
max_new_tokens
checkpoint_keep_last
max_train_samples
max_validation_samples
seed
lora_r
lora_alpha
lora_dropout
lora_target_modules
lambda_level
level_class_count
level_head_hidden_dim
level_head_dropout
level_alignment_policy
target_contract
```

Para reanudar un run interrumpido, se debe lanzar el mismo comando con el mismo
`--run-id` y los mismos parametros de identidad. El script buscara el ultimo
checkpoint bajo:

```text
results/two_head_sft_kubernetes_v1/<run-id>/checkpoints/
```

y cargara:

- adapter LoRA;
- `level_head.pt`;
- optimizer;
- scheduler;
- estado de epoch/batch/global_step.

## Comando recomendado para el primer run

```powershell
uv run python scripts\train_kubernetes_two_head_sft.py `
  --output-dir results\two_head_sft_kubernetes_v1 `
  --run-id two-head-sft-v1-20260516 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --epochs 3 `
  --checkpoint-steps 25 `
  --checkpoint-keep-last 0 `
  --eval-checkpoint-steps 54 `
  --eval-max-samples 10 `
  --eval-sample-strategy random `
  --validation-log-every 1 `
  --wandb-mode online `
  --wandb-log-artifacts `
  --oom-recovery skip_batch `
  --max-oom-skips 0
```

Notas:

- `--checkpoint-keep-last 0` conserva todos los checkpoints.
- `--eval-checkpoint-steps 54` valida aproximadamente una vez por epoca.
- `--wandb-mode online` sube metricas en directo.
- `--wandb-log-artifacts` sube artefactos al cerrar correctamente el run.
- `--oom-recovery skip_batch` permite continuar si una microbatch causa CUDA OOM,
  registrando el incidente en `oom_skipped_batches.jsonl`.
- `--max-oom-skips 0` significa que no se impone limite explicito adicional de
  skips por OOM.

## Comprobaciones anti-leakage

Antes de interpretar resultados, debe verificarse:

- las salidas visibles contienen `<content_blocks>`, no `<blocks>`;
- cada linea visible tiene tres columnas, no cuatro;
- no aparece la columna gold `level` en `raw_model_output`;
- `predicted_blocks` se construye con `predicted_level`, no con `level` gold;
- `average_level_exact_match_rate` se calcula solo en evaluacion, comparando
  predicciones contra referencia;
- `test.jsonl` no se usa durante seleccion de modelo.

El run debe presentarse como SFT con supervision estructural explicita, no como
validacion completa de correccion Kubernetes. La validez de dominio sigue siendo
aproximada salvo que se incorpore validacion oficial de schemas o ejecucion
funcional.
