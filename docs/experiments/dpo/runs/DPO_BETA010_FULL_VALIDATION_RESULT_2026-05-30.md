# Resultado DPO beta 0.10 completo - 2026-05-30

Document type: run result

## Resumen

Este documento registra el primer experimento completo de DPO sobre Kubernetes
v1 con `beta=0.10`. A diferencia del smoke test previo, esta run usa el conjunto
completo de pares de preferencia disponibles, guarda checkpoints durante el
entrenamiento, ejecuta una validacion intermedia a dos tercios del proceso y
cierra con validacion completa sobre `validation`.

La lectura principal es prudente: el DPO no mejora de forma global al SFT
serializado del que parte. La run muestra una mejora localizada en algunas
metricas de dominio Kubernetes, especialmente en los niveles intermedios de
validez, pero tambien introduce una ligera perdida de estabilidad generativa y
de fidelidad al prompt. Por tanto, este experimento debe leerse como una primera
medida del efecto de DPO automatico, no como una confirmacion de que DPO sea ya
superior a `serialized_sft`.

## Artefactos

- Run DPO: `dpo-beta010-full-20260529-170249`
- Directorio:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/`
- Variante: `serialized_sft_dpo`
- Variante fuente: `serialized_sft`
- Serializacion objetivo: `blocks_tsv_v1`
- Checkpoint final: `checkpoint-step-57`
- Checkpoints guardados: `checkpoint-step-32`, `checkpoint-step-57`
- Split evaluado: `validation`
- Ejemplos evaluados al final: `70/70`
- Metricas finales:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/metrics.json`
- Predicciones finales:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/validation_predictions.jsonl`
- Validacion intermedia:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/two_thirds_validation_metrics.jsonl`
- W&B:
  `https://wandb.ai/mario-garcia-berenguer-universidad-polit-cnica-de-madrid/llm-structured-semantic-generation/runs/dpo-beta010-full-20260529-170249`

Modelo SFT de referencia:

- Run SFT fuente: `serialized-sft-a-v1-20260505-171226`
- Checkpoint fuente: `checkpoint-step-159`
- Metricas SFT comparadas:
  `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/validation_metrics_recomputed.json`

La comparacion usa el mismo split de `validation` con `70` muestras. En el caso
del SFT se usan las metricas recomputadas offline porque incluyen el mismo stack
de evaluacion ampliado que se usa para DPO, incluyendo los niveles de validez
Kubernetes y las metricas auxiliares BLEU/ROUGE.

## Configuracion

El experimento se lanzo con la siguiente configuracion principal:

- `beta`: `0.10`
- `epochs`: `1`
- `batch_size`: `1`
- `gradient_accumulation_steps`: `8`
- `learning_rate`: `5e-6`
- `max_seq_length`: `2048`
- `max_new_tokens`: `1024`
- `checkpoint_steps`: `32`
- `two_thirds_validation_samples`: `10`
- `two_thirds_validation_sample_strategy`: `random`
- `wandb_mode`: `online`

Comando de referencia:

```powershell
uv run python scripts\train_kubernetes_dpo.py `
  --run-id dpo-beta010-full-20260529-170249 `
  --beta 0.10 `
  --checkpoint-steps 32 `
  --two-thirds-validation-samples 10 `
  --wandb-mode online `
  --wandb-run-name dpo-beta010-full-20260529-170249 `
  --wandb-tags dpo-full,beta-0.10,full-validation,two-thirds-validation
```

El entrenamiento completo produjo `57` pasos de optimizacion. No se registraron
microbatches omitidos por OOM (`oom_skipped_batches=0`). La fase de validacion
final fue lenta, pero termino correctamente.

## Validacion Intermedia A Dos Tercios

La validacion intermedia se ejecuto al llegar al paso `38`, que corresponde a
dos tercios de los `57` pasos totales previstos. Se evaluaron `10` muestras de
validation seleccionadas de forma aleatoria y reproducible.

Metricas principales de esa validacion:

| Metrica | Valor |
| --- | ---: |
| `yaml_parse_success_rate` | 0.8000 |
| `block_parse_success_rate` | 0.8000 |
| `parsed_equal_rate` | 0.1000 |
| `average_line_text_f1` | 0.6691 |
| `average_level_exact_match_rate` | 0.5839 |
| `average_prompt_requirement_f1` | 0.7089 |
| `average_kubernetes_domain_validity_score` | 0.6667 |
| `kubernetes_domain_gate_pass_rate` | 0.0000 |

Esta validacion no debe compararse directamente con la validacion final, porque
solo contiene `10` muestras. Su utilidad principal es operacional: confirma que
el entrenamiento podia interrumpirse, evaluar un subconjunto y volver a entrenar
sin perder el estado de la run.

## Comparacion Final SFT vs DPO

| Metrica | SFT `checkpoint-step-159` | DPO `checkpoint-step-57` | Delta DPO - SFT |
| --- | ---: | ---: | ---: |
| `yaml_parse_success_rate` | 0.9857 | 0.9714 | -0.0143 |
| `block_parse_success_rate` | 0.9857 | 0.9714 | -0.0143 |
| `parsed_equal_rate` | 0.1143 | 0.1429 | +0.0286 |
| `document_count_match_rate` | 0.9286 | 0.9429 | +0.0143 |
| `line_count_match_rate` | 0.3571 | 0.3286 | -0.0286 |
| `average_content_exact_match_rate` | 0.6124 | 0.6060 | -0.0064 |
| `average_level_exact_match_rate` | 0.7578 | 0.7422 | -0.0155 |
| `average_line_text_f1` | 0.8206 | 0.8092 | -0.0113 |
| `average_level_mae` | 0.2723 | 0.2630 | -0.0093 |
| `average_kind_sequence_match_rate` | 0.9143 | 0.9000 | -0.0143 |
| `average_semantic_key_f1` | 0.9552 | 0.9406 | -0.0147 |
| `average_prompt_requirement_f1` | 0.8531 | 0.8368 | -0.0163 |
| `prompt_requirement_exact_match_rate` | 0.5857 | 0.5714 | -0.0143 |
| `average_kubernetes_domain_validity_score` | 0.8310 | 0.8286 | -0.0024 |
| `kubernetes_domain_gate_pass_rate` | 0.1429 | 0.1286 | -0.0143 |
| `average_kubernetes_domain_validity_level` | 3.9000 | 3.9429 | +0.0429 |
| `average_bleu_score` | 0.7327 | 0.7277 | -0.0050 |
| `average_rougeL_f1` | 0.8462 | 0.8417 | -0.0045 |

La tabla muestra un efecto mixto. El DPO aumenta `parsed_equal_rate`, que pasa
de `0.1143` a `0.1429`, y mejora ligeramente el ajuste del numero de documentos.
Sin embargo, empeoran la parseabilidad YAML, el F1 de lineas, el cumplimiento
del prompt y varias metricas semanticas. La caida no es grande en valor absoluto,
pero afecta justo a las metricas que definen la robustez general de la salida.

## Validez Kubernetes Por Niveles

| Nivel Kubernetes | SFT | DPO | Delta |
| --- | ---: | ---: | ---: |
| `kubernetes_level_0_pass_rate` | 0.9857 | 0.9714 | -0.0143 |
| `kubernetes_level_1_pass_rate` | 0.9857 | 0.9714 | -0.0143 |
| `kubernetes_level_2_pass_rate` | 0.9714 | 0.9714 | +0.0000 |
| `kubernetes_level_3_pass_rate` | 0.9143 | 0.9571 | +0.0429 |
| `kubernetes_level_4_pass_rate` | 0.9000 | 0.9429 | +0.0429 |
| `kubernetes_level_5_pass_rate` | 0.1429 | 0.1286 | -0.0143 |

Esta descomposicion matiza la lectura global. Aunque el gate final baja, DPO
mejora los niveles `3` y `4`, que corresponden a invariantes intra-recurso e
inter-recurso. En otras palabras, el entrenamiento con preferencias automaticas
parece empujar al modelo hacia algunas estructuras Kubernetes mas coherentes,
pero no lo suficiente como para compensar los fallos adicionales de parseo YAML
y los requisitos de calidad estatica del nivel `5`.

## Perfil De Errores De Dominio

| Error Kubernetes | SFT | DPO | Delta |
| --- | ---: | ---: | ---: |
| `missing_resource_requirement` | 261 | 253 | -8 |
| `missing_run_as_non_root` | 71 | 69 | -2 |
| `missing_read_only_root_filesystem` | 71 | 69 | -2 |
| `latest_image_tag` | 67 | 65 | -2 |
| `volume_mount_without_volume` | 5 | 1 | -4 |
| `kubernetes_identity` | 1 | 0 | -1 |
| `yaml_parse` | 1 | 2 | +1 |
| `service_selector_without_workload` | 1 | 1 | 0 |

El perfil de errores refuerza la misma idea: DPO reduce varios errores
asociados a checks de dominio, especialmente `volume_mount_without_volume`, pero
introduce un fallo adicional de parseo YAML. Ese fallo pesa mucho porque la
parseabilidad es la primera barrera de evaluacion y condiciona toda la lectura
posterior.

## Analisis Por Muestra

La comparacion por `unit_id` muestra que la mayor parte del split no cambia:

- `parsed_equal_to_reference`: mejora en `2` muestras, no empeora en ninguna.
- `yaml_parse_ok`: empeora en `1` muestra.
- `line_text_f1`: mejora en `11`, empeora en `6`, queda igual en `53`.
- `prompt_requirement_f1`: mejora en `1`, empeora en `2`, queda igual en `67`.
- `kubernetes_domain_validity_score`: mejora en `4`, empeora en `1`, queda
  igual en `65`.
- `kubernetes_domain_validity_level`: mejora en `4`, empeora en `1`, queda
  igual en `65`.

La regresion mas importante aparece en `q140::question_simplified`. En SFT esta
muestra parseaba y pasaba el gate de dominio, mientras que en DPO falla YAML.
Si se excluye solo esa muestra, el DPO queda empatado con SFT en parseabilidad
YAML y mejora el `average_kubernetes_domain_validity_score` de `0.8285` a
`0.8406`, con un aumento del nivel medio de validez de `3.8841` a `4.0145`.
Este analisis no debe usarse para ocultar la regresion, pero si ayuda a
identificar que la conclusion global esta muy afectada por un caso negativo
concreto.

## Interpretacion

A primera vista, DPO deberia beneficiar a este proyecto porque el conjunto de
preferencias fue construido precisamente a partir de senales automaticas de
validez estructural, adecuacion al prompt y dominio Kubernetes. Sin embargo,
este primer experimento muestra que la relacion entre preferencia automatica y
calidad final no es lineal. El modelo aprende algunas preferencias utiles, sobre
todo en invariantes de dominio, pero ese desplazamiento tambien puede modificar
la superficie generada de manera suficiente como para perder estabilidad en
parseo y en fidelidad al prompt.

Esto es importante para la tesis porque situa DPO en su lugar metodologico
correcto. No funciona aqui como una etapa que simplemente mejora al SFT, sino
como una herramienta de alineamiento que repondera el comportamiento del modelo.
Cuando la preferencia automatica enfatiza estructura y dominio, puede mejorar
algunas propiedades internas del manifiesto, pero tambien puede penalizar la
regularidad superficial que el SFT ya habia aprendido muy bien. En un problema
donde la salida debe pasar por un parser, esa regularidad superficial no es un
detalle secundario: es parte de la condicion de posibilidad de la validacion.

La conclusion prudente es que `beta=0.10` produce un resultado mixto y no debe
adoptarse como nueva referencia principal frente a `serialized_sft`. El SFT
sigue siendo el control mas fuerte en esta comparacion. DPO, aun asi, deja una
senal experimental valiosa: los niveles Kubernetes intermedios mejoran y varios
errores de dominio se reducen. El siguiente paso no deberia ser descartar DPO,
sino hacerlo mas conservador o mejorar el balance de las preferencias para que
la optimizacion no degrade parseabilidad, contenido y cumplimiento del prompt.

## Limitaciones

- La evaluacion es sobre `validation`, no sobre `test`.
- El DPO solo se probo con `beta=0.10`; no hay todavia una curva de sensibilidad
  sobre `beta`.
- La comparacion usa una unica semilla de entrenamiento DPO.
- Las preferencias son automaticas y proxy; no equivalen a un conjunto de
  preferencias humanas.
- Las metricas de dominio Kubernetes siguen siendo checks automaticos propios y
  aproximados. No sustituyen una validacion formal completa contra el schema de
  Kubernetes.
- Durante el cierre de W&B aparecieron avisos de steps no monotonos para algunos
  logs finales. Los artefactos locales (`metrics.json`, `state.json` y
  `validation_predictions.jsonl`) si quedaron correctamente escritos.

## Siguientes Pasos

1. Revisar manualmente `q140::question_simplified`, porque concentra la mayor
   regresion del experimento.
2. Ejecutar una variante mas conservadora de DPO, por ejemplo con menor learning
   rate, menos pasos efectivos o early stopping a partir de una validacion
   intermedia.
3. Probar otros valores de `beta`, especialmente menores o cercanos a `0.05`,
   para reducir el desplazamiento respecto al SFT.
4. Revisar la funcion de preferencia para reforzar la penalizacion de salidas no
   parseables y de degradaciones fuertes en requisitos del prompt.
5. Mantener `serialized_sft` como referencia principal hasta que DPO demuestre
   una mejora estable en parseabilidad, fidelidad al prompt y dominio.

## Conclusion

El primer experimento completo de DPO con `beta=0.10` no supera globalmente al
SFT serializado. Mejora algunas senales relevantes de validez Kubernetes, en
particular los niveles `3` y `4`, pero empeora ligeramente la estabilidad de
parseo, el F1 de lineas y la adecuacion al prompt. La conclusion experimental
mas util no es que DPO falle, sino que esta configuracion todavia no esta bien
balanceada para el objetivo del proyecto: mejorar la estructura sin romper la
superficie parseable que hace posible la reconstruccion segura del YAML.
