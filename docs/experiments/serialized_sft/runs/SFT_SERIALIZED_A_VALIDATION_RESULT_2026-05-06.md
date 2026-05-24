# Resultado de validacion SFT Arquitectura A - 2026-05-06

## Resumen

El primer entrenamiento completo de la Arquitectura A (`serialized_sft`) obtiene
un resultado muy fuerte en `validation` para Kubernetes v1. El modelo entrenado
como LM causal estandar con LoRA aprende a generar la superficie estructurada
`blocks_tsv_v1`, y el parser existente reconstruye YAML valido en casi todos los
casos evaluados.

Este resultado no cierra la comparacion principal de la tesis, porque
`two_head_sft` sigue siendo la rama que debe probar si modelar `level` como senal
estructural explicita mejora a la serializacion plana. Pero si fija un control
supervisado mucho mas competitivo de lo esperado: la rama serializada ya no es un
baseline debil, sino una referencia fuerte para las siguientes arquitecturas.

## Artefactos

- Run SFT: `serialized-sft-a-v1-20260505-171226`
- Directorio: `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/`
- Variante: `serialized_sft`
- Serializacion objetivo: `blocks_tsv_v1`
- Checkpoint seleccionado: `checkpoint-step-159`
- Split evaluado: `validation`
- Ejemplos evaluados: `70/70`
- Proyecto W&B: `llm-structured-semantic-generation`

Baseline de referencia:

- Run baseline: `compact-validation70-320-vtfix`
- Directorio: `results/baseline_kubernetes_v1/compact-validation70-320-vtfix/`
- Serializacion baseline: `blocks_tsv_compact_v1`
- Ejemplos evaluados: `65/70`

La comparacion es informativa pero debe leerse con cuidado: el baseline usa
`blocks_tsv_compact_v1` y el SFT usa `blocks_tsv_v1`. Aun asi, ambos pasan por
el mismo enfoque de representacion intermedia de bloques, reconstruccion con
parser y evaluacion estructural.

## Metricas principales

| Metrica | Baseline validation | SFT A validation | Mejora |
| --- | ---: | ---: | ---: |
| `structured_output_parse_success_rate` | 0.9286 | 1.0000 | +0.0714 |
| `yaml_parse_success_rate` | 0.3077 | 0.9857 | +0.6780 |
| `block_parse_success_rate` | 0.3077 | 0.9857 | +0.6780 |
| `average_line_text_f1` | 0.2086 | 0.8206 | +0.6119 |
| `average_level_exact_match_rate` | 0.1131 | 0.7578 | +0.6447 |
| `average_level_mae` | 0.7896 | 0.2723 | -0.5173 |
| `average_prompt_requirement_f1` | 0.2213 | 0.8531 | +0.6318 |
| `average_semantic_key_f1` | 0.2778 | 0.9552 | +0.6774 |
| `average_required_field_complete_resource_rate` | 0.7059 | 1.0000 | +0.2941 |
| `required_field_complete_sample_rate` | 0.7059 | 1.0000 | +0.2941 |
| `average_kind_sequence_match_rate` | 0.2000 | 0.9143 | +0.7143 |
| `primary_kind_match_rate` | 0.6500 | 0.9275 | +0.2775 |
| `primary_api_version_match_rate` | 0.6500 | 0.9275 | +0.2775 |
| `primary_metadata_name_match_rate` | 0.6500 | 0.8986 | +0.2486 |

Las metricas de seleccion del checkpoint tambien son muy altas:

- `yaml_parse_success_rate`: `0.9857`
- `average_prompt_requirement_f1`: `0.8531`
- `average_required_field_complete_resource_rate`: `1.0000`
- `average_line_text_f1`: `0.8206`

## Lectura del resultado

El resultado es especialmente relevante porque ataca directamente el problema
central del proyecto: un modelo autoregresivo plano debe producir una salida que
depende de una jerarquia. En esta rama, la jerarquia no se predice con una cabeza
estructural separada, sino que se serializa como texto dentro de `blocks_tsv_v1`.

La mejora sobre el baseline sugiere que el SFT con LoRA consigue alinear muy bien
al modelo con la superficie estructurada esperada por el parser:

- la salida estructurada es parseable en todos los ejemplos;
- el YAML reconstruido es sintacticamente valido en casi todo el split;
- la prediccion de `level` mejora de forma muy marcada;
- las claves semanticas y los requisitos del prompt mejoran mucho;
- los campos requeridos aparecen completos en todos los recursos evaluados.

Esto valida la utilidad de Arquitectura A como control supervisado. No es solo
un paso previo tecnico para llegar a `two_head_sft`: ya demuestra que una
representacion serializada de la jerarquia puede ser una estrategia fuerte para
generacion estructurada con parser.

## Lo que todavia no demuestra

Aunque el resultado es muy bueno, no debe interpretarse como una prueba completa
de correccion Kubernetes.

Limitaciones importantes:

- La evaluacion es sobre `validation`, no sobre `test`.
- Las metricas semanticas actuales son aproximadas y derivadas de checks
  automaticos propios; no sustituyen una validacion formal contra el schema de
  Kubernetes.
- `parsed_equal_rate` sigue siendo bajo (`0.1143`), asi que el modelo no esta
  reproduciendo exactamente los manifiestos objetivo. Esto no contradice el
  objetivo principal, porque el exito del proyecto prioriza validez estructural,
  adecuacion al prompt y reconstruccion segura sobre igualdad textual exacta.
- La comparacion con el baseline no es perfectamente limpia en superficie de
  salida, porque el baseline reportado usa `blocks_tsv_compact_v1` y el SFT usa
  `blocks_tsv_v1`.
- Durante el entrenamiento se omitieron 2 microbatches por recuperacion de OOM:
  `q229::question` y `q229::question_simplified`. La omision esta registrada en
  `oom_skipped_batches.jsonl` y debe reportarse como condicion operacional del
  experimento.

## Implicacion para la comparacion `serialized_sft` vs `two_head_sft`

Este resultado sube el liston para `two_head_sft`. La rama con cabeza explicita
de `level` ya no compite contra un baseline serializado fragil, sino contra una
Arquitectura A que:

- reconstruye YAML valido casi siempre;
- predice niveles con una exactitud razonablemente alta;
- conserva bastante bien las claves semanticas;
- cumple los campos requeridos en todos los recursos evaluados;
- mejora de forma clara la adecuacion al prompt.

Por tanto, `two_head_sft` deberia justificarse no solo por mejorar una metrica
aislada, sino por demostrar alguna ventaja investigadora clara:

- mejor prediccion de `level`;
- menor error estructural en casos complejos;
- mejor generalizacion fuera del patron visto;
- mayor robustez en manifests multi-recurso;
- mejor eficiencia de muestra o estabilidad de entrenamiento;
- menor dependencia de que el modelo aprenda `level` como texto ordinario.

## Registro operacional

El entrenamiento completo se ejecuto en condiciones locales con memoria limitada.
La recuperacion de OOM permitio continuar sin descartar el run completo, pero
introduce una salvedad reproducible:

- politica usada: `--oom-recovery skip_batch`
- limite: `--max-oom-skips 8`
- skips observados: `2`
- estado final: `completed`
- `global_step`: `159`
- `epoch`: `3`

Esto confirma que el trainer resumible y tolerante a fallos es util para este
proyecto: el experimento pudo completarse a pesar de interrupciones, OOMs y
relanzamientos.

## Siguientes pasos recomendados

1. Revisar manualmente los errores de validacion, especialmente el ejemplo que
   falla en `yaml_parse_success_rate` y los casos con bajo `line_count_match`.
2. Mantener `test` reservado para evaluacion final y no usarlo para seleccionar
   checkpoints.
3. Ejecutar `two_head_sft` con la misma disciplina de artefactos, resume y
   evaluacion.
4. Anadir, si es viable, validacion externa tipo schema/Kubernetes para separar
   mejor validez sintactica, estructural y dominio Kubernetes real.
5. Usar este resultado como referencia fuerte para DPO posterior, no como cierre
   definitivo del ciclo experimental.

## Conclusion

Arquitectura A produce un resultado de validacion claramente positivo. El SFT
serializado convierte un baseline con baja reconstruccion YAML en un modelo que
genera casi siempre una representacion estructurada parseable, reconstruible y
alineada con buena parte de los requisitos del prompt.

La conclusion prudente es que `serialized_sft` funciona muy bien como control
supervisado para Kubernetes v1. La conclusion investigadora mas interesante es
que la serializacion explicita de la jerarquia ya captura gran parte del problema
estructural, por lo que cualquier arquitectura jerarquica posterior debera
demostrar una mejora real frente a una referencia fuerte.
