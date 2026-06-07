# Comparacion DPO beta 0.30 frente a beta 0.10 y SFT - 2026-05-31

Document type: analysis

## Resumen

Este documento registra la lectura comparativa del experimento DPO con
`beta=0.30`, usando las predicciones de validacion generadas para la run
`dpo-beta030-full-20260530-130338`.

Por decision experimental, las `42` muestras generadas en validacion se tratan
como el resultado final disponible de esta run. El archivo principal
`metrics.json` se ha dejado con formato de validacion final, y la comparacion
detallada queda tambien en:

`results/dpo_kubernetes_v1/training/dpo-beta030-full-20260530-130338/comparison_metrics.json`

La comparacion mas justa se hace sobre los mismos `42` `unit_id` evaluados por
`beta=0.30`, filtrando tambien el DPO `beta=0.10` y el SFT serializado a ese
mismo subconjunto.

## Artefactos

- DPO `beta=0.30`:
  `results/dpo_kubernetes_v1/training/dpo-beta030-full-20260530-130338/`
- DPO `beta=0.10`:
  `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/`
- SFT serializado:
  `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/`

## Comparacion Sobre Las Mismas 42 Muestras

| Metrica | SFT | DPO beta 0.10 | DPO beta 0.30 | Delta 0.30 - 0.10 | Delta 0.30 - SFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| `yaml_parse_success_rate` | 0.9762 | 0.9762 | 0.9762 | +0.0000 | +0.0000 |
| `block_parse_success_rate` | 0.9762 | 0.9762 | 0.9762 | +0.0000 | +0.0000 |
| `parsed_equal_rate` | 0.1429 | 0.1667 | 0.1667 | +0.0000 | +0.0238 |
| `average_line_text_f1` | 0.8261 | 0.8300 | 0.8297 | -0.0003 | +0.0036 |
| `average_level_exact_match_rate` | 0.7796 | 0.7788 | 0.7780 | -0.0008 | -0.0016 |
| `average_level_mae` | 0.2530 | 0.2426 | 0.2443 | +0.0017 | -0.0088 |
| `average_prompt_requirement_f1` | 0.8066 | 0.8104 | 0.8104 | +0.0000 | +0.0038 |
| `average_semantic_key_f1` | 0.9586 | 0.9600 | 0.9600 | +0.0000 | +0.0014 |
| `average_kubernetes_domain_validity_score` | 0.7937 | 0.8135 | 0.8135 | +0.0000 | +0.0198 |
| `kubernetes_domain_gate_pass_rate` | 0.0238 | 0.0238 | 0.0238 | +0.0000 | +0.0000 |
| `average_kubernetes_domain_validity_level` | 3.6429 | 3.8571 | 3.8571 | +0.0000 | +0.2143 |

## Lectura Principal

El resultado mas claro es que `beta=0.30` no cambia casi nada frente a
`beta=0.10` en las muestras comparables. En la mayoria de metricas importantes
ambos DPO empatan exactamente, y donde no empatan la diferencia es minima:
`beta=0.30` queda ligeramente por debajo en `average_line_text_f1`,
`average_level_exact_match_rate` y `average_level_mae`.

Frente al SFT, los dos DPO si dejan una senal positiva en este subconjunto:
sube `parsed_equal_rate`, mejora levemente `line_text_f1`, mejora el ajuste al
prompt, y aumenta bastante el nivel medio de validez Kubernetes. La mejora mas
interpretable es que DPO reduce algunos errores de dominio intermedio frente al
SFT, no que produzca un salto general en correccion final.

La parte negativa es importante: el `kubernetes_domain_gate_pass_rate` no mejora.
Se queda en `0.0238`, igual que SFT y que `beta=0.10` en estas mismas muestras.
Es decir, DPO mueve algunas senales internas de dominio, pero no consigue que
mas ejemplos pasen el gate completo de Kubernetes.

## Contexto Con Validacion Completa De 70 Muestras

Como contexto, en la validacion completa de `70` muestras:

| Metrica | SFT 70 | DPO beta 0.10 70 |
| --- | ---: | ---: |
| `yaml_parse_success_rate` | 0.9857 | 0.9714 |
| `parsed_equal_rate` | 0.1143 | 0.1429 |
| `average_line_text_f1` | 0.8206 | 0.8092 |
| `average_level_exact_match_rate` | 0.7578 | 0.7422 |
| `average_prompt_requirement_f1` | 0.8531 | 0.8368 |
| `average_kubernetes_domain_validity_score` | 0.8310 | 0.8286 |
| `kubernetes_domain_gate_pass_rate` | 0.1429 | 0.1286 |

Esta tabla explica por que conviene ser prudente: sobre el split completo,
`beta=0.10` no superaba globalmente al SFT. En cambio, sobre las `42` muestras
que llegaron a generarse para `beta=0.30`, DPO parece algo mejor que SFT en
varias senales. La diferencia sugiere que el subconjunto de `42` muestras no es
necesariamente representativo del split completo.

## Conclusion

`beta=0.30` no aporta una mejora clara frente a `beta=0.10`. La configuracion
mas agresiva no degrada fuertemente las metricas comparables, pero tampoco
produce el salto que buscabamos. En las muestras compartidas, el efecto de DPO
parece venir de aplicar DPO en si, no de subir `beta` de `0.10` a `0.30`.

La conclusion experimental mas util es:

- DPO puede mejorar senales intermedias de estructura y dominio frente al SFT.
- El aumento de `beta` a `0.30` no mejora de forma clara el resultado.
- El gate Kubernetes sigue siendo el cuello de botella.
- Para el siguiente experimento conviene tocar la funcion de preferencia o la
  composicion de errores penalizados antes que seguir subiendo `beta`.

## Diagnostico De Igualdad Entre beta 0.10 Y beta 0.30

Se compararon directamente las predicciones de validacion de:

- `results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/validation_predictions.jsonl`
- `results/dpo_kubernetes_v1/training/dpo-beta030-full-20260530-130338/validation_predictions.jsonl`

El objetivo era distinguir si el empate entre `beta=0.10` y `beta=0.30` era solo
un empate de metricas agregadas o si las salidas generadas eran realmente las
mismas.

Resultado:

| Diagnostico | Valor |
| --- | ---: |
| Filas de validacion `beta=0.10` | 70 |
| Filas de validacion `beta=0.30` | 42 |
| `unit_id` comunes | 42 |
| Salidas `raw_model_output` exactamente iguales | 41 |
| Bloques predichos exactamente iguales | 41 |
| Metricas trazadas exactamente iguales | 41 |
| Unidades diferentes | 1 |

La unica diferencia aparece en `q41::question`. En esa muestra, `beta=0.30`
genera una salida algo mas larga, introduce `terminationGracePeriodSeconds: 30`,
cambia `hostPath.type` de `File` a `Socket` y desplaza algunos niveles de
`volumeMounts` y `volumes`. La diferencia no cambia las senales de prompt ni de
validez Kubernetes:

| Metrica en `q41::question` | beta 0.10 | beta 0.30 | Delta 0.30 - 0.10 |
| --- | ---: | ---: | ---: |
| `line_text_f1` | 0.8125 | 0.8000 | -0.0125 |
| `level_exact_match_rate` | 0.8276 | 0.7931 | -0.0345 |
| `level_mae` | 0.2759 | 0.3448 | +0.0690 |
| `prompt_requirement_f1` | 1.0000 | 1.0000 | +0.0000 |
| `kubernetes_domain_validity_score` | 0.8333 | 0.8333 | +0.0000 |
| `kubernetes_domain_validity_level` | 4 | 4 | +0 |
| `kubernetes_domain_gate_pass` | false | false | +0 |

Esta comprobacion cambia la interpretacion del empate: no estamos viendo dos
politicas que llegan a metricas parecidas por caminos distintos, sino casi la
misma salida en casi todos los ejemplos comparables. Por tanto, el siguiente
paso no deberia ser seguir explorando valores de `beta` sobre el mismo dataset
de preferencias. La palanca mas informativa pasa a ser la construccion de un
dataset de preferencias `v2` mas discriminativo, especialmente en errores de
nivel 5 y en negativos duros que ya sean parseables y plausibles.
