# Resultado DPO beta 0.30 completo - 2026-06-10

Document type: run result

## Resumen

Este documento registra el cierre completo de la validacion de la run DPO v1
con `beta=0.30`.

La run ya tenia entrenamiento completo y un resultado parcial documentado sobre
`42` muestras. El 2026-06-10 se reanudo la evaluacion final desde el artefacto
append-only existente `validation_predictions.jsonl`, que contenia esas `42`
predicciones. La reanudacion salto los `unit_id` ya presentes y genero las `28`
muestras restantes del split `validation`.

El cierre final queda en `70/70` muestras, sin duplicados de `unit_id`, con
`state.status = completed` y `metrics.json` actualizado.

## Artefactos

- Run DPO:
  `results/dpo_kubernetes_v1/training/dpo-beta030-full-20260530-130338/`
- Checkpoint evaluado:
  `checkpoint-step-57`
- Metricas finales:
  `results/dpo_kubernetes_v1/training/dpo-beta030-full-20260530-130338/metrics.json`
- Predicciones finales:
  `results/dpo_kubernetes_v1/training/dpo-beta030-full-20260530-130338/validation_predictions.jsonl`
- Log de reanudacion:
  `results/dpo_kubernetes_v1/training/logs/dpo-beta030-full-20260530-130338.resume-validation-20260610-171937.out.log`
- W&B:
  `https://wandb.ai/mario-garcia-berenguer-universidad-polit-cnica-de-madrid/llm-structured-semantic-generation/runs/dpo-beta030-full-20260530-130338`

Comando de referencia:

```powershell
uv run python scripts\train_kubernetes_dpo.py `
  --run-id dpo-beta030-full-20260530-130338 `
  --beta 0.30 `
  --checkpoint-steps 32 `
  --two-thirds-validation-samples 10 `
  --wandb-mode online `
  --wandb-run-name dpo-beta030-full-20260530-130338 `
  --wandb-tags dpo-full,beta-0.30,full-validation,two-thirds-validation,resume-validation `
  --validation-log-every 1
```

## Comprobacion De Cierre

| Comprobacion | Valor |
| --- | ---: |
| Predicciones en `validation_predictions.jsonl` | 70 |
| `unit_id` unicos | 70 |
| `unit_id` duplicados | 0 |
| `metrics.row_count` | 70 |
| `metrics.evaluated_count` | 70 |
| `state.status` | `completed` |
| `state.global_step` | 57 |

## Comparacion Completa Sobre Validation

| Metrica | SFT | DPO beta 0.10 | DPO beta 0.30 | Delta 0.30 - 0.10 | Delta 0.30 - SFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| `row_count` | 70 | 70 | 70 | 0 | 0 |
| `yaml_parse_success_rate` | 0.9857 | 0.9714 | 0.9857 | +0.0143 | +0.0000 |
| `block_parse_success_rate` | 0.9857 | 0.9714 | 0.9857 | +0.0143 | +0.0000 |
| `parsed_equal_rate` | 0.1143 | 0.1429 | 0.1429 | +0.0000 | +0.0286 |
| `average_line_text_f1` | 0.8206 | 0.8092 | 0.8193 | +0.0101 | -0.0013 |
| `average_level_exact_match_rate` | 0.7578 | 0.7422 | 0.7495 | +0.0073 | -0.0082 |
| `average_level_mae` | 0.2723 | 0.2630 | 0.2747 | +0.0117 | +0.0024 |
| `average_prompt_requirement_f1` | 0.8531 | 0.8368 | 0.8511 | +0.0143 | -0.0020 |
| `average_semantic_key_f1` | 0.9552 | 0.9406 | 0.9548 | +0.0143 | -0.0004 |
| `average_kubernetes_domain_validity_score` | 0.8310 | 0.8286 | 0.8429 | +0.0143 | +0.0119 |
| `kubernetes_domain_gate_pass_rate` | 0.1429 | 0.1286 | 0.1429 | +0.0143 | +0.0000 |
| `average_kubernetes_domain_validity_level` | 3.9000 | 3.9429 | 4.0286 | +0.0857 | +0.1286 |
| `kubernetes_level_3_pass_rate` | 0.9143 | 0.9571 | 0.9714 | +0.0143 | +0.0571 |
| `kubernetes_level_4_pass_rate` | 0.9000 | 0.9429 | 0.9571 | +0.0143 | +0.0571 |
| `kubernetes_level_5_pass_rate` | 0.1429 | 0.1286 | 0.1429 | +0.0143 | +0.0000 |
| `average_bleu_score` | 0.7327 | 0.7277 | 0.7273 | -0.0004 | -0.0054 |
| `average_rougeL_f1` | 0.8462 | 0.8417 | 0.8412 | -0.0005 | -0.0049 |

## Lectura Principal

La validacion completa cambia la lectura provisional de `beta=0.30`. En el
subconjunto de `42` muestras, `beta=0.30` parecia casi indistinguible de
`beta=0.10` y el gate Kubernetes quedaba en `0.0238`. En el cierre sobre
`70/70`, `beta=0.30` alcanza `kubernetes_domain_gate_pass_rate = 0.1429`, igual
que el SFT completo y por encima de `beta=0.10`.

Frente a `beta=0.10`, `beta=0.30` mejora parseabilidad, F1 de lineas, exactitud
de niveles, F1 de prompt, senales semanticas y validez Kubernetes. Frente al
SFT, la lectura sigue siendo mixta: mejora `parsed_equal_rate` y las metricas de
validez Kubernetes, especialmente niveles `3` y `4`, pero queda ligeramente por
debajo en F1 de lineas, exactitud de niveles, MAE de niveles, F1 de prompt y
metricas lexicas.

La conclusion prudente es que `beta=0.30` es mejor que `beta=0.10` en la
validacion completa y recupera el gate final frente al SFT, pero no demuestra
una mejora global clara sobre el SFT serializado. Su efecto mas interesante
esta en las senales de dominio Kubernetes y en el nivel medio de validez, no en
la fidelidad textual o de niveles.

## Limitaciones

- La evaluacion es sobre `validation`, no sobre `test`.
- Las preferencias siguen siendo automaticas y proxy; no equivalen a
  preferencias humanas.
- Las metricas de dominio Kubernetes son checks automaticos propios y
  aproximados, no validacion oficial completa contra schema Kubernetes.
- La reanudacion escribio la validacion restante sobre la misma run y el mismo
  archivo append-only, por lo que el resultado parcial de `42` muestras debe
  leerse como historico, no como resultado final.

