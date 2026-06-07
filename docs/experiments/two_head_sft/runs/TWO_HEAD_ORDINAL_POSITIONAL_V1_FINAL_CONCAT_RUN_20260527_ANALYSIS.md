# Analisis Del Run Ordinal Posicional V1 Final-Concat 2026-05-27

Document type: run result

## Resumen

Este documento registra el cierre manual del experimento
`two_head_ordinal_positional_v1` con positional encoding sinusoidal absoluto
inyectado por concatenacion antes de la lineal final del cabezal ordinal.

El entrenamiento llego al checkpoint final `checkpoint-step-159`. La evaluacion
final se interrumpio manualmente primero con 60/70 muestras, pero se reanudo el
2026-06-02 y quedo completada con las 70 muestras de validation. El artefacto
parcial se conserva como rastro historico, pero la interpretacion de este
documento usa `metrics.json` y la auditoria final completa.

La lectura principal es que el positional encoding V1 no corrige el colapso
inicial hacia `level=0`. En el primer cuartil del YAML, el modelo predice
`level=0` en el 94.7% de las lineas evaluadas, mientras que el gold contiene
`level=0` en el 62.2%. Ademas, la correlacion entre `line_position` y el score
ordinal `z` es alta, lo que encaja con la interpretacion de que la concatenacion
final esta actuando en gran parte como un sesgo temporal aditivo sobre el score,
no como una condicion que permita reinterpretar el hidden state.

## Estado Del Run

- Run id: `two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527`
- Model variant: `two_head_ordinal_positional_v1`
- Dataset: `data/processed/kubernetes_v1/`
- Serialization: `content_blocks_v1`
- W&B: `llm-structured-semantic-generation`
- Final checkpoint available: `checkpoint-step-159`
- Final validation status: completed at 70/70 validation predictions after resume.
- Full metrics artifact: `metrics.json`
- Historical partial metrics artifact: `metrics_partial.json`
- Partial stop reason recorded: `manual_stop_after_training_completed_partial_final_validation`
- Final state: `completed`

The process was stopped after the training phase had produced the final
checkpoint. The final validation pass was later resumed without additional
training: the run stayed at `global_step=159` and loaded `checkpoint-step-159`.

## Configuracion Relevante

The run keeps the same optimization choices as the previous ordinal centered
gap experiment:

- Base learning rate: `2e-4`
- Ordinal MLP LR multiplier: `3.0`
- Ordinal MLP LR: `6e-4`
- Threshold LR multiplier: `50.0`
- Threshold LR: `1e-2`
- Initial threshold gap: `0.5`
- Initial thresholds: `[-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]`
- Epochs: `3`
- Batch size: `1`
- Gradient accumulation: `8`
- Checkpoint every: `8` optimizer steps
- Checkpoint retention: keep all checkpoints
- Intermediate eval: 10 random validation samples at step 80

The ordinal positional head uses:

```text
hidden -> LayerNorm -> Linear(hidden, 512) -> GELU -> Dropout(0.10)
       -> Linear(512, 64) -> GELU -> LayerNorm(64)

line_position -> sinusoidal absolute PE, 16 dims -> LayerNorm(16)

concat(hidden_64, pos_16) -> Linear(80, 1) -> z
```

The positional branch has no dropout. The positional encoding is causal in the
sense that it uses the line index generated so far, not the final YAML length.

## Resultado Final

The following metrics are computed from the complete final validation artifact:

| Metric | Value |
| --- | ---: |
| Validation predictions completed | 70 / 70 |
| Evaluated samples | 69 |
| Structured output parse success rate | 0.9857 |
| YAML parse success rate | 0.0725 |
| Average level MAE | 1.0239 |
| Average level exact match rate | 0.0223 |
| Predicted max level mean | 4.3043 |
| Deep level exact recall 5-8 | 0.1379 |
| Deep level off-by-one recall 5-8 | 0.4966 |
| Compressed deep to 0-4 rate | 0.6966 |
| Kubernetes domain gate pass rate | 0.0000 |
| Kubernetes domain validity score | 0.0507 |

The level distribution remains strongly distorted:

| Level | Pred count | Gold count | Recall |
| ---: | ---: | ---: | ---: |
| 0 | 742 | 312 | 0.9455 |
| 1 | 153 | 269 | 0.0706 |
| 2 | 209 | 210 | 0.1381 |
| 3 | 196 | 299 | 0.1037 |
| 4 | 218 | 394 | 0.3528 |
| 5 | 111 | 93 | 0.2151 |
| 6 | 0 | 40 | 0.0000 |
| 7 | 0 | 3 | 0.0000 |
| 8 | 0 | 9 | 0.0000 |

The model overpredicts `level=0`, underpredicts `level=1`, and never predicts
levels `6`, `7`, or `8` in the complete validation subset.

## Distribucion Por Cuartiles

The audit over the complete final validation predictions shows the same temporal
pattern that motivated this experiment.

| Quartile | Lines | Pred 0 rate | Gold 0 rate | Gold > 0 predicted 0 | Mean pred level | Mean gold level | MAE | z mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | 455 | 0.9473 | 0.6220 | 0.3429 | 0.0725 | 0.5187 | 0.4813 | -3.4343 |
| Q2 | 423 | 0.5177 | 0.0662 | 0.4704 | 1.0071 | 2.2861 | 1.3830 | -1.7728 |
| Q3 | 436 | 0.1583 | 0.0093 | 0.1565 | 2.6193 | 3.4720 | 1.1776 | -0.6115 |
| Q4 | 396 | 0.0581 | 0.0123 | 0.0677 | 3.1616 | 3.8246 | 1.1846 | -0.3088 |

This is the most important diagnostic result. The first quartile is still
dominated by predicted `level=0`. The second quartile also has a large
`level=0` excess. The effect decreases later in the sequence, but does not
translate into reliable high-level prediction.

## Lectura Del Score Ordinal

The audit of the final projector indicates that the positional branch is not
being ignored:

| Quantity | Value |
| --- | ---: |
| Hidden projector norm | 0.5305 |
| Position projector norm | 0.2518 |
| Position/hidden norm ratio | 0.4747 |
| Hidden dims in final projector | 64 |
| Position dims in final projector | 16 |

The correlation diagnostics are:

| Correlation | Value |
| --- | ---: |
| Pearson `line_position -> z` | 0.8460 |
| Spearman `line_position -> z` | 0.8652 |
| Pearson `line_position -> pred_level` | 0.7738 |
| Pearson `line_position -> gold_level` | 0.7999 |
| Pearson `z -> gold_level` | 0.8588 |
| Pearson `pred_level -> gold_level` | 0.7830 |

This supports a cautious interpretation: the final-concat positional encoding
does affect the ordinal score, but the interaction is probably too late and too
linear. In this architecture, the final layer can express:

```text
z = w_hidden * h_64 + w_pos * pe(t) + b
```

That makes the positional signal a direct additive temporal bias over `z`.
It does not force the hidden projection to be interpreted differently depending
on the temporal region of the generated YAML.

## Position-Only Baseline

The audit script also fits simple position-only baselines on the dataset. These
baselines are not intended as model candidates, but as shortcut diagnostics.

| Baseline | Exact match | MAE | Pred 0 rate |
| --- | ---: | ---: | ---: |
| Absolute line position | 0.4933 | 0.9544 | 0.1556 |
| Fixed-width position bin | 0.3733 | 1.0822 | 0.1556 |

The position-only baseline captures part of the temporal regularity of the
dataset, but it does not reproduce the same first-quartile collapse as the V1
head. This suggests that the failure is not just a trivial dataset prior over
absolute line position. The current evidence is more compatible with a mismatch
between temporal hidden-state distributions and a late additive positional
correction.

## Interpretacion

This run should be considered a useful negative/intermediate result:

- It confirms that adding a simple sinusoidal positional vector by final
  concatenation is not enough to fix the early-sequence `level=0` collapse.
- It shows that the positional branch is used, but mainly as a temporal ramp on
  the ordinal score.
- It does not prove that the initial hidden states are intrinsically poorer.
- It remains consistent with the broader hypothesis of temporal
  non-stationarity in the latent space.
- It motivates moving from late positional concatenation to a real conditioning
  mechanism such as FiLM or gating.

The result should not be presented as evidence that positional information is
useless. The more precise conclusion is that this specific injection policy,
`final_concat`, is likely too weak or too late for the kind of positional drift
being studied.

## Limitaciones

- Final validation is complete, but one structured prediction failed to parse,
  so several metrics use 69
  evaluable samples.
- No normal/zero/shuffle positional ablation was run on the final checkpoint
  before closing the experiment.
- YAML parseability remains very low, so structural metrics should be read as
  diagnostics of the block-level prediction rather than as evidence of useful
  Kubernetes manifests.
- This result does not close the broader question of positional conditioning;
  it only closes V1 final-concat as the next main direction.

## Artefactos

- Run directory:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/`
- Metrics:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/metrics.json`
- Historical partial metrics:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/metrics_partial.json`
- Final checkpoint:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/checkpoints/checkpoint-step-159/`
- Final validation predictions:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/validation_predictions.jsonl`
- Final full audit:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/positional_v1_audit/checkpoint-step-159-full/`
- Historical partial audit:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/positional_v1_audit/checkpoint-step-159-partial/`
- Intermediate step-80 audit:
  `results/two_head_ordinal_positional_sft_kubernetes_v1/two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527/positional_v1_audit/checkpoint-step-80/`

## Siguiente Paso

The immediate follow-up is to keep this run as the V1 baseline and test a
variant where position modulates the hidden representation before the final
ordinal bottleneck. The planned V2 is:

```text
hidden -> 512 features
position -> sinusoidal PE -> FiLM/gating parameters
modulated hidden_512 -> 64 -> ordinal score z
```

The purpose of V2 is not to add more positional dimensions, but to test whether
feature-wise conditioning helps the level head interpret hidden states under
temporal distribution shift.
