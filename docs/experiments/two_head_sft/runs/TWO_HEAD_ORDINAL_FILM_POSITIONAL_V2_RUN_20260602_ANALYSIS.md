# Analisis Del Run Ordinal FiLM Posicional V2 2026-06-02

Document type: run result

## Resumen

Este documento registra el cierre del experimento
`two_head_ordinal_film_positional_v2`, lanzado como continuacion del analisis
de drift posicional latente. La variante sustituye la inyeccion posicional
`final_concat` del experimento V1 por una modulacion FiLM sobre el cuello de
botella de 512 features del cabezal ordinal.

El entrenamiento completo llego a `global_step=159` tras 3 epocas y quedo
guardado en `checkpoint-step-159`. La validacion final fue interrumpida
manualmente cuando llevaba 20/70 muestras y se reanudo despues desde el mismo
run, reutilizando `validation_predictions.jsonl`. La validacion final termino
con 70/70 muestras y el estado del run quedo como `completed`.

La lectura principal es mixta. FiLM mejora la parseabilidad YAML respecto a los
experimentos ordinales posicionales anteriores y mantiene el contrato
parser-facing `content_blocks_v1` con parseabilidad estructural completa. Sin
embargo, no resuelve el problema central del cabezal de `level`: el primer
cuartil sigue colapsando casi por completo a `level=0`, y los niveles profundos
siguen comprimidos hacia el rango `0-4`. Por tanto, este run no demuestra que el
condicionamiento posicional FiLM sea suficiente; si acaso, refuerza la idea de
que la posicion temporal esta siendo explotada como senal fuerte, pero todavia
no como una condicion rica que permita reinterpretar correctamente el hidden
state.

## Estado Del Run

- Run id: `two-head-ordinal-film-positional-v2-512-64-gap05-mlp-lr3-threshold-lr50-20260602`
- Model variant: `two_head_ordinal_film_positional_v2`
- Dataset: `data/processed/kubernetes_v1/`
- Serialization: `content_blocks_v1`
- Level alignment policy: `record_prefix_state`
- W&B project: `llm-structured-semantic-generation`
- Final checkpoint: `checkpoint-step-159`
- Final validation: `70/70`
- Final state: `completed`
- Completed at: `2026-06-03T13:07:40.668187Z`

The local checkpoint directory at close contains `checkpoint-step-159`. During
training, intermediate checkpoints were observed at steps 32, 64, 96 and 128,
but the final local artifact available after resume is the final checkpoint.
The final checkpoint is therefore the source of truth for future evaluation or
resume work.

## Configuracion Relevante

The run keeps the same optimization choices as the previous centered gap05
ordinal experiment and the positional V1 run:

- Base learning rate: `2e-4`
- Ordinal MLP LR multiplier: `3.0`
- Ordinal MLP LR: `6e-4`
- Threshold LR multiplier: `50.0`
- Threshold LR: `1e-2`
- Initial threshold center: `0.0`
- Initial threshold gap: `0.5`
- Initial thresholds: `[-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]`
- Epochs: `3`
- Batch size: `1`
- Gradient accumulation: `8`
- Checkpoint interval requested: every `32` optimizer steps
- Intermediate eval: `10` random validation samples at step `107`
- OOM recovery: `skip_batch`
- Lambda level: `2.0`

The ordinal positional head is:

```text
hidden -> LayerNorm -> Linear(hidden, 512) -> GELU -> Dropout(0.10)

line_position -> sinusoidal absolute PE, 16 dims
              -> LayerNorm
              -> Linear(16, 128) -> GELU -> Linear(128, 1024)
              -> gamma, beta

FiLM(hidden_512) = hidden_512 * (1 + 0.10 * gamma) + 0.10 * beta

FiLM(hidden_512) -> Linear(512, 64) -> GELU -> LayerNorm(64)
                 -> Linear(64, 1) -> ordinal score z
```

The positional branch has no dropout. The position used is the causal absolute
line index available at generation time; it does not require the final document
length.

## Entrenamiento

The run completed training without fatal errors:

| Field | Value |
| --- | ---: |
| Final global step | 159 |
| Final epoch state | 3 |
| Train log rows | 159 |
| OOM skipped batches | 2 |
| Final loss | 0.1820 |
| Final LM loss | 0.0713 |
| Final ordinal level loss | 0.0554 |
| Final base LR | 0.000003797 |
| Final ordinal MLP LR | 0.000011392 |
| Final threshold LR | 0.000189873 |

The two OOM events were handled by `skip_batch`. They occurred during training,
not during the resumed final validation. The final validation was resumed
successfully from the partial `validation_predictions.jsonl` artifact.

Final train-time thresholds:

| Threshold | Value |
| --- | ---: |
| tau_0 | -1.7274 |
| tau_1 | -1.1812 |
| tau_2 | -0.6081 |
| tau_3 | -0.0107 |
| tau_4 | 0.6175 |
| tau_5 | 1.2805 |
| tau_6 | 1.9663 |
| tau_7 | 2.6443 |

The final batch diagnostic still showed a very low `z` distribution
(`ordinal_z_mean=-2.6797`), with that batch predicting only levels 0 and 1.
This should be read as a local train-batch diagnostic, not as the full
validation distribution.

## Validacion Intermedia

The intermediate validation at step 107 used 10 random validation samples:

| Metric | Value |
| --- | ---: |
| Structured output parse success rate | 1.0000 |
| YAML parse success rate | 0.1000 |
| Average level MAE | 0.6471 |
| Deep level exact recall 5-8 | 0.2917 |
| Compressed deep to 0-4 rate | 0.5000 |
| Predicted max level mean | 4.7000 |

This intermediate result looked somewhat promising on `level_mae` and deep
recall, but it was only a 10-sample diagnostic. The full validation gives a
more conservative reading.

## Resultado Final

Final metrics from `metrics.json`:

| Metric | Value |
| --- | ---: |
| Evaluated count | 70 |
| Structured output parse success rate | 1.0000 |
| YAML parse success rate | 0.1286 |
| Block parse success rate | 0.1286 |
| Parsed equal rate | 0.0000 |
| Document count match rate | 0.8429 |
| Line count match rate | 0.2857 |
| Average valid block ratio | 1.0000 |
| Average indentation leak rate | 0.0000 |
| Average block count error | 3.8714 |
| Average content exact match rate | 0.0489 |
| Average line text F1 | 0.0782 |
| Average level exact match rate | 0.0278 |
| Average level MAE | 0.8430 |
| Primary kind match rate | 0.5556 |
| Primary API version match rate | 0.5556 |
| Primary metadata name match rate | 0.0000 |
| Prompt requirement F1 | 0.0799 |
| Kubernetes domain validity score | 0.1000 |
| Kubernetes domain gate pass rate | 0.0000 |
| BLEU | 0.6711 |
| ROUGE-1 F1 | 0.8374 |
| ROUGE-2 F1 | 0.7436 |
| ROUGE-L F1 | 0.7782 |

The high BLEU/ROUGE values should not be overinterpreted. In this project,
surface overlap is secondary to parser-facing structural validity, level
prediction, and Kubernetes-domain checks.

## Distribucion Global De Levels

| Level | Gold count | Pred count | Recall | Precision |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 312 | 716 | 0.9487 | 0.4134 |
| 1 | 269 | 179 | 0.0892 | 0.1341 |
| 2 | 213 | 220 | 0.1690 | 0.1636 |
| 3 | 301 | 243 | 0.1030 | 0.1276 |
| 4 | 401 | 189 | 0.2993 | 0.6349 |
| 5 | 88 | 89 | 0.1932 | 0.1910 |
| 6 | 40 | 0 | 0.0000 | 0.0000 |
| 7 | 3 | 0 | 0.0000 | 0.0000 |
| 8 | 9 | 0 | 0.0000 | 0.0000 |

The model predicts many more zeros than the reference distribution and never
predicts levels 6, 7 or 8 in final validation. This is the clearest sign that
FiLM V2 has not solved the deep-level compression problem.

Deep-level summary:

| Metric | Value |
| --- | ---: |
| Deep level support 5-8 | 140 |
| Deep level exact recall 5-8 | 0.1214 |
| Deep level off-by-one recall 5-8 | 0.4143 |
| Compressed deep to 0-4 rate | 0.7429 |
| Predicted max level mean | 4.1571 |
| Target max level >= 5 count | 46 |
| YAML parse success rate when target max level >= 5 | 0.1522 |

The off-by-one deep recall is substantially higher than exact deep recall,
which suggests that the model sometimes approaches the correct depth band, but
the ordinal boundary calibration is still not reliable enough for exact
hierarchy reconstruction.

## Analisis Por Cuartiles Del YAML

This table aligns each predicted line with the corresponding gold line when
available and assigns the line to a quartile by relative position inside its
own target sequence.

| Quartile | Lines | MAE | Exact | Gold level 0 rate | Pred level 0 rate | Gold 5-8 | Pred 5-8 | z mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | 436 | 0.4404 | 0.6284 | 0.6468 | 0.9541 | 0 | 0 | -3.3003 |
| Q2 | 403 | 1.3871 | 0.1290 | 0.0496 | 0.5335 | 0 | 0 | -1.7917 |
| Q3 | 415 | 1.1783 | 0.2627 | 0.0193 | 0.1470 | 39 | 22 | -0.7254 |
| Q4 | 382 | 1.2723 | 0.2330 | 0.0052 | 0.0628 | 101 | 67 | -0.2323 |

The first quartile still shows the original failure mode: `level=0` is predicted
in 95.41% of aligned lines, while gold `level=0` appears in 64.68%. This is not
just a small calibration issue; the head is still very strongly biased toward
the shallowest class at the beginning of the generated structure.

At the same time, the score `z` rises with line position:

| Correlation | Value |
| --- | ---: |
| Absolute line position -> z, Pearson | 0.8442 |
| Absolute line position -> z, Spearman | 0.8647 |
| Absolute line position -> predicted level, Pearson | 0.7734 |
| Predicted level -> gold level, Pearson | 0.7816 |

This supports the interpretation that the model has learned a strong temporal
ramp. That ramp is useful enough to improve some parser-facing metrics, but it
does not yet separate structural level from generation time.

## Comparacion Con Experimentos Previos

| Run | Evaluated | Structured parse | YAML parse | Level MAE | Level exact | Deep exact 5-8 | Deep compressed to 0-4 | Pred max level mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinal density centered gap05 | 70 | 1.0000 | 0.0286 | 0.4559 | 0.0107 | 0.2174 | 0.6014 | 4.4857 |
| Positional V1 final-concat | 69 | 0.9857 | 0.0725 | 1.0239 | 0.0223 | 0.1379 | 0.6966 | 4.3043 |
| FiLM positional V2 | 70 | 1.0000 | 0.1286 | 0.8430 | 0.0278 | 0.1214 | 0.7429 | 4.1571 |

FiLM V2 is better than V1 in YAML parseability and level MAE, but it is worse
than the previous non-positional centered density run on `average_level_mae`,
deep exact recall and deep compression. Therefore, the result should not be
read as a clear modeling win. The useful conclusion is narrower: FiLM is a more
expressive positional conditioning mechanism than final concatenation, but this
particular configuration still lets generation time dominate the ordinal score.

## Comparacion Directa Con El Run Sin Positional Encoding

The most relevant control for this run is:

```text
results/two_head_ordinal_sft_kubernetes_v1/
  two-head-ordinal-density-v2-centered-gap05-mlp-lr3-threshold-lr50-20260523/
```

This control has the same centered threshold initialization, same gap, same
ordinal MLP LR multiplier and same threshold LR multiplier, but it does not add
explicit positional conditioning to the ordinal head.

Global comparison:

| Metric | No positional | FiLM positional V2 | Reading |
| --- | ---: | ---: | --- |
| Evaluated count | 70 | 70 | Same validation size. |
| Structured output parse success | 1.0000 | 1.0000 | No difference. |
| YAML parse success | 0.0286 | 0.1286 | FiLM improves parseability. |
| Block parse success | 0.0286 | 0.1286 | FiLM improves parser-facing YAML reconstruction. |
| Average level MAE | 0.4559 | 0.8430 | FiLM worsens level distance. |
| Average level exact match | 0.0107 | 0.0278 | FiLM slightly improves exact match, but both are very low. |
| Line text F1 | 0.0145 | 0.0782 | FiLM improves surface block text overlap. |
| Prompt requirement F1 | 0.0143 | 0.0799 | FiLM improves prompt requirement overlap. |
| Kubernetes domain validity score | 0.0238 | 0.1000 | FiLM improves this proxy metric, still low in absolute value. |
| Deep exact recall 5-8 | 0.2174 | 0.1214 | FiLM worsens deep exact recovery. |
| Deep off-by-one recall 5-8 | 0.5652 | 0.4143 | FiLM worsens near-deep recovery. |
| Compressed deep to 0-4 rate | 0.6014 | 0.7429 | FiLM increases deep-level compression. |
| Predicted max level mean | 4.4857 | 4.1571 | FiLM predicts shallower maximum depths. |
| YAML parse success when target max level >= 5 | 0.0435 | 0.1522 | FiLM improves parseability on deeper targets, despite worse level accuracy. |

This comparison separates two effects. FiLM V2 helps the generated block output
become more parseable and somewhat more textually aligned, but it hurts the
ordinal hierarchy signal. In other words, the model is better at producing an
output that the parser can consume, but worse at assigning the exact structural
depths needed by the explicit `level` objective.

Quartile comparison:

| Quartile | Metric | No positional | FiLM positional V2 | Delta FiLM - no-pos |
| --- | --- | ---: | ---: | ---: |
| Q1 | Level MAE | 0.4133 | 0.4404 | +0.0270 |
| Q1 | Exact match | 0.6378 | 0.6284 | -0.0093 |
| Q1 | Pred level 0 rate | 0.9067 | 0.9541 | +0.0475 |
| Q1 | Gold level 0 rate | 0.6578 | 0.6468 | -0.0110 |
| Q1 | Pred deep 5-8 count | 0 | 0 | 0 |
| Q1 | Gold deep 5-8 count | 0 | 0 | 0 |
| Q2 | Level MAE | 1.1695 | 1.3871 | +0.2176 |
| Q2 | Exact match | 0.1501 | 0.1290 | -0.0211 |
| Q2 | Pred level 0 rate | 0.4092 | 0.5335 | +0.1243 |
| Q2 | Gold level 0 rate | 0.0291 | 0.0496 | +0.0205 |
| Q2 | Pred deep 5-8 count | 0 | 0 | 0 |
| Q2 | Gold deep 5-8 count | 0 | 0 | 0 |
| Q3 | Level MAE | 0.8372 | 1.1783 | +0.3411 |
| Q3 | Exact match | 0.3558 | 0.2627 | -0.0932 |
| Q3 | Pred level 0 rate | 0.0326 | 0.1470 | +0.1144 |
| Q3 | Gold level 0 rate | 0.0140 | 0.0193 | +0.0053 |
| Q3 | Pred deep 5-8 count | 38 | 22 | -16 |
| Q3 | Gold deep 5-8 count | 33 | 39 | +6 |
| Q4 | Level MAE | 0.9795 | 1.2723 | +0.2928 |
| Q4 | Exact match | 0.3667 | 0.2330 | -0.1337 |
| Q4 | Pred level 0 rate | 0.0179 | 0.0628 | +0.0449 |
| Q4 | Gold level 0 rate | 0.0051 | 0.0052 | +0.0001 |
| Q4 | Pred deep 5-8 count | 95 | 67 | -28 |
| Q4 | Gold deep 5-8 count | 105 | 101 | -4 |

The quartile analysis makes the tradeoff sharper. FiLM V2 does not merely fail
to remove the initial `level=0` bias; it increases it relative to the
non-positional control in every quartile. The biggest deterioration appears in
Q2 and Q3, where gold `level=0` is already rare but FiLM still predicts zero far
more often than the control.

The deep-level comparison points in the same direction. In Q3 and Q4, where
deep levels start to appear, the non-positional model predicts more `5-8` levels
than FiLM V2. FiLM therefore seems to produce more parseable text while
compressing the hierarchy into a shallower range.

Correlation comparison:

| Correlation | No positional | FiLM positional V2 |
| --- | ---: | ---: |
| Absolute line position -> z, Pearson | 0.7804 | 0.8442 |
| Absolute line position -> z, Spearman | 0.8870 | 0.8647 |
| Absolute line position -> predicted level, Pearson | 0.8177 | 0.7734 |
| Predicted level -> gold level, Pearson | 0.8029 | 0.7816 |

The no-positional model already has a strong relationship between line position
and ordinal score, because the hidden state itself carries autoregressive
positional information. Adding FiLM does not create the temporal effect from
zero; it strengthens or reshapes an effect that was already present. This is
important for interpretation: the issue is not only how to provide position to
the head, but how to prevent the head from using position as a shortcut for
hierarchy.

## Analisis De Errores De Parseo Frente Al Control No-Posicional

The parser-facing improvement of FiLM V2 is real. The YAML parse success rate
goes from 2/70 to 9/70, and the set-level comparison is especially informative:

| Parseability set comparison | Unit ids |
| --- | --- |
| Parseable in both runs | `q126::question_simplified`, `q22::question` |
| Parseable only with FiLM V2 | `q126::question`, `q17::question`, `q32::question`, `q35::question_simplified`, `q49::question`, `q49::question_simplified`, `q65::question_simplified` |
| Parseable only without positional encoding | none |

This means FiLM V2 does not lose any example that the non-positional control
already parsed. It strictly adds 7 parseable validation examples in this run.
That is why, despite the worse `level_mae`, it is reasonable to read FiLM V2 as
a better parser-facing generator.

Error classes:

| YAML exception class | No positional | FiLM positional V2 |
| --- | ---: | ---: |
| ParserError | 58 | 44 |
| ScannerError | 10 | 17 |
| Parse failures total | 68 | 61 |

FiLM reduces the total number of parser failures and, more importantly, changes
their shape. The non-positional model is dominated by `ParserError` cases caused
by inconsistent mapping structure. FiLM reduces those but exposes more
`ScannerError` cases and collection-level failures.

Main PyYAML problem messages:

| Problem message | No positional | FiLM positional V2 | Interpretation |
| --- | ---: | ---: | --- |
| `expected <block end>, but found '<block mapping start>'` | 53 | 14 | FiLM greatly reduces broken mapping nesting. |
| `mapping values are not allowed here` | 9 | 16 | FiLM still often places key-value lines at illegal indentation. |
| `expected <block end>, but found '?'` | 3 | 27 | FiLM shifts many failures into block collections/list contexts. |
| `expected <block end>, but found '-'` | 2 | 2 | Similar rate of sequence-entry boundary errors. |
| `could not find expected ':'` | 1 | 1 | Rare broken scalar/mapping syntax in both. |
| `expected <block end>, but found '<scalar>'` | 0 | 1 | Rare FiLM-only scalar boundary error. |

Context of the parser failure:

| Parser context | No positional | FiLM positional V2 |
| --- | ---: | ---: |
| `while parsing a block mapping` | 55 | 16 |
| `while parsing a block collection` | 3 | 28 |
| `while scanning a simple key` | 1 | 1 |

This is the clearest qualitative difference. The no-positional model usually
fails because the mapping hierarchy itself is malformed: parent keys such as
`metadata:`, `spec:`, `template:` or nested mapping fields are assigned levels
that make PyYAML expect the mapping to close, but the next line starts another
mapping at an incompatible indentation.

FiLM V2 reduces that mapping-nesting failure substantially. Its remaining errors
are much more often list/collection errors. Typical cases are:

- command lists where a scalar list item such as `- sleep` or `- while true...`
  is followed by `image:` or `name:` at a level that PyYAML still interprets as
  part of the collection;
- environment variable blocks where `value:` is over-indented relative to
  `- name:`;
- container or host alias sections where a mapping key appears immediately
  after a sequence context without closing the sequence cleanly.

First parser error position:

| First error position | No positional | FiLM positional V2 |
| --- | ---: | ---: |
| Q1 | 0 | 0 |
| Q2 | 15 | 2 |
| Q3 | 46 | 38 |
| Q4 | 7 | 21 |
| Average error line | 14.68 | 18.30 |
| Average relative error position | 0.5811 | 0.7036 |

FiLM therefore tends to fail later. This matters: even when it does not produce
a parseable manifest, it more often gets through the initial document skeleton
before the parser breaks. The non-positional model tends to fail around the
middle of the manifest, while FiLM failures are shifted toward the last third.

The parser-error analysis changes the interpretation of the run. From the
`level` metric alone, FiLM V2 looks worse than the no-positional control. From
the parser-facing perspective, FiLM V2 is better: it preserves all parseable
cases from the control, adds new parseable cases, reduces mapping-nesting
failures, and pushes many remaining errors later into list-valued sections. The
remaining weakness is not generic YAML syntax anymore, but especially the
interaction between sequence scopes and mapping fields inside Kubernetes
substructures such as `containers`, `command`, `env`, `volumeMounts`,
`hostAliases` and similar nested lists.

## Interpretacion

This run does not support the claim that adding positional FiLM solves the
level-head problem. It does support a more cautious claim: the temporal position
is an active and highly predictive variable in the level head, but the model
still confuses temporal progress with structural depth.

The model seems to have learned a monotonic-ish mapping:

```text
early line -> low z -> shallow level
later line -> higher z -> deeper level
```

That pattern is partially aligned with the dataset, because deeper YAML levels
often appear after the beginning of a manifest. But it is not equivalent to
learning hierarchy. The first quartile remains over-compressed to level 0, and
levels 6-8 disappear entirely from the prediction distribution.

The result also does not prove that the initial latent representations are
intrinsically poorer. The data are more consistent with non-stationarity plus a
shortcut: the same structural label is being read through a latent distribution
that changes over generation time, and the positional feature gives the head a
strong temporal prior. Whether the early latent states are less diverse or less
informative still requires direct latent-space analysis or ablations.

## W&B And Logging Notes

W&B resumed the same run successfully. During the resumed final validation,
there were warnings that some logs at step 159 were ignored because W&B had
already advanced to a higher internal step. For this reason, local artifacts
should be treated as the source of truth:

- `metrics.json`
- `validation_predictions.jsonl`
- `validation_metrics_progress.jsonl`
- `validation_example_metrics.jsonl`
- `state.json`

Other warnings were non-fatal:

- `expandable_segments` is not supported on this Windows platform.
- `torch.load(weights_only=False)` FutureWarning.
- generation flags `temperature`, `top_p` and `top_k` may be ignored.

## Conclusion Operativa

The model is trained and evaluable, but FiLM positional V2 should not be adopted
as the next main solution without further audit. It improves YAML parseability
relative to the previous positional V1 run, but it does not fix the structural
level prediction problem and still shows a strong early collapse to `level=0`.

Immediate next steps:

1. Run `normal`, `zero-position` and `shuffle-position` ablations on
   `checkpoint-step-159`.
2. Add a simple `position-only` baseline to quantify how much of the result is
   explained by temporal shortcuts.
3. Compare per-quartile `z` distributions against thresholds to see whether the
   failure is mainly calibration, feature collapse or output-range compression.
4. Inspect examples where deep gold levels 6-8 are compressed to 4-5 versus
   compressed to 0-3.
5. Consider a future variant that predicts a residual correction over a
   position-only prior, instead of injecting position directly into the same
   ordinal score.

## Artefactos

- Run directory:
  `results/two_head_ordinal_film_positional_sft_kubernetes_v1/two-head-ordinal-film-positional-v2-512-64-gap05-mlp-lr3-threshold-lr50-20260602/`
- Config:
  `results/two_head_ordinal_film_positional_sft_kubernetes_v1/two-head-ordinal-film-positional-v2-512-64-gap05-mlp-lr3-threshold-lr50-20260602/config.json`
- State:
  `results/two_head_ordinal_film_positional_sft_kubernetes_v1/two-head-ordinal-film-positional-v2-512-64-gap05-mlp-lr3-threshold-lr50-20260602/state.json`
- Metrics:
  `results/two_head_ordinal_film_positional_sft_kubernetes_v1/two-head-ordinal-film-positional-v2-512-64-gap05-mlp-lr3-threshold-lr50-20260602/metrics.json`
- Final validation predictions:
  `results/two_head_ordinal_film_positional_sft_kubernetes_v1/two-head-ordinal-film-positional-v2-512-64-gap05-mlp-lr3-threshold-lr50-20260602/validation_predictions.jsonl`
- Final checkpoint:
  `results/two_head_ordinal_film_positional_sft_kubernetes_v1/two-head-ordinal-film-positional-v2-512-64-gap05-mlp-lr3-threshold-lr50-20260602/checkpoints/checkpoint-step-159/`
