# Positional Conditioning Next Steps

Document type: analysis

Date: 2026-05-27

## Context

The first positional ordinal head was introduced as a deliberately simple
variant of the two-head model. It keeps the same ordinal `level` output space
and injects a causal absolute line-position encoding just before the final
linear projection:

```text
hidden -> 512 -> 64
line_position -> sin/cos 16
concat(hidden_64, position_16) -> ordinal_score
```

This was intended as a baseline, not as the final positional conditioning
mechanism. The reason for starting with this variant was practical: it isolates
whether a causal temporal signal helps the head without multiplying labels into
`level x bin` classes.

The intermediate validation at `checkpoint-step-80` suggests that the model is
not simply ignoring position. The `ordinal_score` rises strongly with
`line_position`, and the final linear layer gives non-trivial weight to the
position branch. However, the behavior still looks like a temporal shortcut:
early lines remain compressed to `level=0`, while later quartiles receive more
varied levels.

The current reading is therefore:

```text
positional encoding is being used,
but mainly as an additive temporal bias over ordinal_score,
not yet as a rich condition for interpreting the hidden state.
```

Update after resume: the run was first stopped after training completed and
after 60/70 final validation predictions had been generated. The final
validation was resumed on 2026-06-02 and completed at 70/70. The full final
audit is recorded as a run result in:

```text
docs/experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_POSITIONAL_V1_FINAL_CONCAT_RUN_20260527_ANALYSIS.md
```

The full final result preserves the same reading as the step-80 audit: Q1 still
has a very high `level=0` prediction rate, and `line_position` remains strongly
correlated with `ordinal_score`. The normal/zero/shuffle ablations were not run
before closing this V1 experiment.

## Intermediate Evidence

The audit of the 10 random validation samples at `checkpoint-step-80` showed:

| Quartile | pred level 0 rate | gold level 0 rate at same line index | mean ordinal_score |
|---|---:|---:|---:|
| Q1 | 1.000 | 0.644 | -3.370 |
| Q2 | 0.593 | 0.037 | -2.003 |
| Q3 | 0.250 | 0.000 | -0.917 |
| Q4 | 0.059 | 0.000 | -0.409 |

The first threshold was around `tau_0 = -1.78`. This means that Q1 was almost
entirely below the first ordinal cut, so the head had no opportunity to emit
levels above zero in that region.

The same audit found a high correlation between absolute line position and
`ordinal_score`:

```text
Pearson(line_position, ordinal_score) = 0.864
Spearman(line_position, ordinal_score) = 0.886
```

This supports the interpretation that position is active, but currently acts
like a coarse monotonic schedule. That schedule is not enough for YAML, because
hierarchy is not a smooth function of line index. The model must also detect
local structural transitions such as `metadata:`, `spec:`, `template:`, list
items, and returns to shallower scopes.

## Bibliographic Reading

The current result is consistent with the literature on positional and
conditional representations:

- Transformer positional encodings inject order information into a sequence
  model, but they do not by themselves define the structural semantics of the
  target output. See Vaswani et al., 2017:
  https://ar5iv.labs.arxiv.org/html/1706.03762v7
- Fourier features help MLPs represent functions over low-dimensional
  coordinates, but the benefit depends on where and how the coordinate encoding
  interacts with the network. See Tancik et al., 2020:
  https://arxiv.org/abs/2006.10739
- Feature-wise conditioning makes an important distinction between simple
  concatenation and feature-wise transformations. A late concatenation before a
  linear layer mostly behaves like a conditional bias. See:
  https://distill.pub/2018/feature-wise-transformations/
- FiLM applies feature-wise affine modulation and is a stronger fit when the
  conditioning signal should change how intermediate features are read. See
  Perez et al., 2018:
  https://arxiv.org/abs/1709.07871

For this project, the relevant conclusion is not that sinusoidal features are
wrong. The more precise conclusion is that `final_concat` is probably too late
and too linear for the role we want position to play.

## Immediate Audit Plan

The immediate step is to turn the current interpretation into reproducible
evidence. The audit script should produce:

- `quartile_level_z_summary.csv`: predicted/gold level distribution and
  `ordinal_score` statistics by generated quartile.
- `line_index_level_z_summary.csv`: the same comparison by absolute
  `line_index`.
- `correlations.json`: correlations between line position, `ordinal_score`,
  predicted level, and gold level.
- `ordinal_head_weight_balance.json`: norm and absolute-mean comparison between
  hidden weights and positional weights in the final projection.
- `position_only_baseline.json`: a baseline that predicts `level` from absolute
  position or fixed-width causal position bins.

The model ablations should be run on the same generated content blocks, without
regenerating text:

```text
normal  -> real positional encoding
zero    -> zeroed positional encoding
shuffle -> positional encoding shuffled across line positions
```

This isolates the `level` head from content-generation noise. The reading will
be:

- if `zero` or `shuffle` changes the predictions substantially, the position
  branch is materially affecting the head;
- if they barely change predictions, the position branch is mostly decorative;
- if `position-only` reproduces the same early-zero pattern, the model is
  exploiting a temporal shortcut already present in the dataset.

## Next Model Variant

The next implementation should create a new trainer, not modify previous
trainers:

```text
scripts/train_kubernetes_two_head_ordinal_film_positional_sft.py
```

The architecture should keep the same ordinal output and the same causal
absolute line-position encoding, but replace `final_concat` with FiLM after the
first hidden projection:

```text
hidden -> LayerNorm -> Linear(hidden,512) -> GELU -> Dropout(0.10)

line_position -> sin/cos 16 -> LayerNorm
              -> Linear(16,128) -> GELU -> Linear(128,1024)
              -> gamma,beta

hidden_512 = hidden_512 * (1 + scale * tanh(gamma)) + scale * beta

hidden_512 -> Linear(512,64) -> GELU -> LayerNorm(64)
           -> Linear(64,1) -> ordinal_score
```

The final FiLM generator layer should be zero-initialized, so the model starts
from an identity modulation. This makes the new run comparable to the previous
head while allowing the position signal to become multiplicative/feature-wise
when useful.

The position branch should still use no dropout. The initial V2 defaults should
match the current centered-gap ordinal experiment:

```text
learning_rate = 2e-4
ordinal_mlp_learning_rate_multiplier = 3
threshold_learning_rate_multiplier = 50
initial_threshold_center = 0.0
initial_threshold_gap = 0.5
position_frequencies = 1,2,4,8,16,32,64,128
```

## Interpretation Boundaries

This document should not be read as evidence that early hidden states are
intrinsically poorer. The stronger and currently supported claim remains:

```text
the latent space used by the level head is temporally non-stationary.
```

The goal of positional conditioning is to help the `level` head interpret hidden
states under this temporal non-stationarity. It is not to create more classes,
repair invalid YAML after generation, or make the parser responsible for hidden
model errors.

## Related Artifacts

```text
docs/analysis/latent/LATENT_POSITIONAL_DRIFT_HYPOTHESIS_2026-05-26.md
docs/analysis/latent/runs/LATENT_POSITIONAL_DRIFT_DISTANCE_ANALYSIS_2026-05-26.md
scripts/audit_positional_head_v1.py
scripts/train_kubernetes_two_head_ordinal_positional_sft.py
scripts/train_kubernetes_two_head_ordinal_film_positional_sft.py
```
