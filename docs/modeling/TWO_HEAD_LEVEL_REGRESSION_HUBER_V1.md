# Two-Head Level Regression Huber v1

Document type: contract

This note defines a separate experimental branch for the two-head Kubernetes SFT
model. It keeps the same parser-facing contract as the ordinal-threshold branch,
but replaces the ordinal cumulative-threshold head with a scalar regression head.

## Motivation

The ordinal density branch already uses an ordered target, but the observed
training runs still compress rare deep levels. This branch tests a simpler
hypothesis:

```text
hidden_state -> MLP -> continuous_level_score
```

The level prediction is then obtained with:

```text
predicted_level = round(continuous_level_score).clamp(0, level_class_count - 1)
```

The point of the branch is not to claim that YAML indentation is truly
continuous. It is a controlled ablation: make distance in level space explicit
in the training objective and measure whether that reduces deep-level collapse.

## Training Surface

The branch keeps the same textual target as `two_head_sft` and
`two_head_ordinal_density_v2`:

```text
prompt -> content_blocks_v1 + level head -> blocks_tsv_v1 -> parser -> YAML
```

The autoregressive LM still learns `content_blocks_v1`, without a serialized
`level` column. The level head is supervised at `record_prefix_state`, the same
alignment policy used by the current two-head ordinal experiments.

## Regression Head

Implemented script:

```text
scripts/train_kubernetes_two_head_regression_sft.py
```

Main artifacts:

```text
results/two_head_regression_sft_kubernetes_v1/<run-id>/
  config.json
  state.json
  train_log.jsonl
  regression_history.jsonl
  gradient_diagnostics.jsonl
  checkpoints/checkpoint-step-*/adapter/
  checkpoints/checkpoint-step-*/regression_level_head.pt
  checkpoints/checkpoint-step-*/training_state.pt
  intermediate_validation_metrics.jsonl
  validation_predictions.jsonl
  metrics.json
```

The checkpoint contract mirrors the ordinal trainer: adapter, tokenizer,
regression head, optimizer, scheduler, and resumable state are saved together.

## Loss

The level loss is density-weighted Huber regression:

```text
level_loss = mean(weight(level) * huber(level_score, gold_level, delta))
loss = lm_loss + lambda_level * level_loss
```

The weights reuse the same smoothed density weighting mechanism as the ordinal
branch:

```text
smoothed_density(level) = kernel_smoothing(train_level_histogram)
weight(level) = min(mean_density / smoothed_density(level), max_density_weight)
```

This keeps the rare-level correction available while changing only the head
geometry and the level objective.

## Learning Rates

The regression head has its own optimizer group:

```text
base_lora:        --learning-rate
regression_head:  --learning-rate * --regression-head-learning-rate-multiplier
```

This preserves the current experiment pattern where the structural head can move
at a different rate from the LoRA/base group.

## Minimal Command Template

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
uv run python scripts\train_kubernetes_two_head_regression_sft.py `
  --output-dir results\two_head_regression_sft_kubernetes_v1 `
  --run-id two-head-level-regression-huber-v1-YYYYMMDD `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --epochs 3 `
  --checkpoint-steps 8 `
  --checkpoint-keep-last 0 `
  --eval-checkpoint-steps 32 `
  --eval-max-samples 10 `
  --eval-sample-strategy random `
  --validation-log-every 1 `
  --oom-recovery skip_batch `
  --max-oom-skips 0 `
  --wandb-mode online `
  --wandb-log-artifacts `
  --regression-head-learning-rate-multiplier 3 `
  --regression-huber-delta 1.0
```

## Interpretation Rules

This branch should be interpreted as an ablation, not as the new default model.
It answers whether an explicit distance-sensitive level objective improves the
failure mode observed in the ordinal runs.

Important comparisons:

- predicted distribution of levels, especially `5..8`;
- `deep_level_exact_recall_5_8`;
- `deep_level_off_by_one_recall_5_8`;
- YAML parse success;
- whether regression overshoots or creates unstable oscillations in
  `level_score`;
- whether better level distance comes at the cost of worse content generation.

If it improves deep-level recall but hurts parseability, the result should be
treated as evidence for adding a distance auxiliary term to the ordinal model,
not necessarily as evidence that pure regression is the final architecture.
