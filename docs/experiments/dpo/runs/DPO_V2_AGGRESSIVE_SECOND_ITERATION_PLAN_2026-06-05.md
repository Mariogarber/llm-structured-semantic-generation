# DPO v2 Aggressive Second-Iteration Plan - 2026-06-05

Document type: analysis

## Scope

This document records the planned second DPO experiment over Kubernetes v1 using
the automatic preference dataset v2.

The experiment is intentionally aggressive. Its purpose is not to start from the
most conservative hyperparameter setting, but to test how far the DPO stage can
be pushed before the model visibly degrades. A failed run is therefore still
useful if it identifies the practical upper bound of the current preference
optimization setup.

## Experimental Interpretation

This run should be interpreted as a second offline DPO iteration, analogous in
spirit to an iterative or online preference-optimization loop:

```text
serialized_sft -> DPO v1 -> DPO v2
```

The base policy and reference policy for this run are the first completed DPO
checkpoint:

```text
results/dpo_kubernetes_v1/training/dpo-beta010-full-20260529-170249/checkpoints/checkpoint-step-57/
```

The new preference data is:

```text
results/dpo_kubernetes_v1/preference_annotation/agent-full-auto-v2/preferences_final.jsonl
```

This differs from an independent DPO-v2-from-SFT run. The experiment asks
whether a stricter second preference dataset can further move an already aligned
DPO policy, not whether v2 alone is better than v1 from the original SFT base.

## Dataset Context

The v2 preference dataset contains:

- `229` final preference pairs;
- `193` train units with at least one pair;
- `62` gate-crossing pairs;
- pair types:
  - `domain_invariant`: `91`;
  - `gate_crossing`: `62`;
  - `prompt_fidelity`: `38`;
  - `structural_fidelity`: `25`;
  - `level5_practice`: `13`;
- `100%` YAML parseability and block-contract parseability for both chosen and
  rejected candidates;
- average score margin `1.007846`.

The dataset is smaller than v1, but stricter. Because it has `229` pairs and the
training configuration uses `gradient_accumulation_steps=8`, one epoch is only:

```text
ceil(229 / 8) = 29 optimizer steps
```

For this reason the planned run uses `3` epochs, for a total of `87` optimizer
steps.

## Planned Configuration

The planned hyperparameters are:

- base/reference checkpoint: DPO v1 `checkpoint-step-57`;
- preference dataset: automatic v2;
- `beta`: `0.30`;
- `learning_rate`: `1e-4`;
- `epochs`: `3`;
- `batch_size`: `1`;
- `gradient_accumulation_steps`: `8`;
- expected optimizer steps: `87`;
- checkpoint cadence: every `29` steps, aligned with epoch boundaries;
- intra-training validation: disabled;
- final validation: enabled;
- W&B logging: online, including training and final validation metrics.

The high learning rate is deliberate. It is expected to expose whether the
current DPO setup can tolerate a much stronger second alignment step. It should
not be described later as a conservative or default DPO-v2 setting.

## Launch Command

```powershell
uv run python scripts\train_kubernetes_dpo.py `
  --run-id dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605 `
  --sft-adapter-path results\dpo_kubernetes_v1\training\dpo-beta010-full-20260529-170249\checkpoints\checkpoint-step-57 `
  --preference-file results\dpo_kubernetes_v1\preference_annotation\agent-full-auto-v2\preferences_final.jsonl `
  --epochs 3 `
  --learning-rate 1e-4 `
  --beta 0.30 `
  --checkpoint-steps 29 `
  --checkpoint-keep-last 0 `
  --two-thirds-validation-samples 0 `
  --wandb-mode online `
  --wandb-run-name dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605 `
  --wandb-tags dpo-v2,second-iteration,base-dpo-beta010,beta-0.30,lr-1e-4,epochs-3,aggressive,full-validation
```

Expected checkpoint layout:

```text
results/dpo_kubernetes_v1/training/dpo-v2-from-dpo-v1-beta030-lr1e4-e3-20260605/
  config.json
  state.json
  reference_logps.jsonl
  reference_logps_summary.json
  train_log.jsonl
  checkpoints/
    checkpoint-step-29/
    checkpoint-step-58/
    checkpoint-step-87/
  validation_predictions.jsonl
  validation_metrics_progress.jsonl
  validation_example_metrics.jsonl
  metrics.json
```

## Tooling Note

The trainer argument is still named `--sft-adapter-path` because the first DPO
implementation was designed for:

```text
serialized_sft -> serialized_sft_dpo
```

For this run, that argument intentionally points to the DPO v1 checkpoint root.
The trainer resolves the contained `adapter/` directory and tokenizer from that
checkpoint. Operationally, the fixed reference log-probabilities for the second
DPO pass are therefore computed from DPO v1, not from the original SFT adapter.

If the generated `config.json` still uses the generic labels
`source_model_variant=serialized_sft` or `stage=dpo_training_v1`, those labels
should be interpreted as tooling names, not as the methodological description of
this run. The methodological description is this document: DPO v2 is initialized
from DPO v1.

## Monitoring

During training, W&B and `train_log.jsonl` should be watched especially for:

- `loss`;
- `reward_margin`;
- `reward_accuracy`;
- `chosen_reward`;
- `rejected_reward`;
- `grad_norm`;
- `learning_rate`;
- skipped or failed batches.

Because intra-training validation is disabled, no structural conclusion should
be drawn before final validation completes. Training metrics only show whether
the DPO objective is being optimized; they do not prove YAML, structural, prompt,
or Kubernetes-domain quality.

## Expected Failure Modes

The main risks are:

- sharp drop in YAML parseability;
- lower block parseability or malformed block surface;
- prompt drift caused by over-optimizing proxy preferences;
- degradation of line text F1 or level exact match;
- worse Kubernetes gate pass rate despite larger DPO reward margin;
- instability from large gradient updates;
- overfitting the stricter but small v2 preference set.

These failure modes are acceptable for this run if they are clearly documented
afterwards. The run is designed to test aggressiveness, not to guarantee a better
checkpoint.

## Result Interpretation Policy

This run should be compared against at least:

- serialized SFT `checkpoint-step-159`;
- DPO v1 `checkpoint-step-57`;
- DPO beta `0.30` v1 result, where relevant.

The primary validation reading should prioritize:

1. YAML parse success rate;
2. block parse success rate;
3. prompt requirement F1;
4. line text F1;
5. level exact match and level MAE;
6. Kubernetes domain validity level;
7. Kubernetes domain gate pass rate;
8. domain error counts, especially level-5 practice errors.

If the run improves DPO reward metrics but damages parseability or prompt
adequacy, it should be interpreted as over-aggressive preference optimization.
If it changes outputs substantially while preserving parseability, it becomes a
useful candidate for a later less aggressive sweep around the observed boundary.

The test split must not be used for this exploratory hyperparameter choice.
