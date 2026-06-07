# DPO Trainer Smoke Result - 2026-05-29

Document type: run result

## Summary

This note records the first operational smoke run of the Kubernetes v1
`serialized_sft_dpo` trainer. It is not a full DPO experiment and must not be
reported as evidence that DPO improves over `serialized_sft`.

The goal was narrower: verify that the new trainer can load the frozen
serialized SFT checkpoint, precompute reference log-probabilities, train a DPO
policy adapter for one preference pair, write local resumable artifacts, and log
training metrics to Weights & Biases in offline mode.

## Run

- Run id: `dpo-smoke-1pair-beta010-20260529`
- Output directory:
  `results/dpo_kubernetes_v1/training/dpo-smoke-1pair-beta010-20260529/`
- Preference file:
  `results/dpo_kubernetes_v1/preference_annotation/agent-full-auto-v1/preferences_final.jsonl`
- Source SFT checkpoint:
  `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/checkpoints/checkpoint-step-159/`
- Source SFT adapter:
  `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/checkpoints/checkpoint-step-159/adapter/`
- Base model: `model/qwen2.5-7b-instruct-4bit/`
- Beta: `0.10`
- Preference pairs used: `1`
- Validation examples used: `1`
- Final validation: skipped
- W&B mode: `offline`

Command:

```powershell
uv run python scripts\train_kubernetes_dpo.py `
  --run-id dpo-smoke-1pair-beta010-20260529 `
  --max-train-samples 1 `
  --max-validation-samples 1 `
  --epochs 1 `
  --batch-size 1 `
  --gradient-accumulation-steps 1 `
  --checkpoint-steps 1 `
  --max-seq-length 768 `
  --skip-final-eval `
  --wandb-mode offline `
  --beta 0.10
```

## Observed Result

The smoke run completed successfully.

Produced artifacts:

- `config.json`
- `state.json`
- `reference_logps.jsonl`
- `reference_logps_summary.json`
- `train_log.jsonl`
- `metrics.json`
- `checkpoints/checkpoint-step-1/`
- local W&B offline run directory

The DPO checkpoint was written under the DPO run directory only. The source SFT
adapter remained a source artifact and was not used as a save target.

Training metrics for the single optimizer step:

- `loss`: `0.6971878409385681`
- `reward_margin`: `-0.008065223693847656`
- `reward_accuracy`: `0.0`
- `chosen_reward`: `0.005223655607551336`
- `rejected_reward`: `0.01328887976706028`
- `chosen_logp`: `-7.93162727355957`
- `rejected_logp`: `-57.74083709716797`
- `logp_margin`: `49.80921173095703`
- `reference_logp_margin`: `49.889862060546875`
- `grad_norm`: `13.934839248657227`

These values are only smoke diagnostics. A single pair is not meaningful as an
alignment result.

## Environment

The trainer recorded the following runtime versions in `config.json`:

- `torch`: `2.5.1+cu121`
- `transformers`: `5.5.0`
- `peft`: `0.18.1`
- `trl`: `1.0.0`
- `bitsandbytes`: `0.49.2`
- `accelerate`: `1.13.0`
- `wandb`: `0.26.1`

CUDA was available on `NVIDIA GeForce RTX 3060 Laptop GPU`.

`uv sync --extra llm` was attempted, but the local `.venv` cleanup hit Windows
permission errors in package metadata directories under `site-packages`. The LLM
dependencies were nevertheless importable through `uv run`, so the smoke run was
executed with the existing environment.

## Next Step

The next experimental step is the real `beta=0.10` DPO run on the full
preference dataset, with W&B online if the local `.env` key should be used. That
run should include validation before any comparison against `serialized_sft`.
