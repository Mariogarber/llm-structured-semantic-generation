# SFT Run Notes - 2026-05-05

## Context

This note records operational issues observed during the first full
Architecture A SFT attempt for Kubernetes v1.

- Architecture: `serialized_sft`
- Target serialization: `blocks_tsv_v1`
- Base model: `model/qwen2.5-7b-instruct-4bit`
- Run id: `serialized-sft-a-v1-20260505-171226`
- Run directory: `results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/`
- W&B project: `llm-structured-semantic-generation`
- Training split size: 426 rows
- Validation split size: 70 rows
- Effective launch defaults: batch size 1, gradient accumulation 8, LoRA r=8,
  alpha=16, dropout=0.05, target modules `q_proj,k_proj,v_proj,o_proj`

## What Worked

- The run initialized W&B online and resumed into the same W&B run id after
  process restarts.
- Checkpoint resume worked from persisted adapter, optimizer, scheduler, and
  trainer state.
- The run reached `checkpoint-step-40` before the CUDA OOM issue below.
- The checkpoint pruning failure did not corrupt the latest checkpoint.

## Issue 1: Windows Checkpoint Pruning Failure

The first interruption was caused by Windows denying deletion of an old
checkpoint adapter directory during retention cleanup:

```text
PermissionError: [WinError 5] Acceso denegado:
results/.../checkpoints/checkpoint-step-5/adapter
```

This was not a model or CUDA failure. It happened after a newer checkpoint had
already been written.

Mitigation added:

- checkpoint pruning now retries;
- if deletion still fails, the trainer logs the failure to
  `checkpoint_prune_errors.jsonl`;
- pruning failures no longer terminate training.

## Issue 2: CUDA OOM At Step 40

The run later stopped repeatedly with CUDA OOM while trying to continue from:

```text
global_step=40
epoch=0
next_batch_index=320
```

The failure occurred during `backward()`:

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 1.10 GiB.
```

Manual inspection showed that this training region contains long examples,
including:

```text
q229::question             ~3113 prompt+target chars
q229::question_simplified  ~3051 prompt+target chars
```

Operational retries with fewer competing processes and with more conservative
GPU memory/offload settings did not pass the same point. This suggests that at
least one microbatch in this region exceeds the practical memory budget for the
current local GPU setup and `max_seq_length=2048`.

## OOM Recovery Policy Added

The trainer now supports an explicit recovery mode:

```bash
--oom-recovery skip_batch
```

In this mode, if a CUDA OOM happens on a microbatch, the trainer:

- records the skipped batch in `oom_skipped_batches.jsonl`;
- stores unit ids, epoch, batch index, global step, tensor shape when available,
  token counts when available, error text, and CUDA memory snapshot;
- clears gradients and CUDA cache;
- discards the current partial gradient accumulation window;
- continues training from the next microbatch;
- logs skip counters to W&B.

This is intentionally not silent repair. Any run using this mode must report
the number and identity of skipped samples, because the effective training set
has changed.

## Metric Visibility Note

This run was launched with:

```text
eval_checkpoint_steps=0
```

Therefore, W&B receives training loss and learning-rate logs during training,
but structural validation metrics are only produced during final validation.

After this run started, support was added for lightweight intermediate
validation:

```bash
--eval-checkpoint-steps 5 --eval-max-samples 5
```

Future runs can use that option to log parser-facing validation metrics under
`validation_sample/...` during training.

## Validation Progress Logging Added For Future Runs

The trainer now supports streaming metric logs during final validation:

```bash
--validation-log-every 1
```

When enabled, every completed validation prediction is still persisted to
`validation_predictions.jsonl`, and the trainer also writes:

- `validation_example_metrics.jsonl`
- `validation_metrics_progress.jsonl`

W&B receives two metric families:

- `validation_example/...`: per-example signals such as YAML parse success,
  `line_text_f1`, `level_exact_match_rate`, prompt requirement F1, semantic key
  F1, required-field rates, and generated token count.
- `validation_progress/...`: cumulative validation metrics over completed
  samples, including parseability, YAML validity, structural agreement,
  hierarchy metrics, prompt fidelity, approximate semantic/domain consistency,
  generated-token counts, and completion progress.

The custom W&B x-axis is:

```text
validation_progress/completed_count
```

This makes the final validation visible while it is running instead of only
after all validation samples finish. The official final metrics remain the
complete-split values logged under `validation/...` and `final/...`.

## Documentation Implication

If the final reported run uses `--oom-recovery skip_batch`, the experiment
description should explicitly include:

- the OOM recovery policy;
- the skipped unit ids from `oom_skipped_batches.jsonl`;
- the fact that skipped samples were excluded from optimizer updates;
- the local hardware constraint motivating the recovery;
- that this is an operational robustness mechanism, not a structural parser
  repair or semantic correction.
