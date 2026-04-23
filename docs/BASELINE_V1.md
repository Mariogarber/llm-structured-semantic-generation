# Kubernetes Baseline v1

This document defines the first reproducible zero-shot baseline path for the
Kubernetes v1 dataset.

## Scope

The baseline uses the local base model at:

```text
model/qwen2.5-7b-instruct-4bit/
```

It performs no supervised weight updates, no LoRA adaptation, and no preference
optimization. It asks the model to produce structural blocks, then sends those
blocks through the deterministic parser.

## Command

Validate the setup without loading the model:

```bash
uv run python scripts/run_kubernetes_baseline.py --dry-run
```

Run the validation split after installing optional LLM dependencies:

```bash
uv sync --extra llm
uv run python scripts/run_kubernetes_baseline.py --split validation
```

The script writes timestamped outputs under:

```text
results/baseline_kubernetes_v1/
```

Each run contains:

- `config.json`
- `predictions.jsonl`
- `metrics.json`

Dry runs write only `config.json` and a dry-run `metrics.json`.

The dry run also records `model_checks`. A full run requires the local model
directory to contain tokenizer files as well as weights/config. The current local
model config declares `bitsandbytes` quantization, so the optional LLM
environment must include that package too.

## Metrics

The baseline records:

- JSON block parse success rate,
- YAML parse success rate after reconstruction,
- parsed-document equality against the reference,
- average line-content exact match rate,
- average level exact match rate.

These are automatic structural checks only. They are not a full semantic
Kubernetes validation suite.

## Current Status

The baseline runner is implemented. No baseline result should be described as a
validated thesis result until the script has been executed and its output has
been reviewed.
