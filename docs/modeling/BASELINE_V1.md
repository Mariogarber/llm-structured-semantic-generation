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

Terminology for structural blocks and evaluation fields is defined in
`docs/reference/TERMINOLOGY.md`.

The recommended baseline output format is now `blocks_tsv_compact_v1`, because
it is substantially shorter than the previous JSON array output and also
shorter than `blocks_tsv_v1`. The compact format removes model-predicted
`line_index` from the inference surface and lets the parser reconstruct it
deterministically from line order.

## Command

Validate the setup without loading the model:

```bash
uv run python scripts/run_kubernetes_baseline.py --dry-run
```

Run the validation split after installing optional LLM dependencies:

```bash
uv sync --extra llm
uv run python scripts/run_kubernetes_baseline.py --split validation --run-id baseline-validation
```

This uses `blocks_tsv_compact_v1` by default. If you need the previous verbose
formats for comparison, you can still request them explicitly:

```bash
uv run python scripts/run_kubernetes_baseline.py --split validation --run-id baseline-validation-json --output-format json_array
uv run python scripts/run_kubernetes_baseline.py --split validation --run-id baseline-validation-tsv --output-format blocks_tsv_v1
```

The script writes timestamped outputs under:

```text
results/baseline_kubernetes_v1/
```

If `--run-id` is passed, the run is created or resumed under:

```text
results/baseline_kubernetes_v1/<run-id>/
```

Each run contains:

- `config.json`
- `state.json`
- `predictions.jsonl`
- `metrics.json`

If `--collect-latent-means` is enabled, the run also writes:

- `latent_mean_vectors.jsonl`

Dry runs write only `config.json` and a dry-run `metrics.json`. They do not
create `state.json` or partial prediction artifacts.

Render an interactive HTML dashboard for any run directory:

```bash
uv run python scripts/render_baseline_dashboard.py --run-dir results/baseline_kubernetes_v1/<run-id>
```

By default the script writes a self-contained `baseline_dashboard.html` inside
that same run directory. You can override the destination with `--output-html`.

## Incremental Execution Contract

The baseline is no longer a monolithic all-or-nothing run. It now follows the
repository-wide LLM execution policy:

- `--output-format` controls the structured text produced by the model.
- `blocks_tsv_compact_v1` is the default and recommended baseline format
  because it reduces output-token pressure substantially relative to both JSON
  arrays and `blocks_tsv_v1`.
- `blocks_tsv_v1` remains available for controlled comparison with the explicit
  SFT serialization family.
- `--run-id` enables stable re-entry into the same run directory.
- `--batch-size` controls how many samples are processed before progress is
  flushed to disk. The default is `1`.
- The script resumes automatically when the target `run-id` already exists and
  its stored configuration is compatible.
- `predictions.jsonl` is the source of truth for completed work during resume.
- `state.json` is reconciled from persisted artifacts when a previous execution
  stopped before updating the state cleanly.
- `latent_mean_vectors.jsonl` is also append-only when latent collection is
  enabled and is reconciled against completed prediction rows during resume.
- `metrics.json` is written only after the run reaches completion.

The dry run also records `model_checks`. A full run requires the local model
directory to contain tokenizer files as well as weights/config. The current local
model config declares `bitsandbytes` quantization, so the optional LLM
environment must include that package too.

## Metrics

The baseline records:

- structured output parse success rate,
- YAML parse success rate after reconstruction,
- parsed-document equality against the reference,
- average line-content exact match rate,
- average level exact match rate.

These are automatic structural checks only. They are not a full semantic
Kubernetes validation suite.

## Current Status

The baseline runner is implemented and has completed recorded validation/test
runs. These runs are valid as baseline references, but they are not final thesis
success claims: they establish the pre-SFT behavior that later supervised and
alignment stages must improve.

## Recorded Runs

The baseline now has a completed recorded run over the full `test` split:

- `run_id`: `compact-test70-320-vtfix`
- split: `test`
- rows: `70`
- output format: `blocks_tsv_compact_v1`
- `max_new_tokens`: `320`
- latent collection: enabled

Main artifacts:

- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/config.json`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/state.json`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/predictions.jsonl`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/latent_mean_vectors.jsonl`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/metrics.json`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/metrics_recomputed.json`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/baseline_dashboard.html`

Headline metrics for that run:

- `structured_output_parse_success_rate = 0.8857`
- `yaml_parse_success_rate = 0.4677`
- `parsed_equal_rate = 0.0161`
- `average_line_text_f1 = 0.3175`
- `average_semantic_key_f1 = 0.4082`
- latent vectors collected for all `70` rows with dimension `3584`

The recomputed metrics artifact adds prompt-requirement, Kubernetes-domain, and
auxiliary text metrics without rerunning inference:

- `average_prompt_requirement_f1 = 0.3230`
- `average_kubernetes_domain_validity_score = 0.3898`
- `kubernetes_domain_gate_pass_rate = 0.0806`
- `average_bleu_score = 0.4576`
- `average_rougeL_f1 = 0.6584`

A shorter pilot run used during prompt and parser iteration is also preserved at:

- `results/baseline_kubernetes_v1/compact-test5-320-prompt3/`

The detailed execution narrative for the completed baseline is documented in:

- `docs/experiments/baseline/runs/BASELINE_EXECUTION_REPORT_2026-04-28.md`

## Observed Baseline Limitations

The completed zero-shot baseline is useful as a reference point, but its failure
modes remain substantial and are exactly the kind of behavior the later SFT
comparison is expected to improve.

Observed limitations in the full `test` run:

- structural surface still fails before evaluation on `8/70` rows;
- only `29/62` evaluated rows reconstruct to valid YAML;
- exact parsed-document equality is almost absent (`0.0161`);
- line coverage is incomplete (`line_count_match_rate = 0.0968`);
- top-level signals such as `kind`, `apiVersion`, and `metadata.name` are often
  captured, but deeper structural coverage remains unstable;
- the most common final failures are YAML `ScannerError` and `ParserError`
  after reconstruction, not missing run artifacts or resume corruption.

This is acceptable for the baseline phase because the main requirement is now:

- reproducible execution;
- resumable long runs;
- stable artifact generation;
- a measurable non-trivial reference before comparing `serialized_sft` against
  the main `two_head_sft` model.
