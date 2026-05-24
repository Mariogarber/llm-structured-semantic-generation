# Baseline Execution Report (2026-04-28)

## Purpose

This note records the concrete execution process used to close the current
zero-shot Kubernetes baseline, including:

- output-format changes made to keep inference affordable on limited hardware;
- resumable execution behavior under interruption;
- pilot runs used to harden the baseline;
- the final completed `test` run and its artifacts;
- the observed limitations that motivate the next SFT stage.

## Starting Point

The original baseline surface was too expensive for long runs on the current
machine:

- the older JSON block output was too verbose;
- several early validation rows were truncated at the generation limit;
- long runs needed to survive interruption or crashes without losing progress.

Because this repository is now required to support incremental and resumable
LLM execution, the baseline was treated as the first full implementation of that
policy.

## Changes Applied During This Baseline Closure

### 1. Resumable execution

The baseline runner was moved to an incremental contract with:

- `config.json`
- `state.json`
- append-only `predictions.jsonl`
- append-only `latent_mean_vectors.jsonl` when latent collection is enabled
- `metrics.json` written only after successful completion

Resume semantics are based on persisted artifacts, not in-memory progress.
During restart, the runner reconciles `state.json` from `predictions.jsonl`.

### 2. Compact structured output

The baseline no longer uses verbose JSON output by default. It now requests
`blocks_tsv_compact_v1`, which keeps:

- `document_index`
- `level`
- `line_text`

and lets the parser reconstruct `line_index` deterministically from line order.

This reduces output-token pressure relative to both:

- the older JSON array format;
- `blocks_tsv_v1`, which remains the explicit SFT target serialization.

### 3. Prompt hardening

The baseline prompt was iterated to reduce structural mistakes:

- explicit distinction between mappings and lists;
- explicit examples for `metadata`, `spec`, and `containers`;
- explicit rule that top-level keys such as `apiVersion`, `kind`, `metadata`,
  and `spec` stay at level `0`.

### 4. Parser hardening

The compact TSV parser was made more tolerant to malformed but recoverable model
output:

- accepts `<TAB>` and `<tab>` markers;
- tolerates truncated tails when valid rows already exist;
- reconstructs missing `line_index` deterministically;
- accepts vertical-tab style separators as another malformed field-separator
  variant observed in real runs.

### 5. Dashboard

The baseline dashboard script was used to render a self-contained HTML report
per run, and its completion-state logic was updated to respect the current
`state.json` contract (`status` / `completed_at`).

## Pilot Runs Used During Iteration

Several small runs were used to improve the baseline before the final test pass.

Representative pilot runs:

- `results/baseline_kubernetes_v1/compact-val2-256/`
- `results/baseline_kubernetes_v1/compact-val2-256-prompt2/`
- `results/baseline_kubernetes_v1/compact-test5-320-prompt3/`

The most useful pilot before the final run was:

- `run_id = compact-test5-320-prompt3`
- split: `test`
- rows: `5`
- `max_new_tokens = 320`
- output format: `blocks_tsv_compact_v1`

Pilot metrics:

- `structured_output_parse_success_rate = 0.6`
- `yaml_parse_success_rate = 1.0` on evaluated rows
- `average_line_text_f1 = 0.6608`
- `average_semantic_key_f1 = 0.6696`

This pilot showed that once the structured surface parsed correctly, the model
could often produce valid YAML, but long-run robustness was still the real
question.

## Crash and Resume Behavior

During the full `test` execution, the machine crashed mid-run.

The repository did not lose completed work because:

- `predictions.jsonl` and `latent_mean_vectors.jsonl` had already been flushed
  batch by batch;
- `state.json` recorded the processed units up to the last persisted batch;
- re-running the baseline with the same `run_id` resumed from the persisted
  artifacts instead of starting from zero.

The interrupted full run had already stored `16` rows before re-entry and later
continued successfully with the same `run_id`.

This confirmed that the incremental execution policy is operational in practice,
not only in design.

## Final Completed Run

### Command

```bash
uv run python scripts/run_kubernetes_baseline.py --split test --max-new-tokens 320 --collect-latent-means --run-id compact-test70-320-vtfix --gpu-memory 5.4GiB
```

### Run identity

- `run_id`: `compact-test70-320-vtfix`
- split: `test`
- rows: `70`
- prompt variants: `question` and `question_simplified`
- output format: `blocks_tsv_compact_v1`
- latent collection: enabled

### Final artifacts

- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/config.json`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/state.json`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/predictions.jsonl`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/latent_mean_vectors.jsonl`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/metrics.json`
- `results/baseline_kubernetes_v1/compact-test70-320-vtfix/baseline_dashboard.html`

### Final metrics

- `row_count = 70`
- `evaluated_count = 62`
- `structured_output_parse_success_rate = 0.8857`
- `yaml_parse_success_rate = 0.4677`
- `parsed_equal_rate = 0.0161`
- `block_parse_success_rate = 0.4677`
- `document_count_match_rate = 0.6774`
- `line_count_match_rate = 0.0968`
- `average_line_text_f1 = 0.3175`
- `average_level_mae = 0.6100`
- `average_semantic_key_f1 = 0.4082`
- `primary_kind_match_rate = 0.6897`
- `primary_api_version_match_rate = 0.6552`
- `primary_metadata_name_match_rate = 0.7241`

Latent collection summary:

- `row_count = 70`
- `rows_with_vector = 70`
- `latent_dims = [3584]`

## Error Pattern Summary

The final baseline is not blocked by infrastructure anymore. Its weaknesses are
now mostly quality-related.

Status breakdown across the `70` rows:

- `29` rows: structured output parsed and reconstructed to valid YAML
- `33` rows: structured output parsed, but reconstructed YAML failed
- `8` rows: structured output itself failed before YAML evaluation

Most common final errors:

- `25` rows: `yaml_parse_error:ScannerError`
- `8` rows: `yaml_parse_error:ParserError`
- `4` rows: `structured_output_parse_error: ... not_enough_compact_tsv_fields`

Interpretation:

- the compact structured surface works often enough to make the run evaluable at
  full split scale;
- the main remaining gap is not resume safety or artifact persistence;
- the main remaining gap is structural generation quality in zero-shot mode.

## Sample Quality Snapshot

Representative successful rows show that the baseline often gets the broad
shape right:

- `apiVersion`
- `kind`
- `metadata.name`
- parts of `spec`

Representative failures still show:

- wrong top-level indentation;
- incomplete or truncated subtrees;
- list/mapping confusion in deeper structures;
- extra or misplaced fields that still produce valid YAML but do not match the
  reference structure closely.

This is consistent with the intended role of the baseline: it is measurable,
reproducible, and non-trivial, but not yet strong.

## What This Means For SFT

The final baseline is sufficient as a pre-SFT reference because it now gives:

- a completed full-split run;
- stable resumable execution;
- stored embeddings for every row;
- a repeatable dashboard and metric report;
- a clear map of where zero-shot generation still fails.

The next supervised stage should compare two ways of targeting exactly these
weaknesses:

- `serialized_sft`, where `level` remains part of the generated textual block
  representation;
- `two_head_sft`, where `level` is predicted by an explicit hierarchical-level
  head.

Both branches should be evaluated against:

- higher structural parse success on the model output itself;
- fewer YAML reconstruction failures;
- better line coverage and level accuracy;
- better semantic-key recall beyond the coarse top-level fields.

## Practical Conclusion

The baseline should now be treated as closed enough for the next phase.

It should not be polished endlessly through prompt engineering. The remaining
quality gap is already informative and is precisely the comparative space where
the serialized SFT control and the two-head SFT model are expected to separate.
