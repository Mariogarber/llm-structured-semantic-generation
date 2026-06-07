# DPO Preference Annotation And Prompt Audit v1

Document type: contract

## Purpose

This document records the implemented local tooling for building and auditing
the first DPO preference dataset.

The tooling is designed for offline DPO. It does not run online policy
rollouts, train a reward model, or replace human judgement with the current
automatic metrics. It creates a review layer over generated candidates so that
human and agent-assisted choices can be made traceable before DPO training.

## Local UI

Run:

```powershell
uv run python scripts/serve_dpo_preference_ui.py --output-dir results/dpo_kubernetes_v1/preference_annotation/manual-v1 --run-id dpo-preference-ui-v1 --batch-size 50
```

The server loads DPO candidates from:

```text
results/dpo_kubernetes_v1/candidate_generation/
```

By default it reviews only the `train` split.

The UI shows:

- natural-language prompt and SFT prompt context;
- reference YAML;
- extracted prompt requirements;
- candidate model output and reconstructed YAML;
- current evaluation metrics and preference score;
- the DPO labeling guide used by both human and agent decisions;
- human `chosen` / `rejected`, `tie`, or `skip` decisions;
- pending agent suggestions.

## Artifacts

The annotation output directory contains:

- `config.json`
- `annotation_state.json`
- `preferences_human.jsonl`
- `preferences_agent_suggestions.jsonl`
- `preferences_final.jsonl`
- `prompt_requirement_gold.jsonl`
- `prompt_requirement_audit_report.json`

The human and agent files are append-only event logs. `preferences_final.jsonl`
is a materialized export of the latest approved preference decisions and is the
file intended for later DPO dataset construction.

## Preference Policy

Approved human decisions take priority over approved agent decisions.
Pending agent suggestions are never exported as final DPO pairs.

The labeling policy is defined in
[`DPO_LABELING_GUIDE_V1.md`](DPO_LABELING_GUIDE_V1.md). The same guide is also
included in the agent decision packet copied from the UI.

Each final row preserves:

- `unit_id`, `sample_id`, `prompt_variant`, and `split`;
- the SFT prompt;
- chosen and rejected model outputs;
- reconstructed YAML for inspection;
- metrics snapshots;
- annotation source, confidence, rationale, and metric flags.

The test split must not be used for preference construction, metric-weight
tuning, beta selection, or failure inspection during DPO development.

## Prompt Requirement Audit

The UI also seeds a small stratified prompt-requirement audit set from the train
split. The seed prioritizes:

- unsupported prompts;
- prompts with low reference F1 under the current extractor;
- prompts that look exact but have very thin extraction;
- multi-resource prompts;
- random train examples.

Human-reviewed gold requirements are appended to `prompt_requirement_gold.jsonl`.
The report compares the current regex/rule extractor against those gold atoms
and records precision, recall, and F1 overall and by category.

Until this gold audit is reviewed, `prompt_requirement_f1` remains an auxiliary
proxy signal. It must not be treated as a complete semantic oracle.
