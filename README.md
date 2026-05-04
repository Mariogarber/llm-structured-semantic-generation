# Structured YAML Generation from Natural Language with Large Language Models

Master's Thesis repository focused on the generation of structured YAML configurations from natural language using large language models (LLMs), with a current effective case study centered on Kubernetes manifests.

## Overview

Large language models are autoregressive systems trained to continue flat token sequences. Many useful technical artifacts, however, are not flat in the way their semantics are organized. Kubernetes manifests are written as YAML text, but their meaning depends on a hierarchy of nested resources, fields, lists, and relations.

This repository studies that mismatch directly: can a flat token generator reliably produce outputs whose correctness depends on hierarchical structure? The current case study is Kubernetes manifest generation from natural-language instructions.

The project compares two formulations of the problem:

- structure as an output-surface problem, where the model emits a serialized structured representation and parser-based reconstruction plus later automatic-preference optimization are used to control the final YAML;
- structure as an explicit hierarchy-prediction problem, where generation order gives the left-to-right line position, but the model must also predict the missing hierarchical variable: the YAML `level` of each line.

The goal is not only to generate syntactically valid YAML, but also to produce configurations that are hierarchically well-formed, domain-compatible, and semantically coherent.

## Research Goal

The main objective of this project is to design and evaluate a system capable of transforming natural language instructions into structured YAML files that are:

- syntactically valid,
- hierarchically correct,
- compatible with the target domain schema,
- semantically coherent with the user request.

The project adopts a mathematical and analytical perspective, with a strong focus on formal evaluation, interpretable intermediate structure, and error analysis.

The central research question is whether explicit hierarchical supervision improves structured Kubernetes generation beyond treating hierarchy as serialized text learned only through the normal token-generation surface.

## Case Study

The long-term thesis motivation is structured YAML generation in technical domains. In the current repository state, the effective case study already prepared for preprocessing and modeling is Kubernetes manifest generation from natural-language descriptions.

This domain is especially useful because it:

- has a clear hierarchical structure,
- allows automatic validation,
- includes well-defined semantic constraints,
- supports the study of intermediate structural representations before final rendering.

## Main Hypothesis

The central hypothesis of this work is that structured YAML generation can be significantly improved when flat autoregressive decoding is complemented with explicit structural control and automatic validity-oriented evaluation.

The project treats `level` as the key missing structural variable. The position of a YAML entry can be recovered from generation order, but its height in the YAML hierarchy cannot be inferred reliably from order alone. That motivates the comparison between serialized hierarchy prediction and a dedicated hierarchical-level head.

More specifically, the project investigates whether:

- supervised fine-tuning (SFT) improves prompt-to-configuration mapping,
- parser-based structural control reduces invalid outputs more effectively than unconstrained YAML decoding,
- automatic preference or reward-based optimization is a practical post-SFT alignment branch without assuming a full RLHF pipeline,
- structural and semantic metrics are more informative than purely textual metrics,
- explicit latent intermediate representations can improve both generation quality and interpretability.

## Methodology

The project is organized around two comparable modeling branches and a shared evaluation stack.

### 1. Surface-Structured Branch

In this branch, hierarchy is represented on the text surface. The model emits a serialized block format that includes both line content and `level`, and the parser reconstructs YAML from that representation.

This branch supports:

- the zero-shot baseline output surface,
- the `serialized_sft` control model,
- later automatic-preference optimization such as DPO over parser-validated candidates.

It treats structural correctness as something that can be improved through output formatting, deterministic reconstruction, and preference signals.

### 2. Explicit-Hierarchy Branch

In this branch, YAML is treated as a tree-like object projected into ordered lines. The line position is supplied by the generation sequence, while the hierarchy is supervised explicitly through `level`.

This branch is represented by `two_head_sft`:

- one output predicts YAML line content,
- one explicit structural head predicts the hierarchical `level`,
- both are projected into the same parser-facing block contract before final YAML reconstruction.

### 3. Shared Semantic Modeling

Both branches must extract the relevant content from the prompt, including:

- requested resources,
- images,
- commands,
- environment variables,
- ports,
- policies and scheduling constraints,
- relations between configuration elements.

### 4. Shared Structural Control

Generation should not be left completely unconstrained. The project explores mechanisms that reduce the probability of invalid YAML structures, such as:

- parser-based structural control,
- rule- or schema-guided reconstruction,
- intermediate canonical representations before YAML rendering,
- richer structural auxiliary signals studied as comparative experiments.

### 5. Validity-Oriented Evaluation And Optimization

Generated outputs are automatically evaluated according to:

- YAML parsing validity,
- schema compliance,
- semantic consistency,
- fidelity to the input prompt,
- absence of internal contradictions.

These signals can be used for:

- best-sample selection,
- preference pair creation,
- reranking,
- post-SFT alignment or automatic-preference optimization.

## Experimental Setup

The repository is designed to compare several configurations:

1. Baseline with the current base model
2. SFT control model with serialized `level` prediction, representing hierarchy as text
3. SFT main model with an explicit hierarchical-level head, representing hierarchy as a supervised structural variable
4. Post-SFT alignment with automatic preferences, preferably DPO first
5. Comparative experiments with additional auxiliary structural signals
6. PPO only if later reward quality and compute make it worthwhile

## Dataset

The dataset consists of `(prompt, YAML file)` pairs focused on Kubernetes manifest generation.

The data pipeline includes:

- YAML normalization and canonicalization,
- removal of superficial stylistic variability,
- construction of intermediate structural targets,
- preparation for future comparative structural signals,
- strict train/validation/test separation to avoid leakage.

Current repository documents:

- Kubernetes preprocessing reference: [docs/KUBERNETES_PREPROCESSING.md](docs/KUBERNETES_PREPROCESSING.md)
- Kubernetes model v1 functional specification: [docs/KUBERNETES_MODEL_V1.md](docs/KUBERNETES_MODEL_V1.md)
- Structural target contract: [docs/STRUCTURAL_TARGETS_V1.md](docs/STRUCTURAL_TARGETS_V1.md)
- Baseline execution contract: [docs/BASELINE_V1.md](docs/BASELINE_V1.md)
- SFT strategy and model comparison: [docs/SFT_STRATEGY_V1.md](docs/SFT_STRATEGY_V1.md)
- Current project status and anteproyecto note: [docs/PROJECT_STATUS_AND_ANTEPROYECTO.md](docs/PROJECT_STATUS_AND_ANTEPROYECTO.md)
- Project terminology and metric definitions: [docs/TERMINOLOGY.md](docs/TERMINOLOGY.md)
- Multi-resource enrichment decision: [docs/MULTI_RESOURCE_STRATEGY_DECISION.md](docs/MULTI_RESOURCE_STRATEGY_DECISION.md)

## Current Implementation Status

The Kubernetes v1 dataset is processed and ready for the next modeling stage. The repository now includes:

- a reproducible Kubernetes preprocessing script,
- processed train/validation/test artifacts under `data/processed/kubernetes_v1/`,
- a line-and-level structural target builder,
- deterministic reconstruction from structural blocks back to YAML,
- structural evaluation helpers,
- fixed SFT serialization rows derived from structural targets,
- the documented two-branch SFT comparison:
  `serialized_sft` vs `two_head_sft`,
- a zero-shot baseline runner for the local Qwen model.

The repository does not yet contain completed SFT, DPO, or PPO result tables.
Baseline artifacts can now be produced end to end with the included runners, but
no baseline output should be treated as a validated thesis conclusion until the
recorded runs have been reviewed explicitly. The current baseline execution
process and recorded test run are documented in
`docs/BASELINE_EXECUTION_REPORT_2026-04-28.md`.

The next dataset-enrichment direction is documented as `kubernetes_v2`: a
derived multi-resource dataset built from `kubernetes_v1` through controlled
compositional oversampling. This is an experimental branch and does not replace
the clean `kubernetes_v1` baseline.

## Incremental LLM Execution Policy

All LLM-related runs in this repository must be incremental and resumable by
default. This is now a permanent project policy, intended for slow or
interruptible machines where long runs cannot safely assume uninterrupted
execution.

The current operative contract is:

- LLM scripts must support `--output-dir`, `--run-id`, and `--batch-size`.
- Work must be persisted at the end of each batch, with `batch_size=1` as the
  default safety setting.
- `config.json` stores the full run configuration.
- `state.json` stores the live run state and processed progress.
- Partial artifacts such as `predictions.jsonl` are append-only during normal
  execution and are the source of truth for completed work during resume.
- `metrics.json` is written only when a run completes successfully.

Future SFT, DPO, validation, and training scripts must follow the same policy
from their first implementation. In particular, training code must also persist
resumable model and optimizer checkpoints rather than assuming monolithic runs.

## Reproducible Commands

Build the base processed dataset:

```bash
uv run python utils/kubernetes_dataset_preprocessor.py
```

Build the line-and-level structural targets:

```bash
uv run python scripts/build_kubernetes_structural_targets.py
```

Build SFT-ready JSONL files from the structural targets:

```bash
uv run python scripts/build_kubernetes_sft_dataset.py
```

Build the descriptive dataset analysis report before training:

```bash
uv run python scripts/analyze_kubernetes_dataset.py
```

Validate the baseline inputs without loading the model:

```bash
uv run python scripts/run_kubernetes_baseline.py --dry-run
```

Run the zero-shot baseline on the validation split after installing the optional LLM dependencies:

```bash
uv sync --extra llm
uv run python scripts/run_kubernetes_baseline.py --split validation --run-id baseline-val
```

The baseline now requests `blocks_tsv_compact_v1` by default instead of the
older JSON array output. This compact inference-only surface removes
model-predicted `line_index`, reconstructs ordering deterministically in the
parser, and is substantially cheaper in output tokens while preserving the same
parser-facing block contract.

The supervised SFT source remains `blocks_tsv_v1`. The inference baseline and
the supervised target format are therefore intentionally related but not
identical:

- `blocks_tsv_compact_v1` is the default for baseline inference because it is
  cheaper and more robust on limited hardware.
- `blocks_tsv_v1` is the direct target for the `serialized_sft` control branch.
- the same `blocks_tsv_v1` rows provide line-content labels and per-line
  `level` labels for the `two_head_sft` main branch.

The central SFT comparison is:

```text
serialized_sft vs two_head_sft
```

The purpose is to test whether an explicit hierarchical-level head improves
structured YAML generation beyond learning hierarchy as serialized text.

The dry run records whether the local model directory has the tokenizer files
and quantization dependencies needed for a full run.

## Evaluation

Traditional text generation metrics such as BLEU are not sufficient for this problem. The evaluation framework therefore emphasizes structural and semantic quality.

### Structural Metrics

- percentage of valid YAML outputs,
- schema validation success rate,
- exact key-level match,
- tree-based structural similarity,
- hierarchical consistency.

### Semantic Metrics

- presence of required resources,
- coherent use of ports, environment variables, and execution settings,
- absence of references to nonexistent elements,
- domain-level validation when possible.

### Prompt Adequacy Metrics

- coverage of requested requirements,
- absence of unnecessary content,
- correspondence between natural language constraints and generated configuration.

### Robustness Metrics

- sensitivity to ambiguous prompts,
- degradation under noisy input,
- generalization to unseen configuration patterns.

## Repository Structure

```text
.
|-- data/                # Raw and processed dataset artifacts
|-- docs/                # Project, preprocessing, modeling, and experiment contracts
|-- exploratory/         # Exploratory notebooks and data profiling artifacts
|-- model/               # Local base model artifacts
|-- scripts/             # Reproducible dataset, SFT, and baseline commands
|-- src/                 # Structural conversion, serialization, and evaluation code
|-- tests/               # Unit tests for structural targets and parser behavior
|-- utils/               # Current preprocessing and utility scripts
|-- README.md
`-- pyproject.toml
```
