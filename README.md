# Structured YAML Generation from Natural Language with Large Language Models

Master's Thesis repository focused on the generation of structured YAML configurations from natural language using large language models (LLMs), with a current effective case study centered on Kubernetes manifests.

## Overview

Large language models are highly effective at free-text generation, but they remain less reliable when the target output must satisfy strict formal and semantic constraints. In this project, the goal is not only to generate syntactically valid YAML, but also to produce configurations that are hierarchically well-formed, domain-compatible, and semantically coherent.

This repository studies how different adaptation and alignment strategies can improve structured generation in technical domains, with an explicit interest in intermediate representations, structural control, and more analyzable formulations of the problem.

## Research Goal

The main objective of this project is to design and evaluate a system capable of transforming natural language instructions into structured YAML files that are:

- syntactically valid,
- hierarchically correct,
- compatible with the target domain schema,
- semantically coherent with the user request.

The project adopts a mathematical and analytical perspective, with a strong focus on formal evaluation, interpretable intermediate structure, and error analysis.

## Case Study

The long-term thesis motivation is structured YAML generation in technical domains. In the current repository state, the effective case study already prepared for preprocessing and modeling is Kubernetes manifest generation from natural-language descriptions.

This domain is especially useful because it:

- has a clear hierarchical structure,
- allows automatic validation,
- includes well-defined semantic constraints,
- supports the study of intermediate structural representations before final rendering.

## Main Hypothesis

The central hypothesis of this work is that structured YAML generation can be significantly improved when supervised fine-tuning is complemented with explicit structural control and automatic validity-oriented evaluation, instead of relying only on unconstrained autoregressive decoding.

More specifically, the project investigates whether:

- supervised fine-tuning (SFT) improves prompt-to-configuration mapping,
- structural control mechanisms reduce invalid outputs more effectively than weak auxiliary signals,
- automatic preference or reward-based optimization is more practical than full RLHF in this setting,
- structural and semantic metrics are more informative than purely textual metrics,
- explicit latent intermediate representations can improve both generation quality and interpretability.

## Methodology

The project is organized into three main levels:

### 1. Semantic Modeling

The model must extract the relevant content from the prompt, including:

- requested resources,
- images,
- commands,
- environment variables,
- ports,
- policies and scheduling constraints,
- relations between configuration elements.

### 2. Structural Control

Generation should not be left completely unconstrained. The project explores mechanisms that reduce the probability of invalid YAML structures, such as:

- parser-based structural control,
- rule- or schema-guided reconstruction,
- intermediate canonical representations before YAML rendering,
- richer structural auxiliary signals studied as comparative experiments.

### 3. Validity-Oriented Optimization

Generated outputs are automatically evaluated according to:

- YAML parsing validity,
- schema compliance,
- semantic consistency,
- fidelity to the input prompt,
- absence of internal contradictions.

This signal can be used for:

- best-sample selection,
- preference pair creation,
- reranking,
- post-SFT alignment or automatic-preference optimization.

## Experimental Setup

The repository is designed to compare several configurations:

1. Baseline with the current base model
2. Supervised Fine-Tuning (SFT) with efficient adaptation (LoRA)
3. Comparative experiments with intermediate representations and auxiliary structural signals
4. Post-SFT alignment with automatic preferences (DPO)
5. PPO only if later reward quality and compute make it worthwhile

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
- Project terminology and metric definitions: [docs/TERMINOLOGY.md](docs/TERMINOLOGY.md)

## Current Implementation Status

The Kubernetes v1 dataset is processed and ready for the next modeling stage. The repository now includes:

- a reproducible Kubernetes preprocessing script,
- processed train/validation/test artifacts under `data/processed/kubernetes_v1/`,
- a line-and-level structural target builder,
- deterministic reconstruction from structural blocks back to YAML,
- structural evaluation helpers,
- fixed SFT serialization rows derived from structural targets,
- a zero-shot baseline runner for the local Qwen model.

The repository does not yet contain completed SFT, DPO, PPO, or validated baseline result tables. Those must be produced by running the experiment scripts and recording their outputs.

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
uv run python scripts/run_kubernetes_baseline.py --split validation
```

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
