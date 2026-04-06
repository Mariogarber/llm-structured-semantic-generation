# Structured YAML Generation from Natural Language with Large Language Models

Master’s Thesis repository focused on the generation of structured YAML configurations from natural language using large language models (LLMs), with `docker-compose.yaml` as the main case study.

## Overview

Large language models are highly effective at free-text generation, but they remain less reliable when the target output must satisfy strict formal and semantic constraints. In this project, the goal is not only to generate syntactically valid YAML, but also to produce configurations that are hierarchically well-formed, domain-compatible, and semantically coherent.

This repository studies how different adaptation and alignment strategies can improve structured generation in technical domains.

## Research Goal

The main objective of this project is to design and evaluate a system capable of transforming natural language instructions into structured YAML files that are:

- syntactically valid,
- hierarchically correct,
- compatible with the target domain schema,
- semantically coherent with the user request.

The project adopts a mathematical and analytical perspective, with a strong focus on formal evaluation and error analysis.

## Case Study

The project is centered on the generation of `docker-compose.yaml` files from natural language descriptions.

This domain is especially useful because it:

- has a clear hierarchical structure,
- allows automatic validation,
- includes well-defined semantic constraints,
- represents a realistic technical structured generation task.

## Main Hypothesis

The central hypothesis of this work is that structured YAML generation can be significantly improved when supervised fine-tuning is complemented with explicit structural control and automatic validity-oriented evaluation, instead of relying only on unconstrained autoregressive decoding.

More specifically, the project investigates whether:

- supervised fine-tuning (SFT) improves prompt-to-configuration mapping,
- structural control mechanisms reduce invalid outputs more effectively than weak auxiliary signals,
- automatic preference or reward-based optimization is more practical than full RLHF in this setting,
- structural and semantic metrics are more informative than purely textual metrics.

## Methodology

The project is organized into three main levels:

### 1. Semantic Modeling
The model must extract the relevant content from the prompt, including:

- requested services,
- ports,
- volumes,
- environment variables,
- dependencies,
- networks,
- deployment constraints.

### 2. Structural Control
Generation should not be left completely unconstrained. The project explores mechanisms that reduce the probability of invalid YAML structures, such as:

- constrained decoding,
- rule- or schema-guided generation,
- intermediate canonical representations before YAML rendering,
- richer structural auxiliary objectives.

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

1. Zero-shot / few-shot baseline
2. Supervised Fine-Tuning (SFT)
3. SFT with efficient adaptation (e.g. LoRA)
4. SFT + intermediate representation
5. SFT + structural control during decoding
6. SFT + structural auxiliary objective
7. Post-SFT optimization using automatic preferences or reward-based reranking

## Dataset

The dataset consists of `(prompt, YAML file)` pairs, focused on a technical domain such as Docker Compose.

The data pipeline includes:

- YAML normalization and canonicalization,
- removal of superficial stylistic variability,
- semantic data augmentation,
- controlled synthetic data generation,
- strict train/validation/test separation to avoid leakage.

## Evaluation

Traditional text generation metrics such as BLEU are not sufficient for this problem. The evaluation framework therefore emphasizes structural and semantic quality.

### Structural Metrics

- percentage of valid YAML outputs,
- schema validation success rate,
- exact key-level match,
- tree-based structural similarity,
- hierarchical consistency.

### Semantic Metrics

- presence of required services,
- correct dependency specification,
- coherent use of ports, volumes, and networks,
- absence of references to nonexistent elements,
- domain-level execution or sandbox validation when possible.

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
├── data/                # Datasets, splits, normalization scripts
├── notebooks/           # Exploratory analysis and experiments
├── src/
│   ├── data/            # Dataset building and preprocessing
│   ├── modeling/        # Model loading, LoRA/SFT pipelines
│   ├── decoding/        # Structural control and constrained decoding
│   ├── reward/          # Automatic scoring / preference generation
│   ├── evaluation/      # Structural, semantic, and robustness metrics
│   └── utils/           # Helpers and utilities
├── experiments/         # Experiment configs and logs
├── results/             # Outputs, tables, plots, and reports
├── README.md
└── requirements.txt