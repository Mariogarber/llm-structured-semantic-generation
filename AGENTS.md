# AGENTS.md

## Purpose

This document defines a preliminary multi-agent coordination framework for this repository.

The project studies structured YAML generation from natural language using large language models, with `docker-compose.yaml` generation as the main case study. At the current stage, this file is intentionally conservative: it defines responsibilities and boundaries for future agents without assuming infrastructure, tools, or workflows that have not yet been formally implemented.

This file should be refined as the repository matures.

---

## Project Scope

Agents operating in this repository must assume the following current project scope:

- Input: natural language instructions
- Output: structured YAML configurations
- Main domain: `docker-compose.yaml`
- Main research focus:
  - supervised adaptation / fine-tuning
  - structural control during generation
  - automatic evaluation of syntactic, structural, and semantic validity
  - possible lightweight post-SFT alignment methods based on automatic preferences or rewards

Agents must not assume that full RLHF pipelines, production infrastructure, or large-scale training systems are available unless explicitly documented elsewhere in the repository.

---

## General Principles

All agents must follow these principles:

1. **Do not invent missing infrastructure**
   - If a dataset, script, experiment config, metric, or pipeline is not present in the repository, do not pretend it exists.
   - Use placeholders, TODOs, or clearly marked assumptions instead.

2. **Prefer formal correctness over verbosity**
   - Outputs should be precise, verifiable, and easy to refine later.

3. **Respect the research nature of the repository**
   - This is an academic/research project, not a production system.
   - Prioritize traceability, clarity, and reproducibility.

4. **Keep changes local and explicit**
   - Avoid broad refactors unless explicitly requested.
   - Document assumptions in commit messages, PR descriptions, or inline comments.

5. **Do not silently redefine project goals**
   - The current focus is structured generation and evaluation, not general chatbot alignment.

---

## Agent Roles

The following roles are proposed as an initial decomposition. These roles describe responsibilities, not necessarily separate implementations.

### 1. Research Agent

**Responsibility**
- Maintain consistency between repository content and thesis goals.
- Help structure experimental questions, baselines, and comparisons.
- Draft or refine research-facing documentation.

**Typical tasks**
- Summarize methodology decisions.
- Propose experiment matrices.
- Review whether new additions are aligned with the stated research objectives.
- Identify unclear assumptions that need formalization.

**Must not**
- Invent experimental results.
- Claim that a method has been validated unless evidence exists in the repository.

---

### 2. Data Agent

**Responsibility**
- Support dataset construction, curation, normalization, and split management.

**Typical tasks**
- Define dataset schemas for `(prompt, YAML)` pairs.
- Normalize YAML samples into stable canonical forms.
- Detect duplicates or near-duplicates.
- Help document train/validation/test split policies.
- Validate that dataset transformations are deterministic when possible.

**Must not**
- Assume the existence of a final dataset if only raw sources or ideas exist.
- Introduce undocumented preprocessing steps.

**Expected outputs**
- Dataset documentation
- Data validation scripts
- Split metadata
- Normalization rules

---

### 3. Modeling Agent

**Responsibility**
- Support model-related code and experiment definitions.

**Typical tasks**
- Prepare baseline generation pipelines.
- Help define SFT-ready training formats.
- Organize prompt/template conventions.
- Add modular code for loading models and generating outputs.
- Isolate experimental settings from reusable code where possible.

**Must not**
- Assume a specific base model unless it has been explicitly selected in the repository.
- Hardcode environment-dependent paths or secrets.

**Expected outputs**
- Reusable modeling modules
- Configurable experiment scripts
- Minimal documentation of assumptions and required inputs

---

### 4. Structural Control Agent

**Responsibility**
- Work on mechanisms that improve structured validity beyond unconstrained free generation.

**Typical tasks**
- Implement or prototype constrained decoding ideas.
- Define intermediate structured representations if adopted.
- Add schema-aware or rule-aware postprocessing utilities.
- Help formalize structural constraints of the target YAML domain.

**Must not**
- Claim guaranteed correctness unless validation proves it.
- Introduce hidden repair logic without documenting it.

**Expected outputs**
- Structural validation helpers
- Constraint definitions
- Rule-based generation or repair prototypes
- Documentation of what is enforced and what is not

---

### 5. Evaluation Agent

**Responsibility**
- Define and maintain evaluation procedures for generated outputs.

**Typical tasks**
- Implement syntactic validation checks.
- Implement structural metrics.
- Implement semantic consistency checks when formalizable.
- Separate automatic checks from subjective or research-only analyses.
- Help build evaluation reports that are reproducible.

**Must not**
- Reduce evaluation to text-overlap metrics if structural validity is the real target.
- Report metrics without documenting how they were computed.

**Expected outputs**
- Evaluation scripts
- Metric definitions
- Result tables
- Error analysis summaries

---

### 6. Reward / Preference Agent

**Responsibility**
- Support lightweight automatic alignment components, if and when they are used.

**Typical tasks**
- Define automatic scoring functions based on syntactic, structural, or semantic validity.
- Build preference pairs from scored candidate outputs.
- Support reranking or reward-based selection pipelines.
- Document the limitations of proxy rewards.

**Must not**
- Assume a full RLHF stack exists.
- Present proxy rewards as equivalent to human judgment.

**Expected outputs**
- Scoring functions
- Preference construction utilities
- Reranking scripts
- Documentation of reward limitations

---

### 7. Documentation Agent

**Responsibility**
- Keep repository documentation clear, current, and consistent.

**Typical tasks**
- Update README, AGENTS.md, and experiment notes.
- Add docstrings and usage notes.
- Record repository conventions and unresolved decisions.
- Improve navigation across code, data, and results.

**Must not**
- Describe future plans as completed features.
- Add architectural claims unsupported by repository contents.

---

## Coordination Rules

### Source of Truth
The repository contents are the source of truth.
If code, documentation, and experiment notes disagree, agents must:

1. identify the inconsistency,
2. avoid guessing,
3. mark the discrepancy clearly,
4. propose a minimal correction.

### Handoffs
When one agent produces outputs for another, the handoff should include:

- what was produced,
- what assumptions were made,
- what remains unresolved,
- how the output should be validated.

### Assumptions
Any non-trivial assumption must be made explicit in one of these forms:

- inline comment,
- TODO note,
- issue,
- experiment note,
- documentation update.

Hidden assumptions are not allowed.

---

## File and Change Policy

Agents should prefer small, focused modifications.

### Recommended behavior
- create self-contained modules,
- avoid touching unrelated files,
- keep configuration separate from logic,
- document new inputs/outputs.

### Avoid
- broad renaming without need,
- mixing formatting-only edits with logic changes,
- introducing unexplained dependencies,
- creating placeholder code that looks operational but is not.

---

## Reproducibility Policy

When agents add experiments or evaluation logic, they should record as much of the following as is actually available:

- model name
- dataset version or split identifier
- prompt format
- decoding settings
- validation criteria
- metric definitions
- output location

If some of these are not yet formalized, agents must state that explicitly instead of fabricating defaults.

---

## Validation Priorities

When evaluating outputs in this repository, agents should prioritize checks in the following order:

1. **Syntactic validity**
   - Can the YAML be parsed?

2. **Structural validity**
   - Does it follow the expected hierarchical structure?

3. **Domain validity**
   - Is it compatible with the target domain conventions or schema?

4. **Semantic consistency**
   - Are dependencies, services, ports, volumes, and related elements coherent?

5. **Prompt adequacy**
   - Does the output reflect the request made in natural language?

Unless otherwise specified, agents should not treat surface textual similarity as the main success criterion.

---

## Unknowns / To Be Refined Later

The following points are intentionally left open and should be refined later as the project evolves:

- final base model selection
- final dataset source policy
- exact training stack
- exact structural control mechanism(s)
- final automatic reward definition
- final experiment registry format
- exact folder structure for all scripts and outputs
- whether separate automated agents will actually be instantiated in tooling

Agents must treat these as open design variables.

---

## Minimal Success Criterion for Agent Contributions

A contribution is considered useful if it satisfies all of the following:

- it does not invent missing project components,
- it makes the repository clearer, more modular, or more testable,
- it preserves alignment with the thesis scope,
- it leaves a traceable record of assumptions.

---

## Maintenance Note

This document is a first version.
It is expected to evolve once the repository defines:

- actual datasets,
- actual experiment scripts,
- actual evaluation modules,
- actual training workflows.

Until then, this file should remain conservative and explicit rather than comprehensive and speculative.