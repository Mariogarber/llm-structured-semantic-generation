# AGENTS.md

## Purpose

This document defines the operational coordination framework for agents working in this repository.

It is intentionally specific to the current state of the project. The repository is no longer at a purely generic "structured YAML generation" stage: it already has a concrete dataset, a fixed preprocessing contract, a current effective case study, and a documented modeling direction.

Agents must use this file to stay aligned with the actual case of use under development, not with a broader or historical formulation of the thesis.

---

## Current Project Scope

Agents operating in this repository must assume the following scope unless a newer repository document explicitly overrides it:

- Input: natural-language instructions
- Output: structured YAML manifests
- Current effective case study: `Kubernetes`
- Historical broader thesis motivation: structured YAML generation in technical domains, including `docker-compose.yaml`
- Base dataset for the main line of experiments: `Kubernetes v1`
- Main processed dataset location: `data/processed/kubernetes_v1/`
- Main preprocessing contract: `docs/KUBERNETES_PREPROCESSING.md`
- Main modeling contract: `docs/KUBERNETES_MODEL_V1.md`
- Current base model explicitly referenced in the documentation: `model/qwen2.5-7b-instruct-4bit/`

The main research focus of the current repository state is:

- supervised adaptation for structured generation
- explicit latent intermediate representations before block generation
- projection from latent structure to YAML lines with hierarchical level
- parser-based structural control at the final generation stage
- automatic evaluation of syntactic, structural, and semantic validity
- lightweight post-SFT alignment methods based on automatic preferences or rewards

Agents must not assume that full RLHF pipelines, production infrastructure, human preference datasets, or large-scale training systems are available unless explicitly documented in the repository.

---

## Closed Decisions For The Main Line

The following decisions are already fixed for the core experimental path and must not be silently redefined:

1. The effective case study is `Kubernetes`, not `docker-compose.yaml`.
2. The base dataset is `Kubernetes v1`.
3. The core project uses a single main dataset version.
4. Oversampling, synthetic enlargement, or dataset growth are not part of the base v1 contract.
5. The main modeling narrative is:
   - `prompt -> latent intermediate representation -> blocks with level -> parser as structural control -> final YAML`
6. The blocks with level are not the latent space itself.
7. The parser is not just a formatter; it is the structural control module.
8. The main experimental sequence is:
   - baseline
   - SFT + LoRA
   - comparative branch with auxiliary structural signals
   - DPO
   - PPO only as an optional later extension
9. Surface text overlap is not the primary success criterion.

If an agent proposes work that conflicts with one of these points, it must first update the relevant documentation explicitly instead of silently changing direction in code or prose.

---

## Main Artifacts Agents Should Know

Before making significant changes, agents should understand the role of these repository artifacts:

- `docs/KUBERNETES_PREPROCESSING.md`
  - defines the preprocessing and dataset contract
- `docs/KUBERNETES_MODEL_V1.md`
  - defines the modeling and research contract
- `README.md`
  - presents the project-level framing
- `utils/kubernetes_dataset_preprocessor.py`
  - current preprocessing implementation for `Kubernetes v1`
- `data/processed/kubernetes_v1/`
  - generated manifests, splits, train-ready exports, and quality report

Agents must treat these artifacts as the current source of truth for the implemented case of use.

---

## General Principles

All agents must follow these principles:

1. **Do not invent missing infrastructure**
   - If a dataset, script, experiment config, metric, parser, latent representation, or training pipeline is not present in the repository, do not pretend it exists.
   - Use placeholders, TODOs, or clearly marked assumptions instead.

2. **Prefer formal correctness over verbosity**
   - Outputs should be precise, verifiable, and easy to refine later.

3. **Respect the research nature of the repository**
   - This is an academic project, not a production system.
   - Prioritize traceability, clarity, reproducibility, and explicit assumptions.

4. **Keep changes local and explicit**
   - Avoid broad refactors unless explicitly requested.
   - Document assumptions in code comments, experiment notes, or documentation updates.

5. **Do not silently redefine project goals**
   - The focus is structured generation, structural control, latent-space analysis, and evaluation.
   - This repository is not a general chatbot project.

6. **Do not silently switch domains**
   - Do not revert the operative case study back to `docker-compose.yaml` in code or docs unless the repository is intentionally re-scoped.
   - If historical thesis text mentions `docker-compose.yaml`, agents should clarify the distinction instead of forcing the whole repo back to that wording.

7. **Do not overstate maturity**
   - A conceptual model contract is not the same thing as a completed implementation.
   - A documented experimental branch is not the same thing as a validated result.

---

## Agent Roles

The following roles describe responsibilities, not necessarily separate implementations.

### 1. Research Agent

**Responsibility**
- Maintain consistency between repository content and the actual thesis direction.
- Keep the experimental narrative aligned with the current Kubernetes-based case of use.
- Ensure the analytical angle of the thesis is reflected in documents and experiment design.

**Typical tasks**
- Summarize methodology decisions.
- Propose experiment matrices.
- Review whether new additions align with the current modeling contract.
- Check whether latent-space analysis, parser-based control, and evaluation are described consistently.

**Must not**
- Invent experimental results.
- Claim that the latent representation, parser, reward, or auxiliary signals have been validated unless evidence exists in the repository.

---

### 2. Data Agent

**Responsibility**
- Support dataset construction, curation, normalization, split management, and structural target derivation for `Kubernetes v1`.

**Typical tasks**
- Maintain schemas for `(prompt, YAML)` pairs.
- Normalize YAML samples into stable canonical forms.
- Detect duplicates or near-duplicates.
- Maintain train/validation/test split policy and leakage-group logic.
- Support derivations from canonical YAML to:
  - final YAML targets
  - line-and-level block targets
  - future latent or auxiliary structural representations

**Must not**
- Assume that raw ideas or source folders already constitute a finished dataset.
- Introduce undocumented preprocessing steps.
- Treat oversampling or synthetic enlargement as part of the base v1 data contract unless explicitly documented as a separate experiment.

**Expected outputs**
- Dataset documentation
- Data validation scripts
- Split metadata
- Normalization rules
- Derived target-format specifications

---

### 3. Modeling Agent

**Responsibility**
- Support model-related code and experiment definitions for the current main path:
  - baseline
  - SFT + LoRA
  - DPO
  - PPO only if later justified

**Typical tasks**
- Prepare baseline generation pipelines.
- Define SFT-ready training formats.
- Organize prompt and output serialization conventions.
- Add modular code for model loading and generation.
- Respect the documented contract:
  - latent intermediate representation first
  - projection to line text plus level
  - parser-controlled YAML output

**Must not**
- Ignore the latent representation layer and collapse the whole system back into unconstrained YAML generation without documenting that decision.
- Hardcode environment-dependent paths or secrets.
- Pretend that comparative auxiliary signals are part of the mandatory core path if they are only experimental.

**Expected outputs**
- Reusable modeling modules
- Configurable experiment scripts
- Minimal documentation of assumptions and required inputs

---

### 4. Structural Control Agent

**Responsibility**
- Work on mechanisms that improve structured validity beyond unconstrained free generation, with the parser as the main current control mechanism.

**Typical tasks**
- Define how predicted blocks with level are converted into final YAML.
- Formalize structural constraints of the target YAML domain.
- Add parser-side validation and deterministic reconstruction utilities.
- Explore auxiliary structural signals useful for control or comparative experiments.

**Must not**
- Claim guaranteed correctness unless validation proves it.
- Introduce hidden repair logic.
- Invent missing content at parse time.
- Use the parser to silently rescue semantically wrong outputs.

**Expected outputs**
- Structural validation helpers
- Constraint definitions
- Deterministic parser or reconstruction utilities
- Documentation of what is enforced and what is not

---

### 5. Evaluation Agent

**Responsibility**
- Define and maintain evaluation procedures for generated Kubernetes manifests and their intermediate forms.

**Typical tasks**
- Implement syntactic validation checks.
- Implement structural metrics.
- Implement semantic consistency checks when formalizable.
- Separate automatic checks from interpretability or research-only analyses.
- Help build reproducible evaluation reports across baseline, SFT, and DPO.
- Support analysis of whether latent-space properties correlate with structural success or failure.

**Must not**
- Reduce evaluation to text-overlap metrics if structural validity is the real target.
- Report metrics without documenting how they were computed.
- Treat parser success alone as a complete measure of semantic correctness.

**Expected outputs**
- Evaluation scripts
- Metric definitions
- Result tables
- Error analysis summaries
- Latent-space analysis summaries when available

---

### 6. Reward / Preference Agent

**Responsibility**
- Support lightweight automatic alignment components for the post-SFT stages.

**Typical tasks**
- Define automatic scoring functions based on syntactic, structural, or semantic validity.
- Build preference pairs from scored candidate outputs.
- Support reranking or reward-based selection pipelines.
- Document the limitations of proxy rewards.

**Must not**
- Assume a full RLHF stack exists.
- Present proxy rewards as equivalent to human judgment.
- Introduce reward terms that are not traceable to documented validation logic.

**Expected outputs**
- Scoring functions
- Preference construction utilities
- Reranking scripts
- Documentation of reward limitations

---

### 7. Documentation Agent

**Responsibility**
- Keep repository documentation clear, current, and aligned with the implemented case of use.

**Typical tasks**
- Update `README.md`, `AGENTS.md`, preprocessing docs, and modeling docs.
- Record repository conventions and unresolved decisions.
- Keep the Kubernetes-based project narrative consistent.
- Flag discrepancies between historical thesis wording and current repository reality.

**Must not**
- Describe future plans as completed features.
- Add architectural claims unsupported by repository contents.
- Reintroduce generic or outdated framing that contradicts current repository documents.

---

## Coordination Rules

### Source of Truth

The repository contents are the source of truth.
If code, documentation, and experiment notes disagree, agents must:

1. identify the inconsistency,
2. avoid guessing,
3. mark the discrepancy clearly,
4. propose a minimal correction.

Current precedence for the operative case of use should be interpreted as:

1. implemented artifacts and generated data
2. explicit repository documentation for Kubernetes preprocessing/modeling
3. older generic or historical wording

### Handoffs

When one agent produces outputs for another, the handoff should include:

- what was produced,
- what assumptions were made,
- what remains unresolved,
- how the output should be validated,
- whether the output belongs to the main experimental path or to a comparative side branch.

### Assumptions

Any non-trivial assumption must be made explicit in one of these forms:

- inline comment,
- TODO note,
- issue,
- experiment note,
- documentation update.

Hidden assumptions are not allowed.

---

## File And Change Policy

Agents should prefer small, focused modifications.

### Recommended behavior

- create self-contained modules,
- avoid touching unrelated files,
- keep configuration separate from logic,
- document new inputs and outputs,
- update the relevant documentation when changing data contracts or modeling contracts.

### Avoid

- broad renaming without need,
- mixing formatting-only edits with logic changes,
- introducing unexplained dependencies,
- creating placeholder code that looks operational but is not,
- adding experimental claims that are not backed by current artifacts.

---

## Reproducibility Policy

When agents add experiments or evaluation logic, they should record as much of the following as is actually available:

- model name
- dataset version or split identifier
- prompt format
- target serialization
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
   - Does it follow the expected hierarchy and block-level organization?

3. **Parser-level control validity**
   - Can the predicted structured representation be safely reconstructed by the parser without hidden repair?

4. **Domain validity**
   - Is it compatible with Kubernetes conventions or schema-like expectations?

5. **Semantic consistency**
   - Are resources, fields, and relations coherent?

6. **Prompt adequacy**
   - Does the output reflect the request made in natural language?

Unless otherwise specified, agents should not treat surface textual similarity as the main success criterion.

---

## Open Variables

The following points are intentionally left open and should be refined later as the project evolves:

- exact parametrization of the latent intermediate representation
- exact serialization used to train latent and block-level targets
- exact loss decomposition for content vs structure
- exact parser implementation details
- final automatic reward definition and weighting
- final experiment registry format
- whether PPO becomes worthwhile after DPO
- whether oversampling or synthetic enlargement later become useful side experiments

Agents must treat these as open design variables and must not silently close them in code or documentation without marking the decision.

---

## Minimal Success Criterion For Agent Contributions

A contribution is considered useful if it satisfies all of the following:

- it does not invent missing project components,
- it is aligned with the current Kubernetes-centered case of use,
- it makes the repository clearer, more modular, or more testable,
- it preserves the latent-plus-structural-control framing of the project,
- it leaves a traceable record of assumptions.

---

## Maintenance Note

This document should evolve together with:

- the preprocessing pipeline,
- the target serialization design,
- the parser implementation,
- the modeling experiments,
- the evaluation modules.

Until then, it should remain explicit, aligned with current repository reality, and conservative about anything that is not yet implemented.
