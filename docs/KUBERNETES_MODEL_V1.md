# Kubernetes Model v1

This document defines the functional and research contract of the first modeling version built on top of the `Kubernetes v1` dataset.

It is intentionally explicit: the project is not only about building a working YAML generator, but also about studying the problem through an intermediate latent representation, explicit structural control, and reproducible evaluation.

Project-wide definitions for terms such as `block`, `level`, `primary_kind`,
`yaml_max_depth`, and `yaml_total_nodes` are maintained in
`docs/TERMINOLOGY.md`.

## 1. Position of this document in the project

This is the reference specification for the first end-to-end modeling pipeline built on top of:

- the processed dataset in `data/processed/kubernetes_v1/`
- the preprocessing policy documented in `docs/KUBERNETES_PREPROCESSING.md`
- the current base model stored in `model/qwen2.5-7b-instruct-4bit/`

This document does not fix exact training script names or class layouts. It fixes what the system is meant to do, what is central to the thesis, and what must be measured.

## 2. Project decisions already fixed

The following decisions are closed for the main line of the project:

- The effective case study of the repository is `Kubernetes`.
- The base dataset for the project is `Kubernetes v1`.
- The project uses a single main dataset version for the core experiments.
- Oversampling or dataset enlargement is not part of the v1 system contract.
- The experimental path is:
  1. baseline with the base model
  2. SFT with LoRA
  3. comparative branch with auxiliary structural signals
  4. DPO as the main post-SFT alignment step
  5. PPO only as an optional later extension
- The scientific core of the thesis is not only final YAML generation, but also the study of an explicit latent intermediate representation before block generation.
- The parser is not only a renderer: it is the module of structural control.
- The official output of inference is only the final YAML.

## 3. System objective

The system must transform a natural-language request into a Kubernetes manifest that is:

- syntactically valid YAML
- hierarchically coherent
- faithful to the prompt
- semantically plausible in the Kubernetes domain

The intended functional pipeline is:

`prompt_text -> latent intermediate representation -> blocks with level -> parser as structural control -> final YAML`

The project therefore studies two things at the same time:

- whether the system can generate valid structured outputs
- whether the intermediate latent representation is useful, analyzable, and predictive of structural quality

The project optimizes for structural correctness and prompt adequacy, not for surface-level text similarity alone.

## 4. Input contract

### Inference input

The user-facing input is a single natural-language request:

- `prompt_text`

Example:

```text
Create a cron job that runs every minute using nginx and forbids concurrent runs.
```

### Training input

During training, each example is defined by:

- `sample_id`
- `prompt_variant`
- `prompt_text`
- `target_yaml_normalized`
- `split`
- `validation_status`

The `prompt_text` comes from one of the two accepted prompt variants:

- `question.txt`
- `question_simplified.txt`

Both variants are valid training inputs and must remain in the same split.

## 5. Output contract

### Internal output of the system

The system should not be documented as directly learning raw YAML as its only target.

Its internal stages are:

1. a latent intermediate representation
2. a projection from that representation to structured blocks
3. parser-based structural control and final YAML reconstruction

### Structured block output

The explicit structured output immediately before the parser is a sequence of YAML lines, where each line contains:

- `line_index`
- `line_text`
- `level`

Optional fields are allowed if needed later:

- `document_index`
- `is_list_item`
- `parent_index`

### Official output of the system

The only official output exposed to the user in inference is:

- the final rendered YAML manifest

The latent and block-level representations are internal and are used for:

- training
- parser input
- structural debugging
- evaluation
- error analysis
- latent-space analysis

## 6. Latent intermediate representation

### Definition

The latent intermediate representation is an explicit internal representation between the prompt and the line-level block sequence.

It is not the final YAML and it is not identical to the blocks with level. It is the conceptual space where the system should organize the semantic content of the prompt before serializing it into a structured output.

### Function

Its function is to:

- condense the relevant semantic content of the prompt
- organize that content before final serialization
- expose an internal structure that can be analyzed mathematically
- provide a bridge between language understanding and structured generation

### Central hypothesis

The main hypothesis of this part of the thesis is:

- if the latent intermediate representation is well defined, generation quality improves
- if it is informative enough, it also becomes interpretable and analyzable

### Viability criterion

The latent representation is considered viable only if both conditions hold:

- it improves practical generation quality
- it supports meaningful structural or geometric analysis

This means that a latent space that is only interesting to visualize, but does not help generation, is insufficient. A latent space that improves generation but offers no analyzable structure also falls short of the intended thesis contribution.

### Role in the thesis

The latent intermediate representation is a central contribution of the project, not a future extension.

## 7. Structural representation after the latent space

### Chosen explicit representation

The explicit structural representation for v1, after the latent stage, is:

- one logical YAML line per prediction unit
- one hierarchical level per line

This means the system predicts a sequence like:

```text
line 1 -> "apiVersion: batch/v1", level 0
line 2 -> "kind: CronJob", level 0
line 3 -> "metadata:", level 0
line 4 -> "name: x-job", level 1
```

### Why this representation is used

This choice is deliberate because it:

- is easier to train than full tree prediction
- is easier to inspect than free-form generation
- separates semantic content from hierarchy
- fits baseline, SFT, DPO, and parser-based control under the same contract
- keeps the parser deterministic and auditable

### Important clarification

The blocks with level are not the latent space itself.

They are the explicit projection of the latent representation into a format that:

- can be supervised
- can be parsed deterministically
- can be evaluated structurally

## 8. Two logical heads after the latent stage

The v1 model must be documented as a system with two coordinated logical outputs after the latent representation is formed.

The conceptual order is:

1. prompt understanding and organization in latent space
2. projection to:
   - a semantic head
   - a structural head
3. parser-based structural control

### Semantic head

Purpose:

- predict the textual content of each YAML line

Target:

- `line_text`

What it is expected to learn:

- keys, values, resources, and configuration content
- domain-specific wording that should become Kubernetes fields
- prompt-to-content mapping

### Structural head

Purpose:

- predict the hierarchical level of each line

Target:

- `level`

What it is expected to learn:

- indentation depth
- parent-child layout
- tree shape consistency
- document organization

### Important clarification

"Two logical heads" is a behavioral contract, not yet a fixed implementation architecture.

This document does not require a specific engineering form such as:

- two literal decoder heads
- multitask losses in a single decoder
- a shared serialization with auxiliary supervision

What is fixed is that the model must be trained and evaluated as if content and structure were distinct signals projected from a prior internal representation.

## 9. Parser as structural control

### Role of the parser

The parser takes the predicted structured blocks and reconstructs final YAML from them.

Its responsibilities are:

- preserve line order
- apply indentation according to `level`
- reconstruct YAML formatting deterministically
- support multi-document YAML if present
- validate that the resulting text parses as YAML

### Why the parser is a control module

The parser should be documented as a structural control mechanism, not as a passive postprocessing step.

It acts as:

- a validator
- a reconstructor
- a control boundary between allowed and invalid structure

This means it operationalizes structural constraints at the final stage of the pipeline.

### Allowed behavior

The parser may perform minor presentation-level cleanup:

- normalize indentation
- normalize spacing
- normalize line formatting
- apply deterministic rendering conventions

### Forbidden behavior

The parser must not:

- invent keys
- invent values
- add missing resources
- silently fix semantic contradictions
- rescue a structurally or semantically wrong prediction with opaque repair logic

### Failure policy

If the prediction cannot be converted into a safe and coherent YAML output, the system must:

- mark the output as invalid
- record the failure
- avoid masking the failure with aggressive repair logic

## 10. Analytical and mathematical perspective

### Motivation

The thesis is not intended as a pure black-box benchmark.

It aims to study the generation problem through more explicit and analyzable internal structure. The goal is to understand not only whether the model works, but also what kind of internal organization makes structured generation possible or fragile.

### Object of analysis

The main object of analysis is the latent intermediate representation before block generation.

The project should study whether that representation:

- separates different structural families
- reflects hierarchical complexity
- distinguishes valid and invalid outputs
- anticipates structural errors before final rendering

### Type of questions the thesis should ask

The documentation should frame questions such as:

- Do similar Kubernetes structures cluster in latent space?
- Does latent geometry reflect structural depth or document complexity?
- Are invalid outputs associated with identifiable latent regions?
- Does the latent representation become more organized after SFT or DPO?

### Connection to performance

The analytical study is not decorative.

It must be connected to generation results. The thesis should treat interpretability and generation quality as related objectives, not independent tracks.

## 11. Dataset policy for modeling

### Base dataset

The official modeling dataset is:

- `Kubernetes v1`

This same dataset is used for:

- baseline
- SFT + LoRA
- comparative experiments with auxiliary signals
- DPO
- PPO if PPO is ever attempted

### Variant policy

Prompt variants are retained as valid inputs:

- `question`
- `question_simplified`

They are part of the same sample identity and must never be separated across splits.

### Oversampling and future enlargement

Oversampling is not part of the core v1 system.

If later experiments use:

- oversampling
- synthetic growth
- any augmentation strategy

they must be documented as additional experiments on top of the fixed v1 baseline, not as a redefinition of the system itself.

The accepted enrichment direction is documented in
`docs/MULTI_RESOURCE_STRATEGY_DECISION.md`. The enriched dataset version will be
`kubernetes_v2`, stored under `data/processed/kubernetes_v2/`, and will focus on
controlled multi-resource and multi-document compositions. `kubernetes_v2` is a
derived experimental branch; `kubernetes_v1` remains the base dataset for the
clean baseline and for comparison.

## 12. Auxiliary structural signals as a comparative branch

### Position in the project

Auxiliary structural signals are not part of the core method contract.

The main line of the project is:

- latent intermediate representation
- baseline / SFT / DPO
- parser-based structural control

Auxiliary signals are a comparative branch intended to test whether additional structural information helps beyond the main approach.

### What they are

At this stage they should be understood as structural signals derived from or used for control, not as a fixed final design.

Possible families include:

- node type
- structural transition type
- parent-child relation
- list markers
- document markers

Depth alone should not be treated as a strong enough formulation.

### How they should be used

Their comparative role is mainly as:

- features or signals useful for parser/control
- additional structured information to compare against the main system

They are not yet fixed as the main target of training.

## 13. Experimental phases

### Phase 1. Baseline

Objective:

- measure how far the base model can go under the latent-plus-structure contract before supervised adaptation

Default base model:

- `Qwen2.5-7B-Instruct-4bit`

Baseline rules:

- same input contract as the final system
- same latent-to-block-to-parser narrative
- same parser as structural control
- same evaluation metrics

What baseline is allowed to change:

- prompt template
- output format instruction
- deterministic parsing pipeline

What baseline does not include:

- supervised weight updates
- LoRA adaptation
- preference learning
- reinforcement learning

### Phase 2. SFT + LoRA

Objective:

- adapt the model to the task using supervised fine-tuning over the latent-aware structured target

Training target:

- line sequence with content and level

Expected gain over baseline:

- better prompt coverage
- fewer structural mistakes
- more parseable predictions
- more stable YAML generation

LoRA is chosen because:

- it is compatible with limited resources
- it allows focused adaptation
- it fits the scope of the thesis better than a full heavy fine-tune

### Phase 3. Comparative branch with auxiliary structural signals

Objective:

- test whether auxiliary structural signals improve the main system

This branch is comparative, not foundational. It exists to measure whether explicit extra structural cues help beyond the latent-plus-parser design.

### Phase 4. DPO

Objective:

- align the SFT model using automatically generated preferences

DPO is the main planned post-SFT method because:

- it is simpler than PPO
- it needs less infrastructure
- it is a realistic continuation of the project

Preference generation must rely on:

- multiple candidate outputs per prompt
- parser validation
- structural and semantic scoring

The chosen output should be the one that:

- parses correctly
- preserves better structure
- reflects the prompt better
- introduces fewer contradictions
- requires less parser cleanup

### Phase 5. PPO as optional extension

PPO is not part of the required main line of the project.

It is only justified if:

- the reward function becomes stable
- DPO is no longer enough
- the available compute makes PPO realistic

PPO should therefore be described as:

- optional
- later
- conditional on reward quality

## 14. Reward definition

If reward-based ranking or post-SFT alignment is used, the reward must include at least:

- YAML validity
- structural consistency
- fidelity to the prompt
- consistency of levels and line ordering
- absence of obvious contradictions
- penalty for outputs that cannot be parsed safely

Recommended priority order:

1. structural sequence parseable
2. final YAML valid
3. hierarchy coherent
4. prompt faithfully covered
5. semantic contradictions minimized

## 15. Evaluation contract

Each phase should report at least:

- percentage of structurally parseable predictions
- percentage of valid YAML outputs
- structural agreement with the reference
- prompt fidelity
- semantic inconsistency rate
- improvement over the previous phase

### Scenarios that must be evaluated

- simple prompts with one resource
- prompts with deeper nesting
- multi-document outputs
- ambiguous prompts
- incomplete prompts
- correct content with wrong level
- wrong content with correct level
- outputs that need only minor formatting cleanup
- outputs that should fail because of real structural inconsistency

### Additional analysis scenarios

The project should also analyze whether the latent space:

- separates structural families
- reflects complexity
- correlates with parser success or failure
- changes meaningfully across baseline, SFT, and DPO

## 16. Criteria for moving to the next phase

### Baseline can start only if

- the dataset is fixed
- YAML-to-line conversion exists
- the parser contract is defined
- the output format expected from the model is fixed
- the latent representation is at least conceptually specified

### SFT can start only if

- the baseline has been measured
- the target serialization is stable
- the parser is stable
- baseline metrics are recorded

### Comparative auxiliary-signal experiments can start only if

- the main SFT system is stable
- the added signals are clearly defined for the experiment
- the comparison preserves the same evaluation protocol

### DPO can start only if

- SFT clearly improves over baseline
- candidate generation exists
- preference construction is meaningful

### PPO can start only if

- reward quality is stable
- PPO is expected to add something beyond DPO
- compute constraints allow it

## 17. What is still intentionally open

This document leaves some implementation choices open on purpose:

- exact model class and training script layout
- exact parametrization of the latent space
- exact loss decomposition for the projected heads
- exact serialization syntax used to train the model
- exact reward formula weights
- exact experiment registry structure

These are implementation details that must respect this contract, not redefine it.

## 18. Implemented bridge to modeling

The first bridge from the processed dataset to modeling is now implemented:

`target_yaml_normalized -> latent intermediate representation -> sequence of lines with level -> training serialization`

The implemented concrete layer covers:

- how lines are extracted from normalized YAML
- how `level` is encoded
- how the model sees both latent and block-level targets
- how parser success and structural fidelity are measured independently and jointly

The current implementation fixes the block-level target and parser boundary:

- `scripts/build_kubernetes_structural_targets.py` derives line-and-level targets.
- `src/llm_structured_semantic_generation/structure.py` reconstructs YAML deterministically.
- `scripts/build_kubernetes_sft_dataset.py` creates the first SFT-ready serialization.
- `scripts/run_kubernetes_baseline.py` defines the zero-shot baseline execution path.

The exact parametrization and supervision of the latent intermediate representation remains open. The implemented block representation must therefore be treated as the explicit projection after the latent stage, not as the latent space itself.

## 19. Next implementation step

The next implementation step is to run and record the baseline:

1. Generate structural targets and confirm `structural_targets_report.json` has `ready_for_baseline: true`.
2. Run the dry-run baseline configuration check.
3. Install optional LLM dependencies if needed.
4. Run the baseline on validation and test.
5. Review `metrics.json` and error examples before starting LoRA/SFT.
