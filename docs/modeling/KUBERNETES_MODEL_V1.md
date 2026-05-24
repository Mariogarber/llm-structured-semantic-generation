# Kubernetes Model v1

This document defines the functional and research contract of the first modeling version built on top of the `Kubernetes v1` dataset.

It is intentionally explicit: the project is not only about building a working YAML generator, but also about studying why flat autoregressive token generation is fragile when the desired output is a hierarchical technical artifact.

Kubernetes manifests are serialized as YAML text, but their correctness depends
on a tree-like structure of nested resources, fields, lists, and relations. The
line order of the output can be supplied by autoregressive generation, but the
height of each entry in the YAML hierarchy is not directly available from that
order. In this project, that missing structural variable is represented by
`level`.

The model contract therefore compares two formulations:

- a surface-structured formulation, where hierarchy is learned as serialized
  text and controlled through parser reconstruction and later
  automatic-preference optimization;
- an explicit-hierarchy formulation, where `level` is predicted through a
  dedicated structural head and then combined with generated line content before
  parser reconstruction.

Project-wide definitions for terms such as `block`, `level`, `primary_kind`,
`yaml_max_depth`, and `yaml_total_nodes` are maintained in
`docs/reference/TERMINOLOGY.md`.

## 1. Position of this document in the project

This is the reference specification for the first end-to-end modeling pipeline built on top of:

- the processed dataset in `data/processed/kubernetes_v1/`
- the preprocessing policy documented in `docs/data/KUBERNETES_PREPROCESSING.md`
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
  2. SFT control model with serialized `level`
  3. SFT main model with an explicit hierarchical-level head
  4. post-SFT alignment experiments, preferably DPO first
  5. comparative branch with additional auxiliary structural signals
  6. PPO only as an optional later extension
- The scientific core of the thesis is not only final YAML generation, but also the study of an explicit latent intermediate representation before block generation.
- The project explicitly compares hierarchy as serialized output text against
  hierarchy as a supervised structural variable.
- The position of a YAML line is supplied by generation order; its hierarchical
  height is represented by `level`.
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

The main comparison is not between YAML as raw text and YAML as a finished tree
object. It is between two ways of making a flat token generator respect a
hierarchical target: serialize the hierarchy into the output surface, or predict
the hierarchy explicitly through a structural head.

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

`line_index` is the explicit version of the left-to-right order already implied
by generation. `level` is the hierarchy height that the project treats as the
central structural prediction target.

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

Conceptually, this representation turns a YAML tree into two coordinated
sequences:

- the ordered content sequence, represented by generated `line_text`
- the hierarchy sequence, represented by supervised `level`

This is the minimal representation needed for the thesis question. The model
does not have to generate a full tree data structure, but it also does not get
to hide hierarchy entirely inside raw text.

### Important clarification

The blocks with level are not the latent space itself.

They are the explicit projection of the latent representation into a format that:

- can be supervised
- can be parsed deterministically
- can be evaluated structurally

## 8. Two model heads after the latent stage

The v1 model must be documented as a system with two coordinated outputs after
the latent representation is formed.

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

The two-head formulation is part of the central modeling hypothesis of the
project.

The main SFT model should therefore include:

- the standard causal language-modeling head for the serialized line content
- an explicit structural head that predicts `level` for each generated YAML line

The serialized `blocks_tsv_v1` target remains useful for dataset preparation,
parser input, baseline comparison, and ablation experiments. In the
surface-structured branch, it lets the language model learn hierarchy as text.
It should not be treated as a replacement for the main hierarchical-level head.

The exact implementation details remain open, but they must preserve this
research claim:

- semantic content and hierarchical structure are distinct predictions projected
  from a shared latent representation
- the structural head is trained with an explicit level supervision signal
- evaluation must report content quality and level quality separately

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
- `serialized_sft`
- `two_head_sft`
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
`docs/decisions/MULTI_RESOURCE_STRATEGY_DECISION.md`. The enriched dataset version will be
`kubernetes_v2`, stored under `data/processed/kubernetes_v2/`, and will focus on
controlled multi-resource and multi-document compositions. `kubernetes_v2` is a
derived experimental branch; `kubernetes_v1` remains the base dataset for the
clean baseline and for comparison.

## 12. Main SFT Comparison

The central supervised comparison in v1 is between two models trained from the
same base checkpoint and dataset:

- `serialized_sft`: the control model, where `level` is generated as part of
  the textual block serialization. This is the surface-structured branch: the
  hierarchy is present, but only as tokens that the causal LM must emit.
- `two_head_sft`: the main model, where `line_text` is predicted through the
  language-modeling head and `level` is predicted through an explicit structural
  head. This is the explicit-hierarchy branch: the hierarchy is supervised as a
  separate structural variable.

This comparison is the cleanest way to test the main thesis hypothesis:

```text
Does explicit hierarchical supervision improve structured YAML generation beyond
learning the hierarchy as serialized text?
```

Both models must use the same:

- `Kubernetes v1` splits
- base model
- LoRA policy, unless the experiment explicitly studies adapter capacity
- parser
- evaluation stack

The preferred first comparison is:

```text
serialized_sft vs two_head_sft
```

Post-SFT alignment can then be applied symmetrically if scope allows:

```text
serialized_sft -> serialized_sft_dpo
two_head_sft -> two_head_sft_dpo
```

The minimal defensible path is to train and evaluate both SFT variants first.
DPO or PPO should not be used to replace this architectural comparison.
Preference optimization belongs after the supervised comparison because it tests
whether validity-oriented feedback can further improve a model that already has
a measurable structural policy. It must not be described as evidence that a full
RLHF pipeline exists in the repository.

## 13. Auxiliary structural signals as a comparative branch

### Position in the project

Auxiliary structural signals are not part of the first two-model SFT comparison.

The main line of the project is:

- latent intermediate representation
- baseline
- SFT control model with serialized `level`
- SFT main model with an explicit `level` head
- parser-based structural control
- optional post-SFT preference optimization

Auxiliary signals are a later comparative branch intended to test whether
additional structural information helps beyond the explicit level-head design.

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

## 14. Experimental phases

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

### Phase 2a. SFT + LoRA Control Model

Objective:

- train a controlled SFT model where both content and `level` are emitted through
  the language-modeling surface

Training target:

- serialized `blocks_tsv_v1`
- `level` represented as text in the block tuple

Purpose:

- provide the ablation needed to measure whether a physical structural head adds
  value
- keep a simple SFT reference close to the current baseline parser contract

### Phase 2b. SFT + LoRA Main Two-Head Model

Objective:

- adapt the model to the task using supervised fine-tuning with an explicit
  split between content prediction and hierarchy prediction

Training target:

- line content through the causal LM objective
- hierarchical `level` through an explicit structural head

Expected gain over baseline:

- better prompt coverage
- fewer structural mistakes
- more parseable predictions
- more stable YAML generation

Expected gain over `serialized_sft`:

- cleaner separation between content mistakes and hierarchy mistakes
- higher `average_level_exact_match_rate`
- better parser reconstruction when line content is mostly correct
- more interpretable structural failure analysis

LoRA is chosen because:

- it is compatible with limited resources
- it allows focused adaptation
- it fits the scope of the thesis better than a full heavy fine-tune

### Phase 3. Post-SFT Alignment

Objective:

- align one or both SFT models using automatically generated preferences

DPO is the preferred first post-SFT method because:

- it is simpler than PPO
- it needs less infrastructure
- it is a realistic continuation of the project

If scope allows, alignment should be applied symmetrically:

```text
serialized_sft -> serialized_sft_dpo
two_head_sft -> two_head_sft_dpo
```

If scope does not allow this, the priority should be:

- first establish `serialized_sft` vs `two_head_sft`
- then apply DPO to the stronger or thesis-central model

Current methodological decision as of 2026-05-24:

- the first DPO branch is `serialized_sft -> serialized_sft_dpo`;
- `serialized_sft` is chosen because it is the strongest complete
  parser-facing generator currently documented;
- the current two-head family remains useful for analysis, but is not the first
  DPO base because its validation behavior is dominated by parseability and
  interface instability;
- the detailed DPO contract is maintained in
  `docs/modeling/DPO_AUTOMATIC_PREFERENCE_V1.md`.

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

### Phase 4. Comparative branch with auxiliary structural signals

Objective:

- test whether additional structural signals improve the main two-head system

This branch is comparative, not foundational. It exists to measure whether
extra structural cues add value beyond the explicit level-head architecture.

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

## 15. Reward definition

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

## 16. Evaluation contract

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

## 17. Criteria for moving to the next phase

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

### Post-SFT alignment can start only if

- `serialized_sft` has been trained and evaluated as the control branch
- `two_head_sft` has been trained and evaluated as the main branch
- the comparison has been reviewed on validation
- the chosen branch for DPO is explicitly justified

The current branch justification is recorded in
`docs/decisions/DPO_POST_SFT_ALIGNMENT_DECISION.md`.

### Comparative auxiliary-signal experiments can start only if

- the main two-head SFT system is stable
- the added signals are clearly defined for the experiment
- the comparison preserves the same evaluation protocol

### DPO can start only if

- at least one SFT branch clearly improves over baseline
- candidate generation exists
- preference construction is meaningful

### PPO can start only if

- reward quality is stable
- PPO is expected to add something beyond DPO
- compute constraints allow it

## 18. What is still intentionally open

This document leaves some implementation choices open on purpose:

- exact model class and training script layout
- exact parametrization of the latent space
- exact loss decomposition for the projected heads
- exact serialization syntax used to train the model
- exact reward formula weights
- exact experiment registry structure

These are implementation details that must respect this contract, not redefine it.

## 19. Implemented bridge to modeling

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
- the recommended baseline output surface is now `blocks_tsv_compact_v1`, which
  keeps the same parser-facing block contract while removing model-predicted
  `line_index` from the inference surface to reduce output-token pressure.
- `blocks_tsv_v1` remains the shared supervised source for:
  - the `serialized_sft` control branch
  - the `two_head_sft` main branch, after extracting line-content labels and
    per-line `level` labels.

The exact parametrization and supervision of the latent intermediate representation remains open. The implemented block representation must therefore be treated as the explicit projection after the latent stage, not as the latent space itself.

## 20. Incremental Execution Policy

All LLM-facing experiment code in this repository must now be incremental and
resumable by default.

This is a project constraint, not an optional implementation detail. The main
motivation is that baseline, validation, SFT, DPO, and future training runs may
need to execute on slow or interruptible hardware without losing accumulated
progress.

The minimum contract for any new LLM script is:

- support `--output-dir`, `--run-id`, and `--batch-size`;
- persist `config.json` and a live `state.json`;
- write partial append-only artifacts batch by batch;
- treat persisted partial artifacts as the source of truth during resume;
- write final aggregate reports such as `metrics.json` only after successful
  completion.

For future training code, this policy also implies resumable checkpoints for the
model, optimizer, and scheduler state. These checkpoints are required when
training is implemented; they must not be deferred to a later cleanup step.

## 21. Next implementation step

The next implementation step is to run and record the baseline:

1. Generate structural targets and confirm `structural_targets_report.json` has `ready_for_baseline: true`.
2. Run the dry-run baseline configuration check.
3. Install optional LLM dependencies if needed.
4. Run the baseline on validation and test with stable `run_id` values so interrupted executions can resume.
5. Review `metrics.json`, `state.json`, and error examples before starting LoRA/SFT.
