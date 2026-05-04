# Kubernetes SFT Strategy v1

This document defines a focused SFT plan for the current repository state.

The plan is driven by the central narrative of the project: LLMs generate flat
token sequences, while Kubernetes YAML requires hierarchical structure. The SFT
stage tests whether the missing hierarchy variable should be learned as part of
the serialized output surface or predicted explicitly as `level`.

It is intentionally narrower than a generic fine-tuning guide. It is based on:

- the current repository contract in `docs/KUBERNETES_MODEL_V1.md`
- the current structural target contract in `docs/STRUCTURAL_TARGETS_V1.md`
- the current processed dataset under `data/processed/kubernetes_v1/`
- the completed baseline run `compact-test70-320-vtfix`
- the local bibliography under `bib/`

The goal is to answer one concrete question:

> Does an explicit hierarchical-level head improve structured Kubernetes generation
> beyond a control SFT model that learns `level` only as serialized text?

This is the supervised version of the broader thesis comparison:

- `serialized_sft` represents the surface-structured approach, where hierarchy
  is emitted as text and later constrained by the parser;
- `two_head_sft` represents the explicit-hierarchy approach, where generation
  order supplies line position and a structural head predicts each line's
  hierarchical `level`.

The first supervised stage should therefore compare two SFT variants:

- `serialized_sft`: control model; the causal LM emits `line_text` and `level`
  inside `blocks_tsv_v1`
- `two_head_sft`: main thesis model; the causal LM predicts line content and a
  physical structural head predicts `level`

## 1. Why SFT is the right next step

The repository already shows three important facts:

1. The baseline is not random.
   - It often predicts the right coarse resource type and some important fields.
   - It is therefore not a case where the model completely fails to map prompt to domain.

2. The main failure is structural realization.
   - The recorded baseline reaches high structured-surface survival compared to final YAML validity.
   - In the recomputed report for `compact-test70-320-vtfix`, `structured_output_parse_success_rate = 0.8857` but `yaml_parse_success_rate = 0.4677`.
   - This strongly suggests that the model often knows roughly what should be produced, but does not serialize it consistently enough.

3. The repository already has a clean supervised target for both variants.
   - `data/processed/kubernetes_v1/sft/` exists.
   - The current target format is `blocks_tsv_v1`.
   - The same rows can train the serialized control model directly.
   - The same rows can also be parsed into line-content labels and level labels
     for the two-head model.
   - The parser and block validation layer already exist, so both variants can
     be evaluated under the same reconstruction contract.

This is exactly the situation where SFT is justified:

- imitation learning is cheap compared to RLHF
- the target representation is explicit and auditable
- the current failures are largely local, structural, and format-sensitive

This is also consistent with the local bibliography:

- `bib/sft.pdf` argues that SFT remains the standard post-training stage because it efficiently transfers expert demonstrations even if its generalization is imperfect
- `bib/dpo_vs_ppo.pdf` explicitly presents SFT as the normal first phase before RLHF-style alignment
- `bib/orpo.pdf` also emphasizes that successful preference alignment still depends on a strong SFT stage

## 2. Bibliography-grounded interpretation

The local bibliography supports the current project direction in a fairly coherent way.

### 2.1 SFT as the necessary first supervised stage

Relevant references:

- `bib/sft.pdf`
- `bib/dpo_vs_ppo.pdf`
- `bib/orpo.pdf`

What they contribute here:

- SFT is the simplest and most stable way to transfer a task-specific input-output mapping.
- Preference optimization methods are easier to justify after a competent supervised policy already exists.
- SFT is especially appropriate when we already have positive demonstrations but no human preference dataset.

What this means for the repo:

- the first serious training stage should compare the two pure SFT + LoRA branches
- DPO should only come after we have a structurally competent supervised checkpoint
- DPO is treated here as practical automatic-preference optimization, not as
  proof that the repository contains a complete human-feedback RLHF pipeline

### 2.2 Structured prediction should not be treated as plain text imitation

Relevant references:

- `bib/structured_pred.pdf`
- `bib/Structured_Prediction_Energy_Networks.pdf`

What they contribute here:

- structured prediction should be evaluated as prediction of interdependent outputs, not just token overlap
- there is value in choosing an explicit output decomposition that makes constraints and dependencies measurable

What this means for the repo:

- training directly on the line-and-level representation is not just an engineering shortcut
- it is aligned with the thesis claim that structure should be modeled explicitly
- `serialized_sft` remains necessary because it tests how far the flat
  generation surface can go when hierarchy is encoded as ordinary output tokens
- `two_head_sft` tests whether making `level` a separate supervised variable
  improves hierarchy-sensitive metrics

### 2.3 Grammar and constraint mechanisms improve valid generation, but do not replace learning

Relevant references:

- `bib/Constraint_LLMS.pdf`
- `bib/Leveraging_Grammar.pdf`
- `bib/Lexically_Constrained_Decoding.pdf`

What they contribute here:

- constrained or grammar-aware generation can sharply reduce invalid outputs
- explicit structural control is especially useful when the target space is narrow and formal
- however, constraints do not magically supply missing semantic content

What this means for the repo:

- the parser is a strong design choice and should stay central
- but parser-side control alone is not enough; the model must learn better structural serialization
- SFT is still needed because the parser cannot legally invent missing content

### 2.4 Alignment should be delayed until the supervised policy is structurally competent

Relevant references:

- `bib/dpo.pdf`
- `bib/orpo.pdf`
- `bib/aligment_tax.pdf`

What they contribute here:

- DPO is attractive because it is simpler than PPO
- ORPO suggests monolithic preference optimization can be efficient
- alignment can introduce an "alignment tax" and degrade other useful abilities if applied too early or too aggressively

What this means for the repo:

- the repository is right to postpone preference optimization
- if SFT is weak, DPO will optimize on top of a structurally fragile base policy
- the first requirement is therefore to improve structural competence and prompt adequacy before preference alignment

## 3. Concrete hypotheses

These hypotheses are intentionally operational and tied to current metrics.

## 3.1 Primary architectural hypothesis

### H1. The explicit `level` head will outperform serialized level prediction on hierarchy-sensitive metrics.

Expected direction:

- `average_level_exact_match_rate` higher for `two_head_sft`
- `yaml_parse_success_rate` higher for `two_head_sft` when content is mostly correct
- fewer cases where line content is plausible but indentation hierarchy breaks reconstruction

Interpretation:

- this is the central thesis hypothesis
- the serialized model is a necessary control, not a competing thesis direction
- if `two_head_sft` does not beat `serialized_sft`, the claimed value of the
  physical hierarchy head must be weakened or reformulated

### Support criterion for H1

Treat H1 as supported if, on the full validation split:

- `two_head_sft` improves `average_level_exact_match_rate` over
  `serialized_sft` by at least `+0.10` absolute
- `two_head_sft` does not reduce `average_prompt_requirement_f1`
- `two_head_sft` improves or matches `yaml_parse_success_rate`

## 3.2 General SFT hypothesis

### H2. Both SFT variants will improve structural validity substantially relative to the current baseline.

Expected direction:

- `yaml_parse_success_rate` up
- `line_count_match_rate` up
- `average_level_exact_match_rate` up
- `average_line_text_f1` up

Interpretation:

- this is the main hypothesis
- if it fails, the current representation or dataset may not be sufficient for the supervised stage

### Support criterion for H2

Treat H2 as supported if, on the full validation split:

- `yaml_parse_success_rate` improves by at least `+0.20` absolute
- `average_line_text_f1` improves by at least `+0.10` absolute
- `structured_output_parse_success_rate` does not deteriorate

These thresholds are deliberately modest but meaningful.

## 3.3 Prompt adequacy hypothesis

### H3. SFT will improve requirement coverage from the prompt more strongly than exact-match fidelity.

Expected direction:

- `average_prompt_requirement_recall` up clearly
- `average_prompt_requirement_f1` up
- `prompt_requirement_exact_match_rate` up, but probably less dramatically

Interpretation:

- supervised learning should help the model cover more requested fields and resource choices
- but exact set equality remains hard because generation may still include extra or misplaced content

### Support criterion for H3

Treat H3 as supported if, on validation:

- `average_prompt_requirement_recall` improves by at least `+0.15` absolute
- `average_prompt_requirement_f1` improves by at least `+0.10` absolute

## 3.4 Minimal domain validity hypothesis

### H4. SFT will reduce missing required fields for the frequent workload kinds.

Expected direction:

- `average_required_field_complete_resource_rate` up
- `required_field_complete_sample_rate` up
- fewer missing groups for `DaemonSet`, `Deployment`, and `Pod`

Interpretation:

- this is important because the recomputed baseline report shows frequent missing required groups in these kinds

### Support criterion for H4

Treat H4 as supported if, on validation:

- `average_required_field_complete_resource_rate` improves by at least `+0.15` absolute
- the count of incomplete `DaemonSet` and `Deployment` predictions is clearly reduced in error analysis

## 3.5 Limited-scope hypothesis

### H5. SFT alone will not solve full semantic consistency across all Kubernetes relations.

Expected outcome:

- improvements in syntax and minimal structure
- incomplete gains in deeper multi-resource or cross-field coherence

Interpretation:

- this is not a pessimistic hypothesis
- it is a realism check

If H5 is observed, that would still be fully compatible with the thesis:

- SFT stabilizes the structural backbone
- parser control enforces legality
- later DPO or auxiliary signals target residual semantic issues

## 4. What should be frozen before SFT

Before the first real SFT run, freeze the following:

- dataset version: `kubernetes_v1`
- processed structural dataset: `data/processed/kubernetes_v1/dataset_structural_targets.jsonl`
- SFT export: `data/processed/kubernetes_v1/sft/`
- base model: `model/qwen2.5-7b-instruct-4bit/`

Also freeze the comparison policy:

- validation is for model selection
- test is touched once per stable candidate, not repeatedly during iteration

Important note:

- the repository currently has a completed full `test` baseline run
- it does not yet document a full frozen validation baseline run of the same quality

So, before comparing SFT properly, the first operational step should be:

- run and freeze a full validation baseline with the same evaluation stack

## 5. How SFT should be executed step by step

This section is intentionally procedural and aligned with current repo reality.

## Step 1. Freeze the supervised target surface

Use the existing SFT export:

- `data/processed/kubernetes_v1/sft/train.jsonl`
- `data/processed/kubernetes_v1/sft/validation.jsonl`
- `data/processed/kubernetes_v1/sft/test.jsonl`

Current status:

- `ready_for_sft = true`
- serialization = `blocks_tsv_v1`

This should remain the shared supervised source for both SFT variants.

For `serialized_sft`, `blocks_tsv_v1` is the direct generation target.

For `two_head_sft`, `blocks_tsv_v1` is parsed into:

- line-content supervision for the causal LM head
- per-line `level` labels for the structural head

Do not silently switch to:

- raw YAML
- compact TSV
- prompt requirement atoms

unless it is explicitly documented as a separate comparative branch.

## Step 2. Match training and first evaluation surfaces

This is a critical repository-specific point.

Current mismatch:

- baseline inference default = `blocks_tsv_compact_v1`
- SFT training target = `blocks_tsv_v1`

For the **first SFT comparison**, evaluation should reconstruct the same
parser-facing block sequence for both models:

- `serialized_sft` emits `blocks_tsv_v1` directly
- `two_head_sft` emits line content and predicts `level`, then combines them
  into the same block contract before parser reconstruction

Reason:

- it removes one confound
- it lets us test whether the physical level head improves the same downstream
  parser-controlled representation
- it keeps the evaluation stack identical across both branches

Only after a clean result should we test whether either checkpoint can also run
well under:

- `blocks_tsv_compact_v1`

as an efficiency-oriented inference variant or deployment surface.

## Step 3. Implement resumable SFT training scripts

The repo does not yet contain the actual SFT trainer, so it must be implemented
conservatively. The minimum defensible implementation can be either:

- one trainer with a `--model-variant serialized_sft|two_head_sft` argument
- two scripts that share the same loading, checkpointing, and evaluation helpers

The first option is preferable because it reduces protocol drift between the two
models.

Minimum contract:

- `--train-file`
- `--validation-file`
- `--base-model-path`
- `--output-dir`
- `--run-id`
- `--batch-size`

Minimum persisted artifacts:

- `config.json`
- `state.json`
- training checkpoints
- validation predictions or validation summaries per checkpoint
- `metrics.json` only after successful completion

Each variant must persist its own run directory and checkpoints. The comparison
is only valid if both variants follow the same resumable policy already used in
baseline inference.

## Step 4. Use LoRA, not full fine-tuning

For the first pass, use:

- LoRA or QLoRA-style adaptation
- the current quantized local Qwen checkpoint

Rationale:

- dataset is small
- hardware is constrained
- the objective is controlled adaptation, not maximal capacity search

The first SFT comparison should therefore be conservative:

- low-rank adapters
- fixed base model
- minimal optimizer complexity

### First LoRA placement policy

The current local model is `Qwen2ForCausalLM` with 28 transformer layers. The
checkpoint exposes the standard Qwen2 projection names:

- attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

For the first SFT comparison, the recommended LoRA target modules are:

```text
q_proj,k_proj,v_proj,o_proj
```

applied across all 28 layers.

Rationale:

- attention adapters are the lowest-risk place to adapt prompt-to-output routing
- the dataset is small enough that adapting the MLPs from the first run may
  overfit structural templates too aggressively
- all-layer attention LoRA still gives the model capacity to adjust both prompt
  conditioning and output serialization without changing the base model

Recommended initial hyperparameter range:

- `r = 8` or `r = 16`
- `lora_alpha = 16` or `32`
- `lora_dropout = 0.05`
- `bias = none`
- `task_type = CAUSAL_LM`

The structural `level` head is not a LoRA target. It is a small newly initialized
trainable module on top of the shared decoder hidden states. LoRA adapts the
backbone; the level head learns the explicit hierarchy prediction.

The second comparison, only if the first run underfits, should expand LoRA to:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

This should be documented as a separate capacity experiment, not as the default
SFT setting.

## Step 5. Train the paired supervised models

The first supervised block should train both SFT variants before moving to DPO,
PPO, or auxiliary structural signals.

### Serialized control model

The control model is:

- shared Qwen2 decoder backbone
- same LoRA target modules as the main model
- standard causal LM head
- direct `blocks_tsv_v1` text generation target

Its purpose is to answer the control question:

```text
How far can the model get if hierarchy is learned only as serialized text?
```

This branch intentionally keeps the hierarchy on the same flat autoregressive
surface as the YAML line content. The parser can validate and reconstruct what
the model emits, but the model itself has no separate supervised mechanism for
the height of each YAML entry.

### Structural level head decision

The main SFT model should introduce an explicit auxiliary classification
head for `level`.

This is the central modeling choice of the project, not a later add-on. The
model should be trained as:

- shared Qwen2 decoder backbone
- LoRA adapters on the selected transformer projections
- causal LM head for `line_text`
- structural classification head for `level`

The supervised loss should combine:

- content loss: token-level causal language-modeling loss over the line-content
  serialization
- level loss: classification loss over the target `level` for each YAML line

The first two-head implementation should keep the loss simple:

- `loss = lm_loss + lambda_level * level_loss`
- start with `lambda_level = 1.0`
- report both losses separately in training logs

The existing `blocks_tsv_v1` serialization remains useful as the data source for
line text and level labels. It should be parsed into two supervised targets
rather than used as the only output channel of the main SFT model.

This branch uses the generation sequence for line order and the structural head
for hierarchy. That separation is the point of the experiment: content and
height are related, but they are not the same prediction.

The exact line-token alignment policy must be documented in the trainer. A
conservative first implementation is:

- predict one `level` per generated YAML line
- attach the level prediction to the hidden state corresponding to the beginning
  or end marker of each line
- ignore non-line control tokens in the level loss

The important point is that `level` is not merely emitted as text in the main
experiment; it is learned through a distinct supervised structural head.

Do not add, in the paired SFT comparison:

- prompt-requirement auxiliary loss
- required-field auxiliary classifier
- reward shaping
- preference loss

Reason:

- the first comparison must answer whether explicit level supervision improves
  structured generation under the current parser-controlled pipeline

If the paired SFT comparison works, only then is it worth testing preference
optimization or auxiliary structural signals.

## Step 6. Evaluate every checkpoint with the current structural stack

For each validation checkpoint of both `serialized_sft` and `two_head_sft`,
compute at least:

- `structured_output_parse_success_rate`
- `yaml_parse_success_rate`
- `parsed_equal_rate`
- `average_line_text_f1`
- `average_level_exact_match_rate`
- `average_prompt_requirement_precision`
- `average_prompt_requirement_recall`
- `average_prompt_requirement_f1`
- `average_required_field_complete_resource_rate`
- `required_field_complete_sample_rate`

The current baseline evaluator is already strong enough to support the final
parser-facing metrics. The two-head trainer must additionally log:

- `lm_loss`
- `level_loss`
- `lambda_level`
- line-alignment policy used for level supervision

## Step 7. Select checkpoints by structural quality, not by LM loss alone

This is essential.

The best checkpoint for each branch should not be chosen only by:

- training loss
- validation cross-entropy

It should be chosen primarily by a structural score.

Recommended primary selection order:

1. `yaml_parse_success_rate`
2. `average_prompt_requirement_f1`
3. `average_required_field_complete_resource_rate`
4. `average_line_text_f1`

Reason:

- raw token imitation can improve while structural validity stagnates
- the thesis objective is structured generation, not language modeling loss minimization

## Step 8. Freeze one best validation checkpoint and evaluate on test once

After selecting the best checkpoint on validation:

- run one test evaluation
- compute the same metrics as in validation
- produce a compact report similar to the recomputed baseline report

This creates the first clean comparison table:

```text
baseline vs serialized_sft vs two_head_sft
```

## 6. Real risks

These are not generic risks. They are the risks that genuinely matter for the current repo.

## Risk 1. Train/inference surface mismatch

Current issue:

- the SFT control branch uses `blocks_tsv_v1` directly
- the two-head branch uses the same rows but decodes `line_text` and `level`
  through separate outputs
- baseline inference defaults to `blocks_tsv_compact_v1`

Why this matters:

- improvements may be hidden or distorted if the checkpoint is evaluated on a different output surface from the one it learned

Mitigation:

- both SFT variants must be converted to the same parser-facing block contract
  before evaluation
- compact inference should be a second experiment, not the first

## Risk 2. The model learns serialization patterns but not deeper prompt grounding

Why this is real:

- the dataset is small
- exact structural targets are repetitive
- the model may overfit common block templates

Symptoms:

- strong `yaml_parse_success_rate`
- moderate or weak `average_prompt_requirement_recall`
- weak generalization on rarer kinds

Mitigation:

- track prompt-requirement metrics from the start
- inspect category-level recall, not only parse success

## Risk 3. Small dataset size limits generalization

Current scale:

- train = `426` prompt rows
- validation = `70`
- test = `70`

Why this matters:

- there is a real chance of narrow memorization of frequent structural families

Mitigation:

- use LoRA, not high-capacity full tuning
- keep checkpoint selection conservative
- report kind-specific failures explicitly

## Risk 4. Long-tail kind coverage may dominate failure interpretation

Why this is real:

- the dataset is not balanced across all Kubernetes kinds
- some important kinds are relatively rare

Mitigation:

- always report both global metrics and frequent-kind error slices
- interpret improvements on common kinds separately from long-tail behavior

## Risk 5. Parser success may hide semantic weakness

Why this matters:

- a YAML file can parse and still be semantically weak or incomplete

Mitigation:

- never rely on `yaml_parse_success_rate` alone
- pair it with:
  - prompt requirement metrics
  - required field completeness
  - targeted error analysis

## Risk 6. The level head requires a careful line-alignment policy

Why this is real:

- the backbone operates at token level, but `level` is defined per YAML line
- a poor alignment choice can make the level loss noisy
- generated line boundaries may not perfectly match target line boundaries

Mitigation:

- define and document exactly which hidden state supervises each line level
- mask all non-line positions from the level loss
- report `average_level_exact_match_rate` separately from line-text metrics
- inspect cases with correct content but wrong `level`
- keep the first alignment policy simple enough to reproduce

## Risk 7. LoRA placement may underfit or overfit the structural task

Why this is real:

- attention-only LoRA may be too weak if the model must learn a new output
  serialization strongly
- adding MLP adapters from the first run may memorize common templates on a small
  dataset

Mitigation:

- start with attention-only LoRA as the controlled baseline
- add MLP LoRA only as a second capacity experiment
- compare both with the same validation baseline and target serialization

## Risk 8. Premature alignment can damage the supervised policy

Why this is real:

- the local bibliography on alignment tax warns that aggressive post-SFT alignment may degrade useful behavior

Mitigation:

- do not jump to DPO before the supervised checkpoint is structurally strong
- measure structural retention before any preference optimization

## 7. What a good first SFT comparison would look like

A good first SFT comparison is **not**:

- perfect YAML equality
- near-complete semantic correctness
- full multi-resource robustness

A good first SFT comparison is:

- much higher YAML parse success
- clearly better prompt requirement recall
- clearly fewer missing required groups in common workload kinds
- visibly lower structural fragmentation in error analysis
- a clear answer to whether `two_head_sft` improves hierarchy prediction over
  `serialized_sft`

In practical terms, the first SFT comparison should be considered successful if
it produces:

- at least one clearly better structural policy than the current zero-shot
  baseline
- a measured `serialized_sft` control
- a measured `two_head_sft` main model
- a stable checkpoint that can serve as the base policy for later DPO

## 8. Recommended experiment order

The first sequence should be:

1. freeze a full validation baseline run
2. implement the shared resumable SFT + LoRA training infrastructure
3. train `serialized_sft` as the control branch
4. implement and train the two-head Qwen2 wrapper with an explicit `level` head
5. evaluate both branches with the current parser/evaluation stack
6. compare `baseline`, `serialized_sft`, and `two_head_sft` on validation
7. select best checkpoints within each branch
8. evaluate the frozen comparison on test once
9. only then consider:
   - compact inference adaptation
   - additional auxiliary structural supervision
   - prompt-requirement auxiliary losses
   - DPO

This order minimizes confounds and keeps the thesis narrative clean.

## 9. Main recommendation

The repository should treat the first SFT stage as a **controlled architectural
comparison for structural stabilization**, not as a final alignment stage.

That means:

- the main target is not fluent YAML text
- the main target is a stronger policy for:
  - content selection
  - line ordering
  - hierarchical levels
  - minimal kind validity

If the SFT comparison achieves that, then the project is in a strong position for
the next phase:

- comparative auxiliary structural signals
- DPO on top of the structurally strongest supervised branch

At the current repository state, this is the most justified, lowest-risk, and most thesis-consistent next step.
