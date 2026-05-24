# Project Status And Anteproyecto Note

This note records the current state of the repository and the wording that
should guide the academic anteproyecto. It is intentionally conservative: it
separates implemented work from planned experiments and avoids presenting future
stages as completed results.

## Current Technical Status

The repository is no longer only a generic structured-YAML generation prototype.
Its current effective case study is Kubernetes manifest generation from natural
language instructions.

The current implemented basis is:

- dataset version: `kubernetes_v1`;
- processed data location: `data/processed/kubernetes_v1/`;
- processed samples: `283`;
- prompt-variant rows: `566`;
- split policy: train/validation/test at leakage-group level;
- structural target representation:
  `document_index`, `line_index`, `level`, `line_text`;
- SFT serialization: `blocks_tsv_v1`;
- baseline inference serialization: `blocks_tsv_compact_v1`;
- current local base model:
  `model/qwen2.5-7b-instruct-4bit/`.

The preprocessing, structural target generation, SFT export, baseline runner,
evaluation helpers, latent-vector collection, prompt-requirement extraction,
SFT trainer for the serialized control branch, offline SFT metric
recomputation, latent `level` probing, and resumable execution utilities exist
in the repository.

The local Python environment currently does not expose `pytest` through
`uv run pytest` or `uv run python -m pytest`, so this note does not claim a
fresh test-suite pass. Older documentation that stated `31 passed` is outdated
as a current verification claim.

## What Is Already Documented Well

The main project direction is documented consistently across:

- `README.md`;
- `AGENTS.md`;
- `docs/data/KUBERNETES_PREPROCESSING.md`;
- `docs/modeling/KUBERNETES_MODEL_V1.md`;
- `docs/data/STRUCTURAL_TARGETS_V1.md`;
- `docs/modeling/BASELINE_V1.md`;
- `docs/modeling/SFT_STRATEGY_V1.md`;
- `docs/evaluation/METRICAS_ACTUALES.md`.

The operative direction is:

```text
prompt -> latent intermediate representation -> blocks with level -> parser as structural control -> final YAML
```

The narrative that should guide new documentation and future chats is:

- LLMs generate flat token sequences;
- Kubernetes YAML encodes hierarchical structure;
- the project compares hierarchy learned as serialized output text against
  hierarchy predicted explicitly through `level`;
- parser-based reconstruction and automatic-preference optimization are control
  and alignment mechanisms, not evidence of a completed full RLHF pipeline.

The historical `docker-compose.yaml` framing is now background motivation only.
It must not override the current Kubernetes-centered implementation.

## Experiment Status

### Dataset And Structural Targets

This part is implemented and reproducible.

Current recorded facts:

- `quality_report.json` reports `sample_count = 283`;
- `quality_report.json` reports `prompt_variant_count = 566`;
- all normalized YAML targets parse successfully;
- structural targets have `status_counts.ok = 566`;
- SFT export has `ready_for_sft = true`.

These artifacts support the next modeling stage, but they are not themselves
model-training results.

### Baseline

The zero-shot baseline runner is implemented and has a completed recorded run on
the full `test` split:

- run id: `compact-test70-320-vtfix`;
- rows: `70`;
- output format: `blocks_tsv_compact_v1`;
- latent collection: enabled.

Headline metrics for that run:

- `structured_output_parse_success_rate = 0.8857`;
- `yaml_parse_success_rate = 0.4677`;
- `parsed_equal_rate = 0.0161`;
- `average_line_text_f1 = 0.3175`;
- `average_semantic_key_f1 = 0.4082`;
- latent vectors collected for all `70` rows with dimension `3584`.

The same run has also been recomputed with the expanded metric stack in
`metrics_recomputed.json`, adding prompt-requirement, Kubernetes-domain, BLEU,
ROUGE, and optional-perplexity fields. In that recomputation:

- `average_prompt_requirement_f1 = 0.3230`;
- `average_kubernetes_domain_validity_score = 0.3898`;
- `kubernetes_domain_gate_pass_rate = 0.0806`;
- `average_bleu_score = 0.4576`;
- `average_rougeL_f1 = 0.6584`.

Interpretation:

- the baseline is useful as a measurable pre-SFT reference;
- it is not a strong final model;
- its main remaining weakness is structural generation quality after
  reconstruction, not missing execution infrastructure.

### SFT

SFT is no longer only the next planned step. The serialized control branch has
been implemented and evaluated on the validation split.

Current status:

- SFT-ready JSONL files exist under `data/processed/kubernetes_v1/sft/`;
- the supervised comparison remains `serialized_sft` vs `two_head_sft`;
- `serialized_sft` is the implemented control branch where `level` is emitted
  as serialized text in `blocks_tsv_v1`;
- `two_head_sft` and later ordinal/regression variants have now been
  implemented and evaluated as experimental explicit-hierarchy branches;
- the current trainer supports LoRA, resumable checkpoints, W&B logging,
  checkpoint-retention error tolerance, streaming validation logs, and explicit
  CUDA OOM recovery through `--oom-recovery skip_batch`;
- there are completed `serialized_sft` and `two_head_sft` validation artifacts,
  but no completed DPO or PPO result tables.

Strongest recorded serialized SFT validation result:

- run id: `serialized-sft-a-v1-20260505-171226`;
- checkpoint: `checkpoint-step-159`;
- epochs: `3`;
- evaluated rows: `70/70`;
- `yaml_parse_success_rate = 0.9857`;
- `parsed_equal_rate = 0.1143`;
- `average_line_text_f1 = 0.8206`;
- `average_level_exact_match_rate = 0.7578`;
- `average_level_mae = 0.2723`;
- `average_prompt_requirement_f1 = 0.8531`;
- `average_semantic_key_f1 = 0.9552`;
- `required_field_complete_sample_rate = 1.0000`.

Operational caveat:

- the run used `--oom-recovery skip_batch`;
- two microbatches were skipped because of CUDA OOM;
- this must be reported as a reproducible training-condition caveat, not hidden
  as if the full train split had been optimized uniformly.

A later one-epoch run also completed:

- run id: `serialized-sft-a-v1-1epoch-20260507-1528`;
- checkpoint: `checkpoint-step-53`;
- evaluated rows: `70/70`;
- `yaml_parse_success_rate = 0.9000`;
- `average_line_text_f1 = 0.7337`;
- `average_level_exact_match_rate = 0.6644`;
- `average_prompt_requirement_f1 = 0.7716`;
- `average_kubernetes_domain_validity_score = 0.7667`;
- `kubernetes_domain_gate_pass_rate = 0.1429`.

Documentation should therefore describe `serialized_sft` as an achieved
supervised control result and as the strongest current practical base for DPO.
The `two_head_sft` family should be described as an explicit-hierarchy
experimental branch whose current results are limited by parseability and
training/interface stability.

Current interpretation after the two-head validation reports:

- `serialized_sft` remains clearly stronger as a complete parser-facing
  generator on validation;
- the first `two_head_sft` result is informative for the thesis, but not the
  preferred base for post-SFT preference optimization;
- DPO should therefore be applied first to `serialized_sft`, not to the current
  two-head checkpoint.

### Latent Level Probe

The repository now contains a completed diagnostic probe for whether the base
model's hidden states already encode the YAML `level` variable.

Completed run:

- run id: `latent-level-probe-real-full-20260513-1528`;
- stage: `all`;
- rows: `496`;
- train units: `426`;
- validation units: `70`;
- validation lines: `1800`;
- feature strategies: `record_prefix_state`, `line_prefix_state`,
  `line_first_token`, `line_last_token`, `line_mean`;
- probe families: majority baseline, previous-level baseline, linear probe,
  small MLP probe.

Best headline results on validation:

- `record_prefix_state + MLP`: `accuracy = 0.8594`, `level_mae = 0.2478`;
- `record_prefix_state + linear`: `accuracy = 0.8583`,
  `balanced_accuracy = 0.7647`, `macro_f1 = 0.7303`;
- `line_prefix_state + MLP`: `accuracy = 0.8072`,
  `balanced_accuracy = 0.7708`, `macro_f1 = 0.7364`;
- majority baseline: `accuracy = 0.1633`;
- previous-level baseline: `accuracy = 0.3522`.

Interpretation:

- the result supports the diagnostic claim that `level` information is
  recoverable from base-model hidden states;
- it does not prove that `two_head_sft` will outperform `serialized_sft`;
- it should be used to motivate the explicit hierarchy branch, not to replace
  the missing training comparison.

### DPO And PPO

DPO is part of the planned post-SFT direction. PPO is only an optional later
extension if reward quality, compute, and project scope justify it.

The first DPO methodology is now defined in:

- `docs/decisions/DPO_POST_SFT_ALIGNMENT_DECISION.md`;
- `docs/modeling/DPO_AUTOMATIC_PREFERENCE_V1.md`;
- `docs/evaluation/DPO_PREFERENCE_SCORING_V1.md`;
- `docs/memoria/notes/DPO_METHODOLOGY_MEMORIA_2026-05-24.md`.

The repository does not currently contain:

- a full preference dataset;
- a trained reward model;
- DPO results;
- PPO results;
- a full RLHF pipeline.

These stages must not be described as completed. The current `serialized_sft`
result is the selected base for the first automatic-preference DPO
construction, but no DPO stage has been executed yet.

### Kubernetes Validation

The current evaluation stack includes YAML parseability, structural checks,
parsed-document equality, block-level comparison, semantic-key overlap,
prompt-requirement approximations, and simple domain consistency checks.

It does not yet provide complete official Kubernetes schema validation or human
semantic evaluation.

When describing evaluation, use this distinction:

- implemented now: automatic syntactic, structural, block-level, and approximate
  semantic checks;
- planned or future: stronger Kubernetes schema validation and deeper semantic
  consistency evaluation.

## Anteproyecto Status

The current file `docs/memoria/templates/plantilla_anteproyecto.docx` is only the official
template. It still contains placeholder text such as `XXXXXX` and empty sections
for objectives, description, abstract, keywords, and approval fields.

Therefore, the anteproyecto description is not yet written in the DOCX. The
best current source for drafting it is this repository documentation, especially
`README.md`, `docs/modeling/KUBERNETES_MODEL_V1.md`, `docs/modeling/BASELINE_V1.md`, and
`docs/modeling/SFT_STRATEGY_V1.md`.

## Recommended Anteproyecto Title

Suggested Spanish title:

```text
Generacion estructurada de manifiestos YAML de Kubernetes a partir de lenguaje natural mediante modelos de lenguaje grandes
```

Alternative shorter title:

```text
Generacion estructurada de YAML de Kubernetes con modelos de lenguaje grandes
```

If the anteproyecto needs to preserve the broader thesis motivation, the title
can remain domain-general, but the description should explicitly say that the
implemented case study is Kubernetes.

## Recommended Anteproyecto Description

Suggested Spanish description:

```text
Este Trabajo Fin de Master parte de una tension entre el funcionamiento de los
modelos de lenguaje grandes y la naturaleza de ciertos artefactos tecnicos. Los
LLMs generan secuencias planas de tokens, mientras que manifiestos como los YAML
de Kubernetes codifican estructuras jerarquicas donde la validez depende de la
posicion, la indentacion, los campos anidados y las relaciones entre recursos.
El caso de estudio actual se centra en Kubernetes, un dominio donde las salidas
deben cumplir restricciones sintacticas, jerarquicas y semanticas.

El proyecto compara dos formas de abordar esa tension. La primera trata la
estructura como una propiedad de la superficie generada: el modelo emite una
representacion estructurada serializada y un parser determinista reconstruye y
valida el YAML final, con una posible fase posterior de optimizacion por
preferencias automaticas. La segunda trata el YAML como una estructura tipo
arbol proyectada en lineas: el orden de generacion proporciona la posicion de
cada entrada, pero el modelo debe predecir explicitamente su nivel jerarquico
mediante un cabezal estructural.

La evaluacion no se limita a similitud textual, sino que prioriza parseabilidad
YAML, consistencia estructural, adecuacion al prompt y senales automaticas de
validez de dominio. La linea experimental contempla un baseline zero-shot,
ajuste supervisado con LoRA y una comparacion supervisada entre un modelo
control que serializa el nivel jerarquico como texto y un modelo principal con
un cabezal explicito para predecir dicho nivel. Tras esa comparacion, el
proyecto contempla una fase posterior de optimizacion por preferencias
automaticas mediante DPO. PPO queda como una extension opcional condicionada a
la calidad de las recompensas automaticas y a la disponibilidad de computo.
```

## Recommended Objectives

Suggested objectives for the anteproyecto:

1. Build and document a reproducible Kubernetes dataset for prompt-to-YAML
   generation.
2. Define an intermediate structural representation based on YAML line blocks
   and explicit hierarchy levels.
3. Implement deterministic parser-based reconstruction from structured blocks
   to final YAML.
4. Establish a zero-shot baseline with resumable execution and automatic
   structural evaluation.
5. Train and evaluate a serialized SFT control model where `level` is generated
   as text.
6. Train and evaluate a main SFT + LoRA model with an explicit hierarchical
   `level` head.
7. Compare structural and semantic validity metrics across baseline,
   serialized SFT, and two-head SFT.
8. Prepare automatic preference or reward signals for a later DPO stage, without
   assuming a full RLHF pipeline.

## Wording To Avoid

Avoid claiming that:

- `two_head_sft` results already exist;
- DPO or PPO has been executed;
- the latent representation has already been validated as a final scientific
  contribution;
- parser-based control guarantees semantic correctness;
- Kubernetes schema validation is complete;
- the project is still mainly about `docker-compose.yaml`.

Use phrasing such as:

- "the repository currently implements";
- "the next experimental stage is";
- "the project plans to evaluate";
- "automatic semantic checks are approximate";
- "the historical motivation includes other YAML-based technical domains, but
  the current case study is Kubernetes".

## Practical Documentation Gap

The main remaining documentation gap is not the technical contract. It is the
formal academic synthesis:

- fill `docs/memoria/templates/plantilla_anteproyecto.docx`;
- keep the anteproyecto aligned with Kubernetes as the operative case study;
- distinguish implemented results from planned methodology;
- avoid presenting either the baseline or `serialized_sft` as final thesis
  closure;
- explain that `serialized_sft` is now a strong supervised control, while
  `two_head_sft` is still needed to test the explicit hierarchy hypothesis
  before DPO.
