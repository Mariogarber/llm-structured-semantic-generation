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
evaluation helpers, latent-vector collection, prompt-requirement extraction, and
resumable execution utilities exist in the repository.

The current test suite passes:

```text
31 passed
```

## What Is Already Documented Well

The main project direction is documented consistently across:

- `README.md`;
- `AGENTS.md`;
- `docs/KUBERNETES_PREPROCESSING.md`;
- `docs/KUBERNETES_MODEL_V1.md`;
- `docs/STRUCTURAL_TARGETS_V1.md`;
- `docs/BASELINE_V1.md`;
- `docs/SFT_STRATEGY_V1.md`;
- `docs/METRICAS_ACTUALES.md`.

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

Interpretation:

- the baseline is useful as a measurable pre-SFT reference;
- it is not a strong final model;
- its main remaining weakness is structural generation quality after
  reconstruction, not missing execution infrastructure.

### SFT

SFT is the correct next experimental step, but it has not yet been run.

Current status:

- SFT-ready JSONL files exist under `data/processed/kubernetes_v1/sft/`;
- the planned supervised comparison is `serialized_sft` vs `two_head_sft`;
- `serialized_sft` is the control branch where `level` is emitted as serialized
  text;
- `two_head_sft` is the main branch with an explicit hierarchical-level head;
- both branches should use LoRA or QLoRA-style adaptation under the same
  evaluation protocol;
- the trainer itself is not yet implemented;
- there are no completed SFT result tables.

Documentation should therefore describe SFT as a planned next experiment, not as
an achieved result.

### DPO And PPO

DPO is part of the planned post-SFT direction. PPO is only an optional later
extension if reward quality, compute, and project scope justify it.

The repository does not currently contain:

- a full preference dataset;
- a trained reward model;
- DPO results;
- PPO results;
- a full RLHF pipeline.

These stages must not be described as completed.

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

The current file `docs/plantilla_anteproyecto.docx` is only the official
template. It still contains placeholder text such as `XXXXXX` and empty sections
for objectives, description, abstract, keywords, and approval fields.

Therefore, the anteproyecto description is not yet written in the DOCX. The
best current source for drafting it is this repository documentation, especially
`README.md`, `docs/KUBERNETES_MODEL_V1.md`, `docs/BASELINE_V1.md`, and
`docs/SFT_STRATEGY_V1.md`.

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

- SFT results already exist;
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

- fill `docs/plantilla_anteproyecto.docx`;
- keep the anteproyecto aligned with Kubernetes as the operative case study;
- distinguish implemented results from planned methodology;
- avoid presenting the baseline as a final successful model;
- explain why the supervised `serialized_sft` vs `two_head_sft` comparison is
  the next necessary step before DPO.
