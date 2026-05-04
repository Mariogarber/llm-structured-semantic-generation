# Multi-Resource Strategy Decision

## Status

Accepted as the next dataset-enrichment direction.

This document records a methodological decision. It does not claim that
`kubernetes_v2` has already been generated, trained, or validated.

## Context

The exploratory analysis of `Kubernetes v1` shows a clear compositional coverage
gap:

- `kubernetes_v1` contains 283 samples.
- These samples contain 338 parsed Kubernetes resource documents.
- 238 out of 283 samples contain a single YAML document.
- 45 samples contain multi-document or multi-resource compositions.

This should not be interpreted as a simple "one technology" limitation. In
Kubernetes, the more precise issue is that the base dataset mostly represents
single-resource manifests. Real Kubernetes configurations often require several
related resources, for example a `Deployment` together with a `Service`, a `Pod`
with a `ConfigMap`, or storage resources linked through `PersistentVolume` and
`PersistentVolumeClaim`.

The resulting risk is not only low variety. It is that a model trained mostly on
single-resource examples may learn to generate syntactically valid isolated YAML
without learning multi-resource relationships.

## Alternative Considered: Atomic Generation And Concatenation

One possible architecture was:

`complex prompt -> resource detection -> atomic generation per resource -> YAML concatenation`

This branch has real advantages:

- it fits the current single-resource-heavy dataset;
- it is modular and easier to debug;
- each resource can be generated and validated independently;
- the parser can still enforce structural validity at the YAML level.

However, it is not the preferred first enrichment path.

## Why The Atomic Branch Is Not Prioritized

The main weakness is semantic coherence across resources.

Generating several individually valid YAML documents does not guarantee that the
combined manifest is coherent. For example:

- a `Service.spec.selector` may not match the labels of the target `Deployment`;
- a `volumeMount.name` may not exist under `volumes`;
- a referenced `ConfigMap` or `Secret` may not be present;
- a `PersistentVolumeClaim` may not match the intended workload mount;
- a `serviceAccountName` may not reference an existing `ServiceAccount`;
- RBAC resources such as `Role`, `RoleBinding`, and `ServiceAccount` may be
  syntactically valid but semantically disconnected.

This approach also moves much of the thesis difficulty into additional
subsystems that are not yet implemented:

- intent decomposition;
- resource selection;
- cross-resource name and reference resolution;
- semantic assembly;
- multi-resource validation.

That would make it harder to know whether improvements come from the model, the
retrieval/decomposition step, the assembler, or hidden validation heuristics.
It could also hide the fact that the model itself has not learned multi-document
generation patterns.

The atomic branch is therefore not considered wrong. It is considered less
appropriate as the next main branch because it creates a new semantic assembler
problem before the model has been evaluated on controlled multi-resource
generation.

## Decision

The project will prioritize a multi-resource and multi-document dataset
enrichment branch.

The derived dataset version will be:

```text
data/processed/kubernetes_v2/
```

`kubernetes_v2` will be built from `kubernetes_v1` using controlled
compositional oversampling. It will not overwrite or redefine `kubernetes_v1`.

The base `kubernetes_v1` dataset remains the clean reference for baseline
experiments and for comparison against enriched training data.

## Methodological Rationale

This choice is aligned with the thesis direction because:

- Kubernetes manifests are often meaningful as sets of related resources;
- multi-resource generation better tests structural and semantic validity than
  isolated YAML generation;
- parser-based control remains useful but is not allowed to hide semantic
  relation errors;
- the latent-plus-block representation can naturally support multi-document
  outputs through `document_index`;
- evaluation can explicitly measure both YAML validity and cross-resource
  coherence.

The approach is more difficult than atomic generation, but it tests the problem
the thesis actually cares about: structured generation where several formal
objects must be produced as one coherent output.

## Rules For `kubernetes_v2`

`kubernetes_v2` must follow these rules:

- Source dataset: `data/processed/kubernetes_v1/`.
- Target dataset: `data/processed/kubernetes_v2/`.
- `kubernetes_v1` must not be modified.
- Synthetic samples must combine sources only within the same split.
- Initial compositions should contain 2 to 4 YAML documents.
- Every synthetic sample must preserve traceability.

Each synthetic row must record at least:

- `synthetic_sample_id`;
- `source_sample_ids`;
- `source_leakage_groups`;
- `split`;
- `composition_strategy`;
- `synthetic_prompt_strategy`;
- `target_yaml_normalized`.

The initial allowed composition families are:

- `Deployment + Service`;
- `Pod + ConfigMap`;
- `Pod + Secret`;
- `PersistentVolume + PersistentVolumeClaim + Pod`;
- `ServiceAccount + Role + RoleBinding`;
- `StatefulSet + Service`.

Additional composition families may be added later, but they must be documented
in the `kubernetes_v2` generation report.

## Validation Requirements

Every `kubernetes_v2` sample must pass at least:

- YAML parsing with `yaml.safe_load_all`;
- structural round-trip validation: `YAML -> blocks -> YAML`;
- split-consistency validation: no synthetic example combines different splits;
- source-traceability validation: every synthetic row points back to existing
  `kubernetes_v1` samples;
- prompt-coverage validation: the synthetic prompt mentions the resource kinds
  included in the target YAML.

The `kubernetes_v2` report must compare `kubernetes_v1` and `kubernetes_v2` on:

- documents per sample;
- `kind` combinations;
- YAML depth;
- total parsed YAML nodes;
- structural block count;
- semantic-key presence;
- duplicate and leakage behavior.

## Expected Research Use

`kubernetes_v2` should be used as an experimental enriched dataset, not as a
silent replacement for `kubernetes_v1`.

Recommended comparisons:

1. Baseline or SFT on `kubernetes_v1`.
2. SFT on `kubernetes_v2`.
3. Evaluation on both atomic and multi-resource scenarios.
4. Error analysis focused on cross-resource coherence.

## Assumptions And Limitations

- The term "multi-technology" is documented technically as multi-resource or
  multi-document Kubernetes generation.
- Oversampling is a controlled intervention to improve compositional coverage,
  not a guarantee of better model behavior.
- Synthetic prompts may initially be template-based; LLM rewriting can be added
  later if the generation method is recorded.
- The atomic branch remains a possible future comparison, especially if
  cross-resource assembly becomes a separate research question.

