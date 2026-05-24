# DPO Post-SFT Alignment Decision

Document type: decision

## Status

Accepted as the first post-SFT alignment methodology for the current
Kubernetes v1 line.

This document records a methodological decision. It does not claim that DPO has
already been implemented, trained, or validated in this repository.

## Context

The repository has already moved beyond a generic structured-output baseline.
The current effective case study is Kubernetes, and the strongest implemented
supervised model is the serialized SFT branch:

```text
serialized_sft
```

This model learns the parser-facing `blocks_tsv_v1` surface directly. In that
surface, each generated line contains:

```text
document_index    line_index    level    line_text
```

The validation result documented for
`serialized-sft-a-v1-20260505-171226` shows that this branch is a strong
supervised reference:

- `structured_output_parse_success_rate = 1.0000`
- `yaml_parse_success_rate = 0.9857`
- `average_prompt_requirement_f1 = 0.8531`
- `average_kubernetes_domain_validity_score = 0.7667`
- `kubernetes_domain_gate_pass_rate = 0.1429`

The two-head branch, in contrast, remains methodologically interesting but
unstable as a complete parser-facing generator in the current implementation.
Its first completed validation result is dominated by a parseability bottleneck.
This makes it a poor first base for post-SFT preference optimization.

The practical question is therefore no longer whether to train from scratch or
whether to replace SFT. The question is whether a structurally competent SFT
policy can be nudged toward better Kubernetes-domain behavior and better prompt
adequacy using automatic preferences.

## Decision

The project will use **offline Direct Preference Optimization (DPO)** as the
first post-SFT alignment method.

The alignment branch will be:

```text
serialized_sft -> automatic preference dataset -> serialized_sft_dpo
```

The reference policy for DPO will be the frozen serialized SFT checkpoint:

```text
results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/
```

The preferred checkpoint is the selected validation checkpoint:

```text
checkpoint-step-159
```

The DPO stage will be documented as automatic-preference optimization, not as a
full RLHF pipeline. The repository does not currently contain:

- a human preference dataset;
- a trained reward model;
- online human feedback;
- PPO results;
- DPO results.

## Why DPO First

DPO is the best first method for this repository state because it matches the
available ingredients:

- a strong SFT policy exists;
- the SFT policy can serve as the DPO reference model;
- candidate outputs can be sampled from the same SFT policy;
- automatic evaluators already provide structural, prompt, and Kubernetes
  validity signals;
- the project has limited compute and should avoid unnecessary online RL
  complexity.

DPO is also aligned with the thesis narrative. The aim is not to replace the
parser or to invent hidden repair logic. The aim is to train the model to prefer
outputs that already score better under explicit, auditable structural and
domain checks.

## Why Not PPO First

PPO remains a possible later extension, but it is not the first alignment
method.

PPO would require substantially more infrastructure:

- an explicit reward model or a very stable scalar reward function;
- online rollouts from the current policy;
- KL control against a reference model;
- value estimation or advantage estimation;
- careful batch-size, clipping, and normalization choices;
- more compute and more sensitivity to hyperparameters.

The local bibliography supports this caution. `dpo_vs_ppo.pdf` argues that PPO
can outperform DPO when implemented and tuned carefully, but the same paper also
shows that this depends on nontrivial implementation details. At the current
project stage, PPO would make it harder to separate model behavior from reward
engineering and training instability.

PPO can be reconsidered only if:

- DPO fails to improve the alignment-retention tradeoff;
- the automatic reward function is shown to be stable;
- the project has enough compute for online training;
- the thesis needs a reward-based RL comparison rather than an offline
  preference comparison.

## Why Not ORPO Or KTO First

ORPO and KTO are relevant alternatives, but neither is the preferred first
method.

ORPO combines supervised adaptation and preference alignment in a single
objective. That is useful when the project is still deciding how to train the
main instruction-following policy. In this repository, however, a strong
serialized SFT checkpoint already exists. Replacing the already measured SFT
stage with a new monolithic objective would blur the comparison.

KTO is attractive when feedback is available as independent desirable or
undesirable labels rather than paired preferences. In this project, paired
preferences can be constructed naturally by sampling several candidate outputs
for the same prompt and ranking them with the existing evaluator. DPO therefore
uses the available signal more directly.

Both ORPO and KTO may be mentioned in the memoria as related preference
optimization alternatives. They should not be presented as implemented unless a
separate experiment is actually run.

## Automatic Preferences

The DPO dataset will contain triples:

```text
prompt, chosen, rejected
```

where:

- `prompt` is the SFT prompt from the Kubernetes v1 training split;
- `chosen` is the candidate output preferred by the automatic scoring function;
- `rejected` is a worse but informative candidate for the same prompt.

Preferences will be built from automatic signals only. The main signals are:

- `prompt_requirement_f1`;
- `kubernetes_domain_validity_score`;
- `kubernetes_domain_gate_pass`;
- required-field completeness;
- structural and level consistency.

The exact scoring contract is defined in:

```text
docs/evaluation/DPO_PREFERENCE_SCORING_V1.md
```

## Alignment Tax And Pareto Analysis

The DPO checkpoint must not be treated as automatically better just because it
optimizes preferences. Preference optimization can improve the target behavior
while degrading useful behavior learned by SFT. This is the alignment-tax risk
documented in the local bibliography.

For this reason, the project will evaluate a Pareto-style interpolation between
the SFT checkpoint and the DPO checkpoint:

```text
theta_alpha = (1 - alpha) theta_sft + alpha theta_dpo
```

When using LoRA adapters, the practical interpolation should be between
compatible adapter weights:

```text
adapter_alpha = (1 - alpha) adapter_sft + alpha adapter_dpo
```

The interpolation sweep will use:

```text
alpha in {0.0, 0.1, 0.2, ..., 1.0}
```

The goal is to identify non-dominated points between:

- alignment metrics: prompt adequacy and Kubernetes validity;
- retention metrics: parseability, block reconstruction, line-text quality, and
  level consistency.

## Expected Thesis Interpretation

If DPO improves Kubernetes validity or prompt adequacy without damaging
parseability, it supports the idea that automatic preference optimization is a
useful post-SFT control mechanism for structured generation.

If DPO improves Kubernetes good-practice metrics but damages prompt fidelity or
parseability, the result should be interpreted as over-optimization of proxy
signals.

If DPO produces no clear improvement, the result is still useful. It would show
that either:

- the serialized SFT model is already close to the limit of the current
  automatic metrics;
- the automatic preference function is too weak;
- the dataset does not contain enough candidate diversity;
- deeper semantic validation is needed before preference optimization can add
  value.

## Assumptions

- Kubernetes v1 remains the base dataset for this experiment.
- The DPO stage uses `serialized_sft`, not the current two-head model.
- The DPO stage is offline.
- The preference data is automatic, not human-labeled.
- The parser remains a structural control boundary, not a semantic repair
  system.
- Test data is reserved for final evaluation and is not used to tune scoring,
  beta, generation temperature, or interpolation alpha.

