# DPO Preference Scoring v1

Document type: contract

## Purpose

This document defines the automatic scoring policy used to construct preference
pairs for the first DPO experiment on Kubernetes v1.

The score is not a complete measure of Kubernetes correctness. It is a
traceable proxy used to rank candidate generations from the same prompt. Its
purpose is to choose which candidate should be treated as `chosen` and which
candidate should be treated as `rejected` for DPO.

The score must be interpreted together with the limitations of the current
evaluation stack:

- YAML parseability is implemented;
- parser-facing structural checks are implemented;
- prompt-requirement checks are approximate;
- Kubernetes-domain validity checks are automatic and partial;
- full official Kubernetes schema validation is not yet implemented;
- human semantic evaluation is not yet implemented.

## Input To The Scorer

The scorer receives one candidate generation for one prompt.

The expected model output surface is:

```text
blocks_tsv_v1
```

The candidate must be evaluated through the same path used for baseline and SFT:

```text
generated text -> block parser -> deterministic reconstruction -> YAML parsing -> metrics
```

The scorer may use only information available to the normal evaluation stack.
It must not inspect the test split when constructing training preferences.

## Hard Validity Gates

Before computing the preference score, the candidate is checked against hard
validity gates.

The candidate is considered invalid for preference ranking if any of the
following fail:

```text
structured_output_parse_success == false
yaml_parse_success == false
block_parse_success == false
```

Invalid candidates may still be used as rejected outputs if they are paired
against a valid candidate. They must not be used as chosen outputs.

Reason:

- DPO should not learn to prefer outputs that cannot pass the parser-facing
  contract.
- A non-parseable YAML output is not useful as a Kubernetes manifest.
- The parser is a structural control boundary, not a hidden repair system.

## Main Preference Score

For candidates that pass the hard gates, compute:

```text
score =
  1.00 * prompt_requirement_f1
+ 0.75 * kubernetes_domain_validity_score
+ 0.50 * required_field_complete_resource_rate
+ 0.25 * level_exact_match_rate
+ 0.25 * kubernetes_domain_gate_pass
- penalties
```

Where:

- `prompt_requirement_f1` measures approximate prompt adequacy.
- `kubernetes_domain_validity_score` gives a graded Kubernetes-domain signal.
- `required_field_complete_resource_rate` protects minimal manifest integrity.
- `level_exact_match_rate` protects hierarchy consistency relative to the
  reference structural target.
- `kubernetes_domain_gate_pass` gives an additional bonus for passing the full
  current KDV gate.

If a metric is unavailable because the candidate is invalid, it must be treated
as `0.0`.

Boolean metrics are converted as:

```text
true  -> 1.0
false -> 0.0
```

## Penalties

The first DPO experiment will use conservative penalties only.

Apply:

```text
penalties =
  0.25 * invented_resource_penalty
+ 0.25 * severe_line_count_penalty
```

`invented_resource_penalty` is `1.0` if the prediction introduces an obvious
primary resource kind that is unsupported by the prompt-requirement extractor
and absent from the reference kind sequence. Otherwise it is `0.0`.

`severe_line_count_penalty` is `1.0` if the generated line count is less than
half or more than twice the reference line count. Otherwise it is `0.0`.

These penalties are intentionally simple. The first DPO run should not introduce
a large hand-engineered reward function that becomes harder to explain than the
model behavior itself.

## Why Gate Pass Is Not The Only Objective

`kubernetes_domain_gate_pass` is important because it captures whether the
candidate passes the full current Kubernetes-domain gate. However, it is too
coarse to use as the only ranking signal.

If most candidates fail the gate, they would all receive the same value. DPO
would then receive little useful preference signal. For that reason, the scoring
function uses `kubernetes_domain_validity_score` as a graded signal and
`kubernetes_domain_gate_pass` as a bonus.

This also prevents the model from being rewarded only for passing a good-practice
check while ignoring prompt adequacy.

## Prompt Adequacy Priority

`prompt_requirement_f1` receives the largest weight because a Kubernetes
manifest can look valid and still fail the user request.

This is especially important for automatic preference optimization. If the
domain-validity component dominates the score too strongly, the model may learn
to add generic good-practice boilerplate while becoming less faithful to the
specific prompt.

The desired behavior is:

```text
first satisfy the request, then prefer the more valid Kubernetes realization
```

not:

```text
first maximize Kubernetes-looking policy defaults, regardless of the request
```

## Pair Construction

For each prompt, generate multiple candidate outputs from the frozen
`serialized_sft` policy.

Recommended first setting:

```text
k = 4 candidates per prompt
temperature in {0.4, 0.7}
top_p = 0.9
max_new_tokens = 1024
```

For each prompt:

1. Evaluate all candidates.
2. Remove duplicate candidate texts.
3. Rank candidates by hard validity first and score second.
4. Select the highest-scoring valid candidate as `chosen`.
5. Select a lower-scoring candidate as `rejected`.
6. Keep the pair only if the margin is large enough.

The minimum score margin is:

```text
score(chosen) - score(rejected) >= 0.15
```

If no pair satisfies the margin, the prompt contributes no DPO pair.

## Hard Negatives

The preferred rejected output is not always the worst possible output.

The first choice for `rejected` should be a hard negative:

- it is parseable;
- it roughly follows the output surface;
- but it is worse on prompt adequacy, Kubernetes validity, required fields, or
  hierarchy.

If no parseable hard negative exists, a non-parseable output may be used as
`rejected`, but those pairs should be counted separately in the dataset report.

Reason:

- trivial pairs teach the model little if `serialized_sft` already parses almost
  always;
- hard negatives better target the residual errors after SFT.

## Dataset Report Requirements

The preference dataset builder must produce a report with at least:

- number of prompts processed;
- number of candidates generated;
- number of duplicate candidates removed;
- number of valid candidates;
- number of DPO pairs retained;
- number of pairs dropped for insufficient margin;
- number of pairs with non-parseable rejected outputs;
- score distribution for chosen outputs;
- score distribution for rejected outputs;
- margin distribution;
- average values of each metric component for chosen and rejected outputs.

The report must also record:

- source split;
- source SFT checkpoint;
- decoding settings;
- scoring formula version;
- random seed;
- date or run id.

## Validation Split Use

The validation split may be used to:

- compare DPO hyperparameters;
- choose the DPO checkpoint;
- evaluate interpolation alpha values;
- audit preference over-optimization.

It must not be used to construct DPO training pairs.

## Test Split Use

The test split is reserved for final evaluation.

It must not be used to:

- choose score weights;
- tune `beta`;
- tune generation temperatures;
- choose interpolation alpha;
- inspect failures during iterative development.

## Failure Modes To Audit

The DPO experiment must explicitly audit these failure modes:

- higher Kubernetes validity but lower prompt adequacy;
- higher prompt adequacy but lower YAML parseability;
- more security or good-practice boilerplate that was not requested;
- degraded line structure or level consistency;
- shorter outputs that pass some metrics by omitting difficult content;
- longer outputs that add plausible but unsupported resources.

## Success Interpretation

DPO is considered successful only if it improves alignment metrics while
preserving the parser-facing structural behavior learned by SFT.

The minimum success condition is:

```text
average_prompt_requirement_f1 improves
or
average_kubernetes_domain_validity_score improves
```

and:

```text
yaml_parse_success_rate drops by no more than 0.02 absolute
```

relative to the frozen `serialized_sft` validation reference.

If DPO improves `kubernetes_domain_gate_pass_rate` but meaningfully reduces
prompt adequacy, the result is not considered a clean success. It is interpreted
as proxy over-optimization.

