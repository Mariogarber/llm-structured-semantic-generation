# DPO Labeling Guide v1

Document type: contract

## Version Note

This file is kept as the labeling contract for the first DPO preference
dataset. Do not reinterpret already exported `preferences_final.jsonl` rows
with later rules.

For the next preference dataset, use
[`DPO_LABELING_GUIDE_V2.md`](DPO_LABELING_GUIDE_V2.md). The v2 guide preserves
the same research framing, but changes the pair-selection policy after the
`beta=0.10` / `beta=0.30` comparison showed that changing `beta` on the current
pairs is not the main useful lever.

## Purpose

This guide defines how to label `chosen` / `rejected` pairs for the first
offline DPO dataset on Kubernetes v1.

The goal is not to reward generic Kubernetes-looking output. The goal is to
prefer the safest and most valid candidate that still answers the prompt.

## Objective

Preference pairs should teach the model to:

- parse YAML correctly and preserve the parser-facing block contract;
- increase `kubernetes_domain_gate_pass_rate`;
- keep high prompt adequacy;
- improve Kubernetes safety and quality practices without changing the request.

## Decision Rule

Do not mark a candidate as `chosen` for DPO if it does not parse as YAML or does
not respect the block contract.

A candidate with `kubernetes_domain_gate_pass = true` should be preferred when
it preserves the prompt or only loses a small amount under the approximate
`Prompt F1` metric. The default automatic tolerance for a gate/practice pair is:

```text
prompt_f1(rejected) - prompt_f1(chosen) <= 0.05
```

A sensitivity run may report a looser tolerance up to `0.10`, but it must be
kept separate in the dataset report. In both cases, the pair should still have
`score_margin >= 0.25`, the chosen candidate must not visibly lose a central
prompt requirement when the YAML is read, and the rejected candidate should not
be better on required-field completeness.

Extra safety fields such as `resources`, `securityContext`, non-latest image
tags, `runAsNonRoot`, or `readOnlyRootFilesystem` are positive when they do not
alter the requested resource names, ports, images, relationships, or intent.

If the strongest gate-pass candidate drifts from the prompt and another
parseable candidate is clearly more faithful, choose the prompt-faithful
candidate or mark `tie` / `skip`.

## Review Order

1. Check `YAML` and `Blocks`. Failed candidates should normally only be
   `rejected`.
2. Check `Gate pass`. Passing the full gate is a strong signal if prompt
   adequacy is preserved or only slightly worse under the approximate
   `Prompt F1` metric.
3. Check `KDV` and `kubernetes_domain_validity_level` as gradual Kubernetes
   quality signals when no candidate passes the full gate.
4. Check `Prompt F1`, but verify it by reading the prompt and YAML.
5. Use `Req fields`, `Level`, `Line F1`, and line counts as structural
   tie-breakers.

## Kubernetes Levels

- Level 0: YAML parses.
- Level 1: block contract and reconstruction are satisfied.
- Level 2: minimal Kubernetes identity: `apiVersion`, `kind`,
  `metadata.name`, known kind, and required fields.
- Level 3: intra-resource invariants: selectors, labels, ports, volumes,
  images, schedules, and replica ranges.
- Level 4: inter-resource invariants: local references and Service/workload
  consistency.
- Level 5: static quality and security checks: CPU/memory requests and limits,
  no `latest`, `runAsNonRoot`, `readOnlyRootFilesystem`, no privileged
  containers, and no host namespaces.

`gate_pass = true` means level 5 under the current automatic checks. It is not
proof of full Kubernetes correctness.

## Pair Types

Keep only informative pairs:

- Strong pair: `chosen` has the best score, `rejected` is worse, and
  `score_margin >= 0.25`.
- Intermediate pair: both parse, but one is clearly worse on prompt adequacy,
  KDV, gate pass, required fields, or hierarchy.
- Gate/practice pair: `chosen` has `gate_pass = true` and `rejected` has
  `gate_pass = false`. The preferred form has no prompt-adequacy regression,
  but an automatic pair may allow `prompt_f1_drop <= 0.05` when
  `score_margin >= 0.25`, required fields are not worse, and reading the YAML
  does not reveal a lost central requirement. A tolerance up to `0.10` is only
  for a separately reported sensitivity run.

Prefer hard negatives: rejected candidates that parse and roughly follow the
surface, but are clearly worse. Keep at most three useful pairs per prompt and
avoid near-duplicate comparisons.

## Prompt F1

`Prompt F1` is an approximate rule-based metric.

Use it as support when it matches the visible YAML behavior. Do not punish a
candidate automatically when Prompt F1 is low but the YAML satisfies the prompt.
Do not over-trust high Prompt F1 when the extractor captured only a trivial atom
such as `kind`.

For gate/practice pairs, a small Prompt F1 regression is allowed because the
extractor is approximate. If `chosen` has lower Prompt F1 than `rejected`, add
`gate_pass_prompt_drift`, use at most `confidence = medium`, and verify that the
difference is not a visible loss of names, ports, images, relations, resource
kinds, or other central prompt requirements.

Use metric flags when the metric is suspicious:

- `prompt_metric_unreliable`
- `under_extracted_prompt`
- `false_positive_requirement`
- `wrong_resource_context`
- `gate_pass_prompt_drift`
- `security_boilerplate_ok`
- `security_boilerplate_changes_intent`
- `hard_negative`

## Tie And Skip

Use `tie` when candidates are effectively equivalent and the pair would not
teach a clear preference.

Use `skip` when all candidates are poor, all drift from the prompt, or the
decision would depend on an unreliable metric.

Use `confidence = low` when the pair depends on semantic judgment not captured
by the current automatic metrics.
