# DPO Labeling Guide v2

Document type: contract

## Purpose

This guide defines how to build the second offline DPO preference dataset for
Kubernetes v1.

The v2 dataset must not overwrite or reinterpret the v1 preference dataset. It
should be exported as a new artifact, preferably under:

```text
results/dpo_kubernetes_v1/preference_annotation/agent-full-auto-v2/
```

The central goal changes slightly from v1. The first DPO runs showed that
`beta=0.10` and `beta=0.30` produce almost identical validation outputs on the
matched subset. Therefore, v2 should focus less on changing the DPO
hyperparameter and more on making the preference signal sharper.

The dataset should teach the model to prefer candidates that:

- keep YAML parseability and the parser-facing block contract;
- preserve the natural-language prompt;
- improve Kubernetes domain validity, especially level 5 static quality and
  security checks;
- avoid rewarding generic security boilerplate that changes the requested
  manifest;
- use hard negatives that are plausible Kubernetes outputs, not only broken
  or trivial failures.

## Scope And Artifacts

Use only the `train` split for preference construction. Do not use validation
or test examples for preference generation, pair selection, threshold tuning, or
manual inspection during dataset construction.

The prompt units remain fixed to Kubernetes v1. Do not add new prompt units to
make the DPO dataset larger. If more evidence is needed, generate more
candidates for existing train units only, and keep that candidate generation run
as a separate resumable artifact.

The exported dataset should keep the v1 row shape when possible:

- `unit_id`, `sample_id`, `prompt_variant`, and `split`;
- SFT prompt context;
- chosen and rejected model outputs;
- reconstructed YAML, if available;
- metric snapshots for both sides;
- annotation source, confidence, rationale, pair type, and metric flags.

The final report should record:

- final pair count and number of train units with at least one pair;
- pair counts by type;
- duplicate count;
- distribution of chosen and rejected `kubernetes_domain_validity_level`;
- number of pairs that cross `kubernetes_domain_gate_pass`;
- average score margin;
- frequency of prompt-drift and metric-unreliability flags.

## Eligibility Rules

A candidate can be `chosen` only if it:

- parses as YAML;
- satisfies the block contract;
- preserves the central resource kind, names, ports, images, relations, and
  requested intent visible in the prompt;
- does not add unrelated resources or safety fields that change the task.

For v2, prefer pairs where both sides parse and satisfy the block contract. The
main training signal should come from hard negatives: outputs that look
plausible but fail a meaningful domain, prompt, or hierarchy criterion.

Do not export a pair when:

- `chosen` and `rejected` are near-duplicates;
- the preference depends only on a tiny metric difference that is not visible in
  the YAML;
- the chosen candidate has worse prompt adequacy in a central requirement;
- the chosen candidate wins only by adding generic security boilerplate that was
  not compatible with the requested resource;
- both candidates are poor and neither represents a useful target behavior.

## Automatic Guardrails

The automatic proposal stage should enforce these default constraints before a
pair is eligible for export:

```text
score_margin >= 0.25
chosen_prompt_f1 >= rejected_prompt_f1 - 0.05
chosen_line_text_f1 >= rejected_line_text_f1 - 0.10
chosen_level_exact_match_rate >= rejected_level_exact_match_rate - 0.15
chosen_required_field_presence_rate >= rejected_required_field_presence_rate - 0.05
```

If one of these metrics is unavailable for a pair, the automatic stage should
not treat the constraint as satisfied by default. Either skip the pair or send
it to manual/agent-assisted review with an explicit rationale.

Manual or agent-assisted review may approve a pair that slightly violates one
guardrail only when the YAML inspection makes the preference clear. Such rows
must use at most `confidence = medium` and include the relevant metric flag.

Do not use a looser prompt tolerance by default in v2. The first DPO runs
already suggest that partial domain improvements are not enough if they do not
generalize across the validation split.

## Pair Types

Use a small set of explicit pair types. Each exported pair should have one main
type, even if several signals are present.

- `gate_crossing`: `chosen` passes `kubernetes_domain_gate_pass` and
  `rejected` does not, while prompt adequacy is preserved.
- `level5_practice`: both candidates fail the full gate, but `chosen` fixes
  meaningful level 5 errors such as missing resources, `latest` images,
  missing `runAsNonRoot`, missing `readOnlyRootFilesystem`, privileged
  containers, or host namespace use.
- `domain_invariant`: `chosen` improves level 3 or level 4 invariants such as
  selectors, labels, service/workload consistency, ports, volumes, schedules,
  images, or replica ranges.
- `prompt_fidelity`: both candidates are structurally comparable, but `chosen`
  better satisfies the natural-language request.
- `structural_fidelity`: both candidates answer the prompt similarly, but
  `chosen` has a clearly better block contract, line structure, or level
  hierarchy.

Avoid exporting many easy `parse_failure` pairs. They are allowed only for a
small diagnostic slice when the chosen output is clean and the report marks
them separately. They should not dominate v2, because the current bottleneck is
not basic YAML parseability.

## Review Order

Review each prompt unit in this order:

1. Remove invalid candidates from the chosen side.
2. Identify any candidate that passes the Kubernetes gate and verify prompt
   adequacy by reading the YAML.
3. If no candidate passes the gate, compare level 5 errors and prefer the
   candidate with safer static practices when intent is preserved.
4. If level 5 is tied or not applicable, compare level 3 and level 4 domain
   invariants.
5. Use prompt adequacy as a hard constraint, not just as a tie-breaker.
6. Use line/level metrics only after YAML, domain validity, and prompt adequacy
   have been checked.

When several rejected candidates express the same failure mode, keep only the
most informative one. Prefer one clear pair per failure category over many
redundant comparisons.

## Redundancy Limits

Use these defaults:

```text
max_pairs_per_unit = 4
preferred_pairs_per_unit = 1 to 2
target_final_pairs = 800 to 1200
ceiling_final_pairs = 1500
```

The target is aspirational and depends on the available candidate pool. Do not
force the count by accepting weak, redundant, or prompt-drifting pairs.
If the final count remains below the target, report the limiting reason instead
of relaxing the guide silently.

Deduplicate by exact `unit_id`, chosen output, rejected output, and pair type.
If two pairs teach the same preference with the same chosen candidate, keep the
pair with the clearer rejected failure or larger score margin.

## Metric Flags

Carry forward the v1 flags and add v2-specific flags where useful:

- `prompt_metric_unreliable`
- `under_extracted_prompt`
- `false_positive_requirement`
- `wrong_resource_context`
- `gate_pass_prompt_drift`
- `security_boilerplate_ok`
- `security_boilerplate_changes_intent`
- `hard_negative`
- `level5_practice_gain`
- `domain_invariant_gain`
- `near_duplicate_rejected`
- `manual_guardrail_override`

Rows with `security_boilerplate_changes_intent`, `near_duplicate_rejected`, or
unresolved prompt drift should normally be skipped.

## Confidence Policy

Use `confidence = high` when the metrics and the visible YAML agree.

Use `confidence = medium` when the preference is visible but one metric is
noisy, approximate, or slightly violates a guardrail.

Do not export `confidence = low` rows into the final DPO dataset. Low-confidence
decisions may remain in annotation logs as `tie` or `skip`, but they should not
be part of `preferences_final.jsonl`.

## Tie And Skip

Use `tie` when candidates are effectively equivalent or when the difference
would not teach a stable preference.

Use `skip` when:

- all candidates drift from the prompt;
- all candidates fail for different severe reasons;
- the preferred output would depend on unverified semantic judgement;
- the only available differences are tiny line/level metric movements;
- the pair would reward Kubernetes-looking text without improving the actual
  requested manifest.

The v2 dataset should be smaller rather than noisier if the candidate pool does
not contain enough hard negatives.
