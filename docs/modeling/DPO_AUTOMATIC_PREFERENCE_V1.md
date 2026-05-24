# DPO Automatic Preference Optimization v1

Document type: contract

## Purpose

This document defines the first DPO methodology for the Kubernetes v1 modeling
line.

The goal is to test whether automatic preference optimization can improve a
strong serialized SFT model without damaging the structural guarantees that make
the model useful as a Kubernetes manifest generator.

The experiment is intentionally narrow. It does not introduce a full RLHF
pipeline, a human preference dataset, an online reward model, or PPO training.
It studies one specific question:

```text
Can a structurally competent serialized SFT model be improved by DPO using
automatic preferences derived from parser, prompt, and Kubernetes-domain checks?
```

## Position In The Project

The current thesis narrative studies a tension between:

- autoregressive language models, which generate flat token sequences;
- Kubernetes YAML manifests, whose correctness depends on hierarchical and
  domain-specific structure.

The repository has already tested:

- a zero-shot baseline;
- a serialized SFT model that emits `level` as text in `blocks_tsv_v1`;
- two-head variants that try to predict `level` through a structural head.

The first DPO experiment follows the strongest currently implemented branch:

```text
prompt -> serialized_sft -> blocks_tsv_v1 -> parser -> YAML
```

DPO is applied after SFT:

```text
serialized_sft -> serialized_sft_dpo
```

The experiment does not replace the earlier supervised comparison. It extends
the best practical supervised policy with automatic preference optimization.

## Bibliographic Basis

The decision is grounded in the local bibliography:

- `bib/dpo.pdf`: DPO optimizes preferences directly without a reward model or
  online RL loop.
- `bib/dpo_vs_ppo.pdf`: DPO is simpler, but can be sensitive to distribution
  shift; PPO can be stronger when tuned carefully, but is substantially more
  complex.
- `bib/27_rlhf_workflow_reward_modeling_online_rlhf.pdf`: SFT is the normal
  base policy before preference-based alignment.
- `bib/26_provably_robust_dpo_noisy_feedback.pdf`: noisy or ambiguous
  preferences can damage DPO, motivating pair margins and conservative
  filtering.
- `bib/aligment_tax.pdf`: preference optimization can create an
  alignment-retention tradeoff, motivating Pareto analysis through
  interpolation.
- `bib/orpo.pdf` and `bib/28_kto_prospect_theoretic_optimization.pdf`: ORPO and
  KTO are relevant alternatives, but not the first method chosen here.

## Formal Setup

Let:

```text
x  = natural-language prompt plus the SFT structural instruction
y  = generated blocks_tsv_v1 output
pi_ref = frozen serialized_sft reference policy
pi_theta = trainable DPO policy initialized from pi_ref
```

The autoregressive policy is:

```text
pi_theta(y | x) = product_t pi_theta(y_t | x, y_<t)
```

The preference dataset is:

```text
D = {(x_i, y_w_i, y_l_i)}
```

where:

- `y_w` is the preferred candidate;
- `y_l` is the rejected candidate;
- both candidates are generated for the same prompt;
- the preference is assigned by automatic scoring.

Under the Bradley-Terry preference model:

```text
P(y_w > y_l | x) = sigma(r*(x, y_w) - r*(x, y_l))
```

The KL-regularized RLHF objective can be written as:

```text
max_pi E[x,y~pi] [r(x,y)] - beta * D_KL(pi(.|x) || pi_ref(.|x))
```

The optimal policy for a reward has the form:

```text
pi*(y|x) = (1 / Z(x)) * pi_ref(y|x) * exp(r(x,y) / beta)
```

Rearranging gives:

```text
r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Because Bradley-Terry uses reward differences, the partition function cancels.
DPO therefore fits the policy directly with:

```text
L_DPO(theta) =
  - E_(x,y_w,y_l)~D log sigma(
      beta * [
        log pi_theta(y_w|x) - log pi_ref(y_w|x)
      - log pi_theta(y_l|x) + log pi_ref(y_l|x)
      ]
    )
```

Equivalently:

```text
Delta_theta = log pi_theta(y_w|x) - log pi_theta(y_l|x)
Delta_ref   = log pi_ref(y_w|x)   - log pi_ref(y_l|x)

L_DPO = - E log sigma(beta * (Delta_theta - Delta_ref))
```

The update increases the relative likelihood of preferred outputs over rejected
outputs, but only relative to the reference policy. This is important because
the reference policy already encodes the SFT behavior that must be retained.

## Reference Model

The DPO reference policy is:

```text
serialized_sft
```

The canonical run is:

```text
results/sft_kubernetes_v1/serialized-sft-a-v1-20260505-171226/
```

The selected checkpoint is:

```text
checkpoint-step-159
```

This model is used in two ways:

1. to generate candidate outputs for the preference dataset;
2. as the frozen `pi_ref` inside the DPO loss.

Using the same SFT family for both roles reduces distribution shift. This is
important because DPO is offline and can become unstable when the preference
dataset is far from the reference policy distribution.

## Candidate Generation

Candidate generation must be incremental and resumable.

Minimum command interface for the future script:

```text
--output-dir
--run-id
--batch-size
--sft-run-dir
--checkpoint
--train-file
--num-candidates
--temperature
--top-p
--max-new-tokens
```

Recommended first settings:

```text
num_candidates = 4
temperature = 0.4 and 0.7 as two candidate sources
top_p = 0.9
max_new_tokens = 1024
```

Candidate generation uses only the Kubernetes v1 training split for DPO
training preferences.

The output artifact should be append-only:

```text
candidates.jsonl
```

Each row should contain at least:

```text
unit_id
sample_id
prompt_variant
split
prompt
candidate_id
generation_seed
decoding_config
raw_generation
normalized_generation
```

## Candidate Evaluation

Each candidate must be passed through the normal parser-facing evaluation stack:

```text
raw_generation
-> extract blocks_tsv_v1
-> reconstruct YAML
-> parse YAML
-> evaluate structure, prompt adequacy, and Kubernetes-domain signals
```

The evaluated artifact should be append-only:

```text
candidate_metrics.jsonl
```

Each row should include:

```text
structured_output_parse_success
yaml_parse_success
block_parse_success
prompt_requirement_f1
kubernetes_domain_validity_score
kubernetes_domain_gate_pass
required_field_complete_resource_rate
level_exact_match_rate
line_text_f1
line_count_match
preference_score
```

The scoring formula is defined in:

```text
docs/evaluation/DPO_PREFERENCE_SCORING_V1.md
```

## Preference Pair Construction

The DPO pair dataset must be derived from evaluated candidates.

For each prompt:

1. group candidates by `unit_id`;
2. remove exact duplicate candidate text;
3. compute preference scores;
4. select the best valid candidate as `chosen`;
5. select an informative lower-scoring candidate as `rejected`;
6. retain the pair only if the score margin is at least `0.15`.

The preferred rejected candidate is a hard negative: parseable but worse. If no
parseable rejected candidate exists, a non-parseable candidate may be used, but
the pair must be marked as a trivial-invalid rejection.

The final DPO dataset artifact is:

```text
preferences.jsonl
```

Each row should contain:

```text
unit_id
sample_id
prompt_variant
prompt
chosen
rejected
chosen_candidate_id
rejected_candidate_id
chosen_score
rejected_score
score_margin
chosen_metrics
rejected_metrics
preference_score_version
```

## DPO Training

The DPO trainer should be resumable and should follow the repository LLM-run
contract.

Minimum command interface:

```text
--output-dir
--run-id
--batch-size
--preference-file
--reference-model-path
--sft-adapter-path
--beta
```

Recommended beta sweep:

```text
beta in {0.05, 0.10, 0.20}
```

The run directory must contain:

```text
config.json
state.json
train_log.jsonl
validation_metrics_progress.jsonl
validation_predictions.jsonl
checkpoints/
metrics.json
```

`metrics.json` is written only after successful completion.

The DPO model must preserve the same output surface:

```text
blocks_tsv_v1
```

No raw-YAML DPO branch is part of this first experiment.

## Checkpoint Selection

Checkpoint selection is based on validation, not training loss alone.

The primary validation criteria are:

1. preserve `yaml_parse_success_rate`;
2. improve `average_prompt_requirement_f1`;
3. improve `average_kubernetes_domain_validity_score`;
4. improve or preserve `kubernetes_domain_gate_pass_rate`;
5. preserve `average_line_text_f1` and `average_level_exact_match_rate`.

The first DPO run is considered useful if:

```text
average_prompt_requirement_f1 improves
or
average_kubernetes_domain_validity_score improves
```

while:

```text
yaml_parse_success_rate >= serialized_sft_yaml_parse_success_rate - 0.02
```

Given the documented serialized SFT reference:

```text
serialized_sft_yaml_parse_success_rate = 0.9857
```

the acceptable lower bound is:

```text
yaml_parse_success_rate >= 0.9657
```

This threshold is a validation criterion, not a claim of final test
performance.

## Pareto Interpolation

After selecting the best DPO checkpoint, perform interpolation between SFT and
DPO.

The conceptual interpolation is:

```text
theta_alpha = (1 - alpha) * theta_sft + alpha * theta_dpo
```

For LoRA, the practical interpolation is between compatible adapter weights:

```text
adapter_alpha = (1 - alpha) * adapter_sft + alpha * adapter_dpo
```

Evaluate:

```text
alpha in {0.0, 0.1, 0.2, ..., 1.0}
```

For each alpha, compute the full validation stack.

The Pareto axes are:

```text
alignment:
  average_prompt_requirement_f1
  average_kubernetes_domain_validity_score
  kubernetes_domain_gate_pass_rate

retention:
  structured_output_parse_success_rate
  yaml_parse_success_rate
  average_line_text_f1
  average_level_exact_match_rate
```

The selected alpha should be a non-dominated point. A point is dominated if
another alpha has equal or better alignment metrics and equal or better
retention metrics, with at least one strict improvement.

## Final Test Evaluation

The test split is evaluated only once for the selected final candidate.

The final candidate can be:

- the best DPO checkpoint directly; or
- an interpolated SFT-DPO checkpoint chosen on validation.

The test report must clearly state which was selected.

It must compare at least:

```text
baseline
serialized_sft
serialized_sft_dpo_selected
```

If interpolation is used, also report:

```text
selected alpha
```

## Expected Result Forms

Possible outcomes:

### Clean improvement

DPO improves prompt adequacy and/or Kubernetes validity while retaining
parseability and structural quality.

Interpretation:

- automatic preferences are useful as a post-SFT control mechanism;
- DPO is justified as the first lightweight alignment method.

### Proxy over-optimization

DPO improves `kubernetes_domain_gate_pass_rate` but reduces prompt adequacy or
parseability.

Interpretation:

- the score overweights domain good-practice signals;
- the preference function needs better balance;
- this is an alignment-tax example, not a clean success.

### No measurable gain

DPO remains close to SFT or underperforms SFT.

Interpretation:

- the SFT model may already be near the limit of the current automatic checks;
- candidate diversity may be too low;
- the preference score may be too weak;
- more semantic validators may be needed.

### Pareto-only gain

The raw DPO checkpoint is not superior, but an interpolated checkpoint improves
the validation tradeoff.

Interpretation:

- DPO moved the model toward the desired behavior but overshot the best
  retention point;
- interpolation provides a useful control mechanism for alignment tax.

## Reporting Requirements

Every DPO report must include:

- source SFT checkpoint;
- candidate generation settings;
- preference scoring version;
- number of generated candidates;
- number of retained preference pairs;
- beta value;
- checkpoint selection policy;
- validation metrics;
- Pareto interpolation table if interpolation is run;
- test metrics only for the final selected candidate.

The report must explicitly state that the preference labels are automatic proxy
labels, not human judgments.

## Things This Experiment Must Not Claim

The DPO experiment must not claim:

- full RLHF has been implemented;
- human preferences were used;
- Kubernetes semantic correctness is guaranteed;
- parser success proves Kubernetes validity;
- DPO is universally superior to PPO;
- the score is a complete reward function for Kubernetes manifests.

## Open Variables

The following remain open for later experiments:

- whether a stronger Kubernetes schema validator should be added;
- whether human inspection should be used for a small audit set;
- whether iterative DPO is useful;
- whether PPO becomes worthwhile after reward validation;
- whether KTO or ORPO should be tested as lighter alternatives.

