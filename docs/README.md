# Documentation Index

This directory is organized by the role each document plays in the project.
Use this file as the first stop when looking for documentation.

## Start Here

- Project framing: [`../README.md`](../README.md)
- Agent operating rules: [`../AGENTS.md`](../AGENTS.md)
- Current status and anteproyecto note: [`project/PROJECT_STATUS_AND_ANTEPROYECTO.md`](project/PROJECT_STATUS_AND_ANTEPROYECTO.md)
- Glossary: [`reference/TERMINOLOGY.md`](reference/TERMINOLOGY.md)

## Core Contracts

- Dataset and preprocessing contract: [`data/KUBERNETES_PREPROCESSING.md`](data/KUBERNETES_PREPROCESSING.md)
- Structural target contract: [`data/STRUCTURAL_TARGETS_V1.md`](data/STRUCTURAL_TARGETS_V1.md)
- Modeling contract: [`modeling/KUBERNETES_MODEL_V1.md`](modeling/KUBERNETES_MODEL_V1.md)
- SFT strategy and main comparison: [`modeling/SFT_STRATEGY_V1.md`](modeling/SFT_STRATEGY_V1.md)
- DPO automatic preference contract: [`modeling/DPO_AUTOMATIC_PREFERENCE_V1.md`](modeling/DPO_AUTOMATIC_PREFERENCE_V1.md)
- Current metrics: [`evaluation/METRICAS_ACTUALES.md`](evaluation/METRICAS_ACTUALES.md)
- DPO preference scoring contract: [`evaluation/DPO_PREFERENCE_SCORING_V1.md`](evaluation/DPO_PREFERENCE_SCORING_V1.md)
- DPO preference annotation and prompt audit: [`evaluation/DPO_PREFERENCE_ANNOTATION_AND_PROMPT_AUDIT_V1.md`](evaluation/DPO_PREFERENCE_ANNOTATION_AND_PROMPT_AUDIT_V1.md)
- DPO labeling guide: [`evaluation/DPO_LABELING_GUIDE_V1.md`](evaluation/DPO_LABELING_GUIDE_V1.md)
- DPO labeling guide v2: [`evaluation/DPO_LABELING_GUIDE_V2.md`](evaluation/DPO_LABELING_GUIDE_V2.md)

## Experiments And Analyses

- Baseline contract: [`modeling/BASELINE_V1.md`](modeling/BASELINE_V1.md)
- Baseline run reports: [`experiments/baseline/runs/`](experiments/baseline/runs/)
- Serialized SFT run reports: [`experiments/serialized_sft/runs/`](experiments/serialized_sft/runs/)
- DPO run reports: [`experiments/dpo/runs/`](experiments/dpo/runs/)
- DPO beta 0.10 full validation result: [`experiments/dpo/runs/DPO_BETA010_FULL_VALIDATION_RESULT_2026-05-30.md`](experiments/dpo/runs/DPO_BETA010_FULL_VALIDATION_RESULT_2026-05-30.md)
- DPO beta 0.30 matched validation comparison: [`experiments/dpo/runs/DPO_BETA030_VALIDATION_COMPARISON_2026-05-31.md`](experiments/dpo/runs/DPO_BETA030_VALIDATION_COMPARISON_2026-05-31.md)
- Two-head SFT contract: [`modeling/TWO_HEAD_SFT_V1.md`](modeling/TWO_HEAD_SFT_V1.md)
- Two-head SFT run reports: [`experiments/two_head_sft/runs/`](experiments/two_head_sft/runs/)
- Ordinal density proposal: [`modeling/TWO_HEAD_ORDINAL_DENSITY_V2.md`](modeling/TWO_HEAD_ORDINAL_DENSITY_V2.md)
- Level regression Huber proposal: [`modeling/TWO_HEAD_LEVEL_REGRESSION_HUBER_V1.md`](modeling/TWO_HEAD_LEVEL_REGRESSION_HUBER_V1.md)
- Ordinal density LR25 run analysis: [`experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_DENSITY_V2_THRESHOLD_LR25_RUN_20260520_ANALYSIS.md`](experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_DENSITY_V2_THRESHOLD_LR25_RUN_20260520_ANALYSIS.md)
- Ordinal density MLP3/THR50 run analysis: [`experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_DENSITY_V2_MLP3_THRESHOLD_LR50_RUN_20260522_ANALYSIS.md`](experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_DENSITY_V2_MLP3_THRESHOLD_LR50_RUN_20260522_ANALYSIS.md)
- Ordinal density centered gap05 MLP3/THR50 run analysis: [`experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_DENSITY_V2_CENTERED_GAP05_MLP3_THRESHOLD_LR50_RUN_20260523_ANALYSIS.md`](experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_DENSITY_V2_CENTERED_GAP05_MLP3_THRESHOLD_LR50_RUN_20260523_ANALYSIS.md)
- Ordinal positional V1 final-concat run analysis: [`experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_POSITIONAL_V1_FINAL_CONCAT_RUN_20260527_ANALYSIS.md`](experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_POSITIONAL_V1_FINAL_CONCAT_RUN_20260527_ANALYSIS.md)
- Ordinal FiLM positional V2 run analysis: [`experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_FILM_POSITIONAL_V2_RUN_20260602_ANALYSIS.md`](experiments/two_head_sft/runs/TWO_HEAD_ORDINAL_FILM_POSITIONAL_V2_RUN_20260602_ANALYSIS.md)
- Latent analysis: [`analysis/latent/`](analysis/latent/)
- Positional latent drift hypothesis: [`analysis/latent/LATENT_POSITIONAL_DRIFT_HYPOTHESIS_2026-05-26.md`](analysis/latent/LATENT_POSITIONAL_DRIFT_HYPOTHESIS_2026-05-26.md)
- Positional latent drift distance analysis: [`analysis/latent/runs/LATENT_POSITIONAL_DRIFT_DISTANCE_ANALYSIS_2026-05-26.md`](analysis/latent/runs/LATENT_POSITIONAL_DRIFT_DISTANCE_ANALYSIS_2026-05-26.md)
- Positional conditioning next steps: [`analysis/latent/POSITIONAL_CONDITIONING_NEXT_STEPS_2026-05-27.md`](analysis/latent/POSITIONAL_CONDITIONING_NEXT_STEPS_2026-05-27.md)

## Decisions And Memoria

- Multi-resource enrichment decision: [`decisions/MULTI_RESOURCE_STRATEGY_DECISION.md`](decisions/MULTI_RESOURCE_STRATEGY_DECISION.md)
- DPO post-SFT alignment decision: [`decisions/DPO_POST_SFT_ALIGNMENT_DECISION.md`](decisions/DPO_POST_SFT_ALIGNMENT_DECISION.md)
- Memoria writing style: [`memoria/MEMORIA_WRITING_STYLE.md`](memoria/MEMORIA_WRITING_STYLE.md)
- Memoria notes: [`memoria/notes/`](memoria/notes/)
- Templates: [`memoria/templates/`](memoria/templates/)

## Placement Rules

Do not add new documents directly to the root of `docs/`. Put new files in the
most specific folder:

- `data/` for preprocessing, dataset, and target contracts.
- `modeling/` for model contracts, training strategy, and architecture notes.
- `evaluation/` for metrics and validation rules.
- `analysis/` for cross-cutting analyses.
- `experiments/<branch>/runs/` for dated run results, audits, and operational notes.
- `decisions/` for stable methodological decisions.
- `reference/` for terminology and reusable lookup material.
- `memoria/` for thesis prose guidance, notes, and templates.
- `archive/` for historical material.

Every new document should state its type near the top: `contract`, `decision`,
`run result`, `analysis`, `memoria note`, `reference`, or `historical`.
