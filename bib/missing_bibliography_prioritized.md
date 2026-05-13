# Missing Bibliography Prioritized

This file records the first internet research pass for papers that were not
already present in `bib/` and are useful for the TFM state of the art.

The personal survey `seminarios_ingles (5).pdf` was used only as an internal
topic map. It should not be cited.

## Priority List

| Priority | File | Paper | Suggested section | Why cite it | Source | Status |
|---:|---|---|---|---|---|---|
| 1 | `01_attention_is_all_you_need.pdf` | Attention Is All You Need | Autoregressive LLMs and Transformers | Foundational Transformer reference for modern LLM sequence generation. | [arXiv](https://arxiv.org/abs/1706.03762) | Downloaded |
| 2 | `02_qwen25_technical_report.pdf` | Qwen2.5 Technical Report | Base model and experimental setup | Supports the documented Qwen2.5 model choice in the repository. | [arXiv](https://arxiv.org/abs/2412.15115) | Downloaded |
| 3 | `03_lora_low_rank_adaptation.pdf` | LoRA: Low-Rank Adaptation of Large Language Models | Efficient adaptation and SFT | Foundational reference for LoRA-style fine-tuning. | [arXiv](https://arxiv.org/abs/2106.09685) | Downloaded |
| 4 | `04_qlora_efficient_finetuning.pdf` | QLoRA: Efficient Finetuning of Quantized LLMs | Efficient adaptation and SFT | Justifies 4-bit/quantized fine-tuning under limited compute. | [arXiv](https://arxiv.org/abs/2305.14314) | Downloaded |
| 5 | `05_scaling_instruction_finetuned_language_models.pdf` | Scaling Instruction-Finetuned Language Models | Instruction tuning | Establishes instruction fine-tuning as a generalization mechanism. | [arXiv](https://arxiv.org/abs/2210.11416) | Downloaded |
| 6 | `06_self_instruct.pdf` | Self-Instruct: Aligning Language Models with Self-Generated Instructions | Instruction tuning and synthetic data | Useful for discussing generated prompts/instruction data, without making it central. | [arXiv](https://arxiv.org/abs/2212.10560) | Downloaded |
| 7 | `07_pretrain_prompt_predict_survey.pdf` | Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP | Prompting and baseline design | Gives a broad prompting taxonomy for explaining zero-shot/few-shot baselines. | [arXiv](https://arxiv.org/abs/2107.13586) | Downloaded |
| 8 | `08_survey_large_language_models.pdf` | A Survey of Large Language Models | LLM background | High-level survey to anchor the LLM background succinctly. | [arXiv](https://arxiv.org/abs/2303.18223) | Downloaded |
| 9 | `09_picard_constrained_decoding.pdf` | PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models | Parser and structural control | Strong precedent for parser-facing constrained autoregressive generation. | [arXiv](https://arxiv.org/abs/2109.05093) | Downloaded |
| 10 | `10_grammar_constrained_decoding_structured_nlp.pdf` | Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning | Grammar and constrained decoding | Modern reference for grammar constraints on structured NLP outputs. | [arXiv](https://arxiv.org/abs/2305.13971) | Downloaded |
| 11 | `11_efficient_guided_generation.pdf` | Efficient Guided Generation for Large Language Models | Guided generation | Useful for modern structured/guided generation framing. | [arXiv](https://arxiv.org/abs/2307.09702) | Downloaded |
| 12 | `12_guidance_controlling_language_models.pdf` | Guidance: A Programming Paradigm for Controlling Language Models | Guided generation | Connects control logic and LLM generation programs. | [arXiv](https://arxiv.org/abs/2306.05285) | Downloaded |
| 13 | `13_structeval_structural_outputs.pdf` | StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs | Structured-output evaluation | Directly relevant because it evaluates structured formats including YAML. | [arXiv](https://arxiv.org/abs/2505.20139) | Downloaded |
| 14 | `14_jsonschemabench_structured_outputs.pdf` | Generating Structured Outputs from Language Models: Benchmark and Studies / JSONSchemaBench | Structured-output evaluation | Useful for discussing schema-constrained output and limits of format guarantees. | [arXiv](https://arxiv.org/abs/2501.10868) | Downloaded |
| 15 | `15_structuredrag_json_response_formatting.pdf` | StructuredRAG: JSON Response Formatting with Large Language Models | Structured-output evaluation | Supports the claim that response-format reliability remains an open problem. | [arXiv](https://arxiv.org/abs/2408.11061) | Downloaded |
| 16 | `16_struc_bench_complex_structured_data.pdf` | Struc-Bench: Are Large Language Models Really Good at Generating Complex Structured Data? | Structured-output evaluation | Adds benchmark evidence for complex structured-data generation failures. | [arXiv](https://arxiv.org/abs/2309.08963) | Downloaded |
| 17 | `17_evaluating_llms_trained_on_code.pdf` | Evaluating Large Language Models Trained on Code | Code and technical artifact generation | Useful for arguing that functional validity matters beyond surface similarity. | [arXiv](https://arxiv.org/abs/2107.03374) | Downloaded |
| 18 | `18_code_llama_open_foundation_models_for_code.pdf` | Code Llama: Open Foundation Models for Code | Code and technical artifact generation | Establishes code-oriented LLMs as a relevant neighboring line. | [arXiv](https://arxiv.org/abs/2308.12950) | Downloaded |
| 19 | `19_doccgen_document_based_controlled_code_generation.pdf` | DocCGen: Document-based Controlled Code Generation | YAML/DSL generation | Very relevant precedent for LLM generation of structured DSLs such as YAML. | [arXiv](https://arxiv.org/abs/2406.11925) | Downloaded |
| 20 | `20_automated_yaml_code_generation_it_tasks.pdf` | Automated Code Generation for Information Technology Tasks in YAML through Large Language Models | YAML/DSL generation | Direct reference for natural-language to YAML IT automation. | [arXiv](https://arxiv.org/abs/2305.02783) | Downloaded |
| 21 | `21_configuration_defects_in_kubernetes.pdf` | Configuration Defects in Kubernetes | Kubernetes domain validity | Justifies Kubernetes configuration as an error-prone domain. | [arXiv](https://arxiv.org/abs/2512.05062) | Downloaded |
| 22 | `22_security_misconfigurations_kubernetes_manifests.pdf` | Security Misconfigurations in Open Source Kubernetes Manifests: An Empirical Study | Kubernetes domain validity | Supports the distinction between YAML parseability and Kubernetes correctness/security. | [Author PDF](https://akondrahman.github.io/files/papers/tosem-k8s.pdf) | Downloaded |
| 23 | `23_inside_job_kubernetes_network_misconfigurations.pdf` | Inside Job: Defending Kubernetes Clusters Against Network Misconfigurations | Kubernetes domain validity | Useful for discussing semantic and security risks in Kubernetes configuration. | [arXiv](https://arxiv.org/abs/2506.21134) | Downloaded |
| 24 | `24_mutiny_kubernetes_failures.pdf` | Mutiny! How does Kubernetes fail, and what can we do about it? | Kubernetes domain validity | Provides background on real-world Kubernetes failure modes. | [arXiv](https://arxiv.org/abs/2404.11169) | Downloaded |
| 25 | `25_diffy_data_driven_bug_finding_configurations.pdf` | Diffy: Data-Driven Bug Finding for Configurations | Configuration defects and validation | Supports structured configuration bug-finding beyond simple syntax checks. | [Author PDF](https://www.sivak.dev/assets/pdf/pldi24_diffy.pdf) | Downloaded |
| 26 | `26_provably_robust_dpo_noisy_feedback.pdf` | Provably Robust DPO: Aligning Language Models with Noisy Feedback | Preference optimization | Useful for limitations of DPO when preference labels are noisy. | [arXiv](https://arxiv.org/abs/2403.00409) | Downloaded |
| 27 | `27_rlhf_workflow_reward_modeling_online_rlhf.pdf` | RLHF Workflow: From Reward Modeling to Online RLHF | RLHF and preference optimization | Complements the local RLHF survey with a workflow-level account. | [OpenReview](https://openreview.net/forum?id=a13aYUU9eU) | Downloaded |
| 28 | `28_kto_prospect_theoretic_optimization.pdf` | KTO: Model Alignment as Prospect Theoretic Optimization | Preference optimization alternatives | Helps mention alternatives to DPO without making them part of the core path. | [arXiv](https://arxiv.org/abs/2402.01306) | Downloaded |
| 29 | `29_survey_rlhf.pdf` | A Survey of Reinforcement Learning from Human Feedback | RLHF and preference optimization | Formal RLHF survey to support definitions such as reward model and feedback types. | [arXiv](https://arxiv.org/abs/2312.14925) | Downloaded |
| 30 | `30_multi_task_uncertainty_loss_weighting.pdf` | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics | Multi-task losses for content and structure | Useful if the two-head model discussion needs principled loss weighting context. | [arXiv](https://arxiv.org/abs/1705.07115) | Downloaded |
| 31 | `31_structural_probe_syntax_word_representations.pdf` | A Structural Probe for Finding Syntax in Word Representations | Structure in learned representations | Supports discussion of hierarchical structure recoverable from representations. | [arXiv](https://arxiv.org/abs/1906.04284) | Downloaded |
| 32 | `32_bert_attention_analysis.pdf` | What Does BERT Look At? An Analysis of BERT's Attention | Structure in learned representations | Useful background for implicit structural information in neural language models. | [arXiv](https://arxiv.org/abs/1906.04341) | Downloaded |
| 33 | `33_abstract_syntax_networks.pdf` | Abstract Syntax Networks for Code Generation and Semantic Parsing | Explicit structural generation | Strong precedent for generation constrained by target syntax/trees. | [arXiv](https://arxiv.org/abs/1704.07535) | Downloaded |
| 34 | `34_syntactic_neural_model_code_generation.pdf` | A Syntactic Neural Model for General-Purpose Code Generation | Explicit structural generation | Supports the idea of generating formal artifacts through syntactic structure. | [arXiv](https://arxiv.org/abs/1704.01696) | Downloaded |

## Notes For The Literature Review

- The strongest narrative arc is: autoregressive LLMs generate flat token
  sequences, but technical artifacts such as YAML and Kubernetes manifests have
  structural and domain constraints.
- The most thesis-specific references are likely priorities 9-16 and 19-25.
- Priorities 26-29 should be used cautiously: they support the post-SFT
  automatic-preference branch, but the repository should not claim a full human
  RLHF pipeline.
- Priorities 30-34 are useful for defending the explicit `level` head as a
  structural modeling choice rather than an arbitrary engineering addition.
